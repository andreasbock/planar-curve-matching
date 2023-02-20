import time
from dataclasses import dataclass
import numpy as np  # for pickle, don't remove!

from firedrake import *
from pyop2.mpi import MPI

import src.utils as utils
from src.mesh_generation import CURVE_TAG
from src.shooting import GeodesicShooter


@dataclass
class InverseProblemParameters:
    # rho/tau/eta are described in section 2.1 of
    #   Iglesias, Marco A. "A regularizing iterative ensemble Kalman method
    #   for PDE-constrained inverse problems." Inverse Problems 32.2 (2016):
    #   025002.
    rho: float = 0.9  # \rho \in (0, 1)
    tau: float = 1 / 0.9 + 1e-04  # \tau > 1/\rho (!)
    gamma_scale: float = 1  # observation variance
    eta: float = 1e-03  # noise limit, equals ||\Lambda^{0.5}(q1 - G(.))||
    relative_tolerance: float = 1e-03  # relative error to previous iteration
    sample_covariance_regularisation: float = 0.001  # alpha_0 regularisation parameter
    dynamic_regularisation: bool = False
    max_iter_regularisation: int = 40


class EnsembleKalmanFilter:
    def __init__(
        self,
        ensemble_object: Ensemble,
        inverse_problem_params: InverseProblemParameters,
        logger: utils.Logger,
        forward_operator: GeodesicShooter = None,
    ):
        self.ensemble = ensemble_object
        self.shooter = forward_operator
        self.inverse_problem_params = inverse_problem_params
        self._logger = logger

        self.ensemble_size = self.ensemble.ensemble_comm.size
        self._rank = self.ensemble.ensemble_comm.Get_rank()

        # initialise dynamic member variables
        self.shape = None
        self.shape_mean = Function(self.shooter.ShapeSpace)
        self.shape_centered = Function(self.shooter.ShapeSpace)

        self.curve_data_indices = self._curve_ids()
        dim_momentum_data = len(self.curve_data_indices)

        self.momentum = self.shooter.momentum_function()
        self.momentum_mean = self.shooter.momentum_function()
        self.momentum_centered = self.shooter.momentum_function()

        self.mismatch = Function(self.shooter.ShapeSpace)
        self.mismatch_local = Function(self.shooter.ShapeSpace)

        self.gamma = None
        self.gamma_inv = None
        self.mean_normaliser = Constant(1 / self.ensemble_size)

        cov_sz = len(self.shape_mean.dat.data)
        self.cov_mismatch = np.empty((cov_sz, cov_sz))
        self.cov_mismatch_all = np.empty(shape=self.cov_mismatch.shape)

        self.xcorr_momentum = np.empty((dim_momentum_data, cov_sz))
        self.xcorr_momentum_all = np.empty(shape=self.xcorr_momentum.shape)

    def run_filter(
        self,
        momentum: Function,
        target: Function,
        max_iterations: int,
        momentum_truth: Function = None,
    ):
        if momentum_truth is not None:
            norm_momentum_truth = np.sqrt(assemble((momentum_truth('+')) ** 2 * dS(CURVE_TAG)))

        self.dump_parameters(target)
        self.momentum.assign(momentum)
        eye = np.eye(product(target.dat.data.shape), dtype='float')
        self.gamma = eye * self.inverse_problem_params.gamma_scale
        self.gamma_inv = eye / self.inverse_problem_params.gamma_scale

        # initialise containers for logging
        errors, alphas = [], []
        consensuses_momentum, relative_error_momentum = [], []

        iteration = 0
        previous_error = float("-inf")
        time_start = time.time()
        while iteration < max_iterations:
            self.info(f"Iteration {iteration}: predicting...")
            self.predict()
            self.mismatch.assign(target - self.shape_mean)
            self.mismatch_local.assign(target - self.shape)

            File(self._logger.logger_data_dir / f"shape_iter={iteration}.pvd").write(self.shooter.shape_function)
            File(self._logger.logger_data_dir / f"smooth_shape_iter={iteration}.pvd").write(self.shooter.smooth_shape_function)
            File(self._logger.logger_data_dir / f"shape_inital_mesh_iter={iteration}.pvd").write(self.shape)
            File(self._logger.logger_data_dir / f"mismatch_iter={iteration}.pvd").write(self.mismatch)
            File(self._logger.logger_data_dir / f"mismatch_local_iter={iteration}.pvd").write(self.mismatch_local)

            new_error = norm(self.mismatch)
            self.info(f"Iteration {iteration}: Error norm: {new_error}")
            consensus_momentum = self._consensus_momentum(self.momentum_mean)
            self.info(f"Iteration {iteration} | Consensus: {consensus_momentum}")
            # log everything
            if self._rank == 0:
                utils.plot_curves(self.shape_mean, self._logger.logger_dir / f"shape_mean_iter={iteration}.pdf")
                utils.plot_curves(self.mismatch, self._logger.logger_dir / f"mismatch_iter={iteration}.pdf")
                consensuses_momentum.append(consensus_momentum)
                if momentum_truth is not None:
                    relative_momentum_norm = np.sqrt(assemble((self.momentum('+') - self.momentum_mean('+')) ** 2 * dS(CURVE_TAG)))
                    relative_error_momentum.append(relative_momentum_norm / norm_momentum_truth)
            self.ensemble.ensemble_comm.Barrier()

            # either we have converged or we correct
            if self.has_converged(new_error, previous_error):
                self.info(f"Filter has converged.")
                break
            else:
                self.shape_centered.assign(self.shape - self.shape_mean)
                cw_alpha_gamma_inv, alpha = self.compute_cw_operator()
                shape_update = np.dot(cw_alpha_gamma_inv, self.mismatch_local.dat.data)

                self.info(f"Iteration {iteration}: correcting momentum...")
                self._correct_momentum(shape_update)
                if self._rank == 0:
                    alphas.append(alpha)

            errors.append(new_error)
            previous_error = new_error
            iteration += 1
        time_end = time.time()
        elapsed = time.gmtime(time_end - time_start)
        elapsed_str = time.strftime("%Hh%Mm%Ss", elapsed)

        if iteration == max_iterations:
            self.info(f"Filter stopped - maximum iteration count reached.")

        self.info(f"Shooting with momentum mean...")
        self.shooter.shoot(self.momentum_mean)
        if self._rank == 0:
            self.info("Time elapsed:")
            self.info(f"\t {elapsed_str}")
            self.info(f"Dumping forward operator of momentum mean...")
            utils.pdump(errors, self._logger.logger_data_dir / "errors")
            utils.pdump(relative_error_momentum, self._logger.logger_data_dir / "relative_error_momentum")
            utils.pdump(alphas, self._logger.logger_data_dir / "alphas")
            utils.pdump(consensuses_momentum, self._logger.logger_data_dir / "consensuses_momentum")
            utils.pdump(self.momentum_mean.dat.data, self._logger.logger_data_dir / "momentum_mean_converged")
            utils.plot_curves(
                self.shooter.shape_function,
                self._logger.logger_dir / f"{self.shooter.mesh_path.stem}_converged.pdf"
            )
        File(self._logger.logger_data_dir / f"{self.shooter.mesh_path.stem}_converged.pvd").write(
            self.shooter.shape_function,
        )
        self.info(f"Done.")

    def predict(self):
        self.ensemble.ensemble_comm.Barrier()

        # shoot
        self.shooter.shoot(self.momentum)
        self.shape = self.shooter.smooth_shape_function_initial_mesh()

        # compute ensemble means
        self._compute_shape_mean()
        self.shooter.update_mesh()
        self._compute_momentum_mean()

    def _correct_momentum(self, shape_update):
        self.ensemble.ensemble_comm.Barrier()
        self.momentum_centered.assign(self.momentum - self.momentum_mean)
        local_C_pw = np.outer(self.momentum_centered.dat.data[self.curve_data_indices], self.shape_centered.dat.data)
        C_pw = np.empty(shape=local_C_pw.shape)
        self._mpi_reduce(local_C_pw, C_pw)
        C_pw /= self.ensemble_size - 1
        self.momentum.dat.data[self.curve_data_indices] += np.dot(C_pw, shape_update)

    def compute_cw_operator(self):
        rhs = self.inverse_problem_params.rho * self.error_norm(self.mismatch)

        alpha = self.inverse_problem_params.sample_covariance_regularisation
        cw = self.compute_cw()

        iteration = 0
        while iteration < self.inverse_problem_params.max_iter_regularisation:
            # compute the operator of which we need the inverse
            cw_alpha_gamma_inv = np.linalg.inv(cw + alpha * self.gamma)

            # compute the error norm (lhs)
            lhs = alpha * self.error_norm(np.dot(cw_alpha_gamma_inv, self.mismatch.dat.data))
            if lhs >= rhs or not self.inverse_problem_params.dynamic_regularisation:
                return cw_alpha_gamma_inv, alpha
            alpha *= 2
            iteration += 1

        error_message = f"!!! alpha failed to converge in {iteration} iterations"
        self.info(error_message)
        raise Exception(error_message)

    def compute_cw(self):
        shape_data = self.shape_centered.dat.data
        np.outer(shape_data, shape_data, self.cov_mismatch)
        self._mpi_reduce(self.cov_mismatch, self.cov_mismatch_all)
        return self.cov_mismatch_all / (self.ensemble_size - 1)

    def has_converged(self, n_err, err):
        if n_err <= self.inverse_problem_params.tau*self.inverse_problem_params.eta:
            self.info(f"Converged, error at noise level ({n_err} <= {self.inverse_problem_params.tau * self.inverse_problem_params.eta}).")
            return True
        elif np.fabs(n_err - err) < self.inverse_problem_params.relative_tolerance:
            self.info("No improvement in residual, terminating filter.")
            return True
        else:
            return False

    def error_norm(self, x):
        if isinstance(x, Function):
            x = x.dat.data
        return np.sqrt(np.dot(x.T, np.dot(self.gamma_inv, x)))

    def dump_parameters(self, target=None):
        self.info(f"Ensemble size: {self.ensemble_size}.")
        self.info(f"Inverse problem parameters: {self.inverse_problem_params}.")
        self.info(f"Momentum dimension: {len(self.curve_data_indices)}.")
        self.shooter.dump_parameters()
        File(self._logger.logger_data_dir / 'target.pvd').write(target)

    def _mpi_reduce(self, f, reduced):
        self.ensemble.ensemble_comm.Allreduce(f, reduced, op=MPI.SUM)

    def _compute_shape_mean(self):
        self.ensemble.allreduce(self.shape, self.shape_mean)
        self.shape_mean.assign(self.shape_mean * self.mean_normaliser)

    def _compute_momentum_mean(self):
        self.ensemble.allreduce(self.momentum, self.momentum_mean)
        self.momentum_mean.assign(self.momentum_mean * self.mean_normaliser)

    def _consensus_momentum(self, momentum_mean):
        _consensus_me = np.sqrt(np.array([assemble((self.momentum('+') - momentum_mean('+')) ** 2 * dS(CURVE_TAG))]))
        _consensus = np.zeros(shape=_consensus_me.shape)
        self._mpi_reduce(_consensus_me, _consensus)
        return _consensus[0] / self.ensemble_size

    def _curve_ids(self):
        interior_bc = DirichletBC(self.shooter.MomentumSpace, 1, CURVE_TAG)
        f = Function(self.shooter.MomentumSpace)
        interior_bc.apply(f)
        return f.dat.data.nonzero()[0]

    def momentum_data(self):
        return self.momentum.dat.data[self.curve_data_indices]

    def info(self, msg):
        if self._rank == 0:
            self._logger.info(msg)



