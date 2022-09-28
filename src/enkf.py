from dataclasses import dataclass, asdict
import shutil
import scipy

import numpy as np  # for pickle, don't remove!

from firedrake import *
from pyop2.mpi import MPI

import src.utils as utils
from src.curves import Reparameterisation
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
    eta: float = 1e-03  # noise limit, equals ||\Lambda^{0.5}(q1-G(.))||
    relative_tolerance: float = 1e-05  # relative error to previous iteration
    sample_covariance_regularisation: float = 1  # alpha_0 regularisation parameter
    max_iter_regularisation: int = 40
    optimise_momentum: bool = True
    optimise_parameterisation: bool = True
    time_steps_exponentiate: int = 10

    def dump(self, logger):
        pass


class EnsembleKalmanFilter:
    def __init__(
        self,
        ensemble_object: Ensemble,
        inverse_problem_params: InverseProblemParameters,
        logger: utils.Logger,
        forward_operator: GeodesicShooter = None,
    ):
        self.ensemble = ensemble_object
        self.forward_operator = forward_operator
        self._inverse_problem_params = inverse_problem_params
        self._logger = logger

        self.ensemble_size = self.ensemble.ensemble_comm.size
        self._rank = self.ensemble.ensemble_comm.Get_rank()

        # initialise dynamic member variables
        self.shape = None
        self.momentum = None
        self.parameterisation = None
        self.reparam = None

        self.gamma = None
        self.sqrt_gamma = None
        self.sqrt_gamma_inv = None

    def run_filter(
        self,
        momentum: Function,
        parameterisation: np.array,
        reparam: Reparameterisation,
        target: np.array,
        max_iterations: int,
        momentum_truth: Function = None,
        param_truth: np.array = None,
        reparam_truth: np.array = None,
    ):
        if momentum_truth is not None:
            norm_momentum_truth = np.sqrt(assemble((momentum_truth('+')) ** 2 * dS(CURVE_TAG)))
        self.dump_parameters(target)
        self.momentum = momentum
        self.parameterisation = parameterisation
        self.reparam = reparam

        target = np.array(target)
        _target_periodic = np.append(target, [target[0, :]], axis=0)
        self.gamma = self._inverse_problem_params.gamma_scale * np.eye(product(target.shape), dtype='float')
        self.sqrt_gamma = scipy.linalg.sqrtm(self.gamma)
        self.sqrt_gamma_inv = np.linalg.inv(self.sqrt_gamma)

        # initialise containers for logging
        errors, alphas = [], []
        consensuses_momentum, consensuses_theta = [], []
        relative_error_momentum, relative_error_param = [], []

        iteration = 0
        previous_error = float("-inf")
        while iteration < max_iterations:
            self._info(f"Iteration {iteration}: predicting...")
            shape_mean, momentum_mean, reparam_mean = self.predict()
            mismatch = np.ndarray.flatten(target - shape_mean)
            new_error = self.error_norm(mismatch)
            self._info(f"Iteration {iteration}: Error norm: {new_error}")

            # log everything
            if self._rank == 0:
                utils.pdump(shape_mean, self._logger.logger_dir / f"q_mean_iter={iteration}")
                utils.pdump(reparam_mean, self._logger.logger_dir / f"t_mean_iter={iteration}")
                utils.pdump(mismatch, self._logger.logger_dir / f"mismatch_iter={iteration}")
                #consensuses_momentum.append(self._consensus_momentum(momentum_mean))
                #consensuses_theta.append(self._consensus_theta(theta_mean))
                errors.append(new_error)
                #if momentum_truth is not None:
                #    relative_momentum_norm = np.sqrt(assemble((self.momentum('+') - momentum_mean('+')) ** 2 * dS(CURVE_TAG)))
                #    relative_error_momentum.append(relative_momentum_norm / norm_momentum_truth)
                #if param_truth is not None:
                #    relative_error_param.append(np.linalg.norm(theta_mean - param_truth) / np.linalg.norm(param_truth))

            # either we have converged or we correct
            if self.has_converged(new_error, previous_error):
                self._info(f"Filter has converged.")
                break
            else:
                centered_shape = np.ndarray.flatten(self.shape - shape_mean)
                cw_alpha_gamma_inv, alpha = self.compute_cw_operator(centered_shape, mismatch)
                mismatch_local = np.ndarray.flatten(target - self.shape)
                shape_update = np.dot(cw_alpha_gamma_inv, mismatch_local)
                if self._inverse_problem_params.optimise_momentum:
                    self._info(f"Iteration {iteration}: correcting momentum...")
                    self._correct_momentum(momentum_mean, centered_shape, shape_update)
                if self._inverse_problem_params.optimise_parameterisation:
                    self._info(f"Iteration {iteration}: correcting parameterisation...")
                    self._correct_reparam(reparam_mean, centered_shape, shape_update)
                if self._rank == 0:
                    alphas.append(alpha)

            previous_error = new_error
            iteration += 1
        if self._rank == 0:
            utils.pdump(errors, self._logger.logger_dir / "errors")
            utils.pdump(relative_error_momentum, self._logger.logger_dir / "relative_error_momentum")
            utils.pdump(relative_error_param, self._logger.logger_dir / "relative_error_param")
            utils.pdump(alphas, self._logger.logger_dir / "alphas")
            utils.pdump(consensuses_momentum, self._logger.logger_dir / "consensuses_momentum")
            utils.pdump(consensuses_theta, self._logger.logger_dir / "consensuses_theta")
        if iteration > max_iterations:
            self._info(f"Filter stopped - maximum iteration count reached.")

    def predict(self):
        if self._inverse_problem_params.optimise_parameterisation:
            # integrate reparameterisation
            reparameterised_points = self.reparam.exponentiate(
                self.parameterisation,
                self._inverse_problem_params.time_steps_exponentiate,
            )
        else:
            reparameterised_points = self.parameterisation

        # shoot with momenta
        curve_result = self.forward_operator.shoot(self.momentum)

        # evaluate curve
        template_points = self.forward_operator.template.at(reparameterised_points)
        self.shape = np.array(curve_result.diffeo.at(template_points))

        # compute ensemble means
        shape_mean = self._compute_shape_mean()
        momentum_mean = self._compute_momentum_mean()
        reparam_mean = self._compute_reparam_mean()

        return shape_mean, momentum_mean, reparam_mean

    def _correct_momentum(self, momentum_mean, centered_shape_flat, shape_update):
        if self._inverse_problem_params.optimise_momentum:
            self._info(f"Correcting momentum...")
            centered_momentum = self.forward_operator.momentum_function()
            centered_momentum.assign(self.momentum - momentum_mean)
            local_C_pw = np.outer(centered_momentum.dat.data, centered_shape_flat)
            C_pw = np.empty(shape=local_C_pw.shape)
            self._mpi_reduce(local_C_pw, C_pw)
            self.momentum.dat.data[:] += np.dot(C_pw, shape_update)

    def _correct_reparam(self, reparam_mean, centered_shape, shape_update):
        centered_param = self.reparam.spline.c - reparam_mean
        c_rq = np.outer(centered_param, centered_shape)
        c_rq_all = np.zeros(shape=c_rq.shape)
        self._mpi_reduce(c_rq, c_rq_all)
        c_rq_all /= self.ensemble_size - 1
        gain = np.dot(c_rq_all, shape_update)
        gain.shape = reparam_mean.shape
        self.reparam.spline.c += gain

    def compute_cw_operator(self, centered_shape, mismatch):
        rhs = self._inverse_problem_params.rho * self.error_norm(mismatch)
        self._info(f"\t rhs = {rhs}")

        alpha = self._inverse_problem_params.sample_covariance_regularisation
        cw = self.compute_cw(centered_shape)

        iteration = 0
        while iteration < self._inverse_problem_params.max_iter_regularisation:
            # compute the operator of which we need the inverse
            cw_alpha_gamma_inv = np.linalg.inv(cw + alpha * self.gamma)

            # compute the error norm (lhs)
            lhs = alpha * self.error_norm(np.dot(cw_alpha_gamma_inv, mismatch))
            self._info(f"\t lhs = {lhs}")

            if lhs >= rhs:
                self._info(f"\t alpha = {alpha}")
                return cw_alpha_gamma_inv, alpha
            alpha *= 2
            iteration += 1

        error_message = f"!!! alpha failed to converge in {iteration} iterations"
        self._info(error_message)
        raise Exception(error_message)

    def compute_cw(self, centered_shape):
        centered_shape_flat = np.ndarray.flatten(centered_shape)
        cov_mismatch = np.outer(centered_shape_flat, centered_shape_flat)
        cov_mismatch_all = np.zeros(shape=cov_mismatch.shape)
        self._mpi_reduce(cov_mismatch, cov_mismatch_all)
        return cov_mismatch_all / (self.ensemble_size - 1)

    def has_converged(self, n_err, err):
        if n_err <= self._inverse_problem_params.tau*self._inverse_problem_params.eta:
            self._info(f"Converged, error at noise level ({n_err} <= {self._inverse_problem_params.tau*self._inverse_problem_params.eta}).")
            return True
        elif np.fabs(n_err - err) < self._inverse_problem_params.relative_tolerance:
            self._info("No improvement in residual, terminating filter.")
            return True
        else:
            return False

    def error_norm(self, mismatch):
        return observation_norm(self.sqrt_gamma_inv, mismatch)

    def dump_parameters(self, target=None):
        if self._rank == 0:
            self._info(f"Ensemble size: {self.ensemble_size}.")
            self._info(f"{self._inverse_problem_params}")
            self.forward_operator.dump_parameters()

            fh = open(self._logger.logger_dir / 'inverse_problem_parameters.log', 'w')
            for key, value in asdict(self._inverse_problem_params).items():
                fh.write(f"{key}: {value}\n")
            fh.close()

            if target is not None:
                utils.pdump(target, self._logger.logger_dir / 'target')

    def _mpi_reduce(self, f, reduced):
        self.ensemble.ensemble_comm.Allreduce(f, reduced, op=MPI.SUM)

    def _clear_cache(self):
        if self._rank == 0:
            fd_cache = os.environ['VIRTUAL_ENV'] + '/.cache/'
            for subdir in ['pyop2', 'tsfc']:
                cache = fd_cache + subdir
                if os.path.exists(cache) and os.path.isdir(cache):
                    shutil.rmtree(cache)
        self.ensemble.ensemble_comm.Barrier()

    def _compute_shape_mean(self):
        shape_mean = np.empty(shape=self.shape.shape)
        self._mpi_reduce(self.shape, shape_mean)
        shape_mean /= self.ensemble_size
        return shape_mean

    def _compute_momentum_mean(self):
        momentum_mean = self.forward_operator.momentum_function()
        self.ensemble.allreduce(self.momentum, momentum_mean)
        momentum_mean.assign(momentum_mean * Constant(1 / self.ensemble_size))
        return momentum_mean

    def _compute_reparam_mean(self):
        _reparam_mean = np.zeros(shape=self.reparam.spline.c.shape)
        self._mpi_reduce(self.reparam.spline.c, _reparam_mean)
        return _reparam_mean / self.ensemble_size

    def _consensus_momentum(self, momentum_mean):
        _consensus_me = np.sqrt(np.array([assemble((self.momentum('+') - momentum_mean('+')) ** 2 * dS(CURVE_TAG))]))
        _consensus = np.zeros(shape=_consensus_me.shape)
        self._mpi_reduce(_consensus_me, _consensus)
        return _consensus[0] / self.ensemble_size

    def _consensus_theta(self, theta_mean):
        _consensus_me = np.linalg.norm(self.parameterisation - theta_mean)
        _consensus = np.zeros(shape=_consensus_me.shape)
        self._mpi_reduce(_consensus_me, _consensus)
        return _consensus / self.ensemble_size

    def _info(self, msg):
        if self._rank == 0:
            self._logger.info(msg)


def observation_norm(sqrt_matrix, x):  # TODO: check order of this!
    inner_norm = np.dot(sqrt_matrix, x)
    return np.sqrt(np.dot(inner_norm, inner_norm))

