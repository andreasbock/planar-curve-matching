from dataclasses import dataclass
import shutil
import scipy
import numpy as np

from firedrake import *
from pyop2.mpi import MPI

import src.utils as utils
from src.shooting import GeodesicShooter


@dataclass
class InverseProblemParameters:
    # rho/tau/eta are described in section 2.1 of
    #   Iglesias, Marco A. "A regularizing iterative ensemble Kalman method
    #   for PDE-constrained inverse problems." Inverse Problems 32.2 (2016):
    #   025002.
    rho: float = 0.9  # \rho \in (0, 1)
    tau: float = 1 / 0.9 + 1e-04  # \tau > 1/\rho (!)
    gamma_scale: float = 0.1  # observation variance
    eta: float = 1e-03  # noise limit, equals ||\Lambda^{0.5}(q1-G(.))||
    relative_tolerance: float = 1e-05  # relative error to previous iteration
    sample_covariance_regularisation: float = 1  # alpha_0 regularisation parameter
    max_iter_regularisation: int = 40
    optimise_momentum: bool = True
    optimise_parameterisation: bool = True

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

        self.gamma = None
        self.sqrt_gamma = None
        self.sqrt_gamma_inv = None

    def run_filter(self, momentum, parameterisation, target, max_iterations):
        self.dump_parameters()
        self.momentum = momentum
        self.parameterisation = parameterisation

        self.gamma = self._inverse_problem_params.gamma_scale * np.eye(sum(target.shape), dtype='float')
        self.sqrt_gamma = scipy.linalg.sqrtm(self.gamma)
        self.sqrt_gamma_inv = np.linalg.inv(self.sqrt_gamma)

        # initialise containers for logging
        errors, alphas = [], []
        consensuses_momentum, consensuses_theta = [], []

        iteration = 0
        previous_error = float("-inf")
        while iteration < max_iterations:
            self._info(f"Iteration {iteration}: predicting...")
            shape_mean, momentum_mean, theta_mean = self.predict()
            mismatch = target - shape_mean
            new_error = self.error_norm(mismatch)
            self._info(f"Iteration {iteration}: Error norm: {new_error}")

            # log everything
            if self._rank == 0:
                utils.pdump(shape_mean, self._logger.logger_path / f"w_mean_iter={iteration}")
                utils.pdump(theta_mean, self._logger.logger_path / f"t_mean_iter={iteration}")
                consensuses_momentum.append(self._consensus_momentum(momentum_mean))
                consensuses_theta.append(self._consensus_theta(theta_mean))
                errors.append(new_error)

            # either we have converged or we correct
            if self.has_converged(new_error, previous_error):
                break
            else:
                self._info(f"Iteration {iteration}: correcting...")
                shape_update, alpha = self.mismatch_covariance(mismatch)
                self._correct_momentum(momentum_mean, mismatch, shape_update)
                self._correct_theta(theta_mean, mismatch, shape_update)
                if self._rank == 0:
                    alphas.append(alpha)

            previous_error = new_error
            iteration += 1
        if self._rank == 0:
            utils.pdump(errors, self._logger.logger_path / "errors")
            utils.pdump(alphas, self._logger.logger_path / "alphas")
            utils.pdump(consensuses_momentum, self._logger.logger_path / "consensuses_momentum")
            utils.pdump(consensuses_theta, self._logger.logger_path / "consensuses_theta")

    def predict(self):
        # shoot using ensemble momenta
        _ = self.forward_operator.shoot(self.momentum)
        self.shape = self.forward_operator.evaluate_curve(self.parameterisation)

        shape_mean = self._compute_shape_mean()
        momentum_mean = self._compute_momentum_mean()
        theta_mean = self._compute_theta_mean()
        # TODO: think about how to dump average shape for inspection
        #if self.verbose:
        #    coords = Function(self.gs.VCG)
        #    self._ensemble.allreduce(q, coords)
        #    coords.assign(coords * self.normaliser)
        #    old_coords = self.gs.mesh.coordinates.copy(deepcopy=True)
        #    self.gs._update_mesh(coords)
        #    File(self.log_dir + "w_mean_{}.pvd".format(iteration)).write(self.shape_function)
        #    self.gs._update_mesh(old_coords)
        return shape_mean, momentum_mean, theta_mean

    def optimise_momentum(self):
        return self._inverse_problem_params.optimise_momentum

    def optimise_parameterisation(self):
        return self._inverse_problem_params.optimise_parameterisation

    def _correct_momentum(self, momentum_mean, mismatch, shape_update):
        self._logger.info('Updating momentum')
        firedrake_mismatch_no_localisation = [Constant(w) for w in np.ndarray.flatten(mismatch)]
        #if self.localise_momentum:
        #    for i, w in enumerate(w_flat_noloc):
        #        self.localisers[i].interpolate(self.localiser(self.T[i//2], w))
        #    w_flat = self.localisers
        #else:
        w_flat = firedrake_mismatch_no_localisation
        m = self.forward_operator.momentum_function()
        mr = self.forward_operator.momentum_function()
        gain = self.forward_operator.momentum_function()
        self.ensemble.ensemble_comm.Barrier()

        for w, _s in zip(w_flat, shape_update):
            m.assign((self.momentum - momentum_mean) * w)
            self.ensemble.allreduce(m, mr)
            gain.assign(gain + mr * Constant(_s))
        self.momentum.assign(self.momentum + gain * Constant(1 / (self.ensemble_size - 1)))

    def _correct_theta(self, theta_mean, mismatch, shape_update):
        self._logger.info('Updating parameterisation')

        cov_theta = np.outer(self.parameterisation - theta_mean, mismatch)
        cov_theta_all = np.zeros(shape=cov_theta.shape)
        self._mpi_reduce(cov_theta, cov_theta_all)
        cov_theta_all /= self.ensemble_size - 1
        gain = np.dot(cov_theta_all, shape_update)

        self.parameterisation += gain
        self.parameterisation = self.parameterisation % (2 * np.pi)

    def _compute_shape_mean(self):
        # TODO: Karcher mean?
        shape_mean = np.zeros(shape=self.shape.shape)
        self._mpi_reduce(self.shape, shape_mean)
        shape_mean /= self.ensemble_size
        return shape_mean

    def _compute_momentum_mean(self):
        momentum_mean = self.forward_operator.momentum_function()
        self.ensemble.allreduce(self.momentum, momentum_mean)
        momentum_mean.assign(momentum_mean * Constant(1 / (self.ensemble_size - 1)))
        return momentum_mean

    def _compute_theta_mean(self):
        theta_mean = np.zeros(shape=self.parameterisation.shape)
        self._mpi_reduce(self.parameterisation, theta_mean)
        theta_mean /= self.ensemble_size
        return theta_mean

    def _consensus_momentum(self, momentum_mean):
        _consensus_me = np.sqrt(np.array([assemble((self.momentum('+') - momentum_mean('+')) ** 2 * dS(10))]))
        _consensus = np.zeros(_consensus_me.shape)
        self._mpi_reduce(_consensus_me, _consensus)
        return _consensus[0] / self.ensemble_size

    def _consensus_theta(self, theta_mean):
        _consensus_me = np.linalg.norm(self.parameterisation - theta_mean)
        _consensus = np.zeros(_consensus_me.shape)
        self._mpi_reduce(_consensus_me, _consensus)
        return _consensus / self.ensemble_size

    def mismatch_covariance(self, mismatch, localise=False):
        cw_alpha_gamma_inv, alpha = self.compute_cw_operator(mismatch, localise)
        return np.dot(cw_alpha_gamma_inv, np.ndarray.flatten(mismatch)), alpha

    def compute_cw_operator(self, mismatch, localise=False):
        _rhs = self._inverse_problem_params.rho * self.error_norm(mismatch)
        alpha = self._inverse_problem_params.sample_covariance_regularisation
        cw = self.compute_cw(mismatch, localise=localise)
        iteration = 0
        while iteration < self._inverse_problem_params.max_iter_regularisation:
            # compute the operator of which we need the inverse
            cw_alpha_gamma_inv = np.linalg.inv(cw + alpha * self.gamma)
            # compute the error norm (lhs)
            w_cw_inv = np.dot(self.sqrt_gamma, np.dot(cw_alpha_gamma_inv, np.ndarray.flatten(mismatch)))
            _lhs = np.sqrt(np.dot(w_cw_inv, w_cw_inv))
            if alpha * _lhs >= _rhs:
                return cw_alpha_gamma_inv, alpha
            alpha *= 2
            iteration += 1

        error_message = f"!!! alpha failed to converge in {iteration} iterations"
        utils.pprint(error_message)
        raise Exception(error_message)

    def compute_cw(self, mismatch, localise=False):
        mismatch_flat = np.ndarray.flatten(mismatch)
        cov_mismatch = np.outer(mismatch_flat, mismatch_flat)
        cov_mismatch_all = np.zeros(shape=cov_mismatch.shape)
        self._mpi_reduce(cov_mismatch, cov_mismatch_all)
        if localise:
            cov_mismatch_all = np.multiply(cov_mismatch_all, np.eye(cov_mismatch_all.shape[0]))
        return cov_mismatch_all / (self.ensemble_size - 1)

    def has_converged(self, n_err, err):
        if n_err <= self._inverse_problem_params.tau*self._inverse_problem_params.eta:
            utils.pprint("Converged, error at noise level.")
            return True
        elif np.fabs(n_err - err) < self._inverse_problem_params.relative_tolerance:
            utils.pprint("No improvement in residual, terminating filter.")
            return True

    def error_norm(self, mismatch):
        return observation_norm(self.sqrt_gamma_inv, mismatch)

    def gamma_norm(self, mismatch):
        return observation_norm(self.sqrt_gamma, mismatch)

    def _mpi_reduce(self, f, reduced):
        self.ensemble.ensemble_comm.Allreduce(f, reduced, op=MPI.SUM)

    def _clear_cache(self):
        self.ensemble.ensemble_comm.Barrier()
        if self._rank == 0:
            fd_cache = os.environ['VIRTUAL_ENV'] + '/.cache/'
            for subdir in ['pyop2', 'tsfc']:
                cache = fd_cache + subdir
                if os.path.exists(cache) and os.path.isdir(cache):
                    shutil.rmtree(cache)
        self.ensemble.ensemble_comm.Barrier()

    def dump_parameters(self):
        self._logger.info(f"{self._inverse_problem_params}")
        self.forward_operator.dump_parameters()

    def _info(self, msg):
        if self._rank == 0:
            self._logger.info(msg)


def observation_norm(sqrt_matrix, x):  # TODO: check order of this!
    inner_norm = np.dot(sqrt_matrix, np.ndarray.flatten(x))
    return np.sqrt(np.dot(inner_norm, inner_norm))

