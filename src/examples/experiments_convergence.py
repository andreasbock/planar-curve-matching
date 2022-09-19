from src.enkf import *
from src.shooting import ShootingParameters
from src.manufactured_solutions import get_solutions

PATH = utils.project_root() / "RESULTS_CONVERGENCE"
PATH.mkdir(exist_ok=True)


def convergence_experiment():
    # parameters
    num_observations = 20
    resolution = 0.5
    max_iterations = 10
    shooting_parameters = ShootingParameters()

    process_per_ensemble_member = 1
    inverse_problem_parameters = InverseProblemParameters()
    ensemble_object = Ensemble(COMM_WORLD, M=process_per_ensemble_member)

    manufactured_solutions = get_solutions(
        shapes=['circle', 'small_triangle'],
        momentum_names=['expand', 'teardrop'],
        resolutions=[resolution],
        landmarks=[num_observations],
        communicator=ensemble_object.comm,
    )

    # run EKI over all manufactured solutions
    for manufactured_solution in manufactured_solutions:
        # set up logging
        logger = utils.Logger(
            PATH / manufactured_solution.name() / "example_enkf.log", ensemble_object.comm
        )

        # set up EnKF
        shooter = GeodesicShooter(
            logger=logger,
            mesh_path=manufactured_solution.mesh_path,
            template=manufactured_solution.template,
            shooting_parameters=shooting_parameters,
            communicator=ensemble_object.comm,
        )

        enkf = EnsembleKalmanFilter(
            ensemble_object,
            inverse_problem_parameters,
            logger,
            shooter,
        )
        enkf._info(f"Loaded solution: '{manufactured_solution.name()}'.")

        # perturb momentum & parameterisation
        pcg = randomfunctiongen.PCG64(seed=12315123)
        rg = randomfunctiongen.Generator(pcg)
        random_part = rg.uniform(enkf.forward_operator.DGT, -1, 1)
        momentum = manufactured_solution.momentum
        x, y = SpatialCoordinate(enkf.forward_operator.mesh)
        momentum_truth = enkf.forward_operator.momentum_function().interpolate(momentum.signal(x, y))
        momentum = enkf.forward_operator.momentum_function().assign(momentum_truth + random_part)

        # perturb parameterisation
        parameterisation = (
                manufactured_solution.parameterisation
                + rg.uniform(low=-1, high=1, size=manufactured_solution.parameterisation.shape)
        ) % 2 * np.pi

        # run the EKI
        enkf.run_filter(
            momentum=momentum,
            parameterisation=parameterisation,
            target=manufactured_solution.target,
            max_iterations=max_iterations,
            momentum_truth=momentum_truth,
            param_truth=manufactured_solution.parameterisation,
        )


if __name__ == "__main__":
    convergence_experiment()
