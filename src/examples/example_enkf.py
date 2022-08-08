from src.enkf import *
from src.shooting import ShootingParameters
from src.manufactured_solutions import get_solutions


EXAMPLES_ENKF_PATH = utils.project_root() / "RESULTS_EXAMPLES_ENKF"


if __name__ == "__main__":
    # set up logging
    logger = utils.Logger(EXAMPLES_ENKF_PATH / "example_enkf.log")

    # parameters
    max_iterations = 50
    shooting_parameters = ShootingParameters()
    process_per_ensemble_member = 1
    inverse_problem_parameters = InverseProblemParameters()
    ensemble_object = Ensemble(COMM_WORLD, M=process_per_ensemble_member)

    # set up EnKF
    enkf = EnsembleKalmanFilter(
        ensemble_object,
        inverse_problem_parameters,
        logger,
    )

    manufactured_solutions = get_solutions(
        momentum_names=['squeeze'],
        shape_names=['circle'],
        resolutions=[0.5],
        landmarks=[10],
    )

    # run EKI over all manufactured solutions
    for manufactured_solution in manufactured_solutions:

        # build shooter based on manufactured solution
        enkf.forward_operator = GeodesicShooter(
            logger=logger,
            mesh_path=manufactured_solution.mesh_path,
            template=manufactured_solution.template,
            shooting_parameters=shooting_parameters,
            communicator=ensemble_object.comm,
        )

        # perturb momentum & parameterisation
        momentum = enkf.forward_operator.momentum_function().interpolate(Constant(".2"))

        # run the EKI
        enkf.run_filter(
            momentum=momentum,
            parameterisation=manufactured_solution.parameterisation,
            target=manufactured_solution.target,
            max_iterations=max_iterations,
        )


