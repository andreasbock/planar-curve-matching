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
    inverse_problem_parameters = InverseProblemParameters()
    ensemble_object = Ensemble(COMM_WORLD, M=1)

    # set up EnKF
    enkf = EnsembleKalmanFilter(
        ensemble_object,
        inverse_problem_parameters,
        logger,
    )

    # run EKI over all manufactured solutions
    for manufactured_solution in get_solutions():

        # build shooter based on manufactured solution
        enkf.forward_operator = GeodesicShooter(
            logger=logger,
            mesh_path=manufactured_solution.mesh_path,
            template=manufactured_solution.template,
            shooting_parameters=shooting_parameters,
        )

        # run the EKI
        enkf.run_filter(
            momentum=manufactured_solution.momentum,
            parameterisation=manufactured_solution.parameterisation,
            target=manufactured_solution.target,
            max_iterations=max_iterations,
        )


