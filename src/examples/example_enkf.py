from src.enkf import *
from src.shooting import ShootingParameters
from src.manufactured_solutions import get_solutions


EXAMPLES_ENKF_PATH = utils.project_root() / "RESULTS_EXAMPLES_ENKF"
EXAMPLES_ENKF_PATH.mkdir(exist_ok=True)


if __name__ == "__main__":
    # parameters
    max_iterations = 10
    shooting_parameters = ShootingParameters()
    shooting_parameters.velocity_solver_parameters = {'ksp_type': 'cg'}
    process_per_ensemble_member = 1
    inverse_problem_parameters = InverseProblemParameters()
    ensemble_object = Ensemble(COMM_WORLD, M=process_per_ensemble_member)

    manufactured_solutions = get_solutions(
        momentum_names=['squeeze'],
        shape_names=['circle'],
        resolutions=[0.5],
        landmarks=[20],
        communicator=ensemble_object.comm,
    )

    # run EKI over all manufactured solutions
    for manufactured_solution in manufactured_solutions:
        # set up logging
        logger = utils.Logger(
            EXAMPLES_ENKF_PATH / manufactured_solution.name() / "example_enkf.log", ensemble_object.comm
        )

        # set up EnKF
        enkf = EnsembleKalmanFilter(
            ensemble_object,
            inverse_problem_parameters,
            logger,
            GeodesicShooter(
                logger=logger,
                mesh_path=manufactured_solution.mesh_path,
                template=manufactured_solution.template,
                shooting_parameters=shooting_parameters,
                communicator=ensemble_object.comm,
            )
        )
        enkf._info(f"Loaded solution: '{manufactured_solution.name()}'.")

        # perturb momentum & parameterisation
        pcg = randomfunctiongen.PCG64(seed=4113)
        rg = randomfunctiongen.Generator(pcg)
        random_part = rg.uniform(enkf.forward_operator.DGT, -4, 4)
        momentum = manufactured_solution.momentum
        x, y = SpatialCoordinate(enkf.forward_operator.mesh)
        momentum = enkf.forward_operator.momentum_function().interpolate(Constant(momentum.signal(x, y)))
        momentum.assign(random_part)
        # run the EKI
        enkf.run_filter(
            momentum=momentum,
            parameterisation=manufactured_solution.parameterisation,
            target=manufactured_solution.target,
            max_iterations=max_iterations,
        )
