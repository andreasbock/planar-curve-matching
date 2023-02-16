from src.enkf import *
from src.shooting import ShootingParameters
from src.manufactured_solutions import get_solutions


if __name__ == "__main__":
    shooting_parameters = ShootingParameters()
    shooting_parameters.time_steps = 15
    shooting_parameters.kappa = 1

    inverse_problem_parameters = InverseProblemParameters()
    process_per_ensemble_member = 1
    ensemble_object = Ensemble(COMM_WORLD, M=process_per_ensemble_member)

    EXAMPLES_ENKF_PATH = utils.project_root() / f"RESULTS_EXAMPLES_ENKF_ESIZE={ensemble_object.ensemble_comm.size}"
    EXAMPLES_ENKF_PATH.mkdir(exist_ok=True)

    manufactured_solutions = get_solutions(
        kappa=shooting_parameters.kappa,
        shapes=['circle'],
        momentum_names=["squeeze", "star", "teardrop", "contract"],
        resolutions=[1],
    )

    # run EKI over all manufactured solutions
    for manufactured_solution in manufactured_solutions:
        # set up logging
        timestamp = utils.date_string_parallel(ensemble_object.ensemble_comm)
        path = EXAMPLES_ENKF_PATH / manufactured_solution.name() / f"REALISATION_{timestamp}"
        logger = utils.Logger(
            path / "example_enkf.log", ensemble_object.ensemble_comm
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
        enkf.info(f"Loaded solution: '{manufactured_solution.name()}'.")

        # perturb momentum
        pcg = randomfunctiongen.PCG64()
        rg = randomfunctiongen.Generator(pcg)
        random_part = rg.uniform(enkf.shooter.MomentumSpace, -4, 4)

        x, y = SpatialCoordinate(enkf.shooter.mesh)
        momentum_truth = enkf.shooter.momentum_function().interpolate(manufactured_solution.momentum.signal(x, y))
        initial_momentum = enkf.shooter.momentum_function().assign(random_part)

        # run the EKI
        enkf.run_filter(
            momentum=initial_momentum,
            target=manufactured_solution.target,
            momentum_truth=momentum_truth,
        )