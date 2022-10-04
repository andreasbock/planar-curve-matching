from src.enkf import *
from src.shooting import ShootingParameters
from src.manufactured_solutions import get_solutions

if __name__ == "__main__":
    # parameters
    max_iterations = 10
    shooting_parameters = ShootingParameters()
    process_per_ensemble_member = 1
    inverse_problem_parameters = InverseProblemParameters()
    inverse_problem_parameters.optimise_parameterisation = False
    ensemble_object = Ensemble(COMM_WORLD, M=process_per_ensemble_member)

    EXAMPLES_ENKF_PATH = utils.project_root() / f"RESULTS_EXAMPLES_ENKF_ESIZE={ensemble_object.ensemble_comm.size}"
    EXAMPLES_ENKF_PATH.mkdir(exist_ok=True)

    manufactured_solutions = get_solutions(
        shapes=['circle'],
        momentum_names=['contract', 'squeeze', 'teardrop'],
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

        # perturb momentum
        pcg = randomfunctiongen.PCG64(seed=4113)
        rg = randomfunctiongen.Generator(pcg)
        random_part = rg.uniform(enkf.forward_operator.MomentumSpace, -4, 4)

        x, y = SpatialCoordinate(enkf.forward_operator.mesh)
        momentum_truth = enkf.forward_operator.momentum_function().interpolate(manufactured_solution.momentum.signal(x, y))
        initial_momentum = enkf.forward_operator.momentum_function().assign(random_part)

        # perturb reparam
        initial_parameterisation = manufactured_solution.parameterisation
        initial_reparam = Reparameterisation(
            n_cells=len(manufactured_solution.parameterisation),
            values=rg.uniform(low=0, high=1, size=manufactured_solution.reparam_values.shape),
        )

        # run the EKI
        enkf.run_filter(
            momentum=initial_momentum,
            parameterisation=initial_parameterisation,
            target=manufactured_solution.target,
            max_iterations=max_iterations,
            reparam=initial_reparam,
            momentum_truth=momentum_truth,
            reparam_truth=manufactured_solution.reparam,
        )
