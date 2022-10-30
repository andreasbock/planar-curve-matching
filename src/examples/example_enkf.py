from src.enkf import *
from src.shooting import ShootingParameters
from src.manufactured_solutions import get_solutions

if __name__ == "__main__":
    # parameters
    max_iterations = 10
    shooting_parameters = ShootingParameters()
    inverse_problem_parameters = InverseProblemParameters()
    inverse_problem_parameters.optimise_momentum = True
    inverse_problem_parameters.optimise_parameterisation = True

    process_per_ensemble_member = 1
    ensemble_object = Ensemble(COMM_WORLD, M=process_per_ensemble_member)

    EXAMPLES_ENKF_PATH = utils.project_root() / f"RESULTS_EXAMPLES_ENKF_ESIZE={ensemble_object.ensemble_comm.size}"
    EXAMPLES_ENKF_PATH.mkdir(exist_ok=True)

    manufactured_solutions = get_solutions(
        shapes=['circle'],
        momentum_names=['contract', 'squeeze', 'teardrop'],
        resolutions=[0.5],
        landmarks=[20],
    )

    # run EKI over all manufactured solutions
    for manufactured_solution in manufactured_solutions:
        # set up logging
        path = EXAMPLES_ENKF_PATH / manufactured_solution.name() / f"REALISATION_{utils.date_string()}"
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
        enkf._info(f"Loaded solution: '{manufactured_solution.name()}'.")

        # perturb momentum
        pcg = randomfunctiongen.PCG64()
        rg = randomfunctiongen.Generator(pcg)
        random_part = rg.uniform(enkf.forward_operator.MomentumSpace, -4, 4)

        x, y = SpatialCoordinate(enkf.forward_operator.mesh)
        momentum_truth = enkf.forward_operator.momentum_function().interpolate(manufactured_solution.momentum.signal(x, y))
        initial_momentum = enkf.forward_operator.momentum_function().assign(random_part)

        # perturb reparam
        parameterisation = manufactured_solution.parameterisation
        initial_reparam = Reparameterisation(
            n_cells=len(manufactured_solution.parameterisation),
            values=rg.uniform(low=-1, high=1, size=manufactured_solution.reparam_values.shape),
        )

        if enkf.inverse_problem_params.optimise_momentum:
            target = manufactured_solution.target
        else:
            target = np.array(manufactured_solution.template.at(manufactured_solution.parameterisation))

        # run the EKI
        enkf.run_filter(
            momentum=initial_momentum,
            parameterisation=parameterisation,
            target=target,
            max_iterations=max_iterations,
            reparam=initial_reparam,
            momentum_truth=momentum_truth,
            reparam_truth=manufactured_solution.reparam,
        )
