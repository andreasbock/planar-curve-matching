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
        shapes=['circle'],
        momentum_names=['expand', 'teardrop', 'star', 'squeeze'],
        resolutions=[resolution],
        landmarks=[num_observations],
        communicator=ensemble_object.comm,
    )

    # run EKI over all manufactured solutions
    for manufactured_solution in manufactured_solutions:
        # set up logging
        logger = utils.Logger(
            PATH / manufactured_solution.name() / "example_enkf.log",
            ensemble_object.comm,
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

        # perturb momentum
        pcg = randomfunctiongen.PCG64(seed=12315123)
        rg = randomfunctiongen.Generator(pcg)

        random_part = rg.normal(enkf.forward_operator.DGT)
        x, y = SpatialCoordinate(enkf.forward_operator.mesh)
        momentum_truth = enkf.forward_operator.momentum_function().interpolate(manufactured_solution.momentum.signal(x, y))
        initial_momentum = enkf.forward_operator.momentum_function().assign(random_part)

        # perturb parameterisation
        initial_parameterisation = (
                + rg.uniform(low=0, high=2*np.pi, size=manufactured_solution.parameterisation.shape)
        ) % (2 * np.pi)
        initial_parameterisation.sort()

        initial_reparam = Reparameterisation(
            n_cells=len(manufactured_solution.parameterisation),
            values=rg.uniform(low=0, high=1, size=manufactured_solution.reparam_values.shape),
        )

        # run the EKI
        enkf.run_filter(
            momentum=initial_momentum,
            parameterisation=initial_parameterisation,
            reparam=initial_reparam,
            target=manufactured_solution.target,
            max_iterations=max_iterations,
            momentum_truth=momentum_truth,
            reparam_truth=manufactured_solution.reparam_values,
            param_truth=manufactured_solution.parameterisation,
        )


if __name__ == "__main__":
    convergence_experiment()
