from firedrake import File

from src import utils
from src.curves import CURVES
from src.manufactured_solutions import (
    MANUFACTURED_SOLUTIONS_PATH, MANUFACTURED_SOLUTIONS_MOMENTUM, MESH_RESOLUTIONS,
    ManufacturedSolution,
)
from src.mesh_generation import MeshGenerationParameters, generate_mesh
from src.shooting import ShootingParameters, GeodesicShooter


if __name__ == "__main__":
    logger = utils.Logger(MANUFACTURED_SOLUTIONS_PATH / "manufactured_solutions.log", )

    shooting_parameters = ShootingParameters()
    shooting_parameters.time_steps = 20
    shooting_parameters.alpha = 0.5
    shooting_parameters.momentum_degree = 0
    shooting_parameters.kappa = 1

    for template in CURVES:
        for resolution in MESH_RESOLUTIONS:
            logger.info(f"Generating mesh for curve: '{template.name}' with resolution: h={resolution}.")
            logger.info(f"Shooting parameters: {shooting_parameters}.")

            mesh_params = MeshGenerationParameters(mesh_size=resolution)
            mesh_path = generate_mesh(mesh_params, template, MANUFACTURED_SOLUTIONS_PATH)

            for momentum in MANUFACTURED_SOLUTIONS_MOMENTUM:
                path = mesh_path.parent / f"{mesh_path.stem}_{momentum.name}" / f"kappa={shooting_parameters.kappa}"
                path.mkdir(exist_ok=True, parents=True)
                logger.info(f"Logging to `{path}`.")

                # shoot
                shooter = GeodesicShooter(logger, mesh_path, template, shooting_parameters)
                shooter.shoot(momentum)
                indicator_moved_original_mesh = shooter.smooth_shape_function_initial_mesh()
                utils.plot_curves(shooter.shape_function, path / f"{mesh_path.stem}_{momentum.name}.pdf")

                # dump the solution
                mf = ManufacturedSolution(
                    template=template,
                    target=indicator_moved_original_mesh.dat.data_ro,
                    mesh_path=mesh_path,
                    momentum=momentum,
                )
                mf.dump(path)
                logger.info(f"Wrote solution to {path / mf.name()}.")
                File(path / f"{mesh_path.stem}_{momentum.name}.pvd").write(shooter.shape_function)
                File(path / f"{mesh_path.stem}_{momentum.name}_smooth_original_mesh.pvd").write(
                    indicator_moved_original_mesh
                )
