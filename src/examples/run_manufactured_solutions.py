import numpy as np

from firedrake import File, Mesh, Function, functionspaceimpl

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
    shooting_parameters.time_steps = 15
    shooting_parameters.alpha = 0.5
    shooting_parameters.momentum_degree = 1

    for template in CURVES:
        for resolution in MESH_RESOLUTIONS:
            logger.info(f"Generating mesh for curve: '{template.name}' with resolution: h={resolution}.")
            logger.info(f"Shooting parameters: {shooting_parameters}.")

            mesh_params = MeshGenerationParameters(mesh_size=resolution)
            mesh_path = generate_mesh(mesh_params, template, MANUFACTURED_SOLUTIONS_PATH)

            for momentum in MANUFACTURED_SOLUTIONS_MOMENTUM:
                template_and_momentum_name = f"{mesh_path.stem}_{momentum.name}"
                path = mesh_path.parent / template_and_momentum_name
                path.mkdir(exist_ok=True)

                # logging
                logger.info(f"Logging to `{path}`.")
                shooter = GeodesicShooter(logger, mesh_path, template, shooting_parameters)

                curve_result = shooter.shoot(momentum)
                new_mesh = Mesh(Function(shooter.VCG1).project(shooter.phi))
                indicator_moved = Function(
                    functionspaceimpl.WithGeometry.create(shooter.shape_function.function_space(), new_mesh),
                    val=shooter.shape_function.topological
                )

                indicator_moved_original_mesh = Function(shooter.ShapeSpace).project(indicator_moved)
                utils.my_heaviside(indicator_moved_original_mesh)
                utils.plot_curves(indicator_moved_original_mesh, path / f"{mesh_path.stem}_{momentum.name}.pdf")

                # dump the solution
                mf = ManufacturedSolution(
                    template=template,
                    target=indicator_moved_original_mesh.dat.data_ro,
                    mesh_path=mesh_path,
                    momentum=momentum,
                )
                mf.dump(path)
                logger.info(f"Wrote solution to {path / mf.name()}.")
                File(path / f"{mesh_path.stem}_{momentum.name}.pvd").write(indicator_moved)
                File(path / f"{mesh_path.stem}_{momentum.name}_original_mesh.pvd").write(indicator_moved_original_mesh)
