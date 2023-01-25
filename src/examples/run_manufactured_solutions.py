import numpy as np

from firedrake import File, Mesh, Function, FunctionSpace

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
    shooting_parameters.momentum_degree = 0

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
                logger.info(f"Logging to `{path}`.")

                # shoot
                shooter = GeodesicShooter(logger, mesh_path, template, shooting_parameters)
                curve_result = shooter.shoot(momentum)

                # set up original mesh & function space
                original_mesh = Mesh(shooter.orig_coords)
                Lagrange_original_mesh = FunctionSpace(original_mesh, "CG", shooter.order_XW)

                # evaluate the moved indicator on the original mesh
                indicator_moved_original_mesh = Function(
                    Lagrange_original_mesh,
                    shooter.shape_function.at(
                        original_mesh.coordinates.dat.data_ro,
                        tolerance=1e-03,
                        dont_raise=True,
                    )
                )
                indicator_moved_original_mesh.dat.data[:] = np.nan_to_num(
                    indicator_moved_original_mesh.dat.data[:],
                    nan=1.0,
                )
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
                File(path / f"{mesh_path.stem}_{momentum.name}.pvd").write(shooter.shape_function)
                File(path / f"{mesh_path.stem}_{momentum.name}_original_mesh.pvd").write(indicator_moved_original_mesh)
