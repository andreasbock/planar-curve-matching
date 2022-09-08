import numpy as np

from firedrake import File

from src import utils
from src.curves import CURVES
from src.manufactured_solutions import (
    MANUFACTURED_SOLUTIONS_PATH, MANUFACTURED_SOLUTIONS_MOMENTUM, MESH_RESOLUTIONS,  MANUFACTURED_SOLUTIONS_PARAMS,
    ManufacturedSolution,
)
from src.mesh_generation import MeshGenerationParameters, generate_mesh
from src.shooting import ShootingParameters, GeodesicShooter


if __name__ == "__main__":
    logger = utils.Logger(MANUFACTURED_SOLUTIONS_PATH / "manufactured_solutions.log", )

    shooting_parameters = ShootingParameters()

    for template in CURVES:
        for resolution in MESH_RESOLUTIONS:
            logger.info(f"Generating mesh for curve: '{template.name}' with resolution: h={resolution}.")

            mesh_params = MeshGenerationParameters(mesh_size=resolution)
            mesh_path = generate_mesh(mesh_params, template, MANUFACTURED_SOLUTIONS_PATH)

            for momentum in MANUFACTURED_SOLUTIONS_MOMENTUM:
                # shooting
                logger.info(f"Shooting with `{momentum.name}`.")
                shooter = GeodesicShooter(logger, mesh_path, template, shooting_parameters)
                curve_result = shooter.shoot(momentum)
                template_and_momentum_name = f"{mesh_path.stem}_{momentum.name}"
                path = mesh_path.parent / template_and_momentum_name
                path.mkdir(exist_ok=True)
                # logging
                logger.info(f"Logging to `{path}`.")

                for parameterisation in MANUFACTURED_SOLUTIONS_PARAMS:
                    template_points = template.at(parameterisation)
                    target = np.array(curve_result.diffeo.at(template_points))

                    # dump the solution
                    ManufacturedSolution(
                        template=template_points,
                        target=target,
                        mesh_path=mesh_path,
                        momentum=momentum,
                        parameterisation=parameterisation,
                    ).dump(path)
                    logger.info(f"Wrote solution to {path}.")

                    # move mesh via linear projection and dump pvd files
                shooter.update_mesh()
                File(path / f"{mesh_path.stem}_{momentum.name}.pvd").write(shooter.shape_function)
