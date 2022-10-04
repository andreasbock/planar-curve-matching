import numpy as np

from firedrake import File, SpatialCoordinate

from src import utils
from src.curves import CURVES, Reparameterisation
from src.manufactured_solutions import (
    MANUFACTURED_SOLUTIONS_PATH, MANUFACTURED_SOLUTIONS_MOMENTUM, MESH_RESOLUTIONS,  MANUFACTURED_SOLUTIONS_PARAMS,
    ManufacturedSolution,
)
from src.mesh_generation import MeshGenerationParameters, generate_mesh
from src.shooting import ShootingParameters, GeodesicShooter


if __name__ == "__main__":
    logger = utils.Logger(MANUFACTURED_SOLUTIONS_PATH / "manufactured_solutions.log", )

    shooting_parameters = ShootingParameters()
    shooting_parameters.time_steps = 15
    time_steps_reparam = 10

    for template in CURVES():
        for resolution in MESH_RESOLUTIONS:
            logger.info(f"Generating mesh for curve: '{template.name}' with resolution: h={resolution}.")

            mesh_params = MeshGenerationParameters(mesh_size=resolution)
            mesh_path = generate_mesh(mesh_params, template, MANUFACTURED_SOLUTIONS_PATH)

            for momentum in MANUFACTURED_SOLUTIONS_MOMENTUM:
                # shooting
                logger.info(f"Shooting with `{momentum.name}`.")
                shooter = GeodesicShooter(logger, mesh_path, template, shooting_parameters)

                template_and_momentum_name = f"{mesh_path.stem}_{momentum.name}"
                path = mesh_path.parent / template_and_momentum_name
                path.mkdir(exist_ok=True)
                # logging
                logger.info(f"Logging to `{path}`.")
                for parameterisation in MANUFACTURED_SOLUTIONS_PARAMS:
                    n_cells = len(parameterisation)
                    values = np.random.normal(loc=0, scale=.1, size=n_cells)
                    reparam = Reparameterisation(n_cells, values=values)
                    reparameterised_points = reparam.exponentiate(parameterisation, time_steps_reparam)
                    template_points = template.at(reparameterised_points)

                    curve_result = shooter.shoot(momentum)
                    target = np.array(curve_result.diffeo.at(template_points))
                    momentum_function = shooter.momentum_function()
                    momentum_function.interpolate(momentum.signal(*SpatialCoordinate(shooter.mesh)))

                    # dump the solution
                    mf = ManufacturedSolution(
                        template=template,
                        target=target,
                        mesh_path=mesh_path,
                        momentum=momentum,
                        reparam_values=values,
                        reparam=reparam,
                        parameterisation=reparameterised_points,
                    )
                    mf.dump(path, momentum_function)
                    logger.info(f"Wrote solution to {path / mf.name()}.")

                    # move mesh via linear projection and dump pvd files
                shooter.update_mesh()
                File(path / f"{mesh_path.stem}_{momentum.name}.pvd").write(shooter.shape_function)
