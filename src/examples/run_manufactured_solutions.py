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
    logger = utils.Logger(MANUFACTURED_SOLUTIONS_PATH / "manufactured_solutions.log")

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
                curve_result = shooter.shoot(momentum.signal)
                template_and_momentum_name = f"{mesh_path.stem}_{momentum.name}"
                path = mesh_path.parent / template_and_momentum_name
                if not path.exists():
                    path.mkdir()

                # logging
                logger.info(f"Logging to `{path}`.")
                utils.plot_norms(curve_result.velocity_norms, curve_result.momentum_norms, shooting_parameters.time_steps, path)
                File(path / f"{mesh_path.stem}_{momentum.name}.pvd").write(shooter.shape_function)
                for parameterisation in MANUFACTURED_SOLUTIONS_PARAMS:
                    try:
                        # TODO: add Xu-Wu point evaluation
                        target = utils.soft_eval(curve_result.diffeo, shooter.VCG, template.at(parameterisation))
                        utils.check_points(mesh_params.min_xy, mesh_params.max_xy, target)
                    except Exception as e:
                        print(e)
                        continue

                    # dump the solution
                    ManufacturedSolution(
                        template=template,
                        target=target,
                        mesh_path=mesh_path,
                        momentum=momentum,
                        parameterisation=parameterisation,
                    ).dump(path)
