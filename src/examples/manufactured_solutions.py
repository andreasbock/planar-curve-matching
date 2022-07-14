from dataclasses import dataclass
from pathlib import Path
from typing import List
from firedrake import *
import numpy as np

import src.mesh_generation
from src.curves import CURVES
from src.mesh_generation.mesh_generation import MeshGenerationParameters
from src.shoot_pullback import GeodesicShooter, ShootingParameters
import src.utils as utils


__all__ = [
    "MANUFACTURED_SOLUTIONS_MOMENTUM",
    "MANUFACTURED_SOLUTIONS_PARAMS",
    "MANUFACTURED_SOLUTIONS_PATH",
    "ManufacturedSolution",
]


@dataclass(frozen=True)
class ManufacturedMomentum:
    name: str
    signal: function


TEMPLATE_MESHES_PATH = utils.project_root() / "TEMPLATE_MESHES"
MANUFACTURED_SOLUTIONS_PATH = utils.project_root() / "MANUFACTURED_SOLUTIONS"
ManufacturedParameterisation = np.array
MANUFACTURED_SOLUTIONS_PARAMS: List[ManufacturedParameterisation] = [
    utils.uniform_parameterisation(n) for n in [10]#, 20, 50]
]


def _expand(x, y):
    return 5


def _contract(x, y):
    return -5


def _star(x, y):
    return 4 * cos(2 * pi * x / 5)


def _teardrop(x, y):
    return conditional(y < 0, -6 * sign(y), 6 * exp(-x ** 2 / 5))


def _squeeze(x, y):
    return conditional(x < -0.3, 3* exp(-(y ** 2 / 5)), -6 * sin(x / 5) * abs(y))


MANUFACTURED_SOLUTIONS_MOMENTUM = [
    ManufacturedMomentum(name="star", signal=_star),
    ManufacturedMomentum(name="teardrop", signal=_teardrop),
    ManufacturedMomentum(name="squeeze", signal=_squeeze),
    ManufacturedMomentum(name="expand", signal=_expand),
    ManufacturedMomentum(name="contract", signal=_contract),
]


@dataclass(frozen=True)
class ManufacturedSolution:
    target: np.array
    mesh_path: Path
    momentum: ManufacturedMomentum
    parameterisation: ManufacturedParameterisation

    def dump(self, base_path: Path, mesh_name: str, shape_function: Function) -> None:
        path = base_path / f"{mesh_name}_{self.momentum.name}_LANDMARKS={len(self.parameterisation)}"
        if not path.parent.exists():
            path.parent.mkdir()
        utils.pdump(self.momentum, path / "momentum")
        utils.pdump(self.parameterisation, path / "parameterisation")
        utils.pdump(self.target, path / "landmarks_target")
        File(path / F"{mesh_name}_{self.momentum.name}.pvd").write(shape_function)
        print(f"Wrote solution to {path}.")

    @staticmethod
    def load(momenta_name: str, resolution: Path) -> "ManufacturedSolution":
        raise NotImplemented


if __name__ == "__main__":
    logger = utils.Logger(MANUFACTURED_SOLUTIONS_PATH / "manufactured_solutions.log")

    shooting_parameters = ShootingParameters()
    MESH_RESOLUTIONS = [1 / (2 * h) for h in range(1, 2)]

    for template in CURVES:
        for resolution in MESH_RESOLUTIONS:
            logger.info(f"Generating mesh for curve: '{template.name}' with resolution: h={resolution}.")

            mesh_params = MeshGenerationParameters(mesh_size=resolution)
            mesh_path = src.mesh_generation.generate_mesh(mesh_params, template, MANUFACTURED_SOLUTIONS_PATH)

            for momentum in MANUFACTURED_SOLUTIONS_MOMENTUM:
                shooter = GeodesicShooter(logger, mesh_path, shooting_parameters)
                diffeo, us = shooter.shoot(momentum.signal)

                # start some logging
                template_and_momentum_name = f"{mesh_path.stem}_{momentum.name}"
                path = mesh_path.parent / template_and_momentum_name
                if not path.parent.exists():
                    path.parent.mkdir()

                utils.pdump(momentum, path / "momentum")
                File(path / f"{template_and_momentum_name}.pvd").write(shooter.shape_function)

                # log evaluation
                for parameterisation in MANUFACTURED_SOLUTIONS_PARAMS:
                    try:
                        target = utils.soft_eval(diffeo, shooter.VCG, template.at(parameterisation))
                        utils.check_points(mesh_params.min_xy, mesh_params.max_xy, target)
                    except Exception as e:
                        print(e)
                        continue
                    utils.pdump(parameterisation, path / f"parameterisation_m={len(parameterisation)}")
                    utils.pdump(target, path / f"target_m={len(parameterisation)}")

