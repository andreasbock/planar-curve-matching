from pathlib import Path
from typing import List
import numpy as np

import firedrake

from src.enkf import *
from src.shoot_pullback import Diffeomorphism
import src.utils as utils


__all__ = [
    "MANUFACTURED_SOLUTIONS_MOMENTUM",
    "MANUFACTURED_SOLUTIONS_PARAMS",
    "MANUFACTURED_SOLUTIONS_PATH",
    "manufacture_solution",
    "ManufacturedSolution",
]


@dataclass(frozen=True)
class ManufacturedMomentum:
    name: str
    signal: function


MANUFACTURED_SOLUTIONS_PATH = Path("MANUFACTURED_SOLUTIONS")
ManufacturedParameterisation = np.array
_NUM_LANDMARKS = [10, 20, 50]
MANUFACTURED_SOLUTIONS_PARAMS: List[ManufacturedParameterisation] = [
    utils.uniform_parameterisation(n) for n in _NUM_LANDMARKS
]
MANUFACTURED_SOLUTIONS_MOMENTUM = [
    ManufacturedMomentum(name="star", signal=lambda x, y: Constant(70)*cos(2*pi*x/5)),
    ManufacturedMomentum(name="teardrop", signal=lambda x, y: conditional(y < 0, -70*sign(y), 90*exp(-x**2/5))),
    ManufacturedMomentum(name="squeeze", signal=lambda x, y: conditional(x < -0.3, 2*10**2*exp(-(y**2/5)), -40*sin(x/5)*abs(y))),
]


@dataclass(frozen=True)
class ManufacturedSolution:
    diffeomorphism: Diffeomorphism  # this is the diffeomorphism
    target: np.array  # the points on the curve
    momentum: ManufacturedMomentum
    parameterisation: ManufacturedParameterisation

    def dump(self, base_path: Path, shape_function: firedrake.Function) -> None:
        path = base_path / f"MANUFACTURED_SOLUTION_MOMENTUM={self.momentum.name}_LANDMARKS={len(self.parameterisation)}"
        File(path / "target.pvd").write(shape_function)
        utils.pdump(self.diffeomorphism, path / "diffeomorphism")
        utils.pdump(self.target, path / "landmarks")
        utils.pdump(self.momentum, path / "momentum")
        utils.pdump(self.parameterisation, path / "parameterisation")

    @staticmethod
    def load(path: Path) -> "ManufacturedSolution":
        return ManufacturedSolution(
            diffeomorphism=utils.pload(path / "diffeomorphism"),
            landmarks=utils.pload(path / "landmarks"),
            momentum_truth=utils.pload(path / "momentum_truth"),
            thetas_truth=utils.pload(path / "thetas_truth"),
        )


def manufacture_solution(
    forward_operator: GeodesicShooter,
    momentum: ManufacturedMomentum,
    param: ManufacturedParameterisation,
) -> ManufacturedSolution:
    diffeomorphism = forward_operator.shoot(momentum.signal)
    landmarks = forward_operator.evaluate_parameterisation(diffeomorphism, param)

    return ManufacturedSolution(
        diffeomorphism,
        landmarks,
        momentum,
        param,
    )


if __name__ == "__main__":
    logger = utils.Logger(MANUFACTURED_SOLUTIONS_PATH / "manufactured_solutions.log")
    mesh = Mesh("../meshes/mesh0.msh")
    forward_operator = GeodesicShooter(mesh, logger)

    for momentum in MANUFACTURED_SOLUTIONS_MOMENTUM:
        for thetas in MANUFACTURED_SOLUTIONS_PARAMS:
            logger.info(f"Manufacturing solution: '{momentum.name}' with {thetas} landmarks.")
            manufactured_solution = manufacture_solution(forward_operator, momentum, thetas)
            manufactured_solution.dump(MANUFACTURED_SOLUTIONS_PATH, forward_operator.shape_function)
