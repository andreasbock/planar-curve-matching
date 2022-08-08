import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Any

from firedrake import *
import numpy as np

from src.curves import Curve, CURVE_NAMES
import src.utils as utils


__all__ = [
    "MANUFACTURED_SOLUTIONS_MOMENTUM",
    "MANUFACTURED_SOLUTIONS_PARAMS",
    "MANUFACTURED_SOLUTIONS_PATH",
    "ManufacturedSolution",
]


TEMPLATE_MESHES_PATH = utils.project_root() / "TEMPLATE_MESHES"
MANUFACTURED_SOLUTIONS_PATH = utils.project_root() / "MANUFACTURED_SOLUTIONS"
Parameterisation = np.array
MANUFACTURED_SOLUTIONS_PARAMS: List[Parameterisation] = [
    utils.uniform_parameterisation(n) for n in [10, 20, 50]
]
MESH_RESOLUTIONS = [1 / (2 * h) for h in range(1, 2)]
MomentumFunction = Any #Union[function, Function]


@dataclass(frozen=True)
class Momentum:
    name: str
    signal: MomentumFunction


@dataclass(frozen=True)
class ManufacturedSolution:
    template: Curve
    target: np.array
    mesh_path: Path
    momentum: Momentum
    parameterisation: Parameterisation

    _template_name: str = "template"
    _target_name: str = "target"
    _mesh_name: str = "mesh_path"
    _momentum_name: str = "momentum"
    _parameterisation_name: str = "parameterisation"

    def dump(self, base_path: Path) -> None:
        mesh_name = self.mesh_path.stem
        path = base_path / f"{mesh_name}_{self.momentum.name}_LANDMARKS={len(self.parameterisation)}"
        if not path.parent.exists():
            path.parent.mkdir()

        utils.pdump(self.template, path / self._template_name)
        utils.pdump(self.target, path / self._target_name)
        utils.pdump(self.mesh_path, path / self._mesh_name)
        utils.pdump(self.momentum, path / self._momentum_name)
        utils.pdump(self.parameterisation, path / self._parameterisation_name)
        print(f"Wrote solution to {path}.")

    @classmethod
    def load(cls, base_path: Path) -> "ManufacturedSolution":
        print(f"Loading solution from {base_path}.")
        return ManufacturedSolution(
            template=utils.pload(base_path / cls._template_name),
            target=utils.pload(base_path / cls._target_name),
            mesh_path=utils.pload(base_path / cls._mesh_name),
            momentum=utils.pload(base_path / cls._momentum_name),
            parameterisation=utils.pload(base_path / cls._parameterisation_name),
        )


def get_solutions(
    momentum_names: List[str] = None,
    shape_names: List[str] = None,
    resolutions: List[float] = None,
    landmarks: List[int] = None,
) -> List[ManufacturedSolution]:

    momenta = momentum_names if momentum_names is not None else MANUFACTURED_SOLUTIONS_MOMENTUM_NAMES
    shapes = shape_names if shape_names is not None else CURVE_NAMES
    resolutions = resolutions if resolutions is not None else MESH_RESOLUTIONS
    landmarks = landmarks if landmarks is not None else MANUFACTURED_SOLUTIONS_PARAMS

    solutions = []
    for momentum, shape, res, lms in itertools.product(momenta, shapes, resolutions, landmarks):
        name = f"{shape}_{momentum}"
        solution_path = MANUFACTURED_SOLUTIONS_PATH / f"h={res}" / name / f"{name}_LANDMARKS={lms}"
        solutions.append(ManufacturedSolution.load(solution_path))
    return solutions


def get_solution(
    momentum_name: str,
    shape_name: str,
    resolution: float,
    landmarks: int,
) -> ManufacturedSolution:
    name = f"{shape_name}_{momentum_name}"
    solution_path = MANUFACTURED_SOLUTIONS_PATH / f"h={resolution}" / name / f"{name}_LANDMARKS={landmarks}"
    return ManufacturedSolution.load(solution_path)


def _expand(x, y):
    return 50.0/72.


def _contract(x, y):
    return -50./72.


def _star(x, y):
    return 40/72. * cos(2 * pi * x / 5)


def _teardrop(x, y):
    return conditional(y < 0, -60/72. * sign(y), 60/72. * exp(-x ** 2 / 5))


def _squeeze(x, y):
    return conditional(x < -0.3, 30/72. * exp(-(y ** 2 / 5)), -60/72. * sin(x / 5) * abs(y))


MANUFACTURED_SOLUTIONS_MOMENTUM = [
    Momentum(name="expand", signal=_expand),
    Momentum(name="contract", signal=_contract),
    Momentum(name="star", signal=_star),
    Momentum(name="teardrop", signal=_teardrop),
    Momentum(name="squeeze", signal=_squeeze),
]
MANUFACTURED_SOLUTIONS_MOMENTUM_NAMES = [m.name for m in MANUFACTURED_SOLUTIONS_MOMENTUM]
