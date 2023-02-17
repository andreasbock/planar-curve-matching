import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Any

from firedrake import *
import numpy as np

import src.utils as utils
from src.curves import Curve

__all__ = [
    "MANUFACTURED_SOLUTIONS_MOMENTUM",
    "MANUFACTURED_SOLUTIONS_PATH",
    "ManufacturedSolution",
]


TEMPLATE_MESHES_PATH = utils.project_root() / "TEMPLATE_MESHES"
MANUFACTURED_SOLUTIONS_PATH = utils.project_root() / "MANUFACTURED_SOLUTIONS"
MESH_RESOLUTIONS = [1]
MomentumFunction = Any


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

    _template_file_name: str = "template"
    _curve_name: str = "curve_name"
    _target_name: str = "target"
    _mesh_name: str = "mesh_path"
    _momentum_name: str = "momentum"

    def name(self) -> str:
        return f"{self.mesh_path.stem}_{self.momentum.name}"

    def dump(self, path: Path) -> None:
        utils.pdump(self.template.points, path / self._template_file_name)
        utils.pdump(self.template.name, path / self._curve_name)
        utils.pdump(self.target, path / self._target_name)
        utils.pdump(self.mesh_path, path / self._mesh_name)
        utils.pdump(self.momentum, path / self._momentum_name)

    @classmethod
    def load(cls, base_path: Path) -> "ManufacturedSolution":

        return ManufacturedSolution(
            template=Curve(
                name=utils.pload(base_path / cls._curve_name),
                points=utils.pload(base_path / cls._template_file_name),
            ),
            target=utils.pload(base_path / cls._target_name),
            mesh_path=utils.pload(base_path / cls._mesh_name),
            momentum=utils.pload(base_path / cls._momentum_name),
        )


def get_solutions(
    shapes: List[str],
    kappa: float,
    momentum_names: List[str] = None,
    resolutions: List[float] = None,
) -> List[ManufacturedSolution]:
    momenta = momentum_names if momentum_names is not None else MANUFACTURED_SOLUTIONS_MOMENTUM_NAMES
    resolutions = resolutions if resolutions is not None else MESH_RESOLUTIONS

    solutions = []
    for momentum, shape, res in itertools.product(momenta, shapes, resolutions):
        solution = get_solution(momentum, shape, res, kappa)
        solutions.append(solution)
    return solutions


def get_solution(
    momentum_name: str,
    shape_name: str,
    resolution: float,
    kappa: float,
) -> ManufacturedSolution:
    name = f"{shape_name}_{momentum_name}"
    smoothness = f"kappa={kappa}"
    solution_path = MANUFACTURED_SOLUTIONS_PATH / f"h={resolution}" / name / smoothness
    return ManufacturedSolution.load(solution_path)


def _expand(x, y):
    return Constant(2*pi) * Constant(50.0/72.)


def _contract(x, y):
    return Constant(2*pi) * Constant(-50./72.)


def _star(x, y):
    return Constant(2*pi) * Constant(1.3) * cos(2 * pi * x / 5)


def _teardrop(x, y):
    return Constant(2*pi) * conditional(y < 0, -60/40. * sign(y), 60/40. * exp(-x ** 2 / 5))


def _squeeze(x, y):
    return Constant(2*pi) * conditional(x < -0.3, 30/72. * exp(-(y ** 2 / 5)), -60/72. * sin(x / 5) * abs(y))


MANUFACTURED_SOLUTIONS_MOMENTUM = [
    Momentum(name="expand", signal=_expand),
    Momentum(name="contract", signal=_contract),
    Momentum(name="star", signal=_star),
    Momentum(name="teardrop", signal=_teardrop),
    Momentum(name="squeeze", signal=_squeeze),
]
MANUFACTURED_SOLUTIONS_MOMENTUM_NAMES = [m.name for m in MANUFACTURED_SOLUTIONS_MOMENTUM]
