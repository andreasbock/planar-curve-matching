import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Any

from firedrake import *
import numpy as np

import src.utils as utils
from src.plot_pickles import target_marker, target_linestyle, plot_landmarks, plot_initial_data
from src.curves import Curve, Reparameterisation

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
    utils.uniform_parameterisation(n) for n in [10, 20]
]
MESH_RESOLUTIONS = [1 / (2 * h) for h in range(1, 2)]
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
    parameterisation: Parameterisation
    reparam_values: np.array
    reparam: Reparameterisation

    _template_file_name: str = "template"
    _curve_name: str = "curve_name"
    _target_name: str = "target"
    _mesh_name: str = "mesh_path"
    _momentum_name: str = "momentum"
    _parameterisation_name: str = "parameterisation"
    _reparam_values_name: str = "reparam_values"
    _reparam_name: str = "reparam_coefs"

    def name(self) -> str:
        return f"{self.mesh_path.stem}_{self.momentum.name}_LANDMARKS={len(self.parameterisation)}"

    def dump(self, base_path: Path, momentum_function=None) -> None:
        path = base_path / self.name()
        if not path.parent.exists():
            path.parent.mkdir()

        utils.pdump(self.template.points, path / self._template_file_name)
        utils.pdump(self.template.name, path / self._curve_name)
        utils.pdump(self.target, path / self._target_name)
        utils.pdump(self.mesh_path, path / self._mesh_name)
        utils.pdump(self.momentum, path / self._momentum_name)
        utils.pdump(self.parameterisation, path / self._parameterisation_name)
        utils.pdump(self.reparam_values, path / self._reparam_values_name)
        utils.pdump(self.reparam.spline.c, path / self._reparam_name)

        plot_landmarks(
            self.target,
            label='Target',
            marker=target_marker,
            linestyle=target_linestyle,
            path=path / 'target.pdf',
        )

        xs = utils.uniform_parameterisation(100)
        ns = self.reparam.at(xs)
        ms = momentum_function.at(self.template.at(xs)) if momentum_function is not None else None
        plot_initial_data(path / 'reparam_and_momentum.pdf', xs, ns, ms)

    @classmethod
    def load(cls, base_path: Path) -> "ManufacturedSolution":
        parameterisation = utils.pload(base_path / cls._parameterisation_name)
        reparam = Reparameterisation(n_cells=len(parameterisation))
        reparam.spline.c = utils.pload(base_path / cls._reparam_name)

        return ManufacturedSolution(
            template=Curve(
                name=utils.pload(base_path / cls._curve_name),
                points=utils.pload(base_path / cls._template_file_name),
            ),
            target=utils.pload(base_path / cls._target_name),
            mesh_path=utils.pload(base_path / cls._mesh_name),
            momentum=utils.pload(base_path / cls._momentum_name),
            parameterisation=parameterisation,
            reparam_values=utils.pload(base_path / cls._reparam_values_name),
            reparam=reparam,
        )


def get_solutions(
    shapes: List[str],
    momentum_names: List[str] = None,
    resolutions: List[float] = None,
    landmarks: List[int] = None,
) -> List[ManufacturedSolution]:
    momenta = momentum_names if momentum_names is not None else MANUFACTURED_SOLUTIONS_MOMENTUM_NAMES
    resolutions = resolutions if resolutions is not None else MESH_RESOLUTIONS
    landmarks = landmarks if landmarks is not None else MANUFACTURED_SOLUTIONS_PARAMS

    solutions = []
    for momentum, shape, res, lms in itertools.product(momenta, shapes, resolutions, landmarks):
        solution = get_solution(momentum, shape, res, lms)
        solutions.append(solution)
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
    return Constant(50.0/72.)


def _contract(x, y):
    return Constant(-50./72.)


def _star(x, y):
    return Constant(1.3) * cos(2 * pi * x / 5)


def _teardrop(x, y):
    return conditional(y < 0, -60/40. * sign(y), 60/40. * exp(-x ** 2 / 5))


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
