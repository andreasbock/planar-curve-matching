from pathlib import Path
from typing import Any, List, Tuple

import firedrake

from src.enkf import *
from src.shoot_pullback import Diffeomorphism
import src.utils as utils

__all__ = ["MANUFACTURED_SOLUTIONS_MOMENTUM", "MANUFACTURED_SOLUTIONS_THETAS", "manufacture_solution"]


class ManufacturedMomentum:
    name: str
    signal: firedrake.Function


ManufacturedParameterisation = List[np.array]
_NUM_LANDMARKS = [10, 20, 50]
MANUFACTURED_SOLUTIONS_THETAS: List[ManufacturedParameterisation] = [
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
    landmarks: np.array  # the points on the curve
    momentum_truth: ManufacturedMomentum
    thetas_truth: ManufacturedParameterisation

    def dump(self, base_path: Path, shape_function: firedrake.Function):
        path = base_path / f"MOMENTUM={self.momentum_truth.name}_LANDMARKS={len(self.thetas_truth)}"
        File(path / "target.pvd").write(shape_function)

        utils.pdump(self.diffeomorphism, path / "diffeomorphism")
        utils.pdump(self.landmarks, path / "landmarks")
        utils.pdump(self.momentum_truth, path / "momentum_truth")
        utils.pdump(self.thetas_truth, path / "thetas_truth")


def manufacture_solution(
        enkf: EnsembleKalmanFilter,
        momentum: ManufacturedMomentum,
        thetas: np.array,
) -> ManufacturedSolution:
    momentum_name, momentum_signal = momentum

    # build true momentum from the average of the ensemble
    momentum_truth = enkf.forward_operator.momentum_function()
    enkf.ensemble.allreduce(momentum_signal, momentum_truth)
    momentum_truth.assign(momentum_truth / enkf.ensemble_size)

    # build true parameterisation from the average
    thetas_truth = thetas
    # TODO: build average!

    # shoot to get manufactured target, q1
    diffeomorphism = enkf.forward_operator.shoot(momentum_truth)
    landmarks = enkf.forward_operator.evaluate_parameterisation(diffeomorphism, thetas_truth)

    return ManufacturedSolution(
        diffeomorphism,
        landmarks,
        momentum_truth,
        thetas_truth,
    )


if __name__=="__main__":
    # set up logger, ensemble, shooter and filter
    base_path = Path("MANUFACTURED_SOLUTIONS")
    logger_path = base_path / "manufactured_solutions.log"
    logger = utils.Logger(logger_path)

    mesh = Mesh("meshes/mesh0.msh")
    forward_operator = GeodesicShooter(mesh, logger)
    enkf = EnsembleKalmanFilter(Ensemble(COMM_WORLD, M=1), forward_operator, InverseProblemParameters(), logger)

    for momentum in MANUFACTURED_SOLUTIONS_MOMENTUM:
        for thetas in MANUFACTURED_SOLUTIONS_THETAS:
            manufactured_solution = manufacture_solution(enkf, momentum, thetas)
            manufactured_solution.dump(base_path, enkf.forward_operator.shape_function)
