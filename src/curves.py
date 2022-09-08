from typing import List
from firedrake import PeriodicIntervalMesh, VectorFunctionSpace, Function

import numpy as np
from numpy.linalg import norm as dist


class Curve:
    def __init__(self, name: str, points: np.array):
        if len(points) == 0:
            raise Exception("Empty list of points do not define a `Curve`.")

        self.name = name
        self.points = points
        self.n = len(points)
        periodic_mesh = PeriodicIntervalMesh(self.n, 2*np.pi)
        V = VectorFunctionSpace(periodic_mesh, "CG", 1, dim=2)
        self.point_function = Function(V, val=points)

    def at(self, param: np.array) -> np.array:
        return self.point_function.at(param)


CURVES: List[Curve] = [
    Curve(
        name="circle",
        points=5 * np.array(
            [
                (np.cos(t), np.sin(t)) for t in np.linspace(0, 2*np.pi, num=25)
            ][:-1]
        )
    ),
    Curve(
        name="small_triangle",
        points=np.array([[-4, -4], [6, 2], [4, 4]]),
    ),
    Curve(
        name="random_shape",
        points=np.array(
            [
                [1, 1],
                [6, 1],
                [7, 3],
                [5, 4],
                [1, 7],
                [-4, 4],
                [-2, 4],
                [0, -2]
            ]
        ),
    ),
]
CURVE_NAMES = [c.name for c in CURVES]