from typing import List
from firedrake import PeriodicIntervalMesh, VectorFunctionSpace, Function, COMM_WORLD

import numpy as np


class Curve:
    def __init__(self, name: str, points: np.array, communicator=COMM_WORLD):
        if len(points) == 0:
            raise Exception("Empty list of points do not define a `Curve`.")

        self.name = name
        self.points = points
        n_cells, dim = self.points.shape
        periodic_mesh = PeriodicIntervalMesh(n_cells, 2*np.pi, comm=communicator)
        V = VectorFunctionSpace(periodic_mesh, "CG", 1, dim=dim)
        self.point_function = Function(V, val=self.points)

    def at(self, param: np.array) -> np.array:
        return self.point_function.at(param)

    @classmethod
    def make(cls, name: str, points: np.array) -> "Curve":
        return Curve(name, points)


def CURVES(communicator=COMM_WORLD) -> List[Curve]:
    return [
        Curve(
            name="circle",
            points=5 * np.array(
                [
                    (np.cos(t), np.sin(t)) for t in np.linspace(0, 2*np.pi, num=25)
                ][:-1]
            ),
            communicator=communicator,
        ),
        Curve(
            name="small_triangle",
            points=np.array([[-4, -4], [6, 2], [4, 4]]),
            communicator=communicator,
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
            communicator=communicator,
        ),
    ]
