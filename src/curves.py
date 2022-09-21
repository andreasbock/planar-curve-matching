from typing import List
from scipy.interpolate import CubicSpline
import numpy as np

from firedrake import PeriodicIntervalMesh, VectorFunctionSpace, Function, COMM_WORLD


class Reparameterisation:

    def __init__(self, n_cells, values: np.array = None):
        theta = 2*np.pi*np.linspace(0, 1, n_cells)

        if values is None:
            values = theta.copy()

        values[-1] = values[0]
        self.spline = CubicSpline(theta, values, bc_type='periodic')

    def exponentiate(self, points, time_steps) -> np.array:
        dt = 1 / time_steps
        new_points = points.copy()
        for t in range(time_steps):
            new_points = (new_points + self.spline(new_points) * dt)
        return new_points % (2 * np.pi)

    def at(self, param: np.array) -> np.array:
        return self.spline(param)


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

    @classmethod
    def make(cls, name: str, points: np.array) -> "Curve":
        return Curve(name, points)

    def at(self, param: np.array) -> np.array:
        return self.point_function.at(param)


def CURVES(communicator=COMM_WORLD) -> List[Curve]:
    circle = Curve(
        name="circle",
        points=5 * np.array(
            [
                (np.cos(t), np.sin(t)) for t in np.linspace(0, 2*np.pi, num=25)
            ][:-1]
        ),
        communicator=communicator,
    )
    small_triangle = Curve(
        name="small_triangle",
        points=np.array([[-4, -4], [6, 2], [4, 4]]),
        communicator=communicator,
    )
    random_shape = Curve(
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
    )
    return [circle]