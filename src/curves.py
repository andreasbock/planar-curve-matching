from typing import List
from scipy.interpolate import CubicSpline
import numpy as np

from firedrake import PeriodicIntervalMesh, VectorFunctionSpace, Function, COMM_WORLD


class Reparameterisation:

    def __init__(self, n_cells, values: np.array = None):
        theta = 2*np.pi*np.linspace(0, 1, n_cells)

        if values is None:
            values = np.zeros(shape=theta.shape)

        values[-1] = values[0]
        self.spline = CubicSpline(theta, values, bc_type='periodic')

    def exponentiate(self, points, time_steps) -> np.array:
        dt = 1 / time_steps
        new_points = points.copy()
        for t in range(time_steps):
            new_points = (new_points + self.spline(new_points) * dt) % (2 * np.pi)
        return new_points

    def at(self, param: np.array) -> np.array:
        return self.spline(param)


class Curve:
    def __init__(self, name: str, points: np.array):
        self.name = name
        self.points = points
        n_cells, dim = self.points.shape
        _points = np.vstack([points, points[0]])
        theta = 2*np.pi*np.linspace(0, 1, n_cells + 1)
        self.spline = CubicSpline(theta, _points, bc_type='periodic')

    @classmethod
    def make(cls, name: str, points: np.array) -> "Curve":
        return Curve(name, points)

    def at(self, param: np.array) -> np.array:
        return self.spline(param)


ALL_CURVES = [
    Curve(
        name="circle",
        points=5 * np.array(
            [
                (np.cos(t), np.sin(t)) for t in np.linspace(0, 2*np.pi, num=25)
            ][:-1]
        ),
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
CURVES = [ALL_CURVES[0]]