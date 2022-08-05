from typing import List

import numpy as np
from numpy.linalg import norm as dist


class Curve:
    def __init__(self, name: str, points: np.array):
        if len(points) == 0:
            raise Exception("Empty list of points do not define a `Curve`.")

        self.name = name
        self.points = points
        self.n = len(points)
        dists = [dist(self.points[i] - self.points[(i + 1) % self.n]) for i in range(self.n)]
        self._plist = 2*np.pi*np.cumsum(dists / np.sum(dists))

    def at(self, param: np.array) -> np.array:
        return list(map(self._at, param))

    def _at(self, angle: float):
        if angle < 0 or 2*np.pi < angle:
            raise Exception(f"Angle {angle} not in [0, 2π].")
        e_prev = 0
        for i, e in enumerate(self._plist):
            if angle < e:
                break
            e_prev = e
        w = (angle - e_prev) / e
        start = self.points[(i - 1) % self.n]
        stop = self.points[i % self.n]
        return w * start + (1 - w) * stop


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