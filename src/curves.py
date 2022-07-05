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
        dists = [0.] + [dist(self.points[i] - self.points[(i + 1) % self.n]) for i in range(self.n)]
        self._plist = 2*np.pi*np.cumsum(dists / np.sum(dists))

    def at(self, param: np.array) -> np.array:
        return list(map(self._at, param))

    def _at(self, angle: float):
        if angle < 0 or 2*np.pi < angle:
            raise Exception(f"Angle {angle} not in [0, 2π].")
        i = None  # PyCharm complains
        for i, e in enumerate(self._plist):
            if angle < e:
                break
        i_next = (i + 1) % self.n
        w = (angle - self._plist[i]) / self.points[i_next]
        start = self.points[i]
        stop = self.points[i_next]
        return w * start + (1 - w) * stop

    def __sub__(self, other) -> List[float]:
        pass


CURVES: List[Curve] = [
    Curve(
        name="small_triangle",
        points=np.array([[1, 1], [3, 1], [2, 2]]),
    ),
    Curve(
        name="circle",
        points=5 * np.array(
            [
                (np.cos(t), np.sin(t)) for t in np.linspace(0, 2*np.pi, num=25)
            ]
        )
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

if __name__ == "__main__":
    curves = None