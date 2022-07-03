from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Curve:
    name: str
    points: np.array

    def at(self, angle: float) -> List[float]:
        pass

    def __sub__(self, other) -> List[float]:
        pass


CURVES: List[Curve] = [
    Curve(
        name="small_triangle",
        points=[[1, 1], [3, 1], [2, 2]],
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
        points=[
            [1, 1],
            [6, 1],
            [7, 3],
            [5, 4],
            [1, 7],
            [-4, 4],
            [-2, 4],
            [0, -2]
        ],
    ),
]
