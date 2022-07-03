from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Curve:
    name: str
    points: np.array


CURVES: List[Curve] = [
    Curve(
        name="small_triangle",
        points=[[1, 1], [3, 1], [2, 2]],
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
