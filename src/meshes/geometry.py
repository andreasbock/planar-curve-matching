from dataclasses import dataclass
import numpy as np

from pathlib import Path

__all__ = ["Curve", "MeshGenerationParameters", "write_geo_file"]


_OFFSET = 5
_RAW_GEOMETRY = """
inside_tag = {INNER_TAG};
outside_tag = {OUTER_TAG};
curve_tag = {CURVE_TAG};

h = {mesh_size};
min_xy = {min_xy};
max_xy = {max_xy};

Point(1) = {{min_xy, min_xy, 0, h}};
Point(2) = {{max_xy, min_xy, 0, h}};
Point(3) = {{max_xy, max_xy, 0, h}};
Point(4) = {{min_xy, max_xy, 0, h}};

Line(1) = {{1, 2}};
Line(2) = {{3, 2}};
Line(3) = {{3, 4}};
Line(4) = {{4, 1}};

Physical Line(1) = {{1}};
Physical Line(2) = {{2}};
Physical Line(3) = {{3}};
Physical Line(4) = {{4}};

Line Loop(1) = {{4, 1, -2, 3}};

/* CUSTOM POINTS BELOW */
{POINTS}
/* CUSTOM POINTS ABOVE */
/* CUSTOM LINES BELOW */
{LINES}
Line Loop(2) = {LOOP_ARRAY};
/* CUSTOM LINES ABOVE */

// Create surfaces inside and outside the curve
Plane Surface(1) = {{1, 2}};  // outer + inner
Plane Surface(2) = {{2}};  // inner

// Tag the loop, inside and outside
Physical Line(curve_tag) = {{5, 6 ,7}};  // loop
Physical Surface(17) = {{1}};  // inner loop
Physical Surface(18) = {{2}};
"""


@dataclass
class Curve:
    name: str
    points: np.array


@dataclass
class MeshGenerationParameters:
    min_xy: int = -10
    max_xy: int = 10
    mesh_size: float = .25
    curve_tag: int = 10
    inner_tag: int = 6
    outer_tag: int = 7


def write_geo_file(
    params: MeshGenerationParameters,
    curve: Curve,
    path_to_geo: Path,
):
    n = len(curve.points)
    points = ""
    for i, point in enumerate(curve.points):
        x, y = point
        points += f"Point({i + _OFFSET}) = {{{x}, {y}, 0, h}};\n"

    lines = ""
    for i in range(n):
        j = i + _OFFSET
        lines += f"Line({j}) = {{ {j}, {(i+1) % (_OFFSET - 2) + _OFFSET} }};\n"

    loop_array = "{" + ",".join([str(i + _OFFSET) for i in range(n)]) + "}"

    txt = _RAW_GEOMETRY.format(
        min_xy=params.min_xy,
        max_xy=params.max_xy,
        mesh_size=params.mesh_size,
        CURVE_TAG=params.curve_tag,
        INNER_TAG=params.inner_tag,
        OUTER_TAG=params.outer_tag,
        POINTS=points,
        LINES=lines,
        LOOP_ARRAY=loop_array,
    )

    path_to_geo.parent.mkdir(exist_ok=True, parents=True)
    path_to_geo.write_text(txt)
