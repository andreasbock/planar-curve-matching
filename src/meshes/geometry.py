from dataclasses import dataclass
import numpy as np

from pathlib import Path


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
    points: np.array
    curve_tag: int = 10
    inner_tag: int = 6
    outer_tag: int = 7


def write_geo_file(
    min_xy: int,
    max_xy: int,
    mesh_size: float,
    curve: Curve,
    geo_path: Path,
):
    n = len(curve.points)
    points = ""
    for i, point in enumerate(curve.points):
        x, y = point
        points += f"Point({i + _OFFSET}) = {{{x}, {y}, 0, {mesh_size}}};\n"

    lines = ""
    for i in range(n):
        j = i + _OFFSET
        lines += f"Line({j}) = {{ {j}, {(i+1) % (_OFFSET - 2) + _OFFSET} }};\n"

    loop_array = "{" + ",".join([str(i + _OFFSET) for i in range(n)]) + "}"

    txt = _RAW_GEOMETRY.format(
        min_xy=min_xy,
        max_xy=max_xy,
        mesh_size=mesh_size,
        POINTS=points,
        LINES=lines,
        LOOP_ARRAY=loop_array,
        CURVE_TAG=curve.curve_tag,
        INNER_TAG=curve.inner_tag,
        OUTER_TAG=curve.outer_tag,
    )

    f = open(geo_path, "w")
    f.write(txt)
    f.close()
