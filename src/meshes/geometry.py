__all__ = ["_OFFSET", "_RAW_GEOMETRY"]


_OFFSET = 5
_RAW_GEOMETRY = """
inside_tag = {INNER_TAG};
outside_tag = {OUTER_TAG};
curve_tag = {CURVE_TAG};
n = {NUMBER_OF_POINTS};

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
Physical Line(curve_tag) = {LOOP_ARRAY};  // loop
Physical Surface(outside_tag) = {{1}};  // inner loop
Physical Surface(inside_tag) = {{2}};
"""
