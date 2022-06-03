//Point(1) = {0, 0, 0, .005};
//Point(2) = {-.001, 0, 0, .005};
//Point(3) = {.001, 0, 0, .005};
//Point(4) = {-.00125, -.00125, 0, .005};
//Point(5) = {.00125, -.00125, 0, .005};
//Point(6) = {.00125, .00125, 0, .005};
//Point(7) = {-.00125, .00125, 0, .005};
Point(1) = {0, 0, 0, .5};
Point(2) = {-5, 0, 0, .5};
Point(3) = {5, 0, 0, .5};
Point(4) = {-12.5, -12.5, 0, .5};
Point(5) = {12.5, -12.5, 0, .5};
Point(6) = {12.5, 12.5, 0, .5};
Point(7) = {-12.5, 12.5, 0, .5};
Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 2};

Line(3) = {7, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};

Line Loop(7) = {3, 4, 5, 6};  // exterior loop
Line Loop(8) = {2, 1};  // circle loop

Plane Surface(9) = {7, 8};  // no idea

Physical Line(10) = {2, 1};  // circle loop
Physical Line(11) = {3};
Physical Line(12) = {5};
Physical Line(13) = {4};
Physical Line(14) = {6};

Physical Surface(1) = {9};

//
Plane Surface(15) = {8};
Physical Surface(2) = {15};

Field[1] = Box;
Field[1].VIn = 0.005;
Background Field = 1;
