
lc = 8;

Point(1) = {0, 0, 0, lc};
Point(2) = {-5, 0, 0, lc};
Point(3) = {5, 0, 0, lc};
Point(4) = {-12.5, -12.5, 0, lc};
Point(5) = {12.5, -12.5, 0, lc};
Point(6) = {12.5, 12.5, 0, lc};
Point(7) = {-12.5, 12.5, 0, lc};
Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 2};

Line(3) = {7, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};

Line Loop(7) = {3, 4, 5, 6};  // exterior loop
Line Loop(8) = {2, 1};  // circle loop

Physical Line(10) = {2, 1};  // circle loop
Physical Line(11) = {3};
Physical Line(12) = {5};
Physical Line(13) = {4};
Physical Line(14) = {6};

Plane Surface(9) = {7, 8};
Physical Surface(15) = {9};

Plane Surface(10) = {8};
Physical Surface(16) = {10};
