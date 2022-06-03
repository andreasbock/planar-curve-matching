Mesh.CharacteristicLengthFactor = 8;

Point(1) = {0, 0, 0, 0};
Point(2) = {1, 0, 0, 0};
Point(3) = {1, 1, 0, 0};
Point(4) = {-1, 1, 0, 0};
Point(5) = {-1, -1, 0, 0};
Point(6) = {0, -1, 0, 0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};

Line Loop(7) = {1, 2, 3, 4, 5, 6};

Plane Surface(8) = {7};
Physical Surface(1) = {8};

// Field[1] = Box;
// Field[1].VIn = 0.005;
// Background Field = 1;
