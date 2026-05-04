//---------------------------------------
// Geometry kernel
//---------------------------------------

SetFactory("OpenCASCADE");



//---------------------------------------
// Parameters of the geometry
//---------------------------------------

// Overetch of the upper plate
overetch = 0.5;

// Distance between the plates
distance = 2.5;

// Discretization over the x axis (for the deformation)
nx = 50; 

// Length of the plate
L = 100 - overetch; 

// Number of modes
n = 4; 

// 1st mode coefficient
coeff(1) = -0.12;
beta(1) = 0.596864 * 3.1415926535 / L;

// 2nd mode coefficient
coeff(2) = -0.12;
beta(2) = 1.49418 * 3.1415926535 / L;

// 3rd mode coefficient
coeff(3) = 0.12;
beta(3) = 2.50025 * 3.1415926535 / L;

// 4th mode coefficient
coeff(4) = 0.12;
beta(4) = 3.49999 * 3.1415926535 / L;



//---------------------------------------
// Upper plate (deformed rectangle)
//---------------------------------------

// Coordiantes of the upper plate
xmin = -50;
xmax = 50 - overetch;

// Height of the upper plate
yheight = 4 - 2 * overetch;

// Indexes of the points
UpperID[] = {};
LowerID[] = {};
LeftID[] = {};
RightID[] = {};

// Point index
p = 1;

// Horizontal edges
For i In {1:nx}
    x = xmin + (i-1)*(xmax - xmin)/(nx-1);

    y_bottom = distance / 2 + overetch;
    For j In {1:n}
        C = (Cosh(beta(j)*L) + Cos(beta(j)*L)) / (Sinh(beta(j)*L) + Sin(beta(j)*L));
        y_bottom  = y_bottom + coeff(j) * (Cosh(beta(j)*(x-xmin)) - Cos(beta(j)*(x-xmin)) - C * (Sinh(beta(j)*(x-xmin)) - Sin(beta(j)*(x-xmin))));
    EndFor

    y_top = y_bottom + yheight;

    Point(p) = {x, y_top, 0, 1.0};
    UpperID[i] = p;
    p = p + 1;

    Point(p) = {x, y_bottom, 0, 1.0};
    LowerID[i] = p;
    p = p + 1;
EndFor

// Create point lists
UpperPts[] = {}; LowerPts[] = {};
For i In {1:nx}
  UpperPts[] += {UpperID[i]};
  LowerPts[] += {LowerID[i]};
EndFor

// Build splines
Spline(1) = {UpperPts[]};
Spline(2) = {LowerPts[]};
Line(3) = {LowerPts[nx-1], UpperPts[nx-1]};
Line(4) = {LowerPts[0], UpperPts[0]};

// Build the plane surface
Curve Loop(1) = {2, 3, -1, -4};
Plane Surface(1) = {1};



//---------------------------------------
// Lower plate (rectangle)
//---------------------------------------

// Vertices
Point(1001) = {-50, -distance/2, 0, 1.0};
Point(1002) = { 50, -distance/2, 0, 1.0};
Point(1003) = { 50, -distance/2-4, 0, 1.0};
Point(1004) = {-50, -distance/2-4, 0, 1.0};

// Edges
Line(5) = {1001, 1002};
Line(6) = {1002, 1003};
Line(7) = {1003, 1004};
Line(8) = {1004, 1001};

// Plane surface
Line Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2}; 



//---------------------------------------
// Outer boundary
//---------------------------------------

// Points
R = 200;
Point(1005) = {0, 0, 0};
Point(1006) = {R, 0, 0};
Point(1007) = {0,  R, 0};
Point(1008) = {-R, 0, 0};
Point(1009) = {0, -R, 0};

// Circle arcs (each needs start, center, end)
Circle(9) = {1006, 1005, 1007};
Circle(10) = {1007, 1005, 1008};
Circle(11) = {1008, 1005, 1009};
Circle(12) = {1009, 1005, 1006};

// Plane surface for the entire domain
Curve Loop(3) = {9, 10, 11, 12};
Plane Surface(3) = {3};

// Subtract the plates from the whole domain to obtain the actual domain where we want to solve the equations
air[] = BooleanDifference{ Surface{3}; Delete; }{ Surface{1}; Surface{2}; Delete; };


//---------------------------------------
// Transfinite Curves
//---------------------------------------

// Set the number of points on the boundaries
r = 16;
d = 0.15;

// Plates
Transfinite Curve {1}    = 50*r/4   Using Progression 1;
Transfinite Curve {2}    = 50*r     Using Progression 1;
Transfinite Curve {3, 4} = 2*r      Using Progression 1+d;
Transfinite Curve {5}    = 50*r     Using Progression 1;
Transfinite Curve {7}    = 50*r/4   Using Progression 1;
Transfinite Curve {6}    = 2*r      Using Progression 1+d;
Transfinite Curve {8}    = 2*r      Using Progression 1-d;

// Outer boundary
Transfinite Curve {9, 10, 11, 12} = 20 Using Progression 1;


//---------------------------------------
// Physical groups
//---------------------------------------                

// Physical curves (boundaries)
Physical Line("force_segment",  10) = {2};
Physical Line("upper_plate",    11) = {1, 3, 4};
Physical Line("lower_plate",    12) = {5, 6, 7, 8};
Physical Line("boundary",       20) = {9, 10, 11, 12};

// Physical Surfaces
Physical Surface("air",         30) = {air[]};



//---------------------------------------
// 6. Generate the Mesh
//---------------------------------------

// 2D mesh generation
Mesh.Algorithm = 6;
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;

Mesh 2;