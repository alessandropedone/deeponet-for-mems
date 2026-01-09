//---------------------------------------
// actuator_air_only_transfinite.geo
// Units: MICRONS
//---------------------------------------
SetFactory("OpenCASCADE");

//---------------------------------------
// Parameters
//---------------------------------------
overetch = 0.0;
distance = 1.5;
nx = 50;

L = 100 - overetch;
n = 4;

xmin = -50;
xmax =  50 - overetch;
yheight = 4 - 2*overetch;

// ---- Modal coefficients (microns) ----
coeff(1) = __COEFF1__;
coeff(2) = __COEFF2__;
coeff(3) = __COEFF3__;
coeff(4) = __COEFF4__;

// Cantilever roots beta*L = 1.8751, 4.6941, 7.8548, 10.9955
beta(1) =  1.875104068711961 / L;
beta(2) =  4.694091132974174 / L;
beta(3) =  7.854757438237612 / L;
beta(4) = 10.995540734875466 / L;

//---------------------------------------
// Upper plate boundary curves (used as hole boundary)
//---------------------------------------
UpperID[] = {};
LowerID[] = {};
p = 1;

For i In {1:nx}
  x = xmin + (i-1)*(xmax - xmin)/(nx-1);

  y_bottom = distance/2 + overetch;

  For j In {1:n}
    C = (Cosh(beta(j)*L) + Cos(beta(j)*L)) / (Sinh(beta(j)*L) + Sin(beta(j)*L));
    y_bottom = y_bottom + coeff(j) * (
      Cosh(beta(j)*(x - xmin)) - Cos(beta(j)*(x - xmin))
      - C*(Sinh(beta(j)*(x - xmin)) - Sin(beta(j)*(x - xmin)))
    );
  EndFor

  y_top = y_bottom + yheight;

  Point(p) = {x, y_top,    0, 1.0}; UpperID[i] = p; p++;
  Point(p) = {x, y_bottom, 0, 1.0}; LowerID[i] = p; p++;
EndFor

UpperPts[] = {}; LowerPts[] = {};
For i In {1:nx}
  UpperPts[] += {UpperID[i]};
  LowerPts[] += {LowerID[i]};
EndFor

Spline(1) = {UpperPts[]};                     // top edge of upper electrode
Spline(2) = {LowerPts[]};                     // bottom edge of upper electrode (gap-facing)
Line(3)   = {LowerPts[nx-1], UpperPts[nx-1]};  // right edge
Line(4)   = {LowerPts[0],    UpperPts[0]};     // left edge

Curve Loop(1) = {2, 3, -1, -4};
Plane Surface(1) = {1};   // only used to cut a hole

//---------------------------------------
// Lower plate (hole)
//---------------------------------------
Point(1001) = {xmin, -distance/2,     0, 1.0};
Point(1002) = {xmax, -distance/2,     0, 1.0};
Point(1003) = {xmax, -distance/2 - 4, 0, 1.0};
Point(1004) = {xmin, -distance/2 - 4, 0, 1.0};

Line(5) = {1001, 1002}; // top edge (gap-facing)
Line(6) = {1002, 1003};
Line(7) = {1003, 1004};
Line(8) = {1004, 1001};

Line Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2};   // only used to cut a hole

//---------------------------------------
// Outer boundary (air container)
//---------------------------------------
R = 200;
Point(1005) = {0, 0, 0};
Point(1006) = { R, 0, 0};
Point(1007) = {0,  R, 0};
Point(1008) = {-R, 0, 0};
Point(1009) = {0, -R, 0};

Circle(9)  = {1006, 1005, 1007};
Circle(10) = {1007, 1005, 1008};
Circle(11) = {1008, 1005, 1009};
Circle(12) = {1009, 1005, 1006};

Curve Loop(3) = {9, 10, 11, 12};
Plane Surface(3) = {3};

//---------------------------------------
// Transfinite Curves (meshing strategy like your example)
//---------------------------------------
r = 16;
d = 0.15;

// Upper electrode boundary
Transfinite Curve {2} = 50*r     Using Progression 1;
Transfinite Curve {1} = 50*r/4   Using Progression 1;
Transfinite Curve {3,4} = 2*r    Using Progression (1 + d);

// Lower electrode boundary
Transfinite Curve {5} = 50*r     Using Progression 1;
Transfinite Curve {7} = 50*r/4   Using Progression 1;
Transfinite Curve {6} = 2*r      Using Progression (1 + d);
Transfinite Curve {8} = 2*r      Using Progression (1 - d);

// Outer boundary
Transfinite Curve {9,10,11,12} = 20 Using Progression 1;

//---------------------------------------
// BooleanDifference: keep only AIR
//---------------------------------------
air[] = BooleanDifference{ Surface{3}; Delete; }{ Surface{1}; Surface{2}; Delete; };

//---------------------------------------
// Physical groups (for electrostatics BCs + traction integration)
//---------------------------------------
Physical Line("force_segment", 10) = {2};         // gap-facing upper boundary
Physical Line("upper_plate",   11) = {1,3,4};     // rest of upper electrode boundary
Physical Line("lower_plate",   12) = {5,6,7,8};   // lower electrode boundary
Physical Line("boundary",      20) = {9,10,11,12};

Physical Surface("air",        30) = {air[]};

//---------------------------------------
// Mesh options
//---------------------------------------
Mesh.Algorithm = 6;
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;

Mesh 2;
