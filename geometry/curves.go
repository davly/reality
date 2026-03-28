package geometry

// ---------------------------------------------------------------------------
// Parametric Curves
//
// Interpolation and curve evaluation for animation, path following, and
// procedural content generation. All functions are allocation-free.
// ---------------------------------------------------------------------------

// LinearInterpolate performs linear interpolation between a and b at
// parameter t. No clamping is applied; t outside [0, 1] extrapolates.
//
// Definition: lerp(a, b, t) = a + t * (b - a) = (1 - t) * a + t * b
// Precision: exact for IEEE 754 float64.
func LinearInterpolate(a, b, t float64) float64 {
	return a + t*(b-a)
}

// BezierCubic evaluates a cubic Bezier curve at parameter t for a single
// scalar component. Control points are p0 (start), p1, p2, p3 (end).
//
// Definition:
//
//	B(t) = (1-t)^3 * p0 + 3*(1-t)^2 * t * p1 + 3*(1-t) * t^2 * p2 + t^3 * p3
//
// Valid input range: t in [0, 1] for on-curve evaluation.
// Precision: exact for IEEE 754 float64.
func BezierCubic(p0, p1, p2, p3, t float64) float64 {
	u := 1 - t
	uu := u * u
	tt := t * t
	return uu*u*p0 + 3*uu*t*p1 + 3*u*tt*p2 + tt*t*p3
}

// BezierCubic3D evaluates a cubic Bezier curve at parameter t in 3D space.
// Control points are p0 (start), p1, p2, p3 (end).
//
// Definition: applies the cubic Bezier formula component-wise.
// Precision: exact for IEEE 754 float64.
func BezierCubic3D(p0, p1, p2, p3 [3]float64, t float64) [3]float64 {
	u := 1 - t
	uu := u * u
	tt := t * t
	a := uu * u       // (1-t)^3
	b := 3 * uu * t   // 3*(1-t)^2*t
	c := 3 * u * tt   // 3*(1-t)*t^2
	d := tt * t        // t^3
	return [3]float64{
		a*p0[0] + b*p1[0] + c*p2[0] + d*p3[0],
		a*p0[1] + b*p1[1] + c*p2[1] + d*p3[1],
		a*p0[2] + b*p1[2] + c*p2[2] + d*p3[2],
	}
}

// CatmullRom evaluates a Catmull-Rom spline at parameter t for a single
// scalar component. The curve passes through p1 at t=0 and p2 at t=1,
// using p0 and p3 as tangent guides.
//
// Definition (uniform Catmull-Rom, tau = 0.5):
//
//	C(t) = 0.5 * ((2*p1) +
//	              (-p0 + p2) * t +
//	              (2*p0 - 5*p1 + 4*p2 - p3) * t^2 +
//	              (-p0 + 3*p1 - 3*p2 + p3) * t^3)
//
// Valid input range: t in [0, 1] for interpolation between p1 and p2.
// Precision: exact for IEEE 754 float64.
func CatmullRom(p0, p1, p2, p3, t float64) float64 {
	tt := t * t
	ttt := tt * t
	return 0.5 * ((2 * p1) +
		(-p0+p2)*t +
		(2*p0-5*p1+4*p2-p3)*tt +
		(-p0+3*p1-3*p2+p3)*ttt)
}
