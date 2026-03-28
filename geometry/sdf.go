package geometry

import "math"

// ---------------------------------------------------------------------------
// Signed Distance Functions (SDF)
//
// These are the building blocks for Pistachio's procedural generation pipeline.
// Each function returns the signed distance from a point to a primitive:
// negative inside, zero on the surface, positive outside.
//
// All SDF functions are allocation-free and suitable for 60 FPS evaluation.
// ---------------------------------------------------------------------------

// SDFSphere returns the signed distance from point p to a sphere defined by
// center and radius.
//
// Definition: sdf(p) = ||p - center|| - radius
// Precision: exact for IEEE 754 float64.
func SDFSphere(p, center [3]float64, radius float64) float64 {
	dx := p[0] - center[0]
	dy := p[1] - center[1]
	dz := p[2] - center[2]
	return math.Sqrt(dx*dx+dy*dy+dz*dz) - radius
}

// SDFBox returns the signed distance from point p to an axis-aligned box
// defined by center and halfExtents (half-widths along each axis).
//
// Definition: standard box SDF using component-wise distance to the box
// surface, handling interior points via max of negative penetration depths.
// Precision: exact for IEEE 754 float64.
func SDFBox(p, center, halfExtents [3]float64) float64 {
	// Distance from center, then subtract half-extents.
	dx := math.Abs(p[0]-center[0]) - halfExtents[0]
	dy := math.Abs(p[1]-center[1]) - halfExtents[1]
	dz := math.Abs(p[2]-center[2]) - halfExtents[2]

	// Clamp negative components to zero for exterior distance.
	ex := math.Max(dx, 0)
	ey := math.Max(dy, 0)
	ez := math.Max(dz, 0)

	// Exterior distance + interior distance.
	exterior := math.Sqrt(ex*ex + ey*ey + ez*ez)
	interior := math.Min(math.Max(dx, math.Max(dy, dz)), 0)

	return exterior + interior
}

// SDFCapsule returns the signed distance from point p to a capsule (line
// segment with rounded ends) defined by endpoints a and b with radius.
//
// Definition: distance from p to the nearest point on segment [a, b], minus
// the radius.
// Precision: exact for IEEE 754 float64.
func SDFCapsule(p, a, b [3]float64, radius float64) float64 {
	// Vector from a to b and a to p.
	abx := b[0] - a[0]
	aby := b[1] - a[1]
	abz := b[2] - a[2]

	apx := p[0] - a[0]
	apy := p[1] - a[1]
	apz := p[2] - a[2]

	// Project ap onto ab, clamped to [0, 1].
	t := (apx*abx + apy*aby + apz*abz) / (abx*abx + aby*aby + abz*abz)
	if t < 0 {
		t = 0
	}
	if t > 1 {
		t = 1
	}

	// Nearest point on segment.
	nx := a[0] + t*abx - p[0]
	ny := a[1] + t*aby - p[1]
	nz := a[2] + t*abz - p[2]

	return math.Sqrt(nx*nx+ny*ny+nz*nz) - radius
}

// SDFTorus returns the signed distance from point p to a torus centered at
// center, lying in the XZ plane, with major radius majorR and minor radius
// minorR.
//
// Definition:
//
//	q = (||p.xz - center.xz|| - majorR, p.y - center.y)
//	sdf = ||q|| - minorR
//
// Precision: exact for IEEE 754 float64.
func SDFTorus(p, center [3]float64, majorR, minorR float64) float64 {
	dx := p[0] - center[0]
	dy := p[1] - center[1]
	dz := p[2] - center[2]

	// Distance in XZ plane from center, minus major radius.
	qx := math.Sqrt(dx*dx+dz*dz) - majorR
	qy := dy

	return math.Sqrt(qx*qx+qy*qy) - minorR
}

// ---------------------------------------------------------------------------
// Boolean operations (crisp)
// ---------------------------------------------------------------------------

// SDFUnion returns the union of two distance fields: min(d1, d2).
//
// Definition: U(d1, d2) = min(d1, d2)
// Precision: exact.
func SDFUnion(d1, d2 float64) float64 {
	return math.Min(d1, d2)
}

// SDFIntersection returns the intersection of two distance fields: max(d1, d2).
//
// Definition: I(d1, d2) = max(d1, d2)
// Precision: exact.
func SDFIntersection(d1, d2 float64) float64 {
	return math.Max(d1, d2)
}

// SDFSubtraction subtracts d1 from d2: max(-d1, d2).
// This carves d1 out of d2.
//
// Definition: S(d1, d2) = max(-d1, d2)
// Precision: exact.
func SDFSubtraction(d1, d2 float64) float64 {
	return math.Max(-d1, d2)
}

// ---------------------------------------------------------------------------
// Boolean operations (smooth / polynomial)
// ---------------------------------------------------------------------------

// SDFSmoothUnion returns a smooth blend of two distance fields using
// polynomial smooth minimum. Parameter k controls blend radius (k > 0).
// When k = 0, behaves like crisp union (min).
//
// Definition (polynomial smooth min):
//
//	h = clamp(0.5 + 0.5*(d2-d1)/k, 0, 1)
//	result = mix(d2, d1, h) - k*h*(1-h)
//
// Reference: Inigo Quilez, "smooth minimum" (2013).
// Precision: exact for IEEE 754 float64.
func SDFSmoothUnion(d1, d2, k float64) float64 {
	if k <= 0 {
		return math.Min(d1, d2)
	}
	h := 0.5 + 0.5*(d2-d1)/k
	if h < 0 {
		h = 0
	}
	if h > 1 {
		h = 1
	}
	return d2 + (d1-d2)*h - k*h*(1-h)
}

// SDFSmoothSubtraction subtracts d1 from d2 with smooth blending.
// Parameter k controls blend radius (k > 0).
//
// Definition:
//
//	h = clamp(0.5 - 0.5*(d2+d1)/k, 0, 1)
//	result = mix(d2, -d1, h) + k*h*(1-h)
//
// Reference: Inigo Quilez, "smooth subtraction" (2013).
// Precision: exact for IEEE 754 float64.
func SDFSmoothSubtraction(d1, d2, k float64) float64 {
	if k <= 0 {
		return math.Max(-d1, d2)
	}
	h := 0.5 - 0.5*(d2+d1)/k
	if h < 0 {
		h = 0
	}
	if h > 1 {
		h = 1
	}
	return d2 + (-d1-d2)*h + k*h*(1-h)
}

// SDFSmoothIntersection returns a smooth intersection of two distance fields.
// Parameter k controls blend radius (k > 0).
//
// Definition:
//
//	h = clamp(0.5 - 0.5*(d2-d1)/k, 0, 1)
//	result = mix(d2, d1, h) + k*h*(1-h)
//
// Reference: Inigo Quilez, "smooth intersection" (2013).
// Precision: exact for IEEE 754 float64.
func SDFSmoothIntersection(d1, d2, k float64) float64 {
	if k <= 0 {
		return math.Max(d1, d2)
	}
	h := 0.5 - 0.5*(d2-d1)/k
	if h < 0 {
		h = 0
	}
	if h > 1 {
		h = 1
	}
	return d2 + (d1-d2)*h + k*h*(1-h)
}
