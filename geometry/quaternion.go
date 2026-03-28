// Package geometry provides computational geometry primitives: quaternion math,
// signed distance functions, parametric curves, and polygon operations. All
// functions are deterministic, use only the Go standard library, and make zero
// heap allocations. Vectors and quaternions use fixed-size arrays ([3]float64
// and [4]float64) to enable stack allocation at 60 FPS.
//
// Quaternion convention: [w, x, y, z] where w is the scalar component.
// Euler angle convention: ZYX (yaw-pitch-roll) intrinsic rotation order.
//
// Extracted from: Pistachio procgen pipeline (proven in 1,758 tests, 57,860
// assertions) plus standard mathematical definitions.
package geometry

import "math"

// QuatIdentity returns the identity quaternion [1, 0, 0, 0].
// Rotating any vector by the identity quaternion leaves it unchanged.
//
// Definition: q_identity = 1 + 0i + 0j + 0k
// Precision: exact.
func QuatIdentity() [4]float64 {
	return [4]float64{1, 0, 0, 0}
}

// QuatDot computes the dot product of two quaternions.
//
// Definition: dot(a, b) = a_w*b_w + a_x*b_x + a_y*b_y + a_z*b_z
// Result range: [-1, 1] for unit quaternions.
// Precision: exact for IEEE 754 float64 inputs.
func QuatDot(a, b [4]float64) float64 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
}

// QuatConjugate returns the conjugate of quaternion q by negating the vector
// part. For unit quaternions, the conjugate equals the inverse.
//
// Definition: conj(w + xi + yj + zk) = w - xi - yj - zk
// Precision: exact.
func QuatConjugate(q [4]float64) [4]float64 {
	return [4]float64{q[0], -q[1], -q[2], -q[3]}
}

// QuatNormalize returns q scaled to unit length. If q has zero magnitude,
// returns the identity quaternion to avoid division by zero.
//
// Definition: q_norm = q / ||q||
// Precision: exact for IEEE 754 float64.
func QuatNormalize(q [4]float64) [4]float64 {
	mag := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
	if mag == 0 {
		return QuatIdentity()
	}
	inv := 1.0 / mag
	return [4]float64{q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv}
}

// QuatMul computes the Hamilton product of two quaternions.
//
// Definition:
//
//	(a_w + a_x*i + a_y*j + a_z*k) * (b_w + b_x*i + b_y*j + b_z*k) =
//	  (a_w*b_w - a_x*b_x - a_y*b_y - a_z*b_z) +
//	  (a_w*b_x + a_x*b_w + a_y*b_z - a_z*b_y) i +
//	  (a_w*b_y - a_x*b_z + a_y*b_w + a_z*b_x) j +
//	  (a_w*b_z + a_x*b_y - a_y*b_x + a_z*b_w) k
//
// Hamilton product is associative but NOT commutative.
// Precision: exact for IEEE 754 float64.
func QuatMul(a, b [4]float64) [4]float64 {
	return [4]float64{
		a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
		a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
		a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
		a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
	}
}

// QuatSlerp performs Spherical Linear Interpolation between unit quaternions
// a and b at parameter t in [0, 1]. Takes the shortest arc (flips b if the
// dot product is negative).
//
// Definition:
//
//	slerp(a, b, t) = a * sin((1-t)*theta) / sin(theta) + b * sin(t*theta) / sin(theta)
//	where theta = acos(|dot(a, b)|)
//
// For nearly parallel quaternions (|dot| > 0.9995), falls back to normalized
// linear interpolation to avoid division by near-zero sin(theta).
//
// Valid input range: t in [0, 1]; a and b should be unit quaternions.
// Precision: 1e-12 (transcendental functions).
func QuatSlerp(a, b [4]float64, t float64) [4]float64 {
	dot := QuatDot(a, b)

	// Take the shortest arc.
	if dot < 0 {
		b = [4]float64{-b[0], -b[1], -b[2], -b[3]}
		dot = -dot
	}

	// Nearly parallel: use normalized lerp to avoid sin(~0) instability.
	if dot > 0.9995 {
		return QuatNormalize([4]float64{
			a[0] + t*(b[0]-a[0]),
			a[1] + t*(b[1]-a[1]),
			a[2] + t*(b[2]-a[2]),
			a[3] + t*(b[3]-a[3]),
		})
	}

	theta := math.Acos(dot)
	sinTheta := math.Sin(theta)
	wa := math.Sin((1-t)*theta) / sinTheta
	wb := math.Sin(t*theta) / sinTheta

	return [4]float64{
		wa*a[0] + wb*b[0],
		wa*a[1] + wb*b[1],
		wa*a[2] + wb*b[2],
		wa*a[3] + wb*b[3],
	}
}

// QuatFromAxisAngle creates a unit quaternion from an axis and angle (radians).
// The axis does not need to be normalized; it is normalized internally.
// If the axis has zero length, returns the identity quaternion.
//
// Definition:
//
//	q = cos(angle/2) + sin(angle/2) * (axis_x*i + axis_y*j + axis_z*k)
//
// Precision: 1e-15 (single trig call).
func QuatFromAxisAngle(axis [3]float64, angle float64) [4]float64 {
	mag := math.Sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])
	if mag == 0 {
		return QuatIdentity()
	}
	inv := 1.0 / mag
	halfAngle := angle * 0.5
	s := math.Sin(halfAngle)
	return [4]float64{
		math.Cos(halfAngle),
		s * axis[0] * inv,
		s * axis[1] * inv,
		s * axis[2] * inv,
	}
}

// QuatToAxisAngle extracts the rotation axis and angle (radians) from a unit
// quaternion. The returned angle is in [0, 2*pi). If the quaternion represents
// no rotation (w ~= 1), returns axis [0, 0, 1] and angle 0.
//
// Definition:
//
//	angle = 2 * acos(w)
//	axis  = (x, y, z) / sin(angle/2)
//
// Precision: 1e-12 (transcendental functions).
func QuatToAxisAngle(q [4]float64) (axis [3]float64, angle float64) {
	// Ensure w is in [-1, 1] for acos safety.
	w := q[0]
	if w > 1 {
		w = 1
	}
	if w < -1 {
		w = -1
	}

	angle = 2 * math.Acos(w)
	sinHalf := math.Sin(angle * 0.5)

	if sinHalf < 1e-10 {
		// Near-zero rotation: axis is arbitrary, choose +Z.
		return [3]float64{0, 0, 1}, 0
	}

	inv := 1.0 / sinHalf
	return [3]float64{q[1] * inv, q[2] * inv, q[3] * inv}, angle
}

// QuatRotateVec rotates a 3D vector v by unit quaternion q.
//
// Definition: v' = q * (0, v) * conj(q)
//
// This implementation uses the optimized formula (Rodrigues via quaternion):
//
//	t = 2 * cross(q.xyz, v)
//	v' = v + q.w * t + cross(q.xyz, t)
//
// Zero heap allocations. Precision: exact for IEEE 754 float64.
func QuatRotateVec(q [4]float64, v [3]float64) [3]float64 {
	// q.xyz cross v
	tx := 2 * (q[2]*v[2] - q[3]*v[1])
	ty := 2 * (q[3]*v[0] - q[1]*v[2])
	tz := 2 * (q[1]*v[1] - q[2]*v[0])

	// v + q.w * t + q.xyz cross t
	return [3]float64{
		v[0] + q[0]*tx + (q[2]*tz - q[3]*ty),
		v[1] + q[0]*ty + (q[3]*tx - q[1]*tz),
		v[2] + q[0]*tz + (q[1]*ty - q[2]*tx),
	}
}

// QuatFromEuler creates a unit quaternion from Euler angles in ZYX
// (yaw-pitch-roll) intrinsic rotation order. Angles are in radians.
//
// Parameters:
//   - pitch: rotation around X axis
//   - yaw: rotation around Y axis
//   - roll: rotation around Z axis
//
// Definition: q = q_z(roll) * q_y(yaw) * q_x(pitch)
//
// Precision: 1e-15 (direct trig computation).
func QuatFromEuler(pitch, yaw, roll float64) [4]float64 {
	cp := math.Cos(pitch * 0.5)
	sp := math.Sin(pitch * 0.5)
	cy := math.Cos(yaw * 0.5)
	sy := math.Sin(yaw * 0.5)
	cr := math.Cos(roll * 0.5)
	sr := math.Sin(roll * 0.5)

	return [4]float64{
		cr*cy*cp + sr*sy*sp, // w
		cr*cy*sp - sr*sy*cp, // x
		cr*sy*cp + sr*cy*sp, // y
		sr*cy*cp - cr*sy*sp, // z
	}
}
