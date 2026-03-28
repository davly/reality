package geometry

import (
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file tests — shared test vectors across Go, Python, C++, C#
// ═══════════════════════════════════════════════════════════════════════════

func TestGolden_QuatSlerp(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/geometry/quaternion_slerp.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			a := inputFloat64Array4(t, tc, "a")
			b := inputFloat64Array4(t, tc, "b")
			tParam := testutil.InputFloat64(t, tc, "t")
			got := QuatSlerp(a, b, tParam)
			gotSlice := got[:]
			testutil.AssertFloat64Slice(t, tc, gotSlice)
		})
	}
}

func TestGolden_SDFPrimitives(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/geometry/sdf_primitives.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			prim := tc.Inputs["primitive"].(string)
			switch prim {
			case "sphere":
				p := inputFloat64Array3(t, tc, "p")
				center := inputFloat64Array3(t, tc, "center")
				radius := testutil.InputFloat64(t, tc, "radius")
				got := SDFSphere(p, center, radius)
				testutil.AssertFloat64(t, tc, got)
			case "box":
				p := inputFloat64Array3(t, tc, "p")
				center := inputFloat64Array3(t, tc, "center")
				half := inputFloat64Array3(t, tc, "halfExtents")
				got := SDFBox(p, center, half)
				testutil.AssertFloat64(t, tc, got)
			case "capsule":
				p := inputFloat64Array3(t, tc, "p")
				a := inputFloat64Array3(t, tc, "a")
				b := inputFloat64Array3(t, tc, "b")
				radius := testutil.InputFloat64(t, tc, "radius")
				got := SDFCapsule(p, a, b, radius)
				testutil.AssertFloat64(t, tc, got)
			case "torus":
				p := inputFloat64Array3(t, tc, "p")
				center := inputFloat64Array3(t, tc, "center")
				majorR := testutil.InputFloat64(t, tc, "majorR")
				minorR := testutil.InputFloat64(t, tc, "minorR")
				got := SDFTorus(p, center, majorR, minorR)
				testutil.AssertFloat64(t, tc, got)
			default:
				t.Fatalf("unknown primitive: %s", prim)
			}
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Quaternion identity, conjugate, dot, normalize
// ═══════════════════════════════════════════════════════════════════════════

func TestQuatIdentity(t *testing.T) {
	q := QuatIdentity()
	assertQuat(t, "identity", q, [4]float64{1, 0, 0, 0}, 0)
}

func TestQuatDot_IdentityWithSelf(t *testing.T) {
	id := QuatIdentity()
	got := QuatDot(id, id)
	assertClose(t, "dot-id-id", got, 1.0, 1e-15)
}

func TestQuatDot_Orthogonal(t *testing.T) {
	// Two quaternions representing 90° rotations around different axes.
	a := [4]float64{0, 1, 0, 0} // pure i
	b := [4]float64{0, 0, 1, 0} // pure j
	got := QuatDot(a, b)
	assertClose(t, "dot-orthogonal", got, 0.0, 1e-15)
}

func TestQuatConjugate(t *testing.T) {
	q := [4]float64{1, 2, 3, 4}
	c := QuatConjugate(q)
	assertQuat(t, "conjugate", c, [4]float64{1, -2, -3, -4}, 0)
}

func TestQuatConjugate_Identity(t *testing.T) {
	c := QuatConjugate(QuatIdentity())
	assertQuat(t, "conj-id", c, [4]float64{1, 0, 0, 0}, 0)
}

func TestQuatNormalize_UnitQuat(t *testing.T) {
	q := QuatIdentity()
	n := QuatNormalize(q)
	assertQuat(t, "normalize-unit", n, [4]float64{1, 0, 0, 0}, 1e-15)
}

func TestQuatNormalize_ScaledQuat(t *testing.T) {
	q := [4]float64{2, 0, 0, 0}
	n := QuatNormalize(q)
	assertQuat(t, "normalize-scaled", n, [4]float64{1, 0, 0, 0}, 1e-15)
}

func TestQuatNormalize_ArbitraryQuat(t *testing.T) {
	q := [4]float64{1, 1, 1, 1}
	n := QuatNormalize(q)
	half := 0.5
	assertQuat(t, "normalize-1111", n, [4]float64{half, half, half, half}, 1e-15)
}

func TestQuatNormalize_ZeroQuat(t *testing.T) {
	q := [4]float64{0, 0, 0, 0}
	n := QuatNormalize(q)
	assertQuat(t, "normalize-zero", n, [4]float64{1, 0, 0, 0}, 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Quaternion multiplication
// ═══════════════════════════════════════════════════════════════════════════

func TestQuatMul_IdentityLeft(t *testing.T) {
	id := QuatIdentity()
	q := [4]float64{0.5, 0.5, 0.5, 0.5}
	got := QuatMul(id, q)
	assertQuat(t, "mul-id-left", got, q, 1e-15)
}

func TestQuatMul_IdentityRight(t *testing.T) {
	id := QuatIdentity()
	q := [4]float64{0.5, 0.5, 0.5, 0.5}
	got := QuatMul(q, id)
	assertQuat(t, "mul-id-right", got, q, 1e-15)
}

func TestQuatMul_IJ_equals_K(t *testing.T) {
	// i * j = k (Hamilton's rule)
	i := [4]float64{0, 1, 0, 0}
	j := [4]float64{0, 0, 1, 0}
	k := QuatMul(i, j)
	assertQuat(t, "i*j=k", k, [4]float64{0, 0, 0, 1}, 1e-15)
}

func TestQuatMul_JK_equals_I(t *testing.T) {
	j := [4]float64{0, 0, 1, 0}
	k := [4]float64{0, 0, 0, 1}
	got := QuatMul(j, k)
	assertQuat(t, "j*k=i", got, [4]float64{0, 1, 0, 0}, 1e-15)
}

func TestQuatMul_KI_equals_J(t *testing.T) {
	k := [4]float64{0, 0, 0, 1}
	i := [4]float64{0, 1, 0, 0}
	got := QuatMul(k, i)
	assertQuat(t, "k*i=j", got, [4]float64{0, 0, 1, 0}, 1e-15)
}

func TestQuatMul_NonCommutative(t *testing.T) {
	a := [4]float64{0, 1, 0, 0}
	b := [4]float64{0, 0, 1, 0}
	ab := QuatMul(a, b)
	ba := QuatMul(b, a)
	// i*j = k, but j*i = -k
	assertQuat(t, "ab", ab, [4]float64{0, 0, 0, 1}, 1e-15)
	assertQuat(t, "ba", ba, [4]float64{0, 0, 0, -1}, 1e-15)
}

func TestQuatMul_QTimesConjugate_IsNormSquared(t *testing.T) {
	q := [4]float64{1, 2, 3, 4}
	c := QuatConjugate(q)
	got := QuatMul(q, c)
	normSq := q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]
	assertQuat(t, "q*conj(q)", got, [4]float64{normSq, 0, 0, 0}, 1e-12)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Quaternion slerp
// ═══════════════════════════════════════════════════════════════════════════

func TestQuatSlerp_T0(t *testing.T) {
	a := QuatIdentity()
	b := QuatFromAxisAngle([3]float64{0, 0, 1}, math.Pi/2)
	got := QuatSlerp(a, b, 0)
	assertQuat(t, "slerp-t0", got, a, 1e-15)
}

func TestQuatSlerp_T1(t *testing.T) {
	a := QuatIdentity()
	b := QuatFromAxisAngle([3]float64{0, 0, 1}, math.Pi/2)
	got := QuatSlerp(a, b, 1)
	assertQuat(t, "slerp-t1", got, b, 1e-12)
}

func TestQuatSlerp_Midpoint(t *testing.T) {
	a := QuatIdentity()
	b := QuatFromAxisAngle([3]float64{0, 0, 1}, math.Pi/2)
	got := QuatSlerp(a, b, 0.5)
	// Should be 45° around Z
	expected := QuatFromAxisAngle([3]float64{0, 0, 1}, math.Pi/4)
	assertQuat(t, "slerp-mid", got, expected, 1e-12)
}

func TestQuatSlerp_SameQuat(t *testing.T) {
	a := QuatFromAxisAngle([3]float64{1, 0, 0}, 1.0)
	got := QuatSlerp(a, a, 0.5)
	assertQuat(t, "slerp-same", got, a, 1e-12)
}

func TestQuatSlerp_ShortestArc(t *testing.T) {
	a := QuatIdentity()
	b := QuatFromAxisAngle([3]float64{0, 0, 1}, math.Pi/2)
	// Negate b — should still take the shortest arc
	nb := [4]float64{-b[0], -b[1], -b[2], -b[3]}
	got := QuatSlerp(a, nb, 0.5)
	expected := QuatFromAxisAngle([3]float64{0, 0, 1}, math.Pi/4)
	assertQuat(t, "slerp-shortest-arc", got, expected, 1e-12)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Axis-angle conversion
// ═══════════════════════════════════════════════════════════════════════════

func TestQuatFromAxisAngle_ZAxis90(t *testing.T) {
	q := QuatFromAxisAngle([3]float64{0, 0, 1}, math.Pi/2)
	s := math.Sin(math.Pi / 4)
	c := math.Cos(math.Pi / 4)
	assertQuat(t, "from-axis-z90", q, [4]float64{c, 0, 0, s}, 1e-15)
}

func TestQuatFromAxisAngle_ZeroAngle(t *testing.T) {
	q := QuatFromAxisAngle([3]float64{1, 0, 0}, 0)
	assertQuat(t, "from-axis-zero-angle", q, [4]float64{1, 0, 0, 0}, 1e-15)
}

func TestQuatFromAxisAngle_ZeroAxis(t *testing.T) {
	q := QuatFromAxisAngle([3]float64{0, 0, 0}, math.Pi)
	assertQuat(t, "from-axis-zero-axis", q, QuatIdentity(), 0)
}

func TestQuatFromAxisAngle_UnnormalizedAxis(t *testing.T) {
	// Axis (2, 0, 0) should be normalized internally.
	q := QuatFromAxisAngle([3]float64{2, 0, 0}, math.Pi/2)
	expected := QuatFromAxisAngle([3]float64{1, 0, 0}, math.Pi/2)
	assertQuat(t, "from-axis-unnormalized", q, expected, 1e-15)
}

func TestQuatToAxisAngle_Identity(t *testing.T) {
	axis, angle := QuatToAxisAngle(QuatIdentity())
	assertClose(t, "to-axis-angle-id", angle, 0.0, 1e-12)
	// Axis is arbitrary for zero rotation; just check it's valid.
	_ = axis
}

func TestQuatToAxisAngle_Roundtrip(t *testing.T) {
	origAxis := [3]float64{0, 1, 0}
	origAngle := math.Pi / 3
	q := QuatFromAxisAngle(origAxis, origAngle)
	axis, angle := QuatToAxisAngle(q)
	assertClose(t, "roundtrip-angle", angle, origAngle, 1e-12)
	assertClose(t, "roundtrip-axis-x", axis[0], origAxis[0], 1e-12)
	assertClose(t, "roundtrip-axis-y", axis[1], origAxis[1], 1e-12)
	assertClose(t, "roundtrip-axis-z", axis[2], origAxis[2], 1e-12)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Quaternion rotation
// ═══════════════════════════════════════════════════════════════════════════

func TestQuatRotateVec_Identity(t *testing.T) {
	v := [3]float64{1, 2, 3}
	got := QuatRotateVec(QuatIdentity(), v)
	assertVec3(t, "rotate-identity", got, v, 1e-15)
}

func TestQuatRotateVec_90AroundZ(t *testing.T) {
	// Rotating (1, 0, 0) by 90° around Z -> (0, 1, 0)
	q := QuatFromAxisAngle([3]float64{0, 0, 1}, math.Pi/2)
	got := QuatRotateVec(q, [3]float64{1, 0, 0})
	assertVec3(t, "rotate-z90", got, [3]float64{0, 1, 0}, 1e-12)
}

func TestQuatRotateVec_90AroundX(t *testing.T) {
	// Rotating (0, 1, 0) by 90° around X -> (0, 0, 1)
	q := QuatFromAxisAngle([3]float64{1, 0, 0}, math.Pi/2)
	got := QuatRotateVec(q, [3]float64{0, 1, 0})
	assertVec3(t, "rotate-x90", got, [3]float64{0, 0, 1}, 1e-12)
}

func TestQuatRotateVec_90AroundY(t *testing.T) {
	// Rotating (0, 0, 1) by 90° around Y -> (1, 0, 0)
	q := QuatFromAxisAngle([3]float64{0, 1, 0}, math.Pi/2)
	got := QuatRotateVec(q, [3]float64{0, 0, 1})
	assertVec3(t, "rotate-y90", got, [3]float64{1, 0, 0}, 1e-12)
}

func TestQuatRotateVec_180AroundY(t *testing.T) {
	// Rotating (1, 0, 0) by 180° around Y -> (-1, 0, 0)
	q := QuatFromAxisAngle([3]float64{0, 1, 0}, math.Pi)
	got := QuatRotateVec(q, [3]float64{1, 0, 0})
	assertVec3(t, "rotate-y180", got, [3]float64{-1, 0, 0}, 1e-12)
}

func TestQuatRotateVec_PreservesLength(t *testing.T) {
	q := QuatFromAxisAngle([3]float64{1, 1, 1}, 1.23)
	v := [3]float64{3, 4, 5}
	got := QuatRotateVec(q, v)
	origLen := math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
	gotLen := math.Sqrt(got[0]*got[0] + got[1]*got[1] + got[2]*got[2])
	assertClose(t, "rotate-preserves-len", gotLen, origLen, 1e-12)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Euler conversion
// ═══════════════════════════════════════════════════════════════════════════

func TestQuatFromEuler_AllZero(t *testing.T) {
	q := QuatFromEuler(0, 0, 0)
	assertQuat(t, "euler-zero", q, QuatIdentity(), 1e-15)
}

func TestQuatFromEuler_PitchOnly(t *testing.T) {
	q := QuatFromEuler(math.Pi/2, 0, 0)
	expected := QuatFromAxisAngle([3]float64{1, 0, 0}, math.Pi/2)
	assertQuat(t, "euler-pitch", q, expected, 1e-12)
}

func TestQuatFromEuler_YawOnly(t *testing.T) {
	q := QuatFromEuler(0, math.Pi/2, 0)
	expected := QuatFromAxisAngle([3]float64{0, 1, 0}, math.Pi/2)
	assertQuat(t, "euler-yaw", q, expected, 1e-12)
}

func TestQuatFromEuler_RollOnly(t *testing.T) {
	q := QuatFromEuler(0, 0, math.Pi/2)
	expected := QuatFromAxisAngle([3]float64{0, 0, 1}, math.Pi/2)
	assertQuat(t, "euler-roll", q, expected, 1e-12)
}

func TestQuatFromEuler_RoundtripRotation(t *testing.T) {
	// Apply Euler rotation, rotate a vector, compare with sequential rotations.
	// ZYX intrinsic: q = q_z(roll) * q_y(yaw) * q_x(pitch)
	q := QuatFromEuler(math.Pi/6, math.Pi/4, math.Pi/3)
	v := [3]float64{1, 0, 0}
	got := QuatRotateVec(q, v)
	// Compose in ZYX intrinsic order: q = qz * qy * qx
	qr := QuatFromAxisAngle([3]float64{0, 0, 1}, math.Pi/3)
	qy := QuatFromAxisAngle([3]float64{0, 1, 0}, math.Pi/4)
	qp := QuatFromAxisAngle([3]float64{1, 0, 0}, math.Pi/6)
	combined := QuatMul(qr, QuatMul(qy, qp))
	expected := QuatRotateVec(combined, v)
	assertVec3(t, "euler-roundtrip", got, expected, 1e-12)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — SDF sphere
// ═══════════════════════════════════════════════════════════════════════════

func TestSDFSphere_Outside(t *testing.T) {
	got := SDFSphere([3]float64{3, 0, 0}, [3]float64{0, 0, 0}, 1)
	assertClose(t, "sphere-outside", got, 2.0, 1e-15)
}

func TestSDFSphere_Inside(t *testing.T) {
	got := SDFSphere([3]float64{0.5, 0, 0}, [3]float64{0, 0, 0}, 1)
	assertClose(t, "sphere-inside", got, -0.5, 1e-15)
}

func TestSDFSphere_OnSurface(t *testing.T) {
	got := SDFSphere([3]float64{0, 1, 0}, [3]float64{0, 0, 0}, 1)
	assertClose(t, "sphere-surface", got, 0.0, 1e-15)
}

func TestSDFSphere_OffsetCenter(t *testing.T) {
	got := SDFSphere([3]float64{5, 0, 0}, [3]float64{3, 0, 0}, 1)
	assertClose(t, "sphere-offset", got, 1.0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — SDF box
// ═══════════════════════════════════════════════════════════════════════════

func TestSDFBox_Inside(t *testing.T) {
	got := SDFBox([3]float64{0, 0, 0}, [3]float64{0, 0, 0}, [3]float64{1, 1, 1})
	assertClose(t, "box-inside", got, -1.0, 1e-15)
}

func TestSDFBox_OnFace(t *testing.T) {
	got := SDFBox([3]float64{1, 0, 0}, [3]float64{0, 0, 0}, [3]float64{1, 1, 1})
	assertClose(t, "box-face", got, 0.0, 1e-15)
}

func TestSDFBox_Outside(t *testing.T) {
	got := SDFBox([3]float64{2, 0, 0}, [3]float64{0, 0, 0}, [3]float64{1, 1, 1})
	assertClose(t, "box-outside", got, 1.0, 1e-15)
}

func TestSDFBox_Corner(t *testing.T) {
	got := SDFBox([3]float64{2, 2, 2}, [3]float64{0, 0, 0}, [3]float64{1, 1, 1})
	assertClose(t, "box-corner", got, math.Sqrt(3), 1e-12)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — SDF capsule
// ═══════════════════════════════════════════════════════════════════════════

func TestSDFCapsule_AtEndpointA(t *testing.T) {
	// Point at endpoint A, offset by radius
	got := SDFCapsule([3]float64{0.5, 0, 0}, [3]float64{0, 0, 0}, [3]float64{0, 1, 0}, 0.5)
	assertClose(t, "capsule-endA", got, 0.0, 1e-15)
}

func TestSDFCapsule_Inside(t *testing.T) {
	got := SDFCapsule([3]float64{0, 0.5, 0}, [3]float64{0, 0, 0}, [3]float64{0, 1, 0}, 0.5)
	assertClose(t, "capsule-inside", got, -0.5, 1e-15)
}

func TestSDFCapsule_BeyondEndpoint(t *testing.T) {
	// Point beyond endpoint B, 1 unit away from B along segment direction
	got := SDFCapsule([3]float64{0, 2, 0}, [3]float64{0, 0, 0}, [3]float64{0, 1, 0}, 0.5)
	assertClose(t, "capsule-beyond", got, 0.5, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — SDF torus
// ═══════════════════════════════════════════════════════════════════════════

func TestSDFTorus_OnRingCenter(t *testing.T) {
	got := SDFTorus([3]float64{2, 0, 0}, [3]float64{0, 0, 0}, 2, 0.5)
	assertClose(t, "torus-ring-center", got, -0.5, 1e-15)
}

func TestSDFTorus_OnSurface(t *testing.T) {
	got := SDFTorus([3]float64{2.5, 0, 0}, [3]float64{0, 0, 0}, 2, 0.5)
	assertClose(t, "torus-surface", got, 0.0, 1e-15)
}

func TestSDFTorus_Outside(t *testing.T) {
	got := SDFTorus([3]float64{3, 0, 0}, [3]float64{0, 0, 0}, 2, 0.5)
	assertClose(t, "torus-outside", got, 0.5, 1e-15)
}

func TestSDFTorus_AtOrigin(t *testing.T) {
	// Point at center of torus, distance is majorR - minorR
	got := SDFTorus([3]float64{0, 0, 0}, [3]float64{0, 0, 0}, 2, 0.5)
	assertClose(t, "torus-origin", got, 1.5, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — SDF boolean operations
// ═══════════════════════════════════════════════════════════════════════════

func TestSDFUnion(t *testing.T) {
	assertClose(t, "union", SDFUnion(1, 2), 1.0, 0)
	assertClose(t, "union-neg", SDFUnion(-1, 2), -1.0, 0)
}

func TestSDFIntersection(t *testing.T) {
	assertClose(t, "intersection", SDFIntersection(1, 2), 2.0, 0)
	assertClose(t, "intersection-neg", SDFIntersection(-1, 2), 2.0, 0)
}

func TestSDFSubtraction(t *testing.T) {
	assertClose(t, "subtraction", SDFSubtraction(1, 2), 2.0, 0)
	assertClose(t, "subtraction2", SDFSubtraction(-1, 2), 2.0, 0)
}

func TestSDFSmoothUnion_KZero(t *testing.T) {
	got := SDFSmoothUnion(3, 5, 0)
	assertClose(t, "smooth-union-k0", got, 3.0, 1e-15)
}

func TestSDFSmoothUnion_Blending(t *testing.T) {
	// Smooth union should be <= min(d1, d2) due to blending.
	d1 := 0.3
	d2 := 0.5
	k := 0.5
	got := SDFSmoothUnion(d1, d2, k)
	if got > d1 {
		t.Errorf("smooth union %v should be <= min(%v, %v)", got, d1, d2)
	}
}

func TestSDFSmoothUnion_Symmetric(t *testing.T) {
	// When d1 == d2, smooth union should be d - k/4 for polynomial smooth min.
	d := 1.0
	k := 0.4
	got := SDFSmoothUnion(d, d, k)
	expected := d - k*0.25
	assertClose(t, "smooth-union-symmetric", got, expected, 1e-15)
}

func TestSDFSmoothSubtraction_KZero(t *testing.T) {
	got := SDFSmoothSubtraction(1, 2, 0)
	assertClose(t, "smooth-sub-k0", got, math.Max(-1, 2), 1e-15)
}

func TestSDFSmoothIntersection_KZero(t *testing.T) {
	got := SDFSmoothIntersection(1, 2, 0)
	assertClose(t, "smooth-int-k0", got, 2.0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Bezier curves
// ═══════════════════════════════════════════════════════════════════════════

func TestBezierCubic_T0(t *testing.T) {
	got := BezierCubic(0, 1, 2, 3, 0)
	assertClose(t, "bezier-t0", got, 0.0, 1e-15)
}

func TestBezierCubic_T1(t *testing.T) {
	got := BezierCubic(0, 1, 2, 3, 1)
	assertClose(t, "bezier-t1", got, 3.0, 1e-15)
}

func TestBezierCubic_Midpoint(t *testing.T) {
	// For equally-spaced control points 0,1,2,3:
	// B(0.5) = 0.125*0 + 0.375*1 + 0.375*2 + 0.125*3 = 1.5
	got := BezierCubic(0, 1, 2, 3, 0.5)
	assertClose(t, "bezier-mid", got, 1.5, 1e-15)
}

func TestBezierCubic_LinearCase(t *testing.T) {
	// When all control points are on a line, Bezier should be linear.
	got := BezierCubic(0, 10, 20, 30, 0.3)
	assertClose(t, "bezier-linear", got, 9.0, 1e-12)
}

func TestBezierCubic3D_Endpoints(t *testing.T) {
	p0 := [3]float64{0, 0, 0}
	p1 := [3]float64{1, 2, 0}
	p2 := [3]float64{3, 2, 0}
	p3 := [3]float64{4, 0, 0}

	got0 := BezierCubic3D(p0, p1, p2, p3, 0)
	assertVec3(t, "bezier3d-t0", got0, p0, 1e-15)

	got1 := BezierCubic3D(p0, p1, p2, p3, 1)
	assertVec3(t, "bezier3d-t1", got1, p3, 1e-15)
}

func TestBezierCubic3D_Midpoint(t *testing.T) {
	p0 := [3]float64{0, 0, 0}
	p1 := [3]float64{0, 1, 0}
	p2 := [3]float64{1, 1, 0}
	p3 := [3]float64{1, 0, 0}
	got := BezierCubic3D(p0, p1, p2, p3, 0.5)
	// At t=0.5: x = 0.125*0 + 0.375*0 + 0.375*1 + 0.125*1 = 0.5
	//           y = 0.125*0 + 0.375*1 + 0.375*1 + 0.125*0 = 0.75
	assertVec3(t, "bezier3d-mid", got, [3]float64{0.5, 0.75, 0}, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Catmull-Rom spline
// ═══════════════════════════════════════════════════════════════════════════

func TestCatmullRom_T0(t *testing.T) {
	got := CatmullRom(0, 1, 2, 3, 0)
	assertClose(t, "catmull-t0", got, 1.0, 1e-15)
}

func TestCatmullRom_T1(t *testing.T) {
	got := CatmullRom(0, 1, 2, 3, 1)
	assertClose(t, "catmull-t1", got, 2.0, 1e-15)
}

func TestCatmullRom_Midpoint_Linear(t *testing.T) {
	// Equally spaced -> midpoint should be 1.5
	got := CatmullRom(0, 1, 2, 3, 0.5)
	assertClose(t, "catmull-mid", got, 1.5, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Linear interpolation
// ═══════════════════════════════════════════════════════════════════════════

func TestLinearInterpolate_T0(t *testing.T) {
	assertClose(t, "lerp-t0", LinearInterpolate(10, 20, 0), 10.0, 0)
}

func TestLinearInterpolate_T1(t *testing.T) {
	assertClose(t, "lerp-t1", LinearInterpolate(10, 20, 1), 20.0, 0)
}

func TestLinearInterpolate_Mid(t *testing.T) {
	assertClose(t, "lerp-mid", LinearInterpolate(10, 20, 0.5), 15.0, 1e-15)
}

func TestLinearInterpolate_Extrapolate(t *testing.T) {
	assertClose(t, "lerp-extrap", LinearInterpolate(0, 10, 2.0), 20.0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Triangle area
// ═══════════════════════════════════════════════════════════════════════════

func TestTriangleArea2D_CCW(t *testing.T) {
	// (0,0), (1,0), (0,1) — CCW, area = 0.5
	got := TriangleArea2D(0, 0, 1, 0, 0, 1)
	assertClose(t, "tri-area-ccw", got, 0.5, 1e-15)
}

func TestTriangleArea2D_CW(t *testing.T) {
	// (0,0), (0,1), (1,0) — CW, area = -0.5
	got := TriangleArea2D(0, 0, 0, 1, 1, 0)
	assertClose(t, "tri-area-cw", got, -0.5, 1e-15)
}

func TestTriangleArea2D_Degenerate(t *testing.T) {
	// Collinear points
	got := TriangleArea2D(0, 0, 1, 1, 2, 2)
	assertClose(t, "tri-area-degen", got, 0.0, 1e-15)
}

func TestTriangleArea2D_LargeTriangle(t *testing.T) {
	// (0,0), (10,0), (0,10) — area = 50
	got := TriangleArea2D(0, 0, 10, 0, 0, 10)
	assertClose(t, "tri-area-large", got, 50.0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Point in triangle
// ═══════════════════════════════════════════════════════════════════════════

func TestPointInTriangle2D_Inside(t *testing.T) {
	got := PointInTriangle2D(0.25, 0.25, 0, 0, 1, 0, 0, 1)
	if !got {
		t.Error("expected point (0.25, 0.25) inside triangle")
	}
}

func TestPointInTriangle2D_Outside(t *testing.T) {
	got := PointInTriangle2D(2, 2, 0, 0, 1, 0, 0, 1)
	if got {
		t.Error("expected point (2, 2) outside triangle")
	}
}

func TestPointInTriangle2D_OnEdge(t *testing.T) {
	got := PointInTriangle2D(0.5, 0, 0, 0, 1, 0, 0, 1)
	if !got {
		t.Error("expected point (0.5, 0) on edge to be inside")
	}
}

func TestPointInTriangle2D_OnVertex(t *testing.T) {
	got := PointInTriangle2D(0, 0, 0, 0, 1, 0, 0, 1)
	if !got {
		t.Error("expected vertex point to be inside")
	}
}

func TestPointInTriangle2D_CWWinding(t *testing.T) {
	// CW winding: (0,0), (0,1), (1,0). Point at (0.1, 0.1) is inside.
	got := PointInTriangle2D(0.1, 0.1, 0, 0, 0, 1, 1, 0)
	if !got {
		t.Error("expected point inside CW triangle")
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — Convex hull
// ═══════════════════════════════════════════════════════════════════════════

func TestConvexHull2D_Square(t *testing.T) {
	points := [][2]float64{{0, 0}, {1, 0}, {1, 1}, {0, 1}}
	hull := ConvexHull2D(points)
	if len(hull) != 4 {
		t.Fatalf("expected 4 hull points, got %d: %v", len(hull), hull)
	}
}

func TestConvexHull2D_WithInterior(t *testing.T) {
	points := [][2]float64{
		{0, 0}, {2, 0}, {2, 2}, {0, 2},
		{1, 1}, // interior
	}
	hull := ConvexHull2D(points)
	if len(hull) != 4 {
		t.Fatalf("expected 4 hull points (interior excluded), got %d: %v", len(hull), hull)
	}
}

func TestConvexHull2D_Triangle(t *testing.T) {
	points := [][2]float64{{0, 0}, {1, 0}, {0.5, 1}}
	hull := ConvexHull2D(points)
	if len(hull) != 3 {
		t.Fatalf("expected 3 hull points, got %d", len(hull))
	}
}

func TestConvexHull2D_TwoPoints(t *testing.T) {
	points := [][2]float64{{0, 0}, {1, 1}}
	hull := ConvexHull2D(points)
	if len(hull) != 2 {
		t.Fatalf("expected 2 points back, got %d", len(hull))
	}
}

func TestConvexHull2D_Empty(t *testing.T) {
	hull := ConvexHull2D(nil)
	if hull != nil {
		t.Fatalf("expected nil for empty input, got %v", hull)
	}
}

func TestConvexHull2D_LargerSet(t *testing.T) {
	// Pentagon with interior points.
	points := [][2]float64{
		{0, 0}, {4, 0}, {5, 3}, {2, 5}, {-1, 3}, // hull
		{1, 1}, {2, 2}, {3, 1}, {2, 3},            // interior
	}
	hull := ConvexHull2D(points)
	if len(hull) != 5 {
		t.Fatalf("expected 5 hull points, got %d: %v", len(hull), hull)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

func assertClose(t *testing.T, label string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %v, want %v (tol %v)", label, got, want, tol)
	}
}

func assertQuat(t *testing.T, label string, got, want [4]float64, tol float64) {
	t.Helper()
	for i := 0; i < 4; i++ {
		if math.Abs(got[i]-want[i]) > tol {
			t.Errorf("%s[%d]: got %v, want %v (tol %v)", label, i, got[i], want[i], tol)
		}
	}
}

func assertVec3(t *testing.T, label string, got, want [3]float64, tol float64) {
	t.Helper()
	for i := 0; i < 3; i++ {
		if math.Abs(got[i]-want[i]) > tol {
			t.Errorf("%s[%d]: got %v, want %v (tol %v)", label, i, got[i], want[i], tol)
		}
	}
}

// inputFloat64Array3 extracts a named [3]float64 from a test case's Inputs.
func inputFloat64Array3(t *testing.T, tc testutil.TestCase, key string) [3]float64 {
	t.Helper()
	s := testutil.InputFloat64Slice(t, tc, key)
	if len(s) != 3 {
		t.Fatalf("[%s] input %q: expected 3 elements, got %d", tc.Description, key, len(s))
	}
	return [3]float64{s[0], s[1], s[2]}
}

// inputFloat64Array4 extracts a named [4]float64 from a test case's Inputs.
func inputFloat64Array4(t *testing.T, tc testutil.TestCase, key string) [4]float64 {
	t.Helper()
	s := testutil.InputFloat64Slice(t, tc, key)
	if len(s) != 4 {
		t.Fatalf("[%s] input %q: expected 4 elements, got %d", tc.Description, key, len(s))
	}
	return [4]float64{s[0], s[1], s[2], s[3]}
}
