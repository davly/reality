package geometry

import (
	"math"
	"testing"
)

// QuatToAxisAngle against the exact axis and angle of the float64 quaternion
// (50-digit values, see quat_axisangle_data_test.go). The angle is accurate to
// about 1 ulp relative at every size; the former acos(w) form returned angle 0
// below ~2e-8 rad and a relative error of 6e-5 at 1e-6 rad, and the same near
// 2*pi, where w is close to -1.
func TestQuatToAxisAngle_MatchesTheExactAxisAndAngle(t *testing.T) {
	const tol = 1e-15 // relative for the angle, absolute for the axis; measured worst 2.4e-16
	worst := 0.0
	for _, c := range quatAxisAngleCases {
		axis, angle := QuatToAxisAngle(c.q)
		rel := math.Abs(angle-c.angle) / c.angle
		if rel > tol {
			t.Errorf("%s theta=%v q=%v: angle %v, want %v (relative error %.3g)", c.kind, c.theta, c.q, angle, c.angle, rel)
		}
		worst = math.Max(worst, rel)
		for i := range axis {
			if d := math.Abs(axis[i] - c.axis[i]); d > tol {
				t.Errorf("%s theta=%v q=%v: axis[%d] = %v, want %v", c.kind, c.theta, c.q, i, axis[i], c.axis[i])
			}
		}
		if n := math.Abs(math.Sqrt(axis[0]*axis[0]+axis[1]*axis[1]+axis[2]*axis[2]) - 1); n > tol {
			t.Errorf("%s theta=%v: axis %v has length error %.3g", c.kind, c.theta, axis, n)
		}
	}
	t.Logf("%d cases; worst relative angle error %.3g", len(quatAxisAngleCases), worst)
}

// A rotation about a coordinate axis has that axis exactly (the vector part is
// divided by its own norm, and sqrt(x*x) == |x|).
func TestQuatToAxisAngle_CoordinateAxesAreExact(t *testing.T) {
	for _, theta := range []float64{1e-8, 1e-6, 1e-3, 0.1, 1, 3, 3.5, 6} {
		for _, ax := range [][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, -1}} {
			axis, angle := QuatToAxisAngle(QuatFromAxisAngle(ax, theta))
			if axis != ax {
				t.Errorf("theta=%v about %v: axis %v", theta, ax, axis)
			}
			if rel := math.Abs(angle-theta) / theta; rel > 1e-15 {
				t.Errorf("theta=%v about %v: angle %v (relative error %.3g)", theta, ax, angle, rel)
			}
		}
	}
}

// No rotation (a vector part below 1e-10, whatever the sign of w) is reported as
// the +Z axis and angle 0; a vector part above that threshold is a rotation,
// even a tiny one, and is reported accurately.
func TestQuatToAxisAngle_NoRotationThreshold(t *testing.T) {
	for _, q := range [][4]float64{{1, 0, 0, 0}, {-1, 0, 0, 0}, {1, 3e-11, 0, 0}, {-1, 0, 5e-11, 0}} {
		axis, angle := QuatToAxisAngle(q)
		if axis != [3]float64{0, 0, 1} || angle != 0 {
			t.Errorf("q=%v: got axis %v angle %v, want +Z and 0", q, axis, angle)
		}
	}
	for _, c := range quatThresholdCases {
		axis, angle := QuatToAxisAngle(c.q)
		if c.q[1] < 1e-10 {
			if axis != [3]float64{0, 0, 1} || angle != 0 {
				t.Errorf("q=%v: got axis %v angle %v, want +Z and 0", c.q, axis, angle)
			}
			continue
		}
		if axis != [3]float64{1, 0, 0} || math.Abs(angle-c.angle)/c.angle > 1e-15 {
			t.Errorf("q=%v: got axis %v angle %v, want +X and %v", c.q, axis, angle, c.angle)
		}
	}
}

// The double cover: -q is the same rotation. The angle and axis come out as
// (2*pi - angle, -axis), which rotate identically.
func TestQuatToAxisAngle_NegatedQuaternion(t *testing.T) {
	q := QuatFromAxisAngle([3]float64{0, 1, 0}, 1e-3)
	axis, angle := QuatToAxisAngle([4]float64{-q[0], -q[1], -q[2], -q[3]})
	if axis != [3]float64{0, -1, 0} {
		t.Errorf("axis %v, want -Y", axis)
	}
	if want := 2*math.Pi - 1e-3; math.Abs(angle-want) > 1e-14 {
		t.Errorf("angle %v, want %v", angle, want)
	}
}
