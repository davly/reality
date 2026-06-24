package physics

import (
	"math"
	"testing"
)

// TestPendulum_ZeroLengthReturnsNaN pins the doc/behaviour fix. The docstring
// promises "Returns NaN if L == 0", but the raw -(g/0)*sin(theta) is -Inf for
// most theta (NaN only when sin(theta) is exactly 0), so a math.IsNaN guard --
// the guard the doc invites -- used to slip the -Inf through. The L==0 guard now
// honors the contract for every theta.
func TestPendulum_ZeroLengthReturnsNaN(t *testing.T) {
	for _, theta := range []float64{0, 0.001, math.Pi / 6, 1.0, math.Pi / 2, math.Pi} {
		if got := Pendulum(theta, 0, 9.81, 0); !math.IsNaN(got) {
			t.Errorf("Pendulum(theta=%v, L=0) = %v, want NaN (documented contract)", theta, got)
		}
	}
	// A normal call still works.
	if got, want := Pendulum(0.01, 1.0, 9.81, 0), -9.81*math.Sin(0.01); math.Abs(got-want) > 1e-12 {
		t.Errorf("Pendulum(0.01,1,9.81,0) = %v, want %v", got, want)
	}
}
