package prob

import (
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

// Golden-file validators for the Session 38 canonical forge primitives.
// These vectors are the cross-language contract — Pistachio / RubberDuck /
// limitless-browser / Nexus implementations are all expected to produce
// byte-identical outputs for the raw FNV and situation-hash vectors, and
// within tolerance for the floating-point vectors.

func TestGoldenJeffreysConfidence(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/prob/jeffreys.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			s := testutil.InputFloat64(t, tc, "successes")
			f := testutil.InputFloat64(t, tc, "failures")
			got := JeffreysConfidence(s, f)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGoldenWilson(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/prob/wilson.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			p := testutil.InputFloat64(t, tc, "p")
			n := testutil.InputInt(t, tc, "n")
			z := testutil.InputFloat64(t, tc, "z")
			lo, hi := WilsonConfidenceInterval(p, n, z)
			testutil.AssertFloat64Slice(t, tc, []float64{lo, hi})
		})
	}
}

func TestGoldenEMATrajectory(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/prob/ema.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			alpha := testutil.InputFloat64(t, tc, "alpha")
			initial := testutil.InputFloat64(t, tc, "initial")
			steps := testutil.InputFloat64Slice(t, tc, "steps")
			v := initial
			traj := make([]float64, 0, len(steps))
			for _, s := range steps {
				v = EMA(v, s, alpha)
				traj = append(traj, v)
			}
			testutil.AssertFloat64Slice(t, tc, traj)
		})
	}
}

// Sanity: the Jeffreys canonical constant at (0,0) is exactly 0.5.
func TestJeffreysCanonical_ZeroZero_IsExactlyHalf(t *testing.T) {
	if got := JeffreysConfidence(0, 0); math.Abs(got-0.5) > 1e-15 {
		t.Errorf("JeffreysConfidence(0,0) = %v, canonical value is 0.5", got)
	}
}
