package audio

// Precision property tests — pins the Precision: docstring bounds for the
// mel-scale conversions as TESTED INVARIANTS (cross-poll: honesty-as-a-tested-
// invariant, deepened into the Tier-0 math substrate).
//
// Pure Go stdlib only (testing + testing/quick + math): preserves the zero-dep
// law of github.com/davly/reality. ADDITIVE — no math/source change.
//
// Claims pinned here:
//   - melscale.go:38  MelToHz: "HzToMel(MelToHz(m)) round-trips to <= 1e-9 of
//     m for m in [0, 8000]."
//   - melscale.go:15  HzToMel: "monotonically increasing" (qualitative but
//     trivially checkable).
//   - melscale.go:37  MelToHz: "monotonically increasing".

import (
	"math"
	"testing"
	"testing/quick"
)

const melRoundTripBound = 1e-9 // docstring melscale.go:38

// genMelDomain returns a uniform sample in the documented round-trip domain
// [0, 8000] from a quick.Config-supplied uint64.
func genMelDomain(u uint64) float64 {
	return (float64(u) / float64(math.MaxUint64)) * 8000.0
}

// TestMelToHzRoundTripBound pins melscale.go:38:
// HzToMel(MelToHz(m)) round-trips to <= 1e-9 of m for m in [0, 8000].
func TestMelToHzRoundTripBound(t *testing.T) {
	var worst float64
	var worstAt float64
	prop := func(u uint64) bool {
		m := genMelDomain(u)
		back := HzToMel(MelToHz(m))
		err := math.Abs(back - m)
		if err > worst {
			worst, worstAt = err, m
		}
		return err <= melRoundTripBound
	}
	if err := quick.Check(prop, &quick.Config{MaxCount: 200000}); err != nil {
		// Enforced invariant: this bound holds today (worst ~9.09e-13), so a
		// future regression must turn the suite RED, not silently SKIP.
		t.Errorf("PRECISION REGRESSION: MelToHz docstring claims HzToMel(MelToHz(m)) <= %g of m for m in [0,8000], observed worst abs error %g at m=%g — needs a tightened impl or an honest docstring", melRoundTripBound, worst, worstAt)
	}
	t.Logf("PINNED melscale.go:38 round-trip <= %g over [0,8000]: worst observed |HzToMel(MelToHz(m))-m| = %g at m=%g", melRoundTripBound, worst, worstAt)
}

// TestMelToHzRoundTripDeterministicGrid is a deterministic, dependency-free
// dense-grid companion to the randomized property (the random one can miss a
// pathological m; the grid guarantees endpoint + uniform coverage).
func TestMelToHzRoundTripDeterministicGrid(t *testing.T) {
	const steps = 80001 // 0.1 mel spacing across [0, 8000]
	var worst, worstAt float64
	for i := 0; i < steps; i++ {
		m := 8000.0 * float64(i) / float64(steps-1)
		err := math.Abs(HzToMel(MelToHz(m)) - m)
		if err > worst {
			worst, worstAt = err, m
		}
	}
	if worst > melRoundTripBound {
		t.Errorf("PRECISION REGRESSION: MelToHz docstring claims round-trip <= %g over [0,8000], observed %g at m=%g", melRoundTripBound, worst, worstAt)
	}
	t.Logf("PINNED (grid) melscale.go:38: worst round-trip error %g at m=%g (bound %g)", worst, worstAt, melRoundTripBound)
}

// TestHzToMelMonotonicProperty pins melscale.go:15 "monotonically increasing".
func TestHzToMelMonotonicProperty(t *testing.T) {
	prop := func(a, b uint64) bool {
		// Two frequencies in [0, 20000] (well beyond the 8000 round-trip
		// domain; monotonicity is claimed for the whole f >= 0 domain).
		fa := (float64(a) / float64(math.MaxUint64)) * 20000.0
		fb := (float64(b) / float64(math.MaxUint64)) * 20000.0
		if fa == fb {
			return true
		}
		ma, mb := HzToMel(fa), HzToMel(fb)
		if fa < fb {
			return ma < mb
		}
		return ma > mb
	}
	if err := quick.Check(prop, &quick.Config{MaxCount: 100000}); err != nil {
		t.Fatalf("HzToMel monotonicity violated: %v", err)
	}
}

// TestMelToHzMonotonic pins melscale.go:37 "monotonically increasing".
func TestMelToHzMonotonic(t *testing.T) {
	prop := func(a, b uint64) bool {
		ma := (float64(a) / float64(math.MaxUint64)) * 4000.0 // mel up to ~Nyquist of 44.1k
		mb := (float64(b) / float64(math.MaxUint64)) * 4000.0
		if ma == mb {
			return true
		}
		fa, fb := MelToHz(ma), MelToHz(mb)
		if ma < mb {
			return fa < fb
		}
		return fa > fb
	}
	if err := quick.Check(prop, &quick.Config{MaxCount: 100000}); err != nil {
		t.Fatalf("MelToHz monotonicity violated: %v", err)
	}
}

// TestMelScaleZeroExact pins the documented "Returns 0 for f == 0 / m == 0".
func TestMelScaleZeroExact(t *testing.T) {
	if HzToMel(0) != 0 {
		t.Errorf("HzToMel(0) = %v, want exact 0", HzToMel(0))
	}
	if MelToHz(0) != 0 {
		t.Errorf("MelToHz(0) = %v, want exact 0", MelToHz(0))
	}
}
