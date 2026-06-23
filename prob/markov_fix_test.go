package prob

import (
	"math"
	"testing"
)

// Regression test for the MarkovSteadyState periodic-chain fix. Plain power
// iteration oscillates forever on a periodic chain, so the old code returned a
// non-stationary vector (e.g. [1/3,1/3,1/3] for a chain whose true stationary
// is [1/2,1/4,1/4]). The lazy-chain (I+P)/2 iteration converges correctly.

func piTimesP(pi, P []float64, n int) []float64 {
	out := make([]float64, n)
	for j := 0; j < n; j++ {
		s := 0.0
		for i := 0; i < n; i++ {
			s += pi[i] * P[i*n+j]
		}
		out[j] = s
	}
	return out
}

func assertStationary(t *testing.T, name string, P []float64, n int, want []float64) {
	t.Helper()
	pi := MarkovSteadyState(P, n)
	if pi == nil {
		t.Fatalf("%s: got nil", name)
	}
	// Must satisfy the documented contract pi*P == pi.
	pP := piTimesP(pi, P, n)
	maxdev := 0.0
	for i := 0; i < n; i++ {
		maxdev = math.Max(maxdev, math.Abs(pP[i]-pi[i]))
	}
	if maxdev > 1e-9 {
		t.Errorf("%s: pi=%v does NOT satisfy pi*P=pi (max dev %.3g)", name, pi, maxdev)
	}
	if want != nil {
		for i := 0; i < n; i++ {
			if math.Abs(pi[i]-want[i]) > 1e-6 {
				t.Errorf("%s: pi=%v want %v", name, pi, want)
				break
			}
		}
	}
	// Probability distribution sanity.
	sum := 0.0
	for _, p := range pi {
		sum += p
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("%s: sum(pi)=%g != 1", name, sum)
	}
}

func TestMarkovSteadyState_PeriodicChain(t *testing.T) {
	// Period-2 irreducible chain; unique stationary [1/2,1/4,1/4].
	P := []float64{0, 0.5, 0.5, 1, 0, 0, 1, 0, 0}
	assertStationary(t, "period-2", P, 3, []float64{0.5, 0.25, 0.25})
}

func TestMarkovSteadyState_KnownChains(t *testing.T) {
	// 2-state aperiodic: [[0.9,0.1],[0.5,0.5]] -> pi=[5/6,1/6].
	assertStationary(t, "2-state", []float64{0.9, 0.1, 0.5, 0.5}, 2,
		[]float64{5.0 / 6.0, 1.0 / 6.0})

	// Doubly-stochastic -> uniform.
	assertStationary(t, "doubly-stochastic", []float64{0.2, 0.3, 0.5, 0.5, 0.2, 0.3, 0.3, 0.5, 0.2}, 3,
		[]float64{1.0 / 3, 1.0 / 3, 1.0 / 3})

	// A pure 2-cycle [[0,1],[1,0]] (period 2) -> uniform [0.5,0.5].
	assertStationary(t, "2-cycle", []float64{0, 1, 1, 0}, 2, []float64{0.5, 0.5})
}
