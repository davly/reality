package combinatorics

import (
	"math"
	"testing"
)

// bruteForceUpOutCall is the ground-truth oracle: it enumerates all 2^n
// price paths on the binomial tree, knocks out any path whose price reaches
// the barrier at ANY step, and risk-neutrally averages the surviving
// discounted call payoffs. O(n·2^n) — only for small n in tests.
func bruteForceUpOutCall(s0, k, r, sigma, tt, barrier float64, n int) float64 {
	dt := tt / float64(n)
	u := math.Exp(sigma * math.Sqrt(dt))
	d := 1.0 / u
	disc := math.Exp(-r * dt)
	p := (math.Exp(r*dt) - d) / (u - d)
	total := 0.0
	for mask := 0; mask < (1 << uint(n)); mask++ {
		net, ups := 0, 0
		breached := false
		for i := 0; i < n; i++ {
			if mask&(1<<uint(i)) != 0 {
				net++
				ups++
			} else {
				net--
			}
			if s0*math.Pow(u, float64(net)) >= barrier {
				breached = true
				break
			}
		}
		if breached {
			continue
		}
		payoff := math.Max(s0*math.Pow(u, float64(net))-k, 0)
		if payoff == 0 {
			continue
		}
		total += math.Pow(p, float64(ups)) * math.Pow(1-p, float64(n-ups)) * payoff
	}
	return total * math.Pow(disc, float64(n))
}

// TestBarrierOptionReflection_MatchesBruteForce pins the reflection-principle
// pricer against the exhaustive oracle. Before the coordinate fix (jRefl was
// 2h-j instead of n+h-j, and above-barrier ITM nodes were not knocked out)
// the up-and-out call was overpriced by up to ~8x — this test failed.
func TestBarrierOptionReflection_MatchesBruteForce(t *testing.T) {
	cases := []struct {
		name                         string
		s0, k, r, sigma, tt, barrier float64
		n                            int
	}{
		{"moderate", 100, 100, 0.05, 0.20, 1, 130, 10},
		{"high-vol-above-barrier-nodes", 100, 100, 0.05, 0.40, 1, 115, 12},
		{"itm-strike-below", 100, 95, 0.03, 0.25, 0.5, 110, 12},
		{"tight-barrier", 100, 100, 0.05, 0.20, 1, 108, 10},
		{"wide-barrier-near-european", 100, 100, 0.05, 0.20, 1, 200, 10},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := BarrierOptionReflection(c.s0, c.k, c.r, c.sigma, c.tt, c.barrier, c.n)
			want := bruteForceUpOutCall(c.s0, c.k, c.r, c.sigma, c.tt, c.barrier, c.n)
			if math.IsNaN(got) {
				t.Fatalf("got NaN")
			}
			if math.Abs(got-want) > 1e-6 {
				t.Errorf("reflection = %.8f, brute force = %.8f (Δ = %.3e)", got, want, math.Abs(got-want))
			}
		})
	}
}
