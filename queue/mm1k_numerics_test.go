package queue

import (
	"math"
	"testing"
)

// TestMM1K_HighLoadOverflow pins the HIGH bug: for rho>1 and large K, rho^(K+1)
// overflowed to +Inf, poisoning pLoss/L/Lq/W/Wq to NaN (MM1K(10,1,308) -> all
// NaN). The overflow-safe reformulation must return finite values matching the
// large-K analytical limits pLoss -> (rho-1)/rho and L -> (K+1)+rho/(1-rho).
func TestMM1K_HighLoadOverflow(t *testing.T) {
	cases := []struct {
		lam, mu float64
		K       int
	}{
		{10, 1, 308}, {1000, 1, 500}, {5, 1, 500}, {2, 1, 1023},
	}
	for _, tc := range cases {
		Lq, Wq, L, W, rho, pLoss := MM1K(tc.lam, tc.mu, tc.K)
		for name, v := range map[string]float64{"L": L, "Lq": Lq, "W": W, "Wq": Wq, "pLoss": pLoss} {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("MM1K(%v,%v,%d): %s=%v, want finite", tc.lam, tc.mu, tc.K, name, v)
			}
		}
		if wantP := (rho - 1) / rho; math.Abs(pLoss-wantP) > 1e-3 {
			t.Errorf("MM1K(%v,%v,%d): pLoss=%v, want ~%v", tc.lam, tc.mu, tc.K, pLoss, wantP)
		}
		if wantL := float64(tc.K+1) + rho/(1-rho); math.Abs(L-wantL) > 1.0 {
			t.Errorf("MM1K(%v,%v,%d): L=%v, want ~%v", tc.lam, tc.mu, tc.K, L, wantL)
		}
	}
}

// TestMM1K_NearUnitPrecision pins the MED bug: with the 1e-12 special-case window,
// rho just off 1 used the closed form, which catastrophically cancelled and gave a
// wildly wrong L (K=100, rho=1+1e-10 -> L=100 vs true 50). For tiny |rho-1| the
// true L is ~K/2; the direct-sum branch must deliver that.
func TestMM1K_NearUnitPrecision(t *testing.T) {
	cases := []struct {
		rho float64
		K   int
	}{
		{1 + 1e-10, 100}, {1 + 1e-9, 100}, {1 - 1e-9, 1000}, {1 + 1e-8, 10}, {1.0, 100},
	}
	for _, tc := range cases {
		_, _, L, _, _, _ := MM1K(tc.rho, 1.0, tc.K) // lambda=rho, mu=1 => rho
		want := float64(tc.K) / 2.0
		if math.Abs(L-want) > 0.01 {
			t.Errorf("MM1K(rho=%v, K=%d): L=%v, want ~%v (K/2 for rho near 1)", tc.rho, tc.K, L, want)
		}
	}
}
