package mdl

import (
	"math"
	"testing"
)

// bruteLogShtarkov computes log C(n,k) — the multinomial NML normalizing term —
// by full enumeration of count vectors, the independent ground truth:
//
//	C(n,k) = sum_{c_1+...+c_k = n} n!/(c_1!...c_k!) * prod_i (c_i/n)^{c_i}
//
// (with the 0*log(0)=0 / (0/n)^0=1 convention).
func bruteLogShtarkov(n, k int) float64 {
	logFact := func(m int) float64 { lg, _ := math.Lgamma(float64(m + 1)); return lg }
	total := 0.0
	var rec func(cat, rem int, logMultinom, prod float64)
	rec = func(cat, rem int, logMultinom, prod float64) {
		if cat == k-1 {
			c := rem
			p := prod
			if c > 0 {
				p *= math.Pow(float64(c)/float64(n), float64(c))
			}
			total += math.Exp(logMultinom-logFact(c)) * p
			return
		}
		for c := 0; c <= rem; c++ {
			p := prod
			if c > 0 {
				p *= math.Pow(float64(c)/float64(n), float64(c))
			}
			rec(cat+1, rem-c, logMultinom-logFact(c), p)
		}
	}
	rec(0, n, logFact(n), 1.0)
	return math.Log(total)
}

// TestNMLMultinomial_MatchesBruteForce pins the Kontkanen-Myllymäki recurrence
// divisor. The bug used n/(kk-1); the correct KM-2007 divisor (reindexed
// C(n,K)=C(n,K-1)+(n/(K-2))C(n,K-2)) is n/(kk-2). The library recurrence must
// match the exact Shtarkov sum for k >= 3.
func TestNMLMultinomial_MatchesBruteForce(t *testing.T) {
	cases := [][]int{
		{3, 3, 4},    // n=10, k=3
		{2, 4, 2, 2}, // n=10, k=4
		{2, 1, 1},    // n=4,  k=3
		{1, 1, 1, 1}, // n=4,  k=4
		{5, 5, 5, 5, 5}, // n=25, k=5
	}
	for _, counts := range cases {
		n := 0
		for _, c := range counts {
			n += c
		}
		got, err := NMLMultinomial(counts)
		if err != nil {
			t.Fatalf("NMLMultinomial(%v): %v", counts, err)
		}
		want := bruteLogShtarkov(n, len(counts))
		if math.Abs(got-want) > 1e-6 {
			t.Errorf("NMLMultinomial(%v) = %.8f, brute-force exact = %.8f (Δ=%.5f nats)", counts, got, want, got-want)
		}
	}
}
