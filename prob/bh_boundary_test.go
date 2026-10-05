package prob

import (
	"math/big"
	"math/rand"
	"sort"
	"strconv"
	"testing"
)

// bhExactReference applies the Benjamini-Hochberg step-up rule in exact
// rational arithmetic: it rejects the hypotheses ranked 1..k for the largest
// k with p_(k) <= k*alpha/m.
func bhExactReference(p []*big.Rat, alpha *big.Rat) []bool {
	m := len(p)
	idx := make([]int, m)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(x, y int) bool { return p[idx[x]].Cmp(p[idx[y]]) < 0 })
	k := -1
	thr := new(big.Rat)
	for i := m - 1; i >= 0; i-- {
		thr.SetFrac64(int64(i+1), int64(m))
		thr.Mul(thr, alpha)
		if p[idx[i]].Cmp(thr) <= 0 {
			k = i
			break
		}
	}
	out := make([]bool, m)
	for i := 0; i <= k; i++ {
		out[idx[i]] = true
	}
	return out
}

// bhAlphas are significance levels as written in decimal.
var bhAlphas = []string{"0.001", "0.01", "0.025", "0.05", "0.1", "0.15", "0.2", "0.3"}

// ratFloat returns the float64 nearest to r.
func ratFloat(r *big.Rat) float64 {
	f, _ := r.Float64()
	return f
}

// randomDecimal returns a uniformly random multiple of 10^-digits in [lo, hi)
// as an exact rational, or nil if there is none.
func randomDecimal(rng *rand.Rand, lo, hi *big.Rat, digits int) *big.Rat {
	scale := new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(digits)), nil)
	// smallest multiple >= lo, largest < hi
	l := new(big.Rat).Mul(lo, new(big.Rat).SetInt(scale))
	h := new(big.Rat).Mul(hi, new(big.Rat).SetInt(scale))
	li := new(big.Int).Quo(l.Num(), l.Denom())
	if new(big.Rat).SetInt(li).Cmp(l) < 0 {
		li.Add(li, big.NewInt(1))
	}
	hi1 := new(big.Int).Quo(h.Num(), h.Denom())
	if new(big.Rat).SetInt(hi1).Cmp(h) >= 0 {
		hi1.Sub(hi1, big.NewInt(1))
	}
	if hi1.Cmp(li) < 0 {
		return nil
	}
	span := new(big.Int).Sub(hi1, li)
	span.Add(span, big.NewInt(1))
	v := new(big.Int).Rand(rng, span)
	v.Add(v, li)
	return new(big.Rat).SetFrac(v, scale)
}

// TestBenjaminiHochbergExactBoundary generates sets of p-values in which the
// p-value ranked i equals its threshold i*alpha/m exactly as a rational
// number, the smaller ranks hold smaller p-values and the larger ranks hold
// p-values above alpha, so that the boundary comparison alone decides the
// outcome: the rule rejects exactly the i smallest. p-values and alpha reach
// the function as their nearest float64, as a caller's data would.
func TestBenjaminiHochbergExactBoundary(t *testing.T) {
	rng := rand.New(rand.NewSource(20261005))
	const cases = 4000
	wrong := 0
	for trial := 0; trial < cases; trial++ {
		alphaStr := bhAlphas[rng.Intn(len(bhAlphas))]
		alpha, _ := new(big.Rat).SetString(alphaStr)
		alphaF, err := strconv.ParseFloat(alphaStr, 64)
		if err != nil {
			t.Fatal(err)
		}
		m := 1 + rng.Intn(60)
		i := 1 + rng.Intn(m)
		boundary := new(big.Rat).Mul(big.NewRat(int64(i), int64(m)), alpha)
		pr := make([]*big.Rat, 0, m)
		pr = append(pr, boundary)
		for j := 1; j < i; j++ {
			v := randomDecimal(rng, new(big.Rat), boundary, 6)
			if v == nil {
				v = new(big.Rat)
			}
			pr = append(pr, v)
		}
		for j := i; j < m; j++ {
			v := randomDecimal(rng, new(big.Rat).Add(alpha, big.NewRat(1, 1000000)), big.NewRat(1, 1), 6)
			pr = append(pr, v)
		}
		rng.Shuffle(len(pr), func(x, y int) { pr[x], pr[y] = pr[y], pr[x] })
		pf := make([]float64, m)
		for j, v := range pr {
			pf[j] = ratFloat(v)
		}
		want := bhExactReference(pr, alpha)
		got := BenjaminiHochberg(pf, alphaF)
		for j := range want {
			if got[j] != want[j] {
				wrong++
				if wrong <= 5 {
					t.Errorf("alpha=%s m=%d: p=%v (index %d, boundary rank %d): got %v, want %v", alphaStr, m, pf[j], j, i, got[j], want[j])
				}
				break
			}
		}
	}
	if wrong > 0 {
		t.Errorf("%d of %d exact-boundary cases decided differently from exact arithmetic", wrong, cases)
	}
}

// TestBenjaminiHochbergAwayFromBoundary checks random p-values (decimals
// with up to 6 digits) whose distance from every threshold k*alpha/m exceeds
// one part in 10^6 in exact arithmetic: the decisions must equal the exact
// rule, and the rule as it was before the boundary tolerance.
func TestBenjaminiHochbergAwayFromBoundary(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	checked := 0
	for trial := 0; trial < 4000; trial++ {
		alphaStr := bhAlphas[rng.Intn(len(bhAlphas))]
		alpha, _ := new(big.Rat).SetString(alphaStr)
		alphaF, _ := strconv.ParseFloat(alphaStr, 64)
		m := 1 + rng.Intn(60)
		pr := make([]*big.Rat, m)
		pf := make([]float64, m)
		near := false
		for j := range pr {
			hi := big.NewRat(1, 1)
			if rng.Intn(2) == 0 {
				hi = new(big.Rat).Set(alpha) // concentrate near the thresholds
			}
			v := randomDecimal(rng, new(big.Rat), hi, 6)
			pr[j], pf[j] = v, ratFloat(v)
			// The thresholds are spaced alpha/m apart, so only the two
			// around v*m/alpha can be close to v.
			pos := new(big.Rat).Quo(new(big.Rat).Mul(v, big.NewRat(int64(m), 1)), alpha)
			k0 := new(big.Int).Quo(pos.Num(), pos.Denom()).Int64()
			for k := k0; k <= k0+1 && !near; k++ {
				if k < 1 || k > int64(m) {
					continue
				}
				thr := new(big.Rat).Mul(big.NewRat(k, int64(m)), alpha)
				d := new(big.Rat).Sub(v, thr)
				d.Abs(d)
				if d.Cmp(new(big.Rat).Mul(thr, big.NewRat(1, 1000000))) <= 0 {
					near = true
				}
			}
		}
		if near {
			continue
		}
		checked++
		want := bhExactReference(pr, alpha)
		got := BenjaminiHochberg(pf, alphaF)
		old := benjaminiHochbergInsertionReference(pf, alphaF, 0)
		for j := range want {
			if got[j] != want[j] || old[j] != want[j] {
				t.Fatalf("alpha=%s m=%d p=%v: got %v, exact %v, previous rule %v", alphaStr, m, pf, got, want, old)
			}
		}
	}
	if checked < 1000 {
		t.Fatalf("only %d cases were away from every threshold; the generator needs adjusting", checked)
	}
	t.Logf("%d random cases away from every threshold: identical to the exact rule and to the previous rule", checked)
}
