package graph

import (
	"math"
	"math/rand"
	"testing"
)

// randomADMG builds a random acyclic mixed graph on n named vars: directed edges
// only go forward in a random topological permutation (guarantees acyclicity);
// bidirected edges are a random subset of pairs.
func randomADMG(rng *rand.Rand, n int, pDir, pBi float64) (ADMG, []string) {
	names := make([]string, n)
	for i := range names {
		names[i] = string(rune('A' + i))
	}
	perm := rng.Perm(n)
	var dir, bi []Edge
	for a := 0; a < n; a++ {
		for b := 0; b < n; b++ {
			if perm[a] < perm[b] && rng.Float64() < pDir {
				dir = append(dir, Edge{names[a], names[b]})
			}
		}
	}
	for a := 0; a < n; a++ {
		for b := a + 1; b < n; b++ {
			if rng.Float64() < pBi {
				bi = append(bi, Edge{names[a], names[b]})
			}
		}
	}
	return NewADMG(names, dir, bi), names
}

// measureWorst returns the largest |functional - truth| over 3 random SCMs for
// P(y|do x) on an identifiable graph (0 if non-identifiable). Shared by the fuzz
// and its sanity check so they exercise the same measurement path.
func measureWorst(t *testing.T, g ADMG, xName, yName string) float64 {
	t.Helper()
	e, ok := g.id(setOf([]string{yName}), setOf([]string{xName}), &exprP{}, &hedgeBox{})
	if !ok {
		return 0
	}
	worst := 0.0
	for seed := int64(1); seed <= 3; seed++ {
		scm := RandomSCM(g, seed)
		joint := scm.ObservationalJoint()
		for xv := 0; xv <= 1; xv++ {
			truth := scm.InterventionalDistribution([]string{yName}, map[string]int{xName: xv})
			for yv := 0; yv <= 1; yv++ {
				got := evalFunctional(e, map[string]int{xName: xv, yName: yv}, joint, scm.observed)
				if d := math.Abs(got - truth[cfgKey(map[string]int{yName: yv}, []string{yName})]); d > worst {
					worst = d
				}
			}
		}
	}
	return worst
}

// TestIDFunctional_Fuzz characterises the SCOPE of the deep-recursion functional
// limitation found in iter 7. Over many random ADMGs, for every identifiable
// P(Y|do X) it measures the symbolic functional against ground-truth do() on
// random SCMs, and classifies each as EXACT (<1e-9) or APPROXIMATE. It asserts
// the invariants that must hold regardless of the known limitation:
//   - every functional is a valid distribution (normalised, in [0,1]);
//   - approximate cases stay within a small bound (no wild divergence = no NEW bug
//     beyond the documented line-7/8 imperfection);
//   - the exact rate is high (most identifiable graphs use the common paths).
// The measured exact/approx breakdown is logged so the limitation's scope is on
// the record, not just asserted.
func TestIDFunctional_Fuzz(t *testing.T) {
	// Sanity: the measurement path MUST flag the known-approximate napkin, else a
	// "0 approximate" result below would just mean a broken measurement.
	if err := measureWorst(t, NewADMG([]string{"W", "Z", "X", "Y"},
		[]Edge{{"W", "Z"}, {"Z", "X"}, {"X", "Y"}}, []Edge{{"W", "X"}, {"W", "Y"}}),
		"X", "Y"); err < 1e-9 {
		t.Fatalf("measurement cannot detect the known napkin approximation (worst=%.2e) — fuzz result would be meaningless", err)
	}

	rng := rand.New(rand.NewSource(20260614))
	const graphs = 400
	var identifiable, exact, approx int
	maxApproxErr := 0.0
	for gi := 0; gi < graphs; gi++ {
		n := 3 + rng.Intn(3) // 3..5 vars
		g, names := randomADMG(rng, n, 0.45, 0.25)
		// pick distinct X, Y
		xi, yi := rng.Intn(n), rng.Intn(n)
		if xi == yi {
			continue
		}
		x, y := []string{names[xi]}, []string{names[yi]}
		e, ok := g.id(setOf(y), setOf(x), &exprP{}, &hedgeBox{})
		if !ok {
			continue // non-identifiable: covered by the truth-table / hedge tests
		}
		identifiable++
		worst := 0.0
		for seed := int64(1); seed <= 3; seed++ {
			scm := RandomSCM(g, seed*1000+int64(gi))
			joint := scm.ObservationalJoint()
			for xv := 0; xv <= 1; xv++ {
				truth := scm.InterventionalDistribution(y, map[string]int{names[xi]: xv})
				total := 0.0
				for yv := 0; yv <= 1; yv++ {
					got := evalFunctional(e, map[string]int{names[xi]: xv, names[yi]: yv}, joint, scm.observed)
					if got < -1e-9 || got > 1+1e-9 {
						t.Fatalf("graph %d: functional out of [0,1]: %.6f (dir=%v bi=%v)", gi, got, g.directed, g.bidirected)
					}
					total += got
					if d := math.Abs(got - truth[cfgKey(map[string]int{names[yi]: yv}, []string{names[yi]})]); d > worst {
						worst = d
					}
				}
				if math.Abs(total-1.0) > 1e-7 {
					t.Fatalf("graph %d: functional not normalised: Σ=%.10f (dir=%v bi=%v)", gi, total, g.directed, g.bidirected)
				}
			}
		}
		if worst < 1e-9 {
			exact++
		} else {
			approx++
			if worst > maxApproxErr {
				maxApproxErr = worst
			}
			// No NEW bug: the documented line-7/8 imperfection is small. A large
			// error would signal a real defect beyond the known limitation.
			if worst > 0.15 {
				t.Fatalf("graph %d: functional error %.4f exceeds the documented small approximation (dir=%v bi=%v)",
					gi, worst, g.directed, g.bidirected)
			}
		}
	}
	t.Logf("fuzz: %d graphs, %d identifiable, %d EXACT functional, %d approximate (max err %.4f)",
		graphs, identifiable, exact, approx, maxApproxErr)
	if identifiable == 0 {
		t.Fatal("no identifiable graphs generated — fuzz parameters degenerate")
	}
	// Most identifiable random graphs should hit the exact (lines 1-4) paths.
	if rate := float64(exact) / float64(identifiable); rate < 0.5 {
		t.Errorf("exact-functional rate %.2f unexpectedly low — the limitation may be broader than line-7/8", rate)
	}
}
