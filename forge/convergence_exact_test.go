package forge

import (
	"math/big"
	"testing"
)

// ratOracle is an independent exact reference for the convergence rule, computed with
// big.Rat (no float anywhere). DecideExact must agree with it on every well-formed input.
func ratOracle(domNum, domDen, confNum, confDen, total int) Verdict {
	if domDen <= 0 || confDen <= 0 || total < MinObservations {
		return VerdictUncertain
	}
	dom := big.NewRat(int64(domNum), int64(domDen))
	conf := big.NewRat(int64(confNum), int64(confDen))
	if dom.Cmp(big.NewRat(7, 10)) >= 0 && conf.Cmp(big.NewRat(13, 20)) >= 0 {
		return VerdictConverged
	}
	if dom.Cmp(big.NewRat(3, 5)) < 0 {
		return VerdictEscape
	}
	return VerdictUncertain
}

func TestDecideExact_Table(t *testing.T) {
	cases := []struct {
		domN, domD, confN, confD, total int
		want                            Verdict
		note                            string
	}{
		{7, 10, 13, 20, 100, VerdictConverged, "both exactly at floor → converged"},
		{8, 10, 7, 10, 100, VerdictConverged, "clear converged"},
		{59, 100, 99, 100, 100, VerdictEscape, "dominance just below 3/5 → escape"},
		{3, 5, 99, 100, 100, VerdictUncertain, "dominance exactly 3/5: not <3/5, not >=7/10 → uncertain"},
		{7, 10, 64, 100, 100, VerdictUncertain, "dominance ok but confidence below 13/20 → uncertain"},
		{9, 10, 13, 20, 2, VerdictUncertain, "below MinObservations → uncertain"},
		{9, 10, 13, 20, 0, VerdictUncertain, "zero observations → uncertain"},
		{7, 0, 13, 20, 100, VerdictUncertain, "zero dominance denom → fail closed"},
		{7, 10, 13, 0, 100, VerdictUncertain, "zero confidence denom → fail closed"},
		{7, -10, 13, 20, 100, VerdictUncertain, "negative denom → fail closed"},
	}
	for _, c := range cases {
		got := DecideExact(c.domN, c.domD, c.confN, c.confD, c.total)
		if got != c.want {
			t.Errorf("DecideExact(%d/%d, %d/%d, n=%d)=%v want %v (%s)", c.domN, c.domD, c.confN, c.confD, c.total, got, c.want, c.note)
		}
		// and it must equal the independent big.Rat oracle
		if orc := ratOracle(c.domN, c.domD, c.confN, c.confD, c.total); orc != got {
			t.Errorf("DecideExact disagrees with big.Rat oracle on %+v: %v vs %v", c, got, orc)
		}
	}
}

// TestDecideExact_OracleGridAndSeam proves DecideExact is EXACT over a dense grid (matches the
// big.Rat oracle on every input), and counts where the float Decide path disagrees — those
// disagreements are precisely the numeric-ingestion seam DecideExact eliminates, and at each
// one DecideExact equals the exact oracle (i.e. the exact verdict is the correct one).
func TestDecideExact_OracleGridAndSeam(t *testing.T) {
	seam := 0
	checked := 0
	for domD := 2; domD <= 300; domD++ {
		for domN := 0; domN <= domD; domN++ {
			confN, confD := 99, 100 // hold confidence above its floor; dominance drives the verdict
			checked++
			exact := DecideExact(domN, domD, confN, confD, 100)
			if orc := ratOracle(domN, domD, confN, confD, 100); exact != orc {
				t.Fatalf("DecideExact NOT exact: dominance %d/%d → %v but oracle %v", domN, domD, exact, orc)
			}
			flt := Decide(float64(domN)/float64(domD), float64(confN)/float64(confD), 100)
			if flt != exact {
				seam++ // float path disagrees with the exact (correct) verdict — the seam
			}
		}
	}
	t.Logf("grid checked=%d; DecideExact exact on all; float-Decide seam disagreements=%d", checked, seam)
}
