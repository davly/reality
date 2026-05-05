package changepoint

import (
	"math"
	"testing"

	"github.com/davly/reality/infogeo"
)

// TestPosterior_FreshStartConvergence demonstrates the substrate-internal
// composition of changepoint × infogeo. After a changepoint, BOCPD's
// run-length posterior should look like a "fresh-start" posterior — what
// a BOCPD instance constructed from scratch and fed only the post-shift
// data would produce — because mass has piled on low run-lengths. Before
// the changepoint, the posterior is concentrated at the long-running r,
// which a fresh-start instance cannot reproduce.
//
// Quantitatively, using infogeo.TotalVariation and infogeo.Hellinger as
// the witness metrics (both bounded in [0, 1]):
//
//   - TV(post_CP_posterior, fresh_start_posterior) is small (< 0.25):
//     after a clean changepoint, BOCPD-on-full-data looks like
//     BOCPD-on-post-CP-only.
//   - TV(pre_CP_posterior, fresh_start_posterior) is large (> 0.6):
//     the long-running posterior is far from the fresh-start one.
//   - Hellinger reproduces the same ordering (both metrics agree on
//     direction).
//
// This test is the first verified consumer for both
// github.com/davly/reality/changepoint and
// github.com/davly/reality/infogeo (substrate-internal first-consumer
// push, S62 2026-05-05). Each package is the other's substrate witness.
//
// Construction matches TestUpdate_DetectsStepShift in bocpd_test.go
// (Lambda=100, RMax=200, preCP=50 N(0,1) samples + postCP=20 N(5,1)
// samples) so the same regime that's known to produce a clean
// detection produces a clean infogeo-witnessed signature.
func TestPosterior_FreshStartConvergence(t *testing.T) {
	const (
		preCP  = 50
		postCP = 20
		seed   = uint64(0xc0ffee)
		preMu  = 0.0
		postMu = 5.0
		sigma  = 1.0
	)

	cfg := Config{Prior: DefaultNigPrior(), RMax: 200, Lambda: 100.0}

	xs := synthesizeBoxMuller(seed, preCP, preMu, sigma, postMu, sigma, postCP)

	// Run A: BOCPD on the FULL (preCP + postCP) series.
	bocpdFull, err := New(cfg)
	if err != nil {
		t.Fatalf("bocpdFull New: %v", err)
	}
	var preSnapshot, postSnapshot []float64
	for i, x := range xs {
		if _, err := bocpdFull.Update(x); err != nil {
			t.Fatalf("bocpdFull.Update(t=%d): %v", i, err)
		}
		if i == preCP-1 {
			preSnapshot = append([]float64(nil), bocpdFull.RunLengthPosterior()...)
		}
	}
	postSnapshot = append([]float64(nil), bocpdFull.RunLengthPosterior()...)

	// Run B: BOCPD-fresh on JUST the post-CP samples.
	bocpdFresh, err := New(cfg)
	if err != nil {
		t.Fatalf("bocpdFresh New: %v", err)
	}
	for i, x := range xs[preCP:] {
		if _, err := bocpdFresh.Update(x); err != nil {
			t.Fatalf("bocpdFresh.Update(t=%d): %v", i, err)
		}
	}
	freshSnapshot := append([]float64(nil), bocpdFresh.RunLengthPosterior()...)

	// Pad to a common length (the longest of the three).
	n := len(preSnapshot)
	if len(postSnapshot) > n {
		n = len(postSnapshot)
	}
	if len(freshSnapshot) > n {
		n = len(freshSnapshot)
	}
	pre := padTo(preSnapshot, n)
	post := padTo(postSnapshot, n)
	fresh := padTo(freshSnapshot, n)

	// Helper to compute and report both metrics for a posterior pair.
	measure := func(label string, p, q []float64) (tv, hel float64) {
		tv, err := infogeo.TotalVariation(p, q)
		if err != nil {
			t.Fatalf("%s TV: %v", label, err)
		}
		hel, err = infogeo.Hellinger(p, q)
		if err != nil {
			t.Fatalf("%s Hellinger: %v", label, err)
		}
		return tv, hel
	}

	preFreshTV, preFreshHel := measure("pre vs fresh", pre, fresh)
	postFreshTV, postFreshHel := measure("post vs fresh", post, fresh)

	// post-CP-A should be close to fresh-start.
	if postFreshTV > 0.25 {
		t.Errorf("post-CP posterior not close enough to fresh-start: TV=%g (want <= 0.25)", postFreshTV)
	}
	if postFreshHel > 0.35 {
		t.Errorf("post-CP posterior not close enough to fresh-start: Hellinger=%g (want <= 0.35)", postFreshHel)
	}

	// pre-CP-A should be far from fresh-start.
	if preFreshTV < 0.6 {
		t.Errorf("pre-CP posterior not far enough from fresh-start: TV=%g (want >= 0.6)", preFreshTV)
	}
	if preFreshHel < 0.7 {
		t.Errorf("pre-CP posterior not far enough from fresh-start: Hellinger=%g (want >= 0.7)", preFreshHel)
	}

	// Both metrics must agree on direction (post < pre); a contradiction
	// would imply infogeo's TV and Hellinger disagree on monotonic
	// ordering, which would be a real bug in either package.
	if !(postFreshTV < preFreshTV && postFreshHel < preFreshHel) {
		t.Errorf("TV and Hellinger disagree on ordering: pre TV=%g pre Hel=%g post TV=%g post Hel=%g",
			preFreshTV, preFreshHel, postFreshTV, postFreshHel)
	}
}

// padTo zero-extends p to length n. Returns p unchanged if len(p) >= n.
func padTo(p []float64, n int) []float64 {
	if len(p) >= n {
		return p
	}
	out := make([]float64, n)
	copy(out, p)
	return out
}

// synthesizeBoxMuller produces a deterministic length-(n1+n2) series of
// Gaussian samples drawn via Box-Muller from a Lehmer LCG keyed on seed.
// First n1 samples are N(mu1, sigma1); last n2 samples are N(mu2, sigma2).
// Used to produce a reproducible series with a known changepoint at index
// n1 for the BOCPD × infogeo composition test above.
func synthesizeBoxMuller(seed uint64, n1 int, mu1, sigma1 float64, mu2, sigma2 float64, n2 int) []float64 {
	rng := seed
	xs := make([]float64, 0, n1+n2)
	uniform := func() float64 {
		rng = rng*6364136223846793005 + 1442695040888963407
		return (float64(rng>>32) + 1) / (4294967296.0 + 1)
	}
	emit := func(mu, sigma float64) {
		u1 := uniform()
		u2 := uniform()
		r := math.Sqrt(-2.0 * math.Log(u1))
		theta := 2.0 * math.Pi * u2
		z := r * math.Cos(theta)
		xs = append(xs, mu+sigma*z)
	}
	for i := 0; i < n1; i++ {
		emit(mu1, sigma1)
	}
	for i := 0; i < n2; i++ {
		emit(mu2, sigma2)
	}
	return xs
}
