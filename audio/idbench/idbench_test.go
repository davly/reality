package idbench

import (
	"math/rand"
	"testing"
)

// makeSamples draws n Gaussian feature vectors around mean (fixed-seed rng for
// determinism).
func makeSamples(rng *rand.Rand, id string, mean []float64, n int, noise float64) []Sample {
	out := make([]Sample, n)
	for i := 0; i < n; i++ {
		x := make([]float64, len(mean))
		for j := range mean {
			x[j] = mean[j] + rng.NormFloat64()*noise
		}
		out[i] = Sample{ID: id, X: x}
	}
	return out
}

// The discrimination-proof KAT: with well-separated per-individual clusters the
// instrument MUST show strong separation (ratio<<1, low EER, rank-1 ~ 1.0); with
// the SAME vectors but randomized labels it MUST collapse to ~chance. An
// instrument that can't tell signal from shuffled noise is worthless — this is
// the red/green discrimination anchor (the naive no-impostor version forces
// ratio=1 and fails the separated case).
func TestEvaluate_DiscriminatesSeparatedClusters(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	dim := 4
	means := map[string][]float64{
		"a": {0, 0, 0, 0},
		"b": {10, 10, 10, 10},
		"c": {20, 20, 20, 20},
	}
	var enroll, probe []Sample
	for id, m := range means {
		enroll = append(enroll, makeSamples(rng, id, m, 8, 1.0)...)
		probe = append(probe, makeSamples(rng, id, m, 8, 1.0)...)
	}

	rep, err := Evaluate(enroll, probe, dim, 1e-6)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if rep.NIndividuals != 3 {
		t.Errorf("NIndividuals=%d, want 3", rep.NIndividuals)
	}
	if rep.IntraInterRatio >= 0.5 {
		t.Errorf("separated clusters: IntraInterRatio=%v, want < 0.5 (intra << inter)", rep.IntraInterRatio)
	}
	if rep.EER >= 0.2 {
		t.Errorf("separated clusters: EER=%v, want < 0.2", rep.EER)
	}
	if rep.Rank1Accuracy < 0.99 {
		t.Errorf("separated clusters: Rank1Accuracy=%v, want ~1.0", rep.Rank1Accuracy)
	}

	// Shuffle-control: same vectors, randomized labels -> no discrimination.
	all := append(append([]Sample{}, enroll...), probe...)
	ids := []string{"a", "b", "c"}
	shufRng := rand.New(rand.NewSource(7))
	shuffled := make([]Sample, len(all))
	for i, s := range all {
		shuffled[i] = Sample{ID: ids[shufRng.Intn(3)], X: s.X}
	}
	en2 := shuffled[:len(shuffled)/2]
	pr2 := shuffled[len(shuffled)/2:]
	rep2, err := Evaluate(en2, pr2, dim, 1e-6)
	if err != nil {
		t.Fatalf("Evaluate (shuffled): %v", err)
	}
	if rep2.IntraInterRatio < 0.7 {
		t.Errorf("shuffled labels: IntraInterRatio=%v, want >= 0.7 (no discrimination)", rep2.IntraInterRatio)
	}
	if rep2.EER < 0.3 {
		t.Errorf("shuffled labels: EER=%v, want >= 0.3 (near chance)", rep2.EER)
	}
}

func TestEvaluate_Validation(t *testing.T) {
	if _, err := Evaluate(nil, nil, 4, 1e-6); err == nil {
		t.Error("want error on empty enroll/probe")
	}
	one := []Sample{{ID: "a", X: []float64{1, 2, 3, 4}}}
	if _, err := Evaluate(one, one, 0, 1e-6); err == nil {
		t.Error("want error on dim < 1")
	}
}
