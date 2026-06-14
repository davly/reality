package prob

import (
	"math"
	"math/rand"
	"testing"
)

func sampleMean(xs []float64) float64 {
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

// TestMedianOfMeans_RobustToOutliers is the core value claim: on contaminated
// data (a true mean plus a few gross outliers) median-of-means stays close to the
// true mean while the sample mean is wrecked. Discriminating — taking the MEAN of
// block means instead of the median would also be wrecked, so this fails if the
// median step is removed (verified by the revert harness).
func TestMedianOfMeans_RobustToOutliers(t *testing.T) {
	const (
		trueMean = 5.0
		n        = 1200
		blocks   = 49 // MoM tolerates floor((k-1)/2)=24 corrupted blocks
		nOut     = 10 // <= 10 distinct blocks corrupted, well within breakdown
		trials   = 200
	)
	rng := rand.New(rand.NewSource(1))
	momWins := 0
	for tr := 0; tr < trials; tr++ {
		xs := make([]float64, n)
		for i := range xs {
			xs[i] = trueMean + rng.NormFloat64() // clean ~N(5,1)
		}
		// A handful of gross outliers, corrupting fewer than blocks/2 blocks.
		for i := 0; i < nOut; i++ {
			xs[rng.Intn(n)] += 1e6
		}
		mom, err := MedianOfMeans(xs, blocks)
		if err != nil {
			t.Fatalf("MedianOfMeans: %v", err)
		}
		sm := sampleMean(xs)
		if math.Abs(mom-trueMean) < math.Abs(sm-trueMean) {
			momWins++
		}
		// MoM must be close to truth in absolute terms.
		if math.Abs(mom-trueMean) > 0.5 {
			t.Fatalf("trial %d: MoM=%.4f far from true mean %.1f", tr, mom, trueMean)
		}
	}
	if momWins < trials { // MoM should beat the sample mean on every contaminated trial
		t.Fatalf("MoM beat the sample mean on only %d/%d trials", momWins, trials)
	}
}

// TestMedianOfMeans_CleanMatchesMean: on clean Gaussian data MoM is close to the
// sample mean (no robustness penalty worth worrying about).
func TestMedianOfMeans_CleanMatchesMean(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	xs := make([]float64, 5000)
	for i := range xs {
		xs[i] = 3.0 + 2.0*rng.NormFloat64()
	}
	mom, _ := MedianOfMeans(xs, 15)
	if math.Abs(mom-sampleMean(xs)) > 0.15 {
		t.Fatalf("MoM=%.4f far from sample mean %.4f on clean data", mom, sampleMean(xs))
	}
}

// TestMedianOfMeans_SingleBlockIsSampleMean: blocks=1 degenerates to the mean.
func TestMedianOfMeans_SingleBlockIsSampleMean(t *testing.T) {
	xs := []float64{1, 2, 3, 4, 100}
	mom, _ := MedianOfMeans(xs, 1)
	if math.Abs(mom-sampleMean(xs)) > 1e-12 {
		t.Fatalf("blocks=1 MoM=%.6f != sample mean %.6f", mom, sampleMean(xs))
	}
}

func TestMedianOfMeans_Validation(t *testing.T) {
	if _, err := MedianOfMeans(nil, 1); err == nil {
		t.Error("empty input: expected error")
	}
	if _, err := MedianOfMeans([]float64{1, 2, 3}, 0); err == nil {
		t.Error("blocks=0: expected error")
	}
	if _, err := MedianOfMeans([]float64{1, 2, 3}, 4); err == nil {
		t.Error("blocks>n: expected error")
	}
}

func TestMedianOfMeansForConfidence_PicksBlocks(t *testing.T) {
	xs := make([]float64, 1000)
	for i := range xs {
		xs[i] = float64(i)
	}
	_, k, err := MedianOfMeansForConfidence(xs, 0.01)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	want := int(math.Ceil(8 * math.Log(1/0.01))) // ~37
	if k != want {
		t.Fatalf("blocks=%d, want %d", k, want)
	}
	if _, _, err := MedianOfMeansForConfidence(xs, 0); err == nil {
		t.Error("delta=0: expected error")
	}
}

// TestCatoniMean_RobustToOutliers: Catoni's honest claim is to be dramatically
// better than the sample mean under contamination — not a tight absolute bound
// under adversarial gross corruption (that is MoM's breakdown guarantee). So we
// assert Catoni stays in a sane band AND is far closer to the truth than the
// sample mean (which gross outliers wreck).
func TestCatoniMean_RobustToOutliers(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	const trueMean = -2.0
	xs := make([]float64, 2000)
	for i := range xs {
		xs[i] = trueMean + rng.NormFloat64()
	}
	for i := 0; i < 40; i++ { // 2% gross contamination
		xs[rng.Intn(len(xs))] += 5000
	}
	cat, err := CatoniMean(xs, 1.0)
	if err != nil {
		t.Fatalf("CatoniMean: %v", err)
	}
	smErr := math.Abs(sampleMean(xs) - trueMean) // ~100: the mean is wrecked
	catErr := math.Abs(cat - trueMean)
	if catErr > 1.0 {
		t.Fatalf("Catoni=%.4f outside sane band around true mean %.1f", cat, trueMean)
	}
	if catErr > 0.05*smErr {
		t.Fatalf("Catoni error %.4f not dramatically better than sample-mean error %.4f", catErr, smErr)
	}
}

func TestCatoniMean_CleanMatchesMean(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	xs := make([]float64, 4000)
	for i := range xs {
		xs[i] = 10.0 + rng.NormFloat64()
	}
	cat, _ := CatoniMean(xs, 1.0)
	if math.Abs(cat-sampleMean(xs)) > 0.1 {
		t.Fatalf("Catoni=%.4f far from sample mean %.4f on clean data", cat, sampleMean(xs))
	}
}

func TestCatoniMean_Validation(t *testing.T) {
	if _, err := CatoniMean(nil, 1); err == nil {
		t.Error("empty: expected error")
	}
	for _, s := range []float64{0, -1, math.Inf(1), math.NaN()} {
		if _, err := CatoniMean([]float64{1, 2, 3}, s); err == nil {
			t.Errorf("scale=%v: expected error", s)
		}
	}
	// All-equal data returns that value.
	if v, err := CatoniMean([]float64{7, 7, 7}, 1); err != nil || v != 7 {
		t.Fatalf("all-equal: got %v,%v want 7,nil", v, err)
	}
}
