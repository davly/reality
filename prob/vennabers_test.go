package prob

import (
	"math"
	"math/rand"
	"testing"
)

// meanAbs returns the mean |a-b| over paired slices.
func meanAbsErr(pred, truth []float64) float64 {
	s := 0.0
	for i := range pred {
		s += math.Abs(pred[i] - truth[i])
	}
	return s / float64(len(pred))
}

// TestVennAbers_Recalibrates is the core value claim: given MIScalibrated base
// scores, the Venn-Abers point estimate recovers the true calibrated probability,
// beating the raw score used as a probability. Setup: x~U[0,1], y~Bernoulli(x),
// but the observed score is x^2 (systematically too low). The calibrated
// P(Y=1|score=s) is sqrt(s)=x, which VA should recover. This is discriminating —
// a predictor that did NOT recalibrate (returned ~score) would fail.
func TestVennAbers_Recalibrates(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const nCal, nTest = 5000, 1000
	cs := make([]float64, nCal)
	cl := make([]float64, nCal)
	for i := range cs {
		x := rng.Float64()
		cs[i] = x * x // miscalibrated score
		if rng.Float64() < x {
			cl[i] = 1
		}
	}
	va, err := NewVennAbers(cs, cl)
	if err != nil {
		t.Fatalf("NewVennAbers: %v", err)
	}
	vaPred := make([]float64, nTest)
	rawPred := make([]float64, nTest)
	truth := make([]float64, nTest)
	for i := 0; i < nTest; i++ {
		x := rng.Float64()
		s := x * x
		vaPred[i] = va.PredictPoint(s)
		rawPred[i] = s // using the raw (miscalibrated) score as a probability
		truth[i] = x   // true P(Y=1|score=s) = sqrt(s) = x
	}
	vaErr := meanAbsErr(vaPred, truth)
	rawErr := meanAbsErr(rawPred, truth)
	if vaErr > 0.05 {
		t.Errorf("Venn-Abers not close to truth: mean abs err %.4f", vaErr)
	}
	if vaErr >= rawErr {
		t.Errorf("Venn-Abers (%.4f) did not beat the raw miscalibrated score (%.4f)", vaErr, rawErr)
	}
}

// TestVennAbers_MultiprobabilityOrdered: p0 <= p1 for every test score, and both
// lie in [0,1].
func TestVennAbers_MultiprobabilityOrdered(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	cs := make([]float64, 500)
	cl := make([]float64, 500)
	for i := range cs {
		cs[i] = rng.Float64()
		if rng.Float64() < cs[i] {
			cl[i] = 1
		}
	}
	va, _ := NewVennAbers(cs, cl)
	for i := 0; i < 200; i++ {
		s := rng.Float64()
		p0, p1 := va.Predict(s)
		if !(p0 >= -1e-12 && p0 <= p1+1e-12 && p1 <= 1+1e-12) {
			t.Fatalf("multiprobability not ordered/in-range at s=%.3f: [%.4f, %.4f]", s, p0, p1)
		}
	}
}

// TestVennAbers_AlreadyCalibratedIsStable: when the score is already calibrated
// (score=x, y~Bernoulli(x)), the VA estimate stays close to the score (it does
// not distort a good calibration).
func TestVennAbers_AlreadyCalibratedIsStable(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	const n = 5000
	cs := make([]float64, n)
	cl := make([]float64, n)
	for i := range cs {
		x := rng.Float64()
		cs[i] = x
		if rng.Float64() < x {
			cl[i] = 1
		}
	}
	va, _ := NewVennAbers(cs, cl)
	maxDev := 0.0
	for i := 0; i < 500; i++ {
		s := rng.Float64()
		if d := math.Abs(va.PredictPoint(s) - s); d > maxDev {
			maxDev = d
		}
	}
	if maxDev > 0.1 {
		t.Errorf("VA distorts an already-calibrated score: max deviation %.4f", maxDev)
	}
}

// TestVennAbers_EmpiricalCalibration: binning VA predictions, the empirical
// outcome rate per bin tracks the predicted probability.
func TestVennAbers_EmpiricalCalibration(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	cs := make([]float64, 5000)
	cl := make([]float64, 5000)
	for i := range cs {
		x := rng.Float64()
		cs[i] = x * x
		if rng.Float64() < x {
			cl[i] = 1
		}
	}
	va, _ := NewVennAbers(cs, cl)
	const bins = 10
	sum := make([]float64, bins)
	hit := make([]float64, bins)
	cnt := make([]int, bins)
	for i := 0; i < 6000; i++ {
		x := rng.Float64()
		s := x * x
		p := va.PredictPoint(s)
		b := int(p * bins)
		if b >= bins {
			b = bins - 1
		}
		sum[b] += p
		if rng.Float64() < x {
			hit[b]++
		}
		cnt[b]++
	}
	for b := 0; b < bins; b++ {
		if cnt[b] < 50 {
			continue
		}
		predRate := sum[b] / float64(cnt[b])
		obsRate := hit[b] / float64(cnt[b])
		if math.Abs(predRate-obsRate) > 0.06 {
			t.Errorf("bin %d miscalibrated: predicted %.3f vs observed %.3f (n=%d)", b, predRate, obsRate, cnt[b])
		}
	}
}

func TestVennAbers_Validation(t *testing.T) {
	if _, err := NewVennAbers(nil, nil); err == nil {
		t.Error("empty: expected error")
	}
	if _, err := NewVennAbers([]float64{0.1, 0.2}, []float64{1}); err == nil {
		t.Error("length mismatch: expected error")
	}
	if _, err := NewVennAbers([]float64{0.1}, []float64{0.5}); err == nil {
		t.Error("non-binary label: expected error")
	}
}
