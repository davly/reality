package changepoint

import (
	"math"
	"math/rand"
	"testing"
)

// =========================================================================
// Expansion coverage for BOCPD. Pre-this: 11 tests.
// Adds: prior validation edges, default-config sanity, posterior moment
// boundary behaviour, helper edge cases, error-path state preservation,
// pure-stationary long-run convergence.
// =========================================================================

// --- DefaultNigPrior + DefaultConfig ------------------------------------

func TestDefaultNigPrior_Values(t *testing.T) {
	p := DefaultNigPrior()
	if p.Mu0 != 0.0 {
		t.Errorf("Mu0=%v, want 0.0", p.Mu0)
	}
	if p.Kappa0 != 1.0 {
		t.Errorf("Kappa0=%v, want 1.0", p.Kappa0)
	}
	if p.Alpha0 != 1.0 {
		t.Errorf("Alpha0=%v, want 1.0", p.Alpha0)
	}
	if p.Beta0 != 1.0 {
		t.Errorf("Beta0=%v, want 1.0", p.Beta0)
	}
	if err := p.Validate(); err != nil {
		t.Errorf("Validate(): %v, want nil for default prior", err)
	}
}

func TestDefaultConfig_PassesValidation(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.RMax != DefaultRMax {
		t.Errorf("RMax=%d, want DefaultRMax=%d", cfg.RMax, DefaultRMax)
	}
	if cfg.Lambda != DefaultLambda {
		t.Errorf("Lambda=%v, want DefaultLambda=%v", cfg.Lambda, DefaultLambda)
	}
	if _, err := New(cfg); err != nil {
		t.Errorf("DefaultConfig should construct cleanly: %v", err)
	}
}

// --- NigPrior.Validate boundary cases -----------------------------------

func TestNigPrior_Validate_RejectsPosInfMu0(t *testing.T) {
	p := DefaultNigPrior()
	p.Mu0 = math.Inf(1)
	if err := p.Validate(); err == nil {
		t.Error("+Inf Mu0 must error")
	}
}

func TestNigPrior_Validate_RejectsNegInfMu0(t *testing.T) {
	p := DefaultNigPrior()
	p.Mu0 = math.Inf(-1)
	if err := p.Validate(); err == nil {
		t.Error("-Inf Mu0 must error")
	}
}

func TestNigPrior_Validate_RejectsNegativeBeta0(t *testing.T) {
	p := DefaultNigPrior()
	p.Beta0 = -1.0
	if err := p.Validate(); err == nil {
		t.Error("negative Beta0 must error")
	}
}

func TestNigPrior_Validate_RejectsZeroAlpha0(t *testing.T) {
	p := DefaultNigPrior()
	p.Alpha0 = 0.0
	if err := p.Validate(); err == nil {
		t.Error("zero Alpha0 must error")
	}
}

func TestNigPrior_Validate_RejectsNanKappa0(t *testing.T) {
	p := DefaultNigPrior()
	p.Kappa0 = math.NaN()
	if err := p.Validate(); err == nil {
		t.Error("NaN Kappa0 must error")
	}
}

// --- New() additional reject cases --------------------------------------

func TestNew_RejectsZeroLambda(t *testing.T) {
	if _, err := New(Config{Prior: DefaultNigPrior(), RMax: 100, Lambda: 0}); err == nil {
		t.Error("Lambda=0 must error")
	}
}

func TestNew_RejectsInfLambda(t *testing.T) {
	if _, err := New(Config{Prior: DefaultNigPrior(), RMax: 100, Lambda: math.Inf(1)}); err == nil {
		t.Error("Lambda=Inf must error")
	}
}

func TestNew_RejectsNegativeRMax(t *testing.T) {
	if _, err := New(Config{Prior: DefaultNigPrior(), RMax: -5, Lambda: 100}); err == nil {
		t.Error("Negative RMax must error")
	}
}

func TestNew_AcceptsRMaxOne(t *testing.T) {
	// RMax = 1 is the smallest valid choice (run-length in {0, 1}).
	b, err := New(Config{Prior: DefaultNigPrior(), RMax: 1, Lambda: 250})
	if err != nil {
		t.Fatalf("RMax=1 should be accepted: %v", err)
	}
	if b.Step() != 0 {
		t.Errorf("Step()=%d, want 0", b.Step())
	}
}

// --- Initial-state moment queries ---------------------------------------

func TestInitial_ExpectedRunLength_IsZero(t *testing.T) {
	b, _ := New(DefaultConfig())
	if e := b.ExpectedRunLength(); e != 0 {
		t.Errorf("ExpectedRunLength()=%v, want 0 at t=0", e)
	}
}

func TestInitial_MapRunLength_IsZero(t *testing.T) {
	b, _ := New(DefaultConfig())
	if r := b.MapRunLength(); r != 0 {
		t.Errorf("MapRunLength()=%d, want 0 at t=0", r)
	}
}

func TestInitial_CurrentRegimeMean_IsPriorMu0(t *testing.T) {
	prior := NigPrior{Mu0: 7.5, Kappa0: 1, Alpha0: 1, Beta0: 1}
	b, err := New(Config{Prior: prior, RMax: 100, Lambda: 250})
	if err != nil {
		t.Fatal(err)
	}
	if m := b.CurrentRegimeMean(); m != 7.5 {
		t.Errorf("CurrentRegimeMean()=%v, want 7.5 at t=0", m)
	}
}

func TestInitial_CurrentRegimeVariance_AlphaGreaterOne(t *testing.T) {
	// E[sigma^2] = beta / (alpha-1) when alpha > 1.
	prior := NigPrior{Mu0: 0, Kappa0: 1, Alpha0: 3, Beta0: 4}
	b, err := New(Config{Prior: prior, RMax: 100, Lambda: 250})
	if err != nil {
		t.Fatal(err)
	}
	want := 4.0 / (3.0 - 1.0)
	if v := b.CurrentRegimeVariance(); math.Abs(v-want) > 1e-12 {
		t.Errorf("CurrentRegimeVariance()=%v, want %v", v, want)
	}
}

// --- ChangePointProbabilityWithin edge cases ----------------------------

func TestCpWithin_WindowZero_ReturnsZero(t *testing.T) {
	b, _ := New(DefaultConfig())
	if cp := b.ChangePointProbabilityWithin(0); cp != 0 {
		t.Errorf("window=0 should return 0, got %v", cp)
	}
}

func TestCpWithin_NegativeWindow_ReturnsZero(t *testing.T) {
	b, _ := New(DefaultConfig())
	if cp := b.ChangePointProbabilityWithin(-3); cp != 0 {
		t.Errorf("negative window should return 0, got %v", cp)
	}
}

func TestCpWithin_LargeWindow_Saturates(t *testing.T) {
	b, _ := New(DefaultConfig())
	_, _ = b.Update(0.5)
	_, _ = b.Update(0.6)
	if cp := b.ChangePointProbabilityWithin(1_000_000); cp < 0.999 {
		t.Errorf("oversized window should sum to ~1, got %v", cp)
	}
}

// --- Update error path ---------------------------------------------------

func TestUpdate_NaN_StatePreserved(t *testing.T) {
	// Failing observation must not corrupt the running state.
	b, _ := New(DefaultConfig())
	_, _ = b.Update(0.1)
	priorStep := b.Step()
	priorMap := b.MapRunLength()

	if _, err := b.Update(math.NaN()); err == nil {
		t.Fatal("NaN should error")
	}
	if b.Step() != priorStep {
		t.Errorf("Step changed after failed update: %d vs %d", b.Step(), priorStep)
	}
	if b.MapRunLength() != priorMap {
		t.Errorf("MAP run-length changed after failed update")
	}
}

// --- Posterior is still a probability (non-negative + bounded) ----------

func TestPosterior_AllEntriesInUnitInterval_ManySteps(t *testing.T) {
	b, _ := New(DefaultConfig())
	rng := rand.New(rand.NewSource(99))
	for i := 0; i < 200; i++ {
		p, _ := b.Update(rng.NormFloat64())
		for r, v := range p {
			if v < 0 || v > 1 {
				t.Errorf("step %d r=%d: p=%v out of [0,1]", i, r, v)
			}
		}
	}
}

// --- Stationary long-run: ExpectedRunLength climbs --------------------

func TestExpectedRunLength_GrowsUnderStationarity(t *testing.T) {
	cfg := Config{Prior: DefaultNigPrior(), RMax: 500, Lambda: 1e8}
	b, _ := New(cfg)
	rng := rand.New(rand.NewSource(11))
	prev := b.ExpectedRunLength()
	for i := 0; i < 50; i++ {
		_, _ = b.Update(rng.NormFloat64())
	}
	mid := b.ExpectedRunLength()
	for i := 0; i < 50; i++ {
		_, _ = b.Update(rng.NormFloat64())
	}
	end := b.ExpectedRunLength()
	if !(mid > prev && end > mid) {
		t.Errorf("expected run-length should grow monotonically: %v -> %v -> %v",
			prev, mid, end)
	}
}

// --- studentTLogPDF golden values ---------------------------------------

func TestStudentTLogPDF_StandardCauchy_AtZero(t *testing.T) {
	// Cauchy = Student's-t with df=1. PDF at 0 is 1/(pi * scale).
	// log PDF = -log(pi) for scale=1.
	got := studentTLogPDF(0.0, 1.0, 0.0, 1.0)
	want := -math.Log(math.Pi)
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("Cauchy log-PDF at 0: got %v, want %v", got, want)
	}
}

func TestStudentTLogPDF_LocationShift_IsTranslationInvariant(t *testing.T) {
	a := studentTLogPDF(2.5, 4.0, 0.0, 1.0)
	b := studentTLogPDF(7.5, 4.0, 5.0, 1.0)
	if math.Abs(a-b) > 1e-12 {
		t.Errorf("translation-invariance violated: %v vs %v", a, b)
	}
}

func TestStudentTLogPDF_Symmetric_AroundLocation(t *testing.T) {
	a := studentTLogPDF(3.0, 5.0, 0.0, 1.0)
	b := studentTLogPDF(-3.0, 5.0, 0.0, 1.0)
	if math.Abs(a-b) > 1e-12 {
		t.Errorf("symmetry violated: %v vs %v", a, b)
	}
}

// --- logSumExp edge cases beyond the existing five-case table ----------

func TestLogSumExp_BothNegInf_ReturnsNegInf(t *testing.T) {
	got := logSumExp(math.Inf(-1), math.Inf(-1))
	if !math.IsInf(got, -1) {
		t.Errorf("logSumExp(-Inf,-Inf) = %v, want -Inf", got)
	}
}

func TestLogSumExp_Symmetric(t *testing.T) {
	a, b := -3.0, 7.0
	if logSumExp(a, b) != logSumExp(b, a) {
		t.Errorf("logSumExp not symmetric: %v vs %v", logSumExp(a, b), logSumExp(b, a))
	}
}

// --- Truncation interaction with RMax=1 (smallest valid) ---------------

func TestTruncation_RMaxOne_PosteriorNeverExceedsLengthTwo(t *testing.T) {
	// With RMax=1, posterior length is at most 2 (run-length in {0, 1}).
	cfg := Config{Prior: DefaultNigPrior(), RMax: 1, Lambda: 1e6}
	b, _ := New(cfg)
	for i := 0; i < 50; i++ {
		p, _ := b.Update(0.0)
		if len(p) > 2 {
			t.Errorf("step %d: len(P)=%d, want <= 2", i, len(p))
		}
	}
}

// --- Mid-stream large hazard: forced-reset bias ------------------------

func TestSmallLambda_BiasesTowardLowerRunLength_VsLargeLambda(t *testing.T) {
	// Same data, different lambda → smaller lambda must produce smaller
	// expected run-length. This is a relative comparison, not an absolute
	// threshold (predictive likelihood for stable signals can still pull
	// mass to higher r even at small lambda).
	xs := make([]float64, 30)
	for i := range xs {
		xs[i] = 0.0
	}

	bSmall, _ := New(Config{Prior: DefaultNigPrior(), RMax: 200, Lambda: 2.0})
	bLarge, _ := New(Config{Prior: DefaultNigPrior(), RMax: 200, Lambda: 1e8})
	for _, x := range xs {
		_, _ = bSmall.Update(x)
		_, _ = bLarge.Update(x)
	}

	eSmall := bSmall.ExpectedRunLength()
	eLarge := bLarge.ExpectedRunLength()
	if !(eSmall < eLarge) {
		t.Errorf("smaller lambda should give smaller expected run-length: small=%v large=%v",
			eSmall, eLarge)
	}
}
