package control

import (
	"math"
	"math/cmplx"
	"testing"
)

// ═══════════════════════════════════════════════════════════════════════════
// PID — additional edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestPID_DerivativeOnlyFirstStep(t *testing.T) {
	// D-only controller, first step: prevError=0, error=5, dt=0.1
	pid := NewPID(0, 0, 2.0, -100, 100)
	out := pid.Update(5.0, 0.0, 0.1)
	// D = Kd * (5-0) / 0.1 = 2 * 50 = 100
	assertClose(t, "D-only first step", out, 100.0, 1e-12)
}

func TestPID_DerivativeSecondStep(t *testing.T) {
	pid := NewPID(0, 0, 2.0, -200, 200)
	pid.Update(5.0, 0.0, 0.1) // error=5, prevError=0
	out := pid.Update(5.0, 2.0, 0.1)
	// error=3, prevError=5, D = 2 * (3-5)/0.1 = 2 * (-20) = -40
	assertClose(t, "D-only second step", out, -40.0, 1e-12)
}

func TestPID_IntegralOnlyAccumulates(t *testing.T) {
	pid := NewPID(0, 2.0, 0, -100, 100)
	dt := 0.1
	// Step 1: error=10, integral += 10*0.1 = 1, output = 2*1 = 2
	out1 := pid.Update(10.0, 0.0, dt)
	assertClose(t, "I step 1", out1, 2.0, 1e-12)
	// Step 2: error=10 again, integral += 10*0.1 = 2, output = 2*2 = 4
	out2 := pid.Update(10.0, 0.0, dt)
	assertClose(t, "I step 2", out2, 4.0, 1e-12)
}

func TestPID_LargeDt(t *testing.T) {
	pid := NewPID(1.0, 0.5, 0.1, -1000, 1000)
	out := pid.Update(10.0, 0.0, 100.0)
	// P = 10, I = 0.5 * (10*100) = 500, D = 0.1 * (10/100) = 0.01
	expected := 10.0 + 500.0 + 0.01
	assertClose(t, "large dt", out, expected, 1e-10)
}

func TestPID_MultipleResets(t *testing.T) {
	pid := NewPID(1.0, 1.0, 1.0, -100, 100)
	pid.Update(10.0, 0.0, 0.1)
	pid.Reset()
	pid.Update(5.0, 0.0, 0.1)
	pid.Reset()
	out := pid.Update(1.0, 0.0, 0.1)
	fresh := NewPID(1.0, 1.0, 1.0, -100, 100)
	outFresh := fresh.Update(1.0, 0.0, 0.1)
	assertClose(t, "multi reset", out, outFresh, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// LowPassFilter — additional edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestLowPass_ExactMiddle(t *testing.T) {
	out := LowPassFilter(0.0, 100.0, 0.5)
	assertClose(t, "lowpass 50%", out, 50.0, 1e-15)
}

func TestLowPass_NegativeValues(t *testing.T) {
	out := LowPassFilter(-10.0, -20.0, 0.3)
	expected := 0.3*(-20.0) + 0.7*(-10.0)
	assertClose(t, "lowpass negative", out, expected, 1e-15)
}

func TestLowPass_IdenticalValues(t *testing.T) {
	out := LowPassFilter(5.0, 5.0, 0.7)
	assertClose(t, "lowpass identical", out, 5.0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// HighPassFilter — additional edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestHighPass_AlphaOne(t *testing.T) {
	out := HighPassFilter(3.0, 5.0, 8.0, 1.0)
	// 1.0 * (3 + 8 - 5) = 6
	assertClose(t, "highpass alpha=1", out, 6.0, 1e-15)
}

func TestHighPass_LargeChange(t *testing.T) {
	out := HighPassFilter(0.0, 0.0, 100.0, 0.9)
	// 0.9 * (0 + 100 - 0) = 90
	assertClose(t, "highpass large change", out, 90.0, 1e-15)
}

func TestHighPass_NegativeChange(t *testing.T) {
	out := HighPassFilter(0.0, 10.0, 5.0, 0.9)
	// 0.9 * (0 + 5 - 10) = 0.9 * (-5) = -4.5
	assertClose(t, "highpass negative change", out, -4.5, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// ComplementaryFilter — additional edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestComplementary_AlphaHalf(t *testing.T) {
	out := ComplementaryFilter(30.0, 10.0, 0.5, 0.01)
	// 0.5*(30+10*0.01) + 0.5*30 = 0.5*30.1 + 15 = 15.05 + 15 = 30.05
	assertClose(t, "complementary alpha=0.5", out, 30.05, 1e-12)
}

func TestComplementary_AlphaClampedBelow(t *testing.T) {
	out := ComplementaryFilter(45.0, 10.0, -0.5, 0.01)
	// Clamped to 0: returns accel = 45
	assertClose(t, "complementary alpha clamped below", out, 45.0, 1e-12)
}

func TestComplementary_AlphaClampedAbove(t *testing.T) {
	out := ComplementaryFilter(45.0, 10.0, 1.5, 0.01)
	// Clamped to 1: 1*(45+10*0.01)+0*45 = 45.1
	assertClose(t, "complementary alpha clamped above", out, 45.1, 1e-12)
}

// ═══════════════════════════════════════════════════════════════════════════
// RateLimiter — additional edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestRateLimiter_AlreadyAtTarget(t *testing.T) {
	out := RateLimiter(5.0, 5.0, 10.0, 0.1)
	assertClose(t, "already at target", out, 5.0, 1e-15)
}

func TestRateLimiter_NegativeMaxRate(t *testing.T) {
	out := RateLimiter(5.0, 100.0, -1.0, 0.1)
	assertClose(t, "negative maxRate holds current", out, 5.0, 1e-15)
}

func TestRateLimiter_VerySmallDt(t *testing.T) {
	out := RateLimiter(0.0, 100.0, 10.0, 0.0001)
	// maxDelta = 10 * 0.0001 = 0.001
	assertClose(t, "tiny dt", out, 0.001, 1e-15)
}

func TestRateLimiter_NegativeDirection(t *testing.T) {
	out := RateLimiter(10.0, -10.0, 5.0, 0.1)
	// delta = -20, maxDelta = 0.5, return 10 - 0.5 = 9.5
	assertClose(t, "negative direction", out, 9.5, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// TransferFunction — additional edge cases
// ═══════════════════════════════════════════════════════════════════════════

func TestTransfer_Evaluate_AtPole(t *testing.T) {
	// H(s) = 1/(s+1), at s=-1 (pole)
	tf := TransferFunction{
		Numerator:   []float64{1},
		Denominator: []float64{1, 1},
	}
	got := tf.Evaluate(complex(-1, 0))
	if !cmplx.IsInf(got) {
		t.Errorf("expected infinity at pole, got %v", got)
	}
}

func TestTransfer_Evaluate_HighFrequency(t *testing.T) {
	// H(s) = 1/(s+1), at s=j*1000 → magnitude → 0
	tf := TransferFunction{
		Numerator:   []float64{1},
		Denominator: []float64{1, 1},
	}
	got := tf.Evaluate(complex(0, 1000))
	mag := cmplx.Abs(got)
	if mag > 0.01 {
		t.Errorf("expected near-zero magnitude at high frequency, got %v", mag)
	}
}

func TestTransfer_Evaluate_ConstantTF(t *testing.T) {
	// H(s) = 3/5 for all s
	tf := TransferFunction{
		Numerator:   []float64{3},
		Denominator: []float64{5},
	}
	got := tf.Evaluate(complex(100, 200))
	assertComplexClose(t, "constant TF", got, complex(0.6, 0), 1e-12)
}

func TestTransfer_Poles_ScaledDenominator(t *testing.T) {
	// 2s + 4 = 0 → s = -2 (leading coeff 2, normalized to s+2)
	tf := TransferFunction{
		Numerator:   []float64{1},
		Denominator: []float64{2, 4},
	}
	poles := tf.Poles()
	if len(poles) != 1 {
		t.Fatalf("expected 1 pole, got %d", len(poles))
	}
	assertComplexClose(t, "scaled pole", poles[0], complex(-2, 0), 1e-12)
}

func TestTransfer_IsStable_ThirdOrderStable(t *testing.T) {
	// (s+1)(s+2)(s+3) = s^3 + 6s^2 + 11s + 6 — all poles negative
	tf := TransferFunction{
		Numerator:   []float64{1},
		Denominator: []float64{1, 6, 11, 6},
	}
	if !tf.IsStable() {
		t.Error("(s+1)(s+2)(s+3) should be stable")
	}
}

func TestTransfer_IsStable_ThirdOrderUnstable(t *testing.T) {
	// (s-1)(s+2)(s+3) = s^3 + 4s^2 + s - 6 — one positive pole
	tf := TransferFunction{
		Numerator:   []float64{1},
		Denominator: []float64{1, 4, 1, -6},
	}
	if tf.IsStable() {
		t.Error("(s-1)(s+2)(s+3) should be unstable")
	}
}

func TestTransfer_GainAtDC(t *testing.T) {
	// H(s) = 5/(s+5), DC gain H(0) = 5/5 = 1
	tf := TransferFunction{
		Numerator:   []float64{5},
		Denominator: []float64{1, 5},
	}
	got := tf.Evaluate(complex(0, 0))
	assertComplexClose(t, "DC gain", got, complex(1, 0), 1e-12)
}

// Suppress unused import
var _ = math.Abs
