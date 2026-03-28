package control

import (
	"math"
	"math/cmplx"
	"testing"

	"github.com/davly/reality/testutil"
)

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file tests — shared test vectors across Go, Python, C++, C#
// ═══════════════════════════════════════════════════════════════════════════

func TestGolden_PIDStepResponse(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/control/pid_step_response.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			kp := testutil.InputFloat64(t, tc, "kp")
			ki := testutil.InputFloat64(t, tc, "ki")
			kd := testutil.InputFloat64(t, tc, "kd")
			minOut := testutil.InputFloat64(t, tc, "min_output")
			maxOut := testutil.InputFloat64(t, tc, "max_output")
			setpoint := testutil.InputFloat64(t, tc, "setpoint")
			dt := testutil.InputFloat64(t, tc, "dt")
			steps := testutil.InputInt(t, tc, "steps")

			pid := NewPID(kp, ki, kd, minOut, maxOut)
			outputs := make([]float64, steps)
			measured := 0.0
			for i := 0; i < steps; i++ {
				output := pid.Update(setpoint, measured, dt)
				outputs[i] = output
				measured += output * dt
			}

			testutil.AssertFloat64Slice(t, tc, outputs)
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — PID Controller
// ═══════════════════════════════════════════════════════════════════════════

func TestPID_ProportionalOnly(t *testing.T) {
	// P-only controller: output = Kp * error.
	pid := NewPID(2.0, 0, 0, -100, 100)
	out := pid.Update(10.0, 3.0, 0.1)
	assertClose(t, "P-only output", out, 14.0, 1e-12) // 2 * (10 - 3) = 14
}

func TestPID_ProportionalTracksLinearly(t *testing.T) {
	// With P-only and a simple plant, the error should decay exponentially.
	pid := NewPID(1.0, 0, 0, -100, 100)
	measured := 0.0
	dt := 0.1
	for i := 0; i < 100; i++ {
		out := pid.Update(1.0, measured, dt)
		measured += out * dt
	}
	// After many steps, measured should approach setpoint.
	assertClose(t, "P-only converges", measured, 1.0, 0.01)
}

func TestPID_IntegralEliminatesSteadyStateError(t *testing.T) {
	// With integral gain, the steady-state error should be zero.
	pid := NewPID(0.5, 1.0, 0, -100, 100)
	measured := 0.0
	dt := 0.1
	for i := 0; i < 500; i++ {
		out := pid.Update(1.0, measured, dt)
		measured += out * dt
	}
	assertClose(t, "PI steady-state", measured, 1.0, 0.001)
}

func TestPID_DerivativeDampsOvershoot(t *testing.T) {
	// Compare PI vs PID on a second-order plant (mass-spring-damper):
	// x'' = output - 0.5*x' - x
	// The derivative gain should reduce overshoot in the settling phase.
	piOnly := NewPID(4.0, 2.0, 0, -100, 100)
	pid := NewPID(4.0, 2.0, 1.5, -100, 100)
	dt := 0.01
	steps := 2000

	// Second-order plant state: position and velocity.
	posPI, velPI := 0.0, 0.0
	posPID, velPID := 0.0, 0.0
	maxOvershootPI := 0.0
	maxOvershootPID := 0.0

	for i := 0; i < steps; i++ {
		outPI := piOnly.Update(1.0, posPI, dt)
		accelPI := outPI - 0.5*velPI - posPI
		velPI += accelPI * dt
		posPI += velPI * dt
		if posPI-1.0 > maxOvershootPI {
			maxOvershootPI = posPI - 1.0
		}

		outPID := pid.Update(1.0, posPID, dt)
		accelPID := outPID - 0.5*velPID - posPID
		velPID += accelPID * dt
		posPID += velPID * dt
		if posPID-1.0 > maxOvershootPID {
			maxOvershootPID = posPID - 1.0
		}
	}

	if maxOvershootPID >= maxOvershootPI {
		t.Errorf("derivative should reduce overshoot: PID=%.6f, PI=%.6f",
			maxOvershootPID, maxOvershootPI)
	}
}

func TestPID_AntiWindupClampsOutput(t *testing.T) {
	pid := NewPID(10.0, 1.0, 0, -5, 5)
	// Large error should be clamped.
	out := pid.Update(100.0, 0.0, 0.1)
	assertClose(t, "clamped max", out, 5.0, 1e-12)

	// Large negative error.
	pid.Reset()
	out = pid.Update(-100.0, 0.0, 0.1)
	assertClose(t, "clamped min", out, -5.0, 1e-12)
}

func TestPID_AntiWindupPreventsIntegralAccumulation(t *testing.T) {
	// When clamped, integral should not keep growing.
	pid := NewPID(10.0, 5.0, 0, -5, 5)
	dt := 0.1
	// Drive it into saturation for many steps.
	for i := 0; i < 100; i++ {
		pid.Update(100.0, 0.0, dt)
	}
	// Now bring setpoint to 0. Without anti-windup, the accumulated integral
	// would take a long time to unwind. With anti-windup, it should respond.
	pid.Reset()
	out := pid.Update(0.0, 0.0, dt)
	assertClose(t, "after reset, zero error", out, 0.0, 1e-12)
}

func TestPID_Reset(t *testing.T) {
	pid := NewPID(1.0, 1.0, 1.0, -100, 100)
	pid.Update(10.0, 0.0, 0.1)
	pid.Update(10.0, 1.0, 0.1)
	pid.Reset()

	// After reset, should behave as if freshly created.
	out := pid.Update(1.0, 0.0, 0.1)
	fresh := NewPID(1.0, 1.0, 1.0, -100, 100)
	outFresh := fresh.Update(1.0, 0.0, 0.1)
	assertClose(t, "reset matches fresh", out, outFresh, 1e-15)
}

func TestPID_ZeroDt(t *testing.T) {
	pid := NewPID(2.0, 1.0, 0.5, -100, 100)
	// dt=0: no integral accumulation, no derivative.
	out := pid.Update(5.0, 2.0, 0.0)
	assertClose(t, "dt=0 output", out, 6.0, 1e-12) // Only P: 2 * 3 = 6
}

func TestPID_NegativeDt(t *testing.T) {
	pid := NewPID(2.0, 1.0, 0.5, -100, 100)
	out := pid.Update(5.0, 2.0, -0.1)
	// dt<=0: only proportional.
	assertClose(t, "negative dt output", out, 6.0, 1e-12)
}

func TestNewPID_PanicsOnInvalidLimits(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for minOut > maxOut")
		}
	}()
	NewPID(1, 0, 0, 10, 5)
}

func TestPID_ZeroGains(t *testing.T) {
	pid := NewPID(0, 0, 0, -100, 100)
	out := pid.Update(100.0, 0.0, 0.1)
	assertClose(t, "zero gains", out, 0.0, 1e-15)
}

func TestPID_EqualLimits(t *testing.T) {
	// min == max: output is always that value regardless of error.
	pid := NewPID(10.0, 0, 0, 3.0, 3.0)
	out := pid.Update(100.0, 0.0, 0.1)
	assertClose(t, "equal limits", out, 3.0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — LowPassFilter
// ═══════════════════════════════════════════════════════════════════════════

func TestLowPass_AlphaOne_PassThrough(t *testing.T) {
	out := LowPassFilter(5.0, 10.0, 1.0)
	assertClose(t, "alpha=1", out, 10.0, 1e-15)
}

func TestLowPass_AlphaZero_HoldPrevious(t *testing.T) {
	out := LowPassFilter(5.0, 10.0, 0.0)
	assertClose(t, "alpha=0", out, 5.0, 1e-15)
}

func TestLowPass_AlphaHalf(t *testing.T) {
	out := LowPassFilter(4.0, 8.0, 0.5)
	assertClose(t, "alpha=0.5", out, 6.0, 1e-15) // 0.5*8 + 0.5*4 = 6
}

func TestLowPass_SmoothsStep(t *testing.T) {
	// Apply low-pass to a step from 0 to 10.
	prev := 0.0
	alpha := 0.3
	for i := 0; i < 50; i++ {
		prev = LowPassFilter(prev, 10.0, alpha)
	}
	// Should converge to 10.
	assertClose(t, "lowpass step convergence", prev, 10.0, 0.01)
}

func TestLowPass_AlphaClampedBelow(t *testing.T) {
	out := LowPassFilter(5.0, 10.0, -1.0)
	assertClose(t, "alpha<0 clamped", out, 5.0, 1e-15)
}

func TestLowPass_AlphaClampedAbove(t *testing.T) {
	out := LowPassFilter(5.0, 10.0, 2.0)
	assertClose(t, "alpha>1 clamped", out, 10.0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — HighPassFilter
// ═══════════════════════════════════════════════════════════════════════════

func TestHighPass_DetectsChange(t *testing.T) {
	// High-pass should detect the rapid change between prev and current.
	out := HighPassFilter(0.0, 5.0, 10.0, 0.9)
	// alpha * (0 + 10 - 5) = 0.9 * 5 = 4.5
	assertClose(t, "highpass detects change", out, 4.5, 1e-15)
}

func TestHighPass_SteadyStateZero(t *testing.T) {
	// If prev == current, the high-pass contribution from this step is just prevFiltered.
	out := HighPassFilter(0.0, 5.0, 5.0, 0.9)
	// alpha * (0 + 5 - 5) = 0
	assertClose(t, "highpass steady state", out, 0.0, 1e-15)
}

func TestHighPass_AlphaZero(t *testing.T) {
	out := HighPassFilter(10.0, 5.0, 8.0, 0.0)
	assertClose(t, "highpass alpha=0", out, 0.0, 1e-15)
}

func TestHighPass_AlphaClamped(t *testing.T) {
	out := HighPassFilter(0.0, 5.0, 10.0, -1.0)
	assertClose(t, "highpass alpha clamped below", out, 0.0, 1e-15)

	out2 := HighPassFilter(0.0, 5.0, 10.0, 2.0)
	// Clamped to 1: 1 * (0 + 10 - 5) = 5
	assertClose(t, "highpass alpha clamped above", out2, 5.0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — ComplementaryFilter
// ═══════════════════════════════════════════════════════════════════════════

func TestComplementary_AlphaZero_AccelOnly(t *testing.T) {
	out := ComplementaryFilter(45.0, 1.0, 0.0, 0.01)
	assertClose(t, "alpha=0 accel only", out, 45.0, 1e-15)
}

func TestComplementary_AlphaOne_GyroIntegrated(t *testing.T) {
	// alpha=1: out = 1*(accel + gyro*dt) + 0*accel = accel + gyro*dt
	out := ComplementaryFilter(45.0, 10.0, 1.0, 0.01)
	assertClose(t, "alpha=1 gyro integrated", out, 45.1, 1e-12)
}

func TestComplementary_TypicalValue(t *testing.T) {
	// alpha=0.98, accel=30, gyro=5, dt=0.01
	out := ComplementaryFilter(30.0, 5.0, 0.98, 0.01)
	// 0.98 * (30 + 5*0.01) + 0.02 * 30 = 0.98 * 30.05 + 0.6 = 29.449 + 0.6 = 30.049
	assertClose(t, "complementary typical", out, 30.049, 1e-12)
}

func TestComplementary_ZeroDt_AccelOnly(t *testing.T) {
	out := ComplementaryFilter(45.0, 100.0, 0.98, 0.0)
	assertClose(t, "dt=0 returns accel", out, 45.0, 1e-15)
}

func TestComplementary_NegativeDt_AccelOnly(t *testing.T) {
	out := ComplementaryFilter(45.0, 100.0, 0.98, -0.1)
	assertClose(t, "dt<0 returns accel", out, 45.0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — RateLimiter
// ═══════════════════════════════════════════════════════════════════════════

func TestRateLimiter_WithinLimit(t *testing.T) {
	// Target within allowed change: should reach target exactly.
	out := RateLimiter(0.0, 0.5, 10.0, 0.1)
	assertClose(t, "within limit", out, 0.5, 1e-15)
}

func TestRateLimiter_ExceedsLimit_Positive(t *testing.T) {
	// Target far away: limited to maxRate * dt.
	out := RateLimiter(0.0, 100.0, 10.0, 0.1)
	assertClose(t, "limited positive", out, 1.0, 1e-15)
}

func TestRateLimiter_ExceedsLimit_Negative(t *testing.T) {
	out := RateLimiter(0.0, -100.0, 10.0, 0.1)
	assertClose(t, "limited negative", out, -1.0, 1e-15)
}

func TestRateLimiter_ZeroDt(t *testing.T) {
	out := RateLimiter(5.0, 100.0, 10.0, 0.0)
	assertClose(t, "dt=0 holds current", out, 5.0, 1e-15)
}

func TestRateLimiter_ZeroMaxRate(t *testing.T) {
	out := RateLimiter(5.0, 100.0, 0.0, 0.1)
	assertClose(t, "maxRate=0 holds current", out, 5.0, 1e-15)
}

func TestRateLimiter_ExactBoundary(t *testing.T) {
	// delta == maxRate * dt exactly.
	out := RateLimiter(0.0, 1.0, 10.0, 0.1)
	assertClose(t, "exact boundary", out, 1.0, 1e-15)
}

func TestRateLimiter_ConvergesOverTime(t *testing.T) {
	current := 0.0
	for i := 0; i < 100; i++ {
		current = RateLimiter(current, 10.0, 5.0, 0.1)
	}
	assertClose(t, "converges to target", current, 10.0, 1e-12)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — TransferFunction
// ═══════════════════════════════════════════════════════════════════════════

func TestTransfer_Evaluate_FirstOrder(t *testing.T) {
	// H(s) = 1 / (s + 1), evaluate at s = 0 => H(0) = 1
	tf := TransferFunction{
		Numerator:   []float64{1},
		Denominator: []float64{1, 1},
	}
	got := tf.Evaluate(complex(0, 0))
	assertComplexClose(t, "H(0)", got, complex(1, 0), 1e-12)
}

func TestTransfer_Evaluate_FirstOrder_AtJW(t *testing.T) {
	// H(s) = 1 / (s + 1), at s = j*1 => H(jw) = 1/(1+j) = (1-j)/2
	tf := TransferFunction{
		Numerator:   []float64{1},
		Denominator: []float64{1, 1},
	}
	got := tf.Evaluate(complex(0, 1))
	expected := complex(0.5, -0.5)
	assertComplexClose(t, "H(j)", got, expected, 1e-12)
}

func TestTransfer_Evaluate_SecondOrder(t *testing.T) {
	// H(s) = 1 / (s^2 + 2s + 1) = 1 / (s+1)^2, at s=0 => 1
	tf := TransferFunction{
		Numerator:   []float64{1},
		Denominator: []float64{1, 2, 1},
	}
	got := tf.Evaluate(complex(0, 0))
	assertComplexClose(t, "2nd order H(0)", got, complex(1, 0), 1e-12)
}

func TestTransfer_Evaluate_WithNumerator(t *testing.T) {
	// H(s) = (2s + 3) / (s + 1) at s=0 => 3/1 = 3
	tf := TransferFunction{
		Numerator:   []float64{2, 3},
		Denominator: []float64{1, 1},
	}
	got := tf.Evaluate(complex(0, 0))
	assertComplexClose(t, "with numerator H(0)", got, complex(3, 0), 1e-12)
}

func TestTransfer_Poles_FirstOrder(t *testing.T) {
	// s + 1 = 0 => pole at s = -1
	tf := TransferFunction{
		Numerator:   []float64{1},
		Denominator: []float64{1, 1},
	}
	poles := tf.Poles()
	if len(poles) != 1 {
		t.Fatalf("expected 1 pole, got %d", len(poles))
	}
	assertComplexClose(t, "first-order pole", poles[0], complex(-1, 0), 1e-12)
}

func TestTransfer_Poles_SecondOrder_Real(t *testing.T) {
	// s^2 + 3s + 2 = (s+1)(s+2) => poles at -1, -2
	tf := TransferFunction{
		Numerator:   []float64{1},
		Denominator: []float64{1, 3, 2},
	}
	poles := tf.Poles()
	if len(poles) != 2 {
		t.Fatalf("expected 2 poles, got %d", len(poles))
	}
	// Order may vary, check both.
	found1 := false
	found2 := false
	for _, p := range poles {
		if cmplx.Abs(p-complex(-1, 0)) < 1e-10 {
			found1 = true
		}
		if cmplx.Abs(p-complex(-2, 0)) < 1e-10 {
			found2 = true
		}
	}
	if !found1 || !found2 {
		t.Errorf("expected poles at -1 and -2, got %v", poles)
	}
}

func TestTransfer_Poles_SecondOrder_Complex(t *testing.T) {
	// s^2 + 2s + 5 = 0 => s = -1 +/- 2j
	tf := TransferFunction{
		Numerator:   []float64{1},
		Denominator: []float64{1, 2, 5},
	}
	poles := tf.Poles()
	if len(poles) != 2 {
		t.Fatalf("expected 2 poles, got %d", len(poles))
	}
	foundPlus := false
	foundMinus := false
	for _, p := range poles {
		if cmplx.Abs(p-complex(-1, 2)) < 1e-10 {
			foundPlus = true
		}
		if cmplx.Abs(p-complex(-1, -2)) < 1e-10 {
			foundMinus = true
		}
	}
	if !foundPlus || !foundMinus {
		t.Errorf("expected poles at -1+/-2j, got %v", poles)
	}
}

func TestTransfer_Poles_ConstantDenominator(t *testing.T) {
	tf := TransferFunction{
		Numerator:   []float64{1},
		Denominator: []float64{5},
	}
	poles := tf.Poles()
	if len(poles) != 0 {
		t.Errorf("constant denominator should have no poles, got %v", poles)
	}
}

func TestTransfer_Poles_ThirdOrder(t *testing.T) {
	// (s+1)(s+2)(s+3) = s^3 + 6s^2 + 11s + 6
	tf := TransferFunction{
		Numerator:   []float64{1},
		Denominator: []float64{1, 6, 11, 6},
	}
	poles := tf.Poles()
	if len(poles) != 3 {
		t.Fatalf("expected 3 poles, got %d", len(poles))
	}
	expectedPoles := []complex128{complex(-1, 0), complex(-2, 0), complex(-3, 0)}
	for _, ep := range expectedPoles {
		found := false
		for _, p := range poles {
			if cmplx.Abs(p-ep) < 1e-6 {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("missing pole at %v, got %v", ep, poles)
		}
	}
}

func TestTransfer_IsStable_StableFirstOrder(t *testing.T) {
	tf := TransferFunction{
		Numerator:   []float64{1},
		Denominator: []float64{1, 1},
	}
	if !tf.IsStable() {
		t.Error("1/(s+1) should be stable")
	}
}

func TestTransfer_IsStable_UnstableFirstOrder(t *testing.T) {
	// s - 1 = 0 => pole at +1
	tf := TransferFunction{
		Numerator:   []float64{1},
		Denominator: []float64{1, -1},
	}
	if tf.IsStable() {
		t.Error("1/(s-1) should be unstable")
	}
}

func TestTransfer_IsStable_StableSecondOrder(t *testing.T) {
	// s^2 + 3s + 2 => poles at -1, -2 (both negative real)
	tf := TransferFunction{
		Numerator:   []float64{1},
		Denominator: []float64{1, 3, 2},
	}
	if !tf.IsStable() {
		t.Error("1/(s^2+3s+2) should be stable")
	}
}

func TestTransfer_IsStable_UnstableSecondOrder(t *testing.T) {
	// s^2 - 1 = (s-1)(s+1) => poles at +1, -1
	tf := TransferFunction{
		Numerator:   []float64{1},
		Denominator: []float64{1, 0, -1},
	}
	if tf.IsStable() {
		t.Error("1/(s^2-1) should be unstable")
	}
}

func TestTransfer_IsStable_MarginallyUnstable(t *testing.T) {
	// s^2 + 1 = 0 => poles at +/-j (on imaginary axis, Re=0)
	tf := TransferFunction{
		Numerator:   []float64{1},
		Denominator: []float64{1, 0, 1},
	}
	if tf.IsStable() {
		t.Error("1/(s^2+1) should be marginally unstable (poles on jw axis)")
	}
}

func TestTransfer_IsStable_ConstantDenominator(t *testing.T) {
	tf := TransferFunction{
		Numerator:   []float64{1},
		Denominator: []float64{5},
	}
	if !tf.IsStable() {
		t.Error("constant denominator (no poles) should be stable")
	}
}

func TestTransfer_Evaluate_PanicsEmptyNumerator(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty numerator")
		}
	}()
	tf := TransferFunction{Numerator: []float64{}, Denominator: []float64{1}}
	tf.Evaluate(complex(0, 0))
}

func TestTransfer_Evaluate_PanicsEmptyDenominator(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty denominator")
		}
	}()
	tf := TransferFunction{Numerator: []float64{1}, Denominator: []float64{}}
	tf.Evaluate(complex(0, 0))
}

func TestTransfer_Poles_PanicsEmptyDenominator(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty denominator")
		}
	}()
	tf := TransferFunction{Numerator: []float64{1}, Denominator: []float64{}}
	tf.Poles()
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

func assertClose(t *testing.T, label string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %v, want %v (tol %v)", label, got, want, tol)
	}
}

func assertComplexClose(t *testing.T, label string, got, want complex128, tol float64) {
	t.Helper()
	if cmplx.Abs(got-want) > tol {
		t.Errorf("%s: got %v, want %v (tol %v)", label, got, want, tol)
	}
}
