package optim

import (
	"errors"
	"testing"
)

// ---------------------------------------------------------------------------
// R123 — VALIDATED-ITERATE-CONVERGENCE regression suite.
//
// Demonstrates the GARCH-class spurious-convergence trap and proves the
// *Validated variants catch it. The trap shape: the gradient function
// returns sentinel-zero on invalid input as a "RetainPreviousIterate"
// recovery; the original loop terminator (gnorm < tol) trips on iter-0
// because zero-gradient ⇒ zero gnorm ⇒ gnorm < tol.
// ---------------------------------------------------------------------------

// trapObjective constructs a gradient function whose validity domain is
// |x| < 1 and which emits sentinel-zero gradient outside that domain.
// The objective itself is f(x) = sum(x_i^2) inside the domain. A warm-start
// outside the domain (e.g. x0 = [10]) trips the R123 trap because the
// gradient is zero on the warm-start, gnorm == 0 < tol, and the original
// GradientDescent reports converged=true at the invalid iterate.
func trapObjective(invalid func([]float64) bool) (
	f func([]float64) float64,
	grad func([]float64, []float64),
) {
	f = func(x []float64) float64 {
		sum := 0.0
		for _, xi := range x {
			sum += xi * xi
		}
		return sum
	}
	grad = func(x, g []float64) {
		if invalid(x) {
			// RetainPreviousIterate recovery: emit zero gradient.
			// The bug class: outer optimisers that don't validate
			// will read zero as "no progress = converged".
			for i := range g {
				g[i] = 0
			}
			return
		}
		for i, xi := range x {
			g[i] = 2.0 * xi
		}
	}
	return f, grad
}

// validityPredicate returns true when |x_i| < 1 for all i.
func validityPredicate(x []float64) bool {
	for _, xi := range x {
		if xi*xi >= 1.0 {
			return false
		}
	}
	return true
}

// invalidPredicate is the negation of validityPredicate, plumbed into the
// trap objective's gradient.
func invalidPredicate(x []float64) bool {
	return !validityPredicate(x)
}

// --- The trap as observed today ---------------------------------------------

func TestR123ConvergenceTrap_VanillaGradientDescent_FalselyConverges(t *testing.T) {
	// Vanilla GradientDescent has no validate hook. With a warm-start
	// outside the validity domain, the trap gradient returns zeros,
	// gnorm == 0, and the loop exits on iter-0 with the *invalid* x0
	// as the supposed converged solution. This documents the bug class
	// the *Validated variants address.
	f, grad := trapObjective(invalidPredicate)
	x0 := []float64{10.0} // outside the |x| < 1 validity domain

	got := GradientDescent(f, grad, x0, 0.1, 1000, 1e-10)

	// Vanilla returns the warm-start unchanged because it "converged"
	// on iter-0. The returned point IS invalid by construction.
	if got[0] != 10.0 {
		t.Fatalf("vanilla GradientDescent moved x: got %v, want [10] (it should false-converge on the warm-start)", got)
	}
	if validityPredicate(got) {
		t.Fatalf("vanilla GradientDescent returned a valid point; expected false-convergence on invalid warm-start. got=%v", got)
	}
}

// --- The fix as it ships ----------------------------------------------------

func TestR123ConvergenceTrap_GradientDescentValidated_CatchesTrap(t *testing.T) {
	// GradientDescentValidated guards the convergence break with the
	// validity predicate. On an invalid warm-start the trap fires at
	// iter-0 (gnorm==0) but validate(x)==false, so the function
	// returns Converged=false with the diagnostic reason.
	f, grad := trapObjective(invalidPredicate)
	x0 := []float64{10.0}

	res, err := GradientDescentValidated(f, grad, x0, 0.1, 1000, 1e-10, validityPredicate)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Converged {
		t.Fatalf("expected Converged=false (invalid warm-start), got Converged=true (R123 trap not caught)")
	}
	if res.Iters != 1 {
		t.Fatalf("expected Iters=1 (trap fires on iter-0 evaluation), got %d", res.Iters)
	}
	if res.Reason != "tolerance hit on invalid iterate (R123 trap caught)" {
		t.Fatalf("unexpected reason: %q", res.Reason)
	}
}

func TestR123ConvergenceTrap_GradientDescentValidated_StillConvergesOnValidWarmStart(t *testing.T) {
	// Sanity check: with a valid warm-start, the *Validated variant
	// should converge normally to the minimum (x = 0).
	f, grad := trapObjective(invalidPredicate)
	x0 := []float64{0.5} // inside |x| < 1

	res, err := GradientDescentValidated(f, grad, x0, 0.1, 1000, 1e-10, validityPredicate)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.Converged {
		t.Fatalf("expected Converged=true on valid warm-start, got false (reason=%q)", res.Reason)
	}
	if res.X[0]*res.X[0] >= 1e-10 {
		t.Fatalf("did not converge to minimum: got x=%v", res.X)
	}
}

func TestR123ConvergenceTrap_LBFGSValidated_CatchesTrap(t *testing.T) {
	f, grad := trapObjective(invalidPredicate)
	x0 := []float64{10.0}

	res, err := LBFGSValidated(f, grad, x0, 5, 1000, 1e-10, validityPredicate)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Converged {
		t.Fatalf("expected Converged=false (invalid warm-start), got Converged=true (R123 trap not caught)")
	}
	if res.Reason != "tolerance hit on invalid iterate (R123 trap caught)" {
		t.Fatalf("unexpected reason: %q", res.Reason)
	}
}

func TestR123ConvergenceTrap_LBFGSValidated_StillConvergesOnValidWarmStart(t *testing.T) {
	f, grad := trapObjective(invalidPredicate)
	x0 := []float64{0.5}

	res, err := LBFGSValidated(f, grad, x0, 5, 200, 1e-10, validityPredicate)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !res.Converged {
		t.Fatalf("expected Converged=true on valid warm-start, got false (reason=%q)", res.Reason)
	}
}

// --- API hygiene ------------------------------------------------------------

func TestR123_ValidatedVariants_RejectNilValidate(t *testing.T) {
	f := func(x []float64) float64 { return x[0] * x[0] }
	grad := func(x, g []float64) { g[0] = 2 * x[0] }

	_, gdErr := GradientDescentValidated(f, grad, []float64{1.0}, 0.1, 10, 1e-6, nil)
	if !errors.Is(gdErr, ErrNilValidate) {
		t.Fatalf("expected ErrNilValidate, got %v", gdErr)
	}
	_, lbfgsErr := LBFGSValidated(f, grad, []float64{1.0}, 5, 10, 1e-6, nil)
	if !errors.Is(lbfgsErr, ErrNilValidate) {
		t.Fatalf("expected ErrNilValidate, got %v", lbfgsErr)
	}
}

func TestR123_ValidatedVariants_BudgetExhaustedReportsHonestly(t *testing.T) {
	// Tiny step + tight tol + small budget — should hit max iters
	// without converging, and the result should reflect that.
	f, grad := trapObjective(invalidPredicate)
	x0 := []float64{0.5} // valid; will move toward 0 but slowly

	res, err := GradientDescentValidated(f, grad, x0, 1e-6, 5, 1e-12, validityPredicate)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Converged {
		t.Fatalf("expected Converged=false (budget exhausted), got Converged=true")
	}
	if res.Iters != 5 {
		t.Fatalf("expected Iters=5 (budget), got %d", res.Iters)
	}
	if res.Reason != "max iterations exhausted" {
		t.Fatalf("unexpected reason: %q", res.Reason)
	}
}
