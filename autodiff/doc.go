// Package autodiff implements reverse-mode automatic differentiation
// (a.k.a. backpropagation, adjoint algorithmic differentiation, AAD) on
// scalar-valued computations expressed as a tape of elementary operations.
//
// A Tape records the chain of operations that produced a final scalar.
// Backward then walks the tape in reverse, applying each operation's local
// derivative (the "pullback") to accumulate gradients with respect to the
// input variables.  The cost of one full gradient is a small constant
// multiple of the cost of the forward evaluation, regardless of how many
// inputs there are — the asymptotic advantage of reverse-mode over
// forward-mode or finite differences when there are many inputs and one
// output.
//
// # Why this exists in Reality
//
// The math + technique cross-pollination hunts identified reverse-mode AAD
// as a candidate substrate for:
//
//   - GARCH / DCC-GARCH calibration (ill-posed; needs Tikhonov-regularised
//     Newton-CG with adjoint gradients, per PLAN_RISKS.md R3 mitigation)
//   - Heston / SABR / rough-vol calibration (same reason)
//   - Risk-parity Newton iteration (gradient of ERC objective)
//   - NSGA-II contagion-beta gradient operator
//   - Any composite calibration whose loss has more than ~10 parameters
//
// As of 2026-05-05 the second-order Heston / SABR / NSGA-II citations
// remain aspirational (those packages do not yet exist as Reality
// packages; verified by substring grep on github.com/davly/reality/autodiff
// across foundation/, infrastructure/, sdk/, apps/).  The first real
// consumer landed in S62 — see Consumers below.  Without a centralised AAD
// primitive every future consumer would either:
//   - Use finite differences (slow, scales as #params)
//   - Re-implement the chain rule by hand for one specific model (fragile)
//   - Pull in a dependency (violates Reality's zero-dep policy)
//
// Consumers (verified):
//   - timeseries/garch/autodiff_test.go:TestNegLogLikGrad_AutodiffEquivalence —
//     reverse-mode AD over the GARCH(1,1) negative-log-likelihood plus
//     Tikhonov penalty; pins the closed-form analytic gradient in
//     timeseries/garch/fit.go:negLogLikGrad against the autodiff path at
//     1e-9 parity. First cross-package consumer for autodiff (S62
//     2026-05-05); the first item in the doc-comment use-case list above
//     (GARCH calibration) is therefore now backed by a real composition
//     test, not a hunt citation.
//   - infogeo/autodiff_test.go:TestKL_AutodiffGradientMatchesQMinusP —
//     pins autodiff's gradient of KL(p || softmax(θ)) against the
//     analytic closed form q − p (the canonical natural-gradient /
//     variational-inference / policy-gradient identity) at 1e-9
//     tolerance across three (p, θ) cases. Second cross-package
//     consumer for autodiff (S62 overnight, 2026-05-06); the
//     R-CLOSED-FORM-PINNED-TO-AUTODIFF pattern is now at 2/3
//     saturation.
//
// Flagship first-consumer push remains queued; see
// LimitlessGodfather/reviews/SESSION_62_PROGRESS.md.
//
// # MVP scope
//
// This package ships:
//
//   - Tape + Variable types (the computational graph)
//   - 12 elementary operations: +, -, *, /, ^, exp, log, sqrt, sin, cos,
//     tanh, neg
//   - 4 vector ops: dot product, sum, vector add, scalar-vector multiply
//   - Backward(out) returning gradients indexed by Variable id
//
// Deferred to v2: Hessian-vector products via forward-over-reverse,
// checkpointing for memory-bounded backprop, taped control flow, broadcast.
//
// # Convention
//
// All ops construct new Variables on the same Tape; mixing tapes panics.
// Forward values propagate eagerly; pullbacks are registered as closures
// and executed lazily on Backward.  Variables are identified by integer
// id; gradients are returned as a slice indexed by id.
//
// # Determinism + allocations
//
// All operations are deterministic.  The tape grows by one entry per op;
// Backward executes one closure per tape entry.  Allocation is concentrated
// at construction; the hot inner pullback closures do not allocate.  Zero
// non-stdlib deps.
//
// # References
//
//   - Griewank A. & Walther A. (2008). Evaluating Derivatives: Principles
//     and Techniques of Algorithmic Differentiation, 2nd ed.  SIAM.
//   - Baydin A. G., Pearlmutter B. A., Radul A. A. & Siskind J. M. (2018).
//     Automatic Differentiation in Machine Learning: a Survey.  JMLR
//     18(153):1-43.
//   - Wengert R. E. (1964). A simple automatic derivative evaluation
//     program.  CACM 7(8):463-464.
package autodiff
