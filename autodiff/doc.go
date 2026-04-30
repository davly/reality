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
// Per the math + technique cross-pollination hunts, reverse-mode AAD is
// load-bearing for:
//
//   - GARCH / DCC-GARCH calibration (ill-posed; needs Tikhonov-regularised
//     Newton-CG with adjoint gradients, per PLAN_RISKS.md R3 mitigation)
//   - Heston / SABR / rough-vol calibration (same reason)
//   - Risk-parity Newton iteration (gradient of ERC objective)
//   - NSGA-II contagion-beta gradient operator
//   - Any composite calibration whose loss has more than ~10 parameters
//
// Without a centralised AAD primitive every consumer either:
//   - Uses finite differences (slow, scales as #params)
//   - Re-implements the chain rule by hand for one specific model (fragile)
//   - Pulls in a dependency (violates Reality's zero-dep policy)
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
