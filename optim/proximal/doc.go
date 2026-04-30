// Package proximal implements proximal-splitting methods for non-smooth
// convex optimization. These methods solve problems of the form
//
//	minimize_x  f(x) + g(x)                        (FBS / FISTA)
//	minimize_x  f(x) + g(z)  subject to  x = z     (consensus ADMM)
//
// where f and g are convex (extended-real-valued) and at least one term may
// be non-smooth (an indicator of a feasible set, an L1 norm, etc.).
//
// # Problem class
//
// Proximal methods are the workhorse for convex problems that are too big
// or too non-smooth for interior-point QP solvers. They are first-order
// (gradient + prox) and parallelisable, with O(1/k) convergence (FBS) or
// O(1/k^2) (FISTA acceleration) for L-smooth f.
//
// Concretely this package targets:
//
//   - LASSO regression  min ||Ax-b||^2 + lambda*||x||_1
//   - Box / non-negative least squares
//   - Constrained quadratic programs with simple feasible sets
//   - Sparse dictionary learning, basis pursuit
//   - Total-variation denoising (via ADMM)
//   - Markowitz portfolio with constraints x in box, sum(x) = 1
//
// # Why this exists in Reality
//
// Per the math + technique cross-pollination hunts, ~25 downstream services
// across the math-frontier and the project-driven hunt cite proximal
// calculus as a foundational substrate. Without a centralised proximal
// package every consumer rolls its own gradient-projection or L-BFGS hack
// for what is structurally the same algorithm.
//
// # MVP-of-MVP scope
//
// This package ships:
//
//   - 8 proximal operators (L1 / L0 / box / non-negative / L2-ball /
//     simplex / squared-L2 / linear)
//   - Forward-Backward Splitting (FBS, Bauschke-Combettes 2011 §28)
//   - FISTA acceleration (Beck-Teboulle 2009 SIAM J Img Sci 2:183)
//   - Consensus ADMM (Boyd-Parikh-Chu-Peleato-Eckstein 2011 FTML 3:1)
//
// Deferred to v2: Davis-Yin three-operator splitting, Chambolle-Pock
// primal-dual, prox of nuclear norm (needs SVD), prox of sorted-L1 (SLOPE).
//
// # Determinism + allocations
//
// All iterations are deterministic (no randomness, no parallelism). Functions
// follow Reality's out-buffer convention: callers pre-allocate work slices
// and pass them in. No allocations occur inside the inner iteration loop
// once buffers are sized.
//
// # References
//
//   - Bauschke H. H. & Combettes P. L. (2011). Convex Analysis and Monotone
//     Operator Theory in Hilbert Spaces. Springer. Chapter 28.
//   - Beck A. & Teboulle M. (2009). A Fast Iterative Shrinkage-Thresholding
//     Algorithm for Linear Inverse Problems. SIAM J Imaging Sci 2(1):183-202.
//   - Boyd S., Parikh N., Chu E., Peleato B. & Eckstein J. (2011).
//     Distributed Optimization and Statistical Learning via the Alternating
//     Direction Method of Multipliers. Found. Trends Mach. Learn. 3(1):1-122.
//   - Parikh N. & Boyd S. (2014). Proximal Algorithms. Found. Trends
//     Optimization 1(3):127-239.
package proximal
