// Package hmm implements the hidden Markov model algorithms of Rabiner's
// tutorial: the Forward and Backward passes (log-space scaled), the Viterbi
// most-likely-path decoder, and Baum-Welch (EM) re-estimation for categorical
// emissions.
//
// A hidden Markov model has N hidden states and, in the discrete case, M
// observation symbols. It is parameterised by
//
//	Pi[i]      initial state distribution,  sum_i Pi[i] = 1
//	A[i*N+j]   state transition matrix,     sum_j A[i][j] = 1
//	B[i*M+k]   emission matrix,             sum_k B[i][k] = 1
//
// (all flat row-major). Rabiner poses three problems: (1) evaluation — the
// likelihood P(O | lambda), solved by Forward; (2) decoding — the single most
// likely state path, solved by Viterbi; (3) learning — re-estimate lambda from
// O, solved by Baum-Welch.
//
// # Why this exists in Reality
//
// Reality's CONTEXT.md:424 promised RubberDuck "real stochastic calculus …
// HMM … validated against golden files with known precision bounds." The HMM
// half never shipped. RubberDuck runs a hand-rolled HiddenMarkovModel.cs whose
// Baum-Welch fit and regime posterior gate live strategy behaviour; a
// deep-dive measured a single fit at ~20,000 hand-composed NormalPDF/Bayesian
// calls with no reference to pin it to. This package is that reference. The
// golden vectors are the canonical Rabiner-tutorial worked example
// (states {Healthy, Fever}, symbols {normal, cold, dizzy}) whose Forward
// likelihood P(O)=0.03628 and Viterbi best path {Healthy,Healthy,Fever} with
// probability 0.01512 are widely published; they are additionally anchored by
// exhaustive brute-force enumeration over all N^T state paths in the tests.
//
// # Log-space scaling
//
// A naive Forward pass multiplies T emission/transition probabilities and
// underflows to zero for even moderate T. Rabiner's remedy is per-step
// scaling; the equivalent, adopted here, is to carry every quantity in log
// space and combine with log-sum-exp. logSumExp subtracts the running maximum
// before exponentiating, so the pass is stable for any T and any state count.
// Viterbi runs in log space too: products become sums and argmax is exact.
//
// The emission model is factored out. ForwardLog / BackwardLog / ViterbiLog
// take a precomputed T×N matrix of emission log-likelihoods
// logB[t*N+i] = log P(o_t | state i), so the same decoding core serves the
// discrete case (built from the emission matrix B) and a Gaussian-emission
// consumer such as RubberDuck (built from a Normal log-pdf). The high-level
// Forward/Backward/Viterbi/Posterior/BaumWelch wrappers take a discrete Model
// and integer observation indices.
//
// # Determinism and precision
//
// Every routine is deterministic. Baum-Welch is deterministic given its
// initial model: the same init and iteration cap always yield the same fit;
// the log-likelihood is non-decreasing across iterations (the EM guarantee)
// and the loop stops on convergence (|Δ log-lik| < tol) or the iteration cap,
// whichever comes first. Viterbi resolves argmax ties toward the lowest state
// index. Zero non-stdlib dependencies (math only). Golden vectors are pinned
// at 1e-9; the exact terminating-decimal cases hold to 1e-12.
//
// # References
//
//   - Rabiner L. R. (1989). A Tutorial on Hidden Markov Models and Selected
//     Applications in Speech Recognition. Proceedings of the IEEE 77(2):
//     257-286. (Forward-Backward, Baum-Welch re-estimation, scaling §V.A.)
//   - Viterbi A. J. (1967). Error Bounds for Convolutional Codes and an
//     Asymptotically Optimum Decoding Algorithm. IEEE Trans. Info. Theory
//     13(2):260-269.
//   - Baum L. E., Petrie T., Soules G. & Weiss N. (1970). A Maximization
//     Technique Occurring in the Statistical Analysis of Probabilistic
//     Functions of Markov Chains. Ann. Math. Statist. 41(1):164-171.
//   - Forney G. D. (1973). The Viterbi Algorithm. Proceedings of the IEEE
//     61(3):268-278.
package hmm
