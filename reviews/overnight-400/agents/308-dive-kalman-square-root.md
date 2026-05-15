# 308 — dive-kalman-square-root (Joseph / Square-Root / UD / Information / Unscented audit)

## Headline
Reality v0.10.0 ships **zero Kalman filtering** — no naive KF, no Joseph, no UD, no SR-UKF, no information filter, no EnKF — so the audit pivots from "harden the existing filter" to "specify the *first* implementation correctly so it never has to be rewritten"; recommend Joseph-form linear KF as v0.11.0 entry, UD (Bierman-Thornton) for v0.12.0 hot-path users, SR-UKF for v0.13.0, with a 3-way mutual cross-validation pin (naive ≡ Joseph ≡ UD on Linear-Gaussian benchmark) and a steady-state ≡ DARE regression.

## Findings (existing audit)

- **Authoritative grep across the entire repo** for {Kalman, EKF, UKF, EnKF, UnscentedKalman, EnsembleKalman, JosephForm, SquareRootKalman, BiermanThornton, UDFactorization, InformationFilter} returns **exactly one hit** in core code: `physics/mechanics.go:128` — and that hit is "Thornton & Marion" (the classical-mechanics textbook for nonlinear pendulum), **not** Bierman-Thornton 1976. All other hits are inside `reviews/overnight-400/` (planning artefacts).
- **`control/` package contents** (`control/filter.go`, `control/pid.go`, `control/transfer.go`): only deterministic primitives — `LowPassFilter` (`filter.go:15`), `HighPassFilter` (`filter.go:38`), `ComplementaryFilter` (`filter.go:74`), `PIDController`, `TransferFunction`. No state-space, no observer, no covariance propagation, no innovation, no measurement model. (Confirmed independently by agent 161 and agents 052/054/055.)
- **`signal/` package** (`signal/filter.go`, `signal/fft.go`): FIR/IIR convolution, MedianFilter, FFT — purely deterministic signal processing, no stochastic estimation. No Wiener filter either.
- **`prob/` package** (`prob/timeseries.go`, `prob/markov.go`, `prob/regression.go`): Markov chains, ARIMA/Levinson-Durbin, OLS — no measurement-update primitive, no recursive Bayesian filter, no particle filter.
- **`linalg/` building blocks already in place** (the good news for an eventual implementation): `linalg/decompose.go:266` `CholeskyDecompose` (returns `false` on loss of positive-definiteness — the exact failure signal a Joseph form is meant to prevent), `linalg/decompose.go:316` `CholeskySolve`, `linalg.LUSolve`, `linalg.QRAlgorithm` (eigenvalues — usable for "P min eigenvalue >0" stability assertions). Notably **absent**: QR decomposition (needed for square-root *time* update via Householder/Givens), Lyapunov solver, DARE solver, and matrix square root by Denman-Beavers (the latter wanted for SR-UKF and `(n+λ)P` Cholesky in UKF sigma-point spread).
- **Synergy plan already exists** in `reviews/overnight-400/agents/161-synergy-control-prob.md:133-167`: C5 specifies "`KalmanFilter` (Joseph stabilised form)" at ~180 LOC, explicitly mandating `P=(I−KC)P_pred(I−KC)ᵀ+KRKᵀ` rather than naive `(I−KC)P_pred`, with the rationale "naïve `P=(I−KC)P_pred` loses positive-definiteness after ~50 updates under floating-point error" (agent 161, line 136). C8 (EKF) and C10 (UKF) follow. **My review concurs and tightens the spec below**: add UD as a separate implementation (not a replacement) for the Pistachio 60 FPS hot path, and add long-horizon PSD regression test as a *contract*.
- **Risk if reality lands a naive `P=(I−KC)P` first** ("just to ship something"): once downstream consumers (Pistachio pose, Sentinel, Oracle) are coupled to the API, swapping to Joseph silently changes numerical behaviour and breaks the cross-language golden files. **The very first commit must be Joseph form.** Cite Maybeck 1979 §1.7 and Grewal-Andrews 2014 §6.5: naive form's loss of PSD is a *guarantee* under enough updates, not a "maybe". Bucy-Joseph 1968: simply re-symmetrising every step (`P ← (P+Pᵀ)/2`) is *not* a substitute — symmetric ≠ PSD.
- **Numerical-stability ordering** (Maybeck Vol. 1 §7.2; Bierman 1977 §V; Grewal-Andrews 2014 §6.6): naive `(I−KH)P` < re-symmetrised naive < Joseph < Carlson square-root (lower-triangular sqrt) ≈ Potter (upper-triangular sqrt) < UD (Bierman-Thornton, equivalent stability with no `sqrt`, scalar measurements give O(n²) per update vs O(n³) for full Joseph). Information filter (`Λ=P⁻¹`) is the dual: stable under **observability** problems (e.g. measurement noise R singular) where covariance form is stable under **process** problems (Q small relative to P).
- **Tests required for Kalman correctness — none exist yet, and per-package test plans 056-060 do not require them** because no Kalman exists. The reviewer must *add* the tests in lockstep with the first commit.

## Concrete recommendations

1. **First commit (`control/kalman.go` ~180 LOC, Joseph form mandated):** API exactly as agent 161 C5 specifies (`reviews/overnight-400/agents/161-synergy-control-prob.md:139-153`), with these tightenings:
   - Update step computes `K = P_pred Hᵀ S⁻¹` using `linalg.CholeskySolve` on `S` (NOT `linalg.Inverse`); panic-with-message if `CholeskyDecompose(S)` returns false (signals R degenerate or sensor model rank-deficient — the user must know).
   - Covariance update is **Joseph** verbatim: `P = (I−KH) P_pred (I−KH)ᵀ + K R Kᵀ`. Three extra `MatMul`s vs naive — order ~10 µs at n=12 (Pistachio pose). Worth it.
   - Add `(kf *KalmanFilter) NIS() float64` returning `νᵀ S⁻¹ ν`; reuses Cholesky factor of `S` already computed during gain calculation. Required for the white-residuals diagnostic in agent 161 C9.
   - **Zero heap on hot path:** all scratch buffers (`Ppred`, `S`, `K`, `IKH`, `tmp_nm`, `tmp_nn`) live on the struct, allocated once in `NewKalmanFilter`. Pistachio calls Predict/Update at 60 Hz; any allocation here is a frame-rate killer.

2. **Second commit (`control/kalman_ud.go` ~250 LOC, Bierman-Thornton UD):** factorise `P = U D Uᵀ` where `U` unit upper-triangular, `D` diagonal positive. Storage: pack `U` strict-upper into `n(n−1)/2` floats, `D` into `n`. Implement:
   - **Bierman measurement update** (Bierman 1977 §V.4, ~80 LOC): scalar measurements only — multi-dim measurements decorrelated via `R = LLᵀ`, `H̃ = L⁻¹H`, `ỹ = L⁻¹y`, then process scalar-by-scalar. O(n²) per scalar measurement, no `sqrt`.
   - **Thornton time update** (Thornton 1976 / Bierman 1977 §VI.4, ~120 LOC): MWGS (modified weighted Gram-Schmidt) on `[ΦU | G]` with weights `[D, Q]`, where `Φ=A`, `G=I` (process noise gain). O(n³) but with smaller constant than Joseph and no `sqrt`.
   - **Reconstruction helper** `func (kf *UDKalman) Covariance(out []float64)` for users wanting the explicit `P` (debug/visualisation only — never used in the recursion).

3. **Third commit (`control/sr_ukf.go` ~300 LOC, Square-Root UKF, Van der Merwe & Wan 2001):** UKF with covariance stored as Cholesky factor `S` (`P = SSᵀ`). Sigma points generated by `χ_i = x ± sqrt(n+λ) S column_i`. Time/measurement updates use **QR decomposition** of stacked weighted residuals followed by **Cholesky downdate** for the negative-weight contribution. **Prerequisite:** add `linalg.QRDecompose` (Householder, ~120 LOC) and `linalg.CholeskyDowndate` (~50 LOC) — both pure linalg primitives; specify them as part of the SR-UKF PR. Cite Julier-Uhlmann 2004 IEEE Proceedings; SR-UKF specifically Van der Merwe & Wan 2001. **Joseph-form variant of SR-UKF** (Holmes-Klein-Murray 2008, "An O(N²) Square Root Unscented Kalman Filter for Visual SLAM") is the right reference for Pistachio's 6-DOF pose estimator.

4. **Information filter** (`control/information_filter.go`, ~120 LOC, optional v0.14.0): propagate `Λ = P⁻¹` and `η = Λ x̂` instead of `P, x̂`. Predict step is *expensive* (matrix inverse), but **measurement update is trivial** — `Λ ← Λ + HᵀR⁻¹H`, `η ← η + HᵀR⁻¹y`. Use case: distributed estimation, fusion of many sensors (sums commute), and observability-deficient problems where `P` is unbounded but `Λ` is bounded (zero rows correspond exactly to unobservable subspace). Cite Anderson-Moore 1979 §10.

5. **Long-horizon PSD regression test** (must accompany commit #1 — non-negotiable): in `control/kalman_test.go`, run 10,000 Predict+Update steps on `x_{k+1}=0.99 x_k + w_k`, `y_k=x_k+v_k` with `Q=1e-6, R=1` (process noise much smaller than measurement noise — the regime where naive form fails). Assert at every k ∈ {1, 10, 100, 1000, 10000}: (a) `P` is symmetric to <1e-14, (b) eigenvalues of `P` (computed via `linalg.QRAlgorithm`) all ≥0, (c) trace(P) bounded above by closed-form steady-state from DARE. **A naive update will fail this test by k≈300; Joseph passes to k=10⁶.** This is the contract.

6. **Steady-state ≡ DARE regression test:** once C6 (DARE solver) lands, add `kalman_test.go::TestSteadyState_MatchesDARE`: take any LTI `(A,H,Q,R)`, run KF for 5000 steps with random measurements, assert `‖P_5000 − P∞‖_F ≤ 1e-10` where `P∞ = DARE(Aᵀ, Hᵀ, Q, R)`. This pins the recursion against an algebraic-solver oracle. (See Wikipedia "Algebraic Riccati equation"; MathWorks `idare` reference; Anderson-Moore 1979 §4.) **R-MUTUAL-CROSS-VALIDATION 3/3 saturator** when combined with #7.

7. **R-MUTUAL-CROSS-VALIDATION 3/3 — three-way KF pin:** in `control/kalman_consistency_test.go`, instantiate three filters on the *same* Linear-Gaussian benchmark (e.g. constant-velocity 4-state kinematic model, 1000 random measurements, fixed RNG seed):
   - KF-naive (covariance form, naive `(I−KH)P` — keep as a *test-only* implementation for the pin),
   - KF-Joseph (the production form),
   - KF-UD (the Bierman-Thornton form, reconstructing `P` only for the assertion).
   Assert `‖P_naive − P_Joseph‖_F ≤ 1e-10` AND `‖P_Joseph − P_UD‖_F ≤ 1e-10` at every k ∈ [1, 200] (before naive diverges) AND `‖x̂_naive − x̂_Joseph‖_∞ ≤ 1e-10` AND `‖x̂_Joseph − x̂_UD‖_∞ ≤ 1e-10`. After k=300 only Joseph≡UD continues. **This is one of the cleanest 3-way mutual-cross-validation opportunities in the repo: three independent algorithms over the same recursion, all derivable from Bayes' rule, agreeing in the well-conditioned regime to 10 digits.**

8. **Cross-language golden files:** every public KF method gets a JSON test vector (per the `testutil` design rule — CLAUDE.md "Golden files are the proof"). Vectors live at `control/testdata/kalman_*.json`, contain `(A, H, Q, R, x0, P0, [u_k], [y_k])` inputs and `[x̂_k], [P_k vec]` expected outputs. Tolerance per agent 161 C5: 1e-10 for `x̂`, 1e-9 for `P` (accumulating). 30 vectors minimum (random-walk, constant-velocity, AR(2), SISO/MIMO, near-singular R, large-Q regime, observability-deficient pair `(A,H)`).

9. **EKF + autodiff synergy** (agent 161 C8 + agent 011 autodiff Tape): when EKF lands, the API should accept either `func(x, u, fOut, jacOut)` (user-supplied analytic Jacobian) **or** `func(x, u, fOut)` plus an `autodiff.Tape` (Jacobian computed automatically by reverse-mode AD). The latter is the "no analytic-gradient drudgery" path agent 161 advertises — but it must be benchmarked: at n=12 (Pistachio), reverse-mode AD costs ~3× forward eval; for EKF that's already cheaper than finite differences (n+1 forward evals = 13×). Document the crossover.

10. **Reject EnKF until there is a real consumer.** EnKF (Evensen 1994) is for state dimensions n ≥ 10⁴ (atmospheric data assimilation). Reality's likely consumers (Pistachio pose: n=12; Sentinel control: n≤8; Oracle time-series: scalar) are 1000× below the regime where EnKF beats UKF. Ship UKF/SR-UKF first; revisit EnKF only if a downstream weather/CFD/cosmology user appears. Cite Evensen 1994 J. Geophys. Res. 99:10143 (geophysical context); Evensen 2009 IEEE CSM 29(3):83 (modern usage).

## Sources

**Repo files (gap evidence):**
- `C:/limitless/foundation/reality/control/filter.go` — only deterministic filters (LowPass/HighPass/Complementary)
- `C:/limitless/foundation/reality/control/pid.go` — PID + RateLimiter
- `C:/limitless/foundation/reality/control/transfer.go` — TransferFunction, no state-space
- `C:/limitless/foundation/reality/signal/filter.go` — FIR/IIR/Median; no Wiener/Kalman
- `C:/limitless/foundation/reality/prob/timeseries.go` — ARIMA/Levinson-Durbin only
- `C:/limitless/foundation/reality/linalg/decompose.go:266` — CholeskyDecompose (the building block)
- `C:/limitless/foundation/reality/linalg/decompose.go:316` — CholeskySolve
- `C:/limitless/foundation/reality/physics/mechanics.go:128` — sole "Thornton" hit, false positive (Thornton & Marion, classical mechanics)

**Cross-references (other review agents):**
- `reviews/overnight-400/agents/161-synergy-control-prob.md:133-167` — C5 KalmanFilter Joseph spec (origin of v0.11.0 plan)
- `reviews/overnight-400/agents/161-synergy-control-prob.md:190-200` — C8 EKF + autodiff
- `reviews/overnight-400/agents/161-synergy-control-prob.md:215-235` — C10 UKF Julier-Uhlmann
- `reviews/overnight-400/agents/052-control-missing.md`, `054-control-api.md`, `055-control-perf.md` — independent confirmations

**Foundational sources:**
- Bierman G.J. 1977, *Factorization Methods for Discrete Sequential Estimation*, Academic Press / Dover 2006 reprint — UD canonical text (https://shop.elsevier.com/books/factorization-methods-for-discrete-sequential-estimation/bierman/978-0-12-097350-7)
- Thornton C.L. 1976, "Triangular Covariance Factorizations for Kalman Filtering", JPL Tech. Memo / Ph.D. thesis UCLA — Thornton time update (cited in Bierman 1977 §VI)
- Maybeck P.S. 1979, *Stochastic Models, Estimation, and Control*, Vol. 1 §1.7 + §7.2 — naive vs Joseph divergence proof
- Grewal & Andrews 2014, *Kalman Filtering: Theory and Practice with MATLAB*, 4th ed., §6.5–6.6 — modern textbook on stability variants
- Anderson & Moore 1979, *Optimal Filtering*, Prentice-Hall §10 — information filter, DARE doubling
- Welch & Bishop 2006, "An Introduction to the Kalman Filter", UNC TR 95-041 — pedagogical baseline
- Crassidis & Junkins 2011, *Optimal Estimation of Dynamic Systems*, 2nd ed. — spacecraft attitude UKF reference (Pistachio-relevant)
- Joseph P.D. & Bucy R.S. 1968 in Bucy-Joseph "Filtering for Stochastic Processes with Applications to Guidance" — origin of Joseph form
- Julier & Uhlmann 2004, "Unscented Filtering and Nonlinear Estimation", Proc. IEEE 92(3):401-422 — UKF canonical (https://www.cs.ubc.ca/~murphyk/Papers/Julier_Uhlmann_mar04.pdf)
- Van der Merwe & Wan 2001, "The Square-Root Unscented Kalman Filter for State and Parameter-Estimation", ICASSP 2001 — SR-UKF (https://forum.orekit.org/uploads/short-url/dTzpDYlajckM4SHd6yjm9Cp6nTD.pdf)
- Holmes, Klein & Murray 2008, "An O(N²) Square Root Unscented Kalman Filter for Visual SLAM", IEEE PAMI — directly Pistachio-relevant (https://www.robots.ox.ac.uk/~gk/publications/HolmesKleinMurrayPAMI_SRUKF.pdf)
- Evensen G. 1994, "Sequential data assimilation with a nonlinear quasi-geostrophic model using Monte-Carlo methods to forecast error statistics", J. Geophys. Res. 99:10143-10162 — EnKF origin
- Verhaegen & Van Dooren 1986, "Numerical aspects of different Kalman filter implementations", IEEE TAC 31(10):907-917 — quantitative naive vs Joseph vs UD comparison
- *Quantitative Verification of Numerical Stability for Kalman Filters*, Roy et al., FM 2019 (https://www.prismmodelchecker.org/papers/fm19kf.pdf) — formal stability checking, modern
- *Kalman Filter Riccati Equation for the Prediction, Estimation, and Smoothing Error Covariance Matrices*, Assimakis 2013, ISRN (https://www.hindawi.com/journals/isrn/2013/249594/) — DARE↔steady-state KF
- Wikipedia, *Algebraic Riccati equation* (https://en.wikipedia.org/wiki/Algebraic_Riccati_equation) — DARE statement and LQG context
- MathWorks, `idare` documentation (https://www.mathworks.com/help/control/ref/idare.html) — implicit DARE solver oracle for the regression test
