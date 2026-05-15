# 309 — dive-kalman-info-form (Information-form Kalman / Particle Filter / Rao-Blackwell / Gaussian-Sum / Variational Bayes)

## Headline
Reality v0.10.0 ships **zero recursive Bayesian filtering** (slot 308 confirmed: no covariance-form Kalman, no info-form Λ/η, no particle filter, no Gaussian-sum, no VB filter); slot 309 is the **dual-form + non-Gaussian** layer that *must* land *after* slot 308's Joseph-form covariance KF, ordered as PR-A info-form (Λ, η, additive measurement update, distributed-fusion API; ~400 LOC) → PR-B Bootstrap PF + 4 resampling schemes (Gordon-Salmond-Smith 1993; ~300 LOC) → PR-C Rao-Blackwellized PF (Schön-Gustafsson-Nordlund 2005; ~350 LOC) → PR-D Gaussian-sum (Alspach-Sorenson 1972; ~250 LOC), with **R-MUTUAL-CROSS-VALIDATION 3/3** anchored on covariance-KF ≡ info-KF ≡ PF(N=10⁴) on a fixed Linear-Gaussian benchmark.

## Findings

### Existing surface (re-verified 2026-05-09)

- **Authoritative grep** across the 22 packages for {InformationFilter, ParticleFilter, SequentialMonteCarlo, RaoBlackwell, GaussianSum, VariationalBayesFilter, BootstrapFilter, Resampling} returns **zero callable matches** in core code. Every match (`prob/conformal/adaptive.go:160 EffectiveSampleSize`, slot 156/161/165/238/265/266 review docs) is either a Kish-window homonym or a planning artefact.
- `control/` contents confirmed: `filter.go` (LowPass/HighPass/Complementary), `pid.go`, `transfer.go`, `control_test.go`, `control_edge_test.go`. **No state-space struct, no innovation step, no covariance, no information matrix.** Same gap slot 308 documented.
- **Closest existing primitive** for slot 309's Bayesian-recursion family: `prob/markov.go` (discrete-state HMM-adjacent), `prob/timeseries.go` (ARIMA/Levinson-Durbin), `prob/conformal/adaptive.go:160` (Kish ESS — co-name hazard with SMC's `(Σw)²/Σw²`; slot 265 already flagged).
- **Linalg readiness** (relevant for info-form): `linalg/decompose.go:266 CholeskyDecompose` (PSD check on Λ), `linalg/decompose.go:316 CholeskySolve` (info → covariance switch-back), `LUSolve` (used in time-update inverse), no `linalg.SymmetricInverse` (one O(n³) op, but with PSD assertion — slot 309 needs to ship this as a 30-LOC linalg helper or accept LUSolve).
- **Cross-link inventory:** slot 161 C11 already specifies "Bootstrap Particle Filter / SIR (Gordon-Salmond-Smith 1993, ~250 LOC)" with `BoxMullerSample` + `SystematicResample`; slot 165 has a `ParticleFilter` API sketch with `EffectiveSampleSize`; slot 265 owns the SMC/PMCMC parameter-inference frontier; slot 266 owns the SMC-design axis (24 primitives including EnKF/EKF-PF/UKF-PF/Twisted-SMC) and the Sorenson-Alspach-1971 Gaussian-mixture filter as S14. **Slot 309's distinct ownership:** the **information-form covariance dual** (Λ = P⁻¹, η = Λx̂) — *not* covered in 161/165/265/266 except as a one-line cross-link in slot 308 recommendation #4. **309 owns the dual-form axis + the Bayesian-recursion *variants* not just SMC**: Gaussian-sum (multi-modal), Variational-Bayes (KL projection), and the hybrid info-form/covariance-form switching strategy.

### Why information form (the math)

The covariance form propagates `(x̂, P)`. The information form propagates `(η, Λ)` where:
- **Information matrix:** `Λ = P⁻¹` (positive-definite when P is)
- **Information vector:** `η = Λ x̂ = P⁻¹ x̂`

**Measurement update is *additive*:**
```
Λ_+ = Λ_- + Hᵀ R⁻¹ H
η_+ = η_- + Hᵀ R⁻¹ z
```
The right-hand sides are **information contributions**: each sensor `i` independently computes `(Hᵢᵀ Rᵢ⁻¹ Hᵢ, Hᵢᵀ Rᵢ⁻¹ zᵢ)`, and the central node sums. Order doesn't matter (associative + commutative). This is *the* reason information form is canonical for distributed sensor fusion (Anderson-Moore 1979 §6) and graph-SLAM (Thrun-Burgard-Fox 2005 §11).

**Time update is *expensive*** (the inverse step):
```
Λ_- = (F Λ_+⁻¹ Fᵀ + Q)⁻¹
η_- = Λ_- F Λ_+⁻¹ η_+
```
or equivalently via the matrix-inversion lemma:
```
M = Fᵀ Λ_+ F + ... (with process-noise correction term)
```
At minimum two matrix inverses per step (vs zero in covariance form). **Hybrid filters switch back to covariance form for the time update** (compute P = Λ⁻¹, propagate P forward, re-invert) when measurements are sparse vs propagation steps.

### Trade-off matrix

| Form | Measurement update | Time update | Initial uninformative prior | Distributed fusion | Observability-deficient |
|------|-------------------|-------------|-----------------------------|--------------------|-------------------------|
| Covariance (P, x̂) | O(n²m + n³) inversion of S | O(n³) MatMul | `P = ∞·I` (numerically broken) | Hard (covariances don't add) | Diverges (P unbounded) |
| Information (Λ, η) | **O(n²m) additive** | **O(n³) inversion** | **Λ = 0** (clean) | **Trivial sum** | **Stable** (Λ has zero eigenvalue, bounded) |

Cite Anderson-Moore 1979 §6 (canonical text); Maybeck 1979 Vol. 1 §5.7 (square-root info filter, SRIF).

### Bayesian recursion variants (the non-Kalman family)

1. **Bootstrap particle filter (Gordon-Salmond-Smith 1993):** sample `N` particles `{x_k^(i)}` from prior, propagate through `f`, weight by likelihood `p(y_k|x_k^(i))`, normalise, resample. **No Gaussian assumption, no linearity assumption.** Convergence: posterior ≡ true Bayes filter as N→∞.
2. **Resampling family:** multinomial (the original GSS-1993 — variance high), **systematic** (Kitagawa 1996, low variance, deterministic stride), **stratified** (Carpenter-Clifford-Fearnhead 1999), **residual** (Liu-Chen 1998, deterministic copies + multinomial residual). Theoretical ordering of variance: residual < stratified ≤ systematic < multinomial (Douc-Cappé-Moulines 2005).
3. **Rao-Blackwellized PF (Schön-Gustafsson-Nordlund 2005, "Marginalized Particle Filters for Mixed Linear/Nonlinear State-Space Models", IEEE TSP 53(7):2279):** when state factors as `x = (x^l, x^n)` with `x^l` linear-Gaussian conditional on `x^n`, sample only `x^n` (M particles) and run a Kalman filter on `x^l|x^n^(i)` per particle. Variance reduction = Rao-Blackwell theorem (E[Var[X|Y]] ≤ Var[X]). Practical 5-50× variance reduction at fixed N.
4. **Gaussian-sum filter (Alspach-Sorenson 1972, IEEE TAC 17(4):439-448):** posterior approximated as `p(x|y_{1:k}) ≈ Σ_j w_j N(x; μ_j, Σ_j)`. Each Gaussian propagated by its own (E)KF; weights updated by likelihood. **Multi-modal capable** (the one thing all Kalman variants cannot do). Component count grows; pruning/merging required.
5. **Variational Bayes filter (Šmídl-Quinn 2005, *The Variational Bayes Method in Signal Processing*, Springer):** approximate `p(x_k, θ | y_{1:k}) ≈ q(x_k) q(θ)` (mean-field) and minimise KL by coordinate ascent. Used when both state and parameters are unknown (joint state-parameter inference). Frontier; ship after PF infrastructure stabilises.
6. **Marginalized particle filter:** synonym of Rao-Blackwellized PF (Schön et al. 2005 use "marginalized"; Doucet-de-Freitas-Murphy-Russell 2000 use "Rao-Blackwellized").

## Concrete recommendations

### PR ordering (depends on slot 308 PR landing first)

**Block-D PR-309-A — `control/information_filter.go` (~400 LOC)**

API:
```go
type InformationFilter struct {
    Lambda []float64 // n×n information matrix (= P⁻¹)
    Eta    []float64 // n information vector (= P⁻¹ x̂)
    F      []float64 // n×n transition
    Qinv   []float64 // n×n process-noise *inverse* (precomputed once)
    n      int
    // scratch buffers (zero heap on hot path)
    tmpNN, tmpNN2, tmpNM []float64
}

func NewInformationFilter(n int, F, Qinv []float64) *InformationFilter
// Uninformative prior: Lambda = 0, Eta = 0 (no init needed)
func (f *InformationFilter) SetPrior(Lambda0, Eta0 []float64)
// Or initialize from covariance form
func (f *InformationFilter) SetFromCovariance(P, xhat []float64)

// Additive measurement update — sensor-i contributes (HᵀR⁻¹H, HᵀR⁻¹z)
func (f *InformationFilter) Update(H, Rinv, z []float64, m int)
// Or contribute information directly (distributed fusion)
func (f *InformationFilter) AddInformation(dLambda, dEta []float64)

// Time update — expensive: requires inverse (uses CholeskySolve internally)
func (f *InformationFilter) Predict()

// Recover (x̂, P) for downstream consumption
func (f *InformationFilter) Mean(xhatOut []float64)        // xhat = Λ⁻¹ η
func (f *InformationFilter) Covariance(Pout []float64)     // P = Λ⁻¹
```

Tightenings:
- `Predict()` MUST use `linalg.CholeskySolve` on Λ (PSD check is free, signals observability deficit). If `CholeskyDecompose(Lambda)` returns false, panic with "InformationFilter: Λ singular at predict step — system is unobservable in current basis; use SetFromCovariance with regularised P". This is the *exact* failure-mode signal a user needs.
- `AddInformation` is the **distributed-fusion entry point**: a remote sensor sends `(dLambda_i, dEta_i)` over the network; central node calls `AddInformation` once per sensor; **order independence** is a contract (must be tested — see #5 below).
- `Mean()` and `Covariance()` are **on-demand** — the recursion never computes them internally. Pistachio's 60 FPS rigid-body pose estimator with 4 cameras + IMU calls `AddInformation` 5× per frame and `Mean()` once.
- **All scratch buffers on struct** — `Predict()` does one `CholeskyDecompose + CholeskySolve` on Λ_+ (n³/3), one `MatMul` for `F Λ⁻¹ Fᵀ`, one `Inverse` of `(F Λ⁻¹ Fᵀ + Q)`. No heap allocations on hot path.

**Block-D PR-309-B — `prob/particle_filter.go` (~300 LOC)**

API matches slot 161 C11 spec (already approved); extends with three resampling schemes:
```go
type ParticleFilter struct {
    F func(x, u, w, out []float64)         // x_{k+1} = F(x_k, u, w)
    H func(x, out []float64)               // y_pred = H(x)
    LogLik func(y, ypred []float64) float64 // log p(y|x)
    N int
    State [][]float64
    Wts   []float64
    // scratch
    cumW, U []float64
    idx     []int
}

// Resampling schemes (all O(N), reuse `idx`):
func (pf *ParticleFilter) ResampleMultinomial(rng RandomSource)
func (pf *ParticleFilter) ResampleSystematic(rng RandomSource)  // RECOMMENDED default
func (pf *ParticleFilter) ResampleStratified(rng RandomSource)
func (pf *ParticleFilter) ResampleResidual(rng RandomSource)

// Conditional: only resample when ESS < N/2 (Liu-Chen 1995)
func (pf *ParticleFilter) EffectiveSampleSize() float64 // = 1 / Σw_i²
func (pf *ParticleFilter) Step(u, y []float64, rng RandomSource)
func (pf *ParticleFilter) Mean(out []float64)
func (pf *ParticleFilter) Covariance(out []float64)
```
Defer to slot 161 C11 + slot 265 P9 for the algorithmic spec; **slot 309 owns the resampling-scheme audit and the cross-validation pin against Kalman.**

**Block-D PR-309-C — `prob/rao_blackwellized_pf.go` (~350 LOC)**

For state `x = (x^l, x^n)` where `x^l` linear-Gaussian conditional on `x^n`:
```go
type RBParticleFilter struct {
    Fn func(xn, u, w, out []float64)            // nonlinear transition
    Fl, Hl, Ql, Rl []float64                    // linear-Gaussian sub-model (parametric in xn)
    LinearKF []control.KalmanFilter             // one KF per particle
    XN [][]float64                              // nonlinear samples
    Wts []float64
}
```
Variance reduction is the *raison d'être*: Schön-Gustafsson-Nordlund 2005 §V demonstrates 5-50× variance reduction over plain bootstrap PF on identical compute budget. Cite Doucet-de-Freitas-Murphy-Russell 2000 (UAI, *Rao-Blackwellised Particle Filtering for Dynamic Bayesian Networks*) as the algorithmic origin.

**Block-D PR-309-D — `prob/gaussian_sum_filter.go` (~250 LOC)**

Bank of `M` Kalman filters with normalised weights:
```go
type GaussianSumFilter struct {
    Components []control.KalmanFilter
    Weights []float64
    PruneThresh float64  // drop components with w < threshold
    MaxComponents int
}
```
Reuses slot-308 `KalmanFilter` for each component. Pruning + merging strategy (Salmond 1990 *Mixture Reduction Algorithms* — pairwise KL-distance merging): when `len(Components) > MaxComponents`, merge two closest pairs by Mahalanobis distance until under the cap. **Multi-modal benchmark:** same-sign-ambiguity bearings-only tracking (Alspach-Sorenson 1972 §V example) — KF/EKF mode-collapse, GSF tracks both peaks, PF tracks both peaks. Cite Alspach-Sorenson 1972 IEEE TAC AC-17(4):439-448 (DOI 10.1109/TAC.1972.1100034).

**Block-D PR-309-E — `prob/variational_bayes_filter.go` (~400 LOC, frontier — defer)**

Šmídl-Quinn 2005 mean-field: `q(x_k, θ) = q(x_k) q(θ)`, coordinate-ascent KL minimisation. Defer until PR-A through PR-D land and a real consumer requests joint state-parameter inference (Pistachio doesn't need this; Sentinel may).

### Tier ordering (refined from prompt's tier suggestion)

| Tier | Primitive | LOC | Depends on |
|------|-----------|-----|------------|
| T0 | Λ + η representation, conversion to/from (P, x̂) | 80 | linalg.CholeskyDecompose + CholeskySolve |
| T1 | Info-form measurement update (additive) | 60 | T0 |
| T2 | Info-form time update (with Cholesky-based inverse) | 100 | T0, T1 |
| T3 | `AddInformation` distributed-fusion API + ordering test | 80 | T1 |
| T4 | Bootstrap PF + multinomial resampling | 150 | prob.RandomSource (slot 156 P11) |
| T5 | Systematic + residual + stratified resampling | 100 | T4 |
| T6 | Rao-Blackwellized PF | 350 | T4, slot-308 KalmanFilter |
| T7 | Gaussian-sum filter (bank-of-Kalmans + merge) | 250 | slot-308 KalmanFilter |
| T8 | Variational-Bayes filter (mean-field, coordinate ascent) | 400 | T0, prob.KLDivergence |

T0+T1+T2+T3 = PR-A (~320 LOC; round up to 400 with tests). T4+T5 = PR-B (~250 LOC). T6 = PR-C. T7 = PR-D. T8 frontier.

### R-MUTUAL-CROSS-VALIDATION 3/3 saturator pins

**Pin #1 — Covariance-KF ≡ Info-KF on Linear-Gaussian benchmark (`control/kalman_consistency_test.go`):**
Run both filters on identical `(A, H, Q, R, x_0, P_0, [u_k], [y_k])` for 1000 steps. After every step, convert info-form to `(x̂_info, P_info) = (Λ⁻¹η, Λ⁻¹)` and assert:
- `‖x̂_cov − x̂_info‖_∞ ≤ 1e-10`
- `‖P_cov − P_info‖_F ≤ 1e-10`
This pins that the dual transformation is *exact*. **Two independent algorithms over the same recursion, agreeing to 10 digits.**

**Pin #2 — Info-form measurement update is associative AND commutative (`control/information_fusion_test.go`):**
Three sensors `(H_1, R_1, z_1), (H_2, R_2, z_2), (H_3, R_3, z_3)`. Run two filters from identical prior `(Λ_-, η_-)`:
- Filter A applies updates in order 1, 2, 3.
- Filter B applies updates in order 3, 1, 2 (any permutation).
Assert `‖Λ_A − Λ_B‖_F ≤ 1e-12 ∧ ‖η_A − η_B‖_∞ ≤ 1e-12`. This is the **distributed-fusion contract**. (For all 3! = 6 permutations.)

**Pin #3 — Particle filter (N=10⁴) ≡ Kalman filter on Linear-Gaussian (`prob/particle_filter_consistency_test.go`):**
Same Linear-Gaussian benchmark as Pin #1. Run KF and PF (Bootstrap, systematic resampling, fixed RNG seed). PF's posterior mean and covariance converge to KF's at rate `O(1/√N)`:
- N=100:  `‖x̂_pf − x̂_kf‖_∞ ≤ 0.5` (loose)
- N=1000: `‖x̂_pf − x̂_kf‖_∞ ≤ 0.15`
- N=10000: `‖x̂_pf − x̂_kf‖_∞ ≤ 0.05`
- N=10000: trace(P_pf) within 5% of trace(P_kf)
Three thresholds = three independent regression points. PF is the non-parametric oracle; KF is the parametric oracle for the LG sub-case; agreement saturates **R-MUTUAL-CROSS-VALIDATION 3/3** (KF ≡ info-KF ≡ PF) over the same Linear-Gaussian benchmark.

(Bonus pin available: GSF with M=1 component must equal KF; with M=10 components on a unimodal problem the merged posterior must equal KF to within 5%.)

### Cross-link consumers

- **Distributed sensor fusion (Pistachio):** rigid-body pose estimation with 4 cameras + IMU — each sensor produces `(dΛ, dη)` independently and asynchronously; info-form sums them on arrival without re-running the whole filter. Covariance-form requires sequential update or stacked measurement vector (much more expensive).
- **Graph-SLAM (future visual SLAM consumer):** the **square-root information matrix** Λ^(1/2) is the natural representation — block-sparse, factored over robot poses + landmarks; Lu-Milios 1997, Thrun-Burgard-Fox 2005 §11; iSAM (Kaess-Ranganathan-Dellaert 2008) and iSAM2 (Kaess et al. 2012) are entirely info-form.
- **High-DOF state estimation (n ≥ 100):** info-form's sparsity is preserved by additive measurement update (covariance form fills in the matrix). Critical for any large-state-vector estimator.
- **Robot localization / Monte Carlo localization (Thrun-Fox-Burgard-Dellaert 2001):** literally a particle filter — bootstrap PF with a known map model. Reality's PR-309-B is the entry point.
- **Multi-target tracking (Salmond 1990):** Gaussian-sum filter with one component per hypothesised target.
- **Joint state-parameter inference:** VB filter (slot-309-T8) when both `x_k` and parameters `θ` are unknown — Šmídl-Quinn 2005 Ch. 5.

### Risks / caveats

- **Λ⁻¹ at predict step** is an O(n³) matrix inverse; for n=12 (Pistachio pose) this is ~1500 flops, negligible. For n≥100 (SLAM), this becomes a hot-path concern — square-root info filter (SRIF, Bierman 1977 §VII) propagates `R` where `Λ = RᵀR`, avoiding the inverse. **Defer SRIF to v0.13.0** alongside square-root UKF (slot 308 PR-C).
- **PF curse of dimensionality:** N must scale exponentially with state dimension. PF at n≥10 needs RBPF (T6) or particle MCMC (slot 265). Document the n≤6 sweet spot for plain bootstrap PF.
- **Resampling RNG dependency:** all PF tests must use a deterministic `prob.RandomSource` (slot 156 P11 / slot 265 PR-0) — without it, golden files cannot be cross-validated against Python/C++.
- **`AddInformation` units bug-magnet:** users will pass `(H_i, R_i, z_i)` instead of `(Hᵢᵀ Rᵢ⁻¹ Hᵢ, Hᵢᵀ Rᵢ⁻¹ zᵢ)` — provide a helper `MeasurementToInformation(H, Rinv, z) (dLambda, dEta []float64)` and document the contract loudly. Type-system can't help (Go float64 slices); rely on naming + 1-line example.
- **GSF component-count explosion:** without pruning/merging, M doubles per update if each component splits. Hard cap `MaxComponents = 32` (Salmond 1990 cited) and merge by Mahalanobis distance.

## Sources

**Repo files (gap evidence):**
- `C:/limitless/foundation/reality/control/filter.go` — only deterministic filters
- `C:/limitless/foundation/reality/control/pid.go`, `transfer.go` — no state-space
- `C:/limitless/foundation/reality/prob/markov.go`, `timeseries.go` — no recursive Bayesian filter
- `C:/limitless/foundation/reality/prob/conformal/adaptive.go:160` — only `EffectiveSampleSize` (Kish, *not* SMC)
- `C:/limitless/foundation/reality/linalg/decompose.go:266 CholeskyDecompose`, `:316 CholeskySolve` — building blocks for Λ inverse

**Cross-references (other review agents):**
- `reviews/overnight-400/agents/308-dive-kalman-square-root.md` — slot 308: zero KF, recommend Joseph-form first; recommendation #4 sketches information filter as "v0.14.0 optional"; **slot 309 promotes that to PR-309-A**
- `reviews/overnight-400/agents/161-synergy-control-prob.md:236-248` — C11 Bootstrap PF spec (Gordon-Salmond-Smith 1993, ~250 LOC); slot 309 PR-B implements
- `reviews/overnight-400/agents/161-synergy-control-prob.md:274-276` — RBPF named as stretch primitive, "ships after C5+C11"; slot 309 PR-C implements
- `reviews/overnight-400/agents/165-synergy-sequence-prob.md:225-240` — `ParticleFilter` API sketch + ESS reuse from `prob/conformal/adaptive.go`
- `reviews/overnight-400/agents/265-new-pmcmc.md`, `266-new-smc.md` — own SMC/PMCMC parameter-inference + SMC-design axes; slot 309 stays in the *filtering-recursion* lane (Λ/η + bootstrap PF + RBPF + GSF + VB), defers tempering / look-ahead / data-assimilation to 265/266
- `reviews/overnight-400/agents/238-new-mcmc.md:127` — M17 SequentialMonteCarlo (parameter inference, separate axis)

**Foundational sources:**
- Anderson B.D.O. & Moore J.B. 1979, *Optimal Filtering*, Prentice-Hall (Dover reprint 2005, 0-486-43938-0) — Ch. 6 information filter, matrix-inversion lemma, dual formulation; Ch. 10 distributed/decentralised filtering. Canonical reference. (https://store.doverpublications.com/0486439380.html)
- Maybeck P.S. 1979, *Stochastic Models, Estimation, and Control* Vol. 1, Academic Press (Mathematics in Science and Engineering 141.1) — §5.7 inverse covariance / information form, §7 square-root filtering (SRIF). (https://www.cs.unc.edu/~welch/kalman/media/pdf/maybeck_ch1.pdf)
- Khan M.E. tutorial, *Matrix Inversion Lemma and Information Filter* (https://emtiyaz.github.io/Writings/MILandIF.pdf) — modern derivation, useful for tests
- Bierman G.J. 1977, *Factorization Methods for Discrete Sequential Estimation*, Academic Press / Dover 2006 §VII — Square-Root Information Filter (SRIF). Future v0.13.0.
- Gordon N.J., Salmond D.J. & Smith A.F.M. 1993, "Novel approach to nonlinear/non-Gaussian Bayesian state estimation", IEE Proc. F 140(2):107-113 — bootstrap particle filter origin. (https://www3.nd.edu/~lemmon/courses/ee67033/pubs/GordonSalmondSmith93.pdf)
- Doucet A., Godsill S. & Andrieu C. 2000, "On sequential Monte Carlo sampling methods for Bayesian filtering", Stat. Comput. 10:197-208 — unifying importance-sampling framework. (https://www.stats.ox.ac.uk/~doucet/doucet_godsill_andrieu_sequentialmontecarloforbayesfiltering.pdf)
- Doucet A., de Freitas N. & Gordon N. (eds.) 2001, *Sequential Monte Carlo Methods in Practice*, Springer — definitive PF book. (https://link.springer.com/book/10.1007/978-1-4757-3437-9)
- Liu J.S. & Chen R. 1998, "Sequential Monte Carlo methods for dynamic systems", JASA 93(443):1032-1044 — residual resampling, ESS-triggered resampling.
- Kitagawa G. 1996, "Monte Carlo filter and smoother for non-Gaussian nonlinear state space models", J. Comput. Graph. Stat. 5(1):1-25 — systematic resampling.
- Carpenter J., Clifford P. & Fearnhead P. 1999, "Improved particle filter for nonlinear problems", IEE Proc. Radar Sonar Navig. 146(1):2-7 — stratified resampling.
- Schön T., Gustafsson F. & Nordlund P.-J. 2005, "Marginalized Particle Filters for Mixed Linear/Nonlinear State-Space Models", IEEE TSP 53(7):2279-2289 — RBPF / marginalized PF canonical paper. (https://user.it.uu.se/~thosc112/research/rao-blackwellized-particle.html)
- Doucet A., de Freitas N., Murphy K. & Russell S. 2000, "Rao-Blackwellised Particle Filtering for Dynamic Bayesian Networks", UAI — RBPF algorithmic origin (uses "Rao-Blackwellised" naming).
- Alspach D.L. & Sorenson H.W. 1972, "Nonlinear Bayesian estimation using Gaussian sum approximations", IEEE TAC AC-17(4):439-448, DOI 10.1109/TAC.1972.1100034 — Gaussian-sum filter origin. (https://ieeexplore.ieee.org/document/1100034/)
- Sorenson H.W. & Alspach D.L. 1971, "Recursive Bayesian estimation using Gaussian sums", Automatica 7(4):465-479 — predecessor; cited by slot 266 S14.
- Salmond D.J. 1990, "Mixture Reduction Algorithms for Target Tracking in Clutter", SPIE 1305:434-445 — pairwise-merge GSF pruning strategy.
- Šmídl V. & Quinn A. 2005, *The Variational Bayes Method in Signal Processing*, Springer (Signals and Communication Technology), ISBN 978-3-540-28819-0 — VB filter book-length treatment. (https://www.amazon.com/Variational-Processing-Signals-Communication-Technology/dp/3540288198)
- Šmídl V. 2008, *Variational Bayesian Filtering*, IEEE TSP 56(10):5020-5030. (https://library.utia.cas.cz/separaty/2008/AS/smidl-variational%20bayesian%20filtering.pdf)
- Thrun S., Burgard W. & Fox D. 2005, *Probabilistic Robotics*, MIT Press, Ch. 3 (info-form Kalman) + Ch. 11 (graph-SLAM info-form).
- Kaess M., Ranganathan A. & Dellaert F. 2008, "iSAM: Incremental Smoothing and Mapping", IEEE T. Robotics 24(6):1365-1378 — square-root info-form for SLAM.
- Lu F. & Milios E. 1997, "Globally Consistent Range Scan Alignment for Environment Mapping", Autonomous Robots 4:333-349 — info-form pose-graph SLAM origin.
- Olfati-Saber R. 2007, "Distributed Kalman filtering for sensor networks", IEEE CDC — distributed info-form fusion via consensus. (https://ieeexplore.ieee.org/document/4434303/)
- Roy P. et al. 2019, "Quantitative Verification of Numerical Stability for Kalman Filters", FM 2019 — formal stability checking of Kalman variants. (https://www.prismmodelchecker.org/papers/fm19kf.pdf)
