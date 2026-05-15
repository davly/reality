# 161 | synergy-control-prob

**Topic:** control × prob — stochastic control, LQG, Kalman family, particle filter
**Block:** B (cross-package synergies)
**Date:** 2026-05-08
**Scope:** capabilities that emerge ONLY when `control/`, `prob/`, and `linalg/`
are composed; not what either is missing in isolation (covered by per-package
agents 056-060 / 116-120 / 071-075). Repo at v0.10.0, 1965 tests passing.

## Two-line summary

Today `control/` ships a stateful PID, three first-order filters, a rate
limiter, and a poles/stability TransferFunction (~490 LOC) — **zero
state-space objects, zero stochastic primitives, no observer**. `prob/`
owns scalar distributions, Markov chains, Bayesian update in log-odds,
ARIMA/Levinson-Durbin, and an embedded LCG — but no measurement-update
primitive. Sitting between them, `linalg/` already provides
LU/Cholesky/MatMul/MatVec/MatAdd/MatSub/Identity/Trace/QRAlgorithm — every
matrix kernel needed below. **Twelve synergy primitives (C1-C12) totalling
~2,100 LOC of glue** stand up the entire LQG/Kalman/PF stack on the
existing bases; cheapest first PR is **C1 `StateSpace` + `StateSpaceStep`**
at ~80 LOC because every other primitive consumes it, and the highest-
leverage one-day unlock is **C5 `KalmanFilter` (Joseph form)** at ~180 LOC
because consumers immediately get sensor fusion that PID + LowPassFilter
cannot deliver (PID has no measurement model, LowPass has no process model
— Kalman is the union).

---

## Bases — what each package exposes today

`control/` (~490 LOC, agents 056-060): `PIDController{Kp,Ki,Kd,clamp}`
scalar with anti-windup (`pid.go:36-122`); `LowPassFilter`,
`HighPassFilter`, `ComplementaryFilter`, `RateLimiter` (all scalar,
first-order, stateless — `filter.go`); `TransferFunction{Num,Den}` with
`Evaluate(s)`, Durand-Kerner `Poles()`, `IsStable()` (Re(pole)<0,
continuous-time only — no `c2d`, no Bode/Nyquist data). **Absent:**
state-space `(A,B,C,D)`, controllability/observability, Lyapunov/Riccati,
any discrete-time filter, any stochastic model, any observer.

`prob/` (~3,500 LOC, agents 116-120): `Distribution` interface
(`Beta/Normal/Exponential/Uniform`); `Normal/Beta/Gamma/Exponential` PDF/
CDF + `NormalQuantile` (Acklam); discrete `Poisson/Binomial`;
`BayesianUpdate` (scalar log-odds only); `MarkovSteadyState/Simulate`
with embedded LCG; `ExponentialSmoothing/HoltLinear/ARIMA` + private
`levinsonDurbin`; `LinearRegression` (closed-form OLS, no recursive);
`ChiSquaredTest/TTestOneSample/TTestTwoSample` and **private**
`chiSquaredCDF/studentTCDF` (`mathutil.go`). **Absent:** any sampler
beyond MarkovSimulate's LCG, multivariate Gaussian, resampling,
publicised information-theoretic helpers.

`linalg/` (load-bearing): `MatMul`, `MatTranspose`, `MatVecMul`, `MatAdd`,
`MatSub`, `MatScale`, `Identity`, `Trace`, `LUDecompose`, `LUSolve`,
`Inverse`, `Determinant`, `CholeskyDecompose`, `CholeskySolve`,
`QRAlgorithm` (symmetric eigensolve). **Absent:** Schur form, matrix
sign function, matrix square root via Denman-Beavers — needed for
classical CARE/DARE Schur solvers (we route around with doubling).

Neither `control/` nor `prob/` imports the other today. There is no
`kf/`, `lqg/`, `mpc/` glue file. `aicore` is the consumer (CLAUDE.md
dependency diagram).

## The conceptual unlock — separation principle

The synergy-defining theorem is the LQG separation principle (Joseph &
Tou 1961, Wonham 1968): for the linear system
`x_{k+1} = Ax_k + Bu_k + w_k`, `y_k = Cx_k + v_k` with `w~N(0,Q)`,
`v~N(0,R)`, the optimal LQG controller is the **cascade** of (a) the
LQR feedback gain `K_LQR` solving DARE on `(A,B,Q_state,R_input)`
applied to (b) the state estimate from the discrete Kalman filter
`(A,C,Q,R)` — designed independently. That decoupling is why `control/`
and `prob/` *deserve* to compose despite living in different packages:
stochastic control = LQR(`control`) ∘ KF(`prob` × `linalg`), and the
Kalman gain `K_KF = PCᵀ(CPCᵀ+R)⁻¹` is literally a multivariate
Bayesian posterior update of `N(x̂,P)` under Gaussian likelihood
`N(Cx,R)` — the multivariate generalisation of `BayesianUpdate`
(`prob.go:85`). The synergies below are unusually shallow because the
math is one bridge primitive (multivariate Gaussian update) plus one
wrapper struct (state-space) and the rest is composition.

---

## C1 — `StateSpace` struct + `StateSpaceStep` (foundation, ~80 LOC)

The single missing object that gates ten of twelve below.

```go
type StateSpace struct {
    A, B, C, D []float64   // dims n×n, n×m, p×n, p×m (row-major)
    N, M, P    int
    Dt         float64     // sample period; 0 = continuous (rejected by Step)
}
func (s *StateSpace) Step(x, u, xNext, y []float64)   // xNext=Ax+Bu, y=Cx+Du
```

Uses `linalg.MatVecMul` twice/call. Zero allocations. Drops cleanly into
60 FPS budget. **Saturation:** Tustin round-trip of any
`TransferFunction` of order ≤4 → `StateSpace` → impulse response
matches `signal.IFFT` of `Evaluate` on unit circle (max err <1e-9 for
stable poles).

## C2 — `Discretize` / `Continuize` (c2d / d2c, ~220 LOC)

Bridge from continuous `TransferFunction` to discrete `StateSpace`.
Three methods: **ZOH** (matrix exp of `[[A,B],[0,0]]` via Padé(6,6),
13 MatMul, exact for piecewise-constant inputs, ~120 LOC); **Tustin**
(`Ad=(I+Acdt/2)(I−Acdt/2)⁻¹` etc — the *only* method preserving
stability across c2d, LHP→unit disc, ~70 LOC, single `LUSolve`);
**Forward Euler** (~10 LOC; warn it can destabilise). API:
`tf.ToStateSpace()`, `ZOHDiscretize(...)`, `TustinDiscretize(...)`.
**Saturation:** `1/(s+1)` → ZOH at dt=0.1 → discrete pole
`exp(-0.1)≈0.9048` to ≤1e-12.

## C3 — `Controllability` & `Observability` matrices + rank test (~120 LOC)

Without these, LQR may yield a controller that physically cannot move
the system. `Co=[B AB A²B … A^{n-1}B]` controllable iff rank `Co=n`;
`Ob=[C; CA; …; CA^{n-1}]` observable iff rank `Ob=n`. API:
`ControllabilityMatrix`, `ObservabilityMatrix`, `IsControllable`,
`IsObservable`. Rank check via thresholded SVD would be cleaner but
linalg has no SVD; fallback is `Inverse`-returns-false (=singular) on
the n×n leading minor plus QR-style column pivoting. **Saturation:**
integrator chain `[[0,1],[0,0]],[[0],[1]]` → `Co=[[0,1],[1,0]]` rank 2.

## C4 — `Lyapunov` solvers (~100 LOC)

`AᵀPA−P+Q=0`. Required by DARE (C6) and stochastic stability (C12).
Skip Bartels-Stewart (needs Schur). Use **Smith doubling** (Smith 1968):
`X₀=Q, A₀=A; X_{k+1}=X_k+A_kᵀX_kA_k; A_{k+1}=A_k²`. Two MatMul + one
MatAdd per step. Quadratic convergence: 20 iterations gives 1e-12 for
`‖A‖<0.9`. API: `DiscreteLyapunov`, `ContinuousLyapunov`.

## C5 — `KalmanFilter` (Joseph stabilised form) — flagship (~180 LOC)

The **central synergy**. Joseph form is mandatory: naïve
`P=(I−KC)P_pred` loses positive-definiteness after ~50 updates under
floating-point error — exactly the regime IMU/GPS fusion lives in.

```go
type KalmanFilter struct {
    SS  *StateSpace
    Q, R, X, Cov []float64               // process/meas cov, posterior mean & cov
    // scratch buffers (no alloc on hot path)
}
func NewKalmanFilter(ss *StateSpace, Q, R, x0, P0 []float64) *KalmanFilter

func (kf *KalmanFilter) Predict(u []float64)                 // x̂=Ax̂+Bu;  P=APAᵀ+Q
func (kf *KalmanFilter) Update (y []float64) []float64       // returns innovation
//   S = CPpredCᵀ + R                                        (CholeskySolve on S)
//   K = Ppred Cᵀ S⁻¹
//   x̂ = x̂pred + K(y − Cx̂pred)
//   P  = (I−KC) Ppred (I−KC)ᵀ + K R Kᵀ                      (Joseph)
```

Pure composition: `linalg.MatMul/MatVecMul/MatAdd/MatSub/Identity/
CholeskySolve` + `StateSpace`. Joseph form costs three extra MatMul vs
naïve — worth it. **Saturation:** ground-truth random walk
`x_{k+1}=x_k+w_k`, scalar measurement `y_k=x_k+v_k`; after 1000 steps
the steady-state Kalman gain matches the algebraic Riccati solution to
≤1e-10 and posterior `P` stays positive (smallest eigenvalue via
`linalg.QRAlgorithm` >0 across all 1000 timesteps).

This single primitive unlocks: IMU/GPS fusion (Pistachio's
`ComplementaryFilter` is *strict subset* of KF under degenerate `Q,R`),
SLAM-style estimation, recursive least squares (degenerate KF with
`A=I, Q=0`).

## C6 — `DARE` → `LQR` (~155 LOC)

`P=AᵀPA−AᵀPB(R+BᵀPB)⁻¹BᵀPA+Q`, `K=(R+BᵀPB)⁻¹BᵀPA`. Solver: **doubling
iteration** (Anderson-Moore 1979) — no Schur, 6 MatMul/step, quadratic
convergence. ~130 LOC `DARE` + 25 LOC `LQR` returning `K` and
steady-state cost `tr(Px₀x₀ᵀ)`. API: `DARE`, `LQR`, `FiniteHorizonLQR`
(back-recurses Riccati from terminal `Qf`). **Saturation:** double
integrator `A=[[1,dt],[0,1]],B=[[0],[dt]],Q=I,R=1` → closed-loop poles
strictly inside unit disc (verify via `IsStable` on `A−BK` extracted
with `QRAlgorithm`).

## C7 — `LQG` controller (~60 LOC)

Trivially `LQR ∘ KalmanFilter`. Separation principle says they're
designed independently, but 99% of users want a single object consuming
`(y, ref)` and emitting `u`. API: `NewLQG(ss, Q_proc, R_meas, Q_state,
R_input, x0, P0)`, `(c *LQGController) Step(y, u)` does
Predict → Update(y) → `u=-K(x̂-Ref)`. For tracking, offset via
`u=-K(x̂−x_ref)+u_ref`; document but don't compute `u_ref` (requires
solving `[A−I,B; C,D][x_ref;u_ref]=[0;r]` which the user knows how to
do with `LUSolve`).

## C8 — Extended Kalman Filter (~150 LOC)

Mildly nonlinear systems with Jacobians `F=∂f/∂x|x̂`, `H=∂h/∂x|x̂`
recomputed each step. **Synergy:** `autodiff.Tape` (agent 011) emits
both `f` and `F` from a single user function — no finite differences,
no analytic-gradient drudgery. EKF replaces `A,C` in `KalmanFilter`
with `F,H` and re-uses Joseph update from C5 verbatim — only `Predict`
differs (state propagation by `f`, covariance by `FPFᵀ+Q`). API:
`ExtendedKalmanFilter{F, H func(x, u, fOut, jacOut []float64), Q, R,
X, Cov}`. **Caveat:** EKF can diverge for strong nonlinearity;
divergence test is `χ²innov > thresh` for k consecutive steps (C9).

## C9 — Innovations covariance test (white residuals, ~100 LOC)

The single most underrated KF diagnostic. If filter is correctly tuned,
innovations `ν_k=y_k−Cx̂_{k|k-1}` are zero-mean white with covariance
`S_k=CP_{k|k-1}Cᵀ+R`. Two tests:

1. **NIS** (Bar-Shalom 2001): `ε_k=ν_kᵀS_k⁻¹ν_k ~ χ²_p`. Sliding window
   of 30 NIS values tested against `χ²_{30p}` flags `Q,R` mistuning at
   p<0.05 (uses publicised `prob.ChiSquaredCDF`).
2. **Whiteness:** sample autocorrelation of `ν_k` at lag 1..L should be
   inside `±1.96/√N` (uses publicised `prob.LjungBoxQ`, bridge primitive
   shared with 154/165).

```go
func (kf *KalmanFilter) NIS() float64
func InnovationsConsistencyTest(innov [][]float64, S [][]float64, p int) (chi2, pValue float64)
```

+~20 LOC to publicise `chiSquaredCDF` (also requested by 151).
**Saturation:** inflate `R` 10× → NIS drops to ~0.1 → mistuning detected
(p<0.001) within 30 samples; deflate → NIS climbs to ~10 → also detected.

## C10 — Unscented Kalman Filter (Julier-Uhlmann 1997, ~220 LOC)

Strongly nonlinear `f,h` where Jacobians are nasty/undefined: UKF
propagates `2n+1` deterministic sigma points through `f,h` unchanged
then rebuilds mean/cov. Matrix square root uses Cholesky of `(n+λ)P`
— already in `linalg.CholeskyDecompose`. API:
`UnscentedKalmanFilter{F, H func(x, u, out), Q, R, X, Cov,
Alpha=1e-3, Beta=2, Kappa=0}`. Cleaner than EKF for users without a
clean Jacobian (most roboticists). **Saturation:** range-bearing
benchmark (Julier 1997 §VI), UKF posterior cov differs from MC truth
by <5%, EKF by ~30% — same code-path comparison.

## C11 — Bootstrap Particle Filter / SIR (Gordon-Salmond-Smith 1993, ~250 LOC)

Arbitrary `f, h` and arbitrary noise (not Gaussian): predict
`x_k^{(i)}~f(x_{k-1}^{(i)},u)+w^{(i)}`; weight `w_k^{(i)}∝p(y_k|x_k^{(i)})`;
normalise; resample (systematic) if `N_eff=1/Σw²<N/2`. Needs two new
prob bridges: `BoxMullerSample(rng) (z1,z2)` (~15 LOC, Gaussian process
noise) and `SystematicResample(weights, rng, indices)` (~30 LOC). Both
belong in `prob/sample.go`, useful far beyond PF (MC integration,
bootstrap CI, dropout). API: `ParticleFilter{F, H, N, State, Wts}`,
`Step(u, y)`, `Mean(out)`, `EffectiveN()`. **Saturation:** 1-D random
walk with `t(ν=3)` measurement noise — KF underperforms PF by 2× MSE;
PF matches Bayes-optimal posterior (offline grid) within MC error.
O(N·n)/step; allocates once at construction.

## C12 — Stochastic Lyapunov stability (Kushner 1967, ~30 LOC)

Discrete `x_{k+1}=Ax_k+w_k` with `E[wwᵀ]=Q` is **mean-square stable**
iff `∃P>0` solving `AᵀPA−P+Q=0` (the discrete Lyapunov of C4). Ship
`IsMeanSquareStable(A, Q []float64, n int) bool` — solves C4 then
verifies `P−Q>0` via Cholesky-success on `P−Q`. Discrete analogue of
`TransferFunction.IsStable` for stochastic systems; closes the symmetry
gap between the two packages.

---

## Stretch primitives (deferred, future agents)

- **H∞ controller** — game-theoretic worst-case noise; modified Riccati
  `AᵀPA−P−AᵀPB[R,0;0,−γ²I+DᵀPD]⁻¹BᵀPA+CᵀC=0`. ~400 LOC; unlocks robust
  control. Defer.
- **MPC with chance constraints** — couples synergy with `optim.Simplex`/
  `LBFGS`. Owned by 163 + future control-optim synergy.
- **SDE/HJB control** — needs `chaos.SDESolver` (Euler-Maruyama) which
  doesn't exist yet (154 candidate). Defer until then.
- **KL control / Todorov 2006** — passive dynamics + KL cost; reduces to
  eigenvalue problem on `linalg.QRAlgorithm`. ~150 LOC; defer.
- **Risk-sensitive (exp-utility) LQR** — Whittle 1981; tweak on DARE
  with parameter θ; ~30 LOC delta from C6; ship after C6.
- **Rao-Blackwellized PF** — partitions state into linear/nonlinear,
  runs KF on linear part conditional on nonlinear samples. ~300 LOC,
  ships after C5+C11.
- **Continuous-time CARE** — ZOH-discretise then DARE is the standard
  2026 trick (Laub 1991 needs Schur). Defer; C2+C6 cover 95% of users.

---

## Cost / value table

| ID  | Primitive                    | LOC  | Depends on                  | Unblocks                         |
|-----|------------------------------|------|-----------------------------|----------------------------------|
| C1  | StateSpace + Step            |  80  | linalg.MatVecMul            | C2-C12                           |
| C2  | c2d/d2c (ZOH/Tustin/Euler)   | 220  | C1, linalg.LUSolve          | continuous-plant Kalman          |
| C3  | Controllability/Observability| 120  | C1, linalg                  | LQR sanity check                 |
| C4  | Discrete/cont Lyapunov       | 100  | linalg.MatMul               | C6, C12                          |
| C5  | KalmanFilter (Joseph)        | 180  | C1, linalg.CholeskySolve    | sensor fusion (Pistachio IMU/GPS)|
| C6  | DARE + LQR                   | 155  | C4, linalg.LUSolve          | LQG, finite-horizon control      |
| C7  | LQG cascade                  |  60  | C5 + C6                     | optimal stochastic control       |
| C8  | EKF                          | 150  | C5, autodiff (optional)     | nonlinear estimation             |
| C9  | NIS / innovations test       | 100  | C5, prob.ChiSquaredCDF*     | filter health diagnostic         |
| C10 | UKF                          | 220  | C5, linalg.Cholesky         | strong-nonlinearity estimation   |
| C11 | Particle filter (SIR)        | 250  | prob.BoxMuller* + Resample* | non-Gaussian estimation          |
| C12 | Mean-square stability        |  30  | C4 + linalg.Cholesky        | stochastic stability certificate |
|     | **Total**                    |**~1,665** | bridges marked * (~65 LOC) |                          |

Plus three small **bridge** primitives in `prob/` (publicising
`ChiSquaredCDF`, adding `BoxMullerSample`, adding `SystematicResample`)
amounting to **~65 LOC**. **Grand total ~2,100 LOC of glue** turns six
existing functions (PID + 3 filters + RateLimiter + TransferFunction)
and ~30 prob distributions/tests into the entire stochastic-control +
filtering stack.

## Recommended PR sequence

1. **C1** (½ day, 80 LOC) — gates everything; ship a `StateSpaceStep` test
   round-tripping the existing `TransferFunction(1/(s+1))` via Tustin.
2. **C5** (1 day, 180 LOC) — flagship; replaces `ComplementaryFilter`
   and gives Pistachio strict-upgrade path.
3. **C9** (½ day, 100 LOC) — pairs with C5; without it KF is a footgun.
4. **C4 + C6 + C7** (2 days, 315 LOC) — LQG triplet shipped together.
5. **C2** (1 day, 220 LOC) — once LQG is real, c2d is the on-ramp from
   `TransferFunction`-style designs.
6. **C8/C10/C11** (3-5 days, 620 LOC) — nonlinear/non-Gaussian extensions.
7. **C3, C12** (½ day, 150 LOC) — small certificates; ship anytime.

Two-week wall-clock total to land all twelve, paced one PR per ½-day,
all changes additive (no API breaks to existing PID/TransferFunction).

## Cycle-hazard resolution

Place all 12 primitives in NEW package `control/stochastic/`
(`statespace.go`, `discretize.go`, `observability.go`, `lyapunov.go`,
`kalman.go` for C5+C9, `lqr.go`, `lqg.go`, `ekf.go`, `ukf.go`,
`particle.go`, `stability.go`) plus 65 LOC in `prob/sample.go`
(BoxMuller+SystematicResample) and `prob/mathutil.go` publicisation
(ChiSquaredCDF) since `control/` cannot import `prob/` without breaking
the sibling-zero-dep invariant. `control/stochastic/` → `control + prob
+ linalg + constants` is a one-way DAG. Matches 151/153/154/155/156/
157/158/159/160 consumer-side-placement precedent (10th consecutive
synergy review confirming this pattern).

## Anti-overlap notes

- **vs 151 (signal × prob):** owns spectral / Whittle / Lomb-Scargle.
  KF whiteness diagnostic shares `LjungBoxQ` request — coordinate
  bridge primitive ownership in `prob/timeseries.go`. Shares
  `ChiSquaredCDF` publicisation request.
- **vs 154 (chaos × timeseries):** owns SDE / HJB / state-space chaos;
  defer SDE control to that synergy. Levinson-Durbin export shared.
- **vs 162 (graph × prob):** disjoint — `MarkovSteadyState` there is
  graph-theoretic, here it is dynamics-theoretic.
- **vs 163 (optim × autodiff):** EKF Jacobians via autodiff is the
  only cross-touch; mention but don't claim ownership.

## Saturation witnesses (R-MUTUAL-CROSS-VALIDATION 3/3 pins)

Each primitive ships with ≥20 golden-file vectors per CLAUDE.md plus
one or more cross-validation pins:

1. **Random-walk KF** — steady-state Kalman gain matches algebraic
   Riccati to ≤1e-10 + posterior P stays positive across 1000 timesteps
   via `QRAlgorithm` minimum-eigenvalue cross-check.
2. **Double-integrator LQR** — closed-loop poles inside unit disc via
   three independent paths: DARE-derived `A−BK` extracted via
   `QRAlgorithm` + `IsStable` on the discretised TransferFunction +
   offline grid-search of cost minimum.
3. **Range-bearing UKF** — UKF posterior covariance differs from MC
   truth by <5%, EKF by ~30% on Julier 1997 §VI benchmark — same
   code-path comparison establishes UKF strictly better than EKF.
4. **NIS deflation/inflation** — inflate R 10× → NIS drops to ~0.1 +
   χ²-test rejects null + mistuning detected (p<0.001) within 30
   samples; deflate symmetric — mirrors recent commits 6a55bb4
   audio-onset 3-detector and 365368a copula×autodiff Clayton
   log-PDF gradient saturation patterns.

## Bottom line

`reality` is unusually well-positioned for stochastic control because
`linalg/` already ships every matrix primitive Kalman/LQR needs and
`prob/` ships every distribution Bayesian estimation needs — the only
missing pieces are the **bridge struct** (`StateSpace`) and the
**multivariate Gaussian update primitive** (Joseph KF). Twelve glue
files at ~2,100 LOC turn the current scalar PID library into a full
LQG/EKF/UKF/PF stack — a ~4× capability multiplier for ~5× current
`control/` LOC, paid for entirely by composition rather than new
mathematics. The separation principle is doing the work; we just
need to write it down.
