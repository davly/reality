# 191 | synergy-chaos-control

**Topic:** chaos × control — UPO stabilisation (OGY), Pyragas time-delay
feedback, chaos synchronisation, reservoir computing, targeting.
**Block:** B (cross-package synergies). **Date:** 2026-05-08.
**Scope.** Capabilities that emerge ONLY when `chaos/`, `control/`, and
`linalg/` compose. Per-package isolation gaps already captured by 026–030
(chaos), 051–055 (control), 096–100 (linalg); this report is **disjoint**
from those. Closely related but disjoint siblings: **154** (chaos × timeseries:
Takens/Rosenstein/RQA), **161** (control × prob: Kalman/EKF/UKF), **186**
(graph × control), **187** (orbital × control), **168** (physics × autodiff).

## Two-line summary

`chaos/` ships forward-only RK4+Lorenz/Rossler/VanDerPol+1-D-Lyapunov and
`control/` ships PID + transfer-function poles + first-order filters; **the
seam — closing the loop on a chaotic plant to stabilise an unstable periodic
orbit — is one Jacobian helper and one delay buffer away from working, and
reality currently ships zero of the eight closed-loop chaos primitives (OGY,
Pyragas, ETDAS, Pecora-Carroll, generalised sync, targeting, reservoir-computing,
sensitivity-control)**. Twelve synergy primitives totalling ~1180 LOC of pure
glue stand up the canonical Boccaletti-Grebogi-Lai-Mancini-Maza
control-of-chaos surface; cheapest first PR is **S1 NumericalJacobian** (~50
LOC) because every UPO/sync/spectrum synergy depends on it, and the
highest-leverage one-day unlock is **S6 Pyragas time-delay feedback** (~80
LOC: the closed-loop variant of `control.LowPassFilter`'s state model with a
ring buffer) which gives a model-free chaos suppressor that needs no UPO
knowledge and composes directly with `RK4Step` / `PIDController.Update`.

---

## 1. Bases — what each side ships today

### 1.1 `chaos/` (461 LOC, agents 026–030)

`RK4Step`, `EulerStep`, `SolveODE` (open-loop integrators);
`LorenzSystem`, `RosslerSystem`, `VanDerPol`, `LotkaVolterra`, `SIRModel`,
`LogisticMap`, `GameOfLife` (derivative-builders / scalar maps);
`LyapunovExponent(f, x0, n)` — **1-D map only**, numerical `f'`;
`BifurcationDiagram(f, …)`, `RecurrencePlot(traj, ε)`. **Open-loop. No
Jacobian helper, no UPO finder, no tangent-space integrator, no state-object.**

### 1.2 `control/` (490 LOC, agents 051–055)

`PIDController.Update`, `TransferFunction.Evaluate/Poles/IsStable`
(continuous-time LTI, Durand-Kerner), `LowPassFilter`, `HighPassFilter`,
`ComplementaryFilter`, `RateLimiter`. **SISO, linear, no state-space type,
no observer, no LQR, no delay-line.**

### 1.3 `linalg/` (847 LOC, agents 096–100)

Matrix arithmetic, `LUDecompose/Solve`, `Inverse`, `Determinant`,
`CholeskyDecompose/Solve`, **`QRAlgorithm` is symmetric-only**
(Householder + tqli), `PCA`. Plenty for everything below except
**non-symmetric eigendecomposition**, the structural blocker — see §3.

### 1.4 The seam — what is missing on the boundary

Control-of-chaos primitives partition into **(A) UPO-aware closed-loop**
(OGY family), **(B) UPO-agnostic closed-loop** (Pyragas family),
**(C) coupling-based synchronisation** (Pecora-Carroll family), **(D)
data-driven model-free** (reservoir / SINDy). Reality ships none. Each is
modest glue:

- (A) needs Jacobian + Poincaré-section + 2×2 closed-form eigensolve (or §3);
- (B) needs a delay-line ring buffer (one `[]float64` of length τ/dt);
- (C) needs co-stepped derivatives + a coupling matrix;
- (D) needs reservoir update + ridge regression via `linalg.CholeskySolve`.

Every glue step uses primitives that already ship.

---

## 2. The twelve synergy primitives (ranked by leverage)

### S1. `NumericalJacobian(f, y, t, J)` — the foundation

**Capability.** Compute the n×n Jacobian ∂f_i/∂y_j of a `chaos.`-style
derivative function by central finite-difference. **Every** UPO method (S2/S3),
multi-D Lyapunov spectrum (S5), Pecora-Carroll auxiliary-system test (S9),
and bifurcation co-dimension analysis depends on this single helper.

**Composition.** For `j ∈ 0..n-1`, save `y[j]`, perturb ±ε, call `f` twice,
fill column j of `J` from `(dydt₊ − dydt₋)/(2ε)`. Step
`ε = √macheps · max(1, |y[j]|)`. Stride into a flat `J []float64` of length
n² consistent with linalg's row-major convention. **LOC ~50.** Lives in
`chaos/jacobian.go`.

### S2. `OGYStabilize` — Ott-Grebogi-Yorke (1990)

**Capability.** Stabilise an unstable periodic orbit of a discrete map `M_p`
by tiny parameter perturbations `δp_n = −Kᵀ(x_n − x*)` whenever the
trajectory enters a small neighbourhood of `x*`, with K projecting along the
unstable eigenvector.

**Composition.** (1) Find `x*` by Newton on `M_p(x) − x = 0` (compose
`linalg.LUSolve` + S1; multi-D Newton missing — see 097/102). (2) Jacobian
`DM(x*)` via S1. (3) Eigendecomposition: until §3 lands, restrict to **2×2**
where eigenvalues fall out of `tr(DM)`/`det(DM)` in closed form (Hénon, baker,
standard map — covers the canonical OGY pedagogy). (4) Bi-orthogonal left
eigenvector via one inverse-iteration step (`linalg.LUSolve`). (5) Control law
`δp = (λ_u/(λ_u−1)) · f_uᵀ(x_n − x*) / g`, with `g = ∂M/∂p` from one extra
S1-style perturbation along p. Reference: Ott-Grebogi-Yorke, *PRL* **64** 1196.

**LOC ~140.** Caps at 2×2 without §3.

### S3. `OGYTargeting` — Shinbrot-Ott-Grebogi-Yorke (1990)

**Capability.** Drive a chaotic trajectory from arbitrary IC into the
δ-neighbourhood of `x*` *before* engaging S2; without it, time-to-control
scales with 1/(basin measure).

**Composition.** Finite-horizon parameter search via
`optim.GoldenSectionSearch` on `p ↦ |M_p(x_n) − x*|²` each step until
`|x − x*| < δ`, then hand off to S2. Reuses S2. **LOC ~60.**

### S4. `PoincareSection(f, hyperplane, normal, dir)`

**Capability.** Convert a continuous flow into a discrete return map by
hyperplane-crossing detection — the canonical reduction OGY and Floquet
analysis assume.

**Composition.** Wrap `RK4Step`. After each step, check sign of
`(y_new − p)·n` vs previous; on sign change matching `dir`, switch the
integration variable from `t` to `y·n` (Hénon's 1982 trick) and take one
final step landing exactly on the hyperplane. Returns successive crossings
as the iterates of the discrete map. **LOC ~90.**

### S5. `LyapunovSpectrum(f, y0, n, dt, λ_out)` — Benettin (1980)

**Capability.** Estimate the **full** spectrum λ₁ ≥ … ≥ λₙ of a continuous
flow via QR-renormalisation, not just the largest scalar exponent. The
single most-cited gap in agent 026's numerics review.

**Composition.** Augment state from `y` (length n) to `(y, Q)` (length n+n²),
co-evolve `dy = f(t,y)` and `dQ = J(t,y)·Q` (J via S1); every k steps, QR-
decompose Q, accumulate `log(diag R)/(k·dt)` per direction, set Q ← Q'.
Symmetric `linalg.QRAlgorithm` is **wrong tool** here (footgun docstring
warning needed); inline modified Gram-Schmidt (~25 LOC) suffices for n ≤ 10.
**LOC ~140.** Golden vectors: Lorenz {0.91, 0, −14.57}, Rossler {0.071, 0,
−5.39}, Hénon {0.42, −1.62}.

### S6. `PyragasFeedback(K, τ)` — time-delay feedback (1992) — **highest-leverage**

**Capability.** Stabilise a UPO of period τ **without** knowing it by
applying continuous feedback `u(t) = K·(x(t) − x(t−τ))`. On any τ-periodic
orbit the feedback vanishes — never distorts the target dynamics, only
suppresses deviations. Removes OGY's "must know the UPO" prerequisite.
Reference: Pyragas, *Phys. Lett. A* **170** 421 (1992) — 7800+ citations.

**Composition.** Stateful struct holding ring buffer
`history [N][]float64` of size N = τ/dt; `Step` looks up `y_delayed`,
computes `u = K·(y − y_delayed)` via one `linalg.MatVecMul`, subtracts from
`dydt`, pushes `y` into the ring. Caller wraps their derivative in
`g(t,y,dydt) {f(t,y,tmp); pyr.Step(t,y,tmp,dydt,dt)}` and integrates `g`
with `RK4Step`. **LOC ~80.** Self-contained — ships before S1.

**Why this is the killer first PR.** (a) No Jacobian, no linearisation, no
UPO knowledge; (b) closes the loop entirely with primitives already
shipped; (c) the 1992 paper anchors the entire 1995–2010 control-of-chaos
literature. Pistachio's particle-NPC chaos-suppression use-case lands today.

### S7. `ExtendedTimeDelayFeedback (ETDAS)` — Socolar-Sukow-Gauthier (1994)

**Capability.** Geometric-series extension of S6:
`u(t) = K·Σ_{k=0}^∞ R^k(x(t−kτ) − x(t−(k+1)τ))`, lifting S6's odd-number
limitation for many UPOs (limit clarified by Fiedler-Flunkert-Georgi-Hövel-
Schöll 2007).

**Composition.** Reuses S6's ring buffer (depth K_max·N, ~6 typical) plus
one geometric accumulator `S(t) = (1−R)·x(t) + R·S(t−τ)`. **LOC ~50** on
top of S6.

### S8. `PecoraCarrollSync(driver_f, response_f, h)` — chaos sync (1990)

**Capability.** Couple two chaotic systems so the response asymptotically
tracks the driver. The foundational 1990 result; ~10 000 citations.

**Composition.** Two `chaos.`-style derivatives of identical state dimension;
a coupling **mask** `h []bool` (cascade replacement: where `h[i]==true`,
response state is replaced by driver state each step) **or** a coupling
matrix C with `dy_r = f_r(y_r) + C·(y_d − y_r)` (continuous coupling, with
the conditional-Lyapunov-exponent stability test composing with S5). Returns
sync-error norm via `linalg.L2Norm(driver − response)`. **LOC ~70.**

### S9. `GeneralizedSync` (Rulkov 1995) / Auxiliary-System Test (Abarbanel 1996)

**Capability.** Detect generalised sync `y_r = ψ(y_d)` when identical sync
fails. Spawn two response copies with different ICs, drive both from same
`y_d`, test `‖y_r − y_r'‖ → 0`.

**Composition.** Two parallel S8 instances sharing driver and coupling.
**LOC ~30** on top of S8.

### S10. `LagSync` / `PhaseSync` — Rosenblum-Pikovsky-Kurths (1996/1997)

**Capability.** Two weaker forms of sync: phase-locking
`|m·φ_d − n·φ_r| < c` even when amplitudes are uncorrelated; lag
`y_r(t) ≈ y_d(t−τ)`.

**Composition.** Phase: instantaneous phase via `signal.HilbertTransform`
(already ships per 131); phase-difference modulo 2π histogrammed via
`prob.Histogram`. Lag: cross-correlation peak via time-shifted dot products.
**LOC ~60**, requires S8.

### S11. `EchoStateNetwork` — reservoir computing (Maass-Jaeger 2002, Pathak et al. 2018)

**Capability.** Train a fixed-random recurrent "reservoir" to predict
chaotic trajectories. Pathak-Hunt-Girvan-Lu-Ott (*PRL* **120** 024102,
2018) showed an ESN trained on Lorenz outperformed every other published
forecaster up to 8 Lyapunov times. Pure linear regression on the readout —
no backprop, no gradient.

**Composition.**
1. Reservoir update `r_{t+1} = (1−α)·r_t + α·tanh(W_in·u_t + W·r_t)` with
   sparse random W rescaled to spectral radius ρ < 1 (certify via
   `linalg.QRAlgorithm` on `WᵀW` ⇒ singular values ⇒ `ρ(W) ≤ σ_max(W)`).
   Two `linalg.MatVecMul` calls + elementwise tanh per step.
2. Train: collect states R (N×T); regress
   `W_out = U·Rᵀ·(R·Rᵀ + λI)⁻¹` via `linalg.CholeskySolve` on the n×n
   positive-definite system — exactly what Cholesky was built for.
3. Predict: closed-loop feedback `u_t ← W_out·r_t`.

The "edge of chaos" Langton-Crutchfield (1990s) result — set ρ ≈ 1 − ε —
makes ρ **the** hyperparameter. **LOC ~250.** Largest of the twelve, also
the highest research leverage; Pulse / Oracle / Horizon all benefit.

### S12. `BasinTargeting / SensitivityControl` — minimum-FTLE path

**Capability.** Drive a trajectory from `y0` to `y*` along the path that
minimises the **finite-time Lyapunov exponent** along the trajectory — the
most predictable / least sensitive route. Shinbrot 1995; Bollt-Lai 2010.

**Composition.** S1 (Jacobian along trajectory) + S5 (windowed FTLE) + outer
optimiser via `optim.LBFGS` over admissible parameter schedules. Objective
`J(p(·)) = ∫ ‖J_f(t, y; p)‖ dt + α·‖y(T) − y*‖²` is differentiable through
the dynamics by `autodiff` (cross-link to 168 / 185). **LOC ~120.**

---

## 3. The thirteenth fix — `linalg.NonsymmetricEigen` (the structural unblocker)

OGY (S2), Lyapunov spectrum at full dim (S5), reservoir spectral-radius
certification (S11), and Floquet theory (S7) all need **eigenvalues of a
non-symmetric real matrix**. Reality's `linalg.QRAlgorithm` is symmetric-
only via Householder + tqli; applying it to non-symmetric matrices is
silently wrong (tqli assumes real-symmetric structure). Agent 097 listed
this. Fix: Hessenberg reduction (~120 LOC) + Francis double-shift QR (~200
LOC, the standard non-symmetric algorithm; LAPACK `dhseqr`). **~320 LOC** in
`linalg/eigen.go`. Until it ships, S2/S5/S7 cap at n≤2 closed-form or
n≤10 Gram-Schmidt-only. **The single linalg gap that unblocks the most
cross-package work in the chaos/control/orbital triangle** — cross-link to
agents 097, 187.

---

## 4. Connective-tissue LOC summary

| ID | Synergy | LOC | Depends on | Lands without §3? |
|---|---|---:|---|---|
| S1 | NumericalJacobian | 50 | — | yes |
| S2 | OGYStabilize | 140 | S1 | 2×2 only |
| S3 | OGYTargeting | 60 | S2 | as S2 |
| S4 | PoincareSection | 90 | RK4Step | yes |
| S5 | LyapunovSpectrum | 140 | S1 | yes (n≤10) |
| **S6** | **Pyragas (priority 1)** | **80** | — | **yes** |
| S7 | ETDAS | 50 | S6 | yes |
| S8 | PecoraCarrollSync | 70 | RK4Step, L2Norm | yes |
| S9 | GeneralizedSync | 30 | S8 | yes |
| S10 | LagSync / PhaseSync | 60 | S8, signal.Hilbert | yes |
| S11 | EchoStateNetwork | 250 | linalg.CholeskySolve | yes |
| S12 | BasinTargeting | 120 | S1, S5, optim.LBFGS | yes |
| **§3** | **NonsymmetricEigen** | **320** | — | unblocks S2/S5/S7 |
| | **Total** | **~1180 (synergy) + 320 (linalg)** | | |

**Sprint ordering.**
1. **S6 Pyragas** — 80 LOC, one day, no Jacobian, highest citation density.
2. **S1 Jacobian** — 50 LOC, unblocks S2/S5/S7/S12.
3. **S5 Lyapunov spectrum** — 140 LOC, the most-cited 026 gap.
4. **S8 Pecora-Carroll** — 70 LOC, foundational sync.
5. **§3 Non-symmetric eigen in linalg/** — 320 LOC, structural unblocker.
6. **S11 Echo State Network** — 250 LOC, lifts reality from "1990 textbook"
   to "2018 SOTA-baseline."
7. Remainder (S2/S3/S4/S7/S9/S10/S12) compose on top.

**Path to a credible chaos-control surface ≈ 1500 LOC of pure composition
over already-shipped primitives**, no new math, all citation-grounded, all
golden-file-testable across Go/Python/C++/C#.

---

## 5. Out-of-scope notes

- **Bifurcation control** (Chen-Moiola-Wang 1999 / Abed-Fu 1986): one-line
  add-on to S2 once parameters become time-varying, but canonical Hopf-normal-
  form formulation drags in symbolic algebra reality does not own — flag
  for a future `chaos/normalform.go` (600+ LOC ceiling).
- **Anticipating sync** (Voss 2000): ~40 LOC variant of S8 once S6's delay
  buffer ships.
- **Chaos shift keying / chaotic communications** (Cuomo-Oppenheim 1993):
  one-line wrapper around S8 with binary parameter modulation — out of
  scope here, lives in `crypto/` if anywhere.
- **Stabilising biological / cardiac chaos** (Schiff-Jerger-Duong-Chang-
  Spano-Ditto, *Nature* 1994): consumer-side glue on S2/S6 inside
  Pulse / Sentinel monitoring — no math additions.
- **Mixmaster / Hamiltonian targeting** (Bollt-Meiss): requires a symplectic
  integrator (out of scope, see 028 axis #10 + agent 109 orbital).
- **Inverse problem (data → dynamics)**: orthogonal seam, owned by 154
  (chaos × timeseries) and any future SINDy work — not duplicated here.
- **Reservoir computing for spatio-temporal chaos** (Pathak-Lu et al. 2018,
  Kuramoto-Sivashinsky): 2-D extension of S11 once `signal.FFT2` ships
  (per 132 / 135). 50 LOC on top of S11.

---

## 6. Cross-package coupling map

```
chaos/         ── RK4Step ───────────────┐
                                         │
control/       ── PIDCtl ────────────────┼── S6 Pyragas (delay buffer)
                                         │
linalg/        ── MatVecMul ─────────────┼── S5 Spectrum / S2 OGY / S11 ESN
                                         │
optim/         ── LBFGS ─────────────────┼── S12 SensitivityControl
                                         │
signal/        ── HilbertTransform ──────┴── S10 PhaseSync

§3 NonsymmEigen (linalg/) is the single inter-package unblocker.
Agent 097 owns it; agents 187, 191 (this), and 154 all consume it.
```

`autodiff/` — flagged for the differentiable-chaos extension (S12 with
gradient through dynamics) once forward-mode duals or HVP land per 012/013.
`testutil/` — every S above ships ≥6 golden JSON vectors; cross-language
port story is "Go canonical, Python validates," same as the rest of reality.

---

## 7. Citations (all closed-form, all pre-2020)

1. Ott, E.; Grebogi, C.; Yorke, J. *PRL* **64** 1196 (1990) — OGY.
2. Pyragas, K. *Phys. Lett. A* **170** 421 (1992) — time-delay feedback.
3. Socolar, J.; Sukow, D.; Gauthier, D. *PRE* **50** 3245 (1994) — ETDAS.
4. Pecora, L.; Carroll, T. *PRL* **64** 821 (1990) — chaos synchronisation.
5. Rulkov, N.; Sushchik, M.; Tsimring, L.; Abarbanel, H. *PRE* **51** 980
   (1995) — generalised synchronisation.
6. Rosenblum, M.; Pikovsky, A.; Kurths, J. *PRL* **76** 1804 (1996); **78**
   4193 (1997) — phase / lag synchronisation.
7. Shinbrot, T.; Ott, E.; Grebogi, C.; Yorke, J. *PRL* **65** 3215 (1990) —
   targeting.
8. Benettin, G.; Galgani, L.; Giorgilli, A.; Strelcyn, J.-M. *Meccanica*
   **15** 9 (1980) — Lyapunov spectrum via QR-renormalisation.
9. Maass, W.; Natschläger, T.; Markram, H. *Neural Comp.* **14** 2531
   (2002) — Liquid State Machines. Jaeger, H. *GMD Tech. Rep.* 148 (2001) —
   Echo State Networks.
10. Pathak, J.; Hunt, B.; Girvan, M.; Lu, Z.; Ott, E. *PRL* **120** 024102
    (2018) — reservoir prediction of spatiotemporal chaos.
11. Boccaletti, S.; Grebogi, C.; Lai, Y.-C.; Mancini, H.; Maza, D. *Phys.
    Rep.* **329** 103 (2000) — review: "The Control of Chaos".
12. Schöll, E.; Schuster, H. (eds.) *Handbook of Chaos Control*, 2nd ed.
    (Wiley-VCH 2008) — Ch. 1 OGY, Ch. 5 Pyragas, Ch. 21 sync.
13. Fiedler, B.; Flunkert, V.; Georgi, M.; Hövel, P.; Schöll, E. *PRL* **98**
    114101 (2007) — refutation of the odd-number limitation for delay
    feedback.
14. Datseris, G.; Parlitz, U. *Nonlinear Dynamics: A Concise Introduction
    Interlaced with Code* (Springer 2022) — modern interface conventions
    cited by agent 028.
