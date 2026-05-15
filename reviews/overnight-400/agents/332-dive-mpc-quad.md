# 332 — dive-mpc-quad (QP / OSQP / Goldfarb-Idnani / explicit MPC / receding-horizon audit)

**Block:** D — specific deep dives. **Date:** 2026-05-09. Repo at v0.10.0, 1965 tests.
**Scope:** algorithm-level deep-dive on Quadratic Programming primitives and Model-Predictive
Control — the *solver* tier that slot 178 (synergy-control-optim) treats as black-box "M4/M4'/M4''"
and slot 102 (optim-missing) names as T1.14. Slot 102 stops at "do this"; slot 178 stops at
"compose this"; slot 332 specifies *which solver, which paper, which 300 LOC, which witness*.

## Headline

`reality` has zero QP solver, zero MPC, and zero KKT object today; the cheapest ship-it-tomorrow
move is **a `Qp` struct (~50 LOC) + OSQP-style ADMM-QP (~300 LOC composed on existing
`optim/proximal.Admm`) + receding-horizon `LinearMpc` (~200 LOC)** — a 550-LOC PR-1 that
unlocks every robotics/drone/process-control consumer of `reality` and immediately saturates
`R-MUTUAL-CROSS-VALIDATION 3/3` (OSQP-ADMM ≡ Goldfarb-Idnani ≡ Mehrotra-IP on the same QP).

## Findings

### F1 — Today's ground state

- `optim/linear.go` ships LP-only: `SimplexMethod` (Bland's rule, full tableau) + `InteriorPoint`
  (gradient-on-barrier heuristic, *not* Mehrotra; flagged at 102-T1.13 / 276-A4).
- `optim/proximal/admm.go:53` ships consensus ADMM with `ProxOp` interface (Boyd 2011 §3.1.1
  scaled form, two prox callbacks). Eight prox ops at `optim/proximal/operators.go:28-197`:
  L1, L0, SquaredL2, NonNeg, Box, L2Ball, Simplex, Linear.
- `control/` has only scalar PID (`pid.go`), three first-order filters (`filter.go`), and a
  polynomial `TransferFunction` with Durand-Kerner (`transfer.go`). No state-space, no horizon,
  no constraint object, no MPC.
- Repo-wide grep: zero hits for `QuadraticProgram | OSQP | Goldfarb | ActiveSet | MPC` in non-review code.
- Slot 178 already names the composition primitives M3/M4'/M5 (KKT struct + ADMM-QP + LinearMpc) at
  *380 LOC* total; this slot specifies the algorithms inside those LOC budgets.

### F2 — The four QP-solver families and which one belongs in v0.x

| Family | Canonical paper | Strength | Weakness | Reality-fit verdict |
|---|---|---|---|---|
| **OSQP / consensus-ADMM** | Stellato-Banjac-Goulart-Bemporad-Boyd 2020 *Math. Prog. Comput.* 12(4):637-672 | robust to ill-conditioning, no requirement on Q PD or A full-rank, division-free post-factorisation, **first ADMM-QP with reliable infeasibility detection**, 10× faster than IPM with warm-start | needs ρ tuning, dual residual converges slowly near boundary | **T1 — ship first.** Reuses `optim/proximal.Admm` verbatim; ~300 LOC including `prox_f = solve KKT linear system`, `prox_g = ProxBox` projection. |
| **Goldfarb-Idnani dual active-set** | Goldfarb-Idnani 1983 *Math. Prog.* 27:1-33 | numerically stable (Cholesky + QR with rank-1 updates), exact in finite steps for SCQP, no tuning | requires Q strictly PD (1983 form), poor for >~200 vars, harder warm-start across changing active sets | **T2 — ship second.** ~250 LOC; the witness for OSQP correctness via R-MUTUAL pin. |
| **qpOASES parametric active-set** | Ferreau-Kirches-Potschka-Bock-Diehl 2014 *Math. Prog. Comput.* 6(4):327-363 | homotopy along previous solution → ideal for MPC where active set barely changes between samples; out-performs others on small/medium QP | C++ template-heavy; 3000+ LOC port; warm-start logic non-trivial | **T4 — defer.** GI dual active-set covers 90%; qpOASES is "GI + homotopy". |
| **Mehrotra IPM / cvxgen** | Mehrotra 1992 *SIAM J. Optim.* 2(4):575-601; Mattingley-Boyd 2012 *Optim. Eng.* 13(1):1-27 | predictor-corrector centring; cvxgen generates branch-free C for fixed dimensions → microsecond solves on embedded targets | code-gen only viable for fixed problem family; pure-runtime IPM beat by OSQP per Stellato 2020 §6 | **T5 — frontier.** Mehrotra IPM is 102-T1.13's job for LP; QP-IPM follows naturally; cvxgen-style code-gen is genuinely future v2. |

The empirical verdict (Stellato 2020 Table 6, on Maros-Mészáros): OSQP wins on ill-conditioned QPs;
qpOASES wins on tiny well-conditioned QPs with warm-start; Mehrotra-IPM wins on large dense QPs.
For an embedded MPC at 1 kHz on robotics-class problems (n,m ≤ 200), OSQP is the modern industry
default — Drake, do-mpc, acados-OSQP-backend, ProxQP all centre on it.

### F3 — The MPC tower

- **Receding-horizon QP** (Mayne-Rawlings-Rao-Scokaert 2000 review): solve QP at each sample,
  apply `u_0`, slide window. Composes M5 in slot 178. ~200 LOC = state-space substitution +
  QP build + warm-start.
- **Explicit MPC / mpQP** (Bemporad-Morari-Dua-Pistikopoulos 2002 *Automatica* 38(1):3-20):
  for linear MPC the optimal `u*(x_0)` is **piecewise affine** over polyhedral critical regions.
  Compute the partition once offline; runtime = point-location + matrix-vector product.
  ~150 LOC for the geometry (slot 178 M11 budgets 380 LOC; the larger figure assumes full
  critical-region enumeration including degenerate-vertex handling, which I'd defer).
- **Wang-Boyd 2010 fast MPC** *IEEE TCST* 18(2):267-278: structure-exploiting interior-point
  using block-tridiagonal Riccati factorisation; 100× faster than generic QP. Pure substrate
  improvement for the receding-horizon path; not a separate algorithm.
- **ACADO real-time iteration** (Houska-Ferreau-Diehl 2011 *Automatica* 47(10):2279-2285;
  also Diehl-Bock-Schlöder 2002): single SQP step per sample on a nonlinear MPC, with
  factorisation reused → microsecond solves. This is a *driver* not a solver — composes on
  GI active-set with warm-start. Slot 178 M12.

### F4 — Concrete primitive list (T0 → T5)

| Tier | Primitive | LOC | Composition | Citation |
|---|---|---|---|---|
| **T0** | `optim/qp.Qp{H, c, Aeq, beq, l, u, Aineq}` struct + `KKTResidual(z, ν, λ)` | 50 | new struct only | Boyd-Vandenberghe §5.5 |
| **T1** | `optim/qp.SolveOSQP(qp, cfg) → (z, λ, status)` ADMM via `proximal.Admm` | 300 | `optim/proximal.Admm` (verbatim) + `linalg.LUSolve` (KKT linear system once, factorised) + `ProxBox` from `proximal/operators.go:103` | Stellato et al 2020 |
| **T2** | `optim/qp.SolveGoldfarbIdnani(qp, cfg)` dual active-set | 250 | `linalg.CholeskyDecompose` + rank-1 update of `R` (QR), iterative add/drop | Goldfarb-Idnani 1983 |
| **T3** | `control/mpc.LinearMpc{ss, Q, R, Qf, N, …}` + `Step(x0)` receding-horizon | 200 | T1 + `linalg.MatMul` for condensed-form build (Bemporad 1998) + warm-start by shifting last `u*` | Mayne et al 2000; Wang-Boyd 2010 |
| **T4** | `control/mpc.ExplicitMpc` precomputed PWA policy | 150 | T0 + multiparametric QP critical-region enumeration (degenerate cases deferred) | Bemporad-Morari-Dua-Pistikopoulos 2002 |
| **T5** | `optim/qp.SolveMehrotraIP` predictor-corrector for large dense QP; cvxgen-style codegen further out | 350 | `linalg.CholeskySolve` for normal equations on `AΘAᵀΔν = r` | Mehrotra 1992; Mattingley-Boyd 2012 |

**T0+T1+T3 = 550 LOC** is the cheapest day-1 PR. Slot 178 PR-1 budget (M3+M4'+M5 = 380) under-counts
because it elides the OSQP-specific bits (ρ-adaptation §5.2, infeasibility-detection §3.4,
factorisation-caching §4); 300 LOC for T1 is realistic.

### F5 — R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities

Saturation per CLAUDE.md golden-file convention. All three are ship-with-PR-1 cheap:

1. **R-QP-3-WAY (T1 vs T2 vs T5).** Random PD `H ∈ R^{n×n}`, random `c, l, u` for n ∈ {5, 20, 50}:
   `SolveOSQP` ≡ `SolveGoldfarbIdnani` ≡ `SolveMehrotraIP` agree to 1e-8 in `‖z*‖_∞` over 1000
   seeded problems. Cross-language: regenerate with Python `quadprog` (GI), `osqp` Python
   bindings, `qpOASES` golden vectors. **Saturates the same idiom as commit 6a55bb4
   audio-onset 3-detector and 365368a Clayton-autodiff.**
2. **R-MPC-EQUALS-LQR (T3 unconstrained ≡ DARE LQR).** Slot 161-C6 `LQR` (assumed landed):
   strip box constraints from T3 LinearMpc, set `Qf = DARE(A,B,Q,R)`, run double-integrator
   `A=[[1,0.1],[0,1]], B=[[0.005],[0.1]]` 1000 steps from random `x_0`: `LinearMpc.Step(x0)`
   ≡ `LQR.K @ x0` to 1e-10. Bellman-principle witness — *finite-horizon receding QP without
   constraints reduces analytically to infinite-horizon Riccati*.
3. **R-EXPLICIT-EQUALS-ONLINE (T3 ≡ T4 on `x_0` test grid).** Same double integrator, box
   `−1≤u≤1`, N=2 horizon: `ExplicitMpc.Step(x_0)` (PWA point-location) ≡ `LinearMpc.Step(x_0)`
   (online QP) to 1e-8 over 10⁴ random `x_0` in the feasibility region. Pin against
   Bemporad-Morari-Dua-Pistikopoulos 2002 Fig 3 (~21 critical regions for that exact problem).

A fourth half-pin (Lyapunov-decrease witness, `V(x_{k+1}) < V(x_k) − x_k'Qx_k − u_k'Ru_k`)
would saturate stability-of-MPC against terminal-cost design (Mayne-Rawlings 2000 Thm 3.1)
but takes ~50 LOC of test scaffolding and isn't blocking for PR-1.

### F6 — Hazards specific to QP/MPC code

- **OSQP `ρ` adaptive update** (Stellato 2020 §5.2): `ρ_eq = 10⁶ · ρ_ineq` for hard equalities;
  current `optim/proximal.AdmmConfig.Rho` is a *scalar*. T1 either (a) extends `AdmmConfig`
  with per-row override, or (b) folds equalities into inequalities `b ≤ Aeq z ≤ b` — the
  OSQP-canonical form (`l ≤ Az ≤ u` everywhere). Recommend (b): keeps `proximal.Admm`
  surface unchanged.
- **Infeasibility detection** (Stellato 2020 §3.4 / Banjac-Goulart 2018): primal-infeasible
  certificate from `δy ≠ 0, A'δy = 0, l'(δy)_+ + u'(δy)_- < 0`. Mandatory or solver loops
  silently on infeasible MPC sub-problems. ~30 LOC.
- **Goldfarb-Idnani positive-definite assumption.** 1983 form requires Q strictly PD. If
  user passes singular Q (e.g. control-only cost in MPC with free state), GI panics; OSQP
  doesn't. Document; gate T2 with `IsPositiveDefinite(H)` precondition.
- **Warm-start correctness for receding-horizon MPC.** Shift previous `u*` by one stage,
  pad last with `K_LQR x_N` (terminal LQR). The naïve "pad with zero" warm-start works
  but degrades convergence by 3-5× in OSQP iterations.
- **Condensed-vs-sparse formulation.** Slot 178 cross-over: `Nm > n²` favours sparse Riccati-IP
  (Wang-Boyd 2010); below that, condense (substitute `x_k = A^k x_0 + Σ A^j B u_{k-1-j}`)
  and ship dense QP. T3 should default condense for `Nm ≤ 100`; document.
- **Explicit-MPC critical-region count** is exponential in worst case (≤ 3^q for q
  inequalities). Bemporad-Morari-Dua-Pistikopoulos 2002 documents 21 regions for the
  double-integrator N=2 example; large-horizon explicit MPC blows up. T4 must cap at 10⁴
  regions and fall back to T3 online QP — non-negotiable for safety.
- **mpQP degenerate-vertex handling.** When critical-region boundaries are linearly
  dependent (degenerate KKT), naive enumeration produces overlapping or missing regions.
  Tøndel-Johansen-Bemporad 2003 *Automatica* 39(3):489-497 fixes this with active-set
  pivoting; slot 332 T4 explicitly defers degenerate-handling to T4-extended.

### F7 — Cross-link consumer demand

QP/MPC unblocks (high-value future-app pull):

- **Drone autopilot** (px4-style): MPC over 6-DOF rigid-body for trajectory tracking.
  Composes T3 + slot 313 (rotation reps) + slot 314 (AHRS) + slot 308 (Kalman square-root) →
  full LQG/MPC autopilot. **Crown synergy.**
- **Vehicle dynamics control** (autonomous-vehicle stack): T3 lateral MPC over bicycle
  model + slot 178 M9 output-feedback MPC = canonical DARPA-Urban-Challenge architecture.
- **Robotics manipulation** (whole-body control): T1 OSQP-QP at 1 kHz solving inverse-dynamics
  with friction/torque/joint-limit constraints. Drake's exact pattern.
- **Process control** (chemical, HVAC): T3 + soft constraints (slot 178 M7) + reference
  governor (M6) — the original 1980s Cutler-Ramaker DMC use case.
- **Slot 102 cousin** (optim-missing T1.14): names T1+T2 directly. Slot 332 specifies algorithms.
- **Slot 308 cousin** (Kalman square-root): T3 + KalmanFilter = LQG = the canonical optimal-
  control textbook problem (Anderson-Moore 1979). Witness: separation principle.
- **Slot 178 superset**: this dive specifies the algorithms that 178 treats as black-box.
- **Slot 334 sibling** (MPPI): sample-based alternative to T3 for nonlinear systems where
  QP linearisation is poor. Williams-Aldrich-Theodorou 2017. Cross-link only.

### F8 — Single-day high-leverage commit

**T0 + T1 + T3 = 550 LOC + ~300 LOC tests in 1.5 engineer-days**, landing:

- First QP solver in `reality` (OSQP-style ADMM, the modern industry standard).
- First MPC controller in `reality` (linear receding-horizon).
- One R-MUTUAL pin (R-MPC-EQUALS-LQR) saturating immediately if slot 161-C6 LQR has landed,
  otherwise a self-witness pin against analytic finite-horizon Riccati closed-form (~30 LOC
  of test scaffolding).

Net new code is small because OSQP-ADMM literally *is* the existing `optim/proximal.Admm`
with two specific prox callbacks (KKT linear-solve as `prox_f`, box-projection as `prox_g`).
This is the tightest example in the repo of the "compose, don't duplicate" doctrine.

## Concrete recommendations

1. **Land T0 + T1 + T3 as PR-1 (550 LOC).** Place at `optim/qp/{qp.go, osqp.go}` and
   `control/mpc/linear.go`. T1 should reformulate equalities into the OSQP-canonical
   `l ≤ Az ≤ u` form so `optim/proximal.AdmmConfig` needs no per-row-ρ extension.

2. **Pin R-MUTUAL-CROSS-VALIDATION 3/3 in PR-1 with two of three witnesses.** R-MPC-EQUALS-LQR
   (T3 unconstrained ≡ DARE LQR closed-form) and R-OSQP-EQUALS-ANALYTIC (T1 on box-constrained
   1-D QP ≡ closed-form `clip(−H⁻¹c, l, u)` for `H` diagonal). Defer R-QP-3-WAY to PR-2 when
   T2 lands.

3. **Land T2 Goldfarb-Idnani as PR-2 (~250 LOC) for the QP cross-validation.** Saturates
   R-QP-3-WAY 3-way against T1 OSQP. Place at `optim/qp/active_set.go`. Use
   `linalg.CholeskyDecompose` + Schreiber-Parlett rank-1 updates of QR factor.

4. **Land T4 Explicit-MPC as PR-3 (~150 LOC).** Required by low-power deployment story.
   Cap at 10⁴ critical regions, fall through to T3 online for x_0 outside enumerated regions.
   Pin R-EXPLICIT-EQUALS-ONLINE 3-way against T3 over 10⁴ random `x_0`.

5. **Defer T5 Mehrotra-IPM and cvxgen-style code-gen.** OSQP outperforms IPM on Stellato 2020
   benchmark for n,m ≤ 500; the IPM fits naturally in the same `optim/qp/` package later as
   `interior_point.go`, but is not on the critical path. cvxgen code-gen is genuinely v2 work
   (it's a *compiler* tool that generates Go from a problem-family description; structurally
   different from the rest of `reality`).

6. **Replace `optim/linear.go::InteriorPoint` LP-heuristic with Mehrotra LP-IPM (slot 102 T1.13)
   *before* QP-IPM lands.** The QP-IPM math reuses the LP-IPM Newton-on-KKT loop verbatim;
   landing them out of order means re-doing the LP version after-the-fact.

7. **Pre-emptively name `optim/qp.Qp` as the "generic constrained-QP" type and reuse from
   slots 178 (control-MPC), 198 (synergy-physics-optim — contact-LCP via QP), 174
   (gametheory-optim — Nash via NCP→QP), 277 (COPO).** Otherwise four sub-packages each
   reinvent a `Qp` struct.

8. **Document the OSQP `ρ`-tuning recipe** in `optim/qp/doc.go`: scale `ρ` so primal/dual
   residuals reach the same magnitude — Stellato 2020 §5.2's `ρ_new = ρ · sqrt(‖r_p‖∞ / ‖r_d‖∞ · scaleD/scaleP)`
   adaptive rule — every 25 ADMM iterations. Without this, OSQP regression tests will fail
   on poorly-scaled problems (e.g., MPC with `Q ~ 1, R ~ 10⁻⁴`).

9. **Add receding-horizon warm-start (T3) by shifting the previous solution.** `u*_warm[0:N-1] = u*_prev[1:N]; u*_warm[N-1] = K_LQR · A · x_prev`. Without it, OSQP iterations per
   MPC step jump from ~30 (warm) to ~200 (cold) — the difference between 100 Hz and 15 Hz
   on a quad-core embedded target. This is the *practical* point of MPC in `reality`.

10. **Cross-link to slot 308 (Kalman square-root) for LQG**: T3 + `KalmanFilter` (slot 308
    or 161-C5) = `OutputFeedbackMpc` (slot 178 M9). The composition test
    "estimation-error MSE ≡ Kalman-NIS bound; tracking-error MSE ≡ LQR closed-form"
    is the canonical separation-principle witness — saturating two adjacent reviews
    simultaneously.

## Sources

**Repo files (absolute paths):**

- `C:\limitless\foundation\reality\optim\linear.go` — current LP simplex/InteriorPoint
- `C:\limitless\foundation\reality\optim\proximal\admm.go:53` — Admm consensus form, the OSQP substrate
- `C:\limitless\foundation\reality\optim\proximal\operators.go:103` — ProxBox, the OSQP `prox_g`
- `C:\limitless\foundation\reality\control\pid.go`, `control\filter.go`, `control\transfer.go` — current control state
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\102-optim-missing.md` — names QP at T1.14
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\178-synergy-control-optim.md` — composition framework (M3/M4'/M5)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\052-control-missing.md` — names MPC as missing
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md:352` — slot 332

**Web sources (algorithms):**

- Stellato, Banjac, Goulart, Bemporad, Boyd. "OSQP: an operator splitting solver for quadratic programs." *Math. Prog. Comput.* 12(4):637-672, 2020. https://web.stanford.edu/~boyd/papers/pdf/osqp.pdf
- Goldfarb, Idnani. "A numerically stable dual method for solving strictly convex quadratic programs." *Math. Prog.* 27:1-33, 1983. https://link.springer.com/article/10.1007/BF02591962
- Ferreau, Kirches, Potschka, Bock, Diehl. "qpOASES: a parametric active-set algorithm for quadratic programming." *Math. Prog. Comput.* 6(4):327-363, 2014. https://link.springer.com/article/10.1007/s12532-014-0071-1
- Bemporad, Morari, Dua, Pistikopoulos. "The explicit linear quadratic regulator for constrained systems." *Automatica* 38(1):3-20, 2002. http://cse.lab.imtlucca.it/~bemporad/publications/papers/acc00-mpqp.pdf
- Wang, Boyd. "Fast Model Predictive Control Using Online Optimization." *IEEE TCST* 18(2):267-278, 2010. https://stanford.edu/~boyd/papers/pdf/fast_mpc.pdf
- Mattingley, Boyd. "CVXGEN: a code generator for embedded convex optimization." *Optim. Eng.* 13(1):1-27, 2012. https://stanford.edu/~boyd/papers/pdf/code_gen_impl.pdf
- Houska, Ferreau, Diehl. "An auto-generated real-time iteration algorithm for nonlinear MPC in the microsecond range." *Automatica* 47(10):2279-2285, 2011. https://www.sciencedirect.com/science/article/abs/pii/S0005109811003918
- Mayne, Rawlings, Rao, Scokaert. "Constrained model predictive control: Stability and optimality." *Automatica* 36(6):789-814, 2000. (review)
- Mehrotra. "On the implementation of a primal-dual interior point method." *SIAM J. Optim.* 2(4):575-601, 1992.
- Tøndel, Johansen, Bemporad. "An algorithm for multi-parametric quadratic programming and explicit MPC solutions." *Automatica* 39(3):489-497, 2003.
- Boyd, Vandenberghe. *Convex Optimization*, Cambridge UP 2004. §5.5 (KKT), §11 (interior-point).
- Banjac, Goulart, Stellato, Bemporad, Boyd. "Infeasibility detection in the alternating direction method of multipliers for convex optimization." 2018. (OSQP infeasibility certificate)
- OSQP documentation. https://osqp.org/docs/
- qpOASES (COIN-OR). https://github.com/coin-or/qpOASES
- ACADO Toolkit. https://acado.github.io/
