# 066 — fluids-numerics

**Topic:** fluids: Bernoulli ill-conditioning at low Δp, Reynolds-regime branching tests
**Scope:** numerical-correctness audit of `C:\limitless\foundation\reality\fluids\`
**Files audited:**
- `fluids\fluids.go` (236 LOC, 11 exported functions)
- `fluids\fluids_test.go` (315 LOC)
- `fluids\fluids_edge_test.go` (250 LOC)
- `fluids\testdata\fluids\*.json` (5 golden files: bernoulli, darcy_weisbach, drag_force, reynolds, terminal_velocity — 4 vectors per file, 20 vectors total)

## Executive summary

`fluids/` is a 236-LOC "fluid mechanics 101" surface — eleven scalar functions, one of which (`PipeFlowFriction`) is a Colebrook–White fixed-point solver and the rest are pure arithmetic over `float64`. Numerically, the package is **mostly IEEE-754 clean on its happy paths** with only `math.Sqrt`, `math.Log10`, `math.Pow`, and `math.Pi` as transcendentals. **Three substantive numerical-correctness defects**, **two latent precision/spec gaps**, and **two missing-but-claimed primitives** dominate this report:

- **N-1 (HIGH)** — Hard 70.2% discontinuous jump at Re=2300 between Hagen–Poiseuille (64/Re=0.0278) and Colebrook (0.0474). No transition smoothing; no transition-region warning. Topic prompt explicitly calls this out.
- **N-2 (HIGH)** — Colebrook iteration uses **absolute** tolerance `1e-12` on `f`, not relative. At very high Re (f → 0.008–0.012) this is ~1e-10 relative; at very low laminar-edge Re (f → 0.05) it is ~2e-11 relative. Spec says "1e-10 relative change" — code does not match spec. Iteration cap of 100 is reached on no tested input but loop returns last `f` silently rather than NaN-on-divergence.
- **N-3 (HIGH)** — `TerminalVelocity` takes `Cd` as a constant input but Cd is itself a strong function of Re (which depends on `v_t`). Empirically, raindrop-scale calls (m=4.19e-6 kg, A=π·1e-6 m², Cd=0.5) yield v_t=6.5 m/s and an implied Re≈885 — well outside the Newton drag regime where Cd≈0.5 is valid. The function is mathematically `√(2mg/(Cd·ρ·A))` (correct), but the docstring's "drag coefficient" hides that the Re-coupled *fixed-point on Cd(Re)* is the user's problem.
- **N-4 (MEDIUM)** — Topic prompt's "Bernoulli ill-conditioning at low Δp" is **not realisable in `BernoulliPressure(rho, v1, p1, h1, v2, h2, g)`** because p1 dominates the sum. Empirically, eps=v2−v1=1e-12 at v=100 m/s is recovered to 0 ULP. The hazard *exists* but only when the user computes Δp = p2−p1 themselves — and the package gives no `BernoulliPressureDrop` primitive that would internally do `0.5*rho*(v1*v1-v2*v2) + rho*g*(h1-h2)` (avoiding the p1 carry).
- **N-5 (MEDIUM)** — `0 <= Re < 2300` branch returns `64/Re` even at Re=0+ where f=∞. `Re==0` returns NaN (correct), but `Re=1e-300` returns 6.4e301 with no overflow guard.
- **N-6 (LOW)** — `PipeFlowFriction(Re, 0, D)` uses `relRough=0` and Swamee–Jain seed `0.25/log10(5.74/Re^0.9)²`; for Re≤~10 this seed is negative-divergent (`logArg < 1` → log10 < 0 → seed positive but small; checked: Re=1e10 seed=0.0037 — fine). No overflow.
- **N-7 (LOW)** — Topic-prompt items **Mach number, compressibility, drag-Cd lookup tables, lift-Cl aerofoil theory, Stokes/inertial regime smoothing, Moody-chart Swamee–Jain accuracy bounds** are not present. Forward-looking spec needed (§Forward).

Six topic-prompt items map to **forward-looking** primitives that do not exist yet (no `MachNumber`, no `CdSphere(Re)`, no `Colebrook` transition smoothing, no compressible Bernoulli, no `LiftCoefficient` from α / aspect-ratio). These are flagged here so when 067-fluids-missing enumerates them they can be co-numbered.

## What the package contains today

| Function | Formula | Hot-path ops | Numerics |
|---|---|---|---|
| `ReynoldsNumber(ρ,v,L,μ)` | `ρvL/μ` | 2 mul, 1 div | exact; +Inf if μ=0 |
| `BernoulliPressure(ρ,v1,p1,h1,v2,h2,g)` | `p1 + ½ρ(v1²−v2²) + ρg(h1−h2)` | 5 mul, 2 sub, 2 add | exact; see N-4 |
| `PipeFlowFriction(Re,ε,D)` | branched; 64/Re or Colebrook | 1 div / Colebrook fixed-point | see N-1, N-2 |
| `DarcyWeisbach(f,L,D,ρ,v)` | `f·(L/D)·(ρv²/2)` | 4 mul, 1 div | exact |
| `DragForce(Cd,ρ,v,A)` | `½Cdρv²A` | 4 mul | exact |
| `LiftForce(Cl,ρ,v,A)` | `½Clρv²A` | 4 mul | exact |
| `TerminalVelocity(m,g,Cd,ρ,A)` | `√(2mg/(CdρA))` | 4 mul, 1 div, 1 sqrt | see N-3 |
| `StokesLaw(μ,r,v)` | `6π·μrv` | 4 mul (π baked into literal) | exact-to-π |
| `MassFlowRate(ρ,v,A)` | `ρvA` | 2 mul | exact |
| `VolumetricFlowRate(v,A)` | `vA` | 1 mul | exact |

Constants: only `math.Pi`. No imports from `constants/`. No allocations. No internal state.

## Topic-prompt items: present vs missing

| Topic-prompt hazard | Applies? | Resolution |
|---|---|---|
| Bernoulli catastrophic cancellation | **Partial** — only at user level | N-4 + Forward |
| Reynolds standard-form | **Yes** | Resolved correctly |
| Reynolds regime branching | **Yes** — laminar/turbulent in `PipeFlowFriction` | **N-1 (open)** |
| Smoothing across regime boundary | **No** — hard 2300 cliff | N-1 + Forward |
| Colebrook convergence | **Yes** | **N-2 (open)** |
| Swamee–Jain accuracy | **Partial** — used as seed only | Forward (§Forward F-3) |
| Drag Cd table vs analytic fits | **No** — `Cd` is user input | Forward (§Forward F-2) |
| Lift Cl aerofoil theory | **No** — `Cl` is user input | Forward (§Forward F-2) |
| Terminal velocity Re-dependent Cd | **No iteration** — single sqrt | **N-3 (open)** |
| Mach number, compressibility | **No** — incompressible only | Forward (§Forward F-1) |
| Stokes vs inertial limit | **Partial** — `StokesLaw` standalone, no validity check | Forward (§Forward F-4) |
| IEEE-754 zero density / zero velocity | **Yes** — most paths | Resolved (Edge tests cover) except N-5 |

## Numerical-correctness defects (detail)

### N-1 — Hard 70% discontinuity at Re=2300 (HIGH, topic-prompt explicit)

`fluids.go:85`:
```go
if Re < 2300 {
    return 64.0 / Re
}
```

Empirical (verified by direct execution against ε=1e-5 m, D=0.1 m):

| Re | f returned | regime |
|---|---|---|
| 2299.999 | 0.027826 | Hagen–Poiseuille `64/Re` |
| 2300.000 | 0.047364 | Colebrook |
| Δ | **0.01954 (+70.2%)** | discontinuous step |

Physical reality: 2000 ≤ Re ≤ 4000 is the **transitional regime** where neither formula is correct. Real engineering practice uses one of: (a) Cheng's blending `f = (1−γ)·f_lam + γ·f_turb` with `γ = 1/(1+exp(−(Re−2700)/360))`; (b) Churchill 1977 single equation valid all-Re; (c) explicit `NaN` (or named-error) on `2300 ≤ Re ≤ 4000`. The current code fakes a sharp transition that is not in any of Reynolds, Moody, or Colebrook's papers and which `aicore` consumers will hit immediately when sweeping flow rates.

**Test coverage gap.** `TestPipeFlowFriction_LaminarBoundary2299` and `TestPipeFlowFriction_TurbulentTransition` test *each side* but never assert continuity — neither test would catch the discontinuity. Topic prompt explicitly asks for "smoothing across regime boundaries" so this is in-scope.

**Cross-language risk.** Python ports calling SciPy's `Colebrook` solver typically use Churchill or LBL Cheng-blend; if those ports follow the local convention rather than reality's discontinuous one, golden files will fail. The four turbulent vectors in `darcy_weisbach.json` (which only tests `f·L·ρv²/(2D)` not the Colebrook solver itself) sidestep this — there is **no golden file for `PipeFlowFriction`**, which means no cross-language pin on the threshold value or the iterate count.

### N-2 — Colebrook tolerance is absolute, spec says relative (HIGH)

`fluids.go:99`:
```go
if math.Abs(fNew - f) < 1e-12 {
    return fNew
}
```

Docstring (`fluids.go:76`): "iterative solve to ~1e-10 relative change". Code is `1e-12 absolute`. Empirically:

| Re | f | iters | tolerance hit at |
|---|---|---|---|
| 2300 | 0.04736 | 14 | 1e-12 abs ≈ 2e-11 relative |
| 1e5 | 0.01851 | 9 | 1e-12 abs ≈ 5e-11 relative |
| 1e10 | 0.01198 | 3 | 1e-12 abs ≈ 8e-11 relative |

Numerically OK for fp64 (all cases comfortable), but spec/code mismatch is a doc bug; if a future contributor follows the spec and changes the tolerance to `< 1e-10 * math.Abs(f)`, behaviour shifts and golden files would need re-pinning. **Cross-language port hazard:** Python and C# reviewers will read the docstring and implement what the docstring says, not what the Go code does — golden files will diverge by a few ULPs.

**Divergence handling.** Loop falls through after 100 iterations and returns the last `f` silently. Spec says "typically converges in < 20 iterations" so the 100 cap is generous, but for pathological input (e.g. `relRough = 1e30`) the user gets a silently-wrong number. Should return NaN (or a structured error) on iteration-cap exhaustion.

### N-3 — TerminalVelocity ignores Re-dependence of Cd (HIGH, topic-prompt explicit)

`fluids.go:175`:
```go
func TerminalVelocity(m, g, Cd, rho, A float64) float64 {
    denom := Cd * rho * A
    if denom <= 0 { return math.NaN() }
    return math.Sqrt(2.0 * m * g / denom)
}
```

This is the closed-form `v_t = √(2mg/(CdρA))` for a Newton-regime Cd that does not depend on Re. In reality, **Cd for a sphere varies from 24/Re (Stokes, Re<1) through ~0.5 (Newton, 10³<Re<2·10⁵) to ~0.1 post-drag-crisis (Re>3·10⁵)**. The user is required to know which regime they are in *before* calling the function — but the regime itself is `v_t`-dependent.

Empirical verification (raindrop, m=4.19e-6 kg, A=πmm² = π·1e-6, ρ_air=1.225, μ=1.81e-5):

| User-supplied Cd | computed v_t (m/s) | implied Re | Cd-actual at that Re |
|---|---|---|---|
| 0.4 (Newton low) | 7.31 | 989 | ~0.4 ✓ marginal |
| 0.5 (Newton typ) | 6.54 | 885 | ~0.4 |
| 1.0 (high) | 4.62 | 626 | ~0.5 |

For sub-mm droplets Cd≈24/Re (Stokes) and the closed form **fails by orders of magnitude**. Required addition: `TerminalVelocityIterative(m, g, ρ, μ, A, d, CdFn func(Re) float64) float64` that fixed-point-iterates on `v_t ↔ Re ↔ Cd(Re)`. Current function should at minimum carry a "valid only when Cd is known to be Re-independent at the resulting v_t" warning, and a sibling `TerminalVelocityStokes(m, g, μ, r) float64 = mg/(6πμr)` for the Re<<1 limit.

**Test coverage gap.** `TestTerminalVelocity_Skydiver` validates 42–43 m/s for human skydiver — Cd=1.0 at Re~3·10⁵ is *plausibly* in Newton regime so the test passes — but no test asserts the function fails or warns when Cd-Re self-consistency is violated.

### N-4 — Bernoulli "ill-conditioning at low Δp" doesn't trigger in the present API (MEDIUM)

The topic prompt says "catastrophic cancellation when Δp << p". The package's `BernoulliPressure` returns **p2**, not Δp, so the dominant `p1` term carries the full magnitude through and there is no cancellation hazard at all in the closed expression `p1 + 0.5ρ(v1²−v2²) + ρg(h1−h2)`. Empirical proof at v1=100, v2=v1+1e-12, p1=1e8 Pa: both straight `(v1*v1 − v2*v2)` and recast `(v1−v2)·(v1+v2)` agree to **0 ULP** with the analytic answer.

The hazard *would* arise if a user computes:
```go
dp := BernoulliPressure(...) - p1   // catastrophic — both ~p1
```
because `p2` and `p1` agree to many leading digits. The package does not expose a `BernoulliPressureDrop(rho, v1, h1, v2, h2, g) float64` (returning just the kinetic+potential terms, **without** the p1 carry) — and *that* function is the one that needs the `(v1−v2)·(v1+v2)` factored form to be cancellation-stable when v1≈v2.

**Action.** Either (a) add `BernoulliPressureDrop` and document the cancellation-safe path, or (b) document on `BernoulliPressure` that consumers must not compute `Δp = result − p1` and should call the (still-to-be-added) drop primitive instead.

### N-5 — `64/Re` overflow at tiny positive Re (MEDIUM)

`fluids.go:81-86`:
```go
if Re <= 0 { return math.NaN() }
if Re < 2300 { return 64.0 / Re }
```

`Re=5e-324` (smallest subnormal) returns `1.28e+325` → `+Inf`. Physically Re < 1 is creeping flow where Hagen–Poiseuille is mostly fine, but Re < 1e-300 is non-physical input that should return NaN-with-reason rather than +Inf. Low severity — no realistic user input will hit it — but the docstring claims `Valid range: Re > 0` without tagging the overflow regime.

### N-6 — Edge-test conventions inconsistent

`TerminalVelocity_NegativeArea` returns NaN (correct, denom = Cd·ρ·(−A) ≤ 0 → NaN guard fires). `DragForce(0, 1.225, 10, 1)` returns 0 cleanly. But:

- `BernoulliPressure_ZeroDensity` returns `p1` exactly (correct: ρ=0 zeros both kinetic and potential terms, **even when v→∞**, which is an over-permissive contract: ρ=0 is non-physical for a liquid streamline).
- `ReynoldsNumber_ZeroLength` returns 0 silently — physically Re at L=0 is undefined (no characteristic scale). Should arguably be NaN, but the test pins it to 0.

These are spec-vs-code consistency gaps not numerical errors per se.

### N-7 — Missing functions named in topic prompt

| Topic-prompt function | Present? |
|---|---|
| `MachNumber(v, c) = v/c` | **No** |
| Compressible Bernoulli (γ-aware) | **No** |
| `CdSphere(Re)` analytic fit (Schiller–Naumann, Clift–Gauvin, Morrison) | **No** |
| `CdCylinder(Re)` lookup/fit | **No** |
| `LiftCoefficientThinAirfoil(α)` = 2π sin α | **No** |
| Stokes-inertial smoothing | **No** |
| Moody-chart Swamee–Jain explicit (as user-callable, not just seed) | **No** |
| Compressibility correction `√(1−M²)` Prandtl–Glauert | **No** |

These belong to `067-fluids-missing` enumeration; flagged here so numerical specs can be co-decided.

## Forward-looking numerical commitments

These commitments must be settled at **first ship** of each missing primitive (golden files cross-language-pin behaviour at first commit):

**F-1 — Mach number and compressibility.** Spec γ-air = 1.4 (constants/), γ-water-vapour = 1.33, γ-monatomic = 5/3. Specify Prandtl–Glauert `√(1−M²)` cutoff at M=0.95 (avoid imaginary). Specify isentropic `T/T0 = (1 + (γ−1)/2·M²)^−1` exact at M=0 to bit.

**F-2 — Cd analytic fits.** Choose between Schiller–Naumann (`24/Re·(1+0.15·Re^0.687)`, valid Re<1000), Clift–Gauvin (extends to Re=2·10⁵), Morrison 2013 (single all-Re equation). Pin per-Re tolerance: 1e-9 in Stokes regime, 1e-6 in transition (drag crisis), 1e-3 elsewhere (transition data scatter).

**F-3 — Colebrook smoothing.** Pick one: (a) **Cheng 2008 sigmoid blend** between Hagen–Poiseuille and Swamee–Jain, valid all Re; (b) **Churchill 1977** single equation `f = 8·((8/Re)^12 + 1/(A+B)^1.5)^(1/12)` valid Re=0 to ∞; (c) explicit `transitional` return on 2000<Re<4000. Recommend Churchill — it is C∞, single-call, no iteration needed, and matches Colebrook to <1% in the fully-turbulent regime.

**F-4 — Stokes/Newton boundary.** Add `DragForceSphere(d, v, ρ, μ)` that internally selects Stokes (`F=6πμrv` if Re<1), Schiller–Naumann (1≤Re<1000), or Newton (`F=½·0.44·ρv²·πr²` if 1000≤Re<2·10⁵). Document hard transitions vs blends.

**F-5 — Terminal-velocity iteration.** Add `TerminalVelocityIterative(m,g,ρ,μ,d,CdFn)` that does Newton or fixed-point on `v ↔ Re ↔ Cd`. Tolerance: 1e-10 relative, max 100 iters, NaN on non-convergence.

**F-6 — Bernoulli drop primitive.** `BernoulliPressureDrop(ρ,v1,h1,v2,h2,g) float64 = 0.5*ρ*(v1-v2)*(v1+v2) + ρ*g*(h1-h2)` factored to avoid cancellation when v1≈v2.

**F-7 — Colebrook tolerance contract.** Change to `math.Abs(fNew−f) < 1e-12 * math.Abs(f)` (relative, matching docstring) and return NaN after 100-iter cap. Re-pin existing tests.

## Recommended fixes (ranked by reach × severity)

1. **N-1 fix** — Replace `if Re < 2300 { return 64/Re }` cliff with Churchill 1977 (single-equation, smooth, no iteration). Estimated effort: 30 min, +20 LOC, +20 golden vectors. Eliminates topic-prompt's headline complaint.
2. **N-2 fix** — Make Colebrook tolerance relative; add NaN-on-cap-exhaust. Estimated effort: 5 min, +2 LOC.
3. **N-3 fix** — Document `TerminalVelocity` Cd-constancy assumption; add `TerminalVelocityIterative` and `TerminalVelocityStokes` siblings. Estimated effort: 1 hr, +60 LOC, +30 golden vectors.
4. **N-4 fix** — Add `BernoulliPressureDrop` (factored form). Estimated effort: 10 min, +10 LOC, +10 golden vectors.
5. **N-5 fix** — Add overflow guard `if Re < 1e-300 { return math.NaN() }`. 5 min.
6. **Coverage** — Add a `PipeFlowFriction` golden file (currently absent — only `darcy_weisbach.json` exists, which tests the post-friction product, not the solver). Estimated 15 vectors covering laminar, transition, smooth-turbulent, rough-turbulent, hydraulically-rough limits, plus convergence-iter-count pins.

## Test-coverage gap summary

| Function | Golden file? | Edge tests? | Numerical hazards covered? |
|---|---|---|---|
| ReynoldsNumber | **Yes** (4 vec) | Yes (5) | mu=0 +Inf, neg v, L=0 |
| BernoulliPressure | **Yes** (4 vec) | Yes (4) | rho=0 spec-debatable; no cancellation case |
| PipeFlowFriction | **NO** | Yes (5) | **No discontinuity test**, no convergence-iter test |
| DarcyWeisbach | **Yes** (3 vec) | Yes (3) | basic |
| DragForce | **Yes** (4 vec) | Yes (4) | covers Cd=0, neg v |
| LiftForce | No | Yes (2) | basic |
| TerminalVelocity | **Yes** (3 vec) | Yes (3) | denom=0 NaN; **no Re-Cd self-consistency** |
| StokesLaw | No | Yes (4) | basic |
| MassFlowRate | No | Yes (2) | basic |
| VolumetricFlowRate | No | Yes (3) | basic |

5 of 11 functions lack golden files (LiftForce, StokesLaw, MassFlowRate, VolumetricFlowRate, **PipeFlowFriction** — the one with the most numerical content). Vector counts (3–4 per file) are well below the project standard of "minimum 20, target 30" stated in `CLAUDE.md`.

## Cross-language port risk

- **High:** `PipeFlowFriction` Re=2300 cliff. Python/C++ ports using SciPy/Boost will use Churchill or smoother and produce different results in the 2000–4000 band — **no golden file** will catch the disagreement.
- **High:** Colebrook absolute-vs-relative tolerance mismatch. Cross-language re-implementers reading the doc will choose `1e-10` relative; iteration count and last-bit will diverge — again no golden file pins iter count or final f to ULP.
- **Medium:** `BernoulliPressure` is associative-arithmetic-sensitive. The expression `p1 + 0.5*rho*(v1*v1-v2*v2) + rho*g*(h1-h2)` can be reassociated in five ways by an optimising compiler. Existing tolerance `1e-6` on golden vector "Elevation change only" suggests the author already noticed; should be locked to a specific evaluation order with explicit fp-contract, or all golden tolerances reviewed.
- **Low:** `StokesLaw` `6.0 * math.Pi * mu * r * v` — operand order matters at extreme magnitude; current order (large constants first) is fine.

## What would change if I had write access

1. Replace lines 80–105 of `fluids.go` with Churchill 1977 (closes N-1 + N-2 in one stroke; eliminates loop).
2. Add `BernoulliPressureDrop` (closes N-4).
3. Add `TerminalVelocityStokes` and `TerminalVelocityIterative` (closes N-3); annotate `TerminalVelocity` with regime warning.
4. Add overflow guard at line 85 (closes N-5).
5. Create `testdata/fluids/pipe_flow_friction.json` with ≥20 vectors spanning Re=10² to 10¹⁰ and ε/D=0 to 0.05 (closes the most consequential coverage gap).
6. Bump existing golden files from 3–4 vectors to 20+ each.

Total estimated effort: 4–6 hours for full numerical-correctness pass + golden-file rebuild.

---

**Auditor:** agent 066
**Status:** complete
**Next:** progress line appended; defects N-1, N-2, N-3 are the topic-prompt's three explicit asks and are all open.
