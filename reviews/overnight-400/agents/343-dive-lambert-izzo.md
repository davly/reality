# 343 — dive-lambert-izzo (Battin / Gauss / Izzo / multi-revolution Lambert audit)

## Headline
`reality/orbital` ships zero Lambert solvers; the cheapest day-1 PR is **Izzo
2014 universal Lambert** (~250–280 LOC, single-rev, Householder-3 on Battin's
`x` parameter, ESA pykep canonical) and it unlocks porkchop plots, intercept,
rendezvous, gravity-assist, asteroid deflection, and Sims-Flanagan match-point
in one stroke — confirming agents 107-T1.6, 108-§9, and 164-L1's prior asks.

## Findings

### State of `reality/orbital` w.r.t. Lambert
- **Zero hits** for `Lambert | Izzo | Battin | Gauss[Ll]ambert | MultiRev` across
  the entire `reality/` tree (`Grep` 2026-05-09). `orbital/orbital.go` is 267
  LOC, 8 closed-form scalars (`KeplerOrbit`, `OrbitalPeriod`, `OrbitalVelocity`,
  `HohmannTransfer`, `EscapeVelocity`, `HillSphere`, `SynodicPeriod`,
  `TrueAnomalyFromMean`). No state-vector type, no inverse `RV→COE`, no
  propagator — Lambert lands on a near-bare base.
- Only `HohmannTransfer` (orbital.go:135) overlaps the BVP territory, and it is
  the trivial special case (two coplanar circular orbits, single rendezvous
  geometry, no time-of-flight argument).
- **Substrate is favourable.** `optim.NewtonRaphson` + `BisectionMethod` +
  `GoldenSectionSearch` (164-§Bases) are precisely the iterative kernels Izzo
  needs; `linalg.CrossProduct` / `DotProduct` / `L2Norm` are the vector
  plumbing; `chaos.SolveODE` provides the cross-validation propagator. Izzo's
  Householder-3 step inlines as a specialised Newton with closed-form
  `dT/dx`, `d²T/dx²`, `d³T/dx³` (Izzo 2014 eqs. 22–23) — no new optim primitive
  required.

### Algorithm comparison (web research × repo cross-reference)

| Method | Year | Iter family | Singularities | Multi-rev | Robustness | Status in repo |
|--------|------|-------------|---------------|-----------|------------|----------------|
| Gauss original | 1809 | Successive subs on F-G | θ=π collapses | No | poor θ→π | absent |
| Sun-Vallado simplified Gauss | 1989 | Bisection on z (univ. var.) | bracket-bound by [-4π, 4π²] | weak | 5 % slower than Battin | absent |
| Battin hyperbolic-tangent | 1987/1999 | Successive subs on `x`/`y` | singularity moved θ=π → 2π | yes (extension) | good for large θ | absent |
| Lancaster-Blanchard | 1969 | (foundational) | unified `x` parameter | yes | proof-of-convergence | absent |
| Gooding | 1990 | Halley cubic on `x` | unified | yes | very robust, slower | absent |
| **Izzo** | 2014 | **Householder-3 on `x`** | unified L-similarity transform | **yes (Tmin curve)** | **2 iters single-rev, 3 iters multi-rev to ULP** | **absent — recommend T0** |
| Klumpp 1999 / Curtis textbook | various | universal-variable + Newton | bracket-tunable | weak | textbook-only | absent |
| von Stryk multi-rev | 1992 | low-thrust transcription | parametric N | N≥1 | low-thrust ctxt | far-future T3 |

### Why Izzo wins as T0
- **Single closed-form initial guess** (Izzo §3.1, eq. 30): `x₀(T,T₀,T₁)` from
  the time-of-flight ratio; no bracketing search, no warm-start.
- **Householder-3 convergence in ≤ 2 iters single-rev**, ≤ 3 iters multi-rev,
  to machine precision (Izzo §5.4 benchmark, 10⁹ random geometries on the unit
  sphere). Gooding needs 4–6 Halley steps; Battin successive subs need 8–15.
- **L-similarity transform** maps every conic — elliptic, parabolic
  (`x=1`), hyperbolic — onto the same `x∈(-1,∞)` axis. Removes the
  branch-on-`a` case-split that Gauss/Curtis carry.
- **Multi-rev branch handled by the same kernel** by adding `2πN` to the time
  scale and tracking the `T_min(x_T*)` inflection (Izzo §4). Yields *both*
  short-period and long-period multi-rev solutions per N.
- **Reference implementation is MIT-licensed** in ESA pykep (`pykep/core/_lambert_problem.cpp`)
  and BSD-licensed in poliastro (`hapsira/iod/izzo.py`); both are 200–300 LOC,
  pure-numeric, zero-dep, golden-file-ready.
- **Per-vector cost dominated by 5 transcendentals** (`acos`, `acosh`, `log`,
  `sqrt`, `atan2`) — one Householder iteration. ESA reports 20–33 % faster
  than Gooding for equivalent ULP, ~5× faster than Battin successive subs.

### Numerical hazards documented in the literature
- **`x = 1` parabolic singularity**: `1/(1-x²)` blows. Izzo §2.4 fixes with
  series expansion of `T(x)` around 1. Both pykep and poliastro implement the
  series; any port must too.
- **Multi-rev existence boundary**: for given N, no solution exists if
  `Δt < T_min(N)`. Detection requires solving an inner 1-D minimisation of
  `T(x;N)` on `x∈[-1,1]` (Izzo §4.2) — Halley iteration on `dT/dx = 0`. Two
  branches (left/right of `x_T*`) must both be reported.
- **`θ ≈ 0` and `θ ≈ 2π` degenerate**: the geometry gives a chord of zero
  length; Izzo declares these singular and requires the caller to detect.
- **Long-way vs. short-way**: a sign convention on the cross-product
  `r₁ × r₂ · ẑ` selects the prograde branch. Battin has the same sign
  ambiguity; both must expose it as a `prograde bool` argument.
- **Initial guess fallback at `Δt → 0`**: Izzo §3.1 eq. 30 uses
  `x₀ = (T₀/T)^(2/(p₂-p₁)) - 1` which → straight-line trajectory as `Δt → 0`.
  This is a free R-MUTUAL-CROSS-VALIDATION pin (see below).

### Cross-language reference implementations (golden-file source-of-truth)
- **ESA pykep** `pykep/core/_lambert_problem.cpp` (~600 LOC C++, MIT) — Izzo's
  own port; treated as the bit-exact reference by every downstream library.
- **poliastro / hapsira** `hapsira/iod/izzo.py` (~250 LOC Python+Numba, BSD-3) —
  Markley-Castelli Python translation, in lock-step with pykep to 1e-13.
- **lamberthub** (Piloto, MIT) — a benchmark suite of *eight* Lambert solvers
  (Gooding, Battin, Avanzini, Arora, Vallado, Izzo, Bate, Gauss) with
  cross-validation harness producing a 1e-12-tolerance JSON corpus. **Direct
  source for golden-file vectors** with no licensing friction.
- **GMAT** `LambertTargeter` (NASA, Apache-2) — bisection + Battin hybrid,
  validated against Vallado test suite; not the speed reference but the
  long-tail mission-design oracle.

### Test problem set (canonical Lambert benchmarks)
- **Vallado §7 example 7-5**: `r1 = (15945.34, 0, 0) km`, `r2 = (12214.84,
  10249.47, 0) km`, `Δt = 76 min`, `μ = 398600 km³/s²`. Single-rev. Reference
  output `v1 ≈ (2.058913, 2.915964, 0)`, `v2 ≈ (-3.451565, 0.910315, 0) km/s`.
  Every Lambert library reports against this. Use as golden vector #1.
- **Curtis Example 5.2** (Earth→Mars 2024 H1 launch window): canonical
  porkchop input. Use as golden vector #2.
- **Izzo 2014 §5 random suite**: 10⁹ uniform-on-sphere geometries × 10⁵ TOF
  ratios — too large for goldens; sample 200 reproducible draws via
  `crypto/rand` seed = "lambert-izzo-2014-§5".
- **lamberthub** ships ~30 multi-rev test cases that distinguish Izzo from
  Battin at the ULP level. Use as golden vector #3+.

## Concrete recommendations

### T0 — Izzo 2014 single-rev universal Lambert (~250 LOC) [DAY-1 PR]
Place in **new file** `orbital/lambert.go`. Signature mirrors 164-L1 spec:
```go
// LambertIzzo solves the orbital boundary-value problem.
// Given two position vectors r1, r2 (m), TOF dt (s), gravitational
// parameter mu (m^3/s^2), and a prograde flag, returns the velocity
// vectors v1, v2 (m/s) at the two endpoints of the connecting orbit.
//
// Reference: D. Izzo, "Revisiting Lambert's problem", Celestial Mechanics
// and Dynamical Astronomy 121:1 (2015), arXiv:1403.2705. Algorithm is the
// Householder-3 iteration on the L-similarity-transformed x parameter,
// with the closed-form initial guess of §3.1.
//
// Valid range: |r1|>0, |r2|>0, dt>0, mu>0, transfer angle theta in (0, 2pi).
// Precision: 1e-12 in v1, v2 components for theta in [0.05, 6.23] rad.
// Failure modes: returns NaN vectors for theta in {0, pi, 2pi}; returns
// NaN for dt below the parabolic minimum (no solution exists).
func LambertIzzo(r1, r2 [3]float64, dt, mu float64, prograde bool) (v1, v2 [3]float64, ok bool)
```
Internal kernel: pure-scalar Householder-3 on `x ∈ (-1, ∞)`; the vector
reconstruction is 30 LOC of cross-product / dot-product on top. Implement the
series expansion around `x = 1` (Izzo §2.4) — copy the 8-term expansion
verbatim from pykep `_lambert_problem.cpp`.

**Goldens (~30 vectors):** Vallado 7-5, Curtis 5.2, three near-parabolic
geometries (`x = 1 ± 1e-3`), three near-singular geometries (`θ → π`), 22
random geometries from a fixed PRNG seed cross-validated against pykep
output (Python script in `testdata/scripts/lambert_izzo_pykep.py`, MIT).

### T1 — Multi-revolution branch (~120 LOC) [WEEK-1 PR]
Extension of T0 same file. Adds `Nrev int` parameter; for `Nrev ≥ 1`,
calls Halley-iterates on `dT/dx = 0` to bracket `x_T*(N)`, then Householder-3
on each side. Returns *two* solutions per `(Nrev, prograde)` quadruple
(left/right of `x_T*`).

```go
// LambertIzzoMultiRev returns up to 2*(Nrev+1) solutions: short-way and
// long-way for each revolution count 0..Nrev. Each output index k corresponds
// to (Nrev = k/2, branch = k%2 in {short, long}). Solutions that violate
// the dt < Tmin(Nrev) existence boundary are returned with ok=false.
func LambertIzzoMultiRev(r1, r2 [3]float64, dt, mu float64, prograde bool, Nrev int) ([]LambertSolution, []bool)
```

Test surface adds the **multi-rev existence boundary** golden: known closed-
form `T_min(N=1)` for `θ = π/2`, validated against pykep `lambert_problem`
constructor output.

### T2 — Cross-validation oracle: Battin 1999 successive subs (~250 LOC)
Place in `orbital/lambert_battin.go`. Same signature as `LambertIzzo`.
Implements Battin's `(x, y)` successive substitutions with the
hyperbolic-tangent transformation (Battin §7.6; see also pykep
`_lambert_problem_battin.cpp` for the canonical port). Slower (~5× per call)
but historically referenced as the JPL Horizons internal solver.

**Sole purpose of Battin** in `reality`: the **R-MUTUAL-CROSS-VALIDATION 3/3
pin** witness against Izzo. The pattern from commit 6a55bb4 (audio onset
3-detector cross-validation) is the template. Specific assertions:

1. **Pin A (mutual agreement on canonical suite):** for the 30-vector Vallado
   + Curtis + lamberthub corpus, `‖LambertIzzo(...) - LambertBattin(...)‖_∞ <
   1e-11` per component. Saturates `R-PIN-MUTUAL-AGREEMENT 1/3`.
2. **Pin B (Izzo zero-rev → straight line at small Δt):** synthesise
   `r1 = (R, 0, 0)`, `r2 = R · (cosθ, sinθ, 0)` with `θ = 0.01 rad`,
   `Δt → 0⁺`. Expect `v1, v2 → (r2-r1)/Δt`. Saturates
   `R-PIN-PARABOLIC-LIMIT 2/3`.
3. **Pin C (time-reversal symmetry):** `LambertIzzo(r1, r2, Δt) =
   reverse(LambertIzzo(r2, r1, Δt))` for the long-way branch (sign-flipped
   prograde). Saturates `R-PIN-TIME-REVERSAL 3/3`.

Three pins, three independent algorithms, three closed forms — full
saturation of the R-MUTUAL-CROSS-VALIDATION 3/3 pattern documented in repo
commits 6a55bb4 (audio), 365368a (autodiff), 154-* (chaos round-trip).

### T3 — Optional: Gooding 1990 / Lancaster-Blanchard tertiary cross-check (~210 LOC)
**Skip in v1.** Battin (T2) already gives the second independent solver; a
third does not strengthen the pin. Add only if (a) a user reports an
Izzo-vs-Battin disagreement at ULP level on some geometry, or (b) the
multi-rev `T_min(N)` curve disagrees between Izzo and Battin on a flagged
test case. Defer to issue tracker.

### T4 — Lambert with J2 perturbation (~200 LOC) [FRONTIER, defer]
Per-line `r̈ = -μr/r³ + a_J2(r)` shooting wrapper around T0 with secant on
`v1` to drive endpoint residual to zero. Requires `J2Acceleration` (107-T1.7,
164-L9) to land first. **Frontier feature**: only `STK Astrogator` and
GMAT's `LambertJ2Targeter` ship this; pykep and poliastro do not. Justifies
a separate sprint, gated on consumer demand from mission-planning consumers.

### T5 — Documentation / consumer cross-link (zero-LOC)
1. Update `orbital/orbital.go` package docstring (line 1–17) to point to
   `lambert.go` for non-Hohmann transfer geometries.
2. Cite Izzo 2014 in the `HohmannTransfer` godoc as "the special-case
   subset of `LambertIzzo` for two coplanar circular orbits".
3. Add to `CLAUDE.md` quick-reference (line 22, package table) once T0 lands:
   `orbital | + Lambert (Izzo universal, multi-rev)` row update.
4. Cross-reference 187-S0 (orbital-control synergy) — its station-keeping
   targeter consumes `LambertIzzo`; flagging this dive in 187's docstring
   removes the "blocked on Lambert" line from that synergy.

## Consumer cross-links
- **Mission planning** (164-L3 porkchop, 164-L4 intercept, 164-L5 rendezvous,
  164-L6 patched-conics): all reduce to repeated Izzo calls. T0 unlocks all
  five.
- **Asteroid rendezvous**: NASA Psyche, ESA Hera, JAXA Hayabusa-2 mission
  classes — Lambert is the workhorse for trajectory chunking between flybys.
- **Earth-Mars porkchop**: 164-§Sprint-3 demo case; Curtis Example 5.2
  reproduction is the headline acceptance test.
- **Gravity assist (164-L6)**: heliocentric Lambert + sphere-of-influence
  hand-off; T0 is the heliocentric leg.
- **Sims-Flanagan low-thrust (164-L11)**: each match-point segment is a
  Lambert call between mid-segment states; T0 is the inner kernel of T11.
- **Slot 187-S0 station-keeping**: target-state acquisition under
  Hill-Clohessy-Wiltshire is a Lambert problem in the rotating frame; T0
  feeds 187's S6 rendezvous primitive.
- **Slot 342 (just done) universal-variable Kepler**: the Stumpff-based
  forward propagator is the natural cross-check for Lambert (forward-and-back
  round-trip pin: `KeplerProp(r1, v1, Δt) = (r2, v2)` then
  `LambertIzzo(r1, r2, Δt) ≈ (v1, v2)` to 1e-10 — Pin C complement).
- **`aicore`** (per CLAUDE.md "Dependency Position"): any orbital-targeting
  consumer (visualizer, simulation, mission-design tool) imports
  `reality/orbital`; T0 is on the import-from list once landed.

## Sources
- **Repo files audited:**
  - `C:\limitless\foundation\reality\orbital\orbital.go` (267 LOC, 8 fns)
  - `C:\limitless\foundation\reality\orbital\orbital_test.go`
  - `C:\limitless\foundation\reality\reviews\overnight-400\agents\107-orbital-missing.md` (T1.6 Izzo Lambert spec, 350 LOC est.)
  - `C:\limitless\foundation\reality\reviews\overnight-400\agents\108-orbital-sota.md` (poliastro/Orekit/GMAT/pykep portability matrix)
  - `C:\limitless\foundation\reality\reviews\overnight-400\agents\164-synergy-orbital-optim.md` (L1 LambertIzzo synergy spec, 280 LOC, eight-PR ordering)
  - `C:\limitless\foundation\reality\reviews\overnight-400\agents\187-synergy-orbital-control.md` (S6 rendezvous downstream consumer)
  - `C:\limitless\foundation\reality\reviews\overnight-400\agents\342-dive-kepler-conic.md` (sister Kepler dive; complementary forward-propagator pin)
  - `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md` line 343
  - Repo commit `6a55bb4` audio-onset 3-detector R-MUTUAL-CROSS-VALIDATION pattern (template)
- **Primary literature:**
  - Izzo, D. "Revisiting Lambert's problem." *Celestial Mechanics and Dynamical Astronomy* 121:1 (2015). arXiv:1403.2705. ESA-ACT preprint at `https://www.esa.int/gsp/ACT/doc/MAD/pub/ACT-RPR-MAD-2014-RevisitingLambertProblem.pdf`. **Primary T0 reference.**
  - Lancaster, E. R. & Blanchard, R. C. "A unified form of Lambert's theorem." NASA Technical Note D-5368, 1969. **Foundational unification of `x` parameter.**
  - Gooding, R. H. "A procedure for the solution of Lambert's orbital boundary-value problem." *Celestial Mechanics and Dynamical Astronomy* 48 (1990) 145–165. **Halley-cubic predecessor; T3 alternate.**
  - Battin, R. H. *An Introduction to the Mathematics and Methods of Astrodynamics, Revised Edition.* AIAA Education Series, 1999. Ch. 7 §7.6. **T2 cross-validation algorithm.**
  - Vallado, D. A. *Fundamentals of Astrodynamics and Applications, 4th ed.* §7. **Test vectors (Example 7-5), bisection bounds zup=4π², zlow=-4π.**
  - Curtis, H. *Orbital Mechanics for Engineering Students.* §5.3 Example 5.2. **Earth-Mars porkchop golden vector source.**
  - de la Torre Sangrà, D. & Fantino, E. "Review of Lambert's problem." arXiv:2104.05283. **Complete algorithm survey, 2021.**
  - Sims & Flanagan. "Preliminary design of low-thrust interplanetary missions." AAS 99-338 (1999). **T1.11 / 164-L11 downstream consumer.**
  - von Stryk, O. & Bulirsch, R. "Direct and indirect methods for trajectory optimization." *Annals of Operations Research* 37 (1992) 357–373. **T4 multi-rev / low-thrust extension reference.**
- **Reference implementations (golden-file source-of-truth):**
  - ESA pykep `pykep/core/_lambert_problem.cpp` (MIT). `https://esa.github.io/pykep/examples/ex2.html`
  - poliastro/hapsira `hapsira/iod/izzo.py` (BSD-3). `https://docs.poliastro.space/en/stable/examples/Multirevolutions%20solution%20in%20Lamberts%20problem.html`
  - lamberthub (Piloto, MIT) — 8-solver benchmark suite. `https://github.com/jorgepiloto/lamberthub`
  - Vallado MATLAB `lambertu.m` (poliastro/vallado-software fork). `https://github.com/poliastro/vallado-software/blob/master/matlab/lambertu.m`
  - GMAT `LambertTargeter` (NASA, Apache-2). Mission-design oracle.
- **Web research executed 2026-05-09:** "Izzo 2014 revisiting Lambert problem Householder iteration universal solver"; "Lancaster Blanchard 1969 unified form Lambert problem Gooding 1990"; "Battin 1987 hyperbolic tangent transformation Lambert problem successive substitutions"; "multi-revolution Lambert solver Izzo benchmark Earth Mars porkchop pykep"; "Sun Vallado 1989 simplified Gauss Lambert algorithm".
