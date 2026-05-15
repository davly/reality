# 027 — chaos: missing systems & analysis tools

**Agent:** 027 / 400
**Topic:** chaos-missing — enumerate canonical chaotic systems and integrators NOT yet in `chaos/`
**Date:** 2026-05-07
**Confirmed by 026:** package contents are `ode.go` (RK4Step, EulerStep, SolveODE), `systems.go` (Lorenz, Rössler, Lotka-Volterra, SIR, Van der Pol, LogisticMap, GameOfLife), `analysis.go` (LyapunovExponent for 1D maps, BifurcationDiagram, RecurrencePlot). Confirmed Hénon, Mackey-Glass, Duffing, Chua, Kuramoto-Sivashinsky, BZ all absent.

## Verified gap

`C:/limitless/foundation/reality/chaos/` ships **2 integrators** (RK4 + forward Euler) and **6 dynamical systems**:

| Present | Type | Status |
|---|---|---|
| Lorenz | 3D continuous flow | yes |
| Rössler | 3D continuous flow | yes |
| Lotka-Volterra | 2D continuous flow | yes |
| SIR | 3D continuous flow | yes |
| Van der Pol | 2D continuous (unforced) | yes |
| Logistic | 1D iterated map | yes |
| Game of Life | 2D cellular automaton | yes |

Cross-reference vs DynamicalSystems.jl `PredefinedDynamicalSystems` (2024-2026 release, ~70 systems), `ChaosTools.jl` analysis stack, `pynamical`, `nolds`, `pyrqa`: **chaos covers ~10% of the canonical surface.** The package is missing the entire discrete-map family except logistic, every named delay-differential / PDE / Hamiltonian / piecewise-smooth chaotic system, and 9 of the 10 named analysis tools in the topic prompt (only RecurrencePlot ships; RQA/Lyapunov-from-time-series/D₂/Higuchi/0-1/permutation-entropy all absent).

This agent enumerates the missing surface in three tiers, every entry citation-grounded, every entry golden-file-testable. Tier 1 = textbook canonical, ≤120 LOC each, single-author overnight; Tier 2 = high-leverage modern additions; Tier 3 = research-grade or PDE-tier (defer or coordinate cross-package).

---

## Tier 1 — must ship (textbook canonical, ≤120 LOC each, golden-file ready)

These are systems and tools every chaos library on PyPI/CRAN/Julia ships, every nonlinear-dynamics textbook (Strogatz, Ott, Hilborn, Alligood-Sauer-Yorke) lists, and every consumer named in `ode.go`'s Consumers tag (Pistachio/Pulse/Oracle/Muse/Horizon) will hit within their first dozen experiments.

### T1 systems

| # | System | Reference | DOF | Why Tier 1 |
|---|---|---|---|---|
| T1.S1 | **Hénon map** `(x,y) → (1−ax²+y, bx)` | Hénon 1976 | 2D map | The canonical 2D dissipative map; constant-Jacobian (det = b) gives an exact area-contraction unit test; standard params (a=1.4, b=0.3) → strange attractor with D₀≈1.261. Also fills 026-N11. ~10 LOC. |
| T1.S2 | **Standard map** (Chirikov-Taylor) `pₙ₊₁ = pₙ + K sin θₙ; θₙ₊₁ = θₙ + pₙ₊₁ mod 2π` | Chirikov 1969, 1979 | 2D area-preserving map | The canonical Hamiltonian/symplectic map; KAM-tori-to-chaos transition at K_c≈0.9716 (Greene's residue). Pinning K_c is a fixed-precision golden test. ~15 LOC. |
| T1.S3 | **Tent map** `x → r·min(x, 1−x)` and **Bernoulli shift** `x → 2x mod 1` | classical | 1D map | Closed-form Lyapunov exponent (`ln r` for tent; `ln 2` for Bernoulli); the only systems with exact analytic invariant measures suitable as λ-estimator unit tests. ~10 LOC combined. |
| T1.S4 | **Ikeda map** `(x,y) → (1+u(x cos t − y sin t), u(x sin t + y cos t))`, t = 0.4 − 6/(1+x²+y²) | Ikeda 1979 | 2D map | Optical-cavity model; standard u=0.6/0.9. Strange attractor with multi-scroll structure. ~20 LOC. |
| T1.S5 | **Tinkerbell map** `(x,y) → (x²−y²+ax+by, 2xy+cx+dy)` | Aulbach-Colonius 1999 (popularised) | 2D map | Standard chaos-textbook map (Wikipedia featured). ~15 LOC. |
| T1.S6 | **Lozi map** (piecewise-linear Hénon) `(x,y) → (1−a|x|+y, bx)` | Lozi 1978 | 2D map | Piecewise-linear; explicitly proven SRB measure (Misiurewicz 1980). The only Tier-1 strange attractor with a rigorous existence proof. ~10 LOC. |
| T1.S7 | **Duffing oscillator (forced)** `ẍ + δẋ + αx + βx³ = γ cos(ωt)` | Duffing 1918 | 2D non-autonomous | Standard params (δ=0.3, α=−1, β=1, γ=0.5, ω=1.2) → period-doubling cascade. The canonical forced nonlinear oscillator; cross-references with `control` (driven systems) and `acoustics` (nonlinear resonance). ~15 LOC. |
| T1.S8 | **Chen system** `ẋ = a(y−x); ẏ = (c−a)x − xz + cy; ż = xy − bz` | Chen-Ueta 1999 | 3D flow | The "double Lorenz" — same family, distinct topology. Standard params (a=35, b=3, c=28). ~10 LOC. |
| T1.S9 | **Lü system** `ẋ = a(y−x); ẏ = −xz + cy; ż = xy − bz` | Lü-Chen 2002 | 3D flow | The bridge between Lorenz and Chen (homotopy in c). Trio (Lorenz/Chen/Lü) is canonical Lorenz-family taxonomy. ~10 LOC. |
| T1.S10 | **Rössler hyperchaos** (4D Rössler) `ẋ=−y−z; ẏ=x+ay+w; ż=b+xz; ẇ=−cz+dw` | Rössler 1979 | 4D flow | Two positive Lyapunov exponents (only Tier-1 system that does); exercises any spectrum estimator's ability to distinguish multiple positive λ's. ~15 LOC. |
| T1.S11 | **Hénon-Heiles** `H = ½(p_x²+p_y²) + ½(x²+y²) + x²y − y³/3` | Hénon-Heiles 1964 | 4D Hamiltonian | The textbook chaotic Hamiltonian; energy E=1/6 is the classic transition. Pairs with the 026-N2 symplectic integrator request — this system *requires* a symplectic integrator to test honestly. ~20 LOC. |
| T1.S12 | **FitzHugh-Nagumo neuron** `v̇ = v − v³/3 − w + I; ẇ = ε(v + a − bw)` | FitzHugh 1961, Nagumo et al 1962 | 2D flow | Reduced Hodgkin-Huxley; ε≈0.08, a=0.7, b=0.8, I=0.5. Limit-cycle and excitability regimes. The "chaos for neuroscience" entry. ~10 LOC. |
| T1.S13 | **Hindmarsh-Rose neuron** `ẋ = y − ax³ + bx² − z + I; ẏ = c − dx² − y; ż = r(s(x − x_R) − z)` | Hindmarsh-Rose 1984 | 3D flow | Bursting & spiking dynamics; standard params (a=1, b=3, c=1, d=5, r=0.006, s=4, x_R=−1.6, I=3.25). Adds chaotic burst structure (slow-fast). ~15 LOC. |
| T1.S14 | **Mackey-Glass** `ẋ(t) = β x(t−τ) / (1 + x(t−τ)ⁿ) − γ x(t)` | Mackey-Glass 1977 | DDE (delay diff eq) | The canonical infinite-dimensional chaotic system (delay → infinite-D phase space); standard τ=17, β=0.2, γ=0.1, n=10 → strange attractor. **REQUIRES** a delay-buffer integrator (~30 LOC scaffolding) — first DDE in repo, blocks any time-series-with-memory consumer. ~50 LOC system + integrator. |
| T1.S15 | **Kuramoto coupled oscillators** `θ̇_i = ω_i + (K/N) Σⱼ sin(θⱼ − θᵢ)` | Kuramoto 1975 | N-D flow | Synchronisation phase transition at K_c = 2/(π g(0̄)) for natural-freq density g; canonical mean-field coupled oscillator. The Pulse/Horizon (trend coupling) consumer's model of choice. ~20 LOC. |

### T1 analysis tools

| # | Tool | Reference | Why Tier 1 |
|---|---|---|---|
| T1.A1 | **Poincaré section** (hyperplane crossings, linear interpolation) | Poincaré 1899 | Universal phase-space dimensionality reducer; substrate for bifurcation diagrams over flows (vs maps); ~50 LOC. The named missing primitive in the topic prompt. |
| T1.A2 | **Bifurcation diagram for flows** (parameter scan + Poincaré section) | Strogatz Ch. 12 | Currently `BifurcationDiagram` is map-only; flow version composes T1.A1 + parameter sweep + transient-discard. ~40 LOC additive. |
| T1.A3 | **Largest Lyapunov from time series — Rosenstein method** | Rosenstein-Collins-De Luca 1993 | Robust to short, noisy series; the standard non-spectral λ₁ estimator; doesn't need Jacobian; pairs with the 026-N3 Benettin estimator (which does need Jacobian). ~80 LOC. |
| T1.A4 | **Largest Lyapunov from time series — Kantz method** | Kantz 1994 | Companion to Rosenstein; better noise robustness, slower. Reference implementation in `nolds` and `TISEAN`. ~80 LOC. |
| T1.A5 | **Correlation dimension D₂ (Grassberger-Procaccia)** | Grassberger-Procaccia 1983 | The classical fractal dimension estimator; C(r) ∝ r^D₂ scaling region fit. Directly consumed by Lorenz/Rössler/Hénon strange-attractor tests. ~60 LOC. |
| T1.A6 | **Box-counting dimension D₀** | Mandelbrot 1982 | Capacity dimension; geometric, model-free. Pairs with D₂ (Hentschel-Procaccia spectrum). ~40 LOC. |
| T1.A7 | **Higuchi fractal dimension** | Higuchi 1988 | Time-series fractal dim without phase-space embedding; widely used in EEG/biomedical signal. ~30 LOC. |
| T1.A8 | **0-1 test for chaos** (Gottwald-Melbourne) | Gottwald-Melbourne 2004, 2009 | Binary-output (regular vs chaotic) test directly on time series; no phase-space reconstruction; cited 2,000+ times. ~50 LOC. |
| T1.A9 | **Permutation entropy** (Bandt-Pompe) | Bandt-Pompe 2002 | Order-pattern complexity measure; embedding-free, fast, robust. The de-facto entropy on observed series in 2020s nonlinear dynamics. ~40 LOC. |
| T1.A10 | **Recurrence Quantification Analysis (RQA)** | Marwan-Romano-Thiel-Kurths 2007 (Phys Rep) | Reality already has `RecurrencePlot` — RQA is the natural follow-on: RR (recurrence rate), DET (determinism / diagonal-line fraction), L_max, ENTR (Shannon entropy of diagonal lengths), LAM (laminarity / vertical-line fraction), TT (trapping time). ~120 LOC; turns a binary matrix into seven scalar diagnostics. |
| T1.A11 | **Time-delay embedding** (Takens) + **mutual-information / FNN heuristics** | Takens 1981; Fraser-Swinney 1986; Kennel-Brown-Abarbanel 1992 | Substrate for every time-series analysis tool above (D₂, Rosenstein, Kantz all need an embedding). Mutual-info picks delay τ; false-nearest-neighbours picks dim m. ~100 LOC combined. |

**Tier-1 total:** 15 systems (~225 LOC) + 11 analysis tools (~690 LOC) ≈ 915 LOC + 26 golden-file JSONs. All citation-grounded, all backwards-compatible, all single-author overnight.

---

## Tier 2 — should ship (modern, validated, fills a real gap)

| # | Item | Reference | Type | Why Tier 2 |
|---|---|---|---|---|
| T2.S1 | **Lorenz 84** (general circulation) | Lorenz 1984 | 3D flow | Climate-prediction toy; secondary Lorenz system in DynamicalSystems.jl. ~10 LOC. |
| T2.S2 | **Lorenz 96** (N-D atmospheric model) | Lorenz 1996 | N-D flow | The standard data-assimilation benchmark; configurable N (typically 5, 36, 40); used in every modern Kalman-filter / ensemble-DA paper. ~15 LOC. |
| T2.S3 | **Rabinovich-Fabrikant** | Rabinovich-Fabrikant 1979 | 3D flow | Plasma-physics chaos; multiple coexisting attractors. ~15 LOC. |
| T2.S4 | **Chua's circuit** (3 sub-cases: cubic, piecewise-linear, smooth Chua) | Chua-Komuro-Matsumoto 1986 | 3D flow | The canonical electronic chaos circuit; double-scroll attractor; piecewise-linear nonlinearity exercises non-smooth ODE corners. Cross-references `em` (Chua diode is an EM circuit element). 3 sub-cases × ~15 LOC = ~45 LOC. |
| T2.S5 | **Belousov-Zhabotinsky / Oregonator (Field-Körös-Noyes)** `ẋ = q y − xy + x(1−x); ẏ = (−q y − xy + f z)/ε; ż = x − z` | Field-Körös-Noyes 1972; Field-Noyes 1974 | 3D flow (stiff!) | The canonical chemical-oscillation chaos; **stiff system** (timescale separation ~10⁵) — exercises the 026-N10 stiff-solver gap directly. Used to validate every Rosenbrock/BDF integrator. ~25 LOC system, requires stiff solver. |
| T2.S6 | **Restricted three-body problem** (planar circular) | Euler 1772; Hill 1878 | 4D Hamiltonian | Canonical Hamiltonian chaos; Lagrange points L₁-L₅ as fixed-point unit tests; exact Jacobi integral for energy-conservation tests. Cross-references `orbital` (vis-viva, escape velocity). ~40 LOC. |
| T2.S7 | **Coupled-map lattices** (CML) `x_{n+1}(i) = (1−ε) f(x_n(i)) + (ε/2)(f(x_n(i−1)) + f(x_n(i+1)))` | Kaneko 1985, 1989 | N-D map | Spatially-extended discrete chaos; substrate for spatiotemporal-chaos analysis (pattern formation, defect dynamics). ~30 LOC + boundary-condition options. |
| T2.S8 | **Double pendulum** | classical | 2D Hamiltonian | Visually-iconic Hamiltonian chaos; Lagrangian L = T−V derivation. Pairs with T1.S11 (Hénon-Heiles) and 026-N2 symplectic integrator. ~30 LOC. |
| T2.S9 | **Sprott systems** (19 algebraically simplest 3D chaotic flows) | Sprott 1994 | 3D flows | The 19 minimal chaotic Jerk systems; exhaustive enumeration of "simplest possible chaos." Useful as a regression test grid. ~5 LOC each × 19 = ~95 LOC. Defer all 19 if budget-bound; ship Sprott-A through Sprott-D. |

### T2 analysis tools

| # | Tool | Reference | Why Tier 2 |
|---|---|---|---|
| T2.A1 | **Sample entropy / Approximate entropy** | Pincus 1991; Richman-Moorman 2000 | Standard physiological signal complexity measure; complements permutation entropy. ~50 LOC each. |
| T2.A2 | **Multiscale entropy** | Costa-Goldberger-Peng 2002 | Coarse-grained sample entropy across timescales; detects long-range correlation structure. ~30 LOC layered on T2.A1. |
| T2.A3 | **Detrended fluctuation analysis (DFA)** + Hurst exponent | Peng et al 1994 | Power-law scaling of variance; Hurst H. The standard tool for long-range memory in time series. ~70 LOC. |
| T2.A4 | **Multifractal DFA (MF-DFA)** | Kantelhardt et al 2002 | q-th order fluctuation exponents → multifractal spectrum f(α). ~120 LOC layered on T2.A3. |
| T2.A5 | **Wolf algorithm** for largest Lyapunov from series | Wolf-Swift-Swinney-Vastano 1985 | Earlier alternative to Rosenstein/Kantz; named in 026-N3 already (their version assumes Jacobian access; series-only variant fits here). ~80 LOC. |
| T2.A6 | **Joint recurrence plot** & **cross-recurrence plot** | Marwan et al 2002 | Two-series coupling diagnostics; extends `RecurrencePlot` to bivariate. ~50 LOC. |
| T2.A7 | **Recurrence-network metrics** | Donner-Zou-Donges-Marwan-Kurths 2010 | Treats recurrence matrix as adjacency matrix → graph metrics (clustering, transitivity). Cross-references `graph` (which has Dijkstra/A*/network analysis). ~40 LOC composing existing graph primitives. |
| T2.A8 | **Transfer entropy** (Schreiber) | Schreiber 2000 | Information-theoretic causal-direction measure; the chaos-community's Granger-causality replacement. Could live in `prob` (info-theory) — cross-package coordination needed. ~80 LOC. |

**Tier-2 total:** 9 systems (~285 LOC, more for stiff-solver substrate) + 8 analysis tools (~520 LOC) ≈ 805 LOC. Several need cross-package coordination (T2.S5 needs stiff solver, T2.A7 composes `graph`, T2.A8 should live in `prob`).

---

## Tier 3 — research-grade, PDE-tier, or large coordination scope

| # | Item | Reference | Why Tier 3 |
|---|---|---|---|
| T3.1 | **Kuramoto-Sivashinsky PDE (1D)** `u_t + u·u_x + u_{xx} + u_{xxxx} = 0` | Kuramoto 1976; Sivashinsky 1977 | Canonical spatiotemporal-chaos PDE; needs spectral (Fourier) discretisation + ETDRK4 (Cox-Matthews 2002) integrator. ~250 LOC, requires `signal.FFT` (exists). The PDE-chaos showpiece, but reality is currently ODE-only — opens a PDE-tier scope question for the package as a whole. |
| T3.2 | **Complex Ginzburg-Landau** `A_t = A + (1+ic₁)A_{xx} − (1+ic₂)|A|²A` | classical | Same scope-decision as T3.1; complex-valued PDE chaos. |
| T3.3 | **Full N-body / restricted N-body integrators** (Verlet, Wisdom-Holman, IAS15) | Wisdom-Holman 1991; Rein-Spiegel 2015 | High-precision symplectic + adaptive symplectic; defer to a future `nbody` package or `orbital` extension. |
| T3.4 | **Stochastic chaotic systems** (Lorenz with additive/multiplicative noise; SDE integrators — Euler-Maruyama, Milstein, stochastic Heun) | Kloeden-Platen 1992 | Itô vs Stratonovich primitive belongs in `prob` (stochastic calculus); chaos imports the SDE substrate. Cross-package coordination required. |
| T3.5 | **Symbolic dynamics & topological entropy** (kneading sequences, Bowen-Ruelle pressure) | Milnor-Thurston 1988 | Computer-algebra-flavour; defer. |
| T3.6 | **Unstable periodic orbit (UPO) detection** (Schmelcher-Diakonos, Newton-Raphson root-finding in shift-mapped phase space) | Schmelcher-Diakonos 1997 | UPO skeleton of strange attractors; advanced topic, ~200 LOC, no current consumer. |
| T3.7 | **Inverse problems / chaos control** (OGY, delayed-feedback Pyragas) | Ott-Grebogi-Yorke 1990; Pyragas 1992 | Stabilising UPOs; cross-references `control` (PID, transfer functions) — natural future extension. |
| T3.8 | **Empirical-mode decomposition / Hilbert-Huang for chaotic series** | Huang et al 1998 | Adaptive time-frequency tool; ~300 LOC; arguably belongs in `signal`. |
| T3.9 | **Synchronisation order parameters** beyond Kuramoto: phase-locking value, generalised synchronisation, complete-synchronisation tests | Pikovsky-Rosenblum-Kurths 2001 | Family of ~10 sync diagnostics; defer until at least 2 chaos consumers exist. |
| T3.10 | **Reservoir computing for chaotic prediction** (echo-state networks) | Jaeger 2001; Pathak et al 2018 | ML-flavour; arguably out of scope for a zero-dep math library; defer. |

---

## Cross-package coordination notes

- **026 (chaos-numerics) overlap:** T1.S1 (Hénon) appears in 026-N11 as a 10-LOC drop. T1.S11 (Hénon-Heiles) and T2.S8 (double pendulum) are blocked on 026-N2 (symplectic integrators) for honest energy-conservation tests. T2.S5 (BZ/Oregonator) is blocked on 026-N10 (stiff solver — Rosenbrock-Wanner). T1.A11 / T2.A5 (Wolf series) are the time-series analogues of 026-N3 (Benettin, Jacobian-based). Recommended sprint order: 026-N1 (workspace) → 026-N2 (symplectic) → 026-N9 (DOPRI5) → 026-N10 (stiff) → 027 systems can then land at zero numerics-debt.
- **`linalg`:** T1.A11 (Takens embedding + FNN) and T2.A6/A7 (joint/network recurrence) compose `linalg` matrix ops; no new linalg primitives needed.
- **`prob`:** T1.A9 (permutation entropy), T2.A1-A4 (sample/approx/multiscale entropy, DFA/MF-DFA), T2.A8 (transfer entropy) all rely on Shannon-entropy primitives that should live in `prob` (which already has "information theory" per the CLAUDE.md table). Coordinate: thin chaos-side wrapper, fat prob-side primitive.
- **`signal`:** T3.1 (Kuramoto-Sivashinsky) requires `signal.FFT` (exists per CLAUDE.md). T1.A11 (mutual-information embedding) wants `signal` window functions for spectral-MI variants.
- **`graph`:** T2.A7 (recurrence-network) composes `graph.Dijkstra` / clustering. Pure additive chaos-side wrapper.
- **`em`:** T2.S4 (Chua's circuit) crosses `em` (Chua diode is a nonlinear circuit element). Currently fine to keep self-contained; flag for later refactor if `em` grows nonlinear-element support.
- **`orbital`:** T2.S6 (restricted three-body) overlaps `orbital` (Hill sphere, Hohmann, Kepler). Either lives in chaos with Hamiltonian framing, or in orbital with astrodynamics framing. Recommend chaos for now (Hamiltonian-chaos pedagogy), with a `// see also orbital.HillSphere` cross-reference.
- **`combinatorics`:** Symbolic-dynamics work (T3.5) would compose Stirling/Bell-number primitives if it ever lands.

---

## Recommended commit ordering (highest-leverage first)

1. **T1.S1 (Hénon) + T1.S2 (Standard map) + T1.S3 (Tent + Bernoulli)** — three iconic discrete maps, ~35 LOC total, immediately exercises the missing iterated-map test surface and gives `LyapunovExponent` real targets with closed-form λ. Closes 026-N11 in passing.
2. **T1.S7 (Duffing forced) + T1.S12 (FitzHugh-Nagumo) + T1.S13 (Hindmarsh-Rose) + T1.S15 (Kuramoto)** — four flow systems, ~60 LOC total. Immediately useful to Pulse/Horizon (Kuramoto coupling) and any neuroscience-adjacent consumer (FHN/HR).
3. **T1.A1 (Poincaré section) + T1.A2 (bifurcation for flows)** — turns the existing flow systems into the same diagnostic pipeline the existing logistic enjoys. ~90 LOC.
4. **T1.A11 (Takens embedding + MI/FNN) + T1.A3 (Rosenstein λ₁)** — unblocks every "I have a time series, what's its largest Lyapunov?" question. ~180 LOC.
5. **T1.A10 (full RQA: RR/DET/LAM/ENTR/L_max/TT)** — turns the existing `RecurrencePlot` from a viz tool into a quantitative pipeline. ~120 LOC.
6. **T1.A5 + T1.A6 + T1.A7 + T1.A8 + T1.A9** — the dimension/complexity battery (D₂, D₀, Higuchi, 0-1 test, permutation entropy). ~220 LOC. Standard nonlinear-time-series analyst's toolkit; closes the topic prompt's analysis-tools list completely.
7. **T1.S14 (Mackey-Glass + DDE delay-buffer integrator)** — first DDE in repo; ~50 LOC. Opens the delay-equation tier.
8. **T1.S11 (Hénon-Heiles)** — pairs with 026-N2 symplectic integrator landing.
9. **T1.S4/S5/S6 (Ikeda/Tinkerbell/Lozi) + T1.S8/S9/S10 (Chen/Lü/Rössler-hyper)** — the "complete the canonical-map and Lorenz-family taxonomy" PRs. ~70 LOC total.
10. **Tier 2 systems and analysis tools** — sequence as cross-package gates clear (stiff solver for BZ, prob entropy for sample/approx entropy, etc).
11. **Tier 3** — open scope question; defer pending an explicit "do we ship PDE chaos?" decision and a `prob`-side SDE primitive.

**Total Tier 1 surface:** ~915 LOC + ~26 golden-file JSONs (≥20 vectors each per CLAUDE.md). All citation-grounded against Strogatz / Ott / Hilborn / Marwan-Romano-Thiel-Kurths / Kantz-Schreiber / DynamicalSystems.jl. All backwards-compatible. All language-agnostic for Python/C++/C# port-target validation.

---

## Topic-prompt coverage check

| Topic-prompt item | Tier | Where addressed |
|---|---|---|
| Kuramoto-Sivashinsky | 3 | T3.1 (PDE-tier scope question) |
| Hénon | 1 | T1.S1 |
| Rössler | present | (already in `systems.go`) |
| Mackey-Glass | 1 | T1.S14 (first DDE) |
| Duffing | 1 | T1.S7 |
| Chua | 2 | T2.S4 (3 sub-cases) |
| Belousov-Zhabotinsky / Oregonator | 2 | T2.S5 (needs stiff solver) |
| Discrete maps: logistic | present | (already) |
| Discrete maps: Hénon, Standard, Tent, Gauss, Bernoulli, Tinkerbell, Ikeda, Lozi | 1 | T1.S1-S6 |
| Coupled-map lattices | 2 | T2.S7 |
| Lorenz | present | (already) |
| Lorenz 84/96, Rabinovich-Fabrikant | 2 | T2.S1-S3 |
| Chen, Lü | 1 | T1.S8-S9 |
| Hindmarsh-Rose, FitzHugh-Nagumo | 1 | T1.S12-S13 |
| Kuramoto coupled oscillators | 1 | T1.S15 |
| Hénon-Heiles | 1 | T1.S11 |
| Three-body / restricted | 2 | T2.S6 |
| Van der Pol | present | (already) |
| Stochastic (Lorenz+noise; Itô-Stratonovich primitives in prob/) | 3 | T3.4 (cross-package) |
| Bifurcation diagrams (flow / Poincaré) | 1 | T1.A1-A2 |
| Poincaré sections | 1 | T1.A1 |
| Recurrence plots | present | (already) |
| RQA (RR/DET/LAM/ENTR) | 1 | T1.A10 |
| Largest Lyapunov from series — Rosenstein, Kantz | 1 | T1.A3-A4 |
| Correlation dimension D₂ (Grassberger-Procaccia) | 1 | T1.A5 |
| Box-counting / Higuchi fractal dim | 1 | T1.A6-A7 |
| 0-1 test for chaos (Gottwald-Melbourne) | 1 | T1.A8 |
| Permutation entropy (Bandt-Pompe) | 1 | T1.A9 |

Every named topic-prompt item placed; nothing dropped silently.
