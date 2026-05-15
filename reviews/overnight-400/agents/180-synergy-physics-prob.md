# 180 | synergy-physics-prob

**Topic:** physics x prob — Boltzmann/MB/BE/FD distributions, partition functions, Ising/XY/Heisenberg/Lennard-Jones MC, Wang-Landau, replica exchange, cluster algorithms, free energy, critical exponents, Brownian/Langevin/Gillespie/Jarzynski/Crooks, polymer SAW.
**Block:** B (cross-package synergies).
**Date:** 2026-05-08.
**Scope:** capabilities that emerge ONLY when `physics/`, `prob/`, `chaos/`, `crypto/`, and `optim/` are composed; not what either package is missing in isolation. Repo at v0.10.0, 1965 tests, 22 packages. Not a per-package review.

## Two-line summary

Today `physics/` ships exactly **6 thermodynamic functions** (`IdealGas`, `StefanBoltzmann`, `CarnotEfficiency`, `HeatEquation1DStep`, `FourierHeatConduction`, `NewtonCooling`, `ThermalExpansion` — all macroscopic, *zero* statistical-mechanics surface; verified `grep -E 'Boltzmann|Maxwell|Bose|Fermi|Planck|partition|Sackur|equipartition|Helmholtz|Gibbs|Ising|Metropolis|Langevin|Brownian|Wiener|Wang|Landau|Jarzynski|Crooks|Gillespie' physics/*.go` returns only the macroscopic `StefanBoltzmann` total-radiance integral); `prob/` ships Normal/Exp/Uniform/Beta/Gamma/Poisson/Binomial CDFs, `MarkovSteadyState` + `MarkovSimulate` (LCG-only, no detailed-balance/MH acceptance kernel), `LogGamma/Erfc/RegularizedBetaInc`, KL divergence — but knows **nothing** about energy levels, partition functions, the Boltzmann factor, the Maxwell-Boltzmann speed distribution, blackbody photon statistics, lattice spin models, SDE integrators, or the Gillespie SSA. The only existing bridge is `optim/metaheuristic.go::SimulatedAnnealing` (line 73: `p := math.Exp(-delta / temp)`) which already implements Boltzmann acceptance — **but as a generic optimizer, not as a stat-mech sampler**, with no tracking of the Markov-chain-level observables (energy, magnetization, autocorrelation, susceptibility) that physical applications need. Cross-edges: `grep github.com/davly/reality/prob physics/*.go` → 0; reverse → 0; `physics/` imports only `constants/`.

**Eighteen synergy primitives (S1–S18) totalling ~3,180 LOC of pure connective tissue** close the gap. Cheapest one-day PR is **S1 BoltzmannFactor + S2 PartitionFunction + S4 MaxwellBoltzmannSpeedPDF/CDF/Quantile = ~140 LOC** consuming only `constants.Boltzmann` + `prob.LogGamma` and unlocking 12 downstream primitives. Highest-leverage architectural lift is **S10 Ising2DMetropolis + S11 IsingObservables (Cv, chi, Binder cumulant) + S18 OnsagerExact2D (closed-form for golden-file pinning)** at ~520 LOC, because the 2D-Ising/Onsager pair is the canonical R-MUTUAL-CROSS-VALIDATION 3/3 pin (Metropolis MC × Wolff cluster × Onsager analytic agreement on Cv-peak location at βc = ln(1+√2)/2 ≈ 0.4407 to 1% on a 32×32 lattice), mirroring commits 6a55bb4 (audio-onset 3-detector) and 365368a (Clayton autodiff-vs-analytic). Crown jewel is **S15 GillespieSSA + S16 LangevinBAOAB + S17 JarzynskiCrooks** (~620 LOC) — exact stochastic chemical kinetics, BAOAB Langevin integrator (Leimkuhler-Matthews 2013), and Jarzynski/Crooks free-energy estimators are *no-zero-dep-library-ships-this* territory. Recommended placement: **NEW sub-package `physics/statmech/`** mirrors `prob/copula/`, `prob/conformal/`, `optim/transport/`, `optim/proximal/` placement convention. Cycle-free DAG: `physics/statmech` → {`physics/`, `prob/`, `crypto/`, `constants/`, occasionally `linalg/`, `optim/`}; reverse direction never. No new abstraction needed.

---

## 0. State of play (verified file-walk)

`physics/` HEAD (4 files, ~520 LOC total numeric core):

- `thermo.go` (148 LOC): `IdealGas` (PV=nRT), `StefanBoltzmann` (σAT⁴, integrated total radiance only — no spectral Planck distribution), `CarnotEfficiency`, `HeatEquation1DStep` (Dirichlet-BC explicit Euler FD), `FourierHeatConduction`, `NewtonCooling`, `ThermalExpansion`. All scalar, deterministic, macroscopic.
- `mechanics.go`, `materials.go`, `optics.go`: outside synergy scope (covered by 058-physics-* per-package agents).
- **No statistical-mechanics surface.** `grep -E 'Boltzmann|Maxwell|Bose|Fermi|Planck.*spec|partition|micro|canonical|grand|Sackur|equipartition|Helmholtz|Gibbs|Ising|spin|lattice|MC|Metropolis|Langevin|Brownian|Wiener|Wang|Landau|Jarzynski|Crooks|Gillespie|SSA' physics/*.go` → only `StefanBoltzmann` (the integrated form, not the spectral Planck B(ν,T) or B(λ,T)).

`prob/` HEAD (~2,800 LOC top-level + `prob/copula/`, `prob/conformal/`):

- `distributions.go`: `NormalPDF/CDF/Quantile`, `ExponentialPDF/CDF/Quantile`, `UniformPDF/CDF`, `BetaPDF/CDF`, `PoissonPMF/CDF`, `GammaPDF/CDF`, `BinomialPMF/CDF`. Acklam inverse-Normal accurate to 1.15e-9. **No Maxwell-Boltzmann speed, Bose-Einstein, Fermi-Dirac, half-normal, chi (not chi²), or Wigner-Dyson.**
- `distribution.go`: `Distribution` interface + `BetaDist`/`NormalDist`/`ExponentialDist`/`UniformDist` + `KLDivergenceNumerical` (trapezoidal).
- `mathutil.go`: `LogGamma` (essential — needed for BE/FD partition-function normalization), `Erfc`, `RegularizedBetaInc`, private `chiSquaredCDF`, `studentTCDF`, `regularizedGammaLowerSeries`.
- `markov.go`: `MarkovSteadyState` (power iteration to L1 < 1e-12), `MarkovSimulate` (private LCG only, **no detailed-balance check, no Metropolis-Hastings acceptance kernel, no Hastings-corrected proposal**, no autocorrelation, no integrated-autocorrelation-time τ_int).
- `prob.go`: Bayesian update, calibration (Brier, log-loss, ECE, isotonic).
- `hypothesis.go`, `nonparametric.go`, `regression.go`, `timeseries.go`, `jeffreys.go`.
- **Absent (verified):** Maxwell-Boltzmann speed PDF/CDF/Quantile, half-normal, Rayleigh, BE/FD occupation, Planck spectral density as a probability distribution, MGF/CGF/Laplace transform, Brownian increments, Wiener bridge, Ornstein-Uhlenbeck, Langevin SDE step, Gillespie SSA, replica-exchange swap, Wolff cluster build, Wang-Landau density-of-states histogram.

`chaos/` (3 files): `RK4Step`, `EulerStep`, `SolveODE`, `LyapunovExponent`, `BifurcationDiagram`, `RecurrencePlot`, `Lorenz`, `VanDerPol`. **No SDE integrator, no Wiener-process driving term, no symplectic Verlet, no BAOAB.** Agent 027 (chaos-missing) already flags SDE as a Tier-1 gap.

`crypto/rng.go`: `MersenneTwister`, `PCG`, `Xoshiro256` — three deterministic PRNGs with `.Float64()` returning [0,1). Seed plumbing every S1–S18 needs already ships.

`optim/metaheuristic.go::SimulatedAnnealing` (line 38): the **only** existing Boltzmann-factor consumer in the entire repo. Line 73 is literally `p := math.Exp(-delta / temp)`. This is a Markov chain at temperature `temp` accepting moves with Metropolis criterion *but with no concept of energy units, no equilibrium state-tracking, and no observable-averaging machinery*. S1–S18 generalize this kernel from "stochastic optimizer" into "stat-mech sampler with physical observables".

`constants/physics.go`: `Boltzmann = 1.380649e-23 J/K` (SI 2019 exact), `Planck = 6.62607015e-34 J*s` (SI 2019 exact), `PlanckReduced` (h/2π), `Avogadro`, `GasConstant`, `StefanBoltzmann = 5.670374419e-8` (derived; comment at line 63 already states σ = 2π⁵k_B⁴/(15h³c²) — the *Planck-distribution integral* is right there, just not exposed as a spectral function).

`linalg/`: `CholeskyDecompose`, `CholeskySolve` — needed for correlated-Gaussian noise in S16 multi-DOF Langevin and for S14 polymer-SAW pivot moves on D-dimensional embeddings.

**Cross-edges.** `grep -r github.com/davly/reality/prob physics/`: 0. `grep -r github.com/davly/reality/physics prob/`: 0. Pristine — like 173 (queue×prob) and 176 (color×prob), this is a clean synergy with no pre-existing entanglement.

---

## 1. The synergy primitives S1–S18

Each row: capability | composition | LOC of connective tissue (excludes test fixtures).

### Tier 1 — Distributions and ensembles (S1–S5, ~340 LOC)

**S1 `BoltzmannFactor(E, T)` + `BoltzmannProbability(E, energies, T)`** — capability: P(state with energy E) = exp(-βE)/Z given enumerable spectrum. Composition: `constants.Boltzmann` + `math.Exp` + log-sum-exp via `prob.mathutil` style. ~40 LOC. Single most-cited primitive in stat-mech curricula; consumed by S2, S6, S7, S10, S15.

**S2 `PartitionFunction(energies, T)` + `LogPartitionFunction`** — capability: Z(β) = Σ exp(-βE_i) with log-sum-exp for overflow safety. Composition: same as S1 plus a stable log-sum-exp helper (canonical 6-line trick: subtract max). Returns Z and log Z separately because U = -∂ log Z / ∂β and F = -k_B T log Z need log Z at machine precision. ~50 LOC.

**S3 `HeatCapacityFromZ(energies, T)`** — capability: Cv = k_B β² ⟨(δE)²⟩ from the canonical fluctuation-dissipation identity, computed via the variance of the Boltzmann-weighted energy ensemble (no numerical differentiation needed). Composition: S1 + `prob.WeightedAverage` (already in `prob.go:278`) for ⟨E⟩ and ⟨E²⟩. ~50 LOC. **Critical:** computing Cv as ∂U/∂T via finite differences on T is well-known to be noisy for MC; the fluctuation form is exact in the ensemble.

**S4 `MaxwellBoltzmannSpeedPDF/CDF/Quantile(v, m, T)`** — capability: f(v) = 4π v² (m/(2π k_B T))^(3/2) exp(-mv²/(2k_B T)), the speed distribution of an ideal gas. CDF closed form via `math.Erf`. Quantile via Newton on CDF (`prob.RegularizedGammaP` style). Composition: `constants.Boltzmann` + `math.Exp/Erf`. ~80 LOC. Sister functions: `MeanSpeed`, `ModeSpeed`, `RMSSpeed` — all closed form.

**S5 `BoseEinsteinOccupation(E, μ, T)` + `FermiDiracOccupation(E, μ, T)`** — capability: ⟨n⟩ = 1/(exp(β(E-μ)) ∓ 1). Composition: `constants.Boltzmann` + careful overflow handling for β(E-μ) → ±∞. ~50 LOC. Companions: `PlanckPhotonOccupation(ν, T) = 1/(exp(hν/k_B T) - 1)` (μ=0 BE); `FermiEnergyZeroT(N, V, m)` (closed form for free-electron gas).

S1–S5 ship today. Rejection-sampling against MB-speed via `crypto.Xoshiro256.Float64()` saturates an R-MUTUAL-CROSS-VALIDATION 3/3 pin (analytic CDF × inverse-CDF sample × rejection sample agreement on ⟨v⟩, ⟨v²⟩, mode to 0.1% at N=10⁶).

### Tier 2 — Spectral density and ideal gases (S6–S9, ~470 LOC)

**S6 `PlanckSpectralDensity(ν, T)`** + wavelength form **`PlanckSpectralDensityLambda(λ, T)`** — capability: B(ν,T) = (2hν³/c²) / (exp(hν/k_B T) - 1) with proper change-of-variable Jacobian for the wavelength version. Composition: `constants.{Planck, Boltzmann, SpeedOfLight}` + `math.Exp`. **Already half-implicit in `StefanBoltzmann` constant comment** but never exposed as a function. ~100 LOC. Cross-validation: ∫₀^∞ B(ν,T) dν integrated by Simpson (`calculus`) over [10⁻⁵, 10²] × ν_max should equal σT⁴/π to 10⁻⁹. **`WienDisplacementWavelength(T) = b/T`** with b = 2.897771955e-3 m·K closed form.

**S7 `IdealGasPartitionFunction(N, V, T, m)` + `SackurTetrodeEntropy`** — capability: Z₁ = V/λ³ where λ = h/√(2πmk_BT) is the thermal de Broglie wavelength; total Z = Z₁^N / N! with `prob.LogGamma` for N!. S = k_B[N ln(V/(Nλ³)) + 5N/2] (Sackur-Tetrode 1912). Composition: `constants.{Planck, Boltzmann, Avogadro}` + `prob.LogGamma`. ~80 LOC. Anchors **R-EXACT-CHECK** against tabulated S(He, 1 atm, 298 K) = 126.15 J/(mol·K).

**S8 `EquipartitionEnergy(dof, T)` + `EinsteinHeatCapacity(T, T_E)` + `DebyeHeatCapacity(T, T_D)`** — capability: U = (dof/2) k_B T per particle (classical); Einstein C_v = 3R(x²e^x/(e^x-1)²) at x = T_E/T; Debye C_v = 9R(T/T_D)³ ∫₀^(T_D/T) x⁴e^x/(e^x-1)² dx. Composition: `constants.{Boltzmann, GasConstant}` + Simpson integration from `calculus`. ~120 LOC. Debye integral is not closed form — calls into `calculus.SimpsonAdaptive`.

**S9 `BlackbodyMonteCarlo(T, N, rng)`** — capability: sample N photon energies from the Planck distribution by either inverse-CDF (closed form via Lambert W or numerical inversion of the polylogarithm) or rejection. Composition: S6 + `crypto` PRNG + `prob.distributions` for proposal-based rejection. ~170 LOC. Used downstream by radiative-transfer Monte Carlo; orthogonal to the deterministic StefanBoltzmann integral.

### Tier 3 — Lattice models and cluster MC (S10–S12, ~720 LOC)

**S10 `Ising2DMetropolis(L, beta, sweeps, rng)`** — capability: 2D nearest-neighbor Ising on L×L torus, single-spin-flip Metropolis at inverse temperature β. Returns equilibrated configuration + (E, M) trace. Composition: 2D bool/int8 lattice + `crypto.Xoshiro256` + Metropolis acceptance kernel literally identical to `optim/metaheuristic.go::SimulatedAnnealing` lines 70-75 but with **discrete energy increments** that are precomputable as a 5-element lookup table {exp(-2β·k) : k ∈ {-4,-2,0,2,4}}. ~200 LOC. **This is the canonical first stat-mech program.**

**S11 `IsingObservables(trace_E, trace_M, beta, L)` returning {U, Cv, chi, U4, autocorr_time}** — capability: ensemble averages with proper jackknife/bootstrap error bars and integrated autocorrelation time τ_int via Sokal's automatic windowing (Madras-Sokal 1988). Composition: `prob.SimpleAverage`, `prob.WeightedAverage`, plus a 60-LOC Sokal integrator. The **Binder cumulant** U4 = 1 - ⟨M⁴⟩/(3⟨M²⟩²) is the standard finite-size-scaling crossing point (Binder 1981). ~180 LOC.

**S12 `Ising2DWolff(L, beta, sweeps, rng)`** — capability: Wolff (1989) single-cluster algorithm — pick a random spin, recursively add neighbors with bond probability p_add = 1 - exp(-2β), flip the entire cluster as one move. **Critical-slowing-down killer** near βc; near critical point τ_int(Wolff) ~ L^0.27 vs τ_int(Metropolis) ~ L^2.17 — a literally polynomial speedup. Composition: BFS queue + same `crypto` PRNG. ~140 LOC. Variant **S12b `SwendsenWang`** (multi-cluster) shares the same bond-percolation step (~80 LOC delta). Both use the same R-MUTUAL-CROSS-VALIDATION pin against Onsager (S18).

Critical-slowing-down language is a *stat-mech-specific concept* that no general MCMC package (PyMC, Stan, Pyro) ships out of the box; ours would.

### Tier 4 — Advanced MC and replica exchange (S13–S14, ~360 LOC)

**S13 `WangLandau(energies_grid, f_initial, f_min, rng)`** — capability: flat-histogram density-of-states sampling (Wang-Landau 2001). Iteratively learn g(E) by accepting moves with probability min(1, g(E_old)/g(E_new)) and updating g(E) ← g(E)·f at the visited bin until histogram is "flat" (canonical 80% criterion), then refine f → √f. Yields S(E) = log g(E) hence F(T), C_v(T) at *every* T from a single simulation. ~180 LOC. Pure prob/MCMC primitive; lattice connection via S10 callback.

**S14 `ReplicaExchange(samplers, betas, swap_interval)`** — capability: parallel tempering — run M independent samplers at increasing β, periodically propose adjacent-replica swaps with acceptance min(1, exp((β_i - β_j)(E_i - E_j))). Composition: M × S10 (or S15, S16) + a single-line swap kernel. ~180 LOC. Saturates **R-MUTUAL-CROSS-VALIDATION 3/3** when paired with Wolff (S12) and Wang-Landau (S13) on the same critical Ising lattice.

### Tier 5 — Continuous SDE and stochastic kinetics (S15–S17, ~620 LOC)

**S15 `GillespieSSA(reactions, x0, t_end, rng)`** — capability: exact stochastic simulation of chemical-master-equation kinetics (Gillespie 1977). At each step: total propensity a₀ = Σa_i; sample τ ~ Exp(a₀) using `prob.ExponentialQuantile`; sample reaction j with weight a_j/a₀. Returns (t_i, x_i) trajectory. Composition: `prob.ExponentialQuantile` + `crypto.Xoshiro256.Float64`. ~200 LOC. Companion **`TauLeap`** for stiff systems (~80 LOC).

**S16 `LangevinBAOAB(x0, v0, force_fn, m, gamma, T, dt, steps, rng)`** — capability: BAOAB (Leimkuhler-Matthews 2013) splitting integrator for the underdamped Langevin equation m·ẍ = F(x) - γ·m·ẋ + √(2γ·m·k_B·T)·ξ(t). BAOAB is the **only** Langevin integrator with O(dt²) configurational error and *exact* canonical sampling in the high-friction limit — strict improvement over Euler-Maruyama and naive velocity-Verlet+OU. Composition: `chaos.RK4Step`-like step structure + `prob.NormalQuantile` for Wiener increments + `crypto.Xoshiro256`. ~220 LOC. Companions: **`OrnsteinUhlenbeckStep`** (~30 LOC), **`BrownianBridge`** (~40 LOC), **`KramersEscapeRate(barrier_height, omega_well, omega_barrier, gamma, T)`** = (ω_well·ω_barrier)/(2π·γ)·exp(-ΔE/k_B T) closed form (~30 LOC).

**S17 `JarzynskiFreeEnergy(W_samples)` + `CrooksFluctuationTheorem(W_fwd, W_rev)`** — capability: Jarzynski equality F = -k_B T log ⟨exp(-βW)⟩ from non-equilibrium work samples; Crooks: P_fwd(W)/P_rev(-W) = exp(β(W-ΔF)) yields ΔF as the crossing point of the two histograms. Composition: log-sum-exp from S2 + `prob.LinearRegression` for the Crooks slope-1 fit. ~200 LOC. **First non-equilibrium-statistical-mechanics primitive in any zero-dep math library**; this is what AlphaFold-2 and FEP/TI alchemical-free-energy calculations consume daily.

### Tier 6 — Polymers and exact closed forms (S18 + extras, ~660 LOC)

**S18 `OnsagerExact2D(beta)` returning {U, Cv, M, F}** — capability: exact closed-form Onsager (1944) solution for the 2D nearest-neighbor Ising model on the infinite square lattice. β_c = ln(1+√2)/2; M(β) = (1 - sinh⁻⁴(2β))^(1/8) for β > β_c, else 0; U = -coth(2β)·[1 + (2/π)(2tanh²(2β) - 1)·K(k)] with K the complete elliptic integral of the first kind; Cv has the famous logarithmic divergence at β_c. Composition: `math` + 50 LOC of Carlson R_F or AGM for K(k). ~150 LOC. **Pure deterministic golden-file pin** for S10/S12/S14.

**S19 `PolymerSAW(L, dim, n_steps, rng)` (pivot algorithm)** — capability: self-avoiding-walk Monte Carlo on Z^d via the Madras-Sokal (1988) pivot algorithm — pick a random vertex, apply a random lattice symmetry to the tail, accept if no self-intersection. Yields ⟨R²⟩ ~ L^(2ν) with Flory exponent ν ≈ 0.5876 (3D) / 3/4 (2D, exact). Composition: hash-set self-intersection check + `crypto` PRNG + a small symmetry group. ~180 LOC.

**S20 `XYModel2D` + `HeisenbergModel3D`** — capability: continuous-spin O(2) and O(3) models with Metropolis + Wolff embeddings. ~200 LOC each; share most code with S10/S12. Defer if ship-list is tight.

**S21 `LennardJonesMC(N, rho, T, n_sweeps, rng)`** — capability: NVT Monte Carlo of N LJ particles with periodic boundary conditions, `r_cut` and tail correction. Composition: `geometry` for periodic distance + Metropolis kernel. ~130 LOC.

---

## 2. Cross-cutting connective-tissue patterns

Three patterns recur across S1–S21 and should be factored into shared helpers up front to avoid 18× duplication:

**P1: Log-sum-exp `LogSumExp(xs []float64) float64`** — ~12 LOC. Subtract max, sum exponentials, add max back. Consumed by S1, S2, S5, S7, S13, S17. Should live in `prob/mathutil.go` next to the existing `LogGamma`.

**P2: Metropolis acceptance kernel `MetropolisAccept(deltaE, beta float64, rng RNG) bool`** — ~6 LOC. Already inlined in `optim/metaheuristic.go:70-75`; should be promoted to a public `prob.MetropolisAccept(deltaE, beta, rng)` and reused by S10, S12, S14, S16. **One-line refactor of `SimulatedAnnealing` to call the new helper** keeps existing tests green and removes the only duplication.

**P3: Integrated autocorrelation time `IntegratedAutocorrelationTime(trace []float64) (tau float64, err float64)`** — ~80 LOC. Sokal-1997 automatic windowing. Used by S11, S13, S14, S17 to compute honest error bars on MC estimators. Should live in `prob/timeseries.go` next to the existing ARIMA machinery (which already does autocovariance via `arimaAutocovariance` at line 281 — partial reuse).

Factoring P1+P2+P3 first **drops S1–S21 total LOC from ~3,180 to ~2,960**.

---

## 3. Recommended package layout

```
physics/
  thermo.go         (existing)
  mechanics.go      (existing)
  materials.go      (existing)
  optics.go         (existing)
  statmech/         (NEW — 18 files, ~3,000 LOC)
    distributions.go   S1, S2, S3, S4, S5
    spectral.go        S6 (Planck, Wien)
    ideal_gas.go       S7, S8 (Sackur-Tetrode, equipartition, Einstein, Debye)
    blackbody_mc.go    S9
    ising.go           S10, S11
    cluster.go         S12 (Wolff, Swendsen-Wang)
    wang_landau.go     S13
    replica.go         S14
    gillespie.go       S15
    langevin.go        S16, OU, KramersEscape
    free_energy.go     S17 (Jarzynski, Crooks)
    onsager.go         S18 (exact 2D Ising, golden-file pin)
    polymer.go         S19 (pivot SAW)
    xy_heisenberg.go   S20
    lennard_jones.go   S21
    statmech_test.go
    testdata/         golden vectors
prob/
  mathutil.go     ← add P1 LogSumExp (~12 LOC)
  markov.go       ← add P2 MetropolisAccept (~10 LOC, reused by optim)
  timeseries.go   ← add P3 IntegratedAutocorrelationTime (~80 LOC)
optim/metaheuristic.go ← refactor SimulatedAnnealing to call prob.MetropolisAccept (~3 LOC delta)
```

DAG: `physics/statmech` → {`physics/`, `prob/`, `crypto/`, `constants/`, `linalg/`, `calculus/`, `optim/`}. **No new edges introduced into `prob/`, `physics/`, or any existing package's import set** — every existing package keeps its current import list. Reverse direction never.

This mirrors the established sub-package convention: `prob/copula/`, `prob/conformal/`, `optim/transport/`, `optim/proximal/`, `audio/onset/`, `audio/spectral/`. No precedent for sub-packages broken.

---

## 4. R-pattern saturation map

R-MUTUAL-CROSS-VALIDATION 3/3 pins available from this synergy (mirroring 6a55bb4 / 365368a / NGramDiceCoefficient / TokenSetRatio / Soundex precedents):

- **MB-speed:** analytic CDF (S4) × inverse-CDF sampling × rejection sampling → ⟨v⟩ ⟨v²⟩ mode within 0.1% at N=10⁶.
- **Planck/Stefan-Boltzmann:** spectral B(ν,T) integrated by Simpson (`calculus`) × σT⁴/π closed form × Monte Carlo (S9) → equal to 10⁻⁹ relative.
- **2D Ising at βc:** Metropolis (S10) × Wolff (S12) × Onsager exact (S18) → Cv-peak location and U(βc) within 1% at L=32, N_sweeps=10⁶.
- **Replica exchange:** parallel-tempered Metropolis × Wang-Landau density-of-states (S13) × Wolff cluster (S12) → free-energy F(T) curve agreement.
- **Jarzynski/Crooks:** forward S17 × reverse S17 × Bennett-acceptance-ratio (Bennett 1976, ~40 LOC bonus) → ΔF triple agreement on toy double-well.
- **Einstein-relation:** Langevin-MSD (S16) × Green-Kubo velocity-autocorrelation × Stokes-Einstein D = k_B T / (6πηr) closed form.

Six 3/3 cross-validation pins fall out essentially for free from the S1–S21 ship list. By repository convention (commits 85a80db, 1e12e80, 3b8413a, 6a55bb4, 365368a) each is a separate landed test commit.

---

## 5. Cheapest-shipping recommendation (one-day PR)

Minimum viable stat-mech surface, shipping today against v0.10.0:

1. P1 `LogSumExp` (12 LOC) into `prob/mathutil.go`.
2. P2 `MetropolisAccept` (10 LOC) into `prob/markov.go`; refactor `optim.SimulatedAnnealing` to call it (3 LOC delta).
3. **S1 `BoltzmannFactor` + `BoltzmannProbability`** (40 LOC) → `physics/statmech/distributions.go`.
4. **S2 `PartitionFunction` + `LogPartitionFunction`** (50 LOC) → same file.
5. **S4 `MaxwellBoltzmannSpeedPDF/CDF/Quantile/MeanSpeed/RMSSpeed/ModeSpeed`** (80 LOC) → same file.
6. **S6 `PlanckSpectralDensity` + `WienDisplacementWavelength`** (100 LOC) → `physics/statmech/spectral.go`.
7. Test fixtures: 30+ golden vectors per primitive, IEEE 754 edge cases (T → 0 → +Inf, ν → 0, m → 0), R-EXACT-CHECK against NIST tabulated thermodynamic values (Sun T=5778 K, He at STP, photon density at 2.725 K CMB).

Total: ~295 LOC connective tissue + ~600 LOC tests + 6 golden-file vectors files = single-day landable.

This is the **synergy MVP**. S10+S11+S18 (Ising/observables/Onsager 3/3 pin, ~520 LOC) is the high-leverage day-2 PR. S15+S16+S17 (Gillespie/BAOAB/Jarzynski-Crooks, ~620 LOC) is the day-3 PR — and arguably the most differentiated relative to scipy/numpy/torch which ship none of these as zero-dep primitives.

---

## 6. Notes and caveats

- `optim.SimulatedAnnealing` already implements Boltzmann acceptance using `math/rand` (not `crypto.Xoshiro256`). The MCMC primitives in `physics/statmech` should use the better PRNGs in `crypto/rng.go` directly; agent 117/120 prob-perf has likely flagged the LCG-only `MarkovSimulate` as a precision concern.
- `physics.StefanBoltzmann` accepts emissivity ∈ [0,1]; the synergy spectral function `PlanckSpectralDensity` should *not* take emissivity (pure blackbody) and a separate `GreyBodySpectralDensity` should multiply by emissivity. Don't conflate them.
- 2D Ising with periodic boundary conditions has a known finite-size correction to βc; Onsager closed form is for the *infinite* lattice. Golden-file tolerances should accommodate finite-size scaling — recommend tolerance 1/L for Cv-peak β-location at finite L.
- `LogGamma` already exists; resist the temptation to reimplement Stirling. Reuse.
- Kramers escape rate is **closed form** in the moderate-friction limit; full PMF-of-escape-time requires S16 simulation. Document precision regimes.
- Wang-Landau (S13) is **the** single-simulation method for getting F(T) curves at all T — but its convergence proofs are weak. Document that the canonical 80% flatness criterion is heuristic; advanced users should use 1/t-modification (Belardinelli-Pereyra 2007, ~30 LOC delta).
- BAOAB (S16) is a 2013 result; before BAOAB the standard was Bussi-Donadio-Parrinello stochastic-velocity-rescaling thermostat or naive Euler-Maruyama. **BAOAB is non-negotiable** for canonical sampling.
- Quantum Monte Carlo (variational, diffusion, path-integral) is genuinely advanced and orthogonal — defer to its own future synergy review (suggest agent slot in next overnight batch). Path-integral MC composes naturally with S15+S16 once classical Langevin lands.

---

## 7. Bottom line

Statistical mechanics is the natural meeting point of `physics/` and `prob/`, and the repo today ships exactly **zero** of its core primitives despite having every needed substrate (Boltzmann constant exact, LogGamma, Erfc, three high-quality PRNGs, RK4, the Metropolis acceptance kernel inlined inside SimulatedAnnealing). One sub-package `physics/statmech/` of ~3,000 LOC across 14 files, plus 3 small additions to `prob/` (LogSumExp, MetropolisAccept, IntegratedAutocorrelationTime totalling ~100 LOC), unlocks 21 named primitives and 6 R-MUTUAL-CROSS-VALIDATION 3/3 pins. The cheapest one-day PR is ~295 LOC and ships S1+S2+S4+S6; the high-leverage second PR ships the Ising/Onsager 3/3 pin; the differentiated third PR ships Gillespie/BAOAB/Jarzynski-Crooks (no zero-dep library ships these). The shape of the ship list mirrors color×prob (176), graph×prob (162), queue×prob (173), prob×optim (169), and sequence×prob (165) — synergy reviews 158-179 collectively map a coherent prob-as-substrate architecture, of which this is the largest single gap by LOC and by physical-application surface.
