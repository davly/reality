# 335 — dive-cmaes-tuning (sep-CMA / IPOP / BIPOP / lm-CMA / aCMA audit)

## Headline
Reality ships **no CMA-ES at all** (only `GeneticAlgorithm` + `SimulatedAnnealing` for derivative-free global search); slot 102 already flagged the standard `(μ/μ_w,λ)`-CMA-ES as T2.18 missing — this dive scopes the *full Hansen-family lineage* (active, sep, IPOP, BIPOP, lm-CMA, BBOB harness) and recommends a single Day-1 PR (~700 LOC) shipping aCMA-ES + IPOP restart together because the marginal cost of restart logic on top of a base CMA-ES loop is ~120 LOC.

## Findings (existing audit)

- **No `cmaes.go`, `cma.go`, `evolution_strategy.go`, or any covariance-adaptation code anywhere in the repo.** `Grep` for `CMA|cmaes|sep_CMA|IPOP|BIPOP|lmCMA|ActiveCMA|covariance.adaptation` over `optim/` returned only false positives in unrelated test names. `optim/cmaes.go` (referenced as expected by the topic prompt) does not exist.
- `optim/` directory contents (file:line):
  - `optim/genetic.go:1-178` — real-coded GA with BLX-α crossover, Gaussian mutation, tournament size 3, hardcoded `[-5,5]` domain (slot 101 F13). 178 LOC.
  - `optim/metaheuristic.go:1-93` — geometric-cooling SA, Metropolis acceptance. 93 LOC.
  - `optim/gradient.go`, `optim/linear.go`, `optim/rootfind.go`, `optim/proximal/`, `optim/transport/` — all gradient-based or LP/proximal; none are evolution strategies.
- Slot 102's missing-algorithm catalogue (`reviews/overnight-400/agents/102-optim-missing.md:104`) lists CMA-ES as **T2.18, ~400 LOC** alongside DE (T1.10) and PSO (T1.11) as the canonical "global-search quartet" gap. Hansen-family variants (sep / IPOP / BIPOP / lm) are not enumerated there — slot 335 is their first proper review.
- Slot 102's Sprint-3 plan (line 230) bundles DE + PSO + CMA-ES into a single ~800 LOC week. This dive **disagrees on ordering**: CMA-ES alone justifies a dedicated PR because its variant tree is wider than DE/PSO and its consumer pull (hyperparameter optimisation, neural architecture search, calibration of stiff models) is higher.
- No `BBOB` benchmark suite; no `Rastrigin`, `Rosenbrock`, or `Ackley` golden test vectors anywhere in `optim/optim_test.go`. Existing GA tests (`optim_test.go:81-100`) use a 1-D function map — no canonical multimodal regression battery exists.
- No `linalg/eigen` symmetric-eigendecomposition is wired into `optim/` for covariance updates; `linalg/cholesky` exists and is the natural primitive for the C^(1/2) factor used in standard CMA-ES sampling.

## Variant taxonomy (research-frontier crosscheck)

| Variant | Year | Key idea | Memory | LOC delta vs base CMA |
|---------|------|----------|--------|----------------------:|
| `(μ/μ_w,λ)`-CMA-ES (Hansen-Ostermeier) | 2001 | Weighted recombination + rank-1 + rank-μ + path-cumulation σ-control | O(n²) | base (~400) |
| **active-CMA / aCMA** (Igel-Suttorp-Hansen) | 2007 | Negative weights for the worst λ−μ samples in rank-μ update | O(n²) | +80 |
| **sep-CMA-ES** (Ros-Hansen) | 2008 | Diagonal-only C, learning rate scaled by n/3 | **O(n)** | +80 |
| **IPOP-CMA-ES** (Auger-Hansen) | 2005 | Restart-with-doubled-λ on stagnation | O(n²) | +120 (wrapper) |
| **BIPOP-CMA-ES** (Hansen 2009) | 2009 | Two interleaved restart regimes (small-λ + large-λ) with budget split | O(n²) | +150 (wrapper) |
| **lm-CMA** (Loshchilov) | 2014 | Limited-memory rank-m update, no full C; viable for n ≥ 1000 | O(mn), m≈10·log n | ~250 |
| **DR3 / DR2** (Hansen) | 2009 | Three-regime restart variant of BIPOP | O(n²) | +50 over BIPOP |
| **MA-ES / fast-MA-ES** (Beyer-Sendhoff) | 2017 | Matrix-adaptation; replaces C-tracking with M-tracking, avoids eigendecomp | O(n²) | rewrite |
| **VkD-CMA** (Akimoto-Hansen) | 2016 | Rank-k diagonal-plus-low-rank C; bridges sep ↔ full | O(kn) | ~200 |

## Concrete recommendations

### T0 — there is no T0 (nothing to harden; CMA-ES does not exist)

### T1 — Standard `(μ/μ_w,λ)`-CMA-ES with active update (aCMA built-in) — ~480 LOC
The Day-1 PR. Hansen-Ostermeier 2001 + Igel-Suttorp-Hansen 2007. Implement once with the negative-weight switch as a flag (`Active bool`), defaulting on (modern Hansen tutorial recommends active-CMA as the default since 2016). Sampling via Cholesky of C; eigendecomposition only every n/(c1+cμ)/10 iterations per Hansen's tutorial Alg. 7. Path-cumulation step-size control ($p_\sigma$, $p_c$). Reuses `linalg/cholesky`. Defaults: `λ = 4 + ⌊3·ln(n)⌋`, `μ = λ/2`, log-decay weights `w_i ∝ ln(μ+1) - ln(i)`, c1 = 2/((n+1.3)²+μ_eff), cμ = min(1-c1, α_μ·(μ_eff-2+1/μ_eff)/((n+2)²+α_μ·μ_eff/2)).

### T2 — sep-CMA-ES — ~80 LOC delta
Ros-Hansen 2008 PPSN. Replace full C with diagonal D²; learning rates multiplied by `(n+2)/3`. Trivial flag on the T1 base. Pays off catastrophically on large separable problems (n ≥ 100 separable Rastrigin/Rosenbrock-rotated benchmarks): O(n) memory and O(n) per-sample cost. Cite: `evolution-es-2008.pdf`.

### T3 — IPOP-CMA-ES restart wrapper — ~120 LOC
Auger-Hansen 2005 CEC. Wrap T1 in a restart loop: when stop-criterion fires (TolFun, TolX, EqualFunVals, ConditionCov, NoEffectAxis, NoEffectCoord per Hansen 2016 Tutorial Alg. 9), restart with `λ ← 2·λ`, fresh σ, fresh C=I, fresh m=uniform-random in box. Run until total budget exhausted; return best-ever.

### T4 — BIPOP-CMA-ES restart wrapper — ~150 LOC
Hansen 2009 GECCO. Two interleaved regimes:
- **Large**: doubled-λ IPOP run.
- **Small**: smaller-λ run with random σ ∈ [σ₀·10⁻²ᵘ, σ₀] where u~U[0,1].
Switch regimes whenever the *total function evaluations spent in that regime* falls behind the other. Track best-of-all-restarts. Empirically the BBOB winner among CMA-ES variants on multimodal-with-weak-structure functions (BBOB f15-f24).

### T5 — lm-CMA — ~250 LOC (defer to a dedicated PR)
Loshchilov 2014 GECCO ("A Computationally Efficient Limited Memory CMA-ES"). Replace explicit C with m direction vectors (`m = 4 + ⌊3·ln(n)⌋`). Sample via $z + \sum_i a_i p_c^{(i)}$. Targets n ≥ 1000 where O(n²) C-storage and O(n³) eigendecomp dominate. Pairs with autodiff slot consumers that would never use this (n ≤ ~50) but matters for any future neural-net hyperparameter optimisation consumer or material-property fitting with thousands of parameters.

### T6 — BBOB benchmark integration (regression harness) — ~150 LOC
Hansen-Finck-Ros-Auger 2009 (COCO platform). Add 24 BBOB functions to `optim/optim_test.go` (Sphere, Ellipsoid, Rastrigin, Rosenbrock-rotated, Schaffer F7, Gallagher-Gaussian, etc.) as benchmark sweep with golden-file expected-success-rate tables. Lets future PRs measure regressions quantitatively. **Required for cross-validation pins below.**

### Day-1 PR (recommendation): T1 + T3 = ~600 LOC
aCMA-ES + IPOP wrapper. The smallest unit that delivers a *modern* derivative-free global optimiser to reality; standard CMA without restarts is no longer competitive on multimodal benchmarks (Hansen's own 2016 tutorial recommends always wrapping IPOP). 9 BBOB functions as Tier-1 regression vectors.

## R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities

Following the project's golden-file precedent, propose three regression pins (each 3/3 — Go reference, Python validator, expected-rank table):

1. **CMA-ES converges on Rosenbrock(n=10) within 1e-6** in ≤ 5000 evals from `x₀ = -2·1, σ₀ = 0.5`. Pin against Hansen's published `purecma.py` reference (the 100-line Python tutorial implementation widely cited as the spec). Saturates pin slot **R-CMAES-ROSENBROCK-CONVERGE 3/3**.
2. **IPOP-CMA-ES finds global optimum of Rastrigin(n=20)** (multimodal, ~10²⁰ local optima) within target 1e-2 with success-rate ≥ 80% over 21 seeds. Compare success-rate to BBOB-2009 published table (Auger-Hansen). Saturates **R-IPOP-RASTRIGIN-MULTIMODAL 3/3**.
3. **sep-CMA-ES ≡ standard CMA-ES on the Sphere function** (separable, identity-covariance optimum). Trajectory rms-difference between sep and standard variants ≤ 1e-9 across first 100 iterations from identical seed. Saturates **R-SEPCMA-EQUIVALENT-ON-SEPARABLE 3/3**.

Bonus: **CMA-ES vs aCMA-ES on Cigar(n=30)**: active-CMA improves convergence by ≥ 1.5× per Igel-Suttorp-Hansen 2007 Table 1.

## Edge cases / hardening checklist (for the Day-1 PR)

- **Covariance singularity**: detect `cond(C) > 10¹⁴`; reset C ← I, σ ← σ₀ (one of Hansen's standard stop conditions, but the wrapper can soft-restart).
- **Stagnation in σ**: NoEffectAxis / NoEffectCoord (Hansen 2016 §4.1).
- **TolFun / TolX**: standard convergence criteria; expose as `Stop` config struct.
- **Reproducibility**: accept `rng *rand.Rand` (matches existing GA / SA convention in `optim/`).
- **Box constraints**: implement reflection-into-box (Hansen 2009) as opt-in; CMA-ES is unconstrained by default.
- **Allocation discipline** (per CLAUDE.md design rule 3): pre-allocate λ × n sample matrix, μ × n parent matrix, C, B, D once; reuse across generations.
- **Cite source per design rule 4**: each function references Hansen 2016 tutorial section.

## Cross-link consumers / synergy slots

- **Slot 102 (optim-missing)** — this dive **fulfills** T2.18 of slot 102; close the loop by referencing back when shipping.
- **Slot 101 (optim-numerics)** F13 (GA hardcoded `[-5,5]`) — CMA-ES has no such issue (sample box is implicit via σ); fixes the "non-portable global optimiser" gap.
- **Slot 099 (linalg-api)** — CMA-ES needs `linalg/cholesky` (already exists) and a symmetric-eigendecomposition (verify presence; if missing, blocker for the periodic eigendecomp of C).
- **Slot 103 (optim-sota)** — CMA-ES is *the* SOTA derivative-free for n ≤ 100; this dive feeds the SOTA gap analysis.
- **Slot 105 (optim-perf)** — sep-CMA-ES (T2 here) is the main perf lever for n ≥ 100; cite this dive.
- **Slot 195 (synergy-optim-prob)** — IPOP-CMA-ES + Bayesian optimisation hybrid (TPE warm-starting BIPOP) is a published technique (Optuna 4.0 ships it).
- **Slot 169 (synergy-prob-optim)** — same hybrid theme, opposite direction.
- **Slot 164 (synergy-orbital-optim)** — orbital-element fitting (Hohmann transfer, low-thrust trajectory) is a textbook CMA-ES consumer (Izzo 2010 GTOC).
- **Slot 279 (new-metaheuristics)** — already audited 21 metaheuristics; CMA-ES would be the marquee addition. Cross-reference.
- **Slot 220 (new-stochastic-opt)** — overlaps with adaptive-LR optimisers; CMA-ES competes with Adam in the n ≤ 100 regime.
- **Slot 195 / aicore** — aicore consumer flagged "hyperparameter optimisation" consumer in earlier reviews; CMA-ES is the primary candidate.

## Sources

### Repository
- `C:\limitless\foundation\reality\optim\genetic.go` (1-178; existing GA, F13 hardcoded domain)
- `C:\limitless\foundation\reality\optim\metaheuristic.go` (1-93; existing SA)
- `C:\limitless\foundation\reality\optim\optim_test.go` (1-100; current 1-D test harness)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\102-optim-missing.md` (T2.18 entry, line 104)
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md` (line 113 — slot 102 missing list; line 355 — this slot)
- `C:\limitless\foundation\reality\CLAUDE.md` (design rules 1-6: golden files, zero deps, no allocations, cite source, document precision, reimplement from first principles)
- `C:\limitless\foundation\reality\optim\proximal\` (precedent for sub-package; `optim/cmaes/` would follow same pattern)

### Web sources (citations for implementation, not WebFetched in this run; standard references in the literature)
- Hansen & Ostermeier (2001) "Completely Derandomized Self-Adaptation in Evolution Strategies", *Evolutionary Computation* 9(2):159-195. Founding paper.
- Hansen (2016) "The CMA Evolution Strategy: A Tutorial", arXiv:1604.00772. **The implementation reference**, used by all major libraries.
- Ros & Hansen (2008) "A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity", PPSN X, LNCS 5199:296-305. (sep-CMA-ES.)
- Auger & Hansen (2005) "A Restart CMA Evolution Strategy with Increasing Population Size", *IEEE CEC* 2005:1769-1776. (IPOP-CMA-ES.)
- Hansen (2009) "Benchmarking a BI-Population CMA-ES on the BBOB-2009 Function Testbed", *GECCO* 2009 Workshop:2389-2396. (BIPOP-CMA-ES.)
- Igel, Suttorp & Hansen (2007) "A Computational Efficient Covariance Matrix Update and a (1+1)-CMA for Evolution Strategies", *GECCO* 2007:453-460. (active-CMA / aCMA.)
- Loshchilov (2014) "A Computationally Efficient Limited Memory CMA-ES for Large Scale Optimization", *GECCO* 2014:397-404. (lm-CMA.)
- Akimoto & Hansen (2016) "Online Model Selection for Restricted Covariance Matrix Adaptation", PPSN XIV. (VkD-CMA, bridges sep↔full.)
- Beyer & Sendhoff (2017) "Simplify Your Covariance Matrix Adaptation Evolution Strategy", *IEEE TEC* 21(5):746-759. (MA-ES, eigendecomposition-free.)
- Hansen, Finck, Ros & Auger (2009) "Real-Parameter Black-Box Optimization Benchmarking 2009: Noiseless Functions Definitions", INRIA RR-6829. (BBOB testbed for T6.)
- Hansen `purecma.py` reference implementation (https://github.com/CMA-ES/pycma) — the canonical 100-line spec used as cross-language regression target.

### Library cross-checks (consumer pull evidence)
- pycma (Python) — Hansen's own reference; ships standard / aCMA / sep / IPOP / BIPOP / lm-CMA.
- Optuna 4.0 `CmaEsSampler` — ships standard + sep + IPOP + warm-starting CMA-ES (Nomura-Watanabe-Akimoto-Ozaki-Onishi 2020).
- Nevergrad — ships ~6 CMA-ES variants in its meta-portfolio.
- BoTorch / Ax — uses CMA-ES as the gradient-free sub-optimiser of acquisition functions in mixed search spaces.
- C-MAES (Hansen's C reference) — used by NLopt as `NLOPT_GN_CMAES`.

## Day-1 PR shopping list (concrete)

```
optim/cmaes/
  cmaes.go         # T1: standard + active CMA-ES, ~480 LOC
  cmaes_test.go    # 9 BBOB function regression vectors, golden tables
  ipop.go          # T3: IPOP restart wrapper, ~120 LOC
  ipop_test.go     # Rastrigin n=20 multi-seed success-rate, golden table
  doc.go           # cite Hansen 2016 tutorial
  testdata/
    bbob_*.json    # golden files per CLAUDE.md design rule 1
```

Total: ~700 LOC code + ~400 LOC tests + 9 golden-file vectors. Two R-MUTUAL-CROSS-VALIDATION 3/3 pins saturated (Rosenbrock + Rastrigin). Closes slot 102 T2.18. Unblocks future T2 (sep-CMA), T4 (BIPOP), T5 (lm-CMA) PRs.
