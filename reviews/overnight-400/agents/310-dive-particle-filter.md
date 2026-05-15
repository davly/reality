# 310 — dive-particle-filter (Multinomial / Stratified / Systematic / Residual / ESS-trigger)

## Headline
Reality has the Kish ESS arithmetic but no resampler — a ~150-220 LOC `prob/resample.go` (T0 generic + T1 multinomial + T3 systematic + T5 ESS-trigger) is the cheapest day-1 PR and unblocks slot 309's Bootstrap PF, Pistachio localization, and any future SMC consumer.

## Findings

### Substrate already present
- `prob/conformal/adaptive.go:160` — `EffectiveSampleSize(n int, halfLife int) float64` computes Kish n_eff = (Σwᵢ)² / Σwᵢ² with a hard-coded recency-weighting kernel `w_i = 0.5^((n-1-i)/halfLife)`. This is the exact arithmetic needed for the PF degeneracy diagnostic but the API is wrong for SMC: it takes `(n, halfLife)` not `weights []float64`. Two consumers want two signatures; correct fix is to extract a low-level `prob.KishESS(weights []float64) float64` and reimplement the conformal helper in terms of it.
- `prob/conformal/adaptive_test.go:194-221` — three tests pin: (a) no-decay limit ≡ n, (b) aggressive decay ≡ ~halfLife/ln2, (c) defensive on bad input. None of these touch the general weights-array case; new tests would not fight existing pins.
- Slot 120 (`120-prob-perf.md:81`) already flags a perf bug: the conformal version recomputes `math.Pow(0.5, k/hl)` for every score where exponential recurrence would suffice. Extracting `KishESS([]float64)` removes the recurrence question entirely and lets the conformal caller cache its own decay vector.
- Reality has zero Kalman/Bayes filter, zero `Resample`, zero generic `Sampler` interface (confirmed by grep; `signal/Resample` references in agent reviews are signal-rate resampling, unrelated).
- Slot 161 (`161-synergy-control-prob.md:242,296,329`) already speccs `SystematicResample(weights, rng, indices)` (~30 LOC) as a control×prob synergy keystone for a SIR particle filter. Slot 165 (`165-synergy-sequence-prob.md:225`) explicitly observes "ESS = 1 / Σwᵢ² triggers resampling when ESS < N/2 (already implemented in `prob/conformal/adaptive.go` — literally the same arithmetic)." Slot 156 lists `Resample` as the keystone P11 missing primitive. **Three independent agents converged on this gap.**

### The four classical schemes (and why systematic is the default)

| Scheme | Variance bound | Bias | Cost | One-line spec |
|---|---|---|---|---|
| Multinomial (Gordon-Salmond-Smith 1993) | O(1/N), highest | unbiased | O(N log N) or O(N) with stored uniforms | N independent draws from categorical(w) |
| Stratified (Kitagawa 1996) | O(1/N²) ✱ | unbiased | O(N) | One uniform `u_i ~ U[i/N, (i+1)/N]` per stratum |
| Systematic (Carpenter-Clifford-Fearnhead 1999) | O(1/N²) typically | unbiased | O(N), ~2× faster than stratified — single RNG draw | One uniform u₀ ~ U[0, 1/N], then u_i = u₀ + i/N |
| Residual (Liu-Chen 1998) | lowest variance among classical | unbiased | O(N) + multinomial on remainder | Take `floor(N·wᵢ)` deterministically; multinomial on remainder weights |

✱ Douc-Cappé-Moulines (2005, arXiv:cs/0507025) proves: residual ≼ stratified ≼ multinomial in the sense of the conditional variance ordering, **but systematic does not satisfy this ordering** — they construct an explicit counter-example. Despite this theoretical wart, Hol-Schön-Gustafsson (2006, IEEE NSSPW) finds systematic empirically dominates on quality × speed for typical SLAM/tracking workloads. This is the consensus default in Stone Soup, FilterPy, and most production particle-filter libraries.

### ESS adaptive trigger
- Kong-Liu-Wong (1994) introduced ESS in the SMC context; the Kish (1965) design-effect formula is mathematically equivalent. n_eff = 1 / Σ w̃ᵢ² for normalized weights w̃ᵢ.
- Standard threshold N/2 (Doucet-Godsill-Andrieu 2000); sparse-observation regimes use N/10. Critical: never resample at every step (loses pre-resampling info), never never (degeneracy collapse to 1 particle in ~10-50 steps).
- Reality's `EffectiveSampleSize(n, halfLife)` does *not* expose the general `KishESS([]float64)` form — it bakes in geometric decay. Refactor: introduce `prob.KishESS(weights []float64) float64`, then `prob/conformal/adaptive.go` computes its decay vector and delegates.

### Suggested API (single file, `prob/resample.go`)

```go
// ResampleScheme identifies one of the four classical SMC resampling
// schemes. Multinomial is the high-variance baseline (use for testing
// only); Systematic is the recommended default.
type ResampleScheme int
const (
    Multinomial ResampleScheme = iota
    Stratified
    Systematic
    Residual
)

// Resample fills indices[i] with the parent-particle index for the
// i-th resampled particle. weights MUST be non-negative and sum to a
// strictly positive value; they are internally normalized. len(indices)
// is the output particle count N (often == len(weights) but not required).
// rng returns U(0,1) draws; pass crypto.NewPCG(seed).Float64.
func Resample(scheme ResampleScheme, weights []float64, rng func() float64, indices []int) error

// KishESS = (Σw)² / Σw²  — the Kong-Liu-Wong / Kish effective sample size.
// For normalized weights this reduces to 1/Σw². Callers typically resample
// when KishESS(w) < N/2.
func KishESS(weights []float64) float64

// ShouldResample is the conventional adaptive trigger: returns true when
// KishESS(weights) < threshold·N. threshold=0.5 is the Doucet default;
// 0.1 is appropriate for sparse-likelihood regimes.
func ShouldResample(weights []float64, threshold float64) bool
```

This is ~150-220 LOC including doc comments and per-function provenance citations. Output-buffer-passing convention (per CLAUDE.md rule 3) is preserved by `indices []int` rather than a returned slice.

### R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities

1. **Weighted-mean preservation across all four schemes (the law of large numbers pin).** For random `weights` and random `x_i`, all four schemes must satisfy `mean(x[indices]) → Σ wᵢ xᵢ` as N→∞ with empirical error ε ~ O(1/√N). N=10,000 drives ε to ~0.01, an easy tolerance. Pin: four schemes × identical weights/x → all four within 0.02 of weighted mean. R-3/3.
2. **Variance ordering pin (Douc-Cappé-Moulines theorem).** For fixed peaked weights, repeat resampling 1000 times and compute `Var(mean(x[indices]))`. Empirically `Var_residual ≤ Var_stratified ≤ Var_multinomial` must hold. Systematic is excluded from this ordering (per the DCM counter-example) and gets its own coarser bound. Direct theoretical pin.
3. **KishESS three-way cross-validation.** (a) closed-form `KishESS(uniform_N) == N` exactly; (b) `KishESS(δ_i) == 1` exactly (degenerate); (c) Monte-Carlo simulation: when ESS < N/2, post-resampling weights are uniform → ESS_after == N. Three identities, one function. R-3/3.

### Cross-link consumers
- **Slot 309 PR-B Bootstrap Particle Filter** — direct primary consumer. Bootstrap PF = propagate via dynamics + reweight by likelihood + adaptive systematic resample. Without this PR, slot 309 cannot ship.
- **Slot 161 (control × prob synergy C11)** — explicitly designed around `SystematicResample`.
- **Slot 165 (sequence × prob synergy)** — already cites `EffectiveSampleSize` reuse.
- **Slot 220 SDE rare-event simulation** — splitting / importance-sampling resampling.
- **Slot 264 SMC for Bayesian inference** — direct.
- **Pistachio localization** (downstream consumer) — particle filter is the canonical robot-localization algorithm; 60 FPS budget tolerates O(N) systematic but not O(N log N) naive multinomial with `sort.SearchFloat64s`.

### Singular cheapest day-1 PR
**T0 (generic dispatch + types) + T1 (multinomial baseline) + T3 (systematic, the default) + T5 (KishESS + ShouldResample).** Skip stratified and residual for v1 — multinomial is the bias-free reference for tests, systematic is the production default. Add stratified/residual in a second PR once the API has settled.

LOC budget:
- `prob/resample.go`: ~180 LOC (4 funcs + types + ~60 LOC doc comments with provenance citations).
- `prob/resample_test.go`: ~250 LOC (3 R-MUTUAL pins + edge cases).
- Golden file `prob/testdata/resample_v1.json`: ~30 vectors (N ∈ {1, 2, 8, 64, 1000}; weight patterns: uniform, peaked, two-mode, degenerate-but-positive). Determinism via fixed seed `crypto.NewPCG(0xCAFE)`.
- Total: ~430 LOC + ~150 lines golden JSON. Well under one focused day.

### Edge cases that bite
- Sum of weights ≠ 1: Hol-Schön-Gustafsson and FilterPy both internally normalize; specify "weights are normalized internally" in doc comment.
- All weights zero: return error (degenerate; caller bug).
- Single weight = 1, rest = 0: must return all indices = winner; a perf bug here will burn N RNG draws unnecessarily — multinomial implementation should early-exit.
- N=1: trivially returns [0] for normalized weights.
- Non-finite weights (NaN, Inf): error (caller bug; PF likelihoods that overflow indicate model mis-specification).
- Negative weights: error.
- Subnormal weights: keep them — particle filters routinely have weights in 1e-300 range after many likelihood multiplications. Document that callers should log-domain reweight before normalizing.

### Subtle implementation notes
- **Systematic resampling cumulative-weight loop**: classical `j` pointer advances monotonically. Correct skeleton:
  ```
  c := w[0]; j := 0; u0 := rng() / N
  for i := 0; i < N; i++ {
      u := u0 + float64(i)/N
      for u > c { j++; c += w[j] }
      indices[i] = j
  }
  ```
  Subtle: floating-point accumulation of `c` can leave `c < 1` after the last step due to round-off, causing the `j++` to walk off the end. Either clamp `j = min(j, N-1)` after the inner loop or use Kahan/Neumaier summation. Most reference implementations clamp; pin the behaviour with a "near-degenerate weight + adversarial floating-point" golden vector.
- **Multinomial via Algorithm of Walker** (alias method, O(N) preprocess + O(1) per sample) vs sorted-uniforms (O(N log N) sort, O(N) walk): for the v1 PR, sorted-uniforms is simpler and sufficient — alias method is a v2 perf optimization once benchmarks demand it.
- **Residual remainder**: `N - Σ floor(N·w_i)` can equal 0 (skip multinomial step) or up to N (full multinomial). Don't allocate a remainder weights vector if remainder = 0.

### What's deliberately *out of scope* for v1
- **Branching algorithms** (Crisan-Lyons 2002): mathematically optimal but high implementation complexity; defer.
- **MoStratified, RA-resampling, GA-resampling** (Frontiers FITEE 2016, S. Seghiri 2025): all variants on stratified — defer until a consumer requests them.
- **Parallel resampling** (Bolic-Djurić-Hong 2003): production necessity for >10⁶ particles, but reality is single-threaded library code; Pistachio's 60 FPS budget at N≈1000-5000 doesn't justify it. Defer.
- **Optimal-transport resampling** (Reich 2013): different paradigm; separate primitive when consumer arrives.

### Provenance citations to embed in doc comments

```
// Multinomial:  Gordon, Salmond, Smith (1993). "Novel approach to
//               nonlinear/non-Gaussian Bayesian state estimation."
//               IEE Proc. F 140(2):107-113.
// Stratified:   Kitagawa, G. (1996). "Monte Carlo filter and smoother
//               for non-Gaussian nonlinear state space models."
//               J. Comp. Graph. Stat. 5(1):1-25.
// Systematic:   Carpenter, J., Clifford, P., Fearnhead, P. (1999).
//               "An improved particle filter for non-linear problems."
//               IEE Proc. Radar Sonar Navig. 146(1):2-7.
// Residual:     Liu, J.S., Chen, R. (1998). "Sequential Monte Carlo
//               methods for dynamic systems." JASA 93(443):1032-1044.
// ESS:          Kong, A., Liu, J.S., Wong, W.H. (1994). "Sequential
//               imputations and Bayesian missing data problems."
//               JASA 89(425):278-288. (Kish 1965 design-effect form.)
// Comparison:   Douc, R., Cappé, O., Moulines, E. (2005). "Comparison
//               of resampling schemes for particle filtering."
//               arXiv:cs/0507025. — variance ordering thm + systematic
//               counter-example.
// Practitioner: Hol, J.D., Schön, T.B., Gustafsson, F. (2006). "On
//               resampling algorithms for particle filters." IEEE NSSPW.
```

## Concrete recommendations

1. **Day-1 PR (~430 LOC + golden JSON):** create `prob/resample.go` with the four-function API above (`ResampleScheme` type, `Resample`, `KishESS`, `ShouldResample`). Implement multinomial (sorted-uniforms) and systematic only; leave stratified/residual as `return errors.New("not yet implemented")` slots so callers can pin against the dispatch interface. Tests pin the three R-MUTUAL identities for the two implemented schemes.
2. **Refactor `prob/conformal/adaptive.go`** to delegate Kish arithmetic to the new `prob.KishESS([]float64)`. The conformal entry-point keeps its `(n, halfLife)` signature for backward compat but its body shrinks to: build decay vector, call `KishESS`. This also addresses slot 120's perf finding (the recomputed `math.Pow` becomes a one-time vector build the caller can cache).
3. **Day-2 PR (~120 LOC):** add stratified and residual to the `Resample` dispatch. Pin the Douc-Cappé-Moulines variance ordering as the third R-MUTUAL identity.
4. **Slot 309 follow-up PR (Bootstrap PF) consumes this directly.** The PF state-update inner loop is `propagate; reweight; if ShouldResample(w, 0.5) { Resample(Systematic, w, rng, idx); reorder; reset weights to 1/N }`. Roughly 80 LOC on top of this PR.
5. **Defer:** branching algorithms, parallel resampling, optimal-transport, MoStratified/RA/GA variants. Open issues but do not implement — let consumer pull demand drive prioritization.
6. **Document the "never resample every step" rule** prominently in the package comment. The most common particle-filter bug in practice is unconditional per-step resampling, which throws away the pre-resampling weight information and inflates Monte Carlo variance.
7. **Golden-file vectors** must include the adversarial near-degenerate weight case (one weight ≈ 1 - ε, rest ≈ ε/(N-1)) to pin the systematic-resampler floating-point clamp behaviour across language ports.

## Sources

### Repo
- `C:\limitless\foundation\reality\prob\conformal\adaptive.go:145-176` — existing `EffectiveSampleSize` (Kish form, recency-weighted; substrate to extract).
- `C:\limitless\foundation\reality\prob\conformal\adaptive_test.go:194-221` — existing ESS pins.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\120-prob-perf.md:81` — perf finding on `math.Pow` recomputation (resolved by the refactor in rec #2).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\155-synergy-crypto-prob.md:252-257,371` — MCMC ESS proposal (Geyer 1992 autocorrelation form, complementary not duplicate).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\156-synergy-topology-prob.md:38,91` — flags missing `Resample` / `Sampler` keystone.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\161-synergy-control-prob.md:242,296,301,329` — control × prob synergy citing `SystematicResample(weights, rng, indices)`.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\165-synergy-sequence-prob.md:225` — sequence × prob synergy citing ESS reuse.

### Web (primary literature)
- [Kitagawa, G. (1996). Monte Carlo Filter and Smoother for Non-Gaussian Nonlinear State Space Models. JCGS 5(1):1-25](https://people.bordeaux.inria.fr/pierre.delmoral/kitagawa-1996.pdf) — stratified resampling original.
- [Carpenter, Clifford, Fearnhead (1999). An Improved Particle Filter for Non-linear Problems. IEE Proc. Radar Sonar Navig. 146(1):2-7](https://www.researchgate.net/publication/2923975_An_Improved_Particle_Filter_for_Non-linear_Problems) — systematic resampling original.
- [Liu, J.S., Chen, R. (1998). Sequential Monte Carlo Methods for Dynamic Systems. JASA 93(443):1032-1044](https://www.researchgate.net/publication/2590205_Sequential_Monte_Carlo_Methods_for_Dynamic_Systems) — residual resampling original.
- [Douc, Cappé, Moulines (2005). Comparison of Resampling Schemes for Particle Filtering. arXiv:cs/0507025](https://arxiv.org/abs/cs/0507025) — variance ordering theorem + systematic counter-example.
- [Hol, Schön, Gustafsson (2006). On Resampling Algorithms for Particle Filters. IEEE NSSPW](https://www.researchgate.net/publication/4288587_On_Resampling_Algorithms_for_Particle_Filters) — empirical comparison; systematic recommended.
- [Kong, Liu, Wong (1994). Sequential Imputations and Bayesian Missing Data Problems. JASA 89(425):278-288](http://www.lucamartino.altervista.org/SigPro2016_ESS.pdf) — ESS = 1/Σw² original (in SMC context).
- [Doucet, Godsill, Andrieu (2000). On Sequential Monte Carlo sampling methods. Stat. Comp. 10:197-208](https://www.stats.ox.ac.uk/~doucet/doucet_godsill_andrieu_sequentialmontecarloforbayesfiltering.pdf) — N/2 adaptive threshold convention.
- [Bolic, Djurić, Hong (2003+). Resampling architectures for parallel particle filters](https://link.springer.com/article/10.1155/S1110865704405149) — deferred (parallel implementations).
- [Stone Soup PF tutorial — ESSResampler N/2 default](https://stonesoup.readthedocs.io/en/latest/auto_tutorials/04_ParticleFilter.html) — practitioner reference for adaptive trigger.
- [Particle Filters: A Hands-On Tutorial. PMC PMC7826670](https://pmc.ncbi.nlm.nih.gov/articles/PMC7826670/) — survey-level reference.
- [arXiv:1809.04129 — Rethinking the Effective Sample Size](https://arxiv.org/pdf/1809.04129) — caveats on ESS as a degeneracy diagnostic; out-of-scope for v1 but worth a doc-comment pointer.
