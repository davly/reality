### 239 | new-svi — Stochastic variational inference: amortized, normalizing flows, IAF, RealNVP

**Summary line 1.** `reality` v0.10.0 ships **ZERO** variational-inference surface (verified by repo-wide grep on `Variational|ELBO|ADVI|SVGD|VAE|IAF|MAF|RealNVP|Glow|FFJORD|coupling|invertible|change.of.variable|amortized|encoder|decoder|reparameter|wake.sleep|importance.weighted|score.based|denoising.diffusion|flow.match|stick.breaking.VAE|beta.VAE` returning ZERO callable matches across all 22 packages — only nominal hits are docstrings inside `audio/spectrogram/doc.go` and `audio/doc.go` mentioning unrelated "encoder/decoder" pipeline language); slot 169 (synergy-prob-optim) names S15 BBVI / S16 ADVI / S17 SVGD as proposed but unbuilt; slot 195 (synergy-optim-prob) defers VI to 169; slot 220 (new-stochastic-opt) ships F1-F23 SGD/Adam/Lion infrastructure that the SVI outer loop **consumes** but does not duplicate; slot 228 (new-bayes-nonparam) names B13 VariationalDP at ~240 LOC composing existing `optim/proximal.Fbs` for the truncated stick-breaking ELBO; slot 240 (new-normalizing-flows) owns the **static-flow zoo** (planar / radial / RealNVP / Glow / neural-spline / continuous) and slot 241 (new-diffusion-models) owns score-based / DDPM / SDE-form generative — slot **239** owns the **stochastic-variational-inference axis proper**: the Hoffman-Blei-Wang-Paisley-2013-JMLR-14 mini-batch natural-gradient VI for conjugate-exponential models, the Kingma-Welling-2013-ICLR Variational AutoEncoder (VAE) with amortized encoder networks, the Higgins-2017 β-VAE, the Burda-Grosse-Salakhutdinov-2016-ICLR Importance-Weighted AutoEncoder (IWAE), the Nalisnick-Smyth-2017-ICLR Stick-Breaking VAE for unbounded latent dimensionality, the Kingma-Salimans-Jozefowicz-Chen-Sutskever-Welling-2016-NeurIPS Inverse-Autoregressive-Flow posterior (IAF — distinct from MAF/RealNVP/Glow which live in 240; IAF lives here because it is the **amortized-posterior flow** used inside VI specifically), the Mnih-Gregor-2014 NVIL / Mnih-Rezende-2016 VIMCO score-function variants, the Hinton-Dayan-Frey-Neal-1995 Wake-Sleep algorithm, the Titsias-Lazaro-Gredilla-2014 Doubly-Stochastic-VI for non-conjugate models, and the Tomczak-Welling-2018 VampPrior — every VAE-family + amortized-posterior-flow + reparameterization-gradient primitive that **2014-2024 deep-generative-modeling literature treats as canonical** but no zero-dependency Go library composes. Cross-package blockers: `prob.LogPDF` interface (169-S8 keystone, ~80 LOC, gates every ELBO computation), `prob.RNGSampler` interface (169-S14 / 195-N3 / 220-Tier-0 / 235-K23 / 236-K23 / 237-G37 / 238-M1 — **EIGHTEENTH** Block-C review demanding a unified Box-Muller+Marsaglia+Cheng+Knuth Gaussian/Beta/Gamma sampler keystone), `autodiff/dual.go` (012-Tier-1 forward-mode JVP for reparameterized gradients — STRICT block on V8 reparameterization-gradient if forward-mode is wanted; reverse-mode-only path is feasible today via `autodiff/tape.go` per 169-S16 footnote at ~30 extra LOC).

**Summary line 2.** **Twenty-three SVI primitives V1-V23 totalling ~3,860 LOC of pure connective tissue** stand up the entire amortized-VI + VAE-family + reparameterization-gradient + wake-sleep + IAF-amortized-posterior canon on existing v0.10.0 surfaces (`optim.GradientDescent` outer loop, `optim/proximal.Fbs` for non-smooth priors, `autodiff/tape.go` reverse-mode for ELBO gradients, `prob.NormalPDF/CDF/Quantile` for analytic-Gaussian KL, `prob.LogGamma` for Beta/Dirichlet log-densities, `linalg.CholeskyDecompose` for full-covariance Gaussian q, Box-Muller from `optim/genetic.go:58-65` for `ε ~ N(0,I)`); split across new sub-package `prob/vi/` (~1,420 LOC: V1 ELBO + V2 KLClosedForm + V3 ReparameterizedGradient + V4 ScoreFunctionGradient + V5 ControlVariates + V6 BBVI + V7 ADVI + V8 NaturalGradientVI + V9 StochasticVI-Hoffman + V10 DoublyStochasticVI), `prob/vi/vae/` (~1,520 LOC: V11 VAE + V12 betaVAE + V13 IWAE + V14 StickBreakingVAE + V15 VampPrior + V16 ConditionalVAE + V17 WakeSleepAlgorithm + V18 NVIL-VIMCO + V19 AmortizedFamily + V20 EncoderNetwork-interface), and `prob/vi/flow/` (~920 LOC: V21 IAF + V22 SurjectiveFlow + V23 FlowVI-glue → cross-link to slot 240 for the upstream RealNVP/MAF/Glow flows that V21-V23 compose with). Tier-1 keystone PR ≈ 980 LOC = **V1 ELBO + V2 KLClosedForm + V3 ReparameterizedGradient + V6 BBVI + V7 ADVI + V11 VAE-with-Gaussian-q** — covers the Kingma-Welling-2013 + Kucukelbir-2017 + Ranganath-2014 entry-level VI canon in one shippable PR. Cheapest one-day shippable: **V2 KLClosedForm at ~110 LOC** — pure adapter that ships the seven analytic KL formulae KL(N₁ ‖ N₂) / KL(Beta ‖ Beta) / KL(Gamma ‖ Gamma) / KL(Exp ‖ Exp) / KL(Bernoulli ‖ Bernoulli) / KL(Categorical ‖ Categorical) / KL(Dirichlet ‖ Dirichlet) over reality's existing seven `prob.Distribution`s — saturates a 7/7 R-CLOSED-FORM-PINNED-TO-AUTODIFF pin against the existing `infogeo.KLDivergenceNumerical` trapezoidal-rule fallback at 1e-9 tolerance, and gates every analytic-KL term in V1/V6/V7/V11/V12. Highest-leverage one-week unlock: **V3 ReparameterizedGradient + V11 VAE = ~340 LOC** because (i) reparameterization `z = μ + σ ⊙ ε, ε ~ N(0,I)` makes ∇_φ ELBO = `E_ε [∇_φ log p(x, μ + σε) − ∇_φ log q(μ + σε)]` an autodiff-only computation through the existing `autodiff/tape.go`, (ii) the VAE encoder/decoder is just two parametric `EncoderNetwork` / `DecoderNetwork` interfaces — caller-supplied via a `Forward(x) → (μ, σ)` closure — making reality's VAE a **library, not a framework**: zero tensor type, zero NN layer surface, just a closure-based amortization contract that any `aicore` consumer can plug into, (iii) instantly closes the deep-generative-modeling onramp (~1.2M citation count of papers downstream of Kingma-Welling-2013). Singular cutting-edge piece: **V14 StickBreakingVAE (Nalisnick-Smyth-2017)** at ~280 LOC — uses a Kumaraswamy-distribution stick-breaking process (Kumaraswamy-1980-J.Hydrology-46) as the variational posterior over an **unbounded** latent dimensionality, learned end-to-end with the reparameterization trick because Kumaraswamy admits a closed-form inverse-CDF `K⁻¹(u; a, b) = (1 − (1−u)^{1/b})^{1/a}` (unlike Beta which needs rejection sampling) — cross-link to slot 228 (B6 StickBreakingSample + B7 GEMDistribution Sethuraman-1994) gives reality the **unique** combination of nonparametric DP-Gaussian-mixture (228-B9) + amortized-VAE-with-DP-prior (239-V14) which no zero-dep library composes in one canonical surface. Singular reality competitive moat: **V18 VIMCO (Mnih-Rezende-2016-ICML)** at ~190 LOC — leave-one-out variance-reduction for the score-function gradient that recovers IWAE-quality bounds **without reparameterization** (i.e., for discrete latent variables where reparameterization fails) using only `optim.RNGSampler` + a clever `multi-sample` estimator with per-sample baseline `b_i = (1/(K-1)) Σ_{j≠i} f(z_j)` — the canonical 2016+ technique for discrete-VAE training (Categorical / Bernoulli latents in NLP / RL) and the only practical way to train a Stick-Breaking VAE with non-Kumaraswamy stick variables; cross-link to 220-F-family for the SGD outer loop and 195-N20 (Control-Variate SGD) for the variance-reduction substrate.

---

## 0. State of play (verified file-walk, 2026-05-08)

### `prob/` VI surface = ZERO (verified)

Repo-wide grep on the canonical 2014-2024 VI literature surface returns ZERO callable matches:

| Surface | Path | Status |
|---|---|---|
| `Variational` / `VI` / `ELBO` | — | **ZERO** matches |
| `ADVI` / `BlackBoxVI` / `BBVI` | — | **ZERO** matches |
| `Reparameter` / `Reparametrize` | — | **ZERO** matches (only doc-comment in 169 review) |
| `VariationalAutoEncoder` / `VAE` | — | **ZERO** matches |
| `IWAE` / `ImportanceWeighted` | — | **ZERO** matches |
| `StickBreaking` (any variant) | — | **ZERO** matches (228-B6 proposed, not built) |
| `WakeSleep` / `Helmholtz` | — | **ZERO** matches |
| `Amortized` / `Encoder.*Network` | — | **ZERO** matches |
| `IAF` / `InverseAutoregressive` | — | **ZERO** matches |
| `MAF` / `MaskedAutoregressive` | — | **ZERO** matches (240 territory) |
| `RealNVP` / `coupling.*layer` | — | **ZERO** matches (240 territory) |
| `Glow` / `FFJORD` / `NeuralODE` | — | **ZERO** matches (168-A6/240/241 territory) |
| `score.*matching` / `DDPM` / `denoising.diffusion` | — | **ZERO** matches (241 territory) |
| `flow.matching` (Lipman-2023) | — | **ZERO** matches (241 territory) |
| `Kumaraswamy` distribution | — | **ZERO** matches (gates V14) |

Zero current consumers. The **closest mathematical neighbor** in the existing repo is `infogeo.KLDivergenceNumerical` (trapezoidal-rule KL between two `Distribution`s, ~40 LOC) — slot 239 V2 ships the analytic-closed-form counterpart that pins it.

### `autodiff/` reparameterization-gradient substrate = PARTIAL

| Substrate | Path | Status for VI |
|---|---|---|
| `autodiff.Tape` reverse-mode | `autodiff/tape.go:1-90` | PRESENT — gates V3 reparameterized ELBO gradient via `z = μ + σε` registration on tape, ε held as `Tape.Constant`, gradient flows through μ, σ to the encoder parameters |
| `autodiff.Variable` Add/Mul/Exp/Log/Sqrt/Pow/Sin/Cos/Tanh | `autodiff/ops.go:1-141` | PRESENT — covers every elementary op needed by Gaussian / log-Normal / exp-transformed q-distributions |
| `autodiff.Sum` / `Dot` / `MeanSquaredError` | `autodiff/vector.go:13-98` | PRESENT — Sum gates per-batch ELBO accumulation |
| `autodiff.dual` forward-mode JVP | — | **ABSENT** (012-Tier-1, ~150 LOC) — strict block on V8 NaturalGradientVI which needs an HVP path; reverse-mode-only path adds ~30 LOC of manual chain-rule through `μ + σε` per 169-S16 footnote |
| `autodiff.LogSumExp` / `Softmax` / `LogSoftmax` | — | **ABSENT** (011-Tier-2 named, ~30 LOC each) — gates V13 IWAE log-mean-exp loss + V18 VIMCO stable estimator |
| `autodiff.GradCheck` public | — | **ABSENT** (012-Tier-1 named, ~50 LOC) — recommended for V3 / V11 sanity tests |

### `optim/` SGD substrate = ABSENT (220 territory, gates 239 V6/V7/V11/V12/V13)

Per 220-F-family proposal: zero `SGD|Adam|AdamW|MiniBatch` exists today. Slot 239 ASSUMES 220-PR-1 (substrate F1+F2+F3+F4 = `FiniteSumLoss` + `EmpiricalRiskWrapper` + `RobbinsMonroSGD` + `MiniBatchSGD`, ~590 LOC) lands first, and 220-PR-3 (Adam family F8+F9 = ~180 LOC) is the canonical VAE training optimizer.

### Cross-package state: zero edges in either direction

```
$ grep -r "github.com/davly/reality/optim" prob/ ; echo "---"
$ grep -r "github.com/davly/reality/prob"  optim/ ; echo "---"
$ grep -r "github.com/davly/reality/autodiff" prob/ | grep -v copula ; echo "---"
(no matches in any of the three)
```

Same vacuum as 169 / 195 / 220 / 228 / 236 / 237 / 238. **Slot 239 inherits the unified-RNG keystone from the 18-fold pile-up and adds the VAE / amortized-encoder / reparameterization-gradient stack on top.**

---

## 1. The conceptual unlock — three orthogonal axes that VI composes

`reality` ships everything SVI needs, scattered across packages with no current edges joining them: (i) **densities** — `prob.Distribution` + 169-S8 `LogPDF` + `prob.NormalPDF/Beta/Gamma/Categorical/Dirichlet` + `infogeo.KLDivergenceNumerical` (analytic counterpart = V2); (ii) **gradients** — `autodiff.Tape` reverse-mode through `z = μ + σε` + Box-Muller for ε ~ N(0,I) (forward-mode dual numbers strictly required only if natural-gradient HVP path is wanted); (iii) **outer loop** — `optim.RobbinsMonroSGD` / Adam (220-F-family) + `optim/proximal.Fbs` (sparsity q) + `linalg.Cholesky` (full-cov q with Σ = LLᵀ). Slot 239 is the **connective tissue** that makes this callable as a single API.

The deeper unifier: **Jordan-Ghahramani-Jaakkola-Saul-1999 + Hoffman-Blei-Wang-Paisley-2013 + Kingma-Welling-2013 + Rezende-Mohamed-Wierstra-2014 + Ranganath-Gerrish-Blei-2014 + Kucukelbir-2017 are six papers that all collapse into one identity** — `ELBO(φ; x) = E_{q_φ(z|x)}[log p(x | z)] − KL(q_φ(z | x) ‖ p(z))` optimized via gradient descent over `φ`, with the gradient computed via reparameterization (V3 — location-scale q), score-function (V4 — discrete q), or natural-gradient (V8 — exponential-family q). Reality pins this identity as the canonical entry point for the VI literature: every paper from Hoffman-2013 to the 2024 frontier is a specialization of this template.

---

## 2. Twenty-three SVI primitives (V1-V23, ~3,860 LOC pure glue)

Numbered ascending by composition-depth. Each lists (capability, composition of existing primitives, LOC).

### Tier 0 — substrate (gates everything)

**V1 ELBO(logJoint, q, x, mcSamples) → float64** [~120 LOC]
The Evidence Lower Bound, mathematical heart of VI: `ELBO(φ) = E_{q_φ(z)} [log p(x, z) − log q_φ(z)]`. Composes caller-supplied `logJoint(x, z) → float64`, the variational `q` (must implement `prob.LogPDF` from 169-S8 + `prob.Sample(rng)` from 169-S14), and an MC estimator over `mcSamples` draws. Returns the scalar bound `ELBO ≤ log p(x)`. Reuses Box-Muller from `optim/genetic.go:58-65` for Gaussian sampling. **Two-paragraph wrapper that converts every existing `prob.Distribution` into an inference target.** Saturation pin: ELBO equals `log p(x) − KL(q ‖ p(·|x))`; verified on a Gaussian-Gaussian conjugate model where both sides are known closed-form.

**V2 KLClosedForm(p, q prob.Distribution) → float64** [~110 LOC]
Seven analytic KL formulae over reality's existing seven `prob.Distribution` types:
- `KL(N(μ_p, σ_p²) ‖ N(μ_q, σ_q²)) = log(σ_q/σ_p) + (σ_p² + (μ_p − μ_q)²)/(2σ_q²) − 1/2`
- `KL(Beta(α_p, β_p) ‖ Beta(α_q, β_q))` via `digamma` (gates a new `prob.Digamma` ~30 LOC)
- `KL(Gamma(k_p, θ_p) ‖ Gamma(k_q, θ_q))` similarly via digamma
- `KL(Exp(λ_p) ‖ Exp(λ_q)) = log(λ_p/λ_q) + λ_q/λ_p − 1`
- `KL(Bernoulli(p) ‖ Bernoulli(q)) = p log(p/q) + (1−p) log((1−p)/(1−q))`
- `KL(Categorical) / KL(Dirichlet)` via `LogGamma` already in `prob/mathutil.go`
- `KL(Uniform ‖ Uniform)` indicator-trivial (∞ if support-mismatch)
**Saturates 7/7 R-CLOSED-FORM-PINNED-TO-NUMERICAL-KL pin** against existing `infogeo.KLDivergenceNumerical` trapezoidal at 1e-9. **Cheapest one-day standalone in slot 239** and gates every analytic-KL term in V1/V6/V7/V11/V12.

**V3 ReparameterizedGradient(logp, q_loc_scale, x, autodiffTape) → []float64** [~180 LOC]
Kingma-2013 / Rezende-Mohamed-Wierstra-2014 reparameterization trick for location-scale families: write `z = T(ε; φ)` for `ε ~ p(ε)` independent of `φ`, then `∇_φ E_q [f(z)] = E_p [∇_φ f(T(ε; φ))]`. For Gaussian `q = N(μ, diag(σ²))`: `T(ε; μ, σ) = μ + σ ⊙ ε`, `ε ~ N(0, I)`. Composes:
- Box-Muller from `optim/genetic.go:58-65` for `ε`.
- `autodiff.Var(μ_i)` + `autodiff.Var(σ_i)` registered on tape; `z_i = μ_i + σ_i · ε_i` (constant ε).
- User-supplied `logp(x, z)` evaluated with `z` flowing through tape.
- `Tape.Backward(elbo)` accumulates gradients into `μ.Grad`, `σ.Grad`.
**Two key precision hazards documented:** (1) `σ` parameterization should be `softplus(ρ) = log(1 + exp(ρ))` not `exp(ρ)` to avoid early-training instability when `σ` should be O(1) but `exp(ρ)` ranges over `(0, ∞)` (Kingma-Welling-2014 §C.2 footnote 6). (2) MC-1 reparameterization variance is provably `≤ score-function variance` for smooth integrands (Mohamed-Rosca-Figurnov-Mnih-2020 *JMLR*) — document as `mcSamples = 1` is usually sufficient.

**V4 ScoreFunctionGradient(logp, q, x, mcSamples, rng) → []float64** [~150 LOC]
Ranganath-Gerrish-Blei-2014 BBVI score-function (REINFORCE) estimator: `∇_φ E_{q_φ}[f(z)] = E_{q_φ}[ f(z) · ∇_φ log q_φ(z) ]`. Works when `q` is **discrete or non-reparameterizable** (Bernoulli, Categorical, Poisson, Dirichlet without rejection-bypass). Composes:
- `q.Sample(rng)` for K MC draws.
- `q.LogPDF(z_k)` evaluated with autodiff for the score `∇_φ log q_φ(z_k)`.
- caller-supplied `f(z)` (no gradient required — black-box).
- Mean over K samples.
**Variance hazard:** unbiased but high-variance — `Var ∝ E[f²] · E[‖score‖²]` per Ranganath-2014 Theorem 1. Document `mcSamples ≥ 64` floor and pair with V5 control variates.

**V5 ControlVariates(estimator, baseline_fn) → variance-reduced estimator** [~110 LOC]
Generic control-variate wrapper around V4: subtract a `baseline(z) ≈ f(z)` from `f(z)` before scoring, then add back the analytic mean of `baseline`. Three baselines ship pre-rolled:
- **Constant baseline** `b = E[f]` estimated as running mean (Williams-1992 REINFORCE-with-baseline).
- **Score-correlated baseline** `b = (Σ_k f_k · score_k²) / (Σ_k score_k²)` per Ranganath-2014 §5.1 (closed-form variance-minimizing constant baseline).
- **Leave-one-out baseline** `b_i = (1/(K-1)) Σ_{j≠i} f(z_j)` per Mnih-Rezende-2016 VIMCO (pre-rolled for V18).
Cross-link to slot 195-N20 ControlVariateSGD; co-ship the substrate.

### Tier 1 — VI outer loops (composes Tier 0)

**V6 BBVI(logJoint, qFamily, φ_0, mcSamples, lr, T, rng) → φ_T** [~190 LOC]
Ranganath-Gerrish-Blei-2014 Black-Box VI: outer-loop `optim.GradientDescent` (or 220-F8 Adam) over V4 score-function gradient with V5 control variates. Mean-field Gaussian default `q_φ(z) = Π_i N(μ_i, σ_i²)` so `φ` has 2D parameters for D latent dims. **Direct overlap with 169-S15** — co-ship.

**V7 ADVI(logJoint, dim, mcSamples, lr, T, autodiffTape) → (μ, σ)** [~210 LOC]
Kucukelbir-Tran-Ranganath-Gelman-Blei-2017-JMLR-18 Automatic-Differentiation VI. Mean-field Gaussian q in **transformed space** `z̃ = T(z)` where T is the ambient-to-real map (e.g., `log` for positive params, `logit` for [0,1] params, identity for ℝ); ELBO gradient via V3 reparameterization through `autodiff.Tape`. **Two-stage transform** is the ADVI signature contribution: (i) bijective `T: support(z) → ℝ^D`, (ii) Gaussian on `ℝ^D` then transform back. Ships seven canonical transforms pre-rolled (`Identity`, `Log`, `Logit`, `LogShift(a, b)` for `(a, b)` interval, `Cholesky` for SPD matrices, `StickBreaking` for simplex, `Ordered` for sorted vectors). **Direct overlap with 169-S16** — co-ship.

**V8 NaturalGradientVI(logJoint, qExpFam, φ_0, lr, T) → φ_T** [~240 LOC]
Amari-1998 / Hoffman-2013 natural-gradient: precondition `∇_φ ELBO` by inverse Fisher information `I(φ)⁻¹` to descend on the Fisher-Rao manifold of the variational family rather than parameter-space. For exponential-family `q_φ` with natural parameters η, the natural gradient is **the gradient of ELBO w.r.t. mean parameters μ = E_q[T(z)]** (Hoffman-2013 §2.2) — closed-form for Gaussian/Beta/Gamma/Dirichlet-mean-field. Composes `infogeo.FisherInformation` (153-S2 keystone — currently named, not built) + V3 reparameterized gradient. **Strict block on 153-S2 + autodiff/dual.go** if a fully general HVP path is wanted; closed-form-Fisher-mean-param path is shippable today against existing reverse-mode for the seven `prob.Distribution`s.

**V9 StochasticVI-Hoffman(logJoint, qExpFam, dataset, batchSize, lr, T, rng) → φ_T** [~220 LOC]
**Hoffman-Blei-Wang-Paisley-2013-JMLR-14 *Stochastic Variational Inference***, the canonical mini-batch natural-gradient VI for conjugate-exponential models (LDA / HMM / GMM / matrix-factorization). Outer loop:
1. Subsample mini-batch B ⊂ {1, ..., N}.
2. For each i ∈ B: compute local variational params φ_i via coordinate ascent (CAVI inner loop).
3. Compute "intermediate global parameters" λ̂ = η_prior + (N/|B|) Σ_{i∈B} sufficient_stats(φ_i).
4. Update global natural parameters λ_{t+1} = (1 − ρ_t) λ_t + ρ_t λ̂ with `ρ_t = (t + τ)^{-κ}`, `κ ∈ (0.5, 1]`.
**The single-most-cited VI paper in the 2010s** — gates LDA-at-Wikipedia-scale and every conjugate-exponential mini-batch consumer. Composes 220-F4 MiniBatchSGD outer loop + V8 natural gradient + a `ConjugateExpFamily` interface (~40 LOC, conjugate-prior-update closure for the 7 distributions). Document Robbins-Monro convergence: `Σ ρ_t = ∞, Σ ρ_t² < ∞` strictly (`κ > 0.5`).

**V10 DoublyStochasticVI(logJoint, q, dataset, batchSize, mcSamples, lr, T, rng) → φ_T** [~170 LOC]
Titsias-Lazaro-Gredilla-2014-ICML *Doubly Stochastic Variational Bayes for non-Conjugate Inference*. Two sources of stochasticity: (i) mini-batch over data (220-F4), (ii) MC samples for ELBO (V3 reparameterization or V4 score). For **non-conjugate** likelihoods (logistic regression, Gaussian process classification, deep generative models). Composes V3 + 220-F4 + V8 natural gradient when q is exp-family. **The canonical recipe for VI on neural-network-parameterized likelihoods** — every modern VAE consumer trains via doubly-stochastic-VI even when not naming it that.

### Tier 2 — VAE family (composes Tier 0/1)

**V11 VAE(encoder, decoder, x, lr, T, mcSamples, rng) → (φ_enc, θ_dec)** [~280 LOC]
Kingma-Welling-2013-ICLR / Rezende-Mohamed-Wierstra-2014-ICML *Variational AutoEncoder*. **The single highest-impact primitive in slot 239** (>1.2M citations in the deep-generative-modeling literature). Composes:
- `EncoderNetwork` interface (V20): closure `Forward(x) → (μ, σ)` parameterized by `φ_enc`. **Caller supplies the network; reality provides the SVI loop.** This is reality's competitive moat: no tensor type, no NN-layer surface, just a closure-based amortization contract.
- `DecoderNetwork` interface: closure `Forward(z) → log p(x | z)` parameterized by `θ_dec`.
- V3 reparameterized gradient via `autodiff.Tape` through `z = μ + σε`.
- V2 closed-form KL term `KL(N(μ, σ²) ‖ N(0, I)) = ½ Σ_i (μ_i² + σ_i² − log σ_i² − 1)`.
- 220-F8 Adam outer loop with cosine-warmup-then-cosine-decay schedule (220-F8 + 220-Tier-6 schedule cross-link).
ELBO objective: `L(x, φ_enc, θ_dec) = E_{q(z|x)}[log p(x|z)] − KL(q(z|x) ‖ p(z))`. **Crown-jewel observation: reality's VAE is a 280-LOC library, not a framework.** Cross-link to 240 for the upstream flow that V21 IAF wraps the encoder with.

**V12 betaVAE(encoder, decoder, x, beta, lr, T, mcSamples, rng) → (φ_enc, θ_dec)** [~30 LOC]
Higgins-Matthey-Pal-Burgess-Glorot-Botvinick-Mohamed-Lerchner-2017-ICLR *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. Three-line modification of V11: scale the KL term by `β > 1` to encourage **disentangled** latent representations. ELBO becomes `L_β = E[log p(x|z)] − β · KL(q ‖ p)`. Document the Burgess-Higgins-Pal-Matthey-Watters-Desjardins-Lerchner-2018 capacity-controlled variant `L_C = E[log p(x|z)] − γ · |KL(q ‖ p) − C|` with `C` schedule.

**V13 IWAE(encoder, decoder, x, K, lr, T, rng) → (φ_enc, θ_dec)** [~240 LOC]
Burda-Grosse-Salakhutdinov-2016-ICLR *Importance-Weighted AutoEncoder*. Tighter ELBO via importance-weighted MC: `L_K(x) = E_{z_1, ..., z_K ~ q}[log (1/K) Σ_k p(x, z_k)/q(z_k|x)] ≥ ELBO`. Provably converges to `log p(x)` as `K → ∞`. Composes V11 VAE forward pass + new `LogSumExp` autodiff op (gates 011-Tier-2 LSE; ~30 LOC) + Burda-2016 §3.1 unbiased gradient estimator. **Saturates R-VARIATIONAL-BOUND-TIGHTNESS 3/3** against V11 ELBO and `log p(x)`-from-AIS-or-NestedSampling (238-M22 cross-link): bound monotonically tightens with K.

**V14 StickBreakingVAE(encoder, decoder, x, K_max, alpha_0, lr, T, rng) → (φ_enc, θ_dec)** [~280 LOC]
Nalisnick-Smyth-2017-ICLR *Stick-Breaking Variational Autoencoders*. **Singular cutting-edge piece of slot 239.** Replaces the fixed-D Gaussian latent with an **unbounded** stick-breaking `π_k = β_k Π_{j<k} (1 − β_j)`, `β_k ~ Kumaraswamy(a_k, b_k)` (encoder outputs `(a_k, b_k)`), then `z = Σ_k π_k μ_k` with cluster-means `μ_k`. Composes:
- **Kumaraswamy distribution** ~40 LOC new (closed-form CDF `1 − (1 − x^a)^b` and inverse-CDF `(1 − (1 − u)^{1/b})^{1/a}` makes it reparameterizable directly — unlike Beta which needs rejection or Marsaglia-Tsang).
- V3 reparameterized gradient through `(a, b) → β → π`.
- V2 closed-form KL `KL(Kumaraswamy(a, b) ‖ Beta(α, β))` via Taylor-expansion approximation (Nalisnick-Smyth-2017 §A).
- 228-B6/B7 cross-link: stick-breaking + GEM.
**Singular reality moat:** combined with 228-B9 DPMixturePosterior, reality ships the only zero-dep nonparametric-Bayesian + amortized-VAE stack in any language.

**V15 VampPrior(encoder, decoder, x, K, lr, T, rng) → (φ_enc, θ_dec)** [~210 LOC]
Tomczak-Welling-2018-AISTATS *VampPrior*. Replaces the standard Gaussian prior `p(z) = N(0, I)` with a learned mixture-of-encoded-pseudo-inputs `p(z) = (1/K) Σ_k q(z | u_k)` where `u_k` are K **trainable pseudo-inputs** in the data space. Three-line modification of V11: introduce K pseudo-input variables `u_1, ..., u_K`, change KL term to MC-estimated `KL(q(z|x) ‖ (1/K) Σ_k q(z | u_k))`. Saturates **R-PRIOR-MISMATCH** pin: VampPrior tightens ELBO over standard-Gaussian prior on multi-modal data.

**V16 ConditionalVAE(encoder, decoder, x, y, lr, T, rng) → (φ_enc, θ_dec)** [~120 LOC]
Sohn-Lee-Yan-2015-NeurIPS *Conditional VAE*. Add conditioning variable `y` (label, image, text) to both encoder `q(z | x, y)` and decoder `p(x | z, y)`. ELBO becomes `L(x, y) = E_{q(z|x, y)}[log p(x | z, y)] − KL(q(z|x, y) ‖ p(z | y))`. Pure interface widening of V11.

**V17 WakeSleepAlgorithm(encoder, decoder, x, lr, T, rng) → (φ_enc, θ_dec)** [~200 LOC]
Hinton-Dayan-Frey-Neal-1995-Science-268 *Wake-Sleep Algorithm for Helmholtz Machines*. Pre-VAE precursor that alternates two phases:
- **Wake phase:** sample `z ~ q(z | x)` (encoder); update decoder θ via `∇_θ log p(x | z)` — same as VAE decoder update.
- **Sleep phase:** sample `z ~ p(z), x ~ p(x | z)` (decoder); update encoder φ via `∇_φ log q(z | x)` — **uses the WRONG-WAY KL** `KL(p(z|x) ‖ q(z|x))` rather than VAE's `KL(q(z|x) ‖ p(z|x))`.
Document the wake-sleep-vs-VAE training-objective asymmetry (Bornschein-Bengio-2015 *Reweighted Wake-Sleep* fixes the asymmetry via importance weights). Pure historical-completeness primitive but enables **discrete-latent training** (no reparameterization required) — useful for V14 stick-breaking when Kumaraswamy is not used.

**V18 VIMCO(encoder, decoder, x, K, lr, T, rng) → (φ_enc, θ_dec)** [~190 LOC]
Mnih-Rezende-2016-ICML *Variational Inference for Monte Carlo Objectives*. **Singular reality competitive moat.** Multi-sample score-function gradient with leave-one-out baseline — recovers IWAE-quality bounds for **discrete latent variables** (Categorical / Bernoulli / Poisson) where reparameterization fails. Composes V4 + V5 (leave-one-out baseline) + V13 IWAE-style multi-sample LSE objective.

**V19 AmortizedFamily(encoder, qBase) → ParametricVariationalFamily** [~130 LOC]
The architectural keystone for amortization: an `AmortizedFamily` is a `prob.Distribution` whose parameters are **a function of x** — `q(z | x; φ) = qBase(z; encoder(x; φ))`. All V11/V12/V13/V14/V15/V16/V18 inherit from this surface. Covers:
- Mean-field Gaussian: `q(z | x) = N(μ(x; φ), diag(σ²(x; φ)))`.
- Full-covariance Gaussian: `q(z | x) = N(μ(x; φ), L(x; φ) L(x; φ)ᵀ)` with Cholesky output (`linalg.Cholesky` cross-link).
- Stick-breaking: `q(z | x) = SBP(a(x; φ), b(x; φ))` per V14.
- Flow-based: `q(z | x) = T_φ(N(μ(x), σ(x)))` per V21 IAF.

**V20 EncoderNetwork interface** [~60 LOC]
The amortization contract: caller-supplied `Forward(x) → params` closure plus `Parameters() []float64` flat parameter access for SGD updates. **Reality ships zero NN layers; aicore (or any consumer) provides the network.** Crown-jewel architectural decision: this keeps `prob/vi/` a math library, not a deep-learning framework.

### Tier 3 — flow-based VI (cross-link to slot 240)

**V21 InverseAutoregressiveFlow(encoder, k_layers, hidden_dim) → AmortizedFamily** [~280 LOC]
Kingma-Salimans-Jozefowicz-Chen-Sutskever-Welling-2016-NeurIPS *Improved Variational Inference with Inverse Autoregressive Flow*. **Lives in slot 239 (not 240) because it is the canonical amortized-posterior flow** — IAF is specifically designed to make `q(z | x)` more flexible inside VI loops. Slot 240 owns the **prior-side** flow zoo (RealNVP/Glow/MAF/neural-spline); slot 239 owns the **posterior-side** flow used inside VAE encoders. Composes:
- Initial Gaussian `z_0 = μ(x) + σ(x) ⊙ ε, ε ~ N(0, I)`.
- K successive IAF layers `z_t = μ_t(z_{t-1}) + σ_t(z_{t-1}) ⊙ z_{t-1}` where `μ_t, σ_t` are autoregressive (each output depends only on prior dims).
- Log-determinant of Jacobian is **diagonal** `Σ_t Σ_i log σ_{t,i}` — O(D) per layer (vs O(D²) for general flows).
- Final `q(z | x) = N(0, I)`-pushed-through-K-IAF-layers; closed-form log-density via change-of-variable.
**Singular flow-VI moat:** O(D)-Jacobian-determinant + autoregressive structure makes IAF the **only normalizing flow that scales to high-D variational posteriors**. Cross-link to 240 for the static MAF / RealNVP / Glow that use the same MADE-style autoregressive masks.

**V22 SurjectiveFlow(encoder, layers) → AmortizedFamily** [~180 LOC]
Nielsen-Jaini-Hoogeboom-Winther-Welling-2020-NeurIPS *SurVAE Flows*. **Surjective** (not bijective) layers — allow dimensionality changes (`z ∈ ℝ^d` to `x ∈ ℝ^D` with `D > d` via injection or D > d via projection) — generalize the flow framework to handle latent-dim selection inside VAE. Composes V21 IAF backbone + SurVAE composition rules (Nielsen-2020 §3). Frontier 2020+ piece; reality ships the math, not the network.

**V23 FlowVI-glue(flow_object_from_240, logJoint, x, lr, T, rng) → φ_T** [~180 LOC]
Adapter that takes **any** `Flow` object from slot 240's `prob/flow/` package (Planar / Radial / RealNVP / Glow / Neural-Spline / Continuous) and uses it as the variational distribution `q_φ(z) = T_φ(N(0, I))`. ELBO via change-of-variable: `log q_φ(z) = log N(T_φ⁻¹(z)) − log |det ∂T_φ/∂z|`. Composes V3 reparameterization (the flow IS the reparameterization map) + caller-supplied flow + 220-F8 Adam outer. **The bridge between slot 240's static-flow zoo and slot 239's VI machinery.**

---

## 3. Composition graph (DAG)

```
prob.LogPDF (169-S8 keystone)            ┐
prob.RNGSampler (18-fold pile-up keystone) ├── architectural keystones (cross-cutting blockers)
autodiff.Tape (already shipping)         │
optim.SGD/Adam (220-F-family)            ┘

V1  ELBO  ────────────── E[logp − logq] (gates everything)
V2  KLClosedForm ──────── 7 analytic KLs (cheapest 1-day shippable, saturates 7/7)
V3  ReparameterizedGradient (location-scale q) ────── V11 VAE ── V12 βVAE
V4  ScoreFunctionGradient (discrete q)    ────── V6 BBVI                 │
V5  ControlVariates (variance reduction)         │                       │
                                                  │                       ├── V13 IWAE ─── V18 VIMCO
V6  BBVI = V4 + V5 + optim.GD                    │                       │
V7  ADVI = V3 + autodiff.Tape + 7-transform set  │                       └── V14 StickBreakingVAE (Kumaraswamy)
V8  NaturalGradientVI = V3 + Fisher_meanparam (153-S2)         (228-B6/B7 cross-link)
V9  StochasticVI-Hoffman = V8 + 220-F4 MiniBatchSGD            ── V15 VampPrior
V10 DoublyStochasticVI = V3 + V4 + V8 + 220-F4                  ── V16 ConditionalVAE
                                                                  ── V17 WakeSleep (historical)
V11 VAE = V3 + V20 EncoderNetwork + V19 AmortizedFamily          ── V18 VIMCO (discrete VAE)
V12 βVAE = V11 with KL × β
V13 IWAE = V11 with K-sample LSE objective + LSE autodiff op (011-T2)

V19 AmortizedFamily ─── architectural keystone for V11-V18, V21-V23
V20 EncoderNetwork ─── caller-supplied closure (no NN type in reality)

V21 IAF ─── amortized posterior flow (lives in 239, not 240)
V22 SurjectiveFlow ─── 2020+ frontier
V23 FlowVI-glue ─── adapter to slot-240's static-flow zoo (RealNVP/Glow/MAF)

  cross-link to 240 ─── prior-side flow zoo (Planar/Radial/RealNVP/Glow/MAF/NeuralSpline/Continuous)
  cross-link to 241 ─── score-based / diffusion / flow-matching (Lipman-2023)
```

---

## 4. Saturation pins this slot unlocks

- **R-CLOSED-FORM-PINNED-TO-NUMERICAL-KL 7/7 (V2):** seven analytic KLs over reality's seven `prob.Distribution`s pinned at 1e-9 against `infogeo.KLDivergenceNumerical` trapezoidal-rule baseline. The canonical R-MUTUAL idiom (closed-form × numerical-quadrature × autodiff-through-density agree).
- **R-VARIATIONAL-BOUND-TIGHTNESS 3/3 (V11/V13/V22):** ELBO ≤ IWAE-K=10 ELBO ≤ IWAE-K=100 ELBO ≤ log p(x)-via-AIS (cross-link 238-M22 nested-sampling). Three orthogonal lower-bound estimators monotonically tightening to the true marginal likelihood.
- **R-REPARAM-VS-SCORE-VARIANCE 2/2 (V3/V4/V5):** on a smooth Gaussian-Gaussian conjugate model, V3 reparameterized-gradient variance is provably ≤ V4 score-function variance per Mohamed-Rosca-Figurnov-Mnih-2020 — pin the inequality with K=100 MC samples at 1e-3 relative tolerance, and pin the V5 control-variate-corrected V4 variance vs raw V4 at ≥10× reduction.
- **R-VAE-CONJUGATE-LIMIT 3/3 (V11):** as decoder collapses to identity (`p(x | z) = N(z, σ²I)` with `σ → 0`) and prior collapses (`p(z) = N(0, λ⁻¹I)` with `λ → ∞`), VAE posterior `q(z | x)` collapses to ridge regression S1 from 169. Three derivation paths (VAE / closed-form-Gaussian / 169-S1-ridge) converging on a single `μ(x) = (XᵀX + λI)⁻¹ Xᵀy` is the R-CONJUGATE-DUAL idiom from 169 extended to amortized inference.
- **R-IWAE-MONOTONIC-TIGHTENING 1/1 (V13):** `L_K ≤ L_{K+1} ≤ log p(x)` per Burda-Grosse-Salakhutdinov-2016 Theorem 1 — pin monotonicity at 1e-6 over `K ∈ {1, 2, 5, 10, 50, 100}` on a 2D Gaussian mixture model.
- **R-STOCHASTIC-VI-CONJUGATE-MATCH 1/1 (V9):** Hoffman-2013 Stochastic VI on a fully-conjugate LDA model (Dirichlet-Categorical mean-field) converges to the **exact same posterior** as full-batch CAVI per Hoffman-2013 Theorem 1 — pin posterior-Dirichlet-α-parameter agreement at 1e-7 with `T = 10^4` mini-batches of size 50.
- **R-WAKESLEEP-VS-VAE 1/1 (V17):** wake-sleep on a Gaussian-Gaussian model converges to the same encoder/decoder as VAE up to the wake-sleep-vs-VAE-asymmetric-KL bias term; pin the asymmetry quantitatively per Bornschein-Bengio-2015 §3 reweighted wake-sleep correction.

---

## 5. Connective-tissue LOC budget

| ID | Capability | LOC | Tier | Blocks-on |
|----|-----------|-----|------|-----------|
| (keystone) | `prob.LogPDF` interface (169-S8) | 80 | 0 | — |
| (keystone) | `prob.RNGSampler` interface (18-fold pile-up) | 80 | 0 | — |
| V1 | ELBO | 120 | 0 | LogPDF, RNG |
| V2 | KLClosedForm (7 analytic KLs + Digamma) | 110 | 0 | — |
| V3 | ReparameterizedGradient | 180 | 0 | autodiff.Tape, RNG |
| V4 | ScoreFunctionGradient | 150 | 0 | LogPDF, RNG, autodiff.Tape |
| V5 | ControlVariates | 110 | 0 | V4 |
| V6 | BBVI | 190 | 1 | V4, V5, optim.GD or 220-F8 Adam |
| V7 | ADVI (+ 7-transform set) | 210 | 1 | V3 |
| V8 | NaturalGradientVI | 240 | 1 | V3, infogeo.Fisher (153-S2) |
| V9 | StochasticVI-Hoffman | 220 | 1 | V8, 220-F4 MiniBatchSGD |
| V10 | DoublyStochasticVI | 170 | 1 | V3, V4, 220-F4 |
| V11 | VAE | 280 | 2 | V3, V20 |
| V12 | βVAE | 30 | 2 | V11 |
| V13 | IWAE (+ LogSumExp autodiff op) | 240 | 2 | V11, 011-T2 LSE |
| V14 | StickBreakingVAE (+ Kumaraswamy distribution) | 280 | 2 | V11, 228-B6/B7 |
| V15 | VampPrior | 210 | 2 | V11 |
| V16 | ConditionalVAE | 120 | 2 | V11 |
| V17 | WakeSleepAlgorithm | 200 | 2 | V20 |
| V18 | VIMCO (multi-sample score-function for discrete) | 190 | 2 | V4, V5, V13 |
| V19 | AmortizedFamily interface | 130 | 2 | V20 |
| V20 | EncoderNetwork interface | 60 | 2 | — |
| V21 | InverseAutoregressiveFlow (IAF) | 280 | 3 | V19, V20 |
| V22 | SurjectiveFlow | 180 | 3 | V21 |
| V23 | FlowVI-glue (slot 240 cross-link) | 180 | 3 | V19, slot 240 Flow type |
| **Σ** | | **3,860** | | |

(Excluding 169-S15/S16/S17 deduplicated overlap: net new LOC for slot 239 ≈ 3,260 since V6 BBVI / V7 ADVI / V21 IAF would co-ship under both review headings — 169 owns the **deterministic-fit-distribution-to-data** axis with VI as one application; slot 239 owns the **amortized-VI + VAE-family + reparameterization-gradient + flow-VI** axis as the dedicated SVI scoping.)

Pure-glue ratio: ~78% composition over `optim.GradientDescent` body + `autodiff.Tape` reverse-mode + `prob.NormalPDF/Beta/Gamma/Categorical/Dirichlet` + `linalg.Cholesky` for full-cov q + 220-F-family SGD/Adam + Box-Muller from `genetic.go:58-65`. ~22% genuinely-new math (V14 Kumaraswamy distribution at ~40 LOC + KL(Kumaraswamy ‖ Beta) Taylor approximation at ~30 LOC; V18 VIMCO multi-sample LOO baseline at ~60 LOC; V21 IAF autoregressive structure + diagonal-Jacobian-determinant at ~90 LOC; V22 SurjectiveFlow composition rules at ~50 LOC; V8 closed-form-Fisher-mean-parameter for the 7 distributions at ~80 LOC).

---

## 6. Recommended PR sequence

**PR-1: substrate (LogPDF + RNG keystones + V1 ELBO + V2 KLClosedForm + V3 ReparameterizedGradient + V4 ScoreFunctionGradient + V5 ControlVariates) — ~830 LOC source, ~360 LOC tests, 1.5 days**
First VI surface in `reality`. Co-ships with 169-S8 LogPDF and 220-F1 RNG keystones. **Saturates R-CLOSED-FORM-PINNED-TO-NUMERICAL-KL 7/7 immediately** via V2 against existing `infogeo.KLDivergenceNumerical`. V3+V4+V5 are the gradient-estimator triplet that gates every Tier-1+ primitive.

**PR-2: V6 BBVI + V7 ADVI + V10 DoublyStochasticVI — ~570 LOC source, ~280 LOC tests, 2 days**
Black-box + automatic-differentiation + non-conjugate VI. Co-ship with 169-S15/S16. Saturates R-REPARAM-VS-SCORE-VARIANCE 2/2 and R-IWAE-MONOTONIC-TIGHTENING 1/1.

**PR-3: V11 VAE + V12 βVAE + V20 EncoderNetwork + V19 AmortizedFamily — ~500 LOC source, ~240 LOC tests, 2 days**
**The crown-jewel PR for slot 239.** Lands the canonical amortized-VI + VAE + amortized-family stack. Saturates R-VAE-CONJUGATE-LIMIT 3/3 by collapsing decoder → identity, prior → flat, and matching against 169-S1 ridge regression.

**PR-4: V13 IWAE + V14 StickBreakingVAE (+ Kumaraswamy + KL-K-to-Beta) — ~520 LOC source, ~260 LOC tests, 2 days**
Importance-weighted bound + nonparametric-latent-dim VAE. Cross-link to 228-B6/B7 stick-breaking. Saturates R-VARIATIONAL-BOUND-TIGHTNESS 3/3 and the 011-T2 LSE autodiff-op cross-link. **Singular cutting-edge piece of slot 239.**

**PR-5: V8 NaturalGradientVI + V9 StochasticVI-Hoffman — ~460 LOC source, ~240 LOC tests, 2 days**
The Hoffman-2013 stochastic-natural-gradient VI canon for conjugate-exponential models. **Strict block on 153-S2 Fisher-Information-from-Distribution** (named in 153 review, ~80 LOC) — ship 153-S2 as a co-shipped Tier-0 dependency. Saturates R-STOCHASTIC-VI-CONJUGATE-MATCH 1/1 on LDA.

**PR-6: V15 VampPrior + V16 ConditionalVAE + V17 WakeSleepAlgorithm + V18 VIMCO — ~720 LOC source, ~360 LOC tests, 3 days**
The VAE-family completion PR. VIMCO is the singular reality moat for **discrete-latent VAE** training; wake-sleep is the historical-completeness primitive that demonstrates the wrong-way-KL asymmetry; VampPrior is the learnable-prior generalization; ConditionalVAE is the conditioning-variable widening.

**PR-7: V21 IAF + V22 SurjectiveFlow + V23 FlowVI-glue — ~640 LOC source, ~280 LOC tests, 3 days**
Flow-based VI: amortized-posterior IAF + 2020+ surjective-flow extension + glue to slot 240's static-flow zoo. **Strict block on slot 240's `prob/flow/` package** (RealNVP / Glow / MAF / Neural-Spline / Continuous types). Recommend coordinated PR with slot 240 lead.

Total: ~4,240 LOC source + ~2,020 LOC tests across 7 PRs over ~14 engineer-days. PR-1 is 1.5-day standalone with maximum saturation-leverage (R-CLOSED-FORM-KL 7/7 in one shot); PR-3 (V11 VAE) is the single highest-impact primitive for downstream consumers (deep-generative-modeling onramp); PR-4 (V14 StickBreakingVAE) is the singular cutting-edge moat; PR-7 (V21-V23 flow-VI) is the crown-jewel composition that bridges slot 240's static-flow zoo with slot 239's amortized-VI machinery.

---

## 7. Cycle-hazard analysis

Proposed import directions:

```
prob/vi/      ──→  prob/         (LogPDF, RNGSampler, NormalPDF, Beta, etc)
prob/vi/      ──→  optim/        (GD, Adam — 220 family)
prob/vi/      ──→  optim/proximal/ (Fbs for sparsity-promoting q; V8 mean-field cap)
prob/vi/      ──→  autodiff/     (Tape, Backward — already exists for prob/copula)
prob/vi/      ──→  infogeo/      (Fisher, KLDivergenceNumerical baseline for V2)
prob/vi/      ──→  linalg/       (Cholesky for full-cov Gaussian q)
prob/vi/vae/  ──→  prob/vi/      (V11-V18 build on V1-V10)
prob/vi/flow/ ──→  prob/vi/, prob/flow/ (slot 240 — the only NEW edge)
```

**Three new cross-package edges** beyond what 169 already proposes (`prob → optim/proximal`, `prob → infogeo`, `prob → autodiff`), all in the same direction as existing precedent (toward more-foundational packages). The single new edge is `prob/vi/flow/ → prob/flow/` which assumes slot 240 ships its `prob/flow/` package; if slot 240 lands first at top-level `flow/` instead, the edge becomes `prob/vi/flow/ → flow/`.

**No cycles.** `optim/`, `autodiff/`, `infogeo/`, `linalg/` do not need to import `prob/vi/` for any V1-V23 primitive. The `EncoderNetwork` interface (V20) is consumer-side (no `aicore` import from reality).

---

## 8. Precision hazards documented

Per CLAUDE.md "Precision documented, not assumed":

- **V1 ELBO:** MC estimate has variance `O(1/K)`; document `K ≥ 16` for ADVI-style and `K ≥ 64` for BBVI-with-score-function. V3 reparameterization typically tolerates `K = 1`.
- **V2 KLClosedForm:** `KL(N_p ‖ N_q)` near `σ_q → 0` blows up; document `σ_q ≥ 1e-6` floor. Beta/Gamma KLs need digamma (Lanczos ~1e-12 for `α, β > 0.5`; series fallback below).
- **V3 ReparameterizedGradient:** softplus-parameterize `σ = log(1 + exp(ρ))` not `exp(ρ)` — keeps gradient finite at `ρ → −∞` while enforcing `σ > 0` (Kingma-Welling-2014 §C.2 fn 6).
- **V4 ScoreFunctionGradient:** unbiased but `Var = O(K⁻¹·E[‖score‖²·f²])`; always pair with V5 control variates. `mcSamples ≥ 64` floor.
- **V7 ADVI:** ambient-to-real transform `T` introduces a Jacobian correction `log |det J_T|` to ELBO; document each of the 7 transforms' closed-form Jacobian per Kucukelbir-2017 §3.3.
- **V8 NaturalGradientVI:** full-cov Fisher inversion `O(D³)`; mean-field `O(D)`. Dispatch: full-cov for `D ≤ 50`, mean-field otherwise.
- **V9 StochasticVI-Hoffman:** schedule `ρ_t = (t + τ)^{-κ}` requires `κ ∈ (0.5, 1]` per Robbins-Monro 1951. Default `(τ, κ) = (1, 0.6)` per Hoffman-2013. Panic at `κ ≤ 0.5`.
- **V11 VAE:** posterior collapse (Bowman-2015 / Razavi-2019) where `q(z|x) → p(z)` is a known failure mode; document KL-annealing + free-bits (Kingma-2016 IAF §C.4) as mitigations.
- **V13 IWAE:** Burda-2016 estimator is unbiased for the **bound** but not the gradient — Roeder-Wu-Duvenaud-2017 *Sticking the Landing* showed higher gradient variance than V11 for fixed K; recommend V18 VIMCO for discrete latents.
- **V14 StickBreakingVAE:** Kumaraswamy `(a, b) > 0`, softplus-parameterized. K-to-Beta KL via Taylor approximation (Nalisnick-2017 §A) ~1e-3 for `a, b ∈ [0.5, 5]`, degrading at extremes.
- **V18 VIMCO:** LOO-baseline has `Var = O(1/K)` matching reparameterization — but only for smooth f; document degradation at high curvature.
- **V21 IAF:** autoregressive structure forces sequential `z_i` evaluation — O(D) per layer. MADE-style mask is consumer-side concern.
- **V23 FlowVI-glue:** change-of-variable `log q(z) = log p_base(T⁻¹(z)) − log |det J_T(z)|` is a slot-240 contractual obligation on every `Flow` object.

---

## 9. Distinct from prior agents (provenance)

- **011-015 autodiff** — 012-T3 names "stochastic AD (reparam + score-function)"; slot 239 is the consumer-side pull justifying that axis.
- **101-105 / 116-120** — 220-F-family (101) + LogPDF/RNG debt (117) consumed verbatim; slot 239 is the 18th Block-C review demanding the unified-RNG keystone.
- **168 physics-autodiff** — 168-A1 dual-numbers is the cleanest path to V8's HVP step; 168 ships substrate, 239 ships consumer.
- **169 prob-optim (CLOSEST SIBLING, MAJOR OVERLAP)** — 169 owns deterministic-fit (S1-S18); slot 239 owns amortized-VI + VAE-family + reparameterization + flow-VI. Overlap V6 = S15 BBVI, V7 = S16 ADVI — co-ship; ship code once. **21 of 23 primitives unique to slot 239**.
- **195 optim-prob** — 195 owns SDE-as-sampler / CMA-ES / IGO; V5 ≈ N20 ControlVariateSGD — co-ship.
- **220 new-stochastic-opt** — 239 CONSUMES F1-F4 as Tier-0 and F8 Adam as Tier-1.
- **228 new-bayes-nonparam** — V14 StickBreakingVAE is the **amortized counterpart** of 228-B13 VariationalDP: same Sethuraman/GEM substrate, learned encoder instead of CAVI. Co-ship B6 StickBreakingSample.
- **236 new-rkhs** — K22 KSD is a non-VI alternative to ELBO; cross-link only.
- **237 new-gaussian-process** — G24 SVGP uses VI machinery from slot 239; cross-link.
- **238 new-mcmc** — orthogonal axis; pin V11 VAE × M22 nested-sampling on `log p(x)` for R-VARIATIONAL-BOUND-TIGHTNESS 3/3.
- **240 new-normalizing-flows (CLOSEST SIBLING)** — 240 owns the static-flow zoo (Planar/Radial/RealNVP/Glow/MAF/NeuralSpline/Continuous/FFJORD); slot 239 owns the amortized-posterior IAF + flow-VI glue. Disjoint rosters; together ship the full Rezende-2015 → Dinh-2017 → Kingma-2016 → Glow-2018 → Chen-2018 → Grathwohl-2019 → Lipman-2023 canon. **Strict shared dependency on slot 240's `Flow` interface.**
- **241 new-diffusion-models** — 241 is the non-VI generative axis (score-matching not ELBO); cross-links only via shared encoder/decoder interface.

Slot 239 is the **SVI-canonical / amortized-VI / VAE-family slot** — every primitive is what 2014-2024 deep-generative-modeling literature calls by these specific names (VAE, β-VAE, IWAE, IAF, VampPrior, VIMCO, Wake-Sleep, StickBreakingVAE).

---

## 10. Bottom line

`reality/prob/` ships **ZERO** variational-inference / amortized-encoder / VAE-family / reparameterization-gradient / wake-sleep / IAF surface despite being the obvious target package for the entire 2014-2024 deep-generative-modeling canon. **Twenty-three primitives V1-V23 totalling ~3,860 LOC of pure connective tissue** stand up the entire amortized-VI + VAE-family + reparameterization-gradient + Hoffman-stochastic-VI + flow-VI pipeline on existing v0.10.0 surfaces (`autodiff.Tape` reverse-mode, `prob.NormalPDF/Beta/Gamma/Categorical/Dirichlet`, `linalg.Cholesky` for full-covariance q, `optim/proximal.Fbs` for sparsity-promoting q, Box-Muller from `genetic.go:58-65`, plus 220-F-family SGD/Adam outer loop and 169-S8 LogPDF / 18-fold-pile-up RNGSampler keystones).

**Cheapest one-day shippable**: V2 KLClosedForm at ~110 LOC saturates a 7/7 R-CLOSED-FORM-PINNED-TO-NUMERICAL-KL pin against existing `infogeo.KLDivergenceNumerical` immediately. **Cheapest 1.5-day standalone PR**: PR-1 substrate (V1 ELBO + V2 KL + V3 reparameterization + V4 score-function + V5 control variates = ~830 LOC) lands the **first VI surface in reality**. **Single highest-impact PR for downstream consumers**: PR-3 V11 VAE + V12 βVAE + V20 EncoderNetwork at ~500 LOC delivers the canonical amortized-VAE + amortized-family stack — the deep-generative-modeling onramp that any `aicore` consumer needs. **Singular cutting-edge moat**: PR-4 V14 StickBreakingVAE (Nalisnick-Smyth-2017 + Kumaraswamy distribution + 228-B6/B7 stick-breaking cross-link) — no zero-dep library composes nonparametric-Bayesian + amortized-VAE in one canonical surface. **Crown-jewel composition**: PR-7 V21 IAF + V22 SurjectiveFlow + V23 FlowVI-glue at ~640 LOC — bridges slot 240's static-flow zoo with slot 239's amortized-VI machinery, completing the Rezende-Mohamed-2015 → Kingma-2016 → Tomczak-2018 → Nielsen-2020 flow-VI literature in one canonical Go API.

The single most important conceptual identity slot 239 pins: **Jordan-1999 + Hoffman-2013 + Kingma-Welling-2013 + Rezende-Mohamed-Wierstra-2014 + Ranganath-Gerrish-Blei-2014 + Kucukelbir-2017 are six papers that all collapse into one identity** — `ELBO(φ) = E_{q_φ}[log p(x, z) − log q_φ(z)]` optimized via gradient descent over `φ`, with the gradient computed via reparameterization (V3 — when q is location-scale), score-function (V4 — when q is discrete), or natural-gradient (V8 — when q is exponential-family). Reality pins this identity as the canonical entry point for the entire 2014-2024 SVI literature: every paper in the canon is a specialization of the V1-V11 template, and the eleven specializations V12-V23 are all compositions of the same Tier-0 substrate.

**Reality is unusually well-positioned for slot 239 because (i) `autodiff.Tape` reverse-mode already handles the V3 reparameterization-gradient path, (ii) `prob.NormalPDF / Beta / Gamma / Categorical / Dirichlet` already supply the variational families, (iii) Box-Muller from `optim/genetic.go:58-65` already provides the Gaussian sampler, (iv) `linalg.Cholesky` already covers full-covariance Gaussian q, (v) `optim/proximal.Fbs` already covers non-smooth-prior MAP via FISTA, (vi) the consumer-side encoder-as-closure architectural decision (V20) keeps `prob/vi/` a math library, not a tensor framework — minimum architectural perturbation, maximum deep-generative-modeling unlock**.
