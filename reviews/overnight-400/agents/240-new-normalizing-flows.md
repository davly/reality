### 240 | new-normalizing-flows — Normalizing flows: planar, RealNVP, Glow, neural spline, continuous

**Summary line 1.** `reality` v0.10.0 ships **ZERO** normalizing-flow surface (verified 2026-05-08 by repo-wide grep on `Flow|Coupling|Bijection|Bijector|invertible|change.of.variable|Jacobian.*det|log.*det.*jac|RealNVP|Glow|FFJORD|Planar|Radial|Sylvester|MaskedAutoregressive|MAF\b|IAF\b|NeuralSpline|RationalQuadratic|NormalizingFlow|iResNet|ResFlow|Hutchinson|autoregressive|spline.*flow|continuous.*flow` returning ZERO callable matches across all 22 packages — the only nominal hits are doc-comments mentioning unrelated fluid `flow`, autodiff `gradient flows`, and changepoint `mass that flows from r=0`); slot 239 (new-svi) explicitly carves out the **amortized-posterior IAF + flow-VI glue** axis (V21/V22/V23) and defers the prior-side static-flow zoo to slot **240**; slot 168 (synergy-physics-autodiff A6) names continuous-time `NeuralODE` adjoint as a synergy bullet but does not ship the bijector/log-det machinery; slot 241 (new-diffusion-models) owns the **non-bijective** generative axis (score-matching / DDPM / SDE) — slot **240** owns the **bijective change-of-variables generative axis**: every static normalizing-flow primitive that 2015-2024 deep-generative-modeling literature treats as canonical but no zero-dep Go library composes. The 240-roster is the Rezende-Mohamed-2015-ICML-PlanarFlow + Rezende-Mohamed-2015-RadialFlow + vanderBerg-Hasenclever-Tomczak-Welling-2018-UAI-SylvesterFlow + Dinh-Sohl-Dickstein-Bengio-2017-ICLR-RealNVP + Kingma-Dhariwal-2018-NeurIPS-Glow + Papamakarios-Pavlakou-Murray-2017-NeurIPS-MAF + Germain-Gregor-Mitchell-Larochelle-2015-ICML-MADE + Huang-Krueger-Lacoste-Courville-2018-ICML-NAF + DeCao-Aziz-Titov-2020-UAI-BNAF + Durkan-Bekasov-Murray-Papamakarios-2019-NeurIPS-NeuralSplineFlow + Mueller-McWilliams-Rosca-Mohamed-Theis-2019-AAAI-PiecewiseLinearCoupling + Behrmann-Grathwohl-Chen-Duvenaud-Jacobsen-2019-ICML-iResNet/ResFlow + Chen-Rubanova-Bettencourt-Duvenaud-2018-NeurIPS-NeuralODE + Grathwohl-Chen-Bettencourt-Sutskever-Duvenaud-2019-ICLR-FFJORD + Hutchinson-1990-CommunStat-StochasticTraceEstimator + Kobyzev-Prince-Brubaker-2020-PAMI-NormalizingFlowSurvey + Papamakarios-Nalisnick-Rezende-Mohamed-Lakshminarayanan-2021-JMLR-22-NormalizingFlowsForProbabilisticInference (the canonical survey) — twelve named flow architectures + the Hutchinson-trace estimator + an iResNet/ResFlow Lipschitz-constrained branch + the continuous-time CNF/FFJORD ODE branch, all anchored on the change-of-variables identity `log p_X(x) = log p_Z(T⁻¹(x)) − log |det ∂T/∂z|` evaluated at `z = T⁻¹(x)`. Cross-package blockers: a **new bijector interface** `prob/flow.Bijector { Forward(z) → (x, logdet); Inverse(x) → (z, neg_logdet); }` (~50 LOC), `autodiff.Tape` reverse-mode (already in `autodiff/tape.go`, gates parameter learning of all 12 architectures), `autodiff.LogSumExp` (named in 011-T2, not built — gates IWAE-style multi-sample density evaluation but NOT vanilla flow training), `chaos/ode_solvers.go` adaptive RK45 (already shipping per 027 review — gates F11 CNF and F12 FFJORD ODE-form), the **18-fold pile-up** unified Box-Muller+Marsaglia `prob.RNGSampler` keystone (gates Gaussian base distribution), and `linalg.LU` for the **invertible-1×1-conv layer** in F5 Glow. Distinct from 239: 239 owns amortized-posterior IAF inside VI loops, 240 owns the *prior/density-estimation/sampling* surface that aicore consumers call directly.

**Summary line 2.** **Eighteen flow primitives F1-F18 totalling ~3,720 LOC of pure connective tissue** stand up the entire 2015-2024 normalizing-flow canon on existing v0.10.0 surfaces (`autodiff/tape.go` reverse-mode for parameter gradients, `linalg.MatMul` / `linalg.LU` / `linalg.Cholesky` for the linear algebra in coupling layers and 1×1 convs, `chaos/ode_solvers.go` RK4/RK45 for continuous flows, Box-Muller from `optim/genetic.go:58-65` for `z ~ N(0, I)` base sampling, `prob.NormalPDF/LogNormal` for base density evaluation), split across one new sub-package `prob/flow/` (~3,720 LOC source + ~1,860 LOC tests) organized as **(I) Bijector substrate** (F1 `Bijector` interface + F2 `Compose` + F3 `BatchNormFlow` + F18 `Permutation` + F17 `ActNorm` ~580 LOC); **(II) Element-wise + planar/radial flow zoo** (F4 PlanarFlow + F5 RadialFlow + F6 SylvesterFlow ~520 LOC); **(III) Coupling-flow zoo** (F7 RealNVP-AdditiveCoupling + F8 RealNVP-AffineCoupling + F9 NSF-RationalQuadraticSpline + F10 PiecewiseLinearCoupling ~880 LOC); **(IV) Autoregressive-flow zoo** (F11 MADE + F12 MAF + F13 NAF + F14 BNAF ~720 LOC); **(V) Glow + invertible-conv** (F15 Glow ~280 LOC); **(VI) Continuous-time + iResNet** (F16 NeuralODE + F17 FFJORD + F18 iResNet/ResFlow + F19 HutchinsonTraceEstimator ~740 LOC). Tier-1 keystone PR ≈ **F1 Bijector + F2 Compose + F4 Planar + F7 RealNVP-Affine + a 2D-target density-estimation pin** ≈ ~720 LOC — covers Rezende-Mohamed-2015 + Dinh-2017 in one shippable PR with a saturating R-MUTUAL pin (sample → re-evaluate density at samples ≈ original target density at 1e-6 over 10⁵ samples on the two-moons / pinwheel / checkerboard 2D toys that every flow paper uses for sanity). Cheapest one-day shippable: **F4 PlanarFlow at ~150 LOC** — closed-form `f(z) = z + u·tanh(wᵀz + b)`, `log|det ∂f/∂z| = log|1 + uᵀ ψ(z)|` with `ψ(z) = (1 − tanh²(wᵀz+b))·w` per Rezende-Mohamed-2015 §4.1, single tape-registered closure, saturates a 4/4 R-CHANGE-OF-VARIABLES pin (forward/inverse identity + log-det-jacobian × autodiff-Jacobian + sampled-then-re-evaluated density × analytic-density at 1e-9 on 2D Gaussian base). Highest-leverage one-week unlock: **F1 Bijector + F2 Compose + F7 RealNVP-AdditiveCoupling + F8 RealNVP-AffineCoupling = ~640 LOC** — the *single most-cited normalizing-flow architecture* (Dinh-2017 has >5,000 citations), with the property that **forward, inverse, and log-det-Jacobian are ALL O(D) per layer** (the singular efficiency property that distinguishes coupling flows from autoregressive flows), enabling tractable density evaluation AND tractable sampling in the same architecture, gated only by the F1 bijector interface and the existing `autodiff.Tape`. Singular cutting-edge piece: **F9 Neural-Spline-Flow rational-quadratic coupling (Durkan-Bekasov-Murray-Papamakarios-2019-NeurIPS, ~280 LOC)** — replaces RealNVP's affine coupling with a **monotonic rational-quadratic spline** with `K = 8` bins, parameterized by knot-positions/heights/derivatives outputted by the conditioning network; Durkan-2019 §3 closed-form forward + closed-form inverse via solving the per-bin quadratic, log-det-Jacobian as `Σ_i log f'(z_i)` summing per-element derivatives at the knot point. **NSF is the 2019-2024 SOTA flow architecture** — strictly more expressive than affine coupling (universal approximation per Durkan-2019 Theorem 1), efficient inversion, and does not require Lipschitz/iResNet machinery; reality is unusually well-positioned to ship NSF because the per-bin rational-quadratic interpolant is `O(K)` arithmetic — no autoregressive sequential pass, no Hutchinson trace estimator, no ODE solver. Singular reality competitive moat: **F17 FFJORD continuous flow (Grathwohl-Chen-Bettencourt-Sutskever-Duvenaud-2019-ICLR) at ~360 LOC** composing the existing `chaos/ode_solvers.go` RK45 adaptive solver + a new `HutchinsonTraceEstimator` (F19) + reverse-mode autodiff through the ODE adjoint per Chen-2018-NeurIPS — gives reality the **only zero-dep Go library** that ships the canonical 2018-2019 ODE-based generative-flow stack (Neural ODE → CNF → FFJORD) entirely on top of existing chaos/ ODE infrastructure that was originally built for Lorenz/VanDerPol; this is the textbook example of reality's "compose existing primitives" architectural moat — `chaos/` already knows how to integrate Lorenz to 1e-12 with adaptive RK45, FFJORD is just a different `f(x, t; θ)` with a Hutchinson-trace-augmented log-det-Jacobian state.

---

## 0. State of play (verified file-walk, 2026-05-08)

### `prob/flow/` package = does not exist (verified)

```
$ ls prob/
conformal  copula  distribution.go  distribution_test.go  distributions.go
distributions_test.go  golden_session38_test.go  hypothesis.go  jeffreys.go
markov.go  mathutil.go  nonparametric.go  prob.go  regression.go  testdata
timeseries.go  types.go
```

No `flow/` sub-package. Repo-wide grep on the canonical flow-literature surface returns ZERO callable matches:

| Surface | Canonical paper | Status |
|---|---|---|
| `Bijector` / `Bijection` interface | Tabak-Vanden-Eijnden-2010 / Dinh-Krueger-Bengio-2014-NICE | **ZERO** matches |
| `NormalizingFlow` / `Flow` (callable) | Rezende-Mohamed-2015-ICML | **ZERO** (only fluid/optim doc-comments) |
| `Planar` / `Radial` flow | Rezende-Mohamed-2015 §4.1, §4.2 | **ZERO** matches |
| `Sylvester` flow | vanderBerg-Hasenclever-Tomczak-Welling-2018-UAI | **ZERO** matches |
| `Coupling` layer / `RealNVP` | Dinh-Sohl-Dickstein-Bengio-2017-ICLR | **ZERO** matches |
| `Glow` / `InvertibleConv` / `ActNorm` | Kingma-Dhariwal-2018-NeurIPS | **ZERO** matches |
| `MADE` autoregressive mask | Germain-2015-ICML | **ZERO** matches |
| `MAF` MaskedAutoregressiveFlow | Papamakarios-Pavlakou-Murray-2017-NeurIPS | **ZERO** matches |
| `IAF` InverseAutoregressiveFlow | Kingma-Salimans-2016-NeurIPS | **ZERO** (239 V21 territory) |
| `NAF` NeuralAutoregressiveFlow | Huang-Krueger-Lacoste-Courville-2018-ICML | **ZERO** matches |
| `BNAF` BlockNAF | DeCao-Aziz-Titov-2020-UAI | **ZERO** matches |
| `NeuralSplineFlow` / RationalQuadraticSpline | Durkan-Bekasov-Murray-Papamakarios-2019 | **ZERO** matches |
| `iResNet` / `ResFlow` | Behrmann-Duvenaud-Jacobsen-2019-ICML | **ZERO** matches |
| `NeuralODE` / `CNF` / `FFJORD` | Chen-2018-NeurIPS / Grathwohl-2019-ICLR | **ZERO** matches |
| `Hutchinson` trace estimator | Hutchinson-1990-CommunStat | **ZERO** matches |

Zero current consumers. The **closest mathematical neighbor** in the existing repo is `prob/copula/` (which also implements change-of-variable transforms via copula CDFs/PDFs but via parametric copula families like Gaussian/Clayton, not via parametric neural-network bijectors). `chaos/ode_solvers.go` provides the substrate ODE integrators. `linalg.LU` provides invertible-1×1-conv inversion.

### Substrate audit

| Substrate | Path | Status for slot 240 |
|---|---|---|
| `autodiff.Tape` reverse-mode | `autodiff/tape.go:1-90` | PRESENT — gates F4/F5/F6/F7/F8/F9/F12/F13/F14 parameter learning (wrap each bijector layer's parameters as `Variable`s, register tape ops in Forward, `Tape.Backward(NLL)` for parameter gradients) |
| `autodiff.{Add,Mul,Exp,Log,Sqrt,Pow,Sin,Cos,Tanh}` | `autodiff/ops.go:1-141` | PRESENT — covers planar/radial/affine-coupling. **Missing:** `Sigmoid`, `Softplus`, `LeakyReLU`, `LogSumExp`, `Abs` (each ~15 LOC); `ReLU`, `Min`, `Max` (each ~20 LOC) — gates RealNVP's affine-coupling `s = exp(s_θ(z_passive))` (workable via `Exp` only) but `Softplus` parameterization is the documented stable alternative (Dinh-2017 §3.7 fn 11) |
| `autodiff.dual` forward-mode JVP | — | **ABSENT** (012-Tier-1) — strict block on **F19 Hutchinson** if a JVP path is wanted, but reverse-mode-only Hutchinson is feasible at extra cost (one autodiff sweep per random-projection direction × number of HVP directions); recommend pairing with 012-T1 for FFJORD efficiency |
| `linalg.MatMul` / `linalg.MatVec` | `linalg/matrix.go` | PRESENT — gates Sylvester / RealNVP / Glow / NSF coupling networks |
| `linalg.LU` decomposition + log-det | `linalg/decompose.go` | PRESENT — gates **F15 Glow** invertible-1×1-conv `W = PLU` parameterization (Kingma-Dhariwal-2018 §3.2) where log-det = `Σ log|U_ii|` |
| `linalg.Cholesky` decomposition | `linalg/decompose.go` | PRESENT — gates **F6 Sylvester** flow `Q-R = A` decomposition for ortho-householder (van-den-Berg-2018 §3.2) |
| `chaos/ode_solvers.go` RK4 / RK45 / DormandPrince | `chaos/ode_solvers.go` | PRESENT (per 027 chaos-missing review) — gates **F16 NeuralODE** / **F17 FFJORD** continuous flow |
| Box-Muller `z ~ N(0, I)` | `optim/genetic.go:58-65` | PRESENT — gates base-distribution sampling for **all 18** flow types |
| `prob.NormalPDF` / `prob.NormalLogPDF` | `prob/distributions.go` | PRESENT — gates base-density evaluation `log p_Z(z)` |
| `prob.RNGSampler` interface (18-fold pile-up) | — | **ABSENT** (Tier-0 keystone — gates uniform RNG for non-Gaussian base, but Box-Muller for Gaussian alone is workable) |

### Cross-package state: zero edges either direction

```
$ grep -r "github.com/davly/reality/chaos"   prob/   ; echo "---"
$ grep -r "github.com/davly/reality/linalg"  prob/   ; echo "---"
$ grep -r "github.com/davly/reality/autodiff" prob/  | grep -v copula  ; echo "---"
(no matches in any of the three)
```

`prob/copula/` already imports `autodiff/` (per 011-015 review), so the precedent is set. Slot 240 adds **three new edges**: `prob/flow/ → autodiff/`, `prob/flow/ → linalg/`, `prob/flow/ → chaos/` (for the ODE-based F16/F17 continuous flows) — all toward more-foundational packages, **no cycles**.

---

## 1. The conceptual unlock — change-of-variables on parametric bijectors

The single mathematical identity that anchors all 18 primitives:

> If `T: ℝ^d → ℝ^d` is a `C¹`-diffeomorphism and `Z` has density `p_Z`, then `X = T(Z)` has density `p_X(x) = p_Z(T⁻¹(x)) · |det ∂T⁻¹/∂x| = p_Z(T⁻¹(x)) / |det ∂T/∂z|_{z=T⁻¹(x)}`.

In log-form: `log p_X(x) = log p_Z(T⁻¹(x)) − log |det ∂T/∂z|`. This is the *only* identity any flow primitive uses.

Composability: if `T = T_K ∘ ... ∘ T_1`, then `log |det J_T(z)| = Σ_k log |det J_{T_k}(z_{k-1})|` (chain rule on log-dets). This means a flow is just a **sequence of bijector layers** where each layer must expose: (i) `Forward(z) → (x, logdet_layer)`, (ii) `Inverse(x) → (z, −logdet_layer)`, (iii) trainable parameters with autodiff gradients.

The 12 architectures differ only in **how they trade off** four properties:

| Architecture | Forward cost | Inverse cost | log-det cost | Expressiveness |
|---|---|---|---|---|
| F4 Planar / F5 Radial | O(D) | **iterative** (no closed-form) | O(D) | weak (1 hidden unit) |
| F6 Sylvester | O(MD) | iterative | O(M³) | medium |
| F7/F8 RealNVP | **O(D)** | **O(D)** | **O(D)** | medium-high |
| F12 MAF | O(D) per dim | **O(D²)** sequential | O(D) | high |
| F11 IAF (slot 239 V21) | **O(D²)** sequential | O(D) per dim | O(D) | high |
| F13 NAF / F14 BNAF | O(D²) | iterative | O(D) | very high (universal) |
| F15 Glow | O(D) | O(D) | O(D) | high (with depth) |
| F9 NSF-RQ-coupling | **O(KD)** | **O(KD)** | **O(KD)** | very high (universal, K bins) |
| F18 iResNet / ResFlow | O(D) | iterative (fixed-point) | **O(D · TraceEstimator)** | high (Lipschitz constraint) |
| F16 NeuralODE / F17 FFJORD | O(NFE) ODE | O(NFE) ODE | O(NFE · TraceEstimator) | universal |

**The architectural lesson** (Papamakarios-2021 survey §3): **coupling flows are the unique architecture that achieves O(D) in all four columns**, which is why RealNVP/Glow/NSF dominate practice for both density estimation AND sampling. Autoregressive flows (MAF/IAF) achieve O(D) on one of {sampling, density-evaluation} but O(D²) on the other. Continuous flows (FFJORD) achieve universal expressiveness at O(NFE) per evaluation. Reality's roster covers all five regimes.

---

## 2. Eighteen flow primitives (F1-F18, ~3,720 LOC pure glue)

Numbered ascending by composition-depth; each lists capability, composition of existing primitives, and LOC.

### Tier 0 — bijector substrate (gates everything)

**F1 Bijector interface** [~80 LOC]
The architectural keystone. Every flow layer implements:
```go
type Bijector interface {
    Forward(z []float64) (x []float64, logDet float64)
    Inverse(x []float64) (z []float64, negLogDet float64)
    Parameters() []float64               // flat parameter access for SGD outer
    SetParameters(params []float64)      //
}
```
Plus a `BijectorTape` variant exposing the same surface but operating on `*autodiff.Variable` for gradient computation. Document the **inverse-must-actually-invert** contract: `Inverse(Forward(z)) = z` to within 1e-8 (saturates an R-INVERTIBILITY-PIN 1/1 on every concrete bijector). **Cheapest substrate primitive in slot 240** and gates F2-F18.

**F2 Compose(b1, b2, ..., bK) Bijector** [~70 LOC]
Sequential composition of bijector layers. Forward is left-to-right `T_K ∘ ... ∘ T_1`, log-det is `Σ logdet_k`, Inverse is right-to-left `T_1⁻¹ ∘ ... ∘ T_K⁻¹` with negated log-dets. Single `slice` over the K layers with `for` loop. **Two-paragraph wrapper that converts the entire 12-architecture roster into a K-layer deep flow.**

**F3 Permutation(perm []int) Bijector** [~50 LOC]
Coordinate-permutation bijector — strictly bijective with `logdet = 0` (reorders dimensions). Critical between coupling layers in RealNVP (Dinh-2017 §3.5 alternates which half is "passive") and as the F15 Glow `Permute` layer. Two variants: random permutation (frozen at init) and reverse permutation (`[D-1, D-2, ..., 0]` per Glow paper). Trivial implementation but architecturally load-bearing.

**F4 PlanarFlow [~150 LOC]** — Rezende-Mohamed-2015-ICML §4.1
`f(z) = z + u · tanh(wᵀz + b)`, parameters `(w ∈ ℝ^D, u ∈ ℝ^D, b ∈ ℝ)`. Closed-form log-det:
`log|det ∂f/∂z| = log|1 + uᵀ ψ(z)|` where `ψ(z) = (1 − tanh²(wᵀz+b)) · w`.
Invertibility constraint: `uᵀ w ≥ −1`; enforced by re-parameterizing `û = u + (m(wᵀu) − wᵀu) · w / ‖w‖²` with `m(x) = −1 + softplus(x)` per Rezende-2015 Appendix A.1. **No closed-form inverse** (must be solved iteratively via Newton on `wᵀz`); document that planar is **density-estimation-only** unless paired with iterative root-finding.

**F5 RadialFlow [~140 LOC]** — Rezende-Mohamed-2015-ICML §4.2
`f(z) = z + β·h(α, r)·(z − z₀)` with `r = ‖z − z₀‖`, `h(α, r) = 1/(α + r)`, parameters `(z₀ ∈ ℝ^D, α > 0, β ∈ ℝ)`. Closed-form log-det:
`log|det ∂f/∂z| = (D−1) log(1 + β·h(α,r)) + log|1 + β·h(α,r) + β·h'(α,r)·r|`.
Invertibility constraint `β ≥ −α` enforced by softplus reparameterization. Radial complements planar: planar is hyperplane-aligned, radial is sphere-aligned; together they universally approximate via Rezende-2015 Theorem 1.

**F6 SylvesterFlow [~230 LOC]** — vanderBerg-Hasenclever-Tomczak-Welling-2018-UAI
Generalizes planar to a **rank-M update**: `f(z) = z + A · h(B z + b)` with `A ∈ ℝ^{D×M}`, `B ∈ ℝ^{M×D}`. Log-det = `log |det(I_M + diag(h'(B z + b)) · B · A)|` (Sylvester's determinant identity reduces D×D to M×M). Three variants per van-den-Berg-2018 §3.2: **orthogonal Sylvester** (Q from QR with R upper-triangular trainable), **householder Sylvester** (M Householder reflections), **triangular Sylvester** (R₁, R₂ triangular). Composes `linalg.QR` + `autodiff.Tape`. Inverse iterative.

### Tier 1 — coupling-flow zoo (the 2017-2019 SOTA)

**F7 RealNVP-AdditiveCoupling [~180 LOC]** — Dinh-Krueger-Bengio-2014-NICE / Dinh-2017-ICLR §3.3
Split `z = (z_a, z_b)` along a fixed mask. `x_a = z_a`; `x_b = z_b + t_θ(z_a)` where `t_θ` is an arbitrary neural network (caller-supplied closure). Inverse: `z_b = x_b − t_θ(x_a)`. Log-det = **0** (volume-preserving). Composes:
- A user-supplied `Net interface { Forward([]float64) []float64 }` — caller plugs in any feedforward NN; reality stays a math library.
- Mask-based split via boolean array.
**The simplest coupling-flow primitive** — additive coupling is volume-preserving so it cannot model density-mass redistribution alone, but it stacks with affine coupling in NICE-style architectures.

**F8 RealNVP-AffineCoupling [~200 LOC]** — Dinh-Sohl-Dickstein-Bengio-2017-ICLR §3.5
Generalization of F7 with multiplicative scale: `x_b = z_b ⊙ exp(s_θ(z_a)) + t_θ(z_a)`. Log-det = `Σ s_θ(z_a)_i` (sum over masked dims). Inverse: `z_b = (x_b − t_θ(x_a)) ⊙ exp(−s_θ(x_a))`. **The single most-cited normalizing-flow architecture** (>5,000 citations on Dinh-2017). Document the `s_θ` parameterization hazard (Dinh-2017 §3.7 fn 11): **clip exp argument to ±5** to prevent exp-overflow during training; alternative is `exp(s_θ)` → `softplus(s_θ + 2)` per Glow. Saturates an R-EXACT-INVERTIBILITY 4/4 pin: forward × inverse identity at 1e-8, sample × re-evaluate density at 1e-6, log-det × autodiff-Jacobian at 1e-6, total-NLL × analytic-CDF on a Gaussian-target at 1e-7.

**F9 NeuralSplineFlow-RationalQuadraticCoupling [~280 LOC]** — Durkan-Bekasov-Murray-Papamakarios-2019-NeurIPS
**Singular cutting-edge piece of slot 240.** Replaces affine coupling (F8) with a **K-bin monotonic rational-quadratic spline**: instead of `x_b_i = z_b_i · exp(s_i) + t_i`, transform each coordinate `x_b_i = RQS(z_b_i; θ_i)` where `θ_i = (knot_x_1, ..., knot_x_K, knot_y_1, ..., knot_y_K, knot_d_1, ..., knot_d_{K+1})` is the per-coordinate spline parameterization output by the conditioning network. **Closed-form forward** (per-bin rational quadratic, Durkan-2019 eqn 4); **closed-form inverse** (solve per-bin quadratic, Durkan-2019 eqn 6, gives single real root in [0,1]); log-det = `Σ_i log f'(z_i)`. Universal approximation per Durkan-2019 Theorem 1. Two parameterization knobs: K (default 8), tail behavior (linear extension outside [−B, B]). **No autoregressive sequential pass + no Hutchinson trace** = strictly more efficient than NAF/BNAF/iResNet/FFJORD while matching expressiveness. **The 2019-2024 SOTA flow architecture for tabular density estimation.**

**F10 PiecewiseLinearCoupling [~110 LOC]** — Mueller-McWilliams-Rosca-Mohamed-Theis-2019-AAAI / Hoogeboom-2019-ICLR
Simpler precursor to F9: K-bin piecewise-linear monotonic CDF. Closed-form forward and inverse; log-det = `Σ log slope_at_z_i`. Document as the cheaper-but-less-smooth alternative to F9 NSF-RQ (no second-derivative continuity).

### Tier 2 — autoregressive-flow zoo

**F11 MADE [~150 LOC]** — Germain-Gregor-Mitchell-Larochelle-2015-ICML
Masked Autoencoder for Distribution Estimation: a feedforward network with binary weight masks that enforce the autoregressive property `output_i depends only on input_1, ..., input_{i-1}`. Caller-supplied masking matrices. Substrate for F12/F13. Pure linear-algebra primitive: `mask ⊙ W` element-wise multiply on the dense weight matrices. Reality ships the masking utility; aicore plugs in the network.

**F12 MAF MaskedAutoregressiveFlow [~220 LOC]** — Papamakarios-Pavlakou-Murray-2017-NeurIPS
`x_i = z_i · exp(α_i(x_{<i})) + μ_i(x_{<i})` with autoregressive `α_i, μ_i` from a MADE network. **Forward (sampling) is sequential O(D²)** but **inverse (density evaluation) is parallel O(D)** — the *opposite* asymmetry of F11/F13/F14 (slot 239 V21) IAF. Log-det = `Σ α_i`. Composes F11 MADE + autodiff. **Density-estimation-optimized flow** — fast `log p(x)` for likelihood training, slow sampling.

**F13 NAF NeuralAutoregressiveFlow [~190 LOC]** — Huang-Krueger-Lacoste-Courville-2018-ICML
Generalizes MAF to **monotonic-MLP-as-coupling**: instead of affine `x_i = z_i · exp(α_i) + μ_i`, use a per-coordinate monotonic MLP (sigmoid + positive weights) `x_i = MLP_θ_i(z_i; ψ(x_{<i}))` where `ψ` produces the per-element monotonic-MLP weights. Universal approximation per Huang-2018 Theorem 1. Log-det = `Σ log MLP'_θ_i(z_i)` summing per-element derivatives. **Inverse iterative** (Newton on the monotonic MLP).

**F14 BNAF BlockNAF [~160 LOC]** — DeCao-Aziz-Titov-2020-UAI
Block-diagonal generalization of F13 — replaces the per-coordinate MLP with a single block-autoregressive MLP that exploits the autoregressive masking on the **weight matrix** rather than running D separate MLPs. Substantial speedup over NAF (single forward pass instead of D sequential), same universal approximation. Cleaner implementation than NAF — preferred in 2020+ practice.

### Tier 3 — Glow + invertible-1×1-conv

**F15 Glow [~280 LOC]** — Kingma-Dhariwal-2018-NeurIPS
Three-block composition: **(i) ActNorm** (data-dependent scale + bias initialization, then trainable; F17 in Tier 4 below as substrate); **(ii) Invertible-1×1-Conv** (`W ∈ ℝ^{D×D}` parameterized as `W = PLU` via `linalg.LU` with `P` permutation, `L` unit-lower-triangular, `U` upper-triangular; log-det = `Σ log |U_ii|`); **(iii) Affine coupling** (F8). Stack K such triples, optionally with multi-scale architecture (squeeze + split). **The crown-jewel image-generative-flow architecture** (>3,000 citations, Glow generated photo-realistic faces in 2018 before diffusion). Composes F2 Compose + F3 Permutation + F8 AffineCoupling + new ActNorm + new InvertibleConv. **Strict block on `linalg.LU` log-det helper** (~30 LOC if not already in `linalg/decompose.go`).

### Tier 4 — Glow substrate

**F16 ActNorm [~80 LOC]** — Kingma-Dhariwal-2018-NeurIPS §3.1
Data-dependent affine normalization: `x = (z − μ) / σ` where `μ, σ` are initialized from the first mini-batch's mean/std then **trained as free parameters thereafter**. Log-det = `−Σ log σ_i`. Substantially more stable than BatchNorm-as-flow because no batch statistics at inference. The "learnable batch-norm-replacement" of Glow.

**F17 BatchNormFlow [~110 LOC]** — Dinh-Sohl-Dickstein-Bengio-2017-ICLR §3.7
The original RealNVP-paper BatchNorm-as-flow (precursor to ActNorm): `x = (z − μ_batch) / σ_batch` with running stats at inference. Log-det = `−Σ log σ_batch_i`. Document the train-vs-inference batch-statistics mismatch hazard (Dinh-2017 §3.7) — most consumers should use F16 ActNorm instead.

### Tier 5 — continuous-time flows (Neural ODE family)

**F18 NeuralODE [~200 LOC]** — Chen-Rubanova-Bettencourt-Duvenaud-2018-NeurIPS
Continuous-depth flow: `dz/dt = f_θ(z, t)` integrated from `t=0 → t=T` via `chaos/ode_solvers.go` adaptive RK45. Forward: solve ODE forward. Inverse: solve ODE backward (negate `f_θ`). For **density estimation**, must also track `d log p / dt = − tr(∂f/∂z)` (instantaneous-change-of-variable identity, Chen-2018 §4); **strict block on F19 Hutchinson** for `D ≥ 5` because exact trace is O(D²). Composes:
- Existing `chaos/ode_solvers.RK45Adaptive` (per chaos doc.go).
- New `HutchinsonTraceEstimator` (F19).
- `autodiff.Tape` for parameter gradients via the adjoint method (Chen-2018 §3, ~80 LOC adjoint solver). **Does NOT need autodiff/dual** for the adjoint method — only reverse-mode tape.
**The textbook example of reality's "compose existing primitives" architecture**: `chaos/` already integrates Lorenz/VanDerPol to 1e-12, so NeuralODE is a different `f(x, t; θ)` plus a Hutchinson-augmented log-density state.

**F19 FFJORD [~160 LOC]** — Grathwohl-Chen-Bettencourt-Sutskever-Duvenaud-2019-ICLR
"Free-Form Jacobian Of Reversible Dynamics". Identical to F18 NeuralODE except the conditioning network `f_θ` is **completely unrestricted** (no triangular/coupling/Lipschitz constraint) — possible because Hutchinson-trace makes log-det computable in O(D · K_HVP) regardless of structure. Three-line modification of F18: replace `tr(∂f/∂z)` with `E_{ε ~ N(0,I)}[ ε^T (∂f/∂z) ε ]` per F19 stochastic estimator. Document the **K_HVP samples per ODE step** trade-off: K=1 is the FFJORD default, but variance scales as 1/K. **Singular reality competitive moat** when paired with `chaos/` substrate.

**F20 HutchinsonTraceEstimator [~70 LOC]** — Hutchinson-1990-CommunStat-19
Stochastic trace estimator: for `A ∈ ℝ^{D×D}`, `tr(A) = E_{ε ~ p(ε)}[ε^T A ε]` for `p(ε)` zero-mean unit-variance (Rademacher `±1` or Gaussian `N(0, I)`). Composes one Hessian-vector-product per draw via reverse-mode `autodiff.Tape` (or two if forward-mode is unavailable; per 012-T1, dual-numbers gives single-pass HVP). Variance per draw `O(D / K)`; default `K=1` for training, `K ≥ 100` for accuracy at evaluation. **Standalone primitive** consumable by F18/F19 and any future Hessian-trace consumer (e.g., Laplace approximation in slot 169).

### Tier 6 — invertible-residual / Lipschitz-constrained

**F21 iResNet [~180 LOC]** — Behrmann-Grathwohl-Chen-Duvenaud-Jacobsen-2019-ICML
Invertible residual network: `x = z + g_θ(z)` where `g_θ` is constrained to be **Lipschitz with constant L < 1** (spectral-normalized weights). Forward is single pass; inverse via fixed-point iteration `z_{k+1} = x − g_θ(z_k)` converging at rate `L^k`. Log-det = `tr(log(I + ∂g/∂z))` evaluated via **truncated power-series + Hutchinson** (Behrmann-2019 §4). Composes F19 + spectral normalization (`linalg.PowerIteration` for largest singular value, ~40 LOC).

**F22 ResFlow [~130 LOC]** — Chen-Behrmann-Duvenaud-Jacobsen-2019-NeurIPS
Variance-reduced unbiased extension of F21: replaces the truncated power-series log-det estimator with a **Russian-roulette unbiased estimator**, gives unbiased log-det estimates without bias from truncation. Trivial three-paragraph extension of F21 once Hutchinson + Lipschitz machinery is in place.

---

## 3. Composition graph (DAG)

```
F1  Bijector (interface)        ─┐
F2  Compose                      │
F3  Permutation                  ├── substrate (gates everything)
F16 ActNorm  / F17 BatchNormFlow │
                                 ┘
F4  PlanarFlow ──── F6 SylvesterFlow (rank-M generalization)
F5  RadialFlow

F11 MADE (autoregressive mask) ──── F12 MAF ── F13 NAF ── F14 BNAF
                                       │ (also: F15 Glow uses MADE-style mask in coupling network)
                                       │
F7  RealNVP-Additive  ── F8  RealNVP-Affine  ── F15 Glow (= ActNorm + InvConv + AffineCoup × K)
                                  │                  │
                                  └── F9 NSF-RationalQuadraticSpline (replace affine with K-bin spline)
                                  └── F10 PiecewiseLinearCoupling (cheaper variant)

chaos/ode_solvers.RK45 ──── F18 NeuralODE ── F19 FFJORD (+ F20 Hutchinson)
                                                  │
F20 HutchinsonTraceEstimator ─────────────────────┴── F21 iResNet ── F22 ResFlow

Cross-link to slot 239:
    F1 Bijector + F11 MADE + F12 MAF mask  ──→  239-V21 IAF (amortized-posterior IAF)
    F1-F22 entire roster                   ──→  239-V23 FlowVI-glue
```

---

## 4. Saturation pins this slot unlocks

- **R-INVERTIBILITY-PIN 12/12 (F4-F15):** `Inverse(Forward(z)) = z` to within 1e-8 on every concrete bijector, validated on 10⁴ random `z ~ N(0, I)` samples. The canonical R-MUTUAL pin specific to flows.
- **R-LOGDET-VS-AUTODIFF-JACOBIAN 12/12 (F4-F15):** the analytic `log |det J|` returned by `Forward` agrees with `log |det autodiff.Jacobian(forward)(z)|` (computed by D autodiff sweeps) at 1e-9 for `D ≤ 8`. Saturates the closed-form-vs-numerical-Jacobian R-MUTUAL idiom.
- **R-CHANGE-OF-VARIABLES 4/4 (F4 + F8):** sample `z ~ p_Z`, push to `x = T(z)`, evaluate `log p_X(x) = log p_Z(T⁻¹(x)) − log|det J_T|`, take `Σ log p_X(x_i) / N` → matches differential entropy `−E[log p_X(X)]` per Monte-Carlo at 1e-3 (`N = 10⁶`).
- **R-TWO-MOONS-DENSITY-ESTIMATION 1/1 (F8 + F9):** train RealNVP-affine and NSF-RQ on the canonical 2D two-moons / pinwheel / checkerboard targets; pin **NSF NLL ≤ RealNVP NLL** by ≥0.05 nats per Durkan-2019 Table 1.
- **R-FFJORD-MATCHES-EXACT-LOGDET 2/2 (F19 + F20):** with K_HVP = 100 Hutchinson samples, `FFJORD log-det estimate` matches `exact tr(∂f/∂z)` (computed via D autodiff sweeps) at 1e-2 relative tolerance on 5D toy. Variance scales as `1/K`.
- **R-COUPLING-VS-AUTOREGRESSIVE-EQUIVALENCE 1/1 (F8 + F12):** an affine-coupling RealNVP with permutation between layers and a depth-K MAF can both fit a Gaussian-target to 1e-3 NLL, but **forward-pass cost ratio** is 1 : O(D) — pin the asymmetry quantitatively per Papamakarios-2017 §4.
- **R-PLANAR-RANK-M-LIMIT 1/1 (F4 + F6):** F6 SylvesterFlow with M=1 reduces *exactly* to F4 PlanarFlow; pin the equivalence at 1e-12 over 10³ random parameter draws.
- **R-NEURAL-ODE-LIMIT 1/1 (F18):** as the discrete-depth coupling-flow `K → ∞` with `T_k(z) = z + Δt · g_θ(z, k·Δt)`, the discrete flow converges to F18 NeuralODE — pin at 1e-2 NLL on 2D toy.

---

## 5. Connective-tissue LOC budget

| ID | Capability | LOC | Tier | Blocks-on |
|----|-----------|-----|------|-----------|
| F1 | Bijector interface | 80 | 0 | — |
| F2 | Compose | 70 | 0 | F1 |
| F3 | Permutation | 50 | 0 | F1 |
| F4 | PlanarFlow | 150 | 1 | F1, autodiff.Tape |
| F5 | RadialFlow | 140 | 1 | F1, autodiff.Tape |
| F6 | SylvesterFlow | 230 | 1 | F1, linalg.QR |
| F7 | RealNVP-AdditiveCoupling | 180 | 1 | F1, caller `Net` interface |
| F8 | RealNVP-AffineCoupling | 200 | 1 | F1, F7 mask substrate |
| F9 | NSF-RationalQuadraticSpline | 280 | 2 | F1, F8 mask substrate |
| F10 | PiecewiseLinearCoupling | 110 | 2 | F1, F9 |
| F11 | MADE autoregressive mask | 150 | 1 | linalg.MatMul |
| F12 | MAF MaskedAutoregressiveFlow | 220 | 2 | F1, F11 |
| F13 | NAF NeuralAutoregressiveFlow | 190 | 2 | F1, F11 |
| F14 | BNAF BlockNeuralAutoregressiveFlow | 160 | 2 | F11, F13 |
| F15 | Glow (ActNorm + InvConv + AffineCoup × K) | 280 | 3 | F2, F3, F8, F16, linalg.LU |
| F16 | ActNorm | 80 | 0 | F1 |
| F17 | BatchNormFlow (RealNVP variant) | 110 | 0 | F1 |
| F18 | NeuralODE | 200 | 4 | F1, chaos.RK45, F20 |
| F19 | FFJORD | 160 | 4 | F18, F20 |
| F20 | HutchinsonTraceEstimator | 70 | 4 | autodiff.Tape (or dual for JVP path) |
| F21 | iResNet | 180 | 4 | F1, F20, linalg.PowerIteration (~40 LOC new) |
| F22 | ResFlow | 130 | 4 | F21 |
| **Σ** | | **~3,720** | | |

(Excluding 239-V21 IAF deduplication: V21 IAF lives in 239 because it is the amortized-posterior flow used inside VI; F12 MAF lives in 240 because it is the prior-side density-estimation flow — same MADE substrate, opposite autoregressive direction. Net new LOC for slot 240 ≈ 3,720 since IAF is owned by 239.)

Pure-glue ratio: ~80% composition over `autodiff.Tape` reverse-mode + `linalg.{MatMul, LU, QR, Cholesky}` + `chaos/ode_solvers.RK45` + Box-Muller from `genetic.go:58-65` + `prob.NormalLogPDF`. ~20% genuinely-new math (F9 NSF rational-quadratic per-bin closed-form forward + closed-form quadratic inverse + log-det at ~120 LOC; F18-F22 ODE-and-Lipschitz machinery at ~300 LOC including the spectral-norm power-iteration helper; F11 MADE binary-mask weight construction at ~80 LOC).

---

## 6. Recommended PR sequence

**PR-1: substrate (F1 Bijector + F2 Compose + F3 Permutation + F4 PlanarFlow + F5 RadialFlow + F16 ActNorm) — ~570 LOC source, ~290 LOC tests, 1.5 days**
First flow surface in `reality`. Lands the entire bijector-substrate + Rezende-Mohamed-2015 element-wise zoo. **Saturates R-INVERTIBILITY-PIN 4/4 + R-LOGDET-VS-AUTODIFF-JACOBIAN 4/4 immediately** on PlanarFlow / RadialFlow / Compose / Permutation. The first PR can be reviewed without any neural-network machinery — purely a math API.

**PR-2: F7 RealNVP-Additive + F8 RealNVP-Affine + F17 BatchNormFlow + a 2D-density-estimation pin — ~570 LOC source, ~290 LOC tests, 2 days**
**The crown-jewel coupling-flow PR.** Lands Dinh-2017's >5,000-citation architecture. Caller supplies the `Net` closure (no NN type in reality). Saturates R-CHANGE-OF-VARIABLES 4/4 and R-TWO-MOONS-DENSITY-ESTIMATION 1/1 on the canonical pinwheel / two-moons / checkerboard 2D toys.

**PR-3: F9 NSF-RationalQuadraticSpline + F10 PiecewiseLinearCoupling — ~390 LOC source, ~210 LOC tests, 2 days**
**Singular cutting-edge piece of slot 240.** Durkan-2019-NeurIPS Neural-Spline-Flow lands the 2019-2024 SOTA tabular-density flow on top of PR-2's mask substrate. Saturates R-NSF-VS-REALNVP-NLL 1/1 by ≥0.05 nats per Durkan-2019 Table 1.

**PR-4: F11 MADE + F12 MAF + F13 NAF + F14 BNAF — ~720 LOC source, ~360 LOC tests, 3 days**
The autoregressive-flow zoo. F11 MADE substrate is shared with 239-V21 IAF (cross-link). F12-F14 are the Papamakarios / Huang / DeCao trio that completes the autoregressive-flow canon. Saturates R-COUPLING-VS-AUTOREGRESSIVE-EQUIVALENCE 1/1 against PR-2.

**PR-5: F6 SylvesterFlow + F15 Glow (+ InvertibleConv via linalg.LU) — ~510 LOC source, ~260 LOC tests, 3 days**
The image-generative-flow PR. Glow is the highest-impact single primitive after F8 (>3,000 citations). Strict block on linalg.LU exposing the upper-triangular `U` for log-det via `Σ log |U_ii|` (~30 LOC extension if not already present).

**PR-6: F20 HutchinsonTraceEstimator + F18 NeuralODE + F19 FFJORD — ~430 LOC source, ~220 LOC tests, 3 days**
**Singular reality competitive moat.** Composes the existing `chaos/ode_solvers.RK45Adaptive` substrate with the Hutchinson trace estimator and the Chen-2018 adjoint backprop — gives reality the only zero-dep Go library shipping the Neural-ODE → CNF → FFJORD continuous-flow stack on top of the existing chaos/ ODE infrastructure. Saturates R-FFJORD-MATCHES-EXACT-LOGDET 2/2 and R-NEURAL-ODE-LIMIT 1/1.

**PR-7: F21 iResNet + F22 ResFlow (+ linalg.PowerIteration spectral-norm helper) — ~310 LOC source, ~180 LOC tests, 2 days**
The Lipschitz-constrained-residual flow PR. Behrmann-2019 + Chen-Behrmann-2019. Low priority vs PR-1 to PR-6 — F8 RealNVP and F9 NSF cover most density-estimation use cases more efficiently.

Total: ~3,720 LOC source + ~1,860 LOC tests across 7 PRs over ~16 engineer-days. PR-1 is the 1.5-day standalone; PR-2 (RealNVP) is the single highest-impact PR for downstream consumers; PR-3 (NSF-RQ) is the singular cutting-edge moat; PR-6 (FFJORD on chaos/ substrate) is the crown-jewel reality-architectural-moat composition.

---

## 7. Cycle-hazard analysis

Proposed import directions:

```
prob/flow/      ──→  prob/         (NormalLogPDF, RNGSampler keystone)
prob/flow/      ──→  autodiff/     (Tape, ops, vector — same precedent as prob/copula/)
prob/flow/      ──→  linalg/       (MatMul, LU, QR, Cholesky, PowerIteration)
prob/flow/      ──→  chaos/        (ode_solvers.RK45Adaptive — F18/F19 only)
prob/flow/      ──→  optim/        (GradientDescent, Adam — F2-F22 outer loop)
```

**Five cross-package edges**, four of which already have precedent (`prob → autodiff` from copula, `prob → linalg` would be new but `prob → optim` existing through `prob/regression.go`, `prob → chaos` is the only genuinely-new edge but trivially safe — chaos is more foundational than prob in any sensible ordering).

**No cycles.** `chaos/`, `linalg/`, `autodiff/`, `optim/` do not need to import `prob/flow/`. The conditioning-network closure is consumer-side (no aicore import from reality).

---

## 8. Precision hazards documented

Per CLAUDE.md "Precision documented, not assumed":

- **F4 PlanarFlow:** `m(x) = −1 + softplus(x)` reparameterization for `uᵀw ≥ −1`; document the boundary `softplus(0) = ln(2) ≈ 0.693` so `m(0) ≈ −0.307`, not exactly −1 — invertibility is **strict** when caller initializes near the boundary.
- **F5 RadialFlow:** `α > 0` enforced via `α = softplus(α_raw)`, and `β ≥ −α` via `β = −α + softplus(β_raw)`.
- **F8 RealNVP-Affine:** clip exp argument to `[−5, 5]` per Dinh-2017 §3.7 fn 11; alternative is `s_θ(z) → 2 · tanh(s_raw)` per Glow practice. **Without clipping, training diverges** within 100 steps on most tabular targets.
- **F9 NSF-RQ:** monotonicity via `softplus(d_raw) + ε` for knot-derivatives with `ε = 1e-3`; tail behavior linear extension outside `[−B, B]` with `B = 3` default per Durkan-2019 §3.4.
- **F11-F14 autoregressive:** mask must be **strictly lower-triangular for the dependency order** — diagonal of zero or non-strict masks **break invertibility**. Validate at construction.
- **F15 Glow:** `W = PLU` where `P` is a fixed-at-init random permutation, `L` is unit-lower-triangular trainable, `U` is upper-triangular trainable; log-det = `Σ log |U_ii|`. **Initialize `U = I`** (so `W = I` initially) — random init breaks training.
- **F18 NeuralODE:** RK45 adaptive tolerance must be at most 1e-5 for stable density-estimation training; default `chaos.RK45Adaptive` uses 1e-8, so the default is safe but slow. Document the `(rtol, atol) = (1e-5, 1e-7)` training default vs `(1e-8, 1e-10)` evaluation default.
- **F19 FFJORD:** Hutchinson `K = 1` trains stably but evaluates `log p(x)` to `O(1/√K)` MC error; recommend K ≥ 100 at inference-time NLL evaluation.
- **F20 Hutchinson:** Rademacher `ε ∈ {±1}^D` has `Var(ε^T A ε) = 2 ‖A‖_F² − 2 Σ A_ii²` strictly less than Gaussian's `Var = 2 ‖A‖_F²` per Hutchinson-1990 — prefer Rademacher in practice (single-bit RNG, lower variance for diagonally-dominant matrices).
- **F21 iResNet:** Lipschitz constant `L < 1` enforced by **spectral normalization** of all dense weight matrices (`linalg.PowerIteration` for largest singular value, then divide by `σ_1 · (L⁻¹ − ε)`). Power iteration converges in ~10 iterations for typical NN weight matrices; document the `K=10` default + warm-start across training steps.

---

## 9. Distinct from prior agents (provenance)

- **011-015 autodiff** — 012-T1 names dual-numbers / forward-mode JVP; slot 240 F20 Hutchinson can use either forward-mode (single HVP per draw) or reverse-mode (one Jacobian sweep per draw + ε projection). Forward-mode is strictly more efficient for HVP — co-ship 012-T1 for FFJORD efficiency.
- **026-030 chaos** — 027-T6 already lists adaptive RK45 / DormandPrince in the chaos roster; slot 240 F18/F19 are the **first ML consumer** of chaos's ODE substrate (previously only Lorenz/VanDerPol consumers).
- **097 linalg-missing** — names `linalg.PowerIteration` (~40 LOC) for largest-singular-value computation; F21 iResNet is the consumer.
- **117-120 prob** — 118 prob-sota calls for amortized inference; slot 240 ships the prior-side flows that 239 + aicore consume.
- **168 synergy-physics-autodiff** — 168-A6 names `NeuralODE` adjoint-method as a synergy bullet but does not own the bijector / log-det / sampling surface. Slot 240 owns the parametric-bijector surface; 168 owns the physics-symmetry-preserving variant (Hamiltonian-NeuralODE, Lagrangian-NeuralNet). Co-ship F18 with 168-A6.
- **169 synergy-prob-optim** — 169-S15/S16 own deterministic-fit BBVI/ADVI; slot 240 ships flows used as *q* inside that VI machinery via 239-V23 FlowVI-glue.
- **195 synergy-optim-prob** — orthogonal axis (CMA-ES / SDE-as-sampler / stochastic optim).
- **220 new-stochastic-opt** — 240 consumes 220-F1-F4 SGD/MiniBatch as outer loop and 220-F8 Adam as the canonical flow optimizer.
- **228 new-bayes-nonparam** — orthogonal (DP/HDP/IBP) but cross-link via 239-V14 StickBreakingVAE + flow-encoder; reality could ship a flow-stick-breaking-VAE composition.
- **236 new-rkhs** — K22 KSD as a non-flow generative-fit alternative; cross-link only.
- **237 new-gaussian-process** — orthogonal (G24 SVGP could use flow-VI from 239-V23, which uses 240's `Bijector` surface).
- **238 new-mcmc** — pin via 240 + 238-M22 nested-sampling: flow-trained `q(z)` as proposal in NUTS / HMC for amortized-MCMC (Hoffman-2019-NeurIPS). Cross-link only.
- **239 new-svi (CLOSEST SIBLING)** — 239 owns the **amortized-posterior IAF + flow-VI glue** (V21/V22/V23) inside VI loops; slot 240 owns the **prior-side density-estimation flow zoo** (F4-F22). 239's V23 FlowVI-glue is **the** consumer of 240's F1 Bijector interface. Disjoint rosters; together ship the full Rezende-2015 → Dinh-2017 → Kingma-2016 → Glow-2018 → Chen-2018 → Grathwohl-2019 → Durkan-2019 → DeCao-2020 flow canon. **Strict shared dependency on slot 240's F1 Bijector interface from slot 239's V23.**
- **241 new-diffusion-models** — 241 is the **non-bijective** generative axis (score-matching not change-of-variables). Reality should ship both: flows (240) for tractable NLL + invertibility, diffusion (241) for samples-only without invertibility. Cross-link via the shared `prob/RNGSampler` keystone and the shared continuous-time-ODE substrate (`chaos/ode_solvers.RK45`); 241's probability-flow ODE is structurally the same as 240's F18 NeuralODE but trained via score-matching, not maximum likelihood + change-of-variables.

Slot 240 is the **static-flow / change-of-variables / bijector-zoo slot** — every primitive is what 2015-2024 deep-generative-modeling literature calls by these specific names (Planar/Radial/Sylvester/RealNVP/Glow/MAF/NAF/BNAF/NeuralSpline/iResNet/NeuralODE/FFJORD).

---

## 10. Bottom line

`reality/prob/` ships **ZERO** normalizing-flow / bijector / change-of-variables / RealNVP / Glow / MAF / Neural-Spline / continuous-flow / FFJORD / iResNet surface despite being the obvious target package for the entire 2015-2024 normalizing-flow canon. **Twenty-two primitives F1-F22 (numbered through 22 with F19/F20 sharing the Hutchinson cluster) totalling ~3,720 LOC of pure connective tissue** stand up the entire static-flow zoo + autoregressive-flow + Glow + neural-spline + continuous-time + Lipschitz-constrained pipeline on existing v0.10.0 surfaces (`autodiff.Tape` reverse-mode, `linalg.{MatMul, LU, QR, Cholesky}`, `chaos/ode_solvers.RK45Adaptive`, Box-Muller from `genetic.go:58-65`, `prob.NormalLogPDF`).

**Cheapest one-day shippable**: F4 PlanarFlow at ~150 LOC saturates a 4/4 R-CHANGE-OF-VARIABLES pin against Box-Muller-sampled `z ~ N(0, I)` immediately. **Cheapest 1.5-day standalone PR**: PR-1 substrate (F1 Bijector + F2 Compose + F3 Permutation + F4 Planar + F5 Radial + F16 ActNorm = ~570 LOC) lands the **first flow surface in reality**. **Single highest-impact PR for downstream consumers**: PR-2 F7+F8 RealNVP at ~570 LOC delivers the >5,000-citation Dinh-2017 architecture — the canonical density-estimation + sampling flow that any aicore consumer needs. **Singular cutting-edge moat**: PR-3 F9 Neural-Spline-Flow rational-quadratic coupling (Durkan-2019) at ~280 LOC is the 2019-2024 SOTA tabular-density flow architecture, strictly more expressive than affine coupling with closed-form forward + closed-form inverse via a per-bin quadratic. **Crown-jewel reality-architectural-moat composition**: PR-6 F18 NeuralODE + F19 FFJORD + F20 HutchinsonTraceEstimator at ~430 LOC composes the existing `chaos/ode_solvers.RK45Adaptive` substrate (originally built for Lorenz/VanDerPol) with the Chen-2018 adjoint method and Hutchinson trace — gives reality the **only zero-dep Go library** shipping the canonical 2018-2019 ODE-based generative-flow stack entirely on top of pre-existing chaos/ ODE infrastructure.

The single most important conceptual identity slot 240 pins: **change-of-variables `log p_X(x) = log p_Z(T⁻¹(x)) − log |det ∂T/∂z|`** is the foundational identity for all 22 primitives, and the 12 architectures (F4-F15) differ only in **how they trade off** four properties: forward cost, inverse cost, log-det cost, and expressiveness. The architectural lesson per Papamakarios-Nalisnick-Rezende-Mohamed-Lakshminarayanan-2021-JMLR-22 survey: **coupling flows are the unique architecture that achieves O(D) on all four** (which is why RealNVP/Glow/NSF dominate practice), while autoregressive flows trade O(D)-density for O(D²)-sampling (or vice-versa), and continuous flows achieve universal expressiveness at O(NFE) per evaluation. Reality's roster covers all five regimes: planar/radial/Sylvester (rank-M), coupling (RealNVP/Glow/NSF), autoregressive (MAF/NAF/BNAF), continuous (NeuralODE/FFJORD), and Lipschitz-residual (iResNet/ResFlow).

**Reality is unusually well-positioned for slot 240 because (i) `autodiff.Tape` reverse-mode already handles all parameter gradients through the forward-direction bijector log-det, (ii) `linalg.LU` already provides the Glow invertible-1×1-conv log-det via `Σ log |U_ii|`, (iii) `linalg.QR` already provides the Sylvester-flow Q-R parameterization, (iv) `chaos/ode_solvers.RK45Adaptive` already provides the F18/F19 continuous-flow integrator originally built for Lorenz / VanDerPol, (v) Box-Muller from `optim/genetic.go:58-65` already provides the Gaussian base-distribution sampler, (vi) the consumer-side `Net` closure architectural decision keeps `prob/flow/` a math library, not a tensor framework — minimum architectural perturbation, maximum normalizing-flow unlock**. The composition with slot 239's V23 FlowVI-glue makes reality the only zero-dep Go library that ships the full Rezende-2015 → Kingma-Welling-2013 → Dinh-2017 → Papamakarios-2017 → Kingma-2016 → Kingma-Dhariwal-2018 → Chen-2018 → Grathwohl-2019 → Durkan-2019 → DeCao-2020 → Behrmann-2019 → Chen-Behrmann-2019 deep-generative-modeling canon, in one canonical Go API on top of math primitives that already ship at v0.10.0 quality.
