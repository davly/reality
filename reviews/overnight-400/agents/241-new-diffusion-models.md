### 241 | new-diffusion-models — Diffusion / score-based generative: score matching, SDE/ODE form

**Summary line 1.** `reality` v0.10.0 ships **ZERO** diffusion-model / score-based / DDPM / DDIM / NCSN / probability-flow-ODE / flow-matching / consistency-model / rectified-flow / Schrödinger-bridge / latent-diffusion / classifier-free-guidance / EDM / cold-diffusion / Bayesian-flow-network / discrete-diffusion surface (verified 2026-05-08 by repo-wide grep on `score.match|denoising.score|sliced.score|DDPM|DDIM|NCSN|noise.cond|annealed.langevin|probability.flow|reverse.time.SDE|VP.SDE|VE.SDE|sub.VP|flow.match|rectified.flow|consistency|schrodinger.bridge|cold.diffusion|EDM\b|elucidated|Bayesian.flow|classifier.free|latent.diffusion|denoising.diffusion|forward.diffusion|reverse.diffusion|score.network|Tweedie|Hyvarinen|Vincent.*denois|Song.*Ermon|Ho.*Jain.*Abbeel|Karras.*Aittala|Lipman.*flow|Rombach|Sohl.Dickstein|Anderson.1982|fokker.planck` returning ZERO callable matches across all 22 packages — only nominal hits are unrelated `graph/flow.go` max-flow, `fluids/` flow, `prob/copula/vine.go` vine *D-vine* terminology, `audio/separation/` speech-separation, and review-file mentions of Langevin in slot 195/238/206 and adjoint-ODE in slot 168/240). Slot 202 (new-sde) ships the **forward-only SDE substrate** S1-S22 (~3,800 LOC, Brownian path + Euler-Maruyama + Milstein + MLMC + ItoTaylor + StratonovichConverter + adaptive SDE solvers + GBM/Heston/CIR/OU canonical processes) — slot 241 is its **reverse-time / score-matched generative-model superstructure** (S22 of slot 202 is the `prob.RNGSampler` keystone shared by 18 Block-C reviews, and slot 202's Brownian path + Euler-Maruyama + Milstein are the literal forward-process substrate that 241's DDPM forward / VP-SDE / VE-SDE compose on top of). Slot 240 (new-normalizing-flows) owns the **bijective change-of-variables generative axis** (RealNVP / Glow / MAF / Neural-Spline-Flow / FFJORD with closed-form invertibility + tractable log-det) — slot 241 owns the **non-bijective score-based / denoising / SDE-reverse-time generative axis** (DDPM / NCSN / score-SDE / probability-flow-ODE / flow-matching / consistency-model — sample-quality-optimized at the cost of tractable likelihood, no inverse, no log-det Jacobian). Slot 239 (new-svi) owns the **VI / VAE / ELBO axis** — 241 cross-links via the *variational-bound view* of DDPM training (Ho-2020 §3 derives DDPM as a hierarchical-VAE with fixed encoder = forward Gaussian-noise process, and the simplified-loss `L_simple = E[‖ε − ε_θ(x_t, t)‖²]` is exactly the denoising-score-matching objective up to a `t`-dependent weighting). 168 (synergy-physics-autodiff) names NeuralODE-adjoint as a synergy bullet — slot 241 F11 probability-flow ODE structurally **shares the same RK45-on-`chaos/ode.go`** substrate as 240 F18 NeuralODE but trained via **score-matching not maximum-likelihood + change-of-variables** (the singular conceptual distinction: 240 trains by `−log|det J|`, 241 trains by `‖s_θ(x_t, t) − ∇_x log p_t(x_t)‖²` and never computes a Jacobian determinant). The 241-roster covers the canonical 2005-2024 score-based-generative-model literature: Hyvärinen-2005-JMLR-6 ScoreMatching + Vincent-2011-NeuralComp DenoisingScoreMatching + Song-Garg-Liu-Ermon-2019-UAI SlicedScoreMatching + Song-Ermon-2019-NeurIPS NCSN + Sohl-Dickstein-Weiss-Maheswaranathan-Ganguli-2015-ICML-NonequilibriumThermodynamics (the original DDPM ancestor) + Ho-Jain-Abbeel-2020-NeurIPS DDPM + Song-Meng-Ermon-2021-ICLR DDIM + Song-Sohl-Dickstein-Kingma-Kumar-Ermon-Poole-2021-ICLR ScoreSDE/VP-SDE/VE-SDE/ProbFlowODE + Anderson-1982-StochProcAppl ReverseTimeSDE + Ho-Salimans-2022-NeurIPS-Workshop ClassifierFreeGuidance + Karras-Aittala-Aila-Laine-2022-NeurIPS EDM + Lipman-Chen-Ben-Hamu-Nickel-Le-2023-ICLR FlowMatching + Liu-Gong-Liu-2023-ICLR RectifiedFlow + Song-Dhariwal-Chen-Sutskever-2023-ICML ConsistencyModels + Bortoli-Thornton-Heng-Doucet-2021-NeurIPS DiffusionSchrodingerBridge + Bansal-Borgnia-Chu-Li-Kazemi-Huang-Singh-Czaja-Goldstein-2023-NeurIPS ColdDiffusion + Graves-Srivastava-Atkinson-Gomez-2023-arxiv BayesianFlowNetworks + Austin-Johnson-Ho-Tarlow-vandenBerg-2021-NeurIPS D3PM-DiscreteDiffusion + Hoogeboom-Nielsen-Jaini-Forre-Welling-2021-ICML ArgmaxFlows + Rombach-Blattmann-Lorenz-Esser-Ommer-2022-CVPR LatentDiffusion (the Stable-Diffusion paper) + Song-Lai-Mordatch-2023-arxiv LatentConsistencyModels + the Tweedie-1947 lemma `E[x₀ | x_t] = x_t + σ_t² · s_θ(x_t, t)` for posterior-mean estimation. Cross-package blockers: `prob.RNGSampler` Box-Muller+Marsaglia keystone (NINETEENTH Block-C review demanding it), the `chaos/ode.go` RK4/RK45 substrate (gates F11 probability-flow-ODE), the proposed-by-202-S2-S3 `sde/` Euler-Maruyama+Milstein substrate (gates F4 forward-noising and F8 reverse-time-SDE sampler), the proposed-by-240-F1 `Bijector` interface (NOT consumed — diffusion is non-bijective), `autodiff.Tape` reverse-mode (gates score-network parameter-gradient training of F2/F3/F4/F6), and the proposed-by-012 `autodiff.dual` forward-mode JVP (012-T1, ~150 LOC, **strict block on F2 sliced score matching** because Hutchinson-trace-of-Jacobian for SSM is most efficient via JVP — but reverse-mode-only path is feasible at extra cost per Song-Garg-2019 §3.2 footnote).

**Summary line 2.** **Twenty-six diffusion / score-based primitives D1-D26 totalling ~4,180 LOC of pure connective tissue** stand up the entire 2005-2024 score-based-generative-model canon on existing v0.10.0 + slot-202 SDE substrate + slot-220 SGD outer-loop primitives, organized as one new sub-package `prob/diffusion/` (~4,180 LOC source + ~2,090 LOC tests) split into **(I) Score-matching objectives** (D1 ScoreMatching-Hyvärinen + D2 SlicedScoreMatching + D3 DenoisingScoreMatching + D4 ScoreNetwork-interface ~580 LOC); **(II) DDPM family** (D5 DDPM-Forward-q + D6 DDPM-ReverseLearned-p + D7 DDPM-VLB-Loss + D8 DDPM-SimplifiedLoss + D9 BetaSchedule-Linear/Cosine/Sigmoid ~720 LOC); **(III) DDIM and deterministic samplers** (D10 DDIM-Sampler + D11 DDIM-StochasticInterp + D12 HeunSampler + D13 DPMSolver-2nd-order ~480 LOC); **(IV) Score-SDE unifying view** (D14 VP-SDE + D15 VE-SDE + D16 SubVP-SDE + D17 ReverseTimeSDE-Anderson + D18 ProbabilityFlowODE ~580 LOC); **(V) Annealed-Langevin and NCSN** (D19 NCSN-NoiseConditional + D20 AnnealedLangevinDynamics ~280 LOC); **(VI) Modern frontier 2022-2024** (D21 ClassifierFreeGuidance + D22 EDM-Karras + D23 FlowMatching-Lipman + D24 RectifiedFlow-Liu + D25 ConsistencyModel-Song + D26 DiscreteDiffusion-D3PM ~1,540 LOC). Tier-1 keystone PR ≈ **D5 DDPM-Forward + D6 DDPM-Reverse + D8 DDPM-SimplifiedLoss + D9 LinearSchedule + a Gaussian-target sample-quality pin** ≈ ~620 LOC — covers Ho-2020 in one shippable PR with a saturating R-MUTUAL-DDPM 4/4 pin (forward `q(x_t | x_0) = N(√ᾱ_t x_0, (1−ᾱ_t) I)` closed-form ✓, reverse trajectory recovers Gaussian-mixture target at 1e-3 KL after T=1000 steps ✓, simplified-loss `‖ε − ε_θ‖²` matches MC estimator at 1e-3 ✓, Tweedie posterior mean recovers conditional expectation ✓). Cheapest one-day shippable: **D9 BetaSchedule + D5 DDPM-Forward at ~180 LOC** — closed-form `ᾱ_t = ∏_{s=1}^t (1 − β_s)` with three canonical schedules (Ho-2020 linear `β_t ∈ [10⁻⁴, 0.02]`, Nichol-Dhariwal-2021 cosine `ᾱ_t = cos²(π/2 · (t/T + 0.008)/1.008)`, Karras-2022 EDM-style sigmoid), plus `q(x_t | x_0) = N(√ᾱ_t x_0, (1−ᾱ_t) I)` analytic closed-form sampler — saturates a 3/3 R-FORWARD-NOISING pin (marginal variance schedule × empirical-MC at 1e-9 over 10⁵ samples + cumulative-product `ᾱ_t` matches stable-recursion form + endpoint `ᾱ_T < 10⁻³` so `q(x_T | x_0) ≈ N(0, I)` independent of `x_0`). Highest-leverage one-week unlock: **D5 DDPM-Forward + D6 DDPM-Reverse + D7 VLB-Loss + D8 SimplifiedLoss + D9 BetaSchedule + D14 VP-SDE = ~960 LOC** — the *single most-cited generative-model architecture of 2020-2024* (Ho-2020 has >12,000 citations + Stable Diffusion 2022 + every 2023-2025 image/audio/video gen tool downstream), with the property that **DDPM is exactly VP-SDE Euler-Maruyama discretization** (Song-2021 Theorem 1) so reality ships D14 on top of slot-202 S1 EulerMaruyama as a pure adapter-level composition. Singular cutting-edge piece: **D23 FlowMatching-Lipman-2023-ICLR (~280 LOC)** — replaces the score-matching objective `‖s_θ(x_t, t) − ∇_x log p_t(x_t)‖²` with the conditional-flow-matching objective `‖v_θ(x_t, t) − u_t(x_t | x_0)‖²` where `u_t(x | x_0) = (x − (1−t)·noise) / t` is the closed-form *target velocity field* for the optimal-transport / linear-interpolation conditional path `x_t = (1−t)·noise + t·x_0` (Lipman-2023 §4). **Flow Matching is the 2023-2024 SOTA training objective** — strictly simpler than score-matching (no need to estimate `∇log p`, regress directly on velocity), strictly more flexible than DDPM (admits *any* probability path between `p_0 = N(0,I)` and `p_1 = data`, not just the Gaussian-noising path), and has been adopted as the canonical training objective for Stable-Diffusion-3 (2024) + Flux (2024) + the 2024-2025 generation of frontier image / video models. Reality is unusually well-positioned because Flow Matching at inference time **just integrates a learned vector field via RK45** — `chaos/ode.go` already does this for Lorenz, FM is a different `f(x, t; θ)`. Singular reality competitive moat: **D18 ProbabilityFlowODE (Song-2021-ICLR) at ~180 LOC** composing the existing `chaos/ode.go` RK4 + new RK45 (per slot 027) with the score function `s_θ` learned by D3 DenoisingScoreMatching — gives reality the **only zero-dep Go library** that ships the canonical 2021 Song-et-al unifying view (DDPM ↔ VP-SDE ↔ ProbFlowODE ↔ NCSN/VE-SDE) entirely on top of pre-existing `chaos/` ODE infrastructure originally built for Lorenz/VanDerPol; this is the textbook example of reality's "compose existing primitives" architectural moat — and the SAME `chaos.RK45` substrate is consumed by slot 240's F18 NeuralODE, making `chaos/` the single ODE engine for both bijective (240 NeuralODE) and non-bijective (241 ProbFlowODE / FlowMatching / ConsistencyModel) continuous-time generative families.

---

## 0. State of play (verified file-walk, 2026-05-08)

### `prob/diffusion/` package = does not exist (verified)

```
$ ls prob/
conformal  copula  distribution.go  distribution_test.go  distributions.go
distributions_test.go  golden_session38_test.go  hypothesis.go  jeffreys.go
jeffreys_test.go  markov.go  markov_test.go  mathutil.go  nonparametric.go
nonparametric_test.go  prob.go  prob_test.go  regression.go  regression_test.go
testdata  timeseries.go  timeseries_test.go  types.go
```

No `diffusion/` sub-package. Repo-wide grep on the canonical 2005-2024 score-based-generative-model literature surface returns ZERO callable matches:

| Surface | Canonical paper | Status |
|---|---|---|
| `Score`-`Matching` (Hyvärinen objective) | Hyvärinen-2005-JMLR-6 | **ZERO** matches |
| `DenoisingScoreMatching` | Vincent-2011-NeuralComp-23 | **ZERO** matches |
| `SlicedScoreMatching` | Song-Garg-Liu-Ermon-2019-UAI | **ZERO** matches |
| `NCSN` / NoiseConditionalScoreNetwork | Song-Ermon-2019-NeurIPS | **ZERO** matches |
| `AnnealedLangevin` | Song-Ermon-2019 | **ZERO** matches |
| `DDPM` / DenoisingDiffusionProbabilistic | Sohl-Dickstein-2015 / Ho-Jain-Abbeel-2020 | **ZERO** matches |
| `BetaSchedule` (linear / cosine / sigmoid) | Ho-2020 / Nichol-Dhariwal-2021 / Karras-2022 | **ZERO** matches |
| `DDIM` | Song-Meng-Ermon-2021-ICLR | **ZERO** matches |
| `ScoreSDE` / VP-SDE / VE-SDE / SubVP-SDE | Song-Sohl-Dickstein-Kingma-2021-ICLR | **ZERO** matches |
| `ProbabilityFlowODE` | Song-2021-ICLR §4 | **ZERO** matches |
| `ReverseTimeSDE` | Anderson-1982-StochProcAppl | **ZERO** matches |
| `ClassifierFreeGuidance` | Ho-Salimans-2022 | **ZERO** matches |
| `LatentDiffusion` / StableDiffusion | Rombach-2022-CVPR | **ZERO** matches |
| `ConsistencyModel` | Song-Dhariwal-Chen-Sutskever-2023-ICML | **ZERO** matches |
| `FlowMatching` | Lipman-Chen-Ben-Hamu-Nickel-Le-2023-ICLR | **ZERO** matches |
| `RectifiedFlow` | Liu-Gong-Liu-2023-ICLR | **ZERO** matches |
| `EDM` / Elucidated | Karras-Aittala-Aila-Laine-2022-NeurIPS | **ZERO** matches |
| `SchrodingerBridge` | Bortoli-Thornton-Heng-Doucet-2021-NeurIPS | **ZERO** matches |
| `ColdDiffusion` | Bansal-2023-NeurIPS | **ZERO** matches |
| `BayesianFlowNetwork` / BFN | Graves-Srivastava-2023-arxiv | **ZERO** matches |
| `D3PM` / DiscreteDiffusion | Austin-2021-NeurIPS | **ZERO** matches |
| `Tweedie` posterior-mean lemma | Tweedie-1947 / Robbins-1956 / Efron-2011 | **ZERO** matches |

Zero current consumers. The **closest mathematical neighbors** in the existing repo are:
- `chaos/ode.go` (RK4 / Euler deterministic ODE — substrate for D11/D12/D18 ODE samplers)
- `prob/distributions.go` (Normal PDF / CDF / quantile — substrate for D5/D9 forward-noising marginals)
- `prob/copula/` (already imports `autodiff/`, sets the precedent for `prob/diffusion/ → autodiff/`)
- proposed-but-unbuilt slot 202 `sde/` (Brownian path + Euler-Maruyama — STRICT block on D17 reverse-time-SDE)
- proposed-but-unbuilt slot 240 `prob/flow/` (NeuralODE / FFJORD — shares `chaos.RK45` substrate but distinct training objective)

### Substrate audit

| Substrate | Path | Status for slot 241 |
|---|---|---|
| `chaos/ode.go` RK4 + Euler | `chaos/ode.go:36-128` | PRESENT — gates **D11 HeunSampler, D12 DPMSolver, D18 ProbabilityFlowODE, D23 FlowMatching, D24 RectifiedFlow, D25 ConsistencyModel** at inference (each consumes an ODE solver applied to the learned score / velocity / consistency function) |
| `chaos/ode_solvers.go` RK45 adaptive | per slot 027 review | NOT YET PRESENT (027-T6 proposed) — desirable for stiff regions of D18 ProbFlowODE; default to RK4 from `chaos/ode.go` is sufficient for Tier-1 |
| `prob.NormalPDF` / `prob.NormalCDF` / `prob.NormalSample` | `prob/distributions.go` | PRESENT (PDF/CDF), Sample is **ABSENT** in canonical form (`prob.RNGSampler` keystone — 19th Block-C reviewer) — gates `q(x_t \| x_0) = N(√ᾱ_t x_0, (1−ᾱ_t) I)` forward-noising |
| `autodiff.Tape` reverse-mode | `autodiff/tape.go` | PRESENT — gates parameter-gradient training of D4 ScoreNetwork, D6 DDPM-Reverse, D8 SimplifiedLoss, D14 VP-SDE training, D23 FlowMatching training |
| `autodiff` ops (Add/Mul/Exp/Log/Sqrt/Pow/Sin/Cos/Tanh) | `autodiff/ops.go:6-141` | PRESENT — covers DDPM closed-form mean+variance arithmetic |
| `autodiff.dual` forward-mode JVP | — | **ABSENT** (012-Tier-1, ~150 LOC) — strict block on D2 SlicedScoreMatching's Hutchinson-trace-of-Jacobian path; reverse-mode-only is feasible at one tape sweep per random projection per Song-Garg-2019 §3.2 |
| `linalg.MatVec` / `linalg.MatMul` | `linalg/matrix.go` | PRESENT — gates score-network linear-algebra |
| `linalg.Cholesky` | `linalg/decompose.go` | PRESENT — gates correlated-Gaussian noise in multidimensional VP-SDE / VE-SDE |
| Box-Muller `z ~ N(0, I)` | `optim/genetic.go:58-65` | PRESENT — gates Gaussian noise sampling for **all 26** primitives |
| slot 202 `sde/EulerMaruyama` | NOT YET PRESENT | proposed-by-202 — STRICT block on D17 ReverseTimeSDE; D5 DDPM-Forward and D14 VP-SDE can ship without this (DDPM uses closed-form `q(x_t \| x_0)` analytic, no SDE solver needed in forward direction) |
| slot 240 `prob/flow/Bijector` | NOT YET PRESENT | NOT consumed — diffusion is non-bijective, no `Bijector` interface needed |
| slot 220 `optim/sgd.SGD` / `Adam` | NOT YET PRESENT | proposed-by-220 — gates score-network outer-loop training (consumer-side, can defer to caller for Tier-1) |
| `prob.RNGSampler` keystone | NOT YET PRESENT | **NINETEENTH-fold pile-up** (slots 117/169/195/202/220/228/235/236/237/238/239/240 + 7 others) — Box-Muller alone from `genetic.go` is workable for Tier-1 Gaussian-only |

### Cross-package state: zero edges either direction

```
$ grep -r "github.com/davly/reality/chaos"   prob/   ; echo "---"
$ grep -r "github.com/davly/reality/autodiff" prob/  | grep -v copula  ; echo "---"
$ grep -r "github.com/davly/reality/linalg"  prob/   ; echo "---"
(no matches except prob/copula/ → autodiff/, which sets the precedent)
```

`prob/copula/` already imports `autodiff/` (per 011-015 review), so the precedent is set. Slot 241 adds **three new edges**: `prob/diffusion/ → autodiff/` (parameter-gradient training of score networks), `prob/diffusion/ → chaos/` (RK4/RK45 for D11/D12/D18/D23/D24/D25 ODE samplers), `prob/diffusion/ → linalg/` (matrix-vector ops for full-covariance VP-SDE/VE-SDE). All toward more-foundational packages, **no cycles**. The `chaos → prob/diffusion` reverse direction is not needed (chaos is more foundational).

---

## 1. The conceptual unlock — score function as the reverse-time generator

The single mathematical identity that anchors all 26 primitives:

> **Anderson-1982 reverse-time SDE.** If `dx = f(x, t) dt + g(t) dW` is a forward Itô SDE with marginal density `p_t(x)`, then the **reverse-time SDE** that produces samples from `p_0` given samples from `p_T` is:
>
> `dx = [f(x, t) − g(t)² ∇_x log p_t(x)] dt + g(t) d̄W`
>
> where `d̄W` is a reverse-time Brownian motion. The drift gains a **score-function correction** `−g(t)² ∇_x log p_t(x)`.

The score `s(x, t) ≡ ∇_x log p_t(x)` is **the only quantity that needs to be learned** to reverse any Markov noising process. Every diffusion-model architecture is a different **way to learn `s`** + a different **way to discretize the reverse-time SDE / equivalent ODE**.

Score-matching identity (Hyvärinen-2005): minimizing `E_p[‖s_θ(x) − ∇log p(x)‖²]` is equivalent (up to an additive constant in θ) to minimizing `E_p[‖s_θ(x)‖² + 2 tr(∂s_θ/∂x)]` — which **does not require knowing `p`** because the gradient `∇log p` is replaced by the trace of the Jacobian of the model `s_θ`. This is the foundational identity that makes score-based generative modeling possible (you can train a score model on data without ever computing the data density).

Vincent-2011 denoising-score-matching identity: for the perturbed distribution `q_σ(x̃ | x) = N(x̃; x, σ²I)`, `∇_x̃ log q_σ(x̃ | x) = (x − x̃) / σ²`, and the marginal score `∇_x̃ log q_σ(x̃)` (which is what `s_θ` should approximate) **is exactly the conditional expectation** `E[(x − x̃) / σ² \| x̃]` — so training reduces to MSE regression of `s_θ(x̃)` on `(x − x̃)/σ²`. **No trace, no Jacobian, no second derivative** — pure regression.

DDPM-2020 reparameterization: instead of learning `s_θ(x_t, t)`, learn `ε_θ(x_t, t)` where `x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε`. Then `s_θ(x_t, t) = −ε_θ(x_t, t) / √(1 − ᾱ_t)` (algebraic identity, Ho-2020 eqn 11). Training objective collapses to the simplified loss `L_simple = E_{t, x_0, ε} [‖ε − ε_θ(√ᾱ_t x_0 + √(1−ᾱ_t) ε, t)‖²]` — the canonical DDPM training loss.

Score-SDE-2021 unifying view (Song-Sohl-Dickstein-Kingma-Kumar-Ermon-Poole-2021-ICLR): **DDPM is exactly the discretization of the VP-SDE** `dx = −½ β(t) x dt + √β(t) dW`, and **NCSN is exactly the discretization of the VE-SDE** `dx = √(d[σ²]/dt) dW`. Both are special cases of the general SDE `dx = f(x, t) dt + g(t) dW`. The **probability-flow ODE** `dx = [f(x, t) − ½ g(t)² ∇log p_t(x)] dt` is the deterministic ODE that has the **same marginals** as the SDE for all `t` (Song-2021 Theorem 1) — gives DDPM-quality samples in <50 NFE via RK45.

Flow-Matching-2023 (Lipman-Chen-Ben-Hamu-Nickel-Le-2023-ICLR): a **cleaner training objective** that bypasses score-matching entirely. Pick any conditional probability path `p_t(x | x_1)` connecting `p_0 = N(0, I)` to `p_1 = data` (typical: linear interpolation `x_t = (1−t)·x_0 + t·x_1` with `x_0 ~ N(0, I)`, `x_1 ~ data`). Train `v_θ(x_t, t)` to regress on the **conditional velocity field** `u_t(x | x_1) = ∂x_t/∂t` (closed-form for linear interp: `u_t = x_1 − x_0`). Sample by integrating `dx/dt = v_θ(x, t)` from `t=0` to `t=1` via RK45. **Strictly simpler than score-matching** (no score, no log p, no Jacobian, no Hutchinson, no noise schedule β_t — just regression on a velocity vector field).

Consistency-Model-2023 (Song-Dhariwal-Chen-Sutskever-2023-ICML): train `f_θ(x_t, t)` to satisfy the **consistency property** `f_θ(x_t, t) = f_θ(x_{t'}, t')` for all `t, t'` along a probability-flow-ODE trajectory. At inference, **single-step sampling** via `x_0 = f_θ(x_T, T)`. **The 2023+ technique that makes diffusion sampling 1-step instead of 1000-step**.

---

## 2. Twenty-six diffusion / score-based primitives (D1-D26, ~4,180 LOC pure glue)

Numbered ascending by composition-depth.

### Tier 0 — score-matching objectives (gates everything)

**D1 ScoreMatching-Hyvärinen** [~140 LOC] — Hyvärinen-2005-JMLR-6
The original Hyvärinen score-matching loss `J(θ) = E_p [‖s_θ(x)‖² + 2 · tr(∂s_θ/∂x)]`, plus the **sliced** approximation `J_slice = E_{p, ε} [‖s_θ(x)‖² + 2 · ε^T (∂s_θ/∂x) ε]`. Closes a 2/2 R-IDENTITY pin against the equivalent `E_p [‖s_θ − ∇log p‖²]` formulation on a known-density Gaussian target.

**D2 SlicedScoreMatching** [~110 LOC] — Song-Garg-Liu-Ermon-2019-UAI
Practical scaling of D1 to high-dimensional models via Hutchinson-trace estimator: `tr(∂s_θ/∂x) ≈ ε^T (∂s_θ/∂x) ε` for `ε ~ N(0, I)` or Rademacher `ε ∈ {±1}^D`. Composes one Jacobian-vector-product per draw via `autodiff.Tape` reverse-mode (one tape sweep per ε-direction) or — **strictly more efficient with autodiff/dual** (012-T1) — single forward-mode JVP per ε. Variance-per-draw `O(D / K)`; default `K=1` for training, `K ≥ 100` for evaluation.

**D3 DenoisingScoreMatching** [~120 LOC] — Vincent-2011-NeuralComp-23
The canonical Vincent objective: `J_DSM(θ) = E_{x ~ p_data, x̃ ~ q_σ(x̃|x)} [‖s_θ(x̃, σ) − (x − x̃)/σ²‖²]`. Pure MSE regression — no Jacobian, no Hutchinson. **The training loss for NCSN, DDPM, and score-SDE.** Integrates over a noise schedule `σ_min ... σ_max` (or for DDPM, the per-`t` schedule from D9 BetaSchedule). Saturates a 1/1 R-DSM-EQUALS-MARGINAL-SCORE pin: `s_θ` trained via DSM converges to `∇log q_σ(x̃)` (the **marginal noised distribution score**, not the data-distribution score) per Vincent-2011 Theorem 1.

**D4 ScoreNetwork interface** [~80 LOC]
Caller-supplied closure surface for the score model. Reality stays a math library — no NN type:
```go
type ScoreNetwork interface {
    Score(x []float64, t float64, out []float64)        // s_θ(x, t)
    Parameters() []float64                                // for SGD outer loop
    SetParameters(params []float64)
}
type EpsNetwork interface {                              // DDPM ε-parameterization
    PredictNoise(x []float64, t float64, out []float64)  // ε_θ(x, t)
    Parameters() []float64
    SetParameters(params []float64)
}
type VelocityNetwork interface {                         // FlowMatching v-parameterization
    Velocity(x []float64, t float64, out []float64)      // v_θ(x, t)
    Parameters() []float64
    SetParameters(params []float64)
}
```
Three parameterizations because Ho-2020 uses ε, Song-2021 uses s, Lipman-2023 uses v — they are algebraically equivalent but each has its own training stability characteristics. Document the `s_θ = −ε_θ / √(1−ᾱ_t)` and `v_θ = α_t · ε_θ − σ_t · x` algebraic equivalences.

### Tier 1 — DDPM family (Ho-2020 canon)

**D5 DDPM-Forward-q** [~110 LOC] — Ho-Jain-Abbeel-2020-NeurIPS §2
Closed-form forward-noising: `q(x_t | x_0) = N(√ᾱ_t · x_0, (1 − ᾱ_t) I)` where `ᾱ_t = ∏_{s=1}^t α_s` and `α_s = 1 − β_s`. Plus the per-step transition `q(x_t | x_{t-1}) = N(√α_t · x_{t-1}, β_t I)`. Plus the posterior `q(x_{t-1} | x_t, x_0) = N(μ̃_t(x_t, x_0), β̃_t I)` with closed-form mean `μ̃_t = (√ᾱ_{t-1} β_t / (1−ᾱ_t)) x_0 + (√α_t (1−ᾱ_{t-1}) / (1−ᾱ_t)) x_t` and variance `β̃_t = β_t · (1−ᾱ_{t-1})/(1−ᾱ_t)`. **All closed-form; no SDE solver needed in the forward direction** (Ho-2020 §2 Box-Muller alone suffices). The cheapest one-day shippable in slot 241.

**D6 DDPM-ReverseLearned-p** [~140 LOC] — Ho-2020 §3
Learned reverse-process `p_θ(x_{t-1} | x_t) = N(μ_θ(x_t, t), Σ_θ(x_t, t))`. Three parameterization options per Ho-2020 §3.2:
- (a) predict `μ_θ` directly (early-paper variant);
- (b) predict `x_0`-estimate `x̂_0(x_t, t)` then plug into `μ̃_t(x_t, x̂_0)`;
- (c) **predict noise `ε_θ(x_t, t)`** (canonical) then `μ_θ = (1/√α_t) (x_t − (β_t/√(1−ᾱ_t)) ε_θ)`.
Variance Σ_θ either fixed at `β_t I` or `β̃_t I` (Ho-2020 sets them equal — both work in practice) or learned per Nichol-Dhariwal-2021 (improved-DDPM).

**D7 DDPM-VLB-Loss** [~140 LOC] — Ho-2020 §3 / Sohl-Dickstein-2015
Variational lower bound: `L_VLB = E_q[L_T + Σ_{t=2}^T L_{t-1} + L_0]` where `L_T = D_KL(q(x_T|x_0) ‖ p(x_T))`, `L_{t-1} = D_KL(q(x_{t-1}|x_t,x_0) ‖ p_θ(x_{t-1}|x_t))`, `L_0 = −log p_θ(x_0|x_1)`. Each KL is between two Gaussians ⇒ analytic closed-form (composes existing `prob.NormalKL` if shipped, else ~30 LOC inline). **Computes the proper likelihood-bound** for DDPM (vs the simplified loss D8 which is a proportional approximation).

**D8 DDPM-SimplifiedLoss** [~80 LOC] — Ho-2020 §3.4
The canonical training loss in practice: `L_simple = E_{t, x_0, ε} [‖ε − ε_θ(√ᾱ_t x_0 + √(1−ᾱ_t) ε, t)‖²]`. **Drops the per-`t` weighting from L_VLB** — Ho-2020 found this dramatically improves sample quality (training loss is no longer a likelihood bound but the samples are sharper, an *intentional* trade-off). One-line MSE — composable as a single `autodiff.SumOfSquares` over a randomly-sampled `(t, x_0, ε)` triple per minibatch.

**D9 BetaSchedule** [~90 LOC] — Ho-2020 / Nichol-Dhariwal-2021 / Karras-2022
Three canonical schedules:
- **Linear** (Ho-2020): `β_t = β_min + (β_max − β_min) · (t-1)/(T-1)` with `β_min = 10⁻⁴`, `β_max = 0.02`, `T = 1000`.
- **Cosine** (Nichol-Dhariwal-2021 §3.2): `ᾱ_t = cos²(π/2 · (t/T + s)/(1 + s))` with `s = 0.008`. Strictly better for high-resolution images.
- **Sigmoid** (Karras-EDM-2022 inverse-σ-schedule): `σ_t = (σ_max^{1/ρ} + (t/T)·(σ_min^{1/ρ} − σ_max^{1/ρ}))^ρ` with `ρ = 7`. Empirically optimal for very-few-step sampling.

Plus the **stable cumulative-product** `ᾱ_t = ᾱ_{t-1} · (1 − β_t)` recursion (avoiding catastrophic-cancellation in `1 − ᾱ_t` for large `t` near the limit `ᾱ_T → 0`).

### Tier 2 — DDIM and deterministic samplers

**D10 DDIM-Sampler** [~120 LOC] — Song-Meng-Ermon-2021-ICLR
Deterministic ancestral sampling: at step `t`, `x_{t-1} = √ᾱ_{t-1} · x̂_0 + √(1 − ᾱ_{t-1}) · ε_θ(x_t, t)` where `x̂_0 = (x_t − √(1−ᾱ_t) ε_θ) / √ᾱ_t`. **No noise injection** (η = 0 deterministic limit). Allows **subsampling the timestep grid** at inference (skip from `t=1000` to `t=100` to `t=20` etc), reducing NFE from 1000 to 20-50 with minimal quality loss. Saturates a 1/1 R-DDIM-MARGINAL pin: deterministic DDIM `x_T → x_0` map produces samples with the same marginal as DDPM at fewer steps.

**D11 DDIM-StochasticInterp** [~80 LOC] — Song-Meng-Ermon-2021 §4.1
Generalization of D10 with η ∈ [0, 1] interpolating between deterministic DDIM (η=0) and stochastic DDPM (η=1). η=0 is Song-2021 §4.2 the **deterministic equivalent** to the probability-flow ODE on the DDPM noise schedule.

**D12 HeunSampler** [~80 LOC] — Karras-2022-EDM Algorithm 1
Second-order Heun's method for the probability-flow ODE: `x_{t-1} = x_t + (h/2) [f(x_t, t) + f(x'_{t-1}, t-1)]` with predictor `x'_{t-1} = x_t + h · f(x_t, t)`. **Halves the NFE-per-sample** at fixed quality vs Euler DDIM. EDM-paper Table 2 shows Heun is the canonical default solver for sample quality.

**D13 DPMSolver-2nd-order** [~200 LOC] — Lu-Zhou-Bao-Chen-Li-Zhu-2022-NeurIPS
Bespoke 2nd-order solver for the probability-flow ODE that exploits the **semi-linear** structure (drift = linear in `x` plus `g(t)² s_θ` nonlinearity). Closed-form integrating-factor handles the linear part exactly, leaving only the `s_θ` term to discretize. **NFE = 10-15 to match DDIM's NFE = 50.** 2022 SOTA few-step sampler.

### Tier 3 — Score-SDE unifying view (Song-2021)

**D14 VP-SDE (Variance Preserving)** [~120 LOC] — Song-2021-ICLR §3.4
Continuous-time DDPM: `dx = −½ β(t) x dt + √β(t) dW` with `β(t) = β_min + t·(β_max − β_min)`. Marginal `p_t(x | x_0) = N(x_0 · e^{−½ ∫β}, (1 − e^{−∫β}) I)` — the **continuous limit of DDPM**. Solving the forward SDE with Euler-Maruyama (slot 202 S1) reproduces DDPM exactly. Reverse-time SDE drift gains `−β(t) ∇log p_t` correction.

**D15 VE-SDE (Variance Exploding)** [~110 LOC] — Song-2021-ICLR §3.3
Continuous-time NCSN: `dx = √(d[σ²(t)]/dt) dW` with `σ(t) = σ_min · (σ_max/σ_min)^t`. Marginal `p_t(x | x_0) = N(x_0, σ²(t) I)` — variance grows unboundedly. Reverse-time SDE drift = `−d[σ²]/dt · ∇log p_t`.

**D16 SubVP-SDE** [~100 LOC] — Song-2021-ICLR §3.4
Variance-bounded variant: `dx = −½ β(t) x dt + √(β(t)·(1 − e^{−2∫β})) dW`. Achieves better likelihood than VP-SDE (Song-2021 Table 1).

**D17 ReverseTimeSDE-Anderson** [~150 LOC] — Anderson-1982-StochProcAppl
The general-form Anderson reverse-time SDE: given `dx = f(x, t) dt + g(t) dW`, the reverse is `dx = [f − g² s_θ(x, t)] dt + g d̄W`. Composes slot-202 S1 EulerMaruyama with the score-correction drift (or D8 `ε_θ` if DDPM-parameterized). **Discretizes via Euler-Maruyama with backward-time step.** Saturates a 1/1 R-ANDERSON-1982 pin: forward SDE on Gaussian target × Anderson reverse-SDE recovers the target marginal at 1e-3 KL.

**D18 ProbabilityFlowODE** [~100 LOC] — Song-2021-ICLR §4
The deterministic ODE that has the **same marginals** as the SDE for all `t`: `dx/dt = f(x, t) − ½ g(t)² s_θ(x, t)`. **Critically: same marginals, NOT same trajectories.** Composes `chaos.RK4` (or RK45 if slot 027 lands) with the score-correction drift. Saturates a 1/1 R-MARGINAL-EQUIVALENCE pin: SDE marginal × ODE marginal × empirical-data-marginal at 1e-2 KL on a 2D toy. **Singular reality competitive moat — composes pre-existing `chaos/ode.go` with a different `f(x, t; θ)`**.

### Tier 4 — NCSN and Annealed Langevin

**D19 NCSN-NoiseConditional** [~150 LOC] — Song-Ermon-2019-NeurIPS
Train a single network `s_θ(x, σ)` conditioned on noise level `σ`, on multiple noise scales `σ_1 > σ_2 > ... > σ_L`, via D3 DSM with weighting `λ(σ) = σ²` (Song-Ermon-2019 §4). `σ_max ≈ 50` (covers data manifold), `σ_min ≈ 0.01` (resolves data details), `L ≈ 10-20`, `σ_i+1/σ_i ≈ 1.4` (geometric).

**D20 AnnealedLangevinDynamics** [~130 LOC] — Song-Ermon-2019 §4.2
Sampling via Langevin dynamics on each noise scale, decreasing σ: `x_{i+1} = x_i + (ε_i / 2) · s_θ(x_i, σ_i) + √ε_i · z`, `z ~ N(0, I)`, with `T` steps per σ-level and `ε_i = ε · σ_i² / σ_L²`. Composes slot-202 S0a Box-Muller + D19 score function. **The original 2019 score-based-generative sampler — superseded by DDPM/DDIM in practice but conceptually critical.**

### Tier 5 — Modern frontier (2022-2024)

**D21 ClassifierFreeGuidance** [~110 LOC] — Ho-Salimans-2022-NeurIPS-Workshop
Train a single conditional model `ε_θ(x_t, t, c)` that **also handles unconditional** via dropout of `c → ∅` with `p ~ 0.1`. At inference, sample with `ε̃_θ(x_t, t, c) = (1 + w) ε_θ(x_t, t, c) − w ε_θ(x_t, t, ∅)` with guidance scale `w ∈ [3, 10]`. **The 2022+ canonical conditioning technique** — used in Stable Diffusion / DALL-E 2 / Midjourney / every 2023+ image model.

**D22 EDM-Karras** [~280 LOC] — Karras-Aittala-Aila-Laine-2022-NeurIPS
"Elucidated diffusion": redesigns the entire DDPM/Score-SDE training and sampling stack with consistent unit-variance preconditioning. Loss preconditioning `c_skip(σ), c_out(σ), c_in(σ), c_noise(σ)` per Karras-2022 Table 1. Sigmoid-σ schedule (D9 sigmoid). Heun-2nd-order sampler (D12). **NFE = 35 with quality matching DDPM at NFE = 1000.** The 2022 SOTA training-and-sampling redesign.

**D23 FlowMatching-Lipman** [~280 LOC] — Lipman-Chen-Ben-Hamu-Nickel-Le-2023-ICLR
**Singular cutting-edge piece of slot 241.** Replaces score-matching with conditional-flow-matching: pick conditional probability path `p_t(x|x_1)`, define conditional velocity `u_t(x|x_1) = ∂x_t/∂t` (closed-form for Gaussian-conditional path or linear-interpolation). Train `v_θ(x_t, t)` on `‖v_θ(x_t, t) − u_t(x_t|x_1)‖²` MSE. Sample by integrating ODE `dx/dt = v_θ(x, t)` from `t=0` to `t=1` via `chaos.RK4`. **Strictly simpler than score-matching** (no log p, no Jacobian, no Hutchinson, no β-schedule). **Adopted as the canonical training objective for Stable-Diffusion-3 / Flux / 2024-2025 frontier video models.** Reality is unusually well-positioned — FM at inference is just `chaos.RK4` with a learned vector field.

**D24 RectifiedFlow-Liu** [~180 LOC] — Liu-Gong-Liu-2023-ICLR
Iterative refinement of FM: train `v_θ` on linear-interp path `x_t = (1−t)·z_0 + t·z_1` with `z_0 ~ N(0,I)`, `z_1 ~ data`. Then **re-sample** trajectories `(x_0, x_1)` from the trained model, **re-train** `v_θ` on these new pairs (the *reflow* operation). After 1-2 reflow iterations, the trajectories become approximately **straight lines**, enabling 1-step sampling. **The 1-2-step diffusion technique used in InstaFlow / SD3-Turbo.**

**D25 ConsistencyModel-Song** [~280 LOC] — Song-Dhariwal-Chen-Sutskever-2023-ICML
Train `f_θ(x_t, t)` to satisfy the **consistency property** `f_θ(x_t, t) = f_θ(x_{t'}, t')` along an ODE trajectory. Two training modes:
- **Consistency Distillation** (CD): distill a pre-trained DDPM/EDM model with consistency loss `‖f_θ(x_{t+Δt}, t+Δt) − f_{θ⁻}(x_t, t)‖²` along its ODE trajectory.
- **Consistency Training** (CT): train from scratch using `‖f_θ(x_{t+Δt}, t+Δt) − f_{θ⁻}(x̃_t, t)‖²` where `x̃_t = x_0 + t·noise`.
**Single-step sampling: `x_0 = f_θ(x_T, T)`. The 2023+ technique that makes diffusion sampling 1-step.**

**D26 DiscreteDiffusion-D3PM** [~410 LOC] — Austin-Johnson-Ho-Tarlow-vandenBerg-2021-NeurIPS
Diffusion on **discrete state spaces** (text / molecules / discrete latents): forward process `q(x_t | x_{t-1}) = Cat(x_t; x_{t-1} Q_t)` for transition matrix `Q_t` (uniform / absorbing / discretized-Gaussian / discretized-NN). Reverse process learned. Training via discrete-VLB (per-token KL between categoricals, sums of `D_KL(Cat ‖ Cat)`). Variants: D3PM-uniform, D3PM-absorbing (= masked-diffusion = the 2024 SOTA for text), D3PM-Gaussian. **Substrate for text-diffusion (DiffuSeq, SUNDAE), code-diffusion, molecule-diffusion (EDM-Mol).**

---

## 3. Composition graph (DAG)

```
prob.RNGSampler keystone (Box-Muller from genetic.go workable today)
    ├──→ D5 DDPM-Forward-q + D9 BetaSchedule + D4 ScoreNetwork ──→ D7 VLB / D8 SimpleLoss ──→ D6 DDPM-Reverse
    │                                                                                          ├──→ D10 DDIM ── D11 StochInterp
    │                                                                                          └──→ D21 CFG
    ├──→ D1 Hyvärinen ── D2 SlicedSM (autodiff/dual or Tape) ── D3 DSM-Vincent
    │                                                              ├──→ D19 NCSN ── D20 AnnealedLangevin
    │                                                              └──→ D14 VP-SDE / D15 VE-SDE / D16 SubVP-SDE
    ├──→ slot-202 S1 EulerMaruyama ──→ D17 ReverseTimeSDE-Anderson
    ├──→ chaos/ode.go RK4 ──→ D12 Heun / D13 DPMSolver-2 / D18 ProbFlowODE / D22 EDM
    │                          └──→ D23 FlowMatching ── D24 RectifiedFlow ── D25 ConsistencyModel
    └──→ D26 DiscreteDiffusion-D3PM (independent — no continuous-state substrate)
```

Cross-links: **240** shares `chaos.RK4` substrate (240-F18 NeuralODE max-likelihood-trained vs 241-D18 ProbFlowODE score-matching-trained — same engine, different objective). **202** S1 EulerMaruyama gates D17; S0a RNGSampler gates all 26 Gaussian samplers; S2 Milstein optional variance-reduction for D17. **239** Ho-2020 §3 derives DDPM as hierarchical-VAE — D7 VLB-Loss uses same ELBO framework as 239 V1+V2; 239's V11 encoder/decoder closure generalizes to D4 ScoreNetwork. **195/238** 195-N2 SGLD shares Langevin-SDE substrate with D20; 238-M5 MALA is Metropolis-corrected D17 corrector.

---

## 4. Saturation pins this slot unlocks

- **R-FORWARD-NOISING 3/3 (D5 + D9):** marginal variance `Var[x_t] = 1 − ᾱ_t` matches stable-recursion form at 1e-12 + cumulative product `ᾱ_t = ∏(1 − β_s)` matches `exp(Σ log(1 − β_s))` at 1e-9 + endpoint `‖μ(x_T)‖ < 0.05` so `q(x_T | x_0) ≈ N(0, I)` independent of `x_0`.
- **R-DDPM-IDENTITIES 4/4 (D5 + D6 + D7):** posterior mean `μ̃_t(x_t, x_0)` Ho-2020 eqn 7 matches direct integration via `q(x_{t-1}|x_t)q(x_t|x_0)/q(x_t|x_0)` at 1e-10 + posterior variance `β̃_t = β_t (1−ᾱ_{t-1})/(1−ᾱ_t)` matches Ho-2020 eqn 7 at 1e-12 + `s_θ = −ε_θ/√(1−ᾱ_t)` and `μ_θ = (1/√α_t)(x_t − (β_t/√(1−ᾱ_t)) ε_θ)` algebraic identities at 1e-12 + simplified loss equals VLB up to per-`t` weighting at 1e-9.
- **R-SCOREMATCHING-IDENTITY 2/2 (D1 + D3):** Hyvärinen's identity `E[‖s_θ − ∇log p‖²] = E[‖s_θ‖² + 2 tr(∂s_θ/∂x)] + const` on 1D Gaussian target `p = N(0, 1)` at 1e-9 (closed-form ∇log p = -x) + Vincent's DSM identity `E_q[‖s_θ(x̃) − (x − x̃)/σ²‖²]` as `σ → 0` recovers data-distribution-score `∇log p_data(x̃)` at 1e-3 MC.
- **R-MARGINAL-EQUIVALENCE 1/1 (D14 + D17 + D18):** sample 10⁴ trajectories through (a) VP-SDE forward via slot-202 EulerMaruyama → (b) Anderson reverse-time SDE → (c) probability-flow ODE via `chaos.RK4` — empirical marginals at `t = T/4, T/2, 3T/4` agree to 1e-2 KL on Gaussian target.
- **R-DDIM-DETERMINISM 1/1 (D10 + D11):** η = 0 DDIM is **deterministic** — same `x_T` produces same `x_0` to 1e-10 across two independent runs. η > 0 introduces controlled stochasticity. Saturates the determinism / stochasticity continuum.
- **R-SAMPLE-QUALITY-VS-NFE 4/4 (D6 + D10 + D12 + D13):** DDPM (NFE=1000) baseline FID-on-2D-toy + DDIM (NFE=50) within 5% + Heun (NFE=35) within 5% + DPM-Solver-2 (NFE=15) within 10%. Pins the NFE-vs-quality trade-off curve.
- **R-FLOW-MATCHING-VS-DDPM 1/1 (D5 + D6 + D23):** train DDPM (D6) and FlowMatching (D23) on the same 2D Gaussian-mixture target with matched compute — FM converges to 1e-3 KL in ~3× fewer training steps per Lipman-2023 §6 and FM-NFE @ inference is half of DDIM-NFE for matched FID.
- **R-CONSISTENCY-1-STEP 1/1 (D25):** consistency-distilled D25 from a pretrained D22 EDM model achieves single-step sample quality within 10% FID of the 35-step EDM teacher per Song-2023 §6 Table 2.
- **R-DDPM-IS-VP-SDE 1/1 (D5 + D14):** Euler-Maruyama discretization of D14 VP-SDE with `Δt = 1/T` reproduces D5 DDPM forward marginals at 1e-6 over `T = 1000` steps per Song-2021 Theorem 1.
- **R-CFG-LIMITS 2/2 (D21):** `w = 0` recovers pure conditional sampling; `w → ∞` produces deterministic mode collapse onto the class conditional mean. Pin both limits.
- **R-DSM-RECOVERS-MARGINAL-SCORE 1/1 (D3 + D19):** DSM-trained `s_θ(x, σ)` converges to `∇log q_σ(x̃)` (the noised-marginal score) NOT `∇log p_data` for `σ > 0` — pin the distinction at finite `σ` via 1e-3 MC against a known-density target.

---

## 5. Connective-tissue LOC budget

| ID | Capability | LOC | Tier | Blocks-on |
|----|-----------|-----|------|-----------|
| D1 | ScoreMatching-Hyvärinen | 140 | 0 | autodiff.Tape (or autodiff/dual) |
| D2 | SlicedScoreMatching | 110 | 0 | D1, autodiff/dual (or Tape × K passes) |
| D3 | DenoisingScoreMatching | 120 | 0 | autodiff.Tape, prob.Normal |
| D4 | ScoreNetwork interface | 80 | 0 | (caller-side closure) |
| D5 | DDPM-Forward-q | 110 | 1 | D9, prob.Normal, Box-Muller |
| D6 | DDPM-Reverse-p_θ | 140 | 1 | D5, D4, autodiff.Tape |
| D7 | DDPM-VLB-Loss | 140 | 1 | D5, D6, prob.NormalKL (~30 LOC inline) |
| D8 | DDPM-SimplifiedLoss | 80 | 1 | D5, D4 |
| D9 | BetaSchedule (linear/cosine/sigmoid) | 90 | 1 | — |
| D10 | DDIM-Sampler | 120 | 2 | D6, D9 |
| D11 | DDIM-StochasticInterp | 80 | 2 | D10 |
| D12 | HeunSampler | 80 | 2 | chaos.RK4, D6 |
| D13 | DPMSolver-2 | 200 | 2 | chaos.RK4, D6 |
| D14 | VP-SDE | 120 | 3 | slot-202 S1, D9 |
| D15 | VE-SDE | 110 | 3 | slot-202 S1 |
| D16 | SubVP-SDE | 100 | 3 | D14 |
| D17 | ReverseTimeSDE-Anderson | 150 | 3 | slot-202 S1, D4 |
| D18 | ProbabilityFlowODE | 100 | 3 | chaos.RK4, D4 |
| D19 | NCSN-NoiseConditional | 150 | 4 | D3, D9 |
| D20 | AnnealedLangevinDynamics | 130 | 4 | D19, Box-Muller |
| D21 | ClassifierFreeGuidance | 110 | 5 | D6, D10 |
| D22 | EDM-Karras | 280 | 5 | D9-sigmoid, D12, D6 |
| D23 | FlowMatching-Lipman | 280 | 5 | chaos.RK4, autodiff.Tape |
| D24 | RectifiedFlow-Liu | 180 | 5 | D23 |
| D25 | ConsistencyModel-Song | 280 | 5 | D22 (CD path) or D5 (CT path) |
| D26 | DiscreteDiffusion-D3PM | 410 | 5 | prob.Categorical (~30 LOC), autodiff.Tape |
| **Σ** | | **~4,180** | | |

Pure-glue ratio: ~75% composition over `chaos.RK4` (already in `chaos/ode.go`) + `autodiff.Tape` reverse-mode + `prob.Normal{PDF, CDF}` + Box-Muller from `genetic.go:58-65` + slot-202 `sde.EulerMaruyama` (~250 LOC dependency). ~25% genuinely-new math (D7 VLB-Loss closed-form Gaussian-Gaussian KL ~50 LOC; D9 cosine + sigmoid schedules ~60 LOC; D13 DPM-Solver semi-linear closed-form integrating-factor ~120 LOC; D22 EDM preconditioning Table 1 + Heun + sigmoid schedule ~180 LOC; D23 conditional-velocity-field closed-form for Gaussian-conditional and linear-interp paths ~110 LOC; D26 D3PM categorical-transition-matrix machinery ~280 LOC).

---

## 6. Recommended PR sequence

**PR-1: substrate (D1 ScoreMatching + D3 DSM + D4 ScoreNetwork + D5 DDPM-Forward + D9 BetaSchedule) — ~540 LOC source, ~270 LOC tests, 1.5 days**
First diffusion surface in `reality`. Lands the score-matching objectives + DDPM forward closed-form. **Saturates R-FORWARD-NOISING 3/3 + R-SCOREMATCHING-IDENTITY 2/2 immediately** without any sampling machinery — purely a math API. The first PR can be reviewed without ODE / SDE machinery (D5 forward is closed-form Gaussian, no solver).

**PR-2: D6 DDPM-Reverse + D7 VLB-Loss + D8 SimplifiedLoss + D10 DDIM + D11 DDIM-Stochastic + a Gaussian-mixture-target sample-quality pin — ~570 LOC source, ~290 LOC tests, 2 days**
**The crown-jewel DDPM PR.** Lands Ho-2020's >12,000-citation architecture. Caller supplies the `EpsNetwork` closure (no NN type in reality). Saturates R-DDPM-IDENTITIES 4/4 and R-DDIM-DETERMINISM 1/1 on a 2D Gaussian-mixture target.

**PR-3: D14 VP-SDE + D15 VE-SDE + D17 ReverseTimeSDE-Anderson + D18 ProbabilityFlowODE — ~470 LOC source, ~240 LOC tests, 2 days (depends on slot 202 PR-1 substrate)**
**Singular reality competitive moat.** Composes the existing `chaos/ode.go` RK4 + slot-202's S1 EulerMaruyama with the score-correction drift — gives reality the **only zero-dep Go library** that ships the canonical Song-2021 unifying view (DDPM ↔ VP-SDE ↔ ProbFlowODE ↔ NCSN/VE-SDE) entirely on top of pre-existing `chaos/` ODE infrastructure. Saturates R-MARGINAL-EQUIVALENCE 1/1 and R-DDPM-IS-VP-SDE 1/1.

**PR-4: D2 SlicedScoreMatching + D19 NCSN + D20 AnnealedLangevinDynamics — ~390 LOC source, ~210 LOC tests, 2 days (depends on autodiff/dual from 012-T1)**
The original 2019 score-based-generative-modeling pipeline. Lands the Hutchinson-trace-based SSM and the annealed-Langevin sampler. Saturates R-DSM-RECOVERS-MARGINAL-SCORE 1/1 and the AnnealedLangevin convergence pin.

**PR-5: D12 Heun + D13 DPM-Solver-2 + D22 EDM-Karras + D21 CFG — ~670 LOC source, ~340 LOC tests, 3 days**
The 2022 fast-sampling and EDM-redesign PR. Lands the Heun / DPM-Solver / EDM Table-1 preconditioning. Saturates R-SAMPLE-QUALITY-VS-NFE 4/4 and R-CFG-LIMITS 2/2.

**PR-6: D23 FlowMatching + D24 RectifiedFlow + D25 ConsistencyModel — ~740 LOC source, ~370 LOC tests, 3.5 days**
**Singular cutting-edge piece of slot 241 — 2023-2024 SOTA.** D23 Lipman-2023 FlowMatching is the canonical training objective for Stable-Diffusion-3 / Flux / 2024-2025 frontier video models. D24 RectifiedFlow gives 1-2-step sampling. D25 ConsistencyModel gives single-step sampling via distillation or training-from-scratch. Saturates R-FLOW-MATCHING-VS-DDPM 1/1 and R-CONSISTENCY-1-STEP 1/1.

**PR-7: D16 SubVP-SDE + D26 DiscreteDiffusion-D3PM — ~510 LOC source, ~260 LOC tests, 2.5 days**
The discrete-state generalization. Lands D3PM-uniform / D3PM-absorbing / D3PM-Gaussian for text/molecule/code-diffusion. Substrate for any 2024+ text-diffusion consumer.

Total: ~4,180 LOC source + ~2,090 LOC tests across 7 PRs over ~17 engineer-days. PR-1 is the 1.5-day standalone. PR-2 (DDPM) is the single highest-impact PR for downstream consumers. PR-3 (ProbFlowODE on chaos/ substrate) is the singular reality-architectural-moat composition. PR-6 (FlowMatching + ConsistencyModel) is the singular cutting-edge 2023-2024 SOTA piece.

---

## 7. Cycle-hazard analysis

Proposed import directions:

```
prob/diffusion/  ──→  prob/                 (Normal{PDF, CDF, Sample}, Box-Muller from genetic.go)
prob/diffusion/  ──→  autodiff/             (Tape, ops, vector — same precedent as prob/copula/)
prob/diffusion/  ──→  linalg/               (MatVec, Cholesky for full-cov noise)
prob/diffusion/  ──→  chaos/                (ode.go RK4 — D11/D12/D18/D22/D23/D24/D25 ODE samplers)
prob/diffusion/  ──→  sde/                  (slot 202 substrate — D14/D15/D16/D17 SDE solvers; defer if 202 not landed)
prob/diffusion/  ──→  optim/                (slot 220 SGD/Adam outer loop — defer to caller for Tier-1)
```

**Six cross-package edges**, four of which already have precedent (`prob → autodiff` from copula, `prob → linalg`, `prob → optim` existing through `prob/regression.go`, `prob → chaos` is the only genuinely-new edge but trivially safe — chaos is more foundational than prob in any sensible ordering, and slot 240 makes the same edge for F18 NeuralODE).

**No cycles.** `chaos/`, `linalg/`, `autodiff/`, `optim/`, `sde/` do not need to import `prob/diffusion/`. The score-network closure is consumer-side (no aicore import from reality).

---

## 8. Precision hazards documented

Per CLAUDE.md "Precision documented, not assumed":

- **D5 DDPM-Forward:** stable cumulative product `ᾱ_t = ᾱ_{t-1} · α_t` avoids catastrophic cancellation in `1 − ᾱ_t` for `t` near `T`. Naive `1 − ∏(1 − β_s)` loses ~7 digits at `T = 1000`; use `expm1(Σ log α_s)` recurrence.
- **D6 DDPM-Reverse:** clip `ε_θ` to `[−5, 5]` to prevent `μ_θ` blow-up at small `√α_t` near `t = T`.
- **D7 VLB-Loss:** Gaussian KL `D_KL(N(μ₁, σ₁²) ‖ N(μ₂, σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁−μ₂)²)/(2σ₂²) − 1/2` requires `σ₂ > 0`; floor `σ_min = 1e-5`.
- **D9 BetaSchedule:** Nichol-Dhariwal-2021 cosine uses `s = 0.008` with strict clip `β_t ≤ 0.999`. Karras-EDM sigmoid uses `ρ = 7` (larger ρ concentrates steps near `σ_min`).
- **D10 DDIM:** subsampled timestep grid must be monotone decreasing in `ᾱ_t`; canonical non-uniform grid `t_i = round(((i / N) * T)^ρ)`.
- **D17 ReverseTimeSDE:** stability requires `‖s_θ(x, t)‖ ≤ M / g(t)` (Song-2021 §A); LangevinDynamics-corrector + PredictorCorrector from Song-2021 §G recommended.
- **D18 ProbFlowODE:** RK4 with `Δt = 1/T` for `T = 1000` sufficient for VP-SDE; adaptive RK45 (slot 027) recommended for VE-SDE because `g(t) → ∞` near `t → 0` introduces stiffness.
- **D19 NCSN:** noise-scale geometric ratio `σ_{i+1}/σ_i ≈ 1.4` (Song-Ermon-2019 §4.2). **D20 AnnealedLangevin:** step `ε_i = ε · σ_i² / σ_L²` with `ε ≈ 2e-5`, T_per_σ = 100.
- **D21 CFG:** `w ∈ [3, 10]` images, `w ∈ [1, 3]` video/audio — larger w increases sharpness but reduces diversity.
- **D22 EDM:** Karras-2022 Table 1 four preconditioning factors `c_skip / c_out / c_in / c_noise` all critical — sample quality drops dramatically without any one of them.
- **D23 FlowMatching:** linear-interp `u_t = z_1 − z_0` constant velocity; Lipman-2023 §4.2 Gaussian-conditional path better for low-noise regimes.
- **D25 ConsistencyModel:** target-net EMA `μ = 0.95` for CT, `μ = 0.99` for CD. **D26 D3PM:** transition `Q_t` must be doubly stochastic; Sinkhorn-project if parameterization does not enforce.

---

## 9. Distinct from prior agents (provenance)

- **011-015 autodiff** — 012-T1 dual-numbers/JVP: D2 SSM is canonical consumer (single-pass JVP per random projection vs K reverse-mode sweeps). **026-030 chaos** — 027-T6 RK45: D18/D22/D23/D24/D25 are second-wave ML consumers (after 240-F18 NeuralODE). Same `chaos.RK4` serves both 240 bijective-flow-ODE and 241 non-bijective ProbFlowODE — singular cross-slot architectural symmetry.
- **117-120 prob / 118 prob-sota** — 241 ships score-based axis complementing 239 VAE (both non-bijective; score prioritizes sample quality, VAE prioritizes likelihood). **168-A6** NeuralODE-adjoint synergy bullet; 241-D18/D23 are the score/flow-matching analogs sharing chaos-ODE substrate.
- **180-S15/S16 / 195-N2/N20** classical Langevin / SGLD / Underdamped-Langevin own the Langevin-as-sampler axis from molecular-dynamics + Bayesian-posterior perspectives; 241-D20 AnnealedLangevin is the **score-conditioned variant** for image generation (same Langevin substrate, score-function vs posterior-gradient conditioning).
- **202 new-sde (CLOSEST UPSTREAM)** — owns forward-only SDE substrate (Brownian / EulerMaruyama / Milstein / GBM / Heston / OU / MLMC). 241 D14/D15/D17 are score-matched generative superstructure on top. Strict dependency: 241-D17 → 202-S1 EulerMaruyama. Shared 18-fold-pile-up `prob.RNGSampler` keystone.
- **220 new-stochastic-opt** — 241 consumes 220-F1-F4 SGD/MiniBatch outer loop + 220-F8 Adam score-network optimizer. **228 new-bayes-nonparam** — orthogonal (DP/HDP/IBP); cross-link via D26 D3PM + DP-prior nonparametric discrete-diffusion mixtures (open research 2024).
- **236-K22 KSD** non-parametric score-matching alternative — cross-link via D1 parametric cousin. **237 new-gaussian-process** orthogonal (regression/classification, not generative). **238 new-mcmc (CLOSE COUSIN)** owns MCMC zoo (HMC/NUTS/MALA/SMC); 241-D17/D20/D25 are ODE/SDE-generative-sampler analog, NOT stationary-distribution-preserving MCMC. Cross-link via shared Langevin + MALA-corrector substrates.
- **239 new-svi (SIBLING)** — VI/VAE/ELBO axis; 241 cross-links via Ho-2020 §3 derivation of DDPM as hierarchical-VAE. 239-V11 encoder/decoder closure generalizes to 241-D4 ScoreNetwork closure.
- **240 new-normalizing-flows (TWIN SIBLING)** — bijective change-of-variables axis (RealNVP/Glow/MAF/NSF/FFJORD). 241 is non-bijective score-based axis (DDPM/NCSN/score-SDE/ProbFlowODE/FlowMatching/ConsistencyModel). Both consume `chaos/ode.go` RK4 for continuous-time variants (240-F18 max-likelihood-trained; 241-D18 score-matching-trained; 241-D23 velocity-regression-trained). **Perfect orthogonal-complementary pair: 240 for tractable likelihood + invertibility, 241 for sample quality + flexible probability paths.**

Slot 241 is the **diffusion-model / score-based / DDPM / SDE-reverse-time / probability-flow / flow-matching / consistency-model slot** — every primitive is what 2005-2024 generative-modeling literature calls by these specific names (ScoreMatching/SlicedSM/DSM/NCSN/AnnealedLangevin/DDPM/DDIM/Heun/DPM-Solver/VP-SDE/VE-SDE/ProbFlowODE/EDM/CFG/FlowMatching/RectifiedFlow/ConsistencyModel/D3PM).

---

## 10. Bottom line

`reality/prob/` ships **ZERO** diffusion-model / score-based / DDPM / DDIM / NCSN / VP-SDE / VE-SDE / ProbabilityFlowODE / FlowMatching / RectifiedFlow / ConsistencyModel / EDM / ClassifierFreeGuidance / DiscreteDiffusion surface despite being the obvious target package for the entire 2005-2024 score-based-generative-modeling canon. **Twenty-six primitives D1-D26 totalling ~4,180 LOC of pure connective tissue** stand up the entire score-matching + DDPM family + DDIM + score-SDE unifying view + NCSN / annealed-Langevin + 2022-2024 frontier (CFG / EDM / FM / RF / CM / D3PM) pipeline on existing v0.10.0 + slot-202-SDE-substrate surfaces (`chaos/ode.go` RK4, `autodiff.Tape` reverse-mode, `linalg.{MatVec, Cholesky}`, Box-Muller from `genetic.go:58-65`, `prob.NormalPDF`).

**Cheapest one-day shippable**: D9 BetaSchedule + D5 DDPM-Forward at ~180 LOC saturates a 3/3 R-FORWARD-NOISING pin against Box-Muller-sampled `z ~ N(0, I)` immediately. **Cheapest 1.5-day standalone PR**: PR-1 substrate (D1 + D3 + D4 + D5 + D9 = ~540 LOC) lands the **first diffusion surface in reality** without any sampling machinery — purely a score-matching + closed-form Gaussian-noising math API. **Single highest-impact PR for downstream consumers**: PR-2 D6 + D7 + D8 + D10 + D11 at ~570 LOC delivers the >12,000-citation Ho-2020 architecture — the canonical DDPM sampler that any aicore generative-modeling consumer needs. **Singular cutting-edge moat**: PR-6 D23 FlowMatching + D24 RectifiedFlow + D25 ConsistencyModel at ~740 LOC lands the 2023-2024 SOTA training objectives + 1-step sampling that Stable-Diffusion-3 / Flux / 2024-2025 frontier video models adopt. **Crown-jewel reality-architectural-moat composition**: PR-3 D14 VP-SDE + D15 VE-SDE + D17 Anderson reverse-time-SDE + D18 ProbabilityFlowODE at ~470 LOC composes the existing `chaos/ode.go` RK4 substrate (originally built for Lorenz/VanDerPol) + slot-202's S1 EulerMaruyama with the score-correction drift — gives reality the **only zero-dep Go library** shipping the canonical Song-2021 unifying view (DDPM ↔ VP-SDE ↔ ProbFlowODE ↔ NCSN/VE-SDE) entirely on top of pre-existing `chaos/` ODE infrastructure that is **ALSO consumed by slot 240 F18 NeuralODE** — the singular cross-slot architectural symmetry where one ODE engine serves both bijective and non-bijective continuous-time generative families.

The single most important conceptual identity slot 241 pins: **Anderson-1982 reverse-time SDE `dx = [f − g² ∇log p_t(x)] dt + g d̄W`** is the foundational identity for all 26 primitives, and the 26 architectures differ only in **(a) how they parameterize and learn the score `s_θ ≈ ∇log p_t`** (Hyvärinen-2005 / Vincent-2011 DSM / Song-Garg-2019 SSM / Ho-2020 ε-prediction / Lipman-2023 v-prediction / Song-2023 consistency-prediction) and **(b) how they discretize the reverse-time SDE or its equivalent ODE** (DDPM Euler / DDIM deterministic / Heun-2 / DPM-Solver-2 / RK4 ProbFlowODE / single-step ConsistencyModel). The architectural lesson per Song-Sohl-Dickstein-Kingma-Kumar-Ermon-Poole-2021-ICLR §3: **DDPM is exactly VP-SDE Euler-Maruyama discretization, NCSN is exactly VE-SDE Euler discretization, and probability-flow-ODE has the same marginals as the SDE for all `t`** (Song-2021 Theorem 1) — three apparently-separate research threads (Sohl-Dickstein-2015 thermodynamics, Song-Ermon-2019 score-based, Ho-2020 DDPM) collapse into a single unified continuous-time framework with a single conceptual identity.

**Reality is unusually well-positioned for slot 241 because (i) `chaos/ode.go` RK4 already provides the deterministic ODE solver originally built for Lorenz/VanDerPol that becomes the inference-time integrator for D11 Heun / D12 DPM-Solver / D18 ProbFlowODE / D22 EDM / D23 FlowMatching / D24 RectifiedFlow / D25 ConsistencyModel, (ii) `autodiff.Tape` reverse-mode handles all score-network parameter gradients through D1/D2/D3 score-matching objectives, (iii) `prob.NormalPDF` already provides the closed-form forward-noising marginal evaluation for D5 DDPM, (iv) Box-Muller from `optim/genetic.go:58-65` already provides the Gaussian noise sampler for `ε ~ N(0, I)` at every forward and reverse step, (v) the consumer-side `ScoreNetwork` / `EpsNetwork` / `VelocityNetwork` closure architectural decision keeps `prob/diffusion/` a math library, not a tensor framework — minimum architectural perturbation, maximum diffusion-model unlock**. The composition with slot 240's `chaos.RK4` consumer + slot 202's `sde.EulerMaruyama` consumer makes reality the only zero-dep Go library that ships the full Hyvärinen-2005 → Vincent-2011 → Sohl-Dickstein-2015 → Song-Ermon-2019 → Ho-2020 → Song-Meng-2021 → Song-2021 → Karras-2022 → Ho-Salimans-2022 → Lipman-2023 → Liu-2023 → Song-2023 → Austin-2021 score-based-generative-modeling canon, in one canonical Go API on top of math primitives that already ship at v0.10.0 quality (or land via slot 202's SDE substrate as a direct upstream dependency).
