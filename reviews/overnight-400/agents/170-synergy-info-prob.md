# 170 | synergy-info-prob

**Topic:** info × prob — rate-distortion, channel capacity, MI lower bounds.
**Block:** B (cross-package synergies). **Date:** 2026-05-08. **Scope:**
capabilities that emerge ONLY when `info/`, `infogeo/`, `compression/`, and
`prob/` are composed; not what each lacks in isolation (087 / 091-095 /
041-045 / 116-120 own those).

## Two-line summary

The repo ships every quadrant Shannon needs (`compression/entropy.go` plug-in
H/MI/KL bits, `infogeo/fdiv.go` 7 f-divergences nats, `optim/transport/
sinkhorn.go` log-domain Sinkhorn = the same alternating-projection structure
Blahut-Arimoto needs, `prob.MarkovSteadyState` power-iteration, `info/mdl/`
NML+BIC+AIC, `info/lz/` LZ76) but **the entire Shannon-coding canon — channel
capacity (BSC/BEC/Z/AWGN), Blahut-Arimoto, rate-distortion R(D),
water-filling, Huffman, arithmetic coding, ANS, Kraft-McMillan witness,
Fano's inequality, transfer entropy, info-bottleneck, MINE/InfoNCE/NWJ
variational MI, KSG estimator — is wholly absent** verified by zero matches
across the four packages. **Sixteen synergy primitives (S1-S16) totalling
~2380 LOC of pure connective tissue** close the gap; cheapest one-day PR is
**S1 BlahutArimotoCapacity (~180 LOC)** because the repo already has
log-domain Sinkhorn whose iteration structure is byte-for-byte the same
alternating KL-projection Arimoto-Blahut needs, plus closed-form BSC/BEC/Z
capacities are 8-line algebra each; **highest-leverage day-one unlock is S2
GaussianMIClosedForm + S3 ShannonHartleyCapacity (~120 LOC)** because every
Wiener / matched-filter / Kalman primitive shipped under `audio/separation/`
+ `signal/` consumer surfaces wants `0.5*log(1+SNR)` as a cap and the repo
literally has zero implementation of it.

---

## Bases — what each package exposes today

- `compression/entropy.go` (177 LOC, 041): `ShannonEntropy/JointEntropy/
  ConditionalEntropy/MutualInformation/KLDivergence/CrossEntropy` — **all
  bits**, no validation, no LSE guard. `coding.go` only RLE+Delta (no
  Huffman/arithmetic/ANS/Kraft). `quantize.go` only uniform scalar (no
  Lloyd-Max).
- `infogeo/fdiv.go` (228 LOC, 091): `KL/ReverseKL/JS/TV/Hellinger/χ²/
  Renyi` — **all nats**, strict `validatePair`. `bregman.go`: 5 generators.
  `mmd.go`: MMD² with Gaussian/Laplacian kernels.
- `info/mdl/`: NML multinomial (Kontkanen-Myllymäki 2007 linear), NML
  Bernoulli, BICShape, AICShape, UniversalIntegerCodeLength (Rissanen
  1983), SelectMDL. `info/lz/`: LZ76 + symbolisation.
- `prob/` (117): 7 distributions (PDF/CDF/Quantile), `MarkovSteadyState`
  power-iter, `KLDivergenceNumerical` trap-rule, copulae, conformal.
- `optim/transport/sinkhorn.go` (105): **log-domain alternating** dual
  update on f, g via LSE — the exact same structure Blahut-Arimoto needs.
- `autodiff/` (011): scalar reverse-mode tape with Exp/Log/Pow — substrate
  for MINE/NWJ/InfoNCE.

**No package imports another for an info-theoretic identity.**
`compression.MutualInformation` (bits, 2-D joint) and `infogeo.KL` (nats,
two simplex vectors) literally cannot be composed. No `info/channel/`,
`info/coding/`, or `info/dynamics/` directory.

---

## Conceptual unlocks (compressed)

1. **Channel capacity = max-MI alternating projection.** `C = max_{p_X}
   I(X;Y)` for fixed `W(y|x)` is Blahut-Arimoto alternation: (a) p(x|y) ∝
   p(x)W(y|x); (b) p(x) ∝ exp(Σ_y W(y|x) log p(x|y)). Same log-domain
   pattern as `optim/transport/sinkhorn.go:226-260`; only ~30 LOC of new
   code (the geometric-mean update).
2. **R(D) is the dual problem.** Same alternation, source ↔ reproduction
   swapped, same Csiszár-1974 convergence proof.
3. **Gaussian channel = water-filling.** `Pᵢ = max(0, ν−σᵢ²)` with `Σ Pᵢ
   = P` solved by bisection on ν (`optim/rootfind.go` ships it). Single-
   channel reduces to Shannon-Hartley `C = ½ log(1+SNR)`.
4. **Variational MI = autodiff + LSE.** Donsker-Varadhan (MINE 2018), NWJ
   (2010), InfoNCE (2018) are all plug-in critic expectations under
   `p(x,y)` vs `p(x)p(y)`; gradients flow through `autodiff.Tape`. Share
   common `LogSumExp`.
5. **Huffman = prefix-code witness for Kraft-McMillan.** `Σ 2⁻ℓᵢ ≤ 1`;
   Huffman achieves `H ≤ L < H+1` (Cover-Thomas 5.4.3). Repo has the
   lower bound (`ShannonEntropy`) but no implementation that achieves it.
6. **Fano = converse to channel coding.** `H(P_e) + P_e log(|X|−1) ≥
   H(X|Y)`. 6-line check; absent.
7. **Transfer entropy = conditional MI on lagged Markov pairs.** Schreiber
   2000: `TE_{Y→X} = I(X_{t+1}; Y_t^{(l)} | X_t^{(k)})`. Repo has
   `prob.MarkovSimulate` (paths) and `compression.MutualInformation`
   (joint matrices) but they don't speak the same data type — ~80 LOC
   bridges binned-TE.

---

## S1 — `BlahutArimotoCapacity` for discrete channels (~180 LOC)

`compression/channel.go` (new). Inputs: `W [][]float64` row-stochastic
`W[x][y] = P(Y=y|X=x)`. Output: `(C, pStar)`. Loop: (1) init p_X uniform;
(2) E-step `q(x|y) = p(x)W(y|x)/Σ_x' p(x')W(y|x')`; (3) M-step `p(x) ←
p(x)exp(Σ_y W(y|x) log q(x|y))/Z`; (4) track `I_t = Σ p(x)W(y|x)
log(q(x|y)/p(x))`, stop on `|ΔI| < tol`.

Closed-form pins for **R-MUTUAL 3/3** saturation:
- BSC(ε): `C = 1 − H₂(ε)`.
- BEC(ε): `C = 1 − ε`.
- Z-channel(ε): `C = log(1+(1−ε)·(1−ε)^(ε/(1−ε)))` (Csiszár-Körner Ex. 7.7).

Mirrors commits `6a55bb4` (audio-onset 3-detector) and `365368a` (Clayton-
autodiff-vs-analytic). LOC: ~80 BA + ~50 closed forms + ~50 tests.

## S2 — `GaussianMIClosedForm` (~50 LOC)

Jointly Gaussian (X,Y): `I = ½ log(|Σ_X||Σ_Y|/|Σ|)` nats. Univariate:
`I = −½ log(1−ρ²)`. Composes `linalg.LogDet` (shipped) with `prob`
covariance estimation. Golden vs MC sample-based MI.

## S3 — `ShannonHartleyCapacity` + parallel-Gaussian water-filling (~120 LOC)

`compression/awgn.go`. `C = B·log₂(1+S/N)` bits/sec. Parallel:
`WaterFillingPower(noiseVars, P) []float64` via bisection on ν using
`optim.Bisect`; capacity `½ Σ log₂(1+Pᵢ/σᵢ²)`. Pins: zero-noise →
+Inf, equal-noise → equal-power, single-channel reduces to S-H.

## S4 — `RateDistortionBA` and Gaussian water-filling (~280 LOC)

`compression/rate_distortion.go`. RD-BA alternation: (1) init q_Y uniform;
(2) `Q(y|x) ∝ q_Y(y)exp(−s·d(x,y))` for Lagrange `s ≥ 0`; (3) `q_Y(y) =
Σ_x p_X(x)Q(y|x)`; (4) sweep s to land on target D, trace R(D).

R-MUTUAL 3/3 pins: Bernoulli-Hamming `R(D) = H₂(p)−H₂(D)` for `D ≤
min(p,1−p)`; Gaussian-squared `R(D) = ½log(σ²/D)` for `D ≤ σ²`; multi-
Gaussian reverse-water-filling `Dᵢ = min(λ,σᵢ²), R = ½ Σ max(0,
log(σᵢ²/Dᵢ))`.

## S5 — `MutualInformationKSG` (Kraskov-Stögbauer-Grassberger 2004, ~250 LOC)

`info/mi/ksg.go`. Canonical continuous-MI estimator via brute-force kNN.
KSG-1: `Î = ψ(k) − ⟨ψ(n_x+1)+ψ(n_y+1)⟩ + ψ(N)`. Needs `prob.Digamma`
(also 117-T1.4, 153-S1; ~40 LOC). KSG-2 lower bias for strong dependence.
Pin: converges to closed-form Gaussian MI as N grows.

## S6 — Variational MI lower bounds: MINE, NWJ, InfoNCE (~280 LOC, ~80 each)

`info/mi/variational.go`. Three families, all expectation-of-critic
through `autodiff.Tape`:

- **MINE / Donsker-Varadhan** (Belghazi 2018):
  `I_DV ≥ E_pxy[T] − log E_pxpy[exp T]`. Bias-corrected EMA on log-mean-
  exp denominator (the Belghazi fix).
- **NWJ** (Nguyen-Wainwright-Jordan 2010):
  `I_NWJ ≥ E_pxy[T] − e⁻¹ E_pxpy[exp T]`. Tighter at small mismatch.
- **InfoNCE** (van den Oord 2018, contrastive):
  `I_NCE ≥ log K − E[−log softmax T(xᵢ,yᵢ)]`. Bounded by `log K` above
  (contrastive budget). Canonical SimCLR / CLIP estimator.

All three share `LogSumExp` (T1.12 of agent 087, currently
duplicated in `info/mdl/nml.go:142-156` and `optim/transport/sinkhorn.go:
229`). Promote it to `prob/mathutil.go` as part of S6.

## S7 — Information bottleneck (Tishby-Pereira-Bialek 1999, ~240 LOC)

`info/bottleneck.go`. Solves `min_{p(t|x)} I(X;T) − β I(T;Y)` via three-
way alternation: (1) `p(t|x) ∝ p(t) exp(−β KL(p(y|x)∥p(y|t)))`; (2) `p(t)
= Σ_x p(x)p(t|x)`; (3) `p(y|t) = Σ_x p(y|x)p(t|x)p(x)/p(t)`. SAME
alternating-KL-projection skeleton as BA and Sinkhorn (Tishby 1999 cited
Csiszár-1974 and Arimoto-1972). Pin both limits: `β→∞ → T=X`, `β→0 →
T⊥X`.

## S8 — `TransferEntropy` (Schreiber 2000, binned, ~80 LOC)

`info/dynamics/te.go`. `TE_{Y→X} = H(X_{t+1}|X_t^{(k)}) − H(X_{t+1}|
X_t^{(k)},Y_t^{(l)})`. Composes `prob.MarkovSimulate` (paths) × bin × 
`compression.JointEntropy`. Pins: Y⊥X → TE≈0; `X_{t+1}=f(Y_t)+noise` →
TE > 0; chain X→Y→Z → `TE_{X→Z|Y}=0`. KSG-TE is 087-T3.1.

## S9 — Huffman, arithmetic, ANS + Kraft-McMillan (~280 LOC)

`compression/coding_optimal.go`. Three coders (Huffman 1952 heap ~120,
Witten-Neal-Cleary 1987 arithmetic ~100, Duda 2013 ANS ~80) plus
`KraftMcMillanWitness` (~30): `Σ 2⁻ℓᵢ ≤ 1` test. Repo currently has the
Shannon lower bound but no implementation that achieves it. R-MUTUAL pin:
3 coders round-trip identically; expected length within `H ≤ L < H+1`.

## S10 — `FanoInequality` witness (~30 LOC)

`info/fano.go`. Confusion matrix → `(LHS, RHS, holds)`. `H(P_e) +
P_e log(|X|−1) ≥ H(X|Y)`. Trivial composition; absent today. Strict form
gives classifier error-rate lower bound from MI.

## S11-S16 — concise

- **S11 DPI witness** (~50). `info/dpi.go`. For X→Y→Z verify
  `I(X;Z) ≤ I(X;Y)` AND `I(X;Z) ≤ I(Y;Z)` with slack diagnostic.
- **S12 ChainRuleMI witness** (~50). `I(X; Y₁..Yₙ) = Σ I(X;Yᵢ|Y_<ᵢ)`.
  Needs conditional-MI primitive (binned ~30, KSG-conditional is 087-T2.4).
- **S13 Fisher↔MI cross-link** (~80). Bridges 153-S2 with S2 here; for
  local deviation `I(θ;X) ≈ ½ δθᵀ F(θ₀) δθ`. Pin against analytic Beta-
  binomial Fisher.
- **S14 Lloyd-Max** (~150). `compression/quantize_lloyd.go`. Alternate
  (a) `bᵢ = ½(c_{i-1}+cᵢ)`, (b) `cᵢ = E[X|X∈[b_{i-1},bᵢ]]` via
  `prob.Distribution.CDF` + bisection. Pin to Max 1960 tabulated values.
- **S15 SampleComplexityFromMI** (~80). PAC bound `N ≥ (1/ε²)(d +
  log(1/δ))/ΔI` composing `prob.HoeffdingBound` × `compression.MI`.
- **S16 EntropyRate** (~80). `info/dynamics/entropy_rate.go`. For
  irreducible aperiodic chain: `h_μ = -Σ πᵢ Σ Pᵢⱼ log Pᵢⱼ = H(X₂|X₁)`.
  Composes `prob.MarkovSteadyState` (shipped) with row-wise
  `compression.ShannonEntropy`.

---

## LOC summary

| Primitive | LOC | Cross-edges | Day-1 ready |
|---|---|---|---|
| S1 BlahutArimotoCapacity (BSC/BEC/Z) | 180 | optim/transport-pattern reuse | YES |
| S2 GaussianMIClosedForm | 50 | linalg.LogDet | YES |
| S3 ShannonHartleyCapacity + water-fill | 120 | optim.Bisect | YES |
| S4 RateDistortionBA + Gaussian water-fill | 280 | S1 skeleton | YES |
| S5 MutualInformationKSG | 250 | needs prob.Digamma (40 LOC) | NO (digamma) |
| S6 MINE+NWJ+InfoNCE variational | 280 | autodiff.Tape | YES (autodiff lands) |
| S7 InformationBottleneck | 240 | infogeo.KL | YES |
| S8 TransferEntropy (binned) | 80 | prob.MarkovSimulate + compression.JointEntropy | YES |
| S9 Huffman+Arithmetic+ANS+Kraft | 280 | none new | YES |
| S10 FanoInequality | 30 | compression.ShannonEntropy | YES |
| S11 DPI witness | 50 | compression.MutualInformation | YES |
| S12 ChainRuleMI witness | 50 | conditional-MI helper | YES |
| S13 FisherInfoVsMI cross-link | 80 | 153-S2 + S2 here | NO (153-S2) |
| S14 Lloyd-Max quantizer | 150 | prob.Distribution.CDF | YES |
| S15 SampleComplexityFromMI | 80 | prob bounds | YES |
| S16 EntropyRate of Markov | 80 | prob.MarkovSteadyState + compression.ShannonEntropy | YES |
| **Total** | **2280** | | **13 of 16** |

Plus shared `LogSumExp` promotion (T1.12 of agent 087): ~30 LOC + ~30 LOC
tests = ~60 LOC overhead, removes one duplication site in
`info/mdl/nml.go:142-156` and one in `optim/transport/sinkhorn.go:229`.

---

## Recommended PR sequence (~9 engineer-days, 6 PRs)

**PR-1 (1 day, ~360 LOC):** `LogSumExp` promotion + S1 BA capacity + S3
Shannon-Hartley + S10 Fano + S11 DPI. Saturates the first
R-MUTUAL-CROSS-VALIDATION 3/3 pin (BSC closed-form × BEC closed-form ×
BA numerical agree to 1e-10) — one-day standalone with maximum lever.

**PR-2 (1 day, ~280 LOC):** S9 Huffman + Arithmetic + ANS + Kraft witness.
Closes the headline gap "compression/ ships RLE+Delta but no optimal
prefix code"; second R-MUTUAL pin (3 coders losslessly round-trip).

**PR-3 (1 day, ~330 LOC):** S2 Gaussian-MI + S4 RD-BA + Gaussian water-
filling. Crown jewel for SNR-driven consumer apps. Third R-MUTUAL pin
(Bernoulli-RD closed × Gaussian-RD closed × BA numerical).

**PR-4 (2 days, ~240 LOC):** S7 Information bottleneck + S8 binned
transfer entropy + S16 entropy rate. Pulls `prob.MarkovSteadyState` into
info-theoretic citizenship for the first time.

**PR-5 (2 days, ~290 LOC):** S5 KSG MI (preceded by `prob.Digamma`
~40 LOC) + S6 MINE+NWJ+InfoNCE variational. The continuous-MI deliverable
that NPEET, JIDT, and the MINE/CPC ML literature all want. Depends on
`autodiff.Tape` (already shipped).

**PR-6 (2 days, ~310 LOC):** S12 + S13 + S14 + S15. Polish: chain rule,
Fisher↔MI cross-link, Lloyd-Max, sample-complexity. Closes the loop with
153 (prob-infogeo) so Fisher matrix from a `prob.Distribution` is one
step from MI under local-deviation analysis.

---

## Architectural placement

Mirrors the consumer-side-placement precedent set by 158 (image/), 159
(em/wave/), 160 (fluids/), 165 (sequence/), 166-167 (acoustics+audio),
168 (chaos+physics+autodiff), 169 (prob+optim) — **15 consecutive synergies
confirming the pattern**. Three new sub-packages plus one root file:

```
info/
  channel/          (S1 BSC/BEC/Z + BA + S3 Shannon-Hartley + S10 Fano)
  rate_distortion/  (S4 RD-BA + Gaussian water-filling + S14 Lloyd-Max)
  mi/               (S2 Gaussian-MI + S5 KSG + S6 MINE/NWJ/InfoNCE)
  dynamics/         (S7 IB + S8 TE + S16 entropy rate)
  fano.go           (root: S10 Fano)  ← OR keep in channel/
  dpi.go            (root: S11 DPI + S12 chain rule)

compression/
  coding_optimal.go (S9 Huffman + Arithmetic + ANS + Kraft)
  quantize_lloyd.go (S14 Lloyd-Max)
  awgn.go           (S3 if not in info/channel/)
```

Cross-edges added (cycle-free, all one-direction):
- `info/channel/` → `infogeo/` (KL projection guard) +
  `optim/transport/-pattern` (no import — same skeleton, different file)
- `info/rate_distortion/` → `info/channel/` (BA skeleton) + `prob/`
  (Distribution.CDF for Lloyd-Max)
- `info/mi/` → `prob/` (Digamma) + `autodiff/` (variational critic) +
  `linalg/` (LogDet)
- `info/dynamics/` → `prob/` (MarkovSteadyState/Simulate) +
  `compression/` (entropy primitives)
- `compression/coding_optimal.go` → none new (pure standalone)

DAG verification: no reverse path. `prob/` does NOT import `info/` or
`compression/` (and shouldn't — prob is closer to the root). `infogeo/`
does NOT import `info/`. All edges go `consumer → primitive`.

---

## Precision and numeric hazards

- **bits vs nats** — agent 087-C1 flagged. Recommend **nats canonical** in
  all new `info/` files, with `BitsToNats = 1/log(2)` and `NatsToBits =
  log(2)` constants exported from `constants/`. `compression/entropy.go`
  is bits and pre-existing; document the wart.
- **BA convergence proof** — Csiszár 1974: alternating I-projection
  decreases MI strictly until fixed point. For tol = 1e-10, ~50 iterations
  suffice on well-conditioned channels; emit `ErrBlahutArimotoNonConv` on
  iter > 1000.
- **InfoNCE saturation** — bounded by `log K`; flag in docstring that
  estimating MI > log(batch_size) is impossible.
- **MINE bias** — Belghazi's EMA fix on log-mean-exp denominator must
  apply, else gradient is biased upward. Pin in test against analytic
  Gaussian-MI ground truth.
- **KSG noise floor** — kNN distances tie at 0 collapse digamma; add
  jitter `+ε·N(0,1)` with ε = 1e-10 (the Kraskov 2004 standard fix).
- **Huffman ties** — sibling-merge ordering matters for golden-file
  parity; canonical convention is min-frequency-first, ties broken by
  insertion order.
- **water-filling** — bisection on ν may fail to bracket if all noise
  variances are equal; degenerate-case fast-path returns equal allocation.
- **Lloyd-Max** — for unbounded sources (Gaussian) bin boundaries diverge
  asymptotically; cap by `±6σ` (the standard truncation).

---

## R-pattern saturations earned

This synergy lands **three new R-MUTUAL-CROSS-VALIDATION 3/3 pins**, each
mirroring the `6a55bb4` audio-onset 3-detector and `365368a` Clayton-
autodiff-vs-analytic idioms:

1. **Channel capacity** — BSC closed `1−H₂(ε)` × BEC closed `1−ε` × BA
   numerical. PR-1.
2. **Lossless coding** — Huffman optimal × Arithmetic Witten-Neal-Cleary ×
   ANS Duda; round-trip identity + expected-length within Shannon ≤ L < H+1.
   PR-2.
3. **Rate-distortion** — Bernoulli-RD `H₂(p)−H₂(D)` × Gaussian-RD `½ log
   σ²/D` × BA-RD numerical. PR-3.

Plus an **R-CLOSED-FORM-PINNED-TO-AUTODIFF expansion** in PR-5: MINE / NWJ
/ InfoNCE numerical × analytic Gaussian-MI ground truth. Mirrors the 169-
S5 EM-GMM × k-means × optim.GA pattern.

---

## What is intentionally NOT in scope

- **Polar codes, LDPC, turbo, RS** — agent 087 T4 (not topic-checklist).
- **Slepian-Wolf, Wyner-Ziv, multi-terminal** — networking; defer to v2.
- **IIT 4.0 Φ** — agent 087 T3.9 (separate sub-package, ~700 LOC).
- **Williams-Beer PID** — agent 087 T3.6, ~450 LOC sub-package.
- **Solomonoff prior, Wallace MML** — flagged v2 in `info/mdl/doc.go:96-
  108`.
- **Sliced Wasserstein** — agent 087 T3.11; lives in `optim/transport/`.

---

## Distinct from prior agents

041-045 (compression isolation, names Huffman/arithmetic gap THIS review
composes with prob for Kraft-McMillan); 086-090 (info isolation, 087
enumerated 30+ primitives THIS picks synergy-only subset reusing 087
LSE/Digamma/KSG flagging); 091-095 (infogeo isolation, THIS reuses
`infogeo.KL` as I-projection without modifying); 116-120 (prob isolation,
117 LogPDF/Digamma debt consumed by S5+S13); 151 signal-prob (orthogonal,
shares LSE promotion); 153 prob-infogeo (S2 cross-linked via S13); 155
crypto-prob (orthogonal, `H_∞` Rényi-∞ limit could share); 161 control-
prob, 162 graph-prob, 165 sequence-prob (orthogonal; NCD via
`info/lz.LempelZivComplexity` could be S17); 163 optim-autodiff (S6
variational MI consumes that infrastructure); 169 prob-optim (orthogonal,
shares R-MUTUAL pin idiom).

---

## Bottom line

`reality/` ships every numerical primitive Shannon's 1948 paper needs
(`compression.ShannonEntropy`, `infogeo.KL`, alternating-LSE-projection in
`optim/transport/sinkhorn.go`, `prob.MarkovSteadyState`, `linalg.LogDet`)
but composes none of them into channel capacity, rate-distortion, IB, TE,
Huffman/Arithmetic/ANS, Fano, DPI, Lloyd-Max, KSG, MINE-NWJ-InfoNCE, or
entropy rate. Six PRs, ~2280 LOC, three R-MUTUAL-CROSS-VALIDATION 3/3
pins, zero new external deps. The Shannon canon is an architectural debt
explicitly named by `info/mdl/doc.go:38-42` ("L10 capacity-bound packages
— when those land"). PR-1 = highest leverage-per-LOC remaining in the
package matrix.

End report.
