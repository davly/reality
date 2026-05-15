# 165 | synergy-sequence-prob — sequence × prob: HMMs, profile HMMs, alignment statistics

`sequence/` ships pure deterministic string tools (Levenshtein/DL/Hamming, Jaro-Winkler, NW/SW, LCS/LCSubstr, n-grams, Soundex, TokenSetRatio, Dice, Shingling). `prob/` ships PDF/CDF/quantile, log-gamma + regularized-incomplete-{beta,gamma}, t/χ², Fisher-exact, Markov{SteadyState,Simulate}, ARIMA, conformal. Neither package speaks to the other today; the missing layer is *probabilistic sequence inference* — HMMs, profile HMMs, pair HMMs, Karlin-Altschul significance — which is exactly the (sequence ⊗ prob) cross-product, ~1300 LOC of pure connective tissue with zero new constants.

---

## 0. Surface roster (the cross-product is empty today)

`sequence/` (15 entry points; agent 127 catalogue): edit-distance and alignment scores, but **nothing returns a probability**. `NeedlemanWunsch`/`SmithWaterman` accept (match, mismatch, gap) `float64` scores — i.e. they already accept what would be log-odds emissions if a caller chose to interpret them that way, but the package has no log-domain APIs.

`prob/` has `MarkovSteadyState` (n-state stochastic matrix, power iteration to fixed point) and `MarkovSimulate` (LCG-driven trajectory sampling). These are state-only — there is no notion of an *emission* distribution conditional on the latent state, which is the one missing piece between a Markov chain and an HMM.

`changepoint/bocpd.go` already implements a pairwise `logSumExp(a, b float64) float64` (file ll.294-305, 12 LOC) and works the run-length filter entirely in log-space. **It is unexported.** Promoting it to `prob/mathutil.go` (alongside `LogGamma`, `RegularizedBetaInc`) is the prerequisite for every entry below; agent 117's T1.4 already names `logsumexp` as missing.

---

## 1. Hidden Markov Models — forward / backward / Viterbi / Baum-Welch

**Capability.** Discrete-state, discrete- or Gaussian-emission HMM with closed-form inference and EM training. Underpins POS-tagging, speech-recognition GMM-HMM baselines, gene-finding (GENSCAN/GeneMark style), DNA-methylation segmentation, sleep-stage classification, anomaly detection. The four-tuple λ = (π, A, B) drives `Forward`, `Backward`, `Viterbi`, `BaumWelch`.

**Composition.** Pure rebind of existing primitives:
- π, A → `prob.MarkovSteadyState` already validates a row-stochastic matrix; reuse the `[]float64`-flat-row-major shape (consistency with `MarkovSimulate`). π is just a length-N vector.
- B (emission) → for discrete: an N×M flat slice. For Gaussian: an `[]prob.NormalDist` of length N (the `Distribution` interface from `distribution.go` is exactly the right abstraction; users plug in `NormalDist` / `BetaDist` / `ExponentialDist` without code change).
- Forward/backward recursions in log-space → need exactly `logSumExp` and `math.Log(A[i,j])`. No new math.
- Viterbi → max-plus algebra; doesn't need `logSumExp`, only `math.Log` of A and B.
- Baum-Welch (EM) → ξ_t(i,j) = αᵗᵢ · Aᵢⱼ · Bⱼ(oₜ₊₁) · βᵗ⁺¹ⱼ / P(O); all reductions are `logSumExp` chains, M-step normalisations are vector sums.

**API sketch:**
```go
type HMM struct {
    N    int             // states
    M    int             // discrete emissions (0 if continuous)
    Pi   []float64       // initial, length N (or LogPi if log-space)
    A    []float64       // N*N transition (row-major, like prob.MarkovSimulate)
    B    []float64       // N*M emission (discrete case)
    Emit []prob.Distribution // length N (continuous case; nil if discrete)
}

func (h *HMM) Forward(obs []int)   (logAlpha []float64, logProb float64) // N*T flat
func (h *HMM) Backward(obs []int)  (logBeta  []float64)
func (h *HMM) Viterbi(obs []int)   (path []int, logProb float64)
func (h *HMM) BaumWelch(obs []int, maxIter int, tol float64) (iters int, llHistory []float64)
func (h *HMM) PosteriorDecode(obs []int) (gamma []float64) // forward-backward γ
```

**LOC budget.** ~120 + 90 + 70 + 180 + 60 = **520 LOC** total inside a new `prob/hmm.go`. Pure connective tissue: nothing leaves the standard library + already-shipping `prob` primitives.

**Where it lives.** `prob/` not `sequence/` — the math is Markov + emissions, no edit-distance involvement. But `sequence/alignment.go`'s NW/SW machinery is the sibling DP; share helpers (`maxFloat` already present, add a `logSumExpVec`).

---

## 2. Log-space arithmetic — the prerequisite that gates all of §1-§4

**Capability.** Underflow-safe addition of probabilities expressed as logs. Without this every HMM forward of length T > ~700 underflows to 0 even for benign emission distributions.

**Composition.** *Already implemented inside `changepoint/bocpd.go`.* It is the canonical 12-line:
```go
if math.IsInf(a, -1) { return b }
if math.IsInf(b, -1) { return a }
if a > b { return a + math.Log1p(math.Exp(b-a)) }
return b + math.Log1p(math.Exp(a-b))
```
Move it to `prob/mathutil.go` as `LogSumExp(a, b float64) float64`, add the vector form `LogSumExpVec(xs []float64) float64` (Kahan-style with explicit max), and the streaming `LogSumExpAdd(running, x float64) float64`. **Connective tissue: 30 LOC.**

`changepoint/bocpd.go` should `import . "prob"` (or the named form) and drop its private copy. Agent 117 T1.4 names this gap; this task fulfils it and unlocks everything below.

Companion gaps that should land in the same commit because they are forced by the same numerical-stability calculus:
- `Log1mExp(x float64) float64` = log(1 - exp(x)), the tail-stable companion. Computed via `if x > -ln(2): math.Log(-math.Expm1(x)); else: math.Log1p(-math.Exp(x))` (Mächler 2012). 8 LOC.
- `Log1pExp(x float64) float64` = log(1 + exp(x)), the soft-plus, used pervasively in CRFs. 10 LOC. (Today reality has `LogOddsToProb` which is the *sigmoid*, not soft-plus.)
- `LogDotExp(logA, logB []float64) float64` = log(Σ exp(aᵢ + bᵢ)) used in HMM forward inner loops.

Total log-space toolbelt: **~80 LOC** in `prob/mathutil.go`.

---

## 3. Profile HMMs (Eddy 1998, HMMER's data structure)

**Capability.** Multiple-sequence-alignment-shaped HMM with three states per column (Match Mₖ, Insert Iₖ, Delete Dₖ) plus Begin/End. Models a *family* of sequences, not a fixed pair. The crown-jewel object of computational biology since 1998; HMMER and Pfam are built on it. Use case in reality: protein-family/sequence-family classification of any token-stream where the "alphabet" is finite — log token streams, command sequences, control-flow traces.

**Composition.** A profile HMM of length L is just three coupled HMMs from §1 plus a tied-parameter Baum-Welch:
- L+1 Match emission distributions (each over the alphabet Σ; reuse `prob.Distribution` if continuous, an `[]float64` otherwise).
- L+1 Insert emission distributions (typically tied to a background distribution, scoring 0 in log-odds).
- 9 transition-type entries per node (M→M, M→I, M→D, I→M, I→I, D→M, D→D, plus B→ and →E).
- Inference: the same forward/backward/Viterbi as §1 with the per-column transition restriction. Eddy's paper presents the recursion in 12 lines; the code is ~250 LOC because the indexing is delicate, not because it adds math.

**API sketch:**
```go
type ProfileHMM struct {
    L int                     // model length
    M []prob.Distribution     // L+1 match emissions
    I []prob.Distribution     // L+1 insert emissions
    T []float64               // (L+1)*9 transition log-probs
}

func TrainProfileHMM(family [][]int, L int, pseudocount float64, maxIter int) *ProfileHMM
func (p *ProfileHMM) ViterbiAlign(seq []int) (path []State, logProb float64)
func (p *ProfileHMM) Forward(seq []int) float64 // log P(seq | model)
```

**LOC budget.** ~250 inference + 120 EM + 60 pseudocount/Dirichlet-prior smoothing = **430 LOC**. Pseudocounts compose with `prob.BetaDist` / a future `DirichletDist` (agent 117 T1.1 names that as missing — this synergy is one motivator).

**Lives in.** `sequence/profile_hmm.go` because the consumer-side semantics (alignment of biological-style sequences) is squarely in `sequence`'s remit, but the inference layer imports `prob`.

---

## 4. Pair HMMs — the probabilistic shadow of NW/SW

**Capability.** Three-state HMM (M, X-gap-in-A, Y-gap-in-B) over the *joint* sequence pair; gives a *probabilistic* pairwise alignment. Durbin-Eddy-Krogh-Mitchison §4. Returns the posterior probability of every alignment column ("posterior decoding" / Maximum Expected Accuracy alignment, BAli-Phy / ProbCons / FSA all use it). Strict generalisation of NW: NW is the Viterbi path of a pair HMM under a specific scoring scheme.

**Composition.** Pair HMMs are exactly NW/SW with the "+ score" replaced by "× probability", carried in log-space. Concretely:
- The DP cell stores `(logFwdM, logFwdX, logFwdY)` instead of one score.
- Recursion: `logFwdM[i,j] = logEmit_M(a[i],b[j]) + LogSumExp{logFwdM[i-1,j-1] + logTau_MM, logFwdX[i-1,j-1] + logTau_XM, logFwdY[i-1,j-1] + logTau_YM}` etc.
- NW today is **`alignment.go` ll.22-88, 67 LOC of DP + traceback**. Pair HMM forward is the same shape, ~110 LOC. Pair HMM backward is its mirror.

**API sketch:**
```go
type PairHMM struct {
    Tau    [9]float64   // 3x3 transition logs (M, X, Y) - row major
    EmitM  func(a, b rune) float64 // log emission of (a,b) pair
    EmitX  func(a rune) float64    // log emission of a in gap state
    EmitY  func(b rune) float64
}

func (p *PairHMM) Forward(a, b string)  (logProb float64, fwd []float64)
func (p *PairHMM) Backward(a, b string) (bwd []float64)
func (p *PairHMM) ViterbiAlign(a, b string) (alignA, alignB string, logProb float64)
func (p *PairHMM) PosteriorMatrix(a, b string) []float64 // P(aligned i↔j | a,b)
func (p *PairHMM) MEAAlign(a, b string) (alignA, alignB string) // max-expected-accuracy
```

**LOC budget.** Forward + backward + posterior matrix + Viterbi + MEA traceback = **~330 LOC**. The MEA traceback is itself a Needleman-Wunsch on the posterior-marginal matrix (re-uses the *existing* NW `dp` skeleton verbatim — third-time-payoff for `alignment.go`).

**Wire-up.** `sequence/pair_hmm.go` next to `alignment.go`. Imports `prob` for `LogSumExp` and `LogSumExpVec`. The connective tissue is *literally three lines* per cell vs the existing `maxFloat` calls in NW — one of the highest synergy-per-LOC opportunities in the catalog.

---

## 5. Karlin-Altschul statistics (BLAST / Gumbel)

**Capability.** Compute E-value, bit-score, p-value of an alignment score under random-string null. The single most-used inferential layer in biology and one of the most cited results in computer science (Karlin-Altschul 1990; the "log-only" half of the BLAST-cited corpus). Output of every BLAST search.

**Composition.**
- Gumbel CDF: F(s) = exp(-K·m·n·exp(-λs)). Closed form — needs `math.Exp` and `math.Log` only.
- E-value: E = K·m·n·exp(-λs). One multiplication + one exp.
- Bit-score: S' = (λS - lnK) / ln2.
- p-value from E-value: p = 1 - exp(-E) (Poisson approximation, valid when m·n is large). One `math.Expm1`.

The constants λ and K must be fitted to the score matrix + gap-penalty regime. Three approaches, in order of complexity:
1. **Pre-tabulated** for canonical regimes (BLOSUM62 + (-11, -1) gap, etc.). 6 LOC, but limits applicability.
2. **Statistical estimation** (`EstimateLambdaK(scoreMatrix, qFreq, dbFreq) (lambda, K float64)`) — solve Σᵢⱼ qᵢ·dⱼ·exp(λ·sᵢⱼ) = 1 for λ via bisection (already in reality: `optim/bisection`!), then compute K via the relative-entropy formula. **~80 LOC** total because root-finding is borrowed.
3. **Empirical fit** by Monte-Carlo: score ~10⁵ random pairs, MLE Gumbel via method-of-moments or Newton. **~120 LOC**, but composes with future `prob` Gumbel sampler (agent 117 T1.2 names Gumbel as missing — building the fitter forces the distribution).

**API sketch:**
```go
// In sequence/karlin_altschul.go:
func GumbelEValue(score, lambda, K float64, m, n int) float64
func GumbelBitScore(score, lambda, K float64) float64
func GumbelPValue(eValue float64) float64
func EstimateKarlinAltschul(scoreMatrix [][]float64, queryFreq, dbFreq []float64) (lambda, K float64, ok bool)

// In prob/gumbel.go (forced sibling):
func GumbelPDF(x, mu, beta float64) float64
func GumbelCDF(x, mu, beta float64) float64
func GumbelQuantile(p, mu, beta float64) float64
```

**LOC budget.** Karlin-Altschul layer **~150 LOC**, Gumbel distribution proper **~80 LOC** (it is one of the missing distributions agent 117 T1.2 named anyway). Total **~230 LOC** but ~80 of that is debt being paid off.

**Connective composition:** `EstimateKarlinAltschul` *must* call `optim.Bisection` (already shipping). This is the rare synergy that bridges three packages: sequence × prob × optim.

---

## 6. Levenshtein distribution under random strings (concentration bounds)

**Capability.** Given two iid uniform-random strings of length n over alphabet Σ, what is the distribution of `LevenshteinDistance`? This is the *null hypothesis* for any string-similarity threshold; without it, the choice of "is JaroWinkler > 0.85 significant?" is a heuristic.

**Composition.** Two paths:
- **Closed-form mean** (Chvátal-Sankoff 1975 constant γ_|Σ|): `E[LCS(X,Y)/n] → γ_|Σ|` as n → ∞. For Σ=2, γ ≈ 0.8118; for Σ=4 (DNA), γ ≈ 0.6602; for ASCII Σ=128, γ ≈ 0.108. Tabulate the canonical values + bilinear interpolation. **~40 LOC** + literature constants.
- **Concentration bound**: by Azuma-McDiarmid, `P(|d(X,Y) - E[d]| > t) ≤ 2·exp(-2t²/(2n))` so `t = sqrt(n·log(2/α)/2)` is a confidence half-width. Already trivially expressible with `math.Sqrt + math.Log`. **~20 LOC**.
- **Monte-Carlo p-value**: sample k random pairs, count how many score ≥ observed. Generic — **~60 LOC** with a callback for the metric.

**API sketch:**
```go
func LevenshteinPValueAzuma(observed, n int, alphabet int) float64
func LevenshteinNullMean(n int, alphabet int) float64                 // Chvátal-Sankoff scaled
func LevenshteinPValueMonteCarlo(observed, n int, alphabet int,
    trials int, rng *prob.Rng) float64
func JaroWinklerPValueMonteCarlo(observed float64, lenA, lenB int,
    alphabet int, trials int, rng *prob.Rng) float64
```

**LOC budget.** **~150 LOC**. Note this requires a `prob.Rng` interface that doesn't exist yet — agent 117 T1.3 names samplers as the gating gap. Build the minimum: a `prob.SplitMix64` PRNG (~30 LOC, deterministic, reproducible, golden-file friendly) — and every other sampling primitive in this synergy stack inherits it.

---

## 7. Conditional Random Fields — the undirected sibling

**Capability.** Linear-chain CRF: discriminative cousin of HMM. P(y | x) ∝ exp(Σ_t Σ_k λ_k · f_k(yᵗ⁻¹, yᵗ, x, t)). Standard for sequence labelling (NER, chunking, segmentation, OCR post-processing). McCallum / Lafferty 2001.

**Composition.** Forward/backward in CRF is *identical to HMM forward/backward in log-space* — only the local factors are unnormalised exp(features · weights) instead of A[i,j]·B[j,oₜ]. So once §1 is built, linear-chain CRF inference is **essentially a thin re-skin** that swaps the local-factor lookup. The training loop is L-BFGS on the log-likelihood — `optim.LBFGS` already exists. Gradient computation reuses CRF forward-backward for the expected-counts, mirror of the HMM E-step.

**API sketch:**
```go
type LinearChainCRF struct {
    NLabels int
    Weights []float64                                // packed feature weights
    Feature func(yPrev, y int, x []float64, t int) []float64 // user callback
}

func (c *LinearChainCRF) Forward(seq [][]float64) (logZ float64, logAlpha []float64)
func (c *LinearChainCRF) Viterbi(seq [][]float64) (labels []int, logProb float64)
func (c *LinearChainCRF) Train(seqs [][][]float64, labels [][]int, l2 float64, maxIter int)
```

**LOC budget.** ~250 LOC reusing `prob.LogSumExp` from §2 and `optim.LBFGS` (which agent 163 reviews and reality already ships). Heaviest cost is feature-templating ergonomics, not math.

---

## 8. Particle filter — sequential Monte-Carlo for sequences

**Capability.** Sequential importance resampling for state-space models with arbitrary likelihood. Generalises the HMM forward pass to non-Gaussian, non-linear emissions. Used everywhere from object tracking to ABC posterior in genealogy reconstruction.

**Composition.** Pure re-bind:
- N particles each holds a state; the proposal/transition is a callback `func(stateᵗ⁻¹) stateᵗ` (samples from p(xᵗ | xᵗ⁻¹)).
- The likelihood is a callback `func(state, obsᵗ) float64` returning `log p(obsᵗ | state)`.
- Resampling step uses systematic resampling on cumulative-weight thresholds (~25 LOC). Effective Sample Size = `1 / Σ wᵢ²` triggers resampling when ESS < N/2 (already implemented in `prob/conformal/adaptive.go` — `EffectiveSampleSize`! literally the same arithmetic, modulo the weighting form). Reuse it.
- No new probability theory — all reductions are weighted means and `LogSumExp`.

**API sketch:**
```go
type ParticleFilter struct {
    N         int
    States    []float64       // N x stateDim, flat
    LogWeights []float64
    Transition func(state []float64, rng *prob.Rng) []float64
    LogLik     func(state []float64, obs []float64) float64
}

func (pf *ParticleFilter) Step(obs []float64, rng *prob.Rng)
func (pf *ParticleFilter) Mean() []float64
func (pf *ParticleFilter) ResampleSystematic(rng *prob.Rng)
```

**LOC budget.** **~180 LOC**. Lives in `prob/particle.go`; not strictly a sequence-package piece, but it is the missing companion to `MarkovSimulate` and prerequisite for "sequence-of-observations posterior".

---

## 9. Variable-order Markov / PPM

**Capability.** Variable-order Markov models (VOM, Begleiter-El-Yaniv-Yona 2004) and Prediction-by-Partial-Matching (Cleary-Witten 1984) generalise the fixed `prob.MarkovSteadyState` n-gram. Used in compression (PPMd in 7-Zip), bioinformatics taxonomy (Phymm), and as on-line predictors. Adaptive context length is the key feature.

**Composition.**
- N-gram extraction → `sequence.NGrams` already does this verbatim (l.15-26). Reuse.
- Suffix tree for O(n)-time variable-order construction → not currently in `sequence/` (agent 127 names it as a Tier-1 missing), but fits inside this synergy.
- Probability estimation with Kneser-Ney / Witten-Bell smoothing → pure arithmetic on `prob` primitives (mostly division and `math.Log`).

**API sketch:**
```go
type VOM struct { /* trie, smoothing params */ }
func TrainVOM(corpus []int, maxOrder int, smoothing Smoothing) *VOM
func (v *VOM) Predict(context []int) []float64    // P(next | context)
func (v *VOM) LogProb(seq []int) float64          // log P(seq | model)
func (v *VOM) Sample(context []int, n int, rng *prob.Rng) []int
```

**LOC budget.** ~250 LOC + suffix-tree dependency (agent 127 §1.6/1.7 names McCreight/Ukkonen suffix-tree, ~250 LOC of its own). Ship a bounded-order trie variant first (~100 LOC) and upgrade to suffix-tree later.

---

## 10. MCMC over alignments / Bayesian alignment

**Capability.** Sample alignments from the posterior P(alignment | sequences, pair HMM) — used by BAli-Phy, MrBayes, StatAlign for phylogenetic-aware alignment. Strict superset of `PairHMM.PosteriorMatrix` (which gives marginals, not joint samples).

**Composition.** Metropolis-Hastings on the alignment lattice:
- Proposal: pick a column, swap to one of the three local moves (advance i, advance j, advance both). 30 LOC.
- Acceptance ratio: ratio of pair HMM joint forward probabilities under proposed vs current → uses the `PairHMM.Forward` from §4 in restricted form. 40 LOC.
- The overall MCMC machinery itself is generic and should land in `prob/mcmc.go` (agent 117 T1.3 names `Metropolis-Hastings` as missing). ~150 LOC for a generic MH driver, plus 60 LOC for the alignment-specific proposal/density.

**LOC budget.** ~250 LOC, of which the generic MH driver (~150) is reusable for everything else (Bayesian regression, etc).

---

## 11. Suffix-array / suffix-automaton statistics

**Capability.** Karkkainen-Sanders DC3 or SA-IS suffix array → background distribution of substring frequencies, k-mer abundance Z-scores, and "is this k-mer enriched?" tests. Used in ChIP-seq peak motif analysis, plagiarism detection, exact-repeat census.

**Composition.** Suffix array is pure-`sequence` (agent 127 §1.6 names this as a Tier-1 missing, ~300 LOC for SA-IS). Once it exists, the statistics layer is:
- k-mer count per position → linear scan of LCP array, ~30 LOC.
- Background k-mer distribution → can be modelled as `prob.PoissonPMF` (agent 117 has it) under a homogeneous-Poisson null, or `BinomialPMF` for fixed-length text.
- Enrichment Z-score: `(observed - n·p̂) / sqrt(n·p̂·(1-p̂))` then `prob.NormalCDF` for p-value. ~40 LOC.

**LOC budget.** ~70 LOC of synergy code (suffix-array itself is ~300 LOC already counted in agent 127).

---

## 12. Edit distance under random source-channel model

**Capability.** Given sender emits string under iid channel-noise (substitution prob ε_sub, insertion ε_ins, deletion ε_del), what is `P(edit_distance = k)`? This is the *generative* probability model whose negative log-likelihood is exactly the score that `LevenshteinDistance` computes. Models OCR errors, typo distributions, sequencing errors. Mitzenmacher 2009.

**Composition.** Same DP shape as `LevenshteinDistance` (`distance.go` ll.24-58, 35 LOC) with `min` replaced by `LogSumExp` and unit costs replaced by log-channel-probabilities. The connective tissue is literally:
```
old: curr[i] = min3(prev[i]+1, curr[i-1]+1, prev[i-1]+cost)
new: curr[i] = LogSumExp3(prev[i]+logEpsDel, curr[i-1]+logEpsIns, prev[i-1]+logCharLik)
```
35 LOC of `distance.go` becomes ~50 LOC of `LevenshteinPosterior` in `sequence/random_channel.go`.

**LOC budget.** ~120 LOC: log-likelihood + posterior + Viterbi-edit-script-decode + a small calibration helper. This is the **highest-density** synergy piece — every line of new code reuses the existing DP scaffolding verbatim.

---

## 13. What this means for `prob.Distribution` and `Markov*`

The single largest design-pressure these synergies put on existing code:

1. **`prob.MarkovSimulate` should accept a `*prob.Rng`.** Today it has a hard-coded LCG seeded from initial state. That's deterministic-for-test (admirable) but it kills compositionality — a particle filter or HMM sampler that wraps `MarkovSimulate` cannot share an RNG, leading to correlated samples between particles. Agent 117 T1.3 already names `prob.Rng` as missing; this synergy is a forcing function.

2. **`prob.Distribution` interface needs a `Sample(rng *prob.Rng) float64`.** Currently PDF + CDF only (l.27-35 of `distribution.go`). HMM/CRF/particle-filter all need to *generate* observations, not just score them. Backward-compatible addition; default impl from `Quantile(rng.Uniform())` for any distribution exposing a quantile.

3. **`sequence/alignment.go` should expose log-domain variants.** `NeedlemanWunsch(a, b, match, mismatch, gap)` is the score-domain face. Add `NeedlemanWunschLog(a, b string, logEmit func(rune,rune)float64, logGap float64)` — same DP, callable as the Viterbi path of any pair HMM. ~30 LOC, reuses `dp` skeleton.

4. **A shared `EditScript` type.** `LevenshteinDistance` returns `int`, NW returns aligned-string-pair, pair-HMM Viterbi returns log-prob — these are three different surface contracts for the same underlying DAG. A common `type EditOp struct { Kind Op; Char rune }` (agent 127 §1.2 already names `LevenshteinEdits`) unifies them and is reusable by every synergy entry.

---

## Summary table — connective tissue LOC

| Synergy entry | New LOC | Reused (sequence + prob + optim) | Net composition |
|---|---:|---|---|
| §2 LogSumExp + Log1mExp + Log1pExp | 80 | `changepoint/bocpd.go` (move existing) | -12 LOC dedup |
| §1 HMM (forward/backward/Viterbi/BaumWelch) | 520 | MarkovSteady, Distribution, LogGamma | High |
| §3 Profile HMM | 430 | §1 HMM, future Dirichlet | High |
| §4 Pair HMM | 330 | NW dp scaffold, §2 LSE | **Highest** (3× reuse) |
| §5 Karlin-Altschul + Gumbel | 230 | optim.Bisection, math.Log | High |
| §6 Levenshtein null distribution | 150 | LevDistance, NormalCDF, McDiarmid | Medium |
| §7 Linear-chain CRF | 250 | §1 forward-backward, optim.LBFGS | Highest reuse |
| §8 Particle filter | 180 | conformal.EffectiveSampleSize | High |
| §9 VOM/PPM | 250 | sequence.NGrams, future suffix-tree | Medium |
| §10 MCMC alignment + generic MH | 250 | §4 PairHMM.Forward | High |
| §11 Suffix-array statistics | 70 | future suffix-array, prob.PoissonPMF | High |
| §12 Edit-distance random-channel | 120 | LevenshteinDistance dp | **Highest density** |
| **Total connective tissue** | **~2860 LOC** | | |

Of this ~2860 LOC, **~530 LOC is gating debt agent 117 already named** (LogSumExp/Log1mExp/Gumbel/Rng/MH driver/Dirichlet) — the synergy ledger forces those gaps to be closed in a coherent batch rather than ad hoc.

The minimal MVP that lights up the most synergy is: **§2 (80 LOC) + §1 HMM (520) + §4 PairHMM (330) + §12 random-channel (120) = 1050 LOC**, which gets you the four most-cited probabilistic-sequence-analysis primitives in the field. Everything else builds on those.

The single highest-yield-per-LOC item is **§4 Pair HMM** — it makes `NeedlemanWunsch` the deterministic Viterbi shadow of a richer probabilistic object, exposes posterior decoding (state-of-the-art for sequence alignment since 2003), and adds 330 LOC against ~67 LOC of existing NW scaffolding. It is the single commit I would land first if I owned this package.
