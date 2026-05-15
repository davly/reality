# 189 | synergy-info-compression

**Topic:** info x compression — minimum description length, BIC/AIC pinning.
**Block:** B (cross-package synergies). **Date:** 2026-05-08. **Scope:**
the model-selection / Kolmogorov / MDL angle, **NOT** the source-coding /
Huffman / arithmetic / Kraft angle 182 already owns. PARTIAL OVERLAP with
182 explicitly delegated: V1-V10 of 182 (Kraft, Huffman, arithmetic, NCD,
Shannon-Fano-Elias-as-prefix-coder) ship there; this report owns refined
MDL, AICc/AICu/HQIC/FPE, gMDL, fNML, KT/CTW universal mixtures, MDL-
applied-to-{regression, changepoint, clustering, ARMA-order}, Elias/
Rice/Golomb-as-MDL-substrate, LZ-as-K-estimator.

## Two-line summary

The repo ships **the cheapest cut of the MDL surface** — `info/mdl/` (619
LOC: NMLMultinomial Kontkanen-Myllymäki-2007 + NMLBernoulli + Bernoulli
CodeLength + GaussianCodeLength + ModelCodeLength + BICShape + AICShape
+ UniversalIntegerCodeLength Rissanen-1983 + SelectMDL + SelectMDLWith
Margin), `info/lz/` (466 LOC LZ76 + Kaspar-Schuster + 3 symbolisations),
`compression/entropy.go` (177 LOC bits-units H/MI/KL) — but **literally
zero refined-MDL machinery** verified by grep (zero matches across
`info/`+`compression/`+`prob/` on `Elias|Golomb|Rice|LZ77|LZ78|LZW|gMDL|
HansenYu|KrichevskyTrofimov|CTW|ContextTree|Willems|fNML|RoosSilander|
AICc|HurvichTsai|AICu|HQIC|MDLRegression|WallaceMML|Solomonoff|VC.dim
|UniversalPortfolio`); explicit v2 deferrals at `info/mdl/doc.go:96-111`
name fNML, Luckiness-NML, CTW, Wallace-MML, Solomonoff. **Twenty-two
synergy primitives W1-W22 totalling ~3210 LOC of pure connective tissue**
close the gap with NO new packages; cheapest one-day PR is **W1 AICc
+ W2 AICu + W3 BICc + W4 HQIC = 90 LOC** four pure-arithmetic adapters
saturating R-MUTUAL-IC-4-WAY (AIC < HQIC < BIC < AICc strict ordering at
finite n, asymptotic agreement at n -> infinity); highest-leverage
architectural lift is **W7 KrichevskyTrofimov + W8 ContextTreeWeighting
+ W9 fNMLMultinomial + W10 LuckinessNML ~510 LOC** because (a) KT is
THE bridge between NML-batch and Bayes-mixture-online, (b) CTW is
the canonical universal-coding-on-tree-sources predictor explicitly
deferred at `info/mdl/doc.go:111`, (c) fNML answers the Causal-engine
GES `local_score_BIC` consumer flagged at `info/lz/doc.go:46-47`,
(d) Luckiness-NML closes the GEV-xi-boundary case at `info/mdl/doc.
go:107-108`. Crown jewel is **W17 MDLLinearRegression + W18 MDL
Changepoint + W19 MDLClusteringSCMS + W20 MDLTimeSeriesOrder ~800 LOC**
— the four canonical MDL-applied-to-statistics drivers retrofitting
RubberDuck Granger lag selection (`info/mdl/doc.go:14`), Causal-engine
GES, and Sensorhub/Nexus hard-coded bin counts. Recommended placement:
NO new packages — all 22 land in `info/mdl/{ic.go, mixture.go, integer_
codes.go, driver.go, k_complexity.go}` plus one cross-package test in
`compression/`. Cycle-free DAG: NEW edges `info/mdl/ -> compression/`,
`info/mdl/ -> info/lz/` (FIRST sibling-info-subpackage edge today co-
located-disjoint per `info/mdl/doc.go:42`), `info/mdl/ -> prob/`; reverse
direction never. Six R-MUTUAL pins fall out: **AIC-vs-AICc-vs-BIC-vs-
HQIC-4-way** (asymptotic agreement + finite-n strict ordering),
**BIC-vs-NML-Laplace-equivalence** (Schwarz-1978 = Rissanen-1996),
**CTW-vs-KT-vs-NML-3-way** (universal-mixture redundancy O((k-1)/2 log
n)), **LZ-rate-vs-empirical-H-vs-Shannon-rate-3-way** (Wyner-Ziv 1989),
**fNML-vs-BIC-on-fully-observed-Bayesian-network**, **Rissanen-log\*-
vs-Elias-gamma-vs-Elias-delta-3-way** (Elias 1975).

---

## 0. State of play (verified file-walk)

`info/mdl/` HEAD (619 LOC + 511 LOC test):

- `nml.go` 157 LOC: `NMLMultinomial(counts)` Kontkanen-Myllymäki 2007
  recurrence `C(n,k) = C(n,k-1) + (n/(k-1))*C(n,k-2)` with O(n) base
  C(n,2) summed in log-space.
- `bernoulli.go` 90 LOC: `NMLBernoulli(s,t)`, `BernoulliCodeLength(s,t)`
  = NLL + regret.
- `codelength.go` 128 LOC: `GaussianCodeLength`, `ModelCodeLength` =
  (k/2)log n, `BICShape`, `AICShape`.
- `universal_int.go` 71 LOC: `UniversalIntegerCodeLength` Rissanen
  1983 log\* + bits variant.
- `select.go` 74 LOC: `SelectMDL`, `SelectMDLWithMargin` argmin + gap.
- `errors.go` 39 LOC: 6 sentinels.

`info/lz/` HEAD (466 LOC): `LempelZivComplexity(symbols, alphabetHint)
-> LzComplexityResult{WordCount, NormalizedComplexity, ...}` +
`SymbolizeByQuantile/Threshold` + `RollingComplexity`.

`compression/` HEAD (380 LOC): `entropy.go` 177 LOC bits-units; `coding.
go` 104 LOC RLE+delta only (no Huffman/arithmetic/Elias/Golomb/LZ77,
per 182 §0); `quantize.go` 99 LOC uniform scalar.

**Explicit v2 deferrals to retrofit:** `info/mdl/doc.go:107-111` Wallace
MML / fNML / Luckiness-NML / CTW; `info/lz/doc.go:90-99` NCD / BDM /
CTW / Solomonoff; `info/mdl/doc.go:14` BIC-from-Laplace; `prob/
timeseries.go:125` ARIMA accepts (p,d,q) but no order-selection helper;
`prob/regression.go:36` LinearRegression returns rSquared but no
log-likelihood / RSS / MDL-regression driver; `changepoint/bocpd.go`
geometric prior hard-coded, no MDL alt.

**Cross-edges today: zero.** `info/mdl/`, `info/lz/`, `compression/`
each import only `math` (+ `errors` in info packages). **Naming
mismatch** (per 182 §0): `info/mdl/*` nats; `compression.ShannonEntropy`
bits; `info/lz` mixed. W-primitives convert at boundaries via
`* math.Ln2` or `/ math.Ln2`.

---

## 1. Conceptual unlocks (model-selection axis)

(1) **Two-part MDL = K-bound** `K(x) <= L(H) + L(D|H)`; both halves
ship, no driver. (2) **Refined MDL (Rissanen 2007)** NLL + parametric-
complexity + integer-prior-on-precision; every ingredient ships, no
composition. (3) **BIC = asymptotic-Laplace** of integrated likelihood
(Schwarz 1978); repo ships shape, no Laplace. (4) **AIC ≡ LOO-CV
squared-error** (Stone 1977). (5) **AICc** Hurvich-Tsai 1989 finite-
n at n/k<40 missing. (6) **AICu** McQuarrie-Tsai 1998, **HQIC** Hannan-
Quinn 1979 strongly-consistent ARMA. (7) **gMDL** Hansen-Yu 2001
hierarchical regression — canonical reference for the Granger-lag
inverse-consumer at `info/mdl/doc.go:14`. (8) **fNML** Roos-Silander-
Kontkanen-Myllymäki 2008 per-context NML for Bayesian-network — drop-
in over NMLMultinomial answering Causal-engine GES. (9) **Luckiness-
NML** Grünwald 2007 §11.4 unbounded params (GEV-xi-on-real-line) —
deferred at doc.go:107-108. (10) **Krichevsky-Trofimov** Bayes-mixture
under Jeffreys: redundancy `((k-1)/2) log n + O(1)` matches NML
asymptotically. (11) **CTW** Willems-Shtarkov-Tjalkens 1995 universal
predictor on tree sources — deferred at doc.go:111. (12) **LZ76 as K-
estimator** `K(x) <= |LZ76(x)| + O(log |x|)` (Lempel-Ziv 1976 §IV).
(13) **LZ-rate -> H_rate** Wyner-Ziv 1989 `c(n) log_A c(n)/n -> H_rate`
a.s.; cross-link 182-V11 same primitive different framing. (14) **Elias
gamma/delta/omega** 1975 universal integer codes; ~15 LOC each. (15)
**Rice/Golomb** 1971/1966 optimal-for-geometric; `compression/coding.go:
12` cites Golomb in RLE comment, ships no Golomb code. (16) **MDL-
regression** Rissanen 1989 §6 / Barron-Rissanen-Yu 1998. (17) **MDL-
changepoint** Davis-Lee-Rodriguez-Yam 2006 cross-link `changepoint/
bocpd.go` Bayesian counterpart. (18) **MDL-clustering SCMS** Kontkanen-
Myllymäki 2008. (19) **VC / covering / PAC** Vapnik-Chervonenkis 1971
mirrors MDL `(d/2) log n / n` redundancy. (20) **Solomonoff M / AIXI /
Wallace MML / Universal portfolio Cover-1991** deferred — incomputable
/ domain-specific / theoretical-ceiling.

---

## 2. Twenty-two synergy primitives (W1-W22)

Each entry: (1) capability, (2) composition, (3) connective LOC.

### Tier A — information criteria adapters (≤30 LOC each)

**W1. AICc(nll, k, n) float64.** Hurvich-Tsai 1989: `AICShape(nll, k)
+ k(k+1)/(n-k-1)`. Critical at `n/k < 40`. Guard `n - k - 1 <= 0` ->
+Inf. **LOC: 25.** R-MUTUAL pin: agreement with AICShape at n=1000 to
1e-3; +2.14 nats divergence at n=20, k=5 per Hurvich-Tsai 1989 Tab.1.

**W2. AICu(nll, k, n) float64.** McQuarrie-Tsai 1998 unbiased-MSE:
`AICShape + (n+k)/(n-k-2)`. **LOC: 20.**
**W3. BICc(nll, k, n) float64.** Tremblay-Wagner 2004: `BICShape +
k(k+1) log(n)/(n-k-1)`. **LOC: 20.** Pin: `n -> inf` recovers BICShape
to 1e-9.
**W4. HQIC(nll, k, n) float64.** Hannan-Quinn 1979: `nll + k log log n`.
Strongly consistent for ARMA-order. Guard `n < 4`. **LOC: 25.**
**R-MUTUAL-IC-4-WAY pin (189-flagship):** at n=10000 four ICs agree to
O(1); at n=20, k=5 strict ordering AIC < HQIC < BIC < AICc per Burnham-
Anderson 2002 Tab.2.5.
**W5. FPE(rss, k, n) float64.** Akaike 1969: `(rss/n) * (n+k)/(n-k)`.
**LOC: 30.** Pin: monotone-related to AIC at large n.
**W6. MallowsCp(rss_full, rss_sub, k_full, k_sub, n) float64.** Mallows
1973: `rss_sub/sigma^2 - (n - 2 k_sub)`. OLS-specific BIC-shape;
`prob.LinearRegression` ships rSquared but not RSS. **LOC: 35.**

### Tier B — universal-mixture estimators (60-250 LOC each)

**W7. KrichevskyTrofimovEstimator(counts) (probs, regret).** KT 1981
Bayes-mixture under Jeffreys prior: `(c_i + 0.5)/(n + k/2)` and worst-
case redundancy `((k-1)/2) log_2 n + O(1)`. Asymptotically matches NML
to O(1). **LOC: 60.** **R-MUTUAL-MIXTURE-3-WAY pin:** NML / KT /
asymptotic-BIC `(k-1)/2 log n` agree to O(1) at n=10000, k=4.

**W8. ContextTreeWeighting(symbols, depth) (logProb, err).** Willems-
Shtarkov-Tjalkens 1995. Tree of KT-mixtures up to bounded depth; leaf-
vs-internal-mixture weighting achieves Rissanen-redundancy on bounded-
depth tree sources. Consumes W7 at each node. **LOC: 250.** Pin:
Willems-Shtarkov-Tjalkens 1995 Tab.II at depth=3 binary, n=128, 1e-9.
Closes explicit `info/mdl/doc.go:111` v2 deferral.

**W9. fNMLMultinomial(counts [][]int) float64.** Roos-Silander-
Kontkanen-Myllymäki 2008 factorised NML for graphical models: `sum_pa
NMLMultinomial(counts[pa])`. Drop-in for Causal-engine GES `local_
score_BIC`. **LOC: 80.** **R-MUTUAL pin:** fNML vs BIC equal-up-to-
O(1) on Roos et al. 2008 Tab.III ALARM-network synthetic.

**W10. LuckinessNML(samples, luckiness) (codelength, err).** Grünwald
2007 §11.4 NML for unbounded parameter spaces. Adds luckiness g(theta)
to the NML denominator. Closes `info/mdl/doc.go:107-108` GEV-xi-boundary
deferral. **LOC: 120.**

**W11. SequentialNML(counts) float64.** Rissanen 1996 sequential
stochastic complexity; on-line NML with prefix-only counts. Cleaner
asymptotics than batch-NML. Composition: prefix-sum over W7. **LOC: 50.**

### Tier C — universal integer codes (15-30 LOC each)

**W12. EliasGammaCodeLength(n) bits.** `1 + 2 floor(log2 n)`. **LOC: 15.**
Pin: `gamma(1)=1, gamma(2)=3, gamma(7)=5` Elias 1975 Tab.II.
**W13. EliasDeltaCodeLength(n) bits.** `floor(log2 n) + 2 floor(log2(1+
floor(log2 n))) + 1`. **LOC: 20.** Pin: Elias Tab.III.
**W14. EliasOmegaCodeLength(n) bits.** Recursive log\*. **LOC: 30.**
**R-MUTUAL-INTEGER-CODE-3-WAY pin:** Rissanen-log\* / Elias-gamma /
Elias-delta / Elias-omega — at n=2 is gamma < delta < omega < log\*;
at n=1000 is log\* ≈ omega < delta < gamma.
**W15. RiceCodeLength(n, k) bits.** `floor(n/2^k) + 1 + k`. Optimal
for `Geometric(p=2^{-k})`. **LOC: 25.**
**W16. GolombCodeLength(n, m) bits.** `floor(n/m) + 1 + ceil(log2 m)`
with truncation refinement when m not a power of 2. **LOC: 30.**

### Tier D — MDL-applied-to-statistics drivers (150-250 LOC each)

**W17. MDLLinearRegression(x, y, candidateDegrees) (bestDegree,
codelengths, err).** For each polynomial degree, fit OLS, code residuals
as Gaussian via `GaussianCodeLength`, code coefficients as integers via
`UniversalIntegerCodeLength * (d+1)`, code degree itself, sum, argmin.
Rissanen 1989 §6, Barron-Rissanen-Yu 1998. Composition: extend
`prob.LinearRegression` to degree>1 via Vandermonde + `linalg.LUSolve`
+ `GaussianCodeLength` + `UniversalIntegerCodeLength`. **LOC: 180.**
Pin: synthetic noisy polynomial degree=3 sigma=0.1 n=100; recover
degree=3 from {1..6} >95% of trials.

**W18. MDLChangepoint(data, maxK) (positions, codelengths, err).** Davis-
Lee-Rodriguez-Yam 2006: DP over K-segmentations, code each segment
Gaussian, code positions via `UniversalIntegerCodeLength`. O(n^2 K)
DP. **LOC: 250.** Pin: vs `changepoint/bocpd` on Truong-Oudre-Vayatis
2020 fixtures, locations agree +/- 2 samples.

**W19. MDLClusteringSCMS(data, maxK) (bestK, codelengths, err).**
Kontkanen-Myllymäki 2008. Requires k-means prereq W19a (~80 LOC); SCMS
driver W19b (~140 LOC) computes NML codelength of resulting partition
via `NMLMultinomial`(cluster_counts) + Gaussian per cluster. **LOC: 220.**
Pin: synthetic Gaussian mixtures K=3, n=300, dim=2; recover K=3 from
{1..6} >90% per Kontkanen-Myllymäki 2008 Tab.II.

**W20. MDLTimeSeriesOrder(data, maxP, maxQ) (bestP, bestQ, codelengths,
err).** ARMA-order via MDL: for each (p,q), fit `prob.ARIMA(p,0,q)`,
compute log-likelihood, code as `BICShape(nll, p+q, n) +
UniversalIntegerCodeLength(p) + UniversalIntegerCodeLength(q)`. Direct
retrofit of RubberDuck Granger `OptimalLag` per `info/mdl/doc.go:14`.
**LOC: 150.** Pin: ARMA(2,1) synthetic; recover (2,1) >85% over (p,q)
in [0,4]^2.

### Tier E — K-complexity / theoretical-ceiling (30-40 LOC each)

**W21. LZ76EntropyRate(symbols, alphabetSize) (rateBits, err).** Wyner-
Ziv 1989: `c(n) log_2 c(n) / n -> H_rate`. 5-LOC composition over
`info/lz.LempelZivComplexity`. Cross-link 182-V11 same primitive
different framing; here K-complexity reading. **LOC: 40.** **R-MUTUAL-
RATE-3-WAY pin:** Bernoulli(0.3) n=10000; LZ76 / arithmetic-coder /
Shannon-empirical agree to within +/- 0.05 bits/symbol of H(0.3)=0.881.

**W22. KComplexityUpperBound(symbols, alphabetSize) (boundBits, err).**
LZ76 K-bound `K(x) <= |LZ76(x)| + O(log |x|)` per Lempel-Ziv 1976 §IV.
`c(n) * (log_2 c(n) + log_2 alphabetSize) + O(log n)`. **LOC: 30.**
Pin: random-iid upper bound matches `H * n + O(log n)`; constant-source
upper bound -> O(log n).

---

## 3. Cycle-free DAG and architectural placement

**No new packages.** All 22 W-primitives ship inside `info/mdl/` plus
ONE new test in `compression/`:

```
info/mdl/                           compression/
  ic.go              <- W1-W6        coding.go  (existing)
  mixture.go         <- W7-W11       entropy.go (existing)
  integer_codes.go   <- W12-W16      mdl_compression_pin_test.go (new)
  driver.go          <- W17-W20
  k_complexity.go    <- W21, W22
```

**New cross-edges:**

- `info/mdl/ -> compression/`: W7 KT consumes `compression.ShannonEntropy`;
  W17 MDLRegression consumes `compression.ScalarQuantize`; new test
  in `compression/` consumes `info/mdl/` for **mdl-codelength-vs-empirical-
  compressed-size** parity demonstration.
- `info/mdl/ -> info/lz/`: W21 + W22 consume `info/lz.LempelZivComplexity`.
  **First sibling-info-subpackage edge** (today co-located-disjoint per
  `info/mdl/doc.go:42`).
- `info/mdl/ -> prob/`: W17 consumes `prob.LinearRegression`; W18 cross-
  validates `changepoint/bocpd`; W20 consumes `prob.ARIMA`.
- **Reverse direction never.** `prob/` does not import `info/mdl/`;
  retrofit pattern is consumer-pull per `info/mdl/doc.go:66-75`.

DAG: `info/mdl/ -> {compression/, info/lz/, prob/}`; `info/lz/ -> {}`;
`compression/ -> {}`; zero cycles.

---

## 4. Recommended landing order

**PR-1 (cheapest, zero blockers): W1+W2+W3+W4 = 90 LOC source + 40 LOC
test.** Pure-arithmetic IC adapters. Saturates **R-MUTUAL-IC-4-WAY**
(asymptotic agreement at n -> inf, strict ordering AIC < HQIC < BIC <
AICc at n=20, k=5 per Burnham-Anderson 2002 Tab.2.5). FIRST 4-way
R-MUTUAL pin in `info/mdl/` extending 6a55bb4-audio-onset / 365368a-
Clayton-autodiff / 1e12e80-token-set-ratio family.

**PR-2: W12+W13+W14+W15+W16 = 120 LOC source + 60 LOC test.** Five
universal integer codes. Pin against Elias 1975 Tab.II/III/IV closed-
forms. **R-MUTUAL-INTEGER-CODE-3-WAY** ordering pin.

**PR-3: W7 + W11 = 110 LOC source + 50 LOC test.** Bayes-mixture
estimator + sequential-NML. Pin against existing NMLMultinomial to
O(1) at n=10000. **R-MUTUAL-MIXTURE-3-WAY**.

**PR-4: W21 + W22 = 70 LOC source + 30 LOC test.** LZ-rate + K-bound.
FIRST `info/mdl/ -> info/lz/` cross-edge. **R-MUTUAL-RATE-3-WAY**.

**PR-5: W17 + W6 = 215 LOC source + 80 LOC test.** FIRST `info/mdl/ ->
prob/` cross-edge. Polynomial-recovery pin.

**PR-6: W20 = 150 LOC source + 60 LOC test.** Direct RubberDuck Granger
`OptimalLag` retrofit per `info/mdl/doc.go:14`.

**PR-7: W18 = 250 LOC source + 80 LOC test.** Davis et al. DP. Cross-
validate against `changepoint/bocpd`.

**PR-8: W19 (W19a k-means + W19b SCMS) = 220 LOC source + 100 LOC test.**
First clustering primitive in repo.

**PR-9: W9 = 80 LOC source + 40 LOC test.** Causal-engine GES retrofit.
Closes `info/mdl/doc.go:106-107` v2 deferral.

**PR-10: W8 = 250 LOC source + 100 LOC test.** Heaviest standalone.
Closes `info/mdl/doc.go:111` v2 deferral.

**PR-11: W10 = 120 LOC source + 60 LOC test.** Closes `info/mdl/doc.go:
107-108` v2 deferral.

**PR-12 (optional): W5 = 30 LOC source + 20 LOC test.**

Total: 12 PRs, ~3210 LOC source + ~1320 LOC test = ~4530 LOC over
~16 engineer-days. Closes 4 of 5 explicit v2 deferrals at `info/mdl/
doc.go:96-111` (Wallace MML still deferred).

---

## 5. Precision hazards

W1 AICc at `n -> k+1` denominator -> 0 guard `n-k-1 <= 0 -> +Inf`
(Burnham-Anderson recommend `n/k > 5` only). W4 HQIC at `n < 4`
log(log(n)) < 0 or NaN guard `n < 4 -> +Inf`. W7 KT at n=0 returns
uniform 1/k under Jeffreys (correct). W8 CTW memory O(2^depth) cap
at 16 (Willems 1998). W12-W16 integer codes at n<=0 -> ErrInvalid
IntegerCode. W17 precision-bits default 12/coeff (Rissanen 1989 §6.2),
expose as parameter. W18 K=0 case must return K=0 if data best-coded
as one segment (Davis et al. 2006 §3.2). W21 c(n)=1 constant-source
rate -> 0 already handled by `info/lz` at `lz76.go:99-108`.

---

## 6. Cross-language pinning targets

W1-W4 Burnham-Anderson 2002 Tab.2.5 1e-6; W7 vs NML Rissanen 1996 Tab.1
O(1)/n at n=10000; W8 CTW Willems-Shtarkov-Tjalkens 1995 Tab.II depth=3
n=128 1e-9; W9 fNML Roos et al. 2008 Tab.III ALARM-net 1e-6; W12-W14
Elias 1975 Tab.II/III/IV closed-form; W17 Barron-Rissanen-Yu 1998 §5
polynomial-recovery >95% over 1000 trials; W18 Davis-Lee-Rodriguez-Yam
2006 Tab.1 +/- 2 samples; W19 Kontkanen-Myllymäki 2008 Tab.II 90%
K-recovery; W21 Bernoulli(0.3) n=10000 within +/- 0.05 bits/symbol of
H(0.3)=0.881 (Ziv-Merhav 1993). Eleven of twenty-two have public-API
equivalents (R `ic`, Python `scipy.stats.modelselection`, MATLAB
`aicbic`) pinning at 1e-6 mirroring 6a55bb4 + 365368a + 1e12e80
R-MUTUAL family.

---

## 7. Differentiation from sibling reviews

**From 182 (source-coding side):** 182 owns Kraft, Huffman, arithmetic,
NCD, rANS/tANS, BWT, MTF, LZ77/78/W. THIS owns model-selection: AICc/
AICu/HQIC/FPE, gMDL, fNML, KT, CTW, Luckiness-NML, MDL-{regression,
changepoint, clustering, ARMA}, Elias/Rice/Golomb-as-MDL-prior-substrate
(NOT as source-coders — same formula, different domain), LZ76-as-K-
estimator. Joint W21=182-V11 recommend joint-land PR-4. **From 170
(channel side):** rate-distortion / capacity / Fano / Blahut-Arimoto
orthogonal axis. **From 086-090, 041-045, 117-120 (isolation reviews):**
those flag missing surface; THIS adds consumer-pull from compression +
prob motivating same primitives. **Placement disagreement with 182:**
Huffman+arithmetic-as-source-coder belong in `compression/` (per 182);
Elias-as-MDL-prior belongs in `info/mdl/integer_codes.go` (per 189) —
same codeword-length formula, different domain-of-use.

**Single-day high-leverage if only one PR: PR-1 W1+W2+W3+W4 = 90 LOC
source + 40 LOC test** because (a) zero upstream blockers — pure
arithmetic on existing `BICShape`/`AICShape`; (b) saturates 4-way
R-MUTUAL pin to 1e-6, second 4-way pin in 189 corpus (188-PR-3 D2/D3/
D14+Cholesky was first); (c) closes largest documented gap in `info/
mdl/codelength.go` — AICShape ships but no AICc despite Hurvich-Tsai
1989 being THE textbook finite-sample correction; (d) immediately
unblocks 8 of 8 inverse-consumer sites at `info/mdl/doc.go:14-21`
(every BIC consumer swaps to AICc with one-line change at finite n);
(e) extends existing 511-LOC `mdl_test.go` with ~40 LOC preserving
the Rissanen-1983 closed-form pin idiom already in use. First three
PRs over three engineer-days lift `info/mdl/` from "v1 cheapest cut"
to "all six ICs + universal mixture predictor + five universal integer
codes" closing FOUR of the five explicit v2 deferrals at `info/mdl/
doc.go:96-111` (only Wallace MML remains deferred; fNML lands PR-9,
Luckiness-NML PR-11, CTW PR-10).
