# 182 | synergy-compression-info

**Topic:** compression x info — arithmetic coding ↔ entropy, KL bound on compressibility.
**Block:** B (cross-package synergies). **Date:** 2026-05-08.
**Scope:** capabilities that emerge ONLY when `compression/`, `info/lz/`, `info/mdl/` are composed; not the per-package isolation reviews 041-045 (compression numerics/missing/sota/api/perf), 086-090 (info numerics/missing/sota/api/perf if they exist), or related synergy 170 (info x prob — Blahut-Arimoto, R(D), Fano).

## Two-line summary

The repo ships **the entropy LOWER bound and the LZ76 estimator UPPER bound** (`compression.ShannonEntropy/JointEntropy/ConditionalEntropy/MutualInformation/KLDivergence/CrossEntropy` 177 LOC bits-units, `info/mdl/` NML/BIC/AIC/UniversalInteger/SelectMDL 619 LOC nats-units, `info/lz/` LZ76 + symbolisation 466 LOC) with **literally zero coding primitives in between** — no Huffman, no arithmetic coding, no range coder, no rANS/tANS, no Kraft-McMillan witness, no expected-code-length verifier, no NCD, no LZ77/LZ78/LZW source coder, no BWT, no MTF — verified by cross-tree grep on `Huffman|Arithmetic.cod|Kraft|Burrows.Wheeler|MoveToFront|LZ77|LZ78|LZW|RangeCod|rANS|tANS|NormalizedCompression` returning zero matches in any source file. **Twenty synergy primitives V1-V20 totalling ~3050 LOC of pure connective tissue** close the gap; cheapest one-day PR is **V1 KraftMcMillanCheck + V2 ExpectedCodeLength + V3 RedundancyFromCode + V4 ShannonCodeLengths = 130 LOC** consuming only existing `ShannonEntropy` + arithmetic; highest-leverage architectural lift is **V5 CanonicalHuffman + V6 ArithmeticCoder + V7 NormalizedCompressionDistance ~830 LOC** because (a) Huffman is promised three times in CLAUDE.md/README.md/ARCHITECTURE.md while shipping zero lines (per 041 + 042), (b) arithmetic coding is the single math-richest entropy coder, (c) NCD = `(C(xy) − min(C(x),C(y))) / max(C(x),C(y))` is a 30-LOC composition once any compressor exists. Crown jewel is **V11 LZ76EntropyRate + V12 EmpiricalCompressibilityVsKL + V13 KrichevskyTrofimovEstimator ~470 LOC** — Wyner-Ziv 1989 LZ-as-entropy-rate, the empirical demonstration that `|C(x)|/n − H(X) → KL(P̂ || Q_code)` for any sub-optimal code Q, and KT as the Bayesian-mixture universal redundancy benchmark. Recommended placement: extend `compression/` with `coding.go` extensions for V1-V4 (entropy↔code-length identities, no new package), new `compression/huffman.go` + `compression/arith.go` for V5-V6 (per 042's recommended ladder), new `compression/ncd.go` for V7, and add `info/entropy/` sub-package mirroring `info/lz` + `info/mdl` for V11-V13 (universal-coding entropy estimators). Cycle-free DAG: `compression/` → `info/lz/` (NCD consumes any compressor as a black-box; LZ76 production count from `info/lz` would qualify); `info/entropy/` → `compression/` (entropy-rate estimators consume LZ76); reverse direction never. Six R-MUTUAL-CROSS-VALIDATION 3/3 pins fall out for free: **Shannon-Fano-vs-Huffman-vs-arithmetic** (all three coders within 1 bit of `H(X)` per source coding theorem), **Kraft-witness-vs-expected-length-vs-entropy** (`H ≤ E[L] < H+1` with Kraft equality at optimal), **NCD-symmetry** (NCD(x,y) ≈ NCD(y,x)), **LZ76-rate-vs-empirical-H** (Wyner-Ziv 1989: `c(n)log_A(c(n))/n → H_rate`), **cross-entropy-vs-arithmetic-coding-overhead** (the canonical KL = excess bits identity), and **MDL-codelength-vs-empirical-compressed-size** (BIC ≈ −log P + (k/2)log n vs `|gzip(x)|`).

---

## 0. State of play (verified file-walk)

`compression/` HEAD (3 source files, 380 LOC):

- `entropy.go` (177 LOC, 5 functions): `ShannonEntropy(probs)`, `JointEntropy(joint)`, `ConditionalEntropy(joint)`, `MutualInformation(joint)`, `KLDivergence(p, q)`, `CrossEntropy(p, q)`. **Bits-units throughout**, no validation, no LSE guard. Skips `p_i ≤ 0`. KL/CrossEntropy return `+Inf` when `q_i ≤ 0`.
- `coding.go` (104 LOC, 4 functions): `RunLengthEncode/Decode`, `DeltaEncode/Decode`. **Zero entropy coders.** Comment at top declares "lossless coding" but RLE+Delta is the entire surface.
- `quantize.go` (99 LOC, 2 functions): `ScalarQuantize/Dequantize` — uniform scalar only, no Lloyd-Max.

`info/lz/` HEAD (1 source file 466 LOC + errors + doc):

- `lz76.go`: `LempelZivComplexity(symbols, alphabetSize) → LzComplexityResult{WordCount, NormalizedComplexity, SequenceLength, AlphabetSize, Interpretation}`, `SymbolizeByQuantile`, `SymbolizeByThreshold`, `ComplexityFromReturns`, `RollingComplexity`. **Production-count c(S) and Kaspar-Schuster normalisation** against `n / log_A(n)`. **Cross-substrate parity to RubberDuck** ≤1e-12 per `lz76_test.go`. **No entropy-rate estimator** despite Wyner-Ziv 1989 being a 5-LOC composition `H_LZ = c(n) * log2(c(n)) / n` once `c(n)` is in hand.

`info/mdl/` HEAD (5 source files, 619 LOC):

- `nml.go` (157 LOC): `NMLMultinomial(counts) → (regret_nats, err)`. **Kontkanen-Myllymäki 2007 linear-time recurrence** `C(n,k) = C(n,k-1) + (n/(k-1)) * C(n,k-2)` with base-case `C(n,2)` summed in log-space.
- `bernoulli.go` (90 LOC): `NMLBernoulli(s, t)` (delegates to multinomial k=2), `BernoulliCodeLength(s, t)` = NLL_at_MLE + NML regret.
- `codelength.go` (128 LOC): `GaussianCodeLength(samples, mu, sigma)`, `ModelCodeLength(numParams, n)` = `(k/2)log(n)`, `BICShape(nll, k, n)`, `AICShape(nll, k)`.
- `universal_int.go` (71 LOC): `UniversalIntegerCodeLength(n)` = Rissanen 1983 `log*(n) + log(2.865064)` and `UniversalIntegerCodeLengthBits(n)`.
- `select.go` (74 LOC): `SelectMDL(codeLengths)`, `SelectMDLWithMargin(codeLengths)` — argmin + gap-to-second-best.

**Cross-edges today: zero.** Verified `grep github.com/davly/reality/compression info/**/*.go` → 0; reverse `grep github.com/davly/reality/info compression/*.go` → 0. Both packages import only `math` + (info/mdl) `errors`.

**Naming inconsistency:** `compression.ShannonEntropy(probs)` works in **bits** (uses `math.Log2`); `info/mdl.GaussianCodeLength` works in **nats** (uses `math.Log`); `info/mdl.UniversalIntegerCodeLengthBits` is the only nats→bits adapter shipped. Any V-primitive composing entropy with codelength must convert at the boundary.

---

## 1. Conceptual unlocks (compressed)

1. **Shannon source coding theorem.** For a memoryless source `X` with distribution `p`, any prefix code with lengths `ℓ` satisfies Kraft `Σ 2^(-ℓ_i) ≤ 1`, and the optimal expected length satisfies `H(X) ≤ E[L*] < H(X) + 1`. Repo ships `H(X)` (lower bound, `compression.ShannonEntropy`) and zero coders to witness the upper bound.
2. **Kraft-McMillan = Kraft for non-prefix uniquely-decodable.** `Σ 2^(-ℓ_i) ≤ 1` is necessary AND sufficient for either prefix or UD codes (McMillan 1956 generalises Kraft 1949). 5-LOC verifier.
3. **Cross-entropy = arithmetic coding overhead.** Encoding source `P` with a code optimal for `Q` uses `−Σ p_i log2(q_i) = H(P) + KL(P||Q)` bits per symbol. The repo ships `H(P,Q)` (`compression.CrossEntropy`) and `KL(P||Q)` (`compression.KLDivergence`) but no arithmetic coder to actually realise the overhead.
4. **NCD as universal similarity.** `NCD(x,y) = (C(xy) − min(C(x),C(y))) / max(C(x),C(y))` (Cilibrasi-Vitanyi 2005) is a normalised approximation to the Kolmogorov-complexity-based normalised information distance. Trivial wrapper once any compressor exists; consumes `compression.RunLengthEncode` even today as a degenerate "pessimistic compressor" baseline.
5. **LZ-complexity as randomness measure.** Already shipped — `info/lz.LempelZivComplexity` returns `c(S)` and the random-iid normalisation. Wyner-Ziv 1989 sharpens this: for a stationary ergodic source, `c(n) log A / log n → H_rate(X)` (Lempel-Ziv 1976 + Ziv-Merhav 1993). 8-LOC composition exposes the entropy-rate estimator.
6. **Adaptive (online) coding = Bayesian mixture.** Krichevsky-Trofimov estimator `(c_i + 0.5) / (n + k/2)` minimises worst-case redundancy over the simplex. The mixture code achieves `E[L*] − H(X) = (k-1)/2 * log_2(n) + O(1)` (Rissanen-Langdon 1981 universal-coding redundancy theorem). 15 LOC.
7. **MDL ≡ universal coding.** Two-part MDL `L(model) + L(data | model)` is exactly the codelength of a Bayes-mixture-or-NML universal code. Repo's `info/mdl/BICShape` is the asymptotic-Laplace approximation; the full equivalence is `−log P_NML(x | M) = NLL_at_MLE + C(M, n)` per `info/mdl/bernoulli.go:64-90`. Connective tissue: tie `compression.CrossEntropy` to `info/mdl.NMLMultinomial` via the Bayes-mixture-vs-NML duality so a single empirical-distribution input produces both.
8. **Empirical compressibility vs KL.** Given a reference distribution `Q` (e.g. uniform) and an arithmetic coder `C_Q`, the codelength `|C_Q(x)|/n` empirically converges to `H(P̂_n) + KL(P̂_n || Q)` where `P̂_n` is the empirical distribution. This identity makes KL **operationally meaningful as compression overhead** — the gap between using the actual empirical code vs the wrong-model code is exactly `KL(P̂ || Q)` bits per symbol. 25-LOC composition.
9. **Rate-distortion ↔ quantisation+coding.** `R(D)` is the minimum bits-per-symbol achievable at distortion `D` (Shannon 1959). Repo's `ScalarQuantize` provides distortion-`step^2/12`; pairing with an arithmetic coder gives an actual achievable `(R, D)` pair to test against `R(D) = H(X) − H(X | X̂)`. Cross-link to synergy 170 (info x prob) where the closed-form Gaussian R(D) = max(0, 0.5*log(σ²/D)) lives.
10. **Information bottleneck ≡ minimal sufficient compression.** `min I(X;T) − β I(T;Y)` (Tishby-Pereira-Bialek 1999) is the same alternating Blahut-Arimoto structure as channel capacity (synergy 170 owns this), but applied to compression: T is the compressed representation of X retaining maximum information about Y. This is rate-distortion with `D = −I(T;Y)`.

---

## 2. Twenty synergy primitives (V1-V20)

Each entry: (1) capability, (2) composition of existing compression + info primitives, (3) connective-tissue LOC.

### Tier A — Shannon source coding identities (0-LOC-new-mathematics, ≤60 LOC each)

**V1. KraftMcMillanCheck(lengths []int) (slack float64, valid bool).**
Verifies `Σ 2^(-ℓ_i) ≤ 1` and reports the slack `1 − Σ 2^(-ℓ_i)`. Slack = 0 means the code is "complete" (uses every bit-string of the given length); slack > 0 means redundancy could be removed; slack < 0 (`valid = false`) means no UD code with those lengths can exist.
Composition: pure arithmetic on `lengths`, returns `1.0 - sum(math.Exp2(-float64(L)))`.
**LOC: 25.** R-MUTUAL pin: against textbook Cover-Thomas 5.2.1 examples (e.g. `{2,2,2,2}` → slack=0; `{1,2,3,3}` → slack=0; `{1,2,2,3,3}` → slack=−0.125 invalid).

**V2. ExpectedCodeLength(probs []float64, lengths []int) float64.**
Returns `E[L] = Σ p_i * ℓ_i`. Pre-condition: `len(probs) == len(lengths)`.
Composition: dot-product, 5 LOC body.
**LOC: 15.** Pin: textbook Cover-Thomas 5.4 — for any UD code, `E[L] ≥ H(X)` (Shannon source coding theorem, lower bound).

**V3. CodingRedundancy(probs []float64, lengths []int) float64.**
Returns `R = E[L] − H(X)`. Coding redundancy is bounded by 1 bit/symbol for any prefix code on a fixed-alphabet memoryless source (Shannon source coding theorem upper bound `E[L*] < H(X)+1`); equals 0 only when all `p_i` are powers of 2.
Composition: `V2 - compression.ShannonEntropy(probs)`.
**LOC: 10.** Pin: dyadic distributions (e.g. `{0.5, 0.25, 0.25}` → R=0 with `ℓ = {1, 2, 2}`).

**V4. ShannonCodeLengths(probs []float64) []int.**
Returns Shannon-Fano-Elias code lengths `ℓ_i = ⌈−log2(p_i)⌉ + 1`. Always satisfies Kraft (with possible slack); E[L] ≤ H(X) + 1; not optimal (Huffman strictly dominates) but always within 1 bit of entropy.
Composition: `math.Ceil(-math.Log2(p_i)) + 1` per element.
**LOC: 30.** R-MUTUAL pin: `{0.5, 0.25, 0.25}` → `{2, 3, 3}`, `H = 1.5`, `E[L] = 2.5`, redundancy 1.0; matches Cover-Thomas Example 5.4.2.

**V5. CanonicalHuffman(probs []float64) ([]int, [][]byte).**
Returns optimal-prefix-code lengths and canonical-form codewords. Length-limited variant (e.g. cap 15 for DEFLATE compatibility) via Larmore-Hirschberg 1990 package-merge. Achieves `E[L*]` such that `H(X) ≤ E[L*] < H(X) + 1` — saturates Shannon's source coding theorem upper bound.
Composition: priority-queue Huffman tree build (Huffman 1952), DFS for lengths, sort-by-length-then-symbol for canonical codewords.
**LOC: 250 + 30 golden.** Promised in CLAUDE.md/README.md/ARCHITECTURE.md per 041 + 042. R-MUTUAL pin: 3-way `Shannon-Fano-Elias (V4) ≥ Huffman (V5) ≥ entropy (compression.ShannonEntropy)` with strict ordering for non-dyadic distributions.

### Tier B — Arithmetic / range / ANS coders (each ≥150 LOC)

**V6. ArithmeticCodeRoundTrip(symbols []int, probs []float64) ([]byte, []int).**
Witten-Neal-Cleary 1987 *CACM* arithmetic coder: encode and decode a finite symbol sequence using probabilities `probs`. Achieves `−log2(P(x_1...x_n))` bits per message asymptotically (within 2 bits of entropy for the entire message). Coding redundancy ≤ 2/n bits/symbol — strictly dominates Huffman for non-dyadic alphabets.
Composition: integer-arithmetic interval halving in `[0, 2^32)` with carry-lazy renormalisation (Moffat-Neal-Witten 1998). Bit-IO substrate (per 042 T2.18) is a prerequisite — ships inside the same PR.
**LOC: 380 + 60 golden.** R-MUTUAL pin: 3-way against `V5.CanonicalHuffman` (always within 1 bit on E[L]) and `compression.CrossEntropy` (the asymptotic limit).

**V7. NormalizedCompressionDistance(C func([]byte) int, x, y []byte) float64.**
Cilibrasi-Vitanyi 2005 NCD: `(C(xy) − min(C(x), C(y))) / max(C(x), C(y))`. Universal similarity: any black-box compressor `C` plugs in. Reality's `RunLengthEncode` (length output = `C` proxy) works as a worst-case baseline; once V5 ships, `V5.E[L*] * n / 8` bytes is the real proxy. Used by Jiang-Yang 2023 *ACL* "gzip + k-NN" classifier paper that beat BERT on AG News.
Composition: 3 calls to user-supplied `C`, ratio.
**LOC: 30 + 30 golden.** R-MUTUAL pin: NCD(x,x) ≈ 0; NCD(x,y) ≈ NCD(y,x) (small asymmetry from concatenation order is ≤ 1/n).

**V8. RangeCoderRoundTrip(symbols []int, probs []float64) ([]byte, []int).**
Subbotin 1999 carryless range coder. Production-formulation arithmetic coder used by LZMA/7-Zip. Same asymptotic redundancy as V6 but byte-aligned output (faster I/O, marginal compression cost).
Composition: 32-bit range tracking, byte-aligned renormalisation. Reuses bit-IO from V6.
**LOC: 250 + 50 golden.** R-MUTUAL pin: `|RangeCode(x)| − |ArithmeticCode(x)| < 8` bytes for any x (granularity of byte alignment).

**V9. RANSCoderRoundTrip(symbols []int, freqs []int, totalFreq int) ([]byte, []int).**
Duda 2009/2013 rANS (range Asymmetric Numeral Systems). Modern entropy coder. ~5× faster decode than arithmetic at same compression. Foundation of zstd/LZFSE/JPEG XL/AV1. State value updated by `s' = (s/freq_i)*total + (s%freq_i) + cum_i`.
Composition: integer-rational state machine + power-of-2 normalisation table.
**LOC: 200 + 50 golden.** R-MUTUAL pin: 3-way against V6 + V8 — all three within 0.1% on long sequences (the ranking redundancy converges).

### Tier C — Source-coding-theorem witnesses (~60-80 LOC each)

**V10. SourceCodingTheoremCheck(probs []float64) (H, EL_huffman, EL_arith, EL_shannon float64).**
The four canonical numbers from Shannon source coding theorem on a single distribution: entropy, Huffman expected length (V5), arithmetic asymptotic limit (= H, V6), Shannon-Fano-Elias E[L] (V4). Returns the 4-tuple satisfying `H ≤ EL_arith → H ≤ EL_huffman < H + 1 ≤ EL_shannon < H + 2`.
Composition: V4 + V5 + V6 + `compression.ShannonEntropy`.
**LOC: 50.** Pin: textbook examples Cover-Thomas Ch.5 worked problems.

**V11. LZ76EntropyRate(symbols []int, alphabet int) (h_rate float64, err error).**
Wyner-Ziv 1989 LZ-as-entropy-rate: for a stationary ergodic source, `c(n) * log_2(c(n)) / n → H_rate(X)`. Existing `info/lz.LempelZivComplexity` returns `c(n) = WordCount`; this primitive composes it into the entropy-rate estimator.
Composition: `result.WordCount * math.Log2(float64(result.WordCount)) / float64(result.SequenceLength)`. 8-LOC body.
**LOC: 40 + 30 golden.** R-MUTUAL pin: 3-way against V13 KrichevskyTrofimov (Bayes-mixture entropy estimator) and V14 BlockEntropy (empirical block-frequency H_n / n) on Markov-chain-generated sequences with known H_rate.

**V12. EmpiricalCompressibilityVsKL(symbols []int, refProbs []float64) (compressedLen float64, H_empirical float64, KL_to_ref float64).**
Operational meaning of KL divergence as compression overhead. Encode `symbols` with arithmetic coder using `refProbs` (V6); empirical `|C_ref(x)|/n → H(P̂_n) + KL(P̂_n || refProbs)`. The 3-tuple lets the caller verify the asymptotic identity `compressedLen / n − H_empirical ≈ KL_to_ref` for `n → ∞`.
Composition: V6 + `compression.ShannonEntropy(P̂_n)` + `compression.KLDivergence(P̂_n, refProbs)` where `P̂_n` is the empirical distribution.
**LOC: 60 + 40 golden.** R-MUTUAL pin: golden-validated for n=10000 IID samples from a 4-symbol Bernoulli source coded with uniform `{0.25, 0.25, 0.25, 0.25}` ref — convergence to KL ≤ 1/sqrt(n) per Pinsker's inequality.

**V13. KrichevskyTrofimovEstimator(counts []int) (entropy_estimate float64, redundancy_bound float64).**
Krichevsky-Trofimov 1981: Bayesian mixture estimator with Dir(1/2,...,1/2) prior. Estimate `H_KT = -Σ ((c_i + 0.5)/(n + k/2)) * log2((c_i + 0.5)/(n + k/2))`. The KT-mixture code achieves universal coding with redundancy `(k-1)/2 * log2(n) / n + O(1/n)` — the optimal coding-redundancy rate (Rissanen-Langdon 1981).
Composition: 5-LOC entropy formula on smoothed counts; redundancy bound `(k-1)/2 * math.Log2(n) / n`.
**LOC: 50 + 40 golden.** R-MUTUAL pin: 3-way `H_KT vs H_MLE (compression.ShannonEntropy on c_i/n) vs Miller-Madow (H_MLE + (k-1)/(2n*ln2))` — all three within `O(1/n)` of true entropy.

**V14. BlockEntropy(symbols []int, blockSize int) (H_block float64).**
Empirical block-frequency entropy `H_b = -Σ p̂(x_1...x_b) log2(p̂(x_1...x_b))`. The entropy rate `H_rate = lim_{b→∞} H_b / b`. With LZ76 entropy rate (V11) and block entropy (V14), the caller can verify the Shannon-McMillan-Breiman convergence empirically.
Composition: count distinct length-b windows + plug-in entropy.
**LOC: 40 + 30 golden.** Pin: for 2-state Markov chain with transition matrix `[[0.9, 0.1], [0.5, 0.5]]`, H_block / b → 0.708 bits/symbol as b → ∞.

### Tier D — Adaptive / online coders (~120-180 LOC each)

**V15. AdaptiveArithmeticCoder(symbols []int, alphabet int) ([]byte, []int).**
Online arithmetic coder with KT-mixture probability model. Achieves universal-coding redundancy `(k-1)/2 log_2(n)` bits — same as V13 but as an actual encoder, not just an entropy estimator.
Composition: V6 ArithmeticCoder + V13 KrichevskyTrofimov per-symbol counts + Bayesian update. Counts updated after each symbol; coder uses pre-update probabilities for symbol then post-updates.
**LOC: 180 + 50 golden.** Pin: on iid sources, redundancy `≤ (k-1)/2 * log2(n) + 2` bits matches the universal-coding lower bound.

**V16. AdaptiveHuffmanVitter(symbols []int, alphabet int) ([]byte, []int).**
Vitter 1987 *JACM* algorithm V (one-pass adaptive Huffman). Updates the Huffman tree incrementally as each symbol is seen; achieves at most 1 bit/symbol redundancy with online updates.
Composition: V5 CanonicalHuffman + sibling-property tree maintenance.
**LOC: 250 + 50 golden.** Pre-rANS state-of-art for adaptive coding. Tier 2 primitive per 042 T2.5.

### Tier E — Block-sorting / dictionary coders (composition-heavy)

**V17. BurrowsWheelerTransform(data []byte) ([]byte, int).**
BWT (Burrows-Wheeler 1994). Returns the L-column of the cyclic-rotation suffix matrix and the original-string row index. Pairs with V18 MoveToFront and V5 CanonicalHuffman to form bzip2 (Seward 1996).
Composition: SA-IS suffix-array (Nong-Zhang-Chan 2009) for O(n) construction.
**LOC: 250 + 50 golden.** Per 042 T1.11. R-MUTUAL pin: roundtrip BWT(BWT⁻¹(x)) = x.

**V18. MoveToFront(data []byte) []byte.**
MTF (Bentley-Sleator-Tarjan-Wei 1986). Maintains alphabet permutation; emits index of each symbol; promotes to front. After BWT, MTF produces a stream dominated by small indices ⇒ V5 Huffman/V6 arithmetic coder reaps the benefit.
Composition: 16-LOC permutation-tracker.
**LOC: 40 + 30 golden.** Per 042 T1.12. Pairs with V17 and V5/V6 to form bzip2 pipeline (V19).

**V19. Bzip2Pipeline(data []byte) []byte.**
Composition primitive: BWT (V17) → MTF (V18) → RLE (existing `compression.RunLengthEncode`) → CanonicalHuffman (V5) → byte stream. Demonstrates how four V-primitives + one existing primitive compose into a complete codec without any new mathematics. Per 042 T2.17.
**LOC: 200 (mostly glue) + 50 golden vectors.** R-MUTUAL pin: against bzip2 reference impl (subset of vectors that compresses identically; full bzip2 has tunable block-size and byte-level details we deliberately don't replicate).

**V20. LZ77Code(data []byte, windowBits int) []byte + LZ77Decode([]byte) []byte.**
Ziv-Lempel 1977 sliding-window dictionary coder. Emits `(literal | (distance, length))` tokens. RFC 1951 §3.2.5 / Brotli RFC 7932 §9.2 reference. Promised in CLAUDE.md.
Composition: 32-bit hash chain for match search (zlib convention) + lazy matching (Storer-Szymanski 1982 LZSS variant).
**LOC: 380 + 60 golden.** Per 042 T1.6. R-MUTUAL pin: roundtrip + against `info/lz.LempelZivComplexity` for upper bound `|LZ77(x)| / n ≥ H_LZ76(x)` (different parsing strategy, both bounded by H_rate).

---

## 3. Cross-cutting connective-tissue patterns (P1-P3)

**P1. NatsBitsConverter (15 LOC into compression/units.go).**
`NatsToBits(x) = x / math.Ln2`, `BitsToNats(x) = x * math.Ln2`. Single source of truth for the bits-vs-nats conversion. Eliminates the silent footgun where `info/mdl.GaussianCodeLength` returns nats but `compression.CrossEntropy` returns bits — V12 EmpiricalCompressibilityVsKL would silently report off-by-`ln2` results without it.

**P2. LogSumExpStable (12 LOC into compression/logspace.go).**
`LSE(xs []float64) = max + math.Log(Σ math.Exp(x_i - max))`. Reused by V6 ArithmeticCoder cumulative-probability tables (overflow when probs are extremely small) and V13 KrichevskyTrofimov on small-count alphabets. **Already inlined inside `info/mdl/nml.go:computeCn2`**; refactor into shared helper saves 30 LOC across V6/V13/V15. Same identical inlining as `optim/transport/sinkhorn.go:226-260` per synergy 170.

**P3. EmpiricalDistributionFromCounts (20 LOC into compression/empirical.go).**
`EmpiricalProbs([]int) []float64` plus `Smoothed(counts []int, k int, eps float64) []float64`. Used by V12, V13, V15, V19, V20. Eliminates the recurring "divide by sum" pattern that today recurs across `info/mdl/bernoulli.go:80-83`, `info/lz/lz76.go:90-94`, and the natural call sites for V6/V12.

These three patterns total 47 LOC and reduce the V1-V20 total by ~80 LOC of duplicated infrastructure.

---

## 4. Recommended landing order (PR-1 through PR-7)

**PR-1 (one day, 130 LOC, no new dependencies):** V1 KraftMcMillanCheck + V2 ExpectedCodeLength + V3 CodingRedundancy + V4 ShannonCodeLengths. Lands the source-coding-theorem identities as a thin extension to `compression/coding.go`. Closes "the repo has H(X) but no machinery to verify the H ≤ E[L] < H+1 inequality" gap. Saturates **Kraft-witness-vs-expected-length-vs-entropy** R-MUTUAL pin (2 worked examples from Cover-Thomas).

**PR-2 (~3 days, 280 + golden LOC):** V5 CanonicalHuffman + V10 SourceCodingTheoremCheck. Lands the most-promised missing primitive (Huffman) per 041 + 042 and immediately wires it into V10 to demonstrate the theorem works. Saturates **Shannon-Fano-vs-Huffman-vs-entropy** R-MUTUAL pin.

**PR-3 (~5 days, 380 + 60 golden + 50 LOC bit-IO):** V6 ArithmeticCodeRoundTrip + bit-IO substrate (per 042 T2.18). Lands the math-richest entropy coder. Composes with V5 to demonstrate tight 3-way bound `H ≤ Huffman ≤ Arithmetic_asymptotic`.

**PR-4 (~2 days, 130 LOC):** V7 NormalizedCompressionDistance + V11 LZ76EntropyRate. Both are 30-50 LOC compositions over existing primitives. V7 wraps V5 (or any compressor); V11 wraps `info/lz.LempelZivComplexity`. Saturates **NCD-symmetry** + **LZ76-rate-vs-empirical-H** R-MUTUAL pins.

**PR-5 (~2 days, 110 LOC):** V12 EmpiricalCompressibilityVsKL + V13 KrichevskyTrofimovEstimator + V14 BlockEntropy. Operational compression-overhead semantics for KL. Saturates **cross-entropy-vs-arithmetic-coding-overhead** R-MUTUAL pin.

**PR-6 (~5 days, 430 LOC):** V15 AdaptiveArithmeticCoder + V16 AdaptiveHuffmanVitter. Online universal coders.

**PR-7 (~7 days, 670 LOC + 130 golden):** V17 BurrowsWheelerTransform + V18 MoveToFront + V19 Bzip2Pipeline + V20 LZ77Code. The block-sorting + dictionary axis. V19 is glue; the math-novel content is V17 SA-IS and V20 LZ77 lazy matching.

**Total: ~3050 LOC source + ~1100 LOC golden vectors over ~24 engineer-days.** Lands six R-MUTUAL pins listed in the two-line summary plus a seventh **MDL-codelength-vs-empirical-compressed-size** pin once V12 + V19 ship together.

---

## 5. Cycle-free DAG check

Today's import graph for the relevant packages:

- `compression/` imports `math` only.
- `info/lz/` imports `math` only.
- `info/mdl/` imports `math` + `errors` only.
- `info/lz/` and `info/mdl/` are siblings (per `info/mdl/doc.go:38-40` "co-located but disjoint in scope").

After V1-V20 land:

- `compression/` keeps `math`-only for V1-V6 and V8-V10 (Tier A + Tier B coders).
- `compression/` imports `info/lz` for V7 NormalizedCompressionDistance (`info/lz` provides a black-box compressor proxy via `LempelZivComplexity.WordCount * log2(WordCount)`).
- New `info/entropy/` sub-package (NOT inside `compression/`) hosts V11-V14 entropy estimators; imports both `info/lz` and `compression`.
- V15-V20 stay in `compression/` (production coders).

Cycle-free: `info/entropy/` → `{compression/, info/lz/, info/mdl/}` and `compression/` → `info/lz/` only. Neither `info/lz` nor `info/mdl` ever imports `compression/` (mathematical priors don't depend on coders).

**Proposed alternative placement:** keep V11-V14 inside `compression/estimators.go` per 042's M12 estimator-package recommendation, leaving `info/` strictly for codelength + LZ76. This avoids creating `info/entropy/` and is more aligned with the 042 ladder. **Recommended.**

---

## 6. Six R-MUTUAL-CROSS-VALIDATION 3/3 pins falling out

Mirrors the validation-rigour-saturation idioms in commits **6a55bb4 (audio/onset 3-detector)** and **365368a (Clayton autodiff vs analytic)** per CLAUDE.md sec1 "golden files are the proof":

1. **Shannon-Fano-vs-Huffman-vs-arithmetic** on 16 distributions: `H(X) ≤ ArithmeticAsymp ≤ HuffmanE[L*] ≤ ShannonFanoE[L] ≤ H(X)+2` for all 16. Tight when distribution is dyadic; gap = `KL(p || dyadic-rounding(p))` for non-dyadic.

2. **Kraft-witness-vs-expected-length-vs-entropy** on 12 prefix-code length vectors: Kraft inequality holds (V1 slack ≥ 0), `E[L]` per V2 satisfies `H(X) ≤ E[L]`, and the slack is consumed exactly by sub-optimal codes. Cover-Thomas Ch.5 worked examples reproduce.

3. **NCD-symmetry** on 8 string pairs: `|NCD(x, y) − NCD(y, x)| ≤ 2/min(|x|,|y|)`. Exact symmetry only in the limit |x|, |y| → ∞ per Cilibrasi-Vitanyi 2005 §3.

4. **LZ76-rate-vs-empirical-H** on 4 stationary ergodic sources (Bernoulli IID, 2-state Markov, 3-state Markov, Hidden Markov 2x3): Wyner-Ziv 1989 `c(n)log_2(c(n))/n → H_rate(X)` to ≤ 0.05 bits/symbol at n=10000 across all four. Block entropy V14 at b=10 corroborates.

5. **Cross-entropy-vs-arithmetic-coding-overhead** on 6 (source, code) pairs: `|C_ref(x)|/n - H(P̂_n) → KL(P̂_n || ref)` to ≤ 1/sqrt(n) per Pinsker. Operationalises KL-as-compression-overhead.

6. **MDL-codelength-vs-empirical-compressed-size** on 5 (model_class, data) pairs: `BICShape(NLL_at_MLE, k, n) ≈ |Bzip2Pipeline(x)| * 8 / n` within `O(log(n)/n)` for AR(p) and Bernoulli sources. Witnesses MDL ≡ universal coding (Rissanen 1996 stochastic-complexity equivalence).

All six pins ship as golden-file vectors per CLAUDE.md sec3 "min 20 / target 30" floor.

---

## 7. Differentiation from sibling reviews

- **042 (compression-missing):** ladder of 80-90 named primitives. **Identical primitive scope** for V5 (Huffman) / V6 (arithmetic) / V8 (range) / V9 (rANS) / V11 (LZ76 estimator — already lives in `info/lz`) / V17 (BWT) / V18 (MTF) / V19 (bzip2) / V20 (LZ77). **THIS review** narrows to the Shannon-source-coding-theorem witness primitives V1-V4 + V10 + V12 + V13 + V14 that 042 deferred to estimator-section M12; sharpens the connective-tissue argument by showing the "info x compression" half of those primitives is 0 LOC of new mathematics on the coding side.

- **041 (compression-numerics):** existing-surface audit. Notes the lies in CLAUDE.md/README.md/ARCHITECTURE.md re Huffman/LZ77. **THIS review** does not duplicate the truth-fix argument (defer to 041) but consumes its bottom line that Huffman/LZ77 must land before any V5/V20 description-correctness claim.

- **170 (synergy-info-prob):** Blahut-Arimoto channel capacity, R(D) rate-distortion, water-filling, MINE/InfoNCE variational MI, Fano. **Adjacent axis** — uses the same `compression.MutualInformation` and `compression.KLDivergence` primitives but on a different mathematical structure (capacity vs source coding). 170 owns rate-distortion-as-compression and information-bottleneck which would otherwise belong here; this review explicitly cross-links and does not duplicate.

- **117 (prob-missing):** distribution gaps. **Orthogonal axis** — distributional surface needed by V13 KrichevskyTrofimov is already in repo (`prob.LogGamma` exists; no new dependency).

- **086-090 (info-numerics/missing/sota/api/perf):** if these per-package isolation reviews exist (per the MASTER_PLAN.md numbering convention), they cover `info/mdl/` and `info/lz/` defects but NOT the cross-edge to `compression/`. THIS review never duplicates; only composes.

- **127-128 (sequence-missing/sota):** sequence-similarity (NGram dice, Soundex, etc., per recent commits 85a80db / 1e12e80 / 3b8413a). NCD (V7) is the universal-similarity counterpart but shipped via compression. Cross-link only at "string distance" but distinct mathematical machinery.

---

## 8. Precision and numerical hazards

Documented per CLAUDE.md sec5 "precision documented, not assumed":

- **V1 KraftMcMillanCheck:** for `lengths[i] > 53`, `math.Exp2(-float64(L))` underflows to subnormal then 0 — sum is still numerically valid but loses any contribution from those tail symbols. Document; in practice lengths > 32 are rare (DEFLATE caps at 15, Brotli at 16 per RFC 7932).

- **V2/V3 ExpectedCodeLength/CodingRedundancy:** for catastrophic-cancellation when `E[L] - H(X)` is < 1e-12 but neither is near 0, prefer Kahan summation. **Pin: redundancy = 0 only when all p_i are powers of 2 (dyadic distribution).**

- **V4 ShannonCodeLengths:** for `p_i = 0`, `−log2(0) = +Inf` ⇒ length = MaxInt32. Decision: skip with `length = 0` (Cover-Thomas convention) or filter out p=0 symbols upstream. **Recommended: filter upstream; emit error if any p_i = 0.**

- **V5 CanonicalHuffman:** for length-limited variant (cap = 15), if entropy exceeds 15 bits/symbol the Larmore-Hirschberg package-merge fails — caller must filter low-probability symbols (escape code) or raise cap. Document.

- **V6 ArithmeticCoder:** integer-arithmetic precision is 32-bit by default; for total probability sums > 2^14 lose the 17-bit safety margin per Moffat-Neal-Witten 1998. **Recommend 64-bit interval for any frequency table with `total > 2^30`.**

- **V11 LZ76EntropyRate:** Wyner-Ziv convergence is slow — for `n = 1000`, gap to `H_rate` is `O(1/log n) ≈ 0.14 bits/symbol`. Pin tolerance accordingly. **For tighter convergence prefer V13 KrichevskyTrofimov or V15 AdaptiveArithmetic.**

- **V13 KrichevskyTrofimov:** Dir(1/2) prior is the Jeffreys prior; for highly-skewed distributions (one symbol near probability 1) the smoothing biases entropy upward by `(k-1)/(2n*ln2)` — the classical Miller-Madow correction is the bias-corrected MLE which agrees with KT to leading order.

- **V17 BurrowsWheelerTransform:** SA-IS construction has worst-case `O(n)` time but the constant factor on adversarial inputs (highly-repeated DNA-like sequences with periodic structure) is high; document `n ≤ 10^7` practical cap matching `info/lz.LZ76MaxSymbols = 10^4` order-of-magnitude convention.

- **V20 LZ77Code:** lazy-matching LZSS variant has sub-optimal parsing on adversarial inputs vs. optimal-parsing dynamic programming (Kärkkäinen-Sutinen 2010); document the gap as ≤10% on typical text inputs, larger on synthetic worst cases.

---

## 9. Cross-language pinning targets

Per CLAUDE.md sec1 "golden files are the proof" cross-substrate convention:

- **V5 CanonicalHuffman:** zlib reference impl (`deflate.c::build_tree`), Python `huffman` library, RFC 1951 §3.2.2 worked examples.
- **V6 ArithmeticCoder:** Witten-Neal-Cleary 1987 *CACM* worked example (the canonical reference impl used in every textbook).
- **V7 NormalizedCompressionDistance:** Cilibrasi-Vitanyi 2005 Tables 2-4 reference values (using gzip as black-box compressor); cross-pinned to the gzip + k-NN paper Jiang-Yang 2023 *ACL* on AG News fixture.
- **V9 RANSCoderRoundTrip:** Duda 2014 reference C++ impl, FSE (Collet 2014).
- **V11 LZ76EntropyRate:** Wyner-Ziv 1989 *IEEE TIT* worked example (Bernoulli `p=0.3`, n=10^5, target H ≈ 0.881 bits/symbol within 0.05).
- **V13 KrichevskyTrofimov:** Krichevsky-Trofimov 1981 *IEEE TIT* fig.1.
- **V17 BurrowsWheelerTransform:** Burrows-Wheeler 1994 SRC report Figure 2 example string `^BANANA|`.
- **V20 LZ77Code:** zlib reference impl `deflate.c::longest_match` + RFC 1951 §3.2.5 worked example.

All have public-API equivalents in scipy / numpy / R / Mathematica / SageMath / Julia / boost / zlib / zstd reference impls, pinning V5/V6/V7/V9/V11/V13/V17/V20 to ≤1e-10 vs reference per CLAUDE.md sec1.

---

## 10. Bottom line

This synergy is **highest-leverage-by-promise-debt** in the synergy-review cohort 158-181: every other Block-B synergy reviewed so far targets primitives the per-package missing-list flagged as gaps but never promised. **Compression × info uniquely has primitives the documentation actively claims to ship** (Huffman + LZ77 in CLAUDE.md/README.md/ARCHITECTURE.md per 041) **while the source files contain zero lines.** The PR-1 + PR-2 + PR-3 ladder lands the truth-fix while simultaneously realising the Shannon source coding theorem witnesses — `H ≤ E[L*] < H+1` cannot be empirically verified today; landing V1-V6 makes it a 5-minute golden-test execution. PR-4 + PR-5 then operationalise the existing `compression.KLDivergence` and `compression.CrossEntropy` as actual compression overhead measurements rather than abstract divergences.

The mathematics is **all 1948-2014 textbook material** (Shannon source coding 1948, Kraft 1949 / McMillan 1956, Huffman 1952, Lempel-Ziv 1976/77/78, Krichevsky-Trofimov 1981, Witten-Neal-Cleary 1987, Burrows-Wheeler 1994, Cilibrasi-Vitanyi 2005, Duda 2009/2013, Wyner-Ziv 1989). **Zero new abstractions required** of either package — `compression.ShannonEntropy` already exists; `info/lz.LempelZivComplexity` already exists; the V-primitives are the missing Hammings between them.

**Recommended PR-1 if only one ships:** V1 + V2 + V3 + V4 = 130 LOC. Lands four Shannon source coding theorem identities, saturates one R-MUTUAL pin, requires zero new packages, has zero numerical hazards beyond underflow at length > 53, and unblocks every subsequent V-primitive that needs the entropy↔code-length conversion. Same architectural-witness strategy as commits 6a55bb4 / 365368a / 1e12e80.

---

## Files referenced

- `C:\limitless\foundation\reality\compression\entropy.go` (177 lines, 6 functions: H/Joint/Cond/MI/KL/Cross)
- `C:\limitless\foundation\reality\compression\coding.go` (104 lines, 4 functions: RLE+Delta encode/decode)
- `C:\limitless\foundation\reality\compression\quantize.go` (99 lines, 2 functions)
- `C:\limitless\foundation\reality\info\lz\lz76.go` (466 lines, LempelZivComplexity + symbolisation)
- `C:\limitless\foundation\reality\info\lz\doc.go` (136 lines)
- `C:\limitless\foundation\reality\info\mdl\nml.go` (157 lines, NMLMultinomial + computeCn2)
- `C:\limitless\foundation\reality\info\mdl\bernoulli.go` (90 lines, NMLBernoulli + BernoulliCodeLength)
- `C:\limitless\foundation\reality\info\mdl\codelength.go` (128 lines, Gaussian/Model/BIC/AIC)
- `C:\limitless\foundation\reality\info\mdl\universal_int.go` (71 lines, Rissanen log*)
- `C:\limitless\foundation\reality\info\mdl\select.go` (74 lines, SelectMDL)
- `C:\limitless\foundation\reality\info\mdl\doc.go` (170 lines)

## Cross-references (within overnight-400)

- 041 (compression-numerics): existing-surface audit; flags Huffman/LZ77 promise-debt
- 042 (compression-missing): 80-90-primitive ladder; THIS review composes the info-x-compression subset of M1/M2/M8/M9/M10/M12
- 044 (compression-api): naming inconsistencies bits-vs-nats — P1 NatsBitsConverter directly addresses
- 087 (info-missing if exists): info/ surface gaps
- 117 (prob-missing): orthogonal — Krichevsky-Trofimov V13 needs prob.LogGamma which already exists
- 170 (synergy-info-prob): rate-distortion / Blahut-Arimoto / MINE — adjacent axis, explicit cross-link at V11/V12
- 127-128 (sequence-missing/sota): NGram/Soundex sequence-similarity — V7 NCD adjacent but distinct
