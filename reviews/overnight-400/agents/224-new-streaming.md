# 224 | new-streaming

**Topic:** Streaming / sketching algorithms — heavy-hitters (Misra-Gries 1982, Space-Saving Metwally-Agrawal-Abbadi 2005, Manku-Motwani Lossy Counting 2002), frequency sketches (Count-Min — Cormode-Muthukrishnan 2005, Count Sketch — Charikar-Chen-Farach-Colton 2002, AMS — Alon-Matias-Szegedy 1996, Conservative-Update CM — Estan-Varghese 2002), cardinality (HyperLogLog — Flajolet-Fusy-Gandouet-Meunier 2007, HLL++ — Heule-Nunkesser-Hall 2013, Linear Counting — Whang-Vander-Zanden-Taylor 1990, k-Minimum-Values, MinHash — Broder 1997), quantiles (t-digest — Dunning-Ertl 2019, KLL — Karnin-Lang-Liberty 2016, Greenwald-Khanna 2001, Q-digest — Shrivastava et al. 2004), reservoir sampling (Algorithm R — Vitter 1985, Algorithm Z — Vitter 1985, weighted A-Res — Efraimidis-Spirakis 2006), set membership (Bloom 1970, Counting Bloom — Fan-Cao-Almeida-Broder 2000, Cuckoo Filter — Fan-Andersen-Kaminsky-Mitzenmacher 2014, XOR Filter — Graf-Lemire 2020, Quotient Filter — Bender et al. 2012, Scalable Bloom — Almeida-Baquero 2007), online matrix sketches (Frequent Directions — Liberty 2013, SRHT, count-sketch JLT — Clarkson-Woodruff 2013), streaming triangles (Pavan et al. 2013), differentially-private CMS (Chan-Shi-Song 2011 / Apple's CMS-DP 2017), DataSketches Apache library reference. **Block:** C (cutting-edge math, what reality is missing). **Date:** 2026-05-08.

**Scope:** the **bounded-memory single-pass-over-stream sublinear-space approximate-aggregate** axis — distinct from 220 (`finite-sum stochastic optimization, full-batch in memory`), 221 (`online-convex-optim regret minimization, decision-then-loss`), 222 (`bandits, action-selection feedback`), 223 (`submodular set-function maximization, oracle access`), 261 (`online SVD streaming PCA, dense matrix updates`), 262 (`randomized NLA sketches, batch with sampling`). 224 owns **bounded-memory aggregates over data streams**: count-distinct, frequency, quantile, top-k, set-membership, sample, second-moment, on a stream you cannot rewind.

## Two-line summary

`reality/` v0.10.0 ships **exactly ONE streaming primitive** (`audio.Fingerprint` Welford 1962 mean+variance via `audio/fingerprint.go:24-76` with parallel-merge Chan-Golub-LeVeque at line 190) and **ZERO sketches** — verified by repo-wide grep returning zero source-code matches for `HyperLogLog|HLL|CountMin|Count-Min|Misra-Gries|MisraGries|tdigest|t-digest|KLL|Greenwald-Khanna|Bloom|CuckooFilter|XORFilter|QuotientFilter|FrequentDirections|CountSketch|AMS|reservoir|Vitter|MinHash|kMinValues|SpaceSaving|LossyCounting|LinearCounting|StreamingTriangles|DPSketch` across all 22 packages (only seven review-document mentions in MASTER_PLAN+218+220+221+222+223 and prior synergies); the underlying machinery is **already 90% present** — `crypto/hash.go:25/40/61` ships `FNV1a32/FNV1a64/MurmurHash3_32` (the canonical CMS+HLL+Bloom+MinHash hash families) and `audio/fingerprint.go:59-76` already demonstrates the canonical `(state-struct, Update, Merge, Query)` idiom that every sketch in this canon follows; **23 primitives ST1-ST23 totalling ~3,250 LOC of pure connective tissue** across NEW `streaming/{sketch.go,heavyhitter.go,cardinality.go,quantile.go,membership.go,sample.go,moment.go,matrix.go,private.go}` — every primitive is a 60-220 LOC composition of `crypto/hash.go` (hash families) + `prob/distributions.go` (variance bounds, gamma/poisson for tail-bound proofs) + `linalg/matrix.go` (FrequentDirections SVD via existing `linalg.QRDecompose`) + `combinatorics/generate.go` (random k-wise hash families) + `audio/fingerprint.go` (Welford pattern as established in-repo idiom). Cheapest one-day standalone is **PR-1 ST1-Sketch-interface + ST2-CountMin + ST3-MisraGries + ST10-HyperLogLog (~520 LOC)** which lands the **first sketch-anything in the repo** and saturates **R-CMS-OVERESTIMATE-1/1** (CMS estimate is provably ≥ true frequency on any stream) plus **R-HLL-RELATIVE-ERROR-1/1** (HLL register variance matches Flajolet-Fusy-Gandouet-Meunier-2007 σ ≈ 1.04/√m). Highest-leverage one-week unlock is **PR-2 ST5-Bloom + ST7-CuckooFilter + ST15-tDigest + ST17-KLL + ST20-ReservoirR + ST21-WeightedReservoir-Efraimidis-Spirakis (~1,180 LOC)** because they collectively saturate the **R-MERGEABLE-SKETCH 6/6** pin (CMS, HLL, Bloom, t-digest, KLL, FrequentDirections all merge associatively + commutatively to byte-identical state — the keystone property that makes sketches map-reducible across shards). Architectural keystone is the **`Sketch[T]` interface (~30 LOC)** — `Update(x T) / Merge(other Sketch[T]) / Query(...) / Bytes() / FromBytes([]byte)` — together with the `Hashing` substrate (3 lines wrapping `crypto.MurmurHash3_32` into k-wise independent families via Carter-Wegman 1979). Co-shipped with 174-G4 `OnlineLearner` + 220-F1 `FiniteSumLoss` + 221-O1 `OnlineConvexLearner` + 222-B1 `StochasticBandit` + 223-S1 `SetFunction` as the **sixth interface in the unified-substrate keystone** consumers reach for first.

---

## 0. State of play (verified file-walk)

### `reality/` streaming surface = ONE primitive (Welford in audio)

Verified by direct read of `audio/fingerprint.go`:

- `Fingerprint{N int; Mean, M2 []float64}` (`fingerprint.go:24-28`) + `UpdateFingerprint(fp, x)` (line 59) + `FingerprintVariance(fp, out)` (line 80) + parallel `MergeFingerprints` Chan-Golub-LeVeque (line 190).
  - This IS the canonical streaming-statistic primitive (Welford 1962; Knuth TAOCP vol 2 §4.2.2).
  - It ALSO establishes the in-repo idiom every sketch will follow: **(struct exporting fields for zero-alloc updates, `Update` mutates in place, `Merge` is associative + commutative, `Query` is a separate read function).** This is exactly the DataSketches mergeable-sketch contract.

Verified ABSENT (repo-wide grep on every `.go` file — match counts are zero):

- **No frequency sketches:** zero matches for `CountMin`, `Count-Min`, `CMS`, `CountSketch`, `Count-Sketch`, `AMS`, `Alon-Matias-Szegedy`, `ConservativeUpdate`, `CU-CMS`, `F2`, `secondMoment`, `frequencyMoment`.
- **No heavy-hitters:** zero matches for `MisraGries`, `Misra-Gries`, `SpaceSaving`, `Space-Saving`, `LossyCounting`, `Lossy-Counting`, `topK`, `top-k`, `heavyHitter`, `frequentItem`, `Manku-Motwani`, `Metwally`.
- **No cardinality / count-distinct:** zero matches for `HyperLogLog`, `HLL`, `HLL++`, `LogLog`, `LinearCounting`, `Linear-Counting`, `kMinValues`, `KMV`, `MinHash`, `Min-Hash`, `Broder`, `bottomK`, `bottom-k`.
- **No quantile sketches:** zero matches for `tdigest`, `t-digest`, `Dunning`, `KLL`, `Karnin-Lang-Liberty`, `GreenwaldKhanna`, `Greenwald-Khanna`, `GK-quantile`, `qDigest`, `q-digest`, `MRL`, `Munro-Paterson`, `streamingQuantile`. (Existing `prob/conformal.SplitQuantile` requires the entire batch in memory — sort-based, O(n) RAM.)
- **No set-membership filters:** zero matches for `Bloom`, `BloomFilter`, `bloom-filter`, `CountingBloom`, `ScalableBloom`, `partitionedBloom`, `Cuckoo`, `cuckoo-filter`, `CuckooFilter`, `XORFilter`, `xor-filter`, `Graf-Lemire`, `QuotientFilter`, `quotient-filter`, `Bender-Farach-Colton`, `RibbonFilter`. (Note: `crypto/structural_hash_test.go` references hashing but zero filter primitives.)
- **No reservoir / streaming sampling:** zero matches for `reservoir`, `Reservoir`, `Vitter`, `algorithmR`, `Algorithm-R`, `algorithmZ`, `Algorithm-Z`, `Efraimidis-Spirakis`, `weighted-reservoir`, `weightedReservoir`, `A-Res`, `A-ExpJ`, `distinctSampling`. (Reality's only sample-related code is `combinatorics/generate.go:RandomSubset` which expects the full ground set — not a stream.)
- **No streaming linear algebra:** zero matches for `FrequentDirections`, `frequent-directions`, `Liberty2013`, `Edith-Liberty`, `SRHT`, `subsampledRandomizedHadamard`, `streamingPCA`, `streamingSVD`, `incrementalSVD`. (261 will own those — but the FrequentDirections one-pass column-space approximation belongs squarely here as a sketch.)
- **No streaming-graph triangles / motifs:** zero matches for `streamingTriangles`, `triangleSketch`, `Pavan`, `BarYossef`, `Doulion`, `Tsourakakis`, `motifCount`. (graph/ has full-batch triangle counting via adjacency matrix only.)
- **No private sketches:** zero matches for `differentialPrivacy`, `differential-privacy`, `DP-CMS`, `DP-HLL`, `Apple-CMS`, `RAPPOR`, `Erlingsson`, `randomizedResponse`, `Warner1965`. (No DP primitive ANYWHERE in repo — 224 introduces the first.)
- **No streaming entropy:** zero matches for `streamingEntropy`, `Chao-Shen`, `Lopez`, `Miller-Madow`, `NSB`, `BUB`, `entropySketch`. (`compression/entropy.go:ShannonEntropy` requires the full distribution in memory — closed-form on the histogram, not an estimator.)

### Primitives already in repo that 224 composes

The remarkable observation: **every sketch in the canon decomposes to at most three existing reality primitives.**

- `crypto/hash.go:25,40,61` — `FNV1a32 / FNV1a64 / MurmurHash3_32`. These are the canonical hash families used by every sketch in this canon. CMS needs `d` independent hash functions (Carter-Wegman tabulation or k-wise polynomial); HLL needs one good 64-bit hash with avalanche; Bloom needs `k` hashes (double-hashing trick — Kirsch-Mitzenmacher 2008); MinHash needs `k` independent permutations; t-digest needs no hashing. **All five MurmurHash3-derived.**
- `prob/distributions.go:NormalQuantile / NormalCDF / GammaCDF / PoissonPMF` — provide the tail-bound machinery for proving the (ε, δ)-guarantees on every sketch. CMS uses Markov-on-expectation for over-estimate bound; HLL uses the harmonic-mean bias-correction derived from Poisson approximation.
- `audio/fingerprint.go:59-194` — the established `(state, Update, Merge, Query)` idiom + parallel-merge formula (Chan-Golub-LeVeque 1979). Every sketch follows this exact shape.
- `linalg/matrix.go` + `linalg/decompose.go:QRDecompose` — needed by Frequent Directions (one-pass SVD-truncate-and-shrink at the 2ℓ-by-d boundary).
- `combinatorics/generate.go:RandomSubset / Permute` — needed for k-wise hash family construction (random tabulation tables).
- `compression/entropy.go:ShannonEntropy / KLDivergence` — needed by the AMS-F2-ratio test (entropy estimator from second-moment sketch).
- `signal/fft.go` — needed by **CountSketch-FFT-acceleration** (Cormode-Hadjieleftheriou 2008 fast top-k).
- `optim/proximal/operators.go:ProxL1` — needed by **L1-error compressed sensing × CountSketch** for the iceberg-query family.

There is essentially zero standalone math to write: **224 is 90% glue.**

---

## 1. The 23-primitive surface ST1-ST23

A new top-level package `streaming/` (not `prob/streaming/` — sketches are dimension-free, language-agnostic primitives, peer to `compression/` and `crypto/`). Files in proposed order:

### Substrate (~280 LOC)

- **ST1 — `Sketch[T]` interface** (`streaming/sketch.go`, ~30 LOC). The keystone: `Update(x T)`, `Merge(other Sketch[T])` (must be commutative + associative), `Query(...)`, `Bytes() []byte`, `FromBytes([]byte) error`. The Bytes/FromBytes contract is what makes sketches **map-reducible across shards** — DataSketches calls this the "mergeable summary" property. Co-shipped with 174-G4/220-F1/221-O1/222-B1/223-S1 as the sixth keystone interface — single substrate for the entire 220-224 review block.
- **ST2 — `Hashing` substrate** (`streaming/hashing.go`, ~120 LOC). Carter-Wegman 1979 k-wise-independent polynomial hash family over Mersenne primes (2³¹-1 and 2⁶¹-1) + Kirsch-Mitzenmacher 2008 double-hashing trick (`h_i(x) = h_a(x) + i·h_b(x)` reduces k MurmurHash3 calls to 2). Wraps `crypto/hash.go:MurmurHash3_32`. Used by CMS (d hashes), Bloom (k hashes), Cuckoo (2 hashes), MinHash (k hashes), CountSketch (d sign × d bucket).
- **ST3 — `RNGSampler` interface** (~30 LOC, = 169-S14 = 195-N3). Already named twice; finally lands here as the `Reservoir` family needs uniform `Float64() ∈ [0,1)` and `Int(n) ∈ [0,n)`.
- **ST4 — Welford promotion** (~100 LOC). Move `audio.Fingerprint` → `streaming.WelfordSketch` (re-export with audio shim — zero churn for audio consumers via type alias). Establishes 224 ownership of the streaming-stat canon. Adds `WelfordExtended` for skewness + kurtosis (Pébaÿ 2008) and `WelfordCovariance` for streaming covariance matrix (Welford 1962 vector form).

### Frequency sketches (~520 LOC)

- **ST5 — `CountMinSketch`** (~140 LOC, Cormode-Muthukrishnan 2005). `d × w` table where `w = ⌈e/ε⌉` and `d = ⌈ln(1/δ)⌉`. `Update` adds 1 to `d` cells; `Estimate` returns the min. Provably **always over-estimates** the true count with prob ≥ 1-δ. With `Conservative Update` toggle (Estan-Varghese 2002): only the min cell increments — empirically 5-10× tighter for skewed streams. Supports negative updates (signed arrival/departure stream) when toggled.
- **ST6 — `CountSketch`** (~140 LOC, Charikar-Chen-Farach-Colton 2002). Like CMS but each cell takes `±1` sign; `Estimate` returns the median across `d` rows. Estimator is **unbiased** (CMS is biased upward); error scales with L₂ instead of L₁ — better for skewed streams once heavy hitters dominate. The mathematical foundation for FrequentDirections + AMS sketch + JL-via-CountSketch (Clarkson-Woodruff 2013).
- **ST7 — `MisraGries`** (~90 LOC, Misra-Gries 1982). The deterministic-pre-quel of Space-Saving. Maintains `k` counter slots; `Update(x)` either increments existing slot, fills empty slot, or decrements ALL slots. Guarantees: any item with frequency > N/(k+1) appears in the slots. Total memory: `k` items, no hashing. **Finds heavy hitters with one pass and zero randomness.**
- **ST8 — `SpaceSaving`** (~90 LOC, Metwally-Agrawal-Abbadi 2005). Strict improvement over Misra-Gries: maintains `k` (item, count, error-bound) triples; on miss, replace the minimum and inherit its count as error bound. Empirically 2-3× tighter than Misra-Gries on Zipfian streams. The de-facto top-k algorithm in industrial deployments (Apache DataSketches, Apache Flink ApproxFreq).
- **ST9 — `LossyCounting`** (~70 LOC, Manku-Motwani 2002). The first-published streaming heavy-hitter algorithm. Buckets-of-width-`⌈1/ε⌉` decrement schedule. Important pedagogical baseline; usually dominated by SpaceSaving in practice but historically load-bearing.
- **ST10 — `AMSSketch`** (~90 LOC, Alon-Matias-Szegedy 1996). The original tug-of-war estimator for the **second frequency moment F₂ = Σ f_i²** = self-join size = L₂². `r × s` table; each cell holds `Σ ε_i(x) · count(x)` for ε_i ∈ {±1}. `Estimate` = median of row-sums-squared. The lower-bound-witness: AMS is provably space-optimal for F₂. Generalizes to F_k via tighter samplers (Indyk-Woodruff 2005) — listed as a stretch primitive ST10b.

### Cardinality (~430 LOC)

- **ST11 — `LinearCounting`** (~60 LOC, Whang-Vander-Zanden-Taylor 1990). Bitmap of `m` bits; hash sets bit; `Estimate = -m·ln(empties/m)`. Best for **small cardinalities** (n ≪ m) where HLL has high relative error. Apache Druid uses LinearCounting < 1e4, HyperLogLog ≥ 1e4 (Heule 2013 hybrid).
- **ST12 — `HyperLogLog`** (`streaming/hyperloglog.go`, ~180 LOC, Flajolet-Fusy-Gandouet-Meunier 2007). `m` registers each holding `max(leading_zeros + 1)` of `MurmurHash3_64(x) mod m`. Estimate = αₘ · m² / Σ 2^(-Mⱼ). Standard error 1.04/√m — `m=1024` (1KB sketch) gives ±3% on cardinalities up to 10⁹. Bias correction at low cardinality via LinearCounting hand-off (Heule-Nunkesser-Hall 2013 "HLL++").
- **ST13 — `HyperLogLogPlusPlus`** (~80 LOC, Heule-Nunkesser-Hall 2013). Sparse representation (sorted list of register indices) for low cardinality + dense fallback + bias-correction LUT for the [3m, 5m] transition zone. The Google production HLL.
- **ST14 — `MinHash` + `kMinValues`** (~110 LOC, Broder 1997 + Beyer-Haas-Reinwald-Sismanis-Gemulla 2007). Maintain `k` smallest hashes. Two sketches: Jaccard ≈ |A_∩|/|A_∪| ≈ #(matching min-hash slots)/k; Cardinality ≈ k / max(top k hash values). The substrate of Broder near-duplicate document detection + Datasketches Theta-Sketch family.

### Quantiles (~470 LOC)

- **ST15 — `tDigest`** (`streaming/tdigest.go`, ~220 LOC, Dunning-Ertl 2019). Cluster-based quantile sketch using a scale function k(q) = δ · arcsin(2q-1) / (2π) + k₀ that concentrates resolution near the extremes (1, 99, 99.9 percentiles get more bins than the median). Critical for **monitoring tail latencies** — every observability stack uses t-digest. Mergeable; serializable; bounded memory δ (typical δ=100 → 100 clusters).
- **ST16 — `KLLSketch`** (~150 LOC, Karnin-Lang-Liberty 2016). The **provably space-optimal** quantile sketch: O((1/ε)·log²log(1/δ)) space for (ε, δ)-quantiles. Hierarchical compactors with random sub-sampling. The DataSketches default. Strictly dominates Greenwald-Khanna in space.
- **ST17 — `GreenwaldKhanna`** (~100 LOC, Greenwald-Khanna 2001). The original deterministic quantile sketch — O((1/ε)·log(εN)) space. Listed for pedagogical completeness + deterministic guarantee (no randomness).

### Set membership (~440 LOC)

- **ST18 — `BloomFilter`** (`streaming/bloom.go`, ~110 LOC, Bloom 1970). `m` bits + `k` hashes. False-positive rate (1-e^(-kn/m))^k optimised at `k = (m/n)·ln 2`. Counting-Bloom variant (Fan-Cao-Almeida-Broder 2000) extends to deletable sets via 4-bit counters. Scalable Bloom (Almeida-Baquero-Preguiça-Hutchison 2007) auto-resizes when fill ratio exceeds threshold.
- **ST19 — `CuckooFilter`** (~120 LOC, Fan-Andersen-Kaminsky-Mitzenmacher 2014). Modern Bloom replacement: supports deletion natively, ~25% smaller for FPR ≤ 3%, faster lookup. Stores `f`-bit fingerprints in a `(2,4)`-cuckoo-hashed table; relocation on insert via the partial-key trick `i₂ = i₁ ⊕ hash(fingerprint)`.
- **ST20 — `XORFilter`** (~100 LOC, Graf-Lemire 2020). Provably space-optimal static membership filter: 1.23·log₂(1/ε) bits per element (vs Bloom's 1.44). Construction is offline (3-XOR-SAT-style peeling); query is `O(1)` and 25% faster than Cuckoo. Best for read-heavy static sets (compiled rule databases, address books).
- **ST21 — `QuotientFilter`** (~110 LOC, Bender-Farach-Colton-Goswami-Johnson-McCauley-Singh 2012). Cache-friendly Bloom alternative; supports merging two filters in O(m+m') instead of Bloom's O(m·k). The basis of RocksDB's Ribbon Filter (Dillinger-Farach-Colton-Walzer 2021 — listed as stretch ST21b).

### Sampling (~310 LOC)

- **ST22 — `ReservoirR`** (`streaming/reservoir.go`, ~80 LOC, Vitter 1985 Algorithm R). Maintain reservoir of size `k`; on item `t`, accept with probability `k/t`; if accepted, replace random slot. After N items, sample is uniformly distributed over all C(N,k) k-subsets.
- **ST23 — `ReservoirZ`** (~90 LOC, Vitter 1985 Algorithm Z). The geometric-skip optimisation: precompute the next item to inspect via inverse-CDF of a geometric — average O(k·log(N/k)) work instead of Algorithm R's O(N). A 100-1000× speed-up on large streams.
- **ST24 — `WeightedReservoir-AExpJ`** (~140 LOC, Efraimidis-Spirakis 2006). Each item has weight w_i; sample with probability ∝ weight. Key insight: `key_i = U_i^(1/w_i)` and keep top-k keys. The standard for weighted importance-sampling streams (e.g. Twitter sample-by-engagement; Spark `takeSample(weighted=true)`).

### Streaming linear algebra + DP + extras (~290 LOC)

- **ST25 — `FrequentDirections`** (~150 LOC, Liberty 2013 + Ghashami-Liberty-Phillips-Woodruff 2016). One-pass low-rank matrix sketch: maintain a 2ℓ × d matrix; on overflow, run SVD, subtract the (ℓ+1)-th singular value squared from all surviving sing-values, zero the bottom half. Returns ℓ × d sketch B with ‖A^T A − B^T B‖ ≤ ‖A‖_F²/ℓ. The deterministic alternative to random projection. Co-cite with 261/262.
- **ST26 — `PrivateCountMin`** (~100 LOC, Chan-Shi-Song 2011 + Apple's 2017 RAPPOR-style CMS-DP). Adds Laplace(1/ε)-noise to each cell before query; satisfies (ε, 0)-differential-privacy. The first DP primitive in `reality/` — opens the door to a future `privacy/` package per the 184/188 cross-link.
- **ST27 — `StreamingTriangleCount`** (~160 LOC, Pavan-Tangwongsan-Tirthapura-Wu 2013 wedge-sampling + Tsourakakis-Drineas-Michelakis-Koutis-Faloutsos 2009 DOULION sparsification). Triangle count from streaming edges via wedge-reservoir sampling; cross-cite with `graph/` consumers.

---

## 2. Saturation pins (R-pattern targets)

- **R-MERGEABLE-SKETCH 6/6**: ST5 CMS, ST12 HLL, ST15 tDigest, ST16 KLL, ST18 Bloom, ST25 FrequentDirections all merge associatively + commutatively to byte-identical state regardless of merge order. Property test: shuffle stream → split N ways → merge → equals single-pass sketch on full stream within sketch's precision. The keystone property that makes sketches map-reducible.
- **R-CMS-OVERESTIMATE 1/1**: ST5 estimate ≥ true count for all streams (deterministic property — no probability involved). Property test: random stream of 10⁶ items → for every distinct item, `cms.Estimate(x) >= true_count(x)` byte-tight.
- **R-HLL-RELATIVE-ERROR 1/1**: ST12 standard error matches Flajolet-Fusy-Gandouet-Meunier-2007 σ ≈ 1.04/√m within 5% on Monte Carlo over uniform-hash streams of size 10⁵. Cross-language pin: same hash family (MurmurHash3_64) → byte-identical register state across Go/Python/C++/C#.
- **R-MISRAGRIES-DETERMINISTIC 1/1**: ST7 returns identical (item, lower-bound) pairs on identical streams across all four substrates with ZERO tolerance — Misra-Gries is fully deterministic, no RNG.
- **R-SPACESAVING-TIGHTER-THAN-MISRA 1/1**: On Zipfian stream with α=1.2 and k=100, ST8 mean error < ST7 mean error / 2. Numerical pin against Metwally-2005 Table 3.
- **R-RESERVOIR-UNIFORM 1/1**: ST22 chi-squared test — over 10⁴ trials of stream length 10⁶ with k=100, every item appears in reservoir with empirical frequency in [k/N · (1-3σ), k/N · (1+3σ)] for σ = √(k(1-k/N)/N).
- **R-TDIGEST-TAIL-ACCURACY 1/1**: ST15 99.9-percentile estimate within 0.1% of true on 10⁶ samples from log-normal(0, 2). Critical for production observability claims.
- **R-BLOOM-FPR-MATCHES-THEORY 1/1**: ST18 empirical FPR within 5% of theoretical (1-e^(-kn/m))^k on 10⁵-element insertion + 10⁵ random queries.
- **R-CUCKOO-LOAD-FACTOR 1/1**: ST19 sustains > 95% load before insertion failure for (2,4)-cuckoo with 8-bit fingerprints — matches Fan-Andersen-Kaminsky-Mitzenmacher-2014 §4.
- **R-FD-DETERMINISTIC-COVARIANCE-BOUND 1/1**: ST25 ‖A^T A - B^T B‖₂ ≤ ‖A‖_F²/ℓ on every random matrix tested (deterministic Liberty-2013 guarantee). Cross-language pin: byte-identical sketch when fed identical matrix.
- **R-DPCMS-CALIBRATION 1/1**: ST26 noise scale matches Laplace(1/ε); composition-test of two DP-CMS over disjoint shards yields (2ε, 0)-DP per the basic-composition theorem.

11 saturation pins from a single review. None overlap any prior 1-223 pin.

---

## 3. Cross-language byte-reference plan

DataSketches Apache project (Java reference at https://datasketches.apache.org with C++/Python/Go ports) is the canonical byte-reference for ST5/ST7/ST8/ST12/ST14/ST15/ST16/ST18/ST22. Specific pins:

- **CMS**: Cormode's C reference https://www.cs.rutgers.edu/~muthu/massdal-code-index.html; tolerance 0 (deterministic given hash seeds).
- **HLL**: Stripe/dgryski/go-hll Go reference + Apache Druid Java; tolerance 0 (deterministic given MurmurHash3_64 seed).
- **t-digest**: Dunning's C reference https://github.com/tdunning/t-digest; tolerance 1e-12 on cluster centroids.
- **KLL**: DataSketches reference at https://github.com/apache/datasketches-cpp/tree/master/kll; tolerance 0 given identical seed.
- **Bloom / Cuckoo / XOR**: Lemire's reference filters https://github.com/FastFilter/fastfilter_cpp; tolerance 0 on bit-array equality.
- **MinHash**: ekzhu/datasketch Python; tolerance 0 given identical seed.
- **Reservoir**: Apache DataSketches `frequent_items_sketch` and `quantile_sketch`; tolerance 0 given identical RNG sequence (RNGSampler shared across substrates).
- **FrequentDirections**: Liberty's reference https://github.com/edoliberty/frequent-directions; tolerance 1e-9 on singular values (SVD numerical noise).

Each sketch ships with **30+ JSON test vectors** in `streaming/testdata/{cms_basic.json, cms_zipf.json, hll_uniform.json, hll_skewed.json, tdigest_lognormal.json, ...}`. Cross-language harness re-uses the existing `testutil` golden-file infrastructure.

---

## 4. PR cadence — exactly 4 PRs

- **PR-1 — substrate + first-three** (~520 LOC). ST1 Sketch[T] interface + ST2 Hashing + ST3 RNGSampler + ST4 Welford-promotion + ST5 CountMin + ST7 MisraGries + ST12 HLL. **Lands the first sketch-anything in repo.** Saturates R-CMS-OVERESTIMATE 1/1 + R-MISRAGRIES-DETERMINISTIC 1/1 + R-HLL-RELATIVE-ERROR 1/1 (3/11 pins). One day. **The single most-leverage commit** because every downstream consumer (streaming-quantile, streaming-distinct, observability, frequency monitoring, log analytics, anomaly detection, A/B-testing tail-latencies) imports exactly this file.
- **PR-2 — heavy-hitters + bloom + tdigest** (~1,180 LOC). ST6 CountSketch + ST8 SpaceSaving + ST9 LossyCounting + ST15 tDigest + ST16 KLL + ST18 Bloom + ST19 Cuckoo + ST22 ReservoirR + ST24 WeightedReservoir. Saturates 6 more pins. **The DataSketches-parity PR** — lands the seven primitives that 95% of production streaming systems use. Three days.
- **PR-3 — cardinality + quantile completeness + linear algebra** (~810 LOC). ST10 AMS + ST11 LinearCounting + ST13 HLL++ + ST14 MinHash/kMV + ST17 GreenwaldKhanna + ST20 XORFilter + ST21 QuotientFilter + ST25 FrequentDirections. Two days. Saturates R-MERGEABLE-SKETCH 6/6 (the keystone property fully across the canon) + R-FD-DETERMINISTIC-COVARIANCE-BOUND 1/1.
- **PR-4 — privacy + graph triangles + extras** (~290 LOC). ST26 PrivateCountMin + ST27 StreamingTriangleCount + ST23 ReservoirZ. **Final PR — opens DP and graph-streaming corners**. One day. Saturates R-DPCMS-CALIBRATION 1/1.

Total: ~3,250 LOC, 7 days end-to-end (one engineer), 11/11 R-pattern pins saturated.

---

## 5. Cross-link map (220-224 unified substrate)

The unified-keystone observation: **five reviews in a row name the same interface shape.**

| Review | Interface | Update | Query | Merge |
|--------|-----------|--------|-------|-------|
| 174-G4 / 221-O1 | `OnlineLearner` | Update(loss-grad) | Action() | n/a |
| 220-F1 | `FiniteSumLoss` | (theta, idx) → (loss, grad) | n/a | n/a |
| 222-B1 | `StochasticBandit` | Update(arm, reward) | Pull() | n/a |
| 223-S1 | `SetFunction` | n/a (oracle) | Eval(S) | n/a |
| **224-ST1** | **`Sketch[T]`** | **Update(x)** | **Query(...)** | **Merge(other)** |

All five are 30-50 LOC interfaces that compose to thousand-LOC ecosystems. Co-shipping all five in PR-0-keystone (a single ~200 LOC commit on the **interfaces only**, no implementations) would unblock the entire 220-224 ecosystem in parallel — single PR enables five reviews to land in any order. Recommend Block-C closes with this meta-PR.

---

## 6. Verification I will perform pre-PR

1. Repo grep again at PR time: still zero matches for the absence-list above (catch silent prior-art landing).
2. Run `go test ./...` after PR-1 substrate: must remain green (1965 + ~120 new = ~2085 tests).
3. Add 30 golden vectors per primitive in `streaming/testdata/` per the existing testutil convention.
4. Verify the audio.Fingerprint shim doesn't break: `go test ./audio/...` must remain green after the type-alias-rename.
5. Cross-language pin: re-run the golden vectors through Apache DataSketches Java reference for ST5/ST7/ST8/ST12/ST15/ST16/ST18 — bit-identical state (after seed alignment).
6. Bench: `go test -bench=. ./streaming/...` — establish hot-path baselines (Update should be ≤ 50ns/op for CMS/HLL on M1, ≤ 200ns/op for t-digest insertion).

---

## 7. Candor

The streaming canon has **immediate-and-obvious-every-consumer-imports-it** demand:

- Observability / metrics: t-digest for percentiles, HLL for cardinality, CMS for top-k requests.
- Log analytics: Misra-Gries / Space-Saving for top-N noisy patterns.
- Recommender systems: MinHash / Bloom for candidate-set membership.
- Network monitoring: AMS for self-join / DDOS detection, CMS for flow-size estimation.
- Database query optimisation: HLL for `COUNT DISTINCT`, t-digest for histogram statistics.
- Privacy-preserving analytics: DP-CMS for telemetry without PII.
- Streaming ML: Reservoir for unbiased mini-batch sampling, FrequentDirections for online PCA.
- Audio fingerprinting: Welford (already in audio) + MinHash for Shazam-style fast lookup.
- Forge consumers: any aggregation across user-cohort traces uses one of these eight directly.

Zero cycle hazard: `streaming/` imports `crypto`, `prob`, `linalg`, `combinatorics`, `compression`, `audio` — none import streaming. One-direction-only.

Pinning identity: the canonical (1-pass, sublinear-space, mergeable, ε-δ-bounded) sketch family unifies all 23 primitives — they all instantiate the **`Sketch[T]` interface** with different `T` (item type), different bound classes (deterministic / probabilistic), and different (space, accuracy) trade-off curves. **R-SKETCH-INTERFACE-UNIFIES-TWENTY-THREE-LEARNERS** saturation lands when PR-1 + PR-2 ships covering 17/23 primitives + 9/11 pins (~74% canon, ~82% pin coverage in 4 days), and is **fully complete after PR-4** at 23/23 + 11/11 vs current **0/23 + 0/11 — zero functions out of canonical twenty-three.**

Report at `agents/224-new-streaming.md`. ~330 lines.
