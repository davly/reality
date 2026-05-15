## 225 | new-ann

**Topic:** Approximate Nearest Neighbor (ANN) — exact baselines (brute-force / kd-tree Bentley 1975 / ball-tree Omohundro 1989 / VP-tree Yianilos 1993 / R-tree Guttman 1984 / R*-tree Beckmann-Kriegel-Schneider-Seeger 1990 / cover-tree Beygelzimer-Kakade-Langford 2006 / M-tree Ciaccia-Patella-Zezula 1997), Locality-Sensitive Hashing (random hyperplane / SimHash Charikar 2002 for cosine, p-stable Datar-Indyk-Immorlica-Mirrokni 2004 / E2LSH for L_p, MinHash Broder 1997 for Jaccard, banding LSH-Forest Bawa-Condie-Ganesan 2005, multi-probe LSH Lv-Josephson-Wang-Charikar-Li 2007, C2LSH Gan-Feng-Fu-Ng 2012, LSH ensemble Zhu-Nargesian-Pu-Bryan-Miller 2016), data-dependent hashing (spectral hashing Weiss-Torralba-Fergus 2008, ITQ Gong-Lazebnik 2011, semi-supervised hashing Wang-Kumar-Chang 2010, deep hashing 2015+), graph-based ANN (KGraph / NN-Descent Dong-Charikar-Li 2011, HNSW Malkov-Yashunin 2018, NSG Fu-Wang-Wang-Cai 2017, NSSG Fu-Wang-Cai 2019, Vamana / DiskANN Subramanya-Devvrit-Kadekodi-Krishaswamy-Simhadri 2019, FreshDiskANN Singh-Subramanya-Krishnaswamy-Simhadri 2021, Filtered DiskANN Gollapudi-Karia-Singh-Singh-Subramanya 2023), inverted-file + quantization (IVF coarse k-means Sivic-Zisserman 2003, Product Quantization PQ Jégou-Douze-Schmid 2011, Optimized PQ Ge-He-Ke-Sun 2013, Locally Optimized PQ Kalantidis-Avrithis 2014, Additive Quantization AQ Babenko-Lempitsky 2014, Composite Quantization Zhang-Du-Wang 2014, Tree-Quantization Babenko-Lempitsky 2015, Residual Quantization Chen-Guan-Wang 2010), inverted multi-index (Babenko-Lempitsky 2012), ScaNN (Guo-Sun-Lindgren-Geng-Simcha-Chern-Kumar 2020 anisotropic-quantization with score-aware loss), binary embedding (ITQ, LSH binary codes, Hamming-distance brute-force), benchmarks (ann-benchmarks.com Aumüller-Bernhardsson-Faithfull 2020, BigANN-Benchmarks 2021), reference impls (FAISS Johnson-Douze-Jégou 2017, Annoy Bernhardsson 2013, NGT Yahoo-Japan 2016, Milvus 2019, Weaviate, Qdrant, ScaNN, hnswlib), filtered/constrained ANN (Filtered Vamana 2021, ACORN Patel-Kraska 2024, Pre-/Post-/In-filtering hybrid), streaming ANN (FreshDiskANN insert-update-delete with bounded memory), multi-vector retrieval (ColBERT Khattab-Zaharia 2020 late-interaction, MaxSim aggregation, PLAID Santhanam-Khattab-Potts-Zaharia 2022), re-ranking (ANN candidate set + exact distance, two-stage retrieval), curse of dimensionality (Beyer-Goldstein-Ramakrishnan-Shaft 1999), c-approximate-NN guarantees (Indyk-Motwani 1998 STOC LSH paper). **Block:** C (cutting-edge math, what reality is missing). **Date:** 2026-05-08.

**Scope:** the **vector-similarity-search axis** — return the k nearest neighbours of a query vector under L_p / cosine / inner-product / Hamming / Jaccard with sublinear (or memory-bounded) time. Distinct from 077-geometry-missing T1-4 kdtree (which is a 3-D geometric primitive for ICP/point-cloud, not a high-D vector index), 081-graph (`graph/` is pure graph algorithms with no spatial-index attachment), 224-new-streaming (set-membership + cardinality + frequency, *not* nearest-neighbour even though MinHash is shared), 220-stochastic-opt / 221-online-learning / 222-bandits (decision-theoretic, not retrieval). 225 owns **k-NN retrieval over vectors / sets / strings** with the four canonical families: tree, hash, graph, quantization.

## Two-line summary

`reality/` v0.10.0 ships **ZERO ANN surface** — repo-wide grep on `KDTree|kdtree|kd-tree|BallTree|ball-tree|VPTree|VP-tree|RTree|HNSW|NSG|Vamana|DiskANN|ScaNN|IVF|ProductQuant|PQ.*Quant|OPQ|SimHash|Sim-Hash|MinHashLSH|MinHash-LSH|RandomProjection|HyperplaneLSH|p-stable|E2LSH|SpectralHashing|ITQ|nearestNeighbor|nearest-neighbor|kNN|approxNN|annIndex` returns **zero source-code matches** across all 22 packages (only mentions are 077-geometry-missing-T1-4 scoping `kdtree.go` for 3-D geometric purposes + 224-streaming-ST14 scoping `MinHash` for Jaccard cardinality + 195-synergy mentioning kd-tree); the **distance machinery is already 100% present** — `linalg/vector.go:23` `CosineSimilarity` + `:48` `EncodingDistance` (L2/√n) + `:71` `DimensionWeightedDistance` + `:134` `DotProduct` + `:150,163,176` `L2Norm/L1Norm/LInfNorm` + `:98` `L2Normalize` + `crypto/hash.go:25,40,61` `FNV1a32/FNV1a64/MurmurHash3_32` (the canonical hash families for SimHash, MinHash, p-stable LSH, k-wise families) + `linalg/decompose.go` SVD/QR (the substrate for OPQ rotation, ITQ rotation, ScaNN whitening) + `linalg/pca.go` (the substrate for spectral hashing eigenvectors); **23 primitives N1-N23 totalling ~4,150 LOC of pure connective tissue** across NEW `nn/{index.go,brute.go,kdtree.go,balltree.go,vptree.go,lsh_hyperplane.go,lsh_pstable.go,lsh_minhash.go,simhash.go,hnsw.go,nsg.go,nndescent.go,ivf.go,pq.go,opq.go,scann.go,itq.go,filter.go,rerank.go,benchmark.go}` — every primitive is a 80-450 LOC composition of `linalg/vector.go` (distances), `crypto/hash.go` (hash families), `linalg/decompose.go` (rotations), `prob/distributions.go` (Gaussian projections), `combinatorics/generate.go` (random k-wise families). Cheapest one-day standalone is **PR-1 N1-Index-interface + N2-BruteForce + N3-KDTree + N7-RandomHyperplaneLSH + N15-ProductQuantization (~720 LOC)** which lands the **first vector-search-anything in the repo** and saturates **R-LSH-RECALL-PROBABILITY 1/1** (random-hyperplane LSH bucket-collision probability matches the cosine-similarity arc-formula 1 - θ/π — Goemans-Williamson 1995, Charikar 2002) plus **R-PQ-RECONSTRUCTION 1/1** (PQ approximate distance is unbiased L2 estimator with variance bounded by sum of sub-codebook quantization errors — Jégou-Douze-Schmid 2011 Theorem 1). Highest-leverage one-week unlock is **PR-2 N10-HNSW + N17-OPQ + N18-ScaNN-anisotropic + N20-FilteredANN + N22-Reranker (~1,750 LOC)** because they collectively saturate the **R-RECALL-AT-10 4/4** pin (HNSW M=16 / IVF-PQ nprobe=8 / ScaNN ah=0.2 / OPQ-on-deep1B all reproduce ann-benchmarks.com 2024 leaderboard recall@10 ≥ 0.95 on SIFT1M / GIST1M / Glove200) and unblock the entire **graph-based ANN family** which is the post-2018 SOTA. Architectural keystone is the **`Index[T]` interface (~30 LOC)** — `Build([]T) / Add(T) / Search(query T, k int) []Hit / SearchRange(query T, radius float64) []Hit / Bytes() / FromBytes([]byte)` — together with the `Distance[T]` substrate (~40 LOC wrapping `linalg.CosineSimilarity / EncodingDistance` plus Hamming + Jaccard + Inner-Product). Co-shipped with 174-G4 `OnlineLearner` + 220-F1 `FiniteSumLoss` + 221-O1 `OnlineConvexLearner` + 222-B1 `StochasticBandit` + 223-S1 `SetFunction` + 224-Sk1 `Sketch[T]` as the **seventh interface in the unified-substrate keystone** consumers reach for first; specifically pairs with 224-ST14 MinHash (which becomes N9 MinHashLSH here by adding banding + LSH-Forest tree on top of the same min-hash sketch).

---

## 0. State at HEAD (2026-05-08, v0.10.0)

Verified by direct read of `linalg/`, `geometry/`, `crypto/`, `graph/` and repo-wide grep.

### `reality/` ANN surface = ZERO (verified)

Repo-wide grep on `KDTree|kdtree|kd-tree|BallTree|VPTree|MTree|RTree|HNSW|NSG|Vamana|DiskANN|ScaNN|IVF|ProductQuant|OPQ|SimHash|MinHashLSH|HyperplaneLSH|p-stable|E2LSH|SpectralHashing|ITQ|nearestNeighbor|kNN|approxNN|annIndex|annoy|hnswlib|FAISS` across all `.go` files returns **zero source-code matches**. The only matches are:

- `reviews/overnight-400/agents/077-geometry-missing.md:216` — Tier-1 `kdtree.go` scoped for 3-D point-cloud / ICP, *not* high-dimensional vector search. Different beast: kd-tree degrades to brute-force for d > ~20 (Beyer-Goldstein-Ramakrishnan-Shaft 1999 curse-of-dimensionality), so the 3-D-geometry kd-tree from 077 cannot serve as a vector ANN index for embeddings (typical d = 128, 384, 768, 1536).
- `reviews/overnight-400/agents/224-new-streaming.md:78` — `ST14 MinHash + kMinValues` scoped as a *cardinality / Jaccard-similarity sketch*, not as an LSH index. 225 N9 (MinHashLSH) imports 224-ST14 and adds the banding + LSH-Forest tree on top.
- `reviews/overnight-400/agents/195-synergy-optim-prob.md` (single mention) — passing reference to "kd-tree-style" splitting in proposal review, no concrete primitive.

### Primitives already in repo that 225 composes

Verified by direct file read.

#### Distance bedrock (`linalg/vector.go`, 241 LOC)

- `CosineSimilarity(a, b []float64) float64` — line 23 — exact float64, returns 0 on length mismatch / zero-magnitude. **Used by every cosine-distance ANN: hyperplane LSH, HNSW-cosine, IVF-cosine, ScaNN-cosine.**
- `EncodingDistance(a, b []float64) float64` — line 48 — L2 / √n. **Almost-canonical L2; ANN reuses but needs a non-normalised `L2Distance` companion (10 LOC trivial wrapper).**
- `DimensionWeightedDistance(a, b, weights []float64) float64` — line 71 — weighted L2. **Used by ScaNN anisotropic loss + filtered re-ranking.**
- `DotProduct(a, b []float64) float64` — line 134 — exact. **Used by inner-product ANN (the Maximum-Inner-Product-Search MIPS family — Shrivastava-Li 2014, Bachrach-Finkelstein-Gilad-Bachrach-Katzir-Koenigstein-Nice-Paquet 2014).**
- `L2Norm / L1Norm / LInfNorm` — lines 150, 163, 176. **Used by p-stable LSH (Datar-Indyk-Immorlica-Mirrokni 2004 — L_p with p ∈ {1, 2}).**
- `L2Normalize(vec) bool` — line 98, in-place. **Required for cosine-as-L2-on-unit-sphere reduction (the standard trick: cos θ = 1 − ½‖a−b‖² for unit vectors, so cosine-NN ≡ L2-NN on the sphere).**

#### Hash bedrock (`crypto/hash.go`, 224 LOC)

- `FNV1a32 / FNV1a64` — lines 25, 40. **Used by MinHash (k independent permutations via Carter-Wegman + FNV-mixing), SimHash (sign-of-FNV-of-token aggregated).**
- `MurmurHash3_32(data, seed)` — line 61. **The canonical 32-bit avalanche hash for LSH bucket-keys (CMS / Bloom / Cuckoo all use it via 224-ST2 hashing substrate, which 225 imports).**
- `ConsistentHash(key, numBuckets)` — line 129. **Used by sharded ANN (distributed IVF / DiskANN cluster routing).**

#### Linear-algebra bedrock (`linalg/decompose.go` 345 LOC + `pca.go` 215 LOC + `eigen.go` 212 LOC + `matrix.go` 208 LOC)

- `SVD` / `QRDecompose` / `LUDecompose` / `Cholesky` (decompose.go). **OPQ rotation = SVD of training data; ITQ rotation = orthogonal Procrustes via SVD; ScaNN whitening = Cholesky of covariance.**
- `PCA(data, k)` (pca.go). **Spectral hashing = top-k eigenvectors of similarity Laplacian; OPQ initialisation rotates by PCA basis; ITQ initialises with PCA then rotates.**
- `MatMul / MatVec` (matrix.go). **Hot path of every projection-based LSH and PQ codebook lookup.**

#### Probability bedrock (`prob/distributions.go`)

- Gaussian sampling. **p-stable LSH for L2 = projection onto N(0, 1) Gaussian vector (Datar-Indyk-Immorlica-Mirrokni 2004 — N(0, 1) is 2-stable); p-stable LSH for L1 = projection onto Cauchy vector (Cauchy is 1-stable).**

#### Combinatorics bedrock (`combinatorics/generate.go`)

- `RandomSubset` etc. **Used to generate random k-wise hash families and HNSW level-assignment probabilities (geometric distribution with parameter 1/ln(M)).**

**Net:** 100 % of the substrate primitives ANN needs are already in the repo. ANN is pure connective tissue — no new low-level primitive is required, only composition.

### Primitives genuinely absent

Repo-wide grep confirms **zero occurrences** of:

- **Tree-based exact NN:** `kdtree`, `KDTree`, `ballTree`, `BallTree`, `vpTree`, `VPTree`, `MTree`, `M-Tree`, `RTree`, `R-tree`, `Rstar`, `R*-tree`, `coverTree`, `Cover-Tree`. (077 scopes a 3-D-geometry kdtree but doesn't ship.)
- **LSH:** `LSH`, `localitySensitive`, `Locality-Sensitive`, `hyperplaneLSH`, `randomHyperplane`, `SimHash`, `Sim-Hash`, `Charikar`, `pStable`, `p-stable`, `E2LSH`, `Datar`, `MinHashLSH`, `Min-Hash-LSH`, `LSHForest`, `multiProbeLSH`, `multi-probe`, `C2LSH`.
- **Data-dependent hashing:** `spectralHashing`, `Spectral-Hashing`, `ITQ`, `IterativeQuantization`, `iterative-quantization`, `semiSupervisedHashing`.
- **Graph-based ANN:** `HNSW`, `H-NSW`, `Malkov`, `Yashunin`, `NSG`, `NavigatingSpreading`, `Navigating-Spreading-out-Graph`, `Vamana`, `DiskANN`, `Disk-ANN`, `Microsoft-DiskANN`, `Subramanya`, `kgraph`, `KGraph`, `nnDescent`, `NN-Descent`, `Dong-Charikar`, `Filtered-Vamana`, `FreshDiskANN`, `ACORN`.
- **Quantization-based ANN:** `IVF`, `invertedFile`, `inverted-file`, `productQuantization`, `Product-Quantization`, `Jegou`, `Jégou`, `OPQ`, `optimizedPQ`, `Optimized-PQ`, `LOPQ`, `Locally-Optimized-PQ`, `additiveQuant`, `Additive-Quantization`, `Babenko-Lempitsky`, `compositeQuantization`, `treeQuantization`, `residualQuantization`, `Residual-Quantization`, `invertedMultiIndex`, `IMI`, `multi-index`.
- **Inner-product-search (MIPS):** `Shrivastava-Li`, `MIPS`, `maximumInnerProduct`, `asymmetricLSH`.
- **ScaNN:** `ScaNN`, `scann`, `anisotropic-quantization`, `Anisotropic-Vector-Quantization`, `Guo-Sun-Lindgren`, `score-aware-loss`.
- **Multi-vector retrieval:** `ColBERT`, `colbert`, `lateInteraction`, `late-interaction`, `MaxSim`, `PLAID`.
- **Filtered ANN:** `filteredANN`, `filtered-ANN`, `constrained-NN`, `pre-filter`, `post-filter`, `in-filter`, `ACORN`, `pinecone-filter`.
- **Re-ranking:** `rerank`, `re-rank`, `reRanker`, `twoStage`, `two-stage`.
- **ANN benchmarks:** `annBenchmark`, `recallAt`, `recall@k`, `qps`, `BigANN`, `Glove`, `SIFT1M`, `Deep1B`.

---

## 1. The 23-primitive surface N1-N23

Total target ~4,150 LOC across 20 files in NEW package `nn/` (sibling to `linalg/`, `graph/`).

### Substrate (~150 LOC)

- **N1 — `Index[T]` + `Hit` + `Distance[T]` interfaces** (`nn/index.go`, ~80 LOC). The keystone interface; mirrors 224's `Sketch[T]`. `Index[T]` exposes `Build([]T)`, `Add(T) error`, `Search(query T, k int) []Hit`, `SearchRange(query T, radius float64) []Hit`, `Size() int`, `Bytes() []byte`, `FromBytes([]byte) error`. `Hit{ID int; Distance float64}`. `Distance[T] func(a, b T) float64` with helpers `L2`, `Cosine`, `InnerProductNeg`, `Hamming`, `Jaccard` thin-wrapping `linalg.CosineSimilarity` / `EncodingDistance` etc.
- **N2 — `nn/distance.go` distance helpers** (~70 LOC). Wrappers over `linalg/vector.go`: `L2(a,b []float64)` (the missing non-normalised companion to `EncodingDistance`), `CosineDist(a,b) = 1 - CosineSimilarity(a,b)` (so smaller = closer like everything else), `InnerProductNeg(a,b) = -DotProduct(a,b)` (negate so smaller = closer), `Hamming([]uint64, []uint64) int` (popcount-bitwise XOR), `Jaccard([]uint64, []uint64) float64` (∩/∪ on bit-sets).

### Exact / tree-based NN (~1,150 LOC)

- **N3 — `BruteForce[T]`** (`nn/brute.go`, ~120 LOC). The reference implementation against which every approximate index is graded. `Build` stores; `Search` linear-scans with a fixed-size top-k heap. **Saturates R-RECALL-1.0 1/1** (recall@k = 1 by definition — this is the ground truth). **Used by N22 (re-ranker) as the "exact distance over candidate set" stage** and by every ANN test as the recall reference.
- **N4 — `KDTree`** (`nn/kdtree.go`, ~280 LOC). Bentley 1975 axis-aligned binary tree. Build via median-of-medians (nth-element); search via best-first with priority queue + branch-and-bound pruning (Friedman-Bentley-Finkel 1977). **Tagged as exact for d ≤ ~20; degrades to brute-force above** (Beyer-Goldstein-Ramakrishnan-Shaft 1999 curse-of-dimensionality — explicitly documented in the docstring with the BGRS 1999 citation). **Reuses 077-T1-4 if/when that lands; this is the d-D generalisation.**
- **N5 — `BallTree`** (`nn/balltree.go`, ~250 LOC). Omohundro 1989; binary tree where each node is a ball (centroid + radius) instead of an axis-aligned box. **Better than kd-tree for L2 in moderate d (~20-60)** because the ball-pruning bound is tight under any rotation, whereas kd-tree's axis-aligned bounds loosen.
- **N6 — `VPTree`** (`nn/vptree.go`, ~250 LOC). Vantage-Point tree (Yianilos 1993). Binary tree where each node picks a "vantage point" and splits by distance-to-vantage-point. **Works for any metric (not just L2)** — supports cosine, Hamming, Jaccard, edit-distance — the only metric-agnostic exact tree in the canon.
- **N7 — `RTree`** (`nn/rtree.go`, ~250 LOC). Guttman 1984 (R-tree) + Beckmann-Kriegel-Schneider-Seeger 1990 (R*-tree improved split heuristic). **Spatial-bounding-rectangle index for low-d range queries** (the GIS-canonical structure). Slot-tree variant for static data. Cross-link to 081-graph if a spatial-graph layer is added later.

### LSH family (~620 LOC)

- **N8 — `RandomHyperplaneLSH` / `SimHash`** (`nn/lsh_hyperplane.go`, ~140 LOC). Charikar 2002 cosine-LSH: signature bit `i` = `sign(<r_i, x>)` for random Gaussian `r_i`. Bucket-collision probability for two unit vectors is `1 − θ(a,b)/π` (Goemans-Williamson 1995). **Saturates R-LSH-COSINE-COLLISION 1/1** — over 10⁶ random pairs the empirical collision rate matches the GW arc-formula to 1/√N. SimHash is the same primitive over feature-token streams (sum of signed projections then sign-bit per dim).
- **N9 — `MinHashLSH`** (`nn/lsh_minhash.go`, ~180 LOC). Banded MinHash for Jaccard. **Imports 224-ST14 MinHash sketch** for the per-set k-min-values; adds `bands × rows = k` partitioning, an LSH-Forest tree (Bawa-Condie-Ganesan 2005) over band-signatures, and the `(b, r)`-band collision-probability `1 − (1 − s^r)^b` (Broder 1997 + Rajaraman-Ullman 2011 *Mining of Massive Datasets* §3.4). **Cross-link to 224**: the MinHash *sketch* lives in `streaming/`; the MinHash *index* (banding + tree + similarity-threshold tuning) lives here.
- **N10 — `pStableLSH` / `E2LSH`** (`nn/lsh_pstable.go`, ~150 LOC). Datar-Indyk-Immorlica-Mirrokni 2004 (E2LSH). For L2: hash bucket = `floor((<a, x> + b) / w)` where `a ∼ N(0, I)` (2-stable) and `b ∼ Uniform(0, w)`. For L1: same but `a ∼ Cauchy` (1-stable). The collision probability is the integral of the stable PDF — closed-form (Datar et al. 2004 Theorem 1). **The standard high-dim L_p LSH; ann-benchmarks-grade implementation**.
- **N11 — `MultiProbeLSH`** (`nn/lsh_multiprobe.go`, ~80 LOC). Lv-Josephson-Wang-Charikar-Li 2007 — at query time, also probe nearby buckets ranked by perturbation-score. **Reduces the number of hash tables by ~10× at the same recall.** Composes over N8/N10.
- **N12 — `C2LSH`** (`nn/lsh_c2.go`, ~70 LOC). Gan-Feng-Fu-Ng 2012 — collision-counting LSH. Single hash family, dynamic radius. **More memory-efficient than E2LSH for large N**.

### Data-dependent hashing (~330 LOC)

- **N13 — `SpectralHashing`** (`nn/spectral_hash.go`, ~150 LOC). Weiss-Torralba-Fergus 2008 — top-k eigenvectors of the data Laplacian + sign-thresholding. **Reuses `linalg.PCA` directly** (k=8/16/32 bits typical). Better recall than data-independent LSH when data has low-dimensional structure.
- **N14 — `ITQ`** (`nn/itq.go`, ~180 LOC). Gong-Lazebnik 2011 Iterative Quantization — initialise with PCA, then iterate orthogonal Procrustes (via `linalg.SVD`) against the binary code centroids. **Best data-dependent binary hashing in the unsupervised regime (per Wang-Liu-Kumar-Chang 2018 survey).**

### Graph-based ANN (~960 LOC)

- **N15 — `NNDescent` / `KGraph`** (`nn/nndescent.go`, ~280 LOC). Dong-Charikar-Li 2011 — iterative neighbor-of-neighbor refinement to build an approximate k-NN graph in O(N · k · log N) without any indexing structure. **Foundational: HNSW, NSG, Vamana all build on top of an initial k-NN graph and NN-Descent is the canonical builder.**
- **N16 — `HNSW`** (`nn/hnsw.go`, ~380 LOC). Malkov-Yashunin 2018 Hierarchical Navigable Small World — multi-layer skip-list-of-graphs with greedy descent. **The post-2018 SOTA; FAISS / hnswlib / Pinecone / Weaviate / Qdrant all default to HNSW.** Level-L assignment via geometric distribution `floor(-log(uniform(0,1)) / log(M))`. Search: greedy descent on layer L → L-1 → ... → 0 with dynamic-candidate-list `efSearch`. **Saturates R-HNSW-RECALL-AT-10 1/1** — `M=16, efConstruction=200, efSearch=100` reproduces ann-benchmarks SIFT1M recall@10 ≥ 0.99.
- **N17 — `NSG`** (`nn/nsg.go`, ~200 LOC). Fu-Wang-Wang-Cai 2017 Navigating Spreading-out Graph — refine an NN-Descent graph by **MRNG (Monotonic Relative Neighborhood Graph) edge selection** + **medoid as fixed entry point**. **Smaller graph than HNSW (no levels), often higher QPS at the same recall on dense datasets.**
- **N18 — `Vamana` / `DiskANN`** (`nn/vamana.go`, ~250 LOC). Subramanya-Devvrit-Kadekodi-Krishnaswamy-Simhadri 2019 — single-layer graph with **α-RNG edge pruning** (α ∈ [1.0, 1.5]) producing a graph with diameter O(log N) that works directly on disk-resident data via SSD-friendly access patterns. **The SOTA for billion-scale on a single machine.** FreshDiskANN (Singh-Subramanya 2021) extension for insert/update/delete adds ~120 LOC if scoped.

### IVF + Quantization (~810 LOC)

- **N19 — `IVF`** (`nn/ivf.go`, ~180 LOC). Inverted File: k-means coarse-quantize the dataset into `nlist` Voronoi cells (uses `linalg/kmeans.go` if/when 097-linalg-missing's k-means lands; otherwise ships a 60-LOC k-means inline). Search: probe the `nprobe` cells closest to the query. **Sivic-Zisserman 2003 (vocabulary trees in image retrieval); Babenko-Lempitsky 2012 added the inverted multi-index.**
- **N20 — `ProductQuantization` / `PQ`** (`nn/pq.go`, ~280 LOC). Jégou-Douze-Schmid 2011 — split each d-dim vector into `M` sub-vectors of `d/M` dims, k-means each sub-space into 2⁸ centroids, encode each vector as `M` bytes. ADC (Asymmetric Distance Computation) lookup table for sub-distances. **Saturates R-PQ-DISTANCE-UNBIASED 1/1** — over 10⁵ random pairs, expected ADC distance equals true L2 distance to the bound proven in Jégou-Douze-Schmid 2011 Theorem 1 (variance = sum of sub-codebook quantisation MSE).
- **N21 — `OptimizedPQ` / `OPQ`** (`nn/opq.go`, ~180 LOC). Ge-He-Ke-Sun 2013 — pre-rotate vectors by an orthogonal matrix `R` chosen to minimise PQ quantisation error (alternate between PQ codebooks and Procrustes-style `R` updates via `linalg.SVD`). **+5-15% recall over vanilla PQ on Deep1B / SIFT1B per the OPQ paper.**
- **N22 — `IVF-PQ`** (`nn/ivfpq.go`, ~80 LOC, almost a wrapper). Composes N19 IVF + N20 PQ + N2-residual-encoding (encode `x − centroid_q[c]` not `x` itself). **The FAISS default for billion-scale; ann-benchmarks 2024 leaderboard champion below 32 GB RAM.**
- **N23 — `ScaNN`** (`nn/scann.go`, ~90 LOC). Guo-Sun-Lindgren-Geng-Simcha-Chern-Kumar 2020 anisotropic-quantization — replaces PQ's "MSE between vector and centroid" with a **score-aware loss** that weights quantisation error along the query-direction more heavily. Closed-form per-sub-codebook update (Guo et al. 2020 §3). **+10-30% recall over OPQ on the Glove200/MSMARCO families** which is why Google retired their internal IVF-PQ for ScaNN circa 2020.

### Filtered + re-ranking (~150 LOC)

- **N24 — `FilteredANN`** (`nn/filter.go`, ~80 LOC). Three sub-strategies: (a) Pre-filter (filter candidate-set then exact-NN over survivors — high recall, low QPS at low selectivity), (b) Post-filter (ANN top-K' >> k then filter — fast but recall-cliff at low selectivity), (c) In-filter (HNSW edge traversal skips filtered nodes — Filtered-Vamana / ACORN-style; needs N16/N18 to expose a per-edge predicate hook). **Singh-Subramanya-Krishnaswamy-Simhadri 2021 Filtered-DiskANN; Patel-Kraska 2024 ACORN.**
- **N25 — `Reranker`** (`nn/rerank.go`, ~70 LOC). Two-stage: ANN gives candidate-set of size `k_cand >> k`, then exact distance via N3 BruteForce reranks to top-k. **Standard production pattern; the ANN papers' "recall@k" benchmark almost always assumes a reranker is present.** Also wraps multi-vector / ColBERT-style late-interaction (MaxSim aggregation over per-token vectors) when `T = [][]float64`.

### Benchmark harness (~110 LOC)

- **N26 — `nn/benchmark.go`** (~110 LOC). `RecallAt(k int, exact, approx []Hit) float64`, `QPS`, `BuildTime`, `IndexSize`. The seven canonical datasets per ann-benchmarks.com (SIFT1M, GIST1M, Glove25/100/200, Deep1M, MSMARCO-passage). **Test vectors against ann-benchmarks 2024 leaderboard**: HNSW M=16 efSearch=100 should hit recall@10 ≥ 0.99 on SIFT1M; IVF-PQ nprobe=8 should hit recall@10 ≥ 0.85 on Deep1M.

---

## 2. Saturation pins (R-pattern targets)

- **R-LSH-COSINE-COLLISION 1/1** (N8 RandomHyperplaneLSH). Empirical collision rate of two unit vectors over 10⁶ random hyperplanes equals `1 - acos(<a,b>)/π` to within 1/√N (Goemans-Williamson 1995 + Charikar 2002).
- **R-LSH-PSTABLE-COLLISION 2/2** (N10 pStableLSH for L1 and L2). Collision rate matches the closed-form integral over the stable distribution PDF (Datar-Indyk-Immorlica-Mirrokni 2004 Theorem 1).
- **R-LSH-MINHASH-JACCARD 1/1** (N9 MinHashLSH banding). For two sets with Jaccard `s`, the bucket-collision rate equals `1 - (1-s^r)^b` to within 1/√N (Broder 1997).
- **R-PQ-DISTANCE-UNBIASED 1/1** (N20 PQ). E[ADC(q, ẑ)] = ‖q − z‖² with variance ≤ Σ MSE(sub_m) (Jégou-Douze-Schmid 2011 Theorem 1).
- **R-HNSW-RECALL-AT-10 1/1** (N16 HNSW). M=16, efConstruction=200, efSearch=100 reproduces ann-benchmarks SIFT1M recall@10 ≥ 0.99.
- **R-IVFPQ-RECALL-AT-10 1/1** (N22 IVF-PQ). nlist=√N, nprobe=8, M=64, ks=256 reproduces ann-benchmarks Deep1M recall@10 ≥ 0.85.
- **R-SCANN-OUTPERFORMS-OPQ 1/1** (N23 ScaNN vs N21 OPQ). At equal index-bytes ScaNN beats OPQ recall@10 by ≥5pp on Glove200 (Guo et al. 2020 Table 2).
- **R-RECALL-1.0-EXACT 1/1** (N3 BruteForce). recall@k = 1.0 by definition.
- **R-RECALL-AT-10 ≥ 0.95 4/4** (PR-2 keystone). HNSW + IVF-PQ + ScaNN + OPQ all ≥ 0.95 recall@10 on SIFT1M.
- **R-MERGEABLE-INDEX 3/3** (N16 HNSW, N19 IVF, N20 PQ). HNSW supports incremental `Add`; IVF supports per-cell `Add`; PQ supports residual-`Add` over fixed codebooks. (Vamana / DiskANN deferred to FreshDiskANN extension.)

---

## 3. Cross-language byte-reference plan

- **HNSW**: hnswlib (Malkov reference impl, C++) is the gold standard. Same seed, same `M / efConstruction / efSearch` should produce byte-identical graph topology after ordering canonicalisation.
- **IVF-PQ**: FAISS Python — same k-means seed + same training set + same M=64, ks=256 should produce byte-identical PQ codes.
- **ScaNN**: scann Python — same anisotropic-quantization-threshold + same training set should reproduce per-sub-codebook centroids to 1e-6.
- **MinHash**: ekzhu/datasketch Python; cross-link to 224-ST14 byte-reference (same Murmur seed → byte-identical k-min-values).
- **SimHash**: a Go reference impl + Python `simhash` package; tolerance 0 given identical Murmur seed and identical token-tokenisation.
- **kd-tree / ball-tree**: scikit-learn `KDTree` / `BallTree`. Same data, same `leaf_size` → byte-identical query results (recall @ k = 1).
- **ann-benchmarks**: pull the published `recall vs QPS` tables directly from ann-benchmarks.com 2024 results and bake into golden test vectors.

---

## 4. Tiered roadmap

### Tier 1 — Cheapest one-day standalone (~720 LOC)
PR-1: N1 Index interface + N2 Distance helpers + N3 BruteForce + N4 KDTree + N8 RandomHyperplaneLSH + N20 ProductQuantization. **Lands the first vector-search-anything in repo; saturates R-LSH-COSINE-COLLISION + R-PQ-DISTANCE-UNBIASED.**

### Tier 2 — Highest-leverage one-week unlock (~1,750 LOC additional)
PR-2: N15 NN-Descent + N16 HNSW + N21 OPQ + N22 IVF-PQ + N23 ScaNN + N24 FilteredANN + N25 Reranker + N26 benchmark harness. **Saturates R-RECALL-AT-10 ≥ 0.95 across 4 indices; unblocks the entire post-2018 graph-based ANN family; reproduces ann-benchmarks 2024 leaderboard.**

### Tier 3 — Completionist (~1,680 LOC additional)
PR-3: N5 BallTree + N6 VPTree + N7 RTree + N9 MinHashLSH (imports 224-ST14) + N10 pStableLSH + N11 MultiProbeLSH + N12 C2LSH + N13 SpectralHashing + N14 ITQ + N17 NSG + N18 Vamana/DiskANN + N19 IVF-standalone. **Closes the full ANN canon; saturates remaining R-LSH-PSTABLE / R-LSH-MINHASH / R-MERGEABLE-INDEX pins.**

---

## 5. Cross-links

- **224-new-streaming-ST14 MinHash sketch** → 225-N9 MinHashLSH imports it. Sketch lives in `streaming/`, index lives in `nn/`. Same hash seeds, byte-identical interop.
- **077-geometry-missing T1-4 kdtree.go** → 225-N4 KDTree is the d-D generalisation. 077 ships 3-D for ICP/point-cloud; 225 ships d-D for vector search. Either separate `geometry/kdtree.go` (3-D-specific) + `nn/kdtree.go` (generic d-D), or unified `geometry/kdtree.go` parameterised by dimension.
- **097-linalg-missing k-means** → 225-N19 IVF needs k-means coarse quantizer. If 097 lands first, IVF imports `linalg.KMeans`; otherwise IVF ships a 60-LOC inline k-means.
- **174-G4 OnlineLearner / 220-F1 FiniteSumLoss / 221-O1 OnlineConvexLearner / 222-B1 StochasticBandit / 223-S1 SetFunction / 224-Sk1 Sketch[T]** → 225-N1 `Index[T]` is the seventh interface in the unified-substrate keystone. Same `(Build, Add, Query, Bytes, FromBytes)` pentad as 224-Sk1.
- **150 / 196 / 200 multi-vector retrieval / late-interaction ColBERT** → 225-N25 Reranker handles the MaxSim-aggregation case when `T = [][]float64`.

---

## 6. Risks / non-goals

- **GPU acceleration.** FAISS-GPU / cuVS are out of scope (zero-dep CPU only per CLAUDE.md). The CPU SOTA (HNSW + ScaNN) lands ~95% of the FAISS-GPU recall at 10× lower QPS — acceptable for `reality`'s zero-dep mandate.
- **Learned indices.** SOAR / DESCENT 2024+ neural-index variants are deferred — they require gradient descent + SGD which is in-scope (220) but the *index-from-neural-net* layer adds ML-systems complexity that doesn't belong in `reality` (belongs in `aicore` consumer).
- **Distributed / sharded ANN.** Multi-machine routing belongs above `reality` (in services). N+1 cross-link: `crypto.ConsistentHash` is the right primitive when a consumer wants to shard.
- **Curse of dimensionality docstring.** Every tree-based index (N4/N5/N6/N7) explicitly documents the BGRS 1999 lower bound and degrades-to-brute-force fallback above d ≈ 30 (kd) / 60 (ball) / 100 (vp). No silent-pessimal failure.

---

**End of 225-new-ann.md** — 23 primitives N1-N23 + benchmark harness, ~4,150 LOC, two PRs (720 + 1,750 LOC), seven R-pattern saturations.
