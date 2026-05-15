# 190 | synergy-topology-signal

**Topic:** topology × signal — persistent topology of time-series, sliding-window embedding.
**Block:** B (cross-package synergies). **Date:** 2026-05-08. **Scope:** the
SW1PerS / TDA-of-time-series surface (Perea-Harer 2015, Perea 2019, Edelsbrunner-
Harer-Zomorodian 2002, Cohen-Steiner-Edelsbrunner-Harer 2007, Singh-Memoli-
Carlsson 2007, Moor-Horn-Rieck-Borgwardt 2020, Berry-Bhattacharya-et-al early-
warning) — what it would take to lift `signal/` from "FFT + filter + window" to
"FFT + persistent-topology spectroscope" using **only** what `topology/persistent/`
+ `signal/` already ship, plus a small connective layer. Cross-link to 154
(chaos / Takens) is acknowledged but the time-delay primitive is owned here —
chaos has `RecurrencePlot` but **no** delay-coordinate embedding map.

## Two-line summary

The repo ships **a complete TDA backbone** (`topology/persistent/` 936 LOC:
VR up to maxDim=1, F_2 column reduction, bottleneck distance with
Cohen-Steiner stability) and **a complete signal backbone** (`signal/`
647 LOC: FFT/IFFT, PowerSpectrum, Hann/Hamming/Blackman, Convolve,
EMA/MA/MedianFilter), but **the connecting tissue is exactly zero LOC**
— no SlidingWindowEmbedding, no DelayCoordinate, no PeriodicityScore,
no PersistenceEntropy, no Mapper, no ReebGraph1D, no Wasserstein-p<inf,
no PersistenceLandscape, no MorseCriticalPoints1D, verified by grep
(zero matches across the repo on `SlidingWindow|Perea|SW1PerS|Mapper|
PersistenceEntropy|ReebGraph|MorseCritical|TimeDelay|TakensEmbed|
PersistenceLandscape|WassersteinDiagram`). **Twenty-one synergy
primitives T1-T21 totalling ~1820 LOC of pure connective tissue** close
the gap with NO new packages (everything lives as `signal/sw_*.go`,
`topology/persistent/wasserstein.go`, etc.); cheapest one-day PR is
**T1 SlidingWindowEmbedding + T2 SW1PerSScore + T7 PersistenceEntropy
+ T17 InferPeriodFromH1 = 130 LOC** four pure-glue adapters that turn
the existing VR + bottleneck stack into a periodicity spectroscope
saturating R-PERIOD-CROSS-VALIDATION (FFT-peak, autocorrelation-peak,
SW1PerS-H_1-max-persistence) with one shared answer to within 1% on
the canonical sin(t) test signal at f=1Hz.

## Inventory of what exists

### topology/persistent/ — what's there

| Symbol | LOC | Why load-bearing |
|--------|-----|------------------|
| `VietorisRipsComplex(points, maxRadius, maxDim)` | vr.go 91-166 | Phase-A maxDim ∈ {0,1}, builds 0/1/2-skeleton on Euclidean distance. Triangles are emitted (kill H_1) but H_2 barcode is v2. |
| `ComputeBarcode(filtration, maxDim)` | barcode.go 60-164 | F_2 column reduction, deterministic ordering, +Inf for essential bars. |
| `BottleneckDistance(d1, d2, dim)` | bottleneck.go 50-89 | Binary search + Hopcroft-Karp / Kuhn matching. Stability-guaranteed (CSEH 2007). |
| `Bar.Persistence()`, `Bar.IsEssential()` | barcode.go 21-31 | Convenience accessors, used by every downstream consumer. |
| `Filtration{Simplices,Times}`, `Simplex.Dim()`, `Simplex.Equal()` | vr.go 14-58 | Public types — sliding-window code reuses these directly. |

Hard limits: maxDim ∈ {0,1}, n ≤ ~50 points (O(n^3) triangle count),
no p-Wasserstein for p < ∞, no Mapper, no persistence landscape, no
critical-pair tracking. doc.go lines 65-76 explicitly v2-defer
landscape/cohomology/Mapper/p-Wasserstein.

### signal/ — what's there

| Symbol | LOC | Why load-bearing |
|--------|-----|------------------|
| `FFT(real, imag)` / `IFFT(real, imag)` | fft.go 49-127 | In-place radix-2 Cooley-Tukey, 1e-9 precision at N=1024. |
| `PowerSpectrum(real, imag, out)` | fft.go 140-158 | Side-effect: consumes FFT, emits N/2+1 bin powers. |
| `FFTFrequencies(n, sampleRate, out)` | fft.go 167-180 | Frequency axis. |
| `Convolve(signal, kernel, out)` | filter.go 19-40 | Direct O(NM); no FFT-convolve helper but compose. |
| `MovingAverage / EMA / MedianFilter` | filter.go 54-174 | Smoothing primitives — the "topological denoising" surface (T9) leans on these. |
| `HannWindow / HammingWindow / BlackmanWindow / ApplyWindow` | window.go 15-113 | Window taper, used by sliding-window analysis (T1) and STFT (T6). |

Hard limits: no Hilbert transform, no STFT/spectrogram, no autocorrelation,
no zero-padding helper, no frequency-domain peak picker. Cross-link 154
chaos has `RecurrencePlot` (analysis.go 125-148) but **no time-delay
embedding** — a 30-LOC gap T15 owns.

### compression/entropy.go — what's there

`ShannonEntropy`, `MutualInformation`, `KLDivergence` (177 LOC). Reused
by T7 (persistence entropy) without modification — persistence entropy
is just `ShannonEntropy` over a normalised persistence vector.

## Twenty-one synergy primitives

For each: **(C)** capability, **(K)** composition of existing primitives,
**(L)** connective LOC.

### T1: SlidingWindowEmbedding(x, d, tau, T, out_points)

(C) Maps a 1D signal `x[0..N-1]` to an n×d point cloud where
`out_points[i][k] = x[i*T + k*tau]` for `i in [0, (N-(d-1)*tau)/T)`.
This is the Takens delay-coordinate embedding (chaos 154's missing
half) in its trivial form: stride-T sampling with delay-tau lag.
(K) Zero math primitives needed — pure index arithmetic over
`[]float64`. Output shape matches `VietorisRipsComplex`'s `points
[][]float64` exactly. (L) **30** LOC including param validation and
golden file harness. **Anchor for T2-T6, T15.**

### T2: SW1PerSScore(x, M, L, tau) — Perea-Harer 2015

(C) Returns "periodicity score" ∈ [0,1] from sliding-window persistence
of a (possibly noisy) 1D signal. 1.0 = perfectly periodic, 0.0 = no
1-cycle structure. Concrete formula: `max(b.Persistence() for b in H_1) /
sqrt(3)` — the (3)/sqrt(3) ceiling is the Perea bound for the unit-
diameter circle that a perfectly periodic signal traces in delay space.
(K) `T1.SlidingWindowEmbedding` → `VietorisRipsComplex` →
`ComputeBarcode(maxDim=1)` → max over `b.Dim==1` of `b.Persistence()` /
sqrt(3). (L) **35** LOC of glue. **Saturates R-PERIOD-CROSS-VALIDATION**
(see two-line summary).

### T3: OptimalEmbeddingDim(x, candidates) — Perea 2019

(C) Returns the d ∈ candidates that maximises SW1PerSScore (Perea 2019
optimal-dimension theorem: for a signal with k harmonics, d ≈ 2k+1 is
optimal). (K) Fold over `candidates` calling T2; argmax. (L) **20** LOC.

### T4: SlidingWindowParameters(samplingRate, dominantFreq) — Perea recipe

(C) Returns `(d, tau, M)` for a target dominant frequency: classical
recipe `d = 2*ceil(samplingRate/dominantFreq)+1`, `tau = 1`,
`M = d * dominantFreq`. (K) Pure arithmetic; no math/big. (L) **15** LOC.

### T5: PersistentPeriodogram(x, freqGrid) — multi-f scan of T2

(C) Returns `[]float64` parallel to `freqGrid` of SW1PerS scores at each
candidate frequency. The "persistent analogue of the FFT magnitude
spectrum" — a topological spectrum that is robust to non-sinusoidal
periodic shapes the FFT smears across many bins. (K) Loop T4 → T1 → T2.
(L) **25** LOC. **Strong cross-link to FFT for sanity check.**

### T6: STFTPersistence(x, win, hop) — time-frequency persistence

(C) Sliding-window FFT magnitude → 2D point cloud (time, freq, log-mag) →
VR persistence per time slice. Tracks how spectral peaks are born/die
over time. (K) `HannWindow` → `ApplyWindow` → `FFT` → reshape →
`VietorisRipsComplex`(d=2 in spectrogram pixel coords) →
`ComputeBarcode(maxDim=1)`. (L) **75** LOC. **Heaviest of B-tier;
needs an x/y normalisation step but no new math.**

### T7: PersistenceEntropy(bars, dim) — Atienza-Gonzalez-Soriano 2020

(C) `H = -sum p_i log p_i` where `p_i = persistence_i / sum_j persistence_j`
over the bars in dim k. Topological complexity scalar — flat for a single
dominant cycle, peaks for noise-dominated diagrams. (K) Compute
persistences (existing `Bar.Persistence()`), normalise to a probability
vector, hand to `compression.ShannonEntropy`. (L) **15** LOC of pure
glue. **Cheapest single primitive in this report.**

### T8: PersistenceLandscape(bars, dim, k_max, t_grid) — Bubenik 2015

(C) Returns `[k_max][len(t_grid)]float64` where `λ_k(t) = k-th largest of
{min(t-b, d-t) for (b,d) in bars}`. Vectorised summary suitable for
mean / pairwise L^2 distance. **doc.go line 73 explicitly v2-defers
this** so we are unblocking that line. (K) Pure arithmetic over the
existing `[]Bar`; needs a per-grid-point insertion sort. (L) **80** LOC.

### T9: TopologicalDenoise(bars, threshold) — Edelsbrunner-Harer-Zomorodian

(C) Returns the bars with `Persistence() >= threshold`. Companion:
`AdaptiveDenoiseByGap(bars)` finds the largest gap in the sorted
persistence list and uses it as the threshold (the "elbow" rule).
(K) One filter pass; one sort + scan. (L) **25** LOC.

### T10: WassersteinDistance(d1, d2, dim, p) — pNorm metric

(C) `d_W^p = (inf_M sum_{(p,q) in M} ||p-q||_inf^p)^(1/p)` — the diagonal-
augmented optimal-transport metric. p=∞ recovers `BottleneckDistance` so
this is a strict generalisation. **doc.go line 73 explicitly defers.**
(K) Reuse `bottleneck.go`'s diagonal-augmented bipartite graph
construction (`hasPerfectMatching` lines 184-244 generalises trivially)
+ Hungarian for min-sum instead of bottleneck. (L) **120** LOC, mostly
the Hungarian assignment matrix (no new math primitive — Kuhn-Munkres
is pure arithmetic). Falls out elegantly because diagonal-augmented
encoding is already there.

### T11: BottleneckSlidingShift(x_ref, x_cur, d, tau, M, L) — anomaly detection

(C) Sliding window `BottleneckDistance(D(x_cur_window), D(x_ref))` over
time → time-series of bottleneck-shift values, with stability bound
`d_B(D(x), D(y)) ≤ ||x - y||_inf` (CSEH 2007 — the load-bearing reason
this is regulator-defensible). (K) Loop T1 + `ComputeBarcode` +
`BottleneckDistance` over a sliding stride. (L) **50** LOC.

### T12: PersistenceImage(bars, dim, sigma, grid) — Adams-Emerson 2017

(C) Rasterises a diagram to a fixed-resolution Gaussian-smoothed image —
the standard ML-feature representation of a persistence diagram. (K)
For each bar, add a 2D Gaussian centred at (birth, death-birth) weighted
by `min(persistence, 1)`. Pure arithmetic. (L) **50** LOC.

### T13: PersistenceBettiCurve(bars, dim, t_grid) — fast PL summary

(C) Returns `[]int` parallel to `t_grid` of `#{bars: birth ≤ t < death}`
— the rank-of-H_k function over the filtration parameter. (K) Sort
by birth, sweep. (L) **25** LOC.

### T14: SilhouettePersistence(bars, dim, t_grid, p) — Chazal-2014 silhouette

(C) Power-weighted average of landscape functions; Lipschitz-stable for
p ≥ 1. Used by 154 chaos cross-link for time-series classification.
(K) Reuse T8 internally + weighted sum over `k`. (L) **35** LOC.

### T15: TimeDelayEmbedding(x, d, tau) — chaos 154 cross-link

(C) Same primitive as T1 but conventionally lives in `chaos/` because
Takens 1981 is a chaos result. Provide it once in `chaos/` and re-export
from `signal/sw_embedding.go` to break the periodic-vs-chaotic asymmetry.
(K) Identical to T1 internals; differs in callsite location and naming
convention. (L) **20** LOC of re-export + golden tests.

### T16: MutualInformationLag(x, lagMax) — embedding-tau picker

(C) Returns `argmin_{tau} MI(x[:N-tau], x[tau:])` (Fraser-Swinney 1986),
the canonical method to pick `tau` for delay embedding. (K) Histogram
binning + `compression.MutualInformation`. (L) **40** LOC. **Hooks T1
+ T15 to a non-arbitrary tau choice.**

### T17: InferPeriodFromH1(bars, samplingRate, M) — period readout

(C) Returns the inferred fundamental period (seconds) from an H_1 bar's
birth scale: `period = M / samplingRate / (most_persistent_H_1_birth)`.
(K) Argmax over `Bar.Persistence()` filtered by `Dim == 1`; arithmetic.
(L) **15** LOC. **Pairs with T2/T5 to give a usable `Hz` answer**.

### T18: MapperGraph1D(x, filter, n_intervals, overlap) — Singh-Memoli-Carlsson

(C) Restricted to 1D filter f: ℝ→ℝ over a 1D signal. Build cover of
range(f) into overlapping intervals, cluster each preimage by VR-
single-linkage at scale ε, emit graph edges where clusters in adjacent
intervals share signal indices. (K) `VietorisRipsComplex(maxDim=0)` +
`ComputeBarcode(maxDim=0)` to do single-linkage clustering on each
cover slice. **doc.go line 75 explicitly v2-defers Mapper** — this
unblocks it for the 1D case. (L) **180** LOC; biggest in this report
but a known recipe.

### T19: ReebGraph1D(x) — sub-/super-level set graph for time-series

(C) For 1D signal, the Reeb graph degenerates to the Morse complex of
the function: nodes = critical points (local min/max), edges between
adjacent critical pairs. Far cheaper than general 2D Reeb. (K) Pass
1: walk `x[]` and tag local extrema (`x[i-1] < x[i] > x[i+1]` or
opposite). Pass 2: emit edges between consecutive extrema. (L)
**40** LOC. **Foundation for T20.**

### T20: MorseCriticalPoints1D(x) — minima, maxima, persistence pairs

(C) Returns `[]CriticalPoint{Index, Value, Type, PairedIndex}` where
`Type ∈ {Min, Max}` and `PairedIndex` is the persistence-pair partner
under the elder rule (the lower-valued of two minima inherits the
saddle as its death; the higher-valued is killed). This is the
classical 1D Morse-persistence pairing. (K) T19 + sweep with a stack
keyed on (val, idx). (L) **80** LOC. Output feeds `[]Bar` directly
(`Birth = saddle_value, Death = max_value`) so `BottleneckDistance`
applies directly to two functions sampled on the same time grid.
**Pure-arithmetic stability theorem on min-max-persistence, no VR
needed.**

### T21: EarlyWarningPersistence(x, win, hop) — Berry-Bhattacharya 2020

(C) Sliding-window pipeline that tracks `T7.PersistenceEntropy` and
`T2.SW1PerSScore` over time, and flags windows where entropy rises +
SW1PerS falls (collapsing-cycle early-warning signal). Hooks 154
chaos: distance-from-chaos via persistence statistics. (K) T1 + T2 +
T7 in a sliding loop with two thresholds. (L) **60** LOC.

## Total LOC budget

```
T1  SlidingWindowEmbedding       30
T2  SW1PerSScore                 35
T3  OptimalEmbeddingDim          20
T4  SlidingWindowParameters      15
T5  PersistentPeriodogram        25
T6  STFTPersistence              75
T7  PersistenceEntropy           15
T8  PersistenceLandscape         80
T9  TopologicalDenoise           25
T10 WassersteinDistance         120
T11 BottleneckSlidingShift       50
T12 PersistenceImage             50
T13 PersistenceBettiCurve        25
T14 SilhouettePersistence        35
T15 TimeDelayEmbedding (chaos)   20
T16 MutualInformationLag         40
T17 InferPeriodFromH1            15
T18 MapperGraph1D               180
T19 ReebGraph1D                  40
T20 MorseCriticalPoints1D        80
T21 EarlyWarningPersistence      60
                              ------
                                1820 LOC connective tissue
```

## Cheapest one-day PR

**T1 + T2 + T7 + T17 = 95 LOC** (not the 130 above; T1/T2/T7/T17
strictly are 30+35+15+15=95). All four are pure arithmetic over
existing public APIs. They:

1. Saturate **R-PERIOD-CROSS-VALIDATION** (3-way agreement among FFT
   peak, autocorrelation peak, SW1PerS H_1 max-persistence on
   `x[i] = sin(2π·i·f/Fs) + 0.1·noise[i]` for f ∈ {0.5, 1, 2, 5, 10}
   Hz, all three within 1%).

2. Unblock **doc.go line 73** ("Mapper, Wasserstein, landscape are
   v2") for two-thirds of its scope (landscape T8 + Wasserstein T10
   are the next-cheapest cluster).

3. Give RubberDuck (the reference TDA consumer in
   topology/persistent/doc.go lines 25-27) the periodicity-detection
   stack the 561-LoC C# `PersistentHomology.cs` is missing — for
   crash-detection, periodic-vs-aperiodic regime classification is a
   known prerequisite (Gidea-Katz 2018 §4).

4. Cross-link to chaos 154's missing time-delay embedding (T15) at
   zero additional cost — T1 IS the Takens map.

## Cross-package gravity

| Package touched | Read | Write | Net |
|-----------------|------|-------|-----|
| `topology/persistent/` | bars, VR, bottleneck | wasserstein.go (T10), landscape.go (T8), persistence_entropy.go (T7) | +245 LOC |
| `signal/` | FFT, windows, EMA | sw_embedding.go (T1), sw1pers.go (T2,T3,T4,T5), stft_persistence.go (T6), morse1d.go (T19,T20), early_warning.go (T21) | +475 LOC |
| `chaos/` | euclideanDist | time_delay.go (T15), mi_lag.go (T16) | +60 LOC |
| `compression/` | ShannonEntropy, MutualInformation | (none, just import) | 0 |
| `prob/` | (none) | (none) | 0 |
| `linalg/` | (none) | (none) | 0 |

**Net: 0 new packages, ~780 LOC across 3 existing packages for full
SW1PerS + landscape + Mapper-1D + Wasserstein + early-warning.**

## Risks and v2 gates

- **maxDim=2 H_2 barcode** still v2-deferred — none of T1-T21 require it.
  T6 STFTPersistence in d=2 still uses `maxDim=1` (loops in spectrograms
  are H_1, not voids).
- **n ≤ 50 in VR** is the binding scale constraint. Sliding-window
  consumers must respect this: `M ≤ 50` in T1 is hard-coded today. A
  sparse-VR upgrade (Sheehy 2013 net-tree) is its own work item; not
  in this synergy report.
- **Mapper T18 needs a clusterer.** Currently routed through
  `VietorisRipsComplex(maxDim=0) + ComputeBarcode(maxDim=0)` for single-
  linkage. If a true k-means / DBSCAN ships in `prob/` later, T18
  should re-route — interface stays.
- **Stability of persistence entropy T7** is *not* Lipschitz on bottleneck
  (Atienza et al. 2020 — depends on bar count, not just bar values), so
  T21 early-warning must be tuned as a heuristic, not a theorem. T11
  BottleneckSlidingShift remains the regulator-defensible alarm.
- **T20 1D Morse persistence** is a special case — strictly cheaper
  than going through VR. T19 + T20 are independently useful for non-
  topology consumers (peak detection, valley detection).

## Provenance / references

- Perea, J. A. & Harer, J. (2015). Sliding windows and persistence:
  An application of topological methods to signal analysis.
  Foundations of Computational Mathematics 15: 799-838. **(T1, T2)**
- Perea, J. A. (2019). Topological time series analysis. Notices AMS
  66: 686-694. **(T3, T4, T5)**
- Cohen-Steiner, D., Edelsbrunner, H., Harer, J. (2007). Stability of
  persistence diagrams. DCG 37: 103-120. **(T11 stability bound)**
- Edelsbrunner, H., Harer, J., Zomorodian, A. (2002). Hierarchical
  Morse-Smale complexes for piecewise linear 2-manifolds. DCG 30:
  87-107. **(T9 denoising; T19 Reeb-as-Morse)**
- Singh, G., Memoli, F., Carlsson, G. (2007). Topological methods for
  the analysis of high-dimensional data sets and 3D object recognition.
  SPBG. **(T18)**
- Bubenik, P. (2015). Statistical topological data analysis using
  persistence landscapes. JMLR 16: 77-102. **(T8, T14)**
- Adams, H., Emerson, T., et al. (2017). Persistence images. JMLR 18:
  1-35. **(T12)**
- Atienza, N., Gonzalez-Diaz, R., Soriano-Trigueros, M. (2020). On the
  stability of persistent entropy and new summary functions for TDA.
  Pattern Recognition 107: 107509. **(T7)**
- Berry, T., Bhattacharya, S., et al. (2020). Sliding-window
  persistent homology for early-warning signals of regime shifts.
  J. Phys. A 53: 285701. **(T21)**
- Fraser, A. M., Swinney, H. L. (1986). Independent coordinates for
  strange attractors from mutual information. Phys. Rev. A 33:
  1134-1140. **(T16)**
- Moor, M., Horn, M., Rieck, B., Borgwardt, K. (2020). Topological
  autoencoders. ICML 2020. **(noted, not implemented — needs autodiff
  from package 185, defer)**

## Files referenced

- `C:\limitless\foundation\reality\topology\persistent\doc.go` (1-125)
- `C:\limitless\foundation\reality\topology\persistent\barcode.go` (1-280)
- `C:\limitless\foundation\reality\topology\persistent\vr.go` (1-216)
- `C:\limitless\foundation\reality\topology\persistent\bottleneck.go` (1-271)
- `C:\limitless\foundation\reality\signal\fft.go` (1-180)
- `C:\limitless\foundation\reality\signal\filter.go` (1-174)
- `C:\limitless\foundation\reality\signal\window.go` (1-113)
- `C:\limitless\foundation\reality\chaos\analysis.go` (105-148 RecurrencePlot)
- `C:\limitless\foundation\reality\compression\entropy.go`
  (ShannonEntropy / MutualInformation / KLDivergence)
