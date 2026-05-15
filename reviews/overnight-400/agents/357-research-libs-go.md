# 357 — research-libs-go (Go math ecosystem 2025-26 + reality positioning)

## Headline
Reality occupies a real gap in Go: pure-Go, MIT, zero-dep, cross-language golden-file-validated coverage of "applied math beyond linalg" (TDA, info-geo, info theory, queueing, color, acoustics, copula autodiff, ZK, conduit/forge) where gonum stops short and gorgonia/gnark are domain-narrow.

## Survey

### 1. gonum/gonum (BSD-3, mature)
The de facto Go numerical library. Submodules: `floats`, `mat`, `lapack/{gonum,native}`, `blas/{gonum,blas64,cblas128}`, `optimize`, `stat`, `graph`, `integrate`, `diff/fd`, `interp`, `dsp/fourier`, `spatial`. Six-month release cadence aligned to Go releases; v0.13.x is the current series in 2026, actively developed. **Overlap with reality:** linalg (gonum is broader/faster — has full LAPACK/BLAS, sparse, eigen), optim (gonum has L-BFGS, CMA-ES, Nelder-Mead — comparable scope), prob/stat (gonum has distributions + samplers but no Bayesian), graph (overlaps Dijkstra/A*/topo — gonum deeper on flow/community), integrate+diff (gonum has Romberg, fd; reality has Simpson/RK4). **No overlap:** color, acoustics, em, fluids, orbital, queue, gametheory, info-geo, topology, sequence, audio, autodiff, zkmark, changepoint. **Lesson:** gonum prefers row-major dense `mat.Dense` with a strong `Matrix` interface; reality's lighter `[][]float64` style is fine for breadth but a `linalg.Matrix` interface boundary would ease interop.

### 2. gorgonia/gorgonia (Apache-2.0, semi-active, pre-1.0)
Dataflow-graph ML library with autodiff, tensors, CUDA via CGO. README explicitly notes API not stable until 1.0. **Overlap:** autodiff (reality's pure-Go forward+reverse competes; gorgonia is reverse-mode tape-based and ML-shaped). **Differences:** gorgonia tensors are heap-allocated graph nodes; reality's autodiff is allocation-conscious (CLAUDE.md rule 3) and avoids CGO entirely. **Lesson:** Go ML autodiff has been pre-1.0 for 5+ years — there is room for a lean, pure-Go autodiff that doesn't pretend to be a DL framework. Reality's `autodiff` package fits this niche.

### 3. cpmech/gosl (Apache-2.0, niche)
Go scientific library with FFT, Bessel, elliptic, NURBS, transfinite interp, Mersenne twister, ODE, optimization. **Strong but CGO-heavy:** links OpenBLAS, LAPACK, UMFPACK, MUMPS, QUADPACK, FFTW3. Docker-only install path on Windows. **Overlap:** signal (FFT), calculus (quadrature), prob distributions, optim. **Lesson:** gosl's depth-via-FFI illustrates exactly the trade reality refuses — gosl gets numerical performance at the cost of CGO, vendoring, and zero Windows-native portability. Reality's "pure Go, no deps" rule is a market position, not a limitation, on Windows/WASM/cross-compile targets.

### 4. itsubaki/autograd, pbenner/autodiff, pointlander/gradient (mixed licenses)
Three small autodiff libraries, all single-author. `pbenner/autodiff` is most complete (MagicScalar with 1st+2nd order derivatives, sparse vec/mat, optim, statistics). `itsubaki/autograd` mimics Chainer/PyTorch DefineByRun. None has cross-language test vectors. **Overlap:** reality/autodiff package. **Lesson:** the Go autodiff niche is fragmented; reality consolidating autodiff into a package with golden-file tests has an immediate differentiation story.

### 5. mjibson/go-dsp (BSD-3, low maintenance)
Original repo now at `madelynnblue/go-dsp`. Contains `fft`, `dsputils`, `spectral` (Pwelch), `wav`, `window`. **Overlap:** reality/signal (FFT, windows, Hilbert), reality/audio. **Differences:** go-dsp has wav I/O reality doesn't; reality has Hilbert + filter design + onset detection go-dsp lacks. **Lesson:** go-dsp shows what minimal-DSP looks like in Go; reality/signal is broader and now has 3-detector cross-validation per recent commits. Reality could absorb a small `wav` reader for parity.

### 6. consensys/gnark (Apache-2.0, production)
Production zk-SNARK library powering Linea zk-rollup. Groth16, Plonk, R1CS, Aurora. Circuits in Go via `frontend.Variable`. **Overlap:** reality/zkmark — but gnark is a full proving system, reality/zkmark is presumably a smaller commitment-oriented primitive. **Lesson:** gnark proves Go can do production cryptography without CGO. If reality/zkmark targets developer-comprehensible commitment proofs (Pedersen, Merkle, KZG) rather than full SNARKs, it complements rather than duplicates.

### 7. shopspring/decimal, cockroachdb/apd, govalues/decimal (mixed)
Three competing decimal libraries. `shopspring/decimal` (MIT, immutable, slowest), `cockroachdb/apd` (Apache-2.0, mutable, fastest, IEEE 754-2008 General Decimal Arithmetic), `govalues/decimal` (MIT, correctly rounded, faster than shopspring without apd's mutability). **Overlap:** none with reality directly (reality is float64-centric). **Lesson:** reality's CLAUDE.md notes 256-bit math/big internal computation for golden-file generation — apd's IEEE 754-2008 spec is the right reference for any decimal-rounded constants, and govalues' correctness focus mirrors reality's golden-file philosophy.

### 8. rocketlaunchr/dataframe-go (MIT, semi-maintained)
Pandas-style dataframes, interoperates with gonum. **Overlap:** none with reality (reality is library-of-functions, not a data layer). **Lesson:** Go's data-science layer is thin and structurally orphaned from gonum; reality stays out of this space, correctly — golden-file purity demands functions, not dataframes.

### 9. gonum/graph [DEPRECATED] / sbinet/gonum-graph (mixed)
`gonum/graph` was deprecated in favor of being merged into main gonum. **Overlap:** reality/graph (Dijkstra, A*, BFS/DFS, topological sort). **Differences:** gonum's network analysis is deeper (flow, betweenness, community); reality is breadth-shaped. **Lesson:** gonum-graph's deprecation history shows the cost of fragmenting the ecosystem — keep reality/graph inside the reality monorepo to avoid the same fate.

### 10. go-gl/mathgl (BSD-3, mature)
Pure-Go 3D math (mgl32, mgl64) for graphics; quaternions, matrix transforms, projection. **Overlap:** reality/geometry (quaternions, projective geometry). **Differences:** mathgl is graphics-API-shaped (column-major, OpenGL conventions); reality/geometry is math-shaped (SDFs, convex hull, curves). **Lesson:** mathgl is the prior art for "pure Go, zero-dep, graphics-math" — confirms reality/geometry's design choice and that there's no value in reimplementing graphics primitives.

### 11. wenta/timeseries-go (MIT, small)
Time-series utilities: slicing, rolling windows, resampling, Z-Score/Robust-Z anomaly detection. **Overlap:** reality/timeseries, reality/changepoint (anomaly+drift detection). **Lesson:** the Go time-series space is split among small single-author libraries; reality/changepoint with proper changepoint algorithms (PELT, BOCPD, CUSUM) and golden files is differentiated.

### 12. samuell/awesome-scientific-go (curated list)
Survey index. Notable: `gostat`, `stats` (montanaflynn), `sparse` (james-bowman), `ODE` (ChristopherRabotin), `gonum/plot`. None overlap reality's exotic packages (info-geo, topology, queueing, copula, audio onset, color CIEDE2000, em, fluids, orbital, gametheory). **Lesson:** the Go applied-math long tail is a sparse archipelago of small unmaintained libraries — reality consolidates these under one MIT roof with shared test infrastructure.

## Reality vs gonum coverage matrix

| Domain | gonum | reality | Verdict |
|---|---|---|---|
| Linear algebra | Mature (LAPACK/BLAS, sparse, eigen, SVD) | Basic (LU/QR/Cholesky, PCA, sparse) | gonum wins; reality is sufficient for downstream packages |
| Optimization | Mature (L-BFGS, CMA-ES, Nelder-Mead, simplex) | Mature (bisection, Newton, L-BFGS, SA, GA, simplex) | Comparable; reality has GA/SA gonum lacks |
| Probability/stats | Mature distributions+samplers; no Bayes | Distributions + Bayesian + hypothesis + info-theory | reality broader |
| Graph | Deeper (flow, centrality, community) | Breadth (Dijkstra, A*, topo, network analysis) | gonum wins on depth |
| Integration/diff | Romberg, fd, ODE quad | Simpson, trapezoidal, RK4, root finding | Comparable |
| Signal/DSP | dsp/fourier (FFT only) | FFT/IFFT, conv, filters, windows, Hilbert | reality wins |
| Geometry | Spatial (kd-tree, R-tree only) | Quaternions, SDF, curves, hull, projective | reality wins |
| Color | none | 8 spaces, CIEDE2000, WCAG, Bradford | reality unique |
| Acoustics | none | Sabine, Doppler, A-weighting, dB SPL | reality unique |
| EM/Fluids/Orbital | none | All three | reality unique |
| Queueing theory | none | M/M/1, M/M/c, M/G/1, Erlang B/C | reality unique |
| Game theory | none | Nash, Shapley, replicator | reality unique |
| Combinatorics | none | Permutations, Catalan, Stirling, Bell | reality unique |
| Compression | none | Entropy, RLE, Huffman, LZ77 | reality unique |
| Crypto / number theory | none | Primality, modular, PRNGs, hashes | reality unique |
| Control theory | none | PID, transfer functions, Bode, stability | reality unique |
| Chaos / dynamics | none | Lorenz, Van der Pol, Lyapunov | reality unique |
| Autodiff | none (gorgonia is separate) | Forward + reverse mode | reality unique among gonum-style libs |
| Topology (TDA) | none | persistent homology (per CLAUDE.md) | reality unique in Go ecosystem |
| Info geometry | none | Fisher, KL, divergences | reality unique in Go ecosystem |
| Sequence (string sim) | none | Soundex, Dice n-gram, TokenSetRatio (recent commits) | reality unique |
| Audio onset | none | 3-detector cross-validated (recent commit) | reality unique |
| ZK proofs | none | zkmark | partial overlap with gnark, different niche |
| Constants | none (uses math.Pi etc) | Full SI 2019, NIST CODATA 2018 | reality unique |

## What's missing in Go ecosystem (reality's USP)

1. **Cross-language golden files.** No Go library publishes deterministic JSON test vectors validated against Python/C++/C#. gonum's tests are Go-internal. This is reality's single most defensible technical moat — anyone consuming reality from a polyglot stack (e.g., aicore + Python ML pipeline + C++ engine) gets bit-exact agreement guarantees nobody else offers in Go.

2. **Pure-Go × MIT × zero-dep triple uniqueness.** gonum is BSD-3 (compatible but viral-attribution); gorgonia is Apache-2.0 + CGO-CUDA; gosl is Apache-2.0 + CGO LAPACK/FFTW. No major Go math library is simultaneously MIT, pure-Go, and zero-dep. Reality is.

3. **Topological data analysis.** No Go TDA library exists. GUDHI/Ripser/giotto-tda are Python/C++/Julia. Reality/topology is unique.

4. **Information geometry.** Fisher information, KL divergence as a manifold, dual connections — no Go library implements this. reality/infogeo is unique.

5. **Stochastic differential equations** (if reality/chaos or a dedicated SDE package adds Euler-Maruyama, Milstein, Heun-SDE). Currently absent in Go entirely.

6. **Queueing theory.** No Go library has M/M/c, M/G/1, Erlang formulas. reality/queue is unique.

7. **Acoustics, EM, fluids, orbital as first-class packages.** Engineering math is wholly missing from Go; reality is the only library covering it.

8. **Domain-aware game theory.** No Go library has Shapley value, Nash equilibrium solvers, replicator dynamics. reality/gametheory is unique.

9. **Audio onset detection with cross-validation between detectors** (per recent commit `audio/onset` 3-detector cross-validation). No Go audio analysis library does this.

10. **Sequence similarity** with Soundex, Sørensen-Dice n-gram, TokenSetRatio (per recent commits). Go has scattered string-distance libs (Levenshtein only); reality/sequence consolidates RapidFuzz parity in pure Go.

11. **Constants with provenance.** No Go library cites NIST CODATA 2018 or SI 2019 as queryable metadata. reality/constants is unique.

## Sources
- [gonum/gonum GitHub](https://github.com/gonum/gonum)
- [Gonum homepage](https://www.gonum.org/)
- [gonum/gonum releases](https://github.com/gonum/gonum/releases)
- [gorgonia/gorgonia GitHub](https://github.com/gorgonia/gorgonia)
- [gorgonia.org pkg.go.dev](https://pkg.go.dev/gorgonia.org/gorgonia)
- [cpmech/gosl GitHub](https://github.com/cpmech/gosl)
- [pbenner/autodiff GitHub](https://github.com/pbenner/autodiff)
- [itsubaki/autograd GitHub](https://github.com/itsubaki/autograd)
- [mjibson/go-dsp / madelynnblue/go-dsp GitHub](https://github.com/madelynnblue/go-dsp)
- [Consensys/gnark pkg.go.dev](https://pkg.go.dev/github.com/consensys/gnark)
- [shopspring/decimal GitHub](https://github.com/shopspring/decimal)
- [cockroachdb/apd GitHub](https://github.com/cockroachdb/apd)
- [govalues/decimal GitHub](https://github.com/govalues/decimal)
- [rocketlaunchr/dataframe-go GitHub](https://github.com/rocketlaunchr/dataframe-go)
- [gonum/graph DEPRECATED](https://github.com/gonum/graph)
- [go-gl/mathgl GitHub](https://github.com/go-gl/mathgl)
- [wenta/timeseries-go GitHub](https://github.com/wenta/timeseries-go)
- [samuell/awesome-scientific-go](https://github.com/samuell/awesome-scientific-go)
- [Awesome Go: Science and Data Analysis](https://awesome-go.com/science-and-data-analysis/)
- [JetBrains: The Go Ecosystem in 2025](https://blog.jetbrains.com/go/2025/11/10/go-language-trends-ecosystem-2025/)
- [sebdah/goldie](https://github.com/sebdah/goldie)
- [gotest.tools/v3/golden](https://pkg.go.dev/gotest.tools/v3/golden)
