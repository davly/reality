# 373 — research-fft-libs (FFT library comparison 2026)

## Headline
For reality's MIT/zero-dep mandate, pocketfft (BSD-3) and PFFFT (BSD-like) are the architectural reference points; FFTW/MKL/KFR are educational only (GPL/proprietary), and pure-Go peers (gonum, gofft) bound what reality should beat in pure code.

## Survey

### FFTW 3.3.11 (Frigo & Johnson, MIT → MIT-licensed only as research; library is GPLv2)
The reference for tuned FFT performance. Codelet-based: precompiles small straight-line FFTs (sizes 2..64, plus radix-{2,3,4,5,7,…}) and a planner stitches them at runtime via dynamic programming over `MEASURE`/`PATIENT` flags. Latest stable is 3.3.11 (2024). Architecture lessons: planner+codelet split, Stockham + mixed-radix, Bluestein/Rader for primes, twiddle precomputation. License is GPLv2 (commercial license sold separately) — **incompatible with reality MIT**. Cannot be vendored, but the codelet idea is reproducible from the 1998 paper without copying code.

### Intel oneAPI MKL FFT (2025.2)
oneMKL DFTI is now redistributable under the Intel Simplified Software License (free of charge, redistribution allowed) but remains proprietary closed-source. 2025 release added distributed SYCL DFT API, GPU image compression, and Xeon-6 tuning. Provides FFTW-compat wrappers — drop-in for FFTW3. **License incompatible** with reality (no source, not MIT-redistributable as code). Ignore for adoption; only relevant as a benchmark target.

### AMD AOCL FFT (AOCL-FFTW)
AMD's fork of FFTW3 with Zen-tuned codelets. BSD-3-Clause for the AOCL wrappers but the underlying FFTW3 sources remain GPLv2 — **so the combined work is GPL**. No useful license advantage over upstream FFTW for reality. Architecture is identical to FFTW.

### KFR 6.x (kfrlib)
Modern C++17 DSP framework, dual-licensed GPLv2/v3 + commercial. Supports SSE/AVX/AVX-512/NEON/RVV with a unified `vec<T,N>` template layer. 2025 LIGO/Virgo benchmarks show parity-or-better vs FFTW for power-of-2 sizes. **License incompatible** (GPL or paid). Architectural lesson worth stealing: `vec<T,N>` abstraction lets one source compile to all SIMD ISAs — analogous in Go would be code generation per GOARCH.

### pocketfft (Reinecke, BSD-3)
NumPy 1.17+ and SciPy backend. ~3000 LOC C++ header-only. Mixed-radix (handles arbitrary N), Bluestein for primes — keeps O(N log N) where FFTPACK degraded. Vectorized only for multi-D / batched 1D. **BSD-3 is MIT-compatible** — could in principle be ported to Go. Best architectural reference for reality: handles non-power-of-2, accuracy on prime lengths matches complex FFT, no auto-tuner overhead. Reality should study its mixed-radix decomposition and Bluestein path before extending beyond radix-2.

### muFFT (Themaister, MIT)
Single-precision only, audio-focused, MIT. ~2000 LOC C. Uses **Stockham autosort** (no explicit bit-reversal — eliminates a non-trivial permutation pass) plus radix-2/4/8 butterflies. Runtime SSE/SSE3/AVX dispatch. Comparable to FFTW for sizes muFFT supports. **MIT — fully compatible with reality.** Two specific lessons: (1) Stockham removes the `bitReverse` pass currently in `signal/fft.go`; (2) radix-4 and radix-8 butterflies cut twiddle multiplications ~25% vs pure radix-2 with no SIMD needed.

### FFTS (Anthony Blake, BSD-3)
"Fastest FFT in the South." Cache-oblivious + runtime code generation; in published 2013 benchmarks beat FFTW, Intel IPP, Apple vDSP on x86 and ARM. Conjugate-pair (split-radix variant). Largely unmaintained since 2017. **BSD-3 compatible** but JIT codegen is unportable to pure Go. Architectural lesson is academic only: dynamic specialization of straight-line codelets per N is the fast path, but reality cannot do this without `unsafe` + assembly.

### KissFFT (Borgerding, BSD-3)
~500 LOC C, mixed-radix (any small-prime factorization). Floating or fixed-point via recompile. The canonical "small, simple, correct" FFT. **BSD-3, MIT-compatible.** Not the fastest, but its API design (one `kiss_fft_cfg` per (N, direction) holding precomputed twiddles, then cheap calls) is the cleanest existing model and maps perfectly onto a Go `type Plan struct { twiddles []complex128; n int }`. Reality should adopt this plan-cache pattern instead of recomputing twiddles every call.

### PFFFT (Pommier; marton78 fork active)
~2000 LOC C, BSD-like. Single-precision, real and complex 1D, SSE/NEON/Altivec. "Half as fast as fastest" by stated goal — yet benchmarks show it within 50% of FFTW on modern CPUs. Bluestein fork (cpuimage) handles non-power-of-2. **BSD compatible.** Two transferable ideas: (1) work-buffer parameter — never allocates inside the FFT call (reality already does this); (2) z-domain layout that keeps FFT output SIMD-friendly. Strong reference for the "small, fast, no-deps" niche reality occupies.

### Apple vDSP / Accelerate
Proprietary Apple framework. Split-complex format, power-of-2 only, plan-based (`vDSP_create_fftsetup`). On Apple Silicon (M1–M4) frequently outperforms MKL. **Incompatible** (closed-source, Darwin-only). Useful only as a benchmark reference; the split-complex memory layout (separate real/imag arrays) is exactly what reality already uses in `FFT(real, imag []float64)`.

### oneAPI MKL FFT (open-source DPC++ interface)
The DPC++ interfaces and the `oneAPI/oneMath` reference are open (Apache-2.0); the high-perf backend remains closed proprietary MKL. The open spec is a useful API design reference (DFTI descriptor with `Set_Value`, `Commit`, `ComputeForward`) but not a code source. Apache-2.0 spec is MIT-compatible.

### gonum/dsp/fourier (BSD-3, pure Go)
Direct port of Fortran FFTPACK. Mixed-radix — handles non-power-of-2. Slower than gofft on power-of-2. **BSD-3, no CGO.** This is reality's most honest competitor in the Go ecosystem. Reality currently only handles power-of-2; gonum already covers arbitrary N.

### gofft (argusdusty, MIT, pure Go)
Pure-Go radix-2 FFT, in-place, **O(1) extra space**. Claims to be the fastest pure-Go FFT. Single-threaded. **MIT — compatible.** Closest peer to reality's current `signal/fft.go`. Worth benchmarking reality directly against gofft as the pure-Go ceiling.

### go-dsp/fft (mjibson, BSD-3, pure Go)
Multi-threaded; allocates O(N) workspace. **BSD-3.** Slower per-core than gofft but scales with cores. Not a model for reality's zero-alloc design.

## Reality lessons

Concrete recommendations for `signal/fft.go`:

1. **Cache twiddles in a Plan struct (KissFFT/PFFFT pattern).** Currently `FFT()` recomputes `cos/sin` inside the loop. Add `type FFTPlan struct { n int; twiddles []float64 }` and `func NewFFTPlan(n int) *FFTPlan`. Per-call cost drops from `O(N log N)` trig evals to zero.

2. **Stockham autosort to drop the bit-reversal pass (muFFT pattern).** `bitReverse()` is currently a separate O(N) pre-pass with poor cache behavior. Stockham interleaves permutation into the butterfly stages at the cost of a second N-element scratch buffer. For reality's API this means an optional `scratch []float64` parameter — still zero-alloc.

3. **Radix-4 and radix-8 butterflies (muFFT/FFTW pattern).** Pure radix-2 does N/2·log2(N) complex muls. Radix-4 saves ~25% multiplications because half the twiddles in each radix-4 stage are ±1 or ±j. Pure-Go, no SIMD needed. Big win for reality before any assembly considered.

4. **Mixed-radix + Bluestein for non-power-of-2 (pocketfft pattern).** `isPow2()` requirement is a real limitation vs gonum. Add a Bluestein chirp-z path for prime/odd N, keeping O(N log N). pocketfft's source is the cleanest BSD-3 reference to study (do not copy — reimplement from the Bluestein 1968 paper, per CLAUDE.md rule 6).

5. **Plan-equality test vectors (golden-file impl).** When adding plans, golden tests must exercise both fresh and cached plans to catch twiddle-recompute drift.

6. **Do not pursue SIMD/JIT (FFTS/KFR pattern).** Out of scope for reality's pure-Go-no-deps mandate. Stop the architectural inspiration at "Stockham + mixed-radix + plan cache."

7. **License hygiene.** When porting algorithmic ideas, only consult MIT/BSD sources (pocketfft, muFFT, KissFFT, PFFFT, gofft, gonum). Do **not** read FFTW/KFR/AOCL source while writing Go — risk of derivative-work taint under GPL.

## Sources
- [FFTW Release Notes](https://www.fftw.org/release-notes.html)
- [pocketfft (NumPy 1.17 release notes)](https://numpy.org/devdocs/release/1.17.0-notes.html)
- [SciPy pocketfft backend issue](https://github.com/scipy/scipy/issues/10175)
- [KissFFT GitHub (mborgerding)](https://github.com/mborgerding/kissfft)
- [muFFT GitHub (Themaister)](https://github.com/themaister/mufft)
- [FFTS GitHub (anthonix)](https://github.com/anthonix/ffts)
- [PFFFT (marton78 fork)](https://github.com/marton78/pffft)
- [KFR GitHub (kfrlib)](https://github.com/kfrlib/kfr) and [pricing/dual license](https://www.kfrlib.com/purchase/)
- [Intel oneAPI Base Toolkit 2025 Release Notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-base-toolkit/2025.html)
- [Apple vDSP.FFT documentation](https://developer.apple.com/documentation/accelerate/vdsp/fft)
- [gonum/dsp/fourier](https://pkg.go.dev/gonum.org/v1/gonum/dsp/fourier)
- [gofft (argusdusty)](https://github.com/argusdusty/gofft)
- [go-dsp (mjibson)](https://github.com/mjibson/go-dsp)
- [Cooley-Tukey FFT (Wikipedia)](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm)
- [Notes on FFTs for implementers (Giesen)](https://fgiesen.wordpress.com/2023/03/19/notes-on-ffts-for-implementers/)
- [Comparison of several FFT libraries in C/C++ (RAL-TR-2020-003)](https://epubs.stfc.ac.uk/manifestation/45434584/RAL-TR-2020-003.pdf)
- [HPK FFT Benchmarks 2025](https://hpkfft.com/pdf/hpkfft-benchmarks-2025.pdf)
- [project-gemmi/benchmarking-fft](https://github.com/project-gemmi/benchmarking-fft)
