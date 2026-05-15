# 301 — dive-fft-correctness (Cooley-Tukey / Split-Radix / Stockham / Bluestein audit + roundoff)

## Headline
Reality's `signal/fft.go` is a clean, zero-alloc, in-place Cooley-Tukey radix-2 (Decimation-In-Time, separated real/imag); it is correct and panics-on-misuse, but leaves three table-stakes pieces on the floor — twiddle-LUT cache (so 60 FPS Pistachio doesn't re-evaluate Cos/Sin every call), an RFFT for real input (~2x speedup), and Bluestein/mixed-radix for non-power-of-2 N. No Parseval test, no benchmarks.

## Findings (existing code audit)

Implementation: **Cooley-Tukey radix-2 DIT, in-place, separated-arrays (`real []float64`, `imag []float64`)**, not interleaved `complex128`. Confirmed by reading `C:\limitless\foundation\reality\signal\fft.go` lines 49–91.

- **Bit-reversal (`fft.go:21–35`).** Implements the Buneman-1986 incremental "next bit-reversed index" trick (no per-element `Reverse` call). Correct and zero-alloc. The `if i < j` guard on line 30 is the standard one-swap-not-two pattern. Good.
- **Butterfly twiddle (`fft.go:65–88`).** Uses **recurrence** `w_{k+1} = w_k · w_step`, not a precomputed LUT. Computes `Cos`/`Sin` once per stage (line 66–67), then advances by complex multiply. Pros: zero alloc. **Cons: error in the twiddle recurrence accumulates linearly within each stage** (k=0..halfSize-1). For N=2^20 this is ~1e6 multiplies in the innermost stage and the twiddle drifts by ~1e6·ε ≈ 2e-10. This is the **dominant error term, not the FFT itself.** A precomputed LUT would eliminate this entirely.
- **Allocation profile.** `FFT`, `IFFT`, `PowerSpectrum`, `FFTFrequencies` all allocate **zero** bytes per call (verified by inspection — no `make`, no slice append, no closures escaping). Good for Pistachio's 60 FPS path.
- **IFFT (`fft.go:101–127`).** Conjugate-FFT-conjugate-scale trick. Correct, but does **two extra O(N) passes** (lines 114–116 and 122–126). A direct DIT with sign flip would save those two passes; minor.
- **API.** Caller-owned slices, in-place mutation, panic-on-bad-input. Idiomatic for reality. **Not numpy-compatible** (numpy returns new array, takes complex; here you mutate parallel real/imag float64). This is fine — but document the convention difference.
- **PowerSpectrum (`fft.go:140–158`).** Mutates `real`/`imag` (calls `FFT` internally). **Documented but easy to miss** — caller cannot reuse the time-domain after calling `PowerSpectrum`. Lines 130–131 do warn.
- **N=1 short-circuit** (`fft.go:57–59`, `fft.go:109–111`): correct, returns identity. **N=0 is NOT short-circuited.** The `isPow2(0)` returns `false` (0 & -1 = 0, but `n>0` guard catches it), so `FFT([], [])` panics with "length must be a power of 2." Defensible but unfriendly — empty input is a degenerate-but-valid FFT. Recommend treating `N==0` as no-op (return).
- **Power-of-2 restriction enforced** (`fft.go:54–55`, panics). No mixed-radix, no Bluestein. Audio sample-rates 44100, 48000 force callers to zero-pad to 65536, paying ~1.5x cost.
- **No RFFT (real-input optimization).** ~50% of audio/spectrogram callers feed `imag = zeros`. A proper RFFT exploits Hermitian symmetry of `FFT(real)` and computes `FFT_N(real)` via a `FFT_{N/2}(complex)` plus O(N) post-processing — half the FLOPs, half the storage. Missing.
- **No Goertzel** for single-bin DFT (Pistachio note-detection use case).
- **No NTT cross-link.** Slot 293 NTT (number-theoretic transform) reuses the same butterfly skeleton with modular w_n. The two implementations don't share a butterfly template (FFT uses recurrence, NTT uses LUT in slot 293). A shared `butterflyDIT(reim []float64, twReal, twImag []float64, n int)` would unify.

### Test coverage (`signal/signal_test.go`)

Present:
- `TestGolden_FFT` (line 14) — golden file `testdata/signal/fft.json`. Good.
- `TestFFT_DCSignal`, `TestFFT_Impulse` (impulse → all-ones spectrum, line 70) — **the spike-detection sanity check.** Good.
- `TestFFT_Sinusoid_Peak` (line 83), `TestFFT_Sinusoid_Bin3` (line 103) — pure cosine → N/2 at bin k and N-k. **Both forms of the cross-check.** Good.
- `TestFFT_Length1`, `TestFFT_Length2` — small N. Good.
- `TestFFT_PanicNotPow2`, `TestFFT_PanicLengthMismatch` — input validation. Good.
- `TestIFFT_Roundtrip` (line 160), `TestIFFT_RoundtripComplex` (line 181) — `IFFT(FFT(x)) ≡ x` at 1e-10. **Round-trip identity pin present.** Good.
- `TestPowerSpectrum_Impulse`, `TestPowerSpectrum_DC`. Good.

Missing:
- **No Parseval/Plancherel test** `‖x‖² == ‖FFT(x)‖²/N`. This is the energy-conservation pin and is *the* second leg of the FFT correctness tripod. Trivial to add (~10 LOC).
- **No DFT-direct cross-validation.** A 5-line O(N²) brute-force DFT for N=8 cross-checked against `FFT` would saturate **R-MUTUAL-CROSS-VALIDATION 3/3** (round-trip + Parseval + direct-DFT).
- **No NaN/Inf input test.** `FFT([NaN,...], ...)` propagates NaN through butterflies — should be documented and tested.
- **No Hermitian symmetry test for real input** (output[k] ≡ conj(output[N-k]) when imag is zero) — would prove RFFT correctness when added.
- **No benchmarks.** `BenchmarkFFT_1024`, `BenchmarkFFT_65536` would catch perf regressions and guide the LUT-cache decision.
- **No accumulated-roundoff test** at large N (e.g., FFT of length-65536 random vector should round-trip to ε·log2(N) ≈ 16ε ≈ 3e-15).

## Concrete recommendations

1. **Add Parseval pin in `signal/signal_test.go` (~12 LOC, risk: zero).** ‖x‖² == (1/N)·Σ|X[k]|² to 1e-12 for N=8,32,1024 random signals. Closes the second R-MUTUAL leg.
2. **Add direct-DFT cross-check** `O(N²)` brute force for N=8, compare to `FFT` at 1e-13 (~25 LOC). Closes R-MUTUAL-CROSS-VALIDATION 3/3 audit pin.
3. **Add benchmarks** `BenchmarkFFT_{64,256,1024,4096,16384,65536}` (~30 LOC). Required to justify any subsequent perf change. Risk: zero.
4. **Twiddle-LUT cache (`fft.go:65–67`, ~50 LOC).** Replace per-stage `Cos`/`Sin` recurrence with a once-allocated `var twiddleCache sync.Map` keyed by N, holding `[]float64` of `2*(N/2)` packed (cosθ, sinθ) for the deepest stage; smaller stages slice into it. Eliminates *both* the recurrence-drift error AND the ~2·log2(N) trig calls. **Trade-off: now allocates per-N once.** For 60 FPS Pistachio at fixed buffer size this is amortized to zero. Risk: low (still in-place, behavior identical to ε).
5. **RFFT (new `signal/rfft.go`, ~120 LOC).** Implement `RFFT(x []float64, outReal, outImag []float64)` for `len(x) == N`, `len(out*) == N/2+1`, using the standard "pack real into complex, FFT length N/2, post-process" algorithm (Sorensen-Heideman-Burrus 1987). Halves time-domain storage and ~halves FLOPs. Risk: medium (post-processing has subtle edge cases at k=0 and k=N/2 — golden tests required).
6. **Bluestein chirp-z for arbitrary N (new `signal/bluestein.go`, ~100 LOC).** Lifts the power-of-2 restriction. O(N log N) for any N via FFT of size next_pow2(2N-1). Lets callers run FFT directly on 44100-sample buffers. Allocates O(N) workspace — provide an `FFTArbitrary` variant that accepts a workspace.
7. **Document N=0 behavior + treat as no-op** (`fft.go:54`). Change to `if n == 0 { return }` before the pow2 check, matching the N=1 short-circuit.
8. **Document the "PowerSpectrum mutates input" gotcha more loudly** (`fft.go:130–131`). Suggest a `PowerSpectrumOOP(x []float64, scratch, out []float64)` variant that doesn't clobber.
9. **NaN/Inf documentation + test.** State "NaN/Inf in input propagate; output is unspecified at those bins."
10. **Long-term: Stockham auto-sort variant** (`signal/fft_stockham.go`, ~150 LOC). Eliminates bit-reversal; ping-pong buffer; cache-friendlier on N>=4096. Required if Pistachio ever moves to GPU. Out-of-place by definition (allocates a length-N scratch). Risk: medium.
11. **Long-term: Split-radix DIT variant** (Yavne-Duhamel-Hollmann; ~200 LOC). 33% fewer multiplies. Worth doing only after #4 (LUT) — without LUT cache, the constant-factor win disappears in trig calls. Risk: high (split-radix is finicky to test against the existing golden file due to slightly different rounding order; new golden vectors needed).
12. **Cross-link with slot 293 NTT.** Refactor butterflies into `signal/butterfly.go` with a shared template parameterized over (twiddle-multiply, modular-add, modular-sub). Lets NTT and FFT share the bit-reversal + outer loop.

## Numerical roundoff analysis

**As-is (Cooley-Tukey radix-2 + recurrence twiddles)**:
- Higham (2002, *Accuracy and Stability of Numerical Algorithms*, ch. 24) bounds radix-2 FFT relative error by `γ_{log2 N} = (log2 N)·ε / (1 - log2 N · ε) ≈ log2(N)·ε` for the butterfly-only contribution. For N=1024, ε=2.2e-16, that gives ~2.2e-15.
- **The dominant error here is twiddle drift from the recurrence**, not the butterflies. The complex recurrence `w_{k+1} = w_k · w_step` has error growth `O(k · ε)` in `k`. Within a stage of size `size = 2^s`, the deepest stage has `halfSize = N/2`, so cumulative twiddle error at the last butterfly is `O((N/2)·ε)` ≈ 1.1e-10 at N=1024, **3 orders of magnitude worse than the FFT itself.**
- Empirically the existing tests pass at 1e-10 (`signal_test.go:176`, `:197`) which matches this analysis. Reality's docstring (`fft.go:45`) claims "1e-9 for 1024-point" — accurate.
- **With LUT cache** (recommendation #4): error drops to ~ε per twiddle (from `Cos`/`Sin` rounding) and overall to `O(log2(N)·ε)` ≈ 2e-15 at N=1024. **5 orders of magnitude better than current.**
- For N=2^20 (1M-point), current implementation has worst-case relative error ~1e-10, lookup version ~5e-15.
- **Stockham vs Cooley-Tukey roundoff**: equivalent (same butterfly count, same numerical structure). Stockham wins on cache, not precision.
- **Split-radix vs radix-2 roundoff**: ~33% fewer multiplies → constant-factor better, asymptotically same `O(log N · ε)`.

## Sources

Repo:
- `C:\limitless\foundation\reality\signal\fft.go` (181 LOC, the implementation)
- `C:\limitless\foundation\reality\signal\signal_test.go:14` (golden), `:52,70,83,103` (DC/impulse/sinusoid), `:160,181` (round-trip)
- `C:\limitless\foundation\reality\signal\filter.go:9–10` (notes that long-kernel convolution should compose FFT — i.e. consumers expect FFT to be perf-critical)
- `C:\limitless\foundation\reality\signal\window.go` (Hann/Hamming etc., feeds FFT)
- Slot 293 NTT review (cousin number-theoretic transform; LUT-twiddle architecture)

External (web-knowledge):
- Cooley & Tukey, *Math. Comp.* 19 (1965), "An algorithm for the machine calculation of complex Fourier series" — original radix-2.
- Yavne 1968 (split-radix discovered); Duhamel & Hollmann, *Electronics Letters* 1984 — modern formulation; Sorensen et al. *IEEE TASSP* 1986 — the standard reference.
- Stockham 1966 (in Cochran et al., *Proc. IEEE* 1967) — auto-sort.
- Bluestein 1968, *NEREM Record* — chirp-z transform for arbitrary N.
- Rader 1968, *Proc. IEEE* — prime-length FFT via cyclic convolution.
- Sorensen, Jones, Heideman, Burrus, *IEEE TASSP* 1987 — "Real-valued fast Fourier transform algorithms."
- Buneman 1986 — incremental bit-reversal index trick.
- FFTW (Frigo & Johnson, *Proc. IEEE* 2005) — codelet/planner architecture; the reference performance benchmark.
- Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., 2002, ch. 24 — `log2(N)·ε` bound.
