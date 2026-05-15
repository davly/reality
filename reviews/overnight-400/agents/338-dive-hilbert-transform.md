# 338 — dive-hilbert-transform (FFT-based / FIR / All-pass IIR / Half-band audit)

## Headline
Hilbert transform is wholly absent from `signal/` — Marple-1999 FFT path is a
~80-LOC composition of existing `signal.FFT`/`IFFT` and unblocks envelope
detection, instantaneous frequency, SSB, beam-forming, and Hilbert-Huang EMD.

## Findings

- **Confirmed gap.** `Hilbert`, `AnalyticSignal`, `EnvelopeDetect`,
  `InstantaneousFrequency`, `AllPassHilbert` — zero matches in any `.go` file
  under `C:/limitless/foundation/reality/`. `signal/` ships only
  `FFT`/`IFFT`/`PowerSpectrum`/`FFTFrequencies` (`signal/fft.go:49,101,140,167`),
  `Convolve`/`MovingAverage`/`EMA`/`MedianFilter` (`signal/filter.go:19,54,97,130`),
  and Hann/Hamming/Blackman/ApplyWindow (`signal/window.go:15,44,76,104`). No
  RFFT, no FFT-convolve, no fractional delay, no Hilbert.
- **Already named by 4 prior agents.** 132-signal-missing §1.7
  ("Hilbert transform (FFT-based analytic signal) — Six lines once RFFT exists.
  API: `Hilbert(signal, outReal, outImag)`, `AnalyticEnvelope`,
  `InstantaneousPhase`"); 133-signal-sota names Hilbert as a SciPy parity gap;
  166-synergy-acoustics-signal lists Hilbert as the blocker for Velvet-noise
  envelope fitting and beam-forming weight design;
  132-signal-missing §2.6 calls Tier 1.7 Hilbert the keystone for EMD
  (Hilbert-Huang transform) and §2.7 VMD. No prior agent has written the code
  or designed the API in detail — this dive does.
- **Marple 1999 algorithm is the standard.** Marple, S. L. (1999). "Computing
  the discrete-time analytic signal via FFT." IEEE Trans. Sig. Proc. 47(9):
  2600–2603. Algorithm: FFT real input → multiply spectrum by `H[k]` where
  `H[0]=H[N/2]=1`, `H[k]=2` for `1≤k<N/2`, `H[k]=0` for `N/2<k<N` → IFFT. The
  imaginary part of the result is `H[x]`; the magnitude is the analytic
  envelope. Exact for length-N power-of-2 inputs; preserves DC and Nyquist
  (matches `scipy.signal.hilbert` byte-for-byte modulo round-off).
  Composition over existing `signal.FFT`+`signal.IFFT`: ~80 LOC including
  buffer management, edge cases (N=1, N=2, odd N panic), and golden vectors.
- **FIR Hilbert (Type-III/IV).** Truncate ideal impulse response
  `h[n] = 2·sin²(πn/2)/(πn)` (zero at even n, `2/(πn)` at odd n) windowed by
  Hamming/Blackman/Kaiser. ~80 LOC. Real-time-friendly (sample-by-sample, no
  block delay beyond filter length); but inferior stop-band attenuation vs
  FFT path for the same compute budget. Mitra (2001) §10.5 is the textbook
  derivation. SciPy provides `firwin`/`remez` with `pass_zero=False` and
  `type='hilbert'` for Parks-McClellan; reality's PM solver is in tier
  316-dive-fir-design (status: not yet shipped) so a windowed-FIR variant is
  the day-1 path.
- **Schüßler-Steffen 1985 all-pass IIR.** Two parallel chains of 2nd-order
  all-pass sections whose phase responses differ by exactly 90° across the
  pass-band. Schüßler, H. W., Steffen, P. (1985). "Halfband filters and
  Hilbert transformers." Circuits, Systems, and Signal Processing 17(2):
  137–164. Reference implementation: 4–8 sections gives 80–120 dB stop-band
  attenuation at ~5% of the FFT-path compute. Cheap real-time (per-sample
  state update, no block delay). ~150 LOC including biquad section coefficient
  tables (Vaidyanathan 1993 design tables) and per-section state. Best fit
  for streaming audio (vibration package, beam-forming).
- **Half-band filter pair.** Decompose a half-band low-pass into a polyphase
  pair where the second branch IS the Hilbert transformer. Gives both LP and
  Hilbert outputs from one filter design at no extra cost. Vaidyanathan 1993
  §4.6. ~120 LOC. Lower priority — useful only when consumer also needs
  half-band decimation (i.e., DSP downsampling pipelines, not present in
  `reality` yet).
- **scipy.signal.hilbert is the cross-language oracle.** SciPy's
  implementation is literally Marple 1999 (FFT path). Golden files generated
  by Go can be byte-equal-validated against SciPy at `1e-12` tolerance for
  pow-2 lengths. Saturates the cross-language R-pin trivially.
- **Three R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities** (regression-test
  ready):
  1. **FFT-Hilbert ≡ FIR-Hilbert ≡ IIR-Hilbert** on a band-limited test
     signal (cosine sweep at 0.1–0.4 of Nyquist). All three implementations
     should agree to `1e-6` in the pass-band centre. Diverges at band edges
     (FIR transition band, IIR cutoff frequency) — pin documents this
     explicitly.
  2. **Envelope of `cos(ωt)` is identically 1.** Analytic signal of pure
     cosine is `exp(jωt)`, so `|x_a| = 1` exactly. Pin to `1e-9` in the
     interior of the buffer (boundaries leak from FFT periodicity, mask out
     first and last few samples).
  3. **Instantaneous frequency of `cos(ωt)` is ω.** Phase derivative of
     `exp(jωt)` is ω. Pin to `1e-7` after central differencing the phase.
- **Cross-link consumers.**
  - `audio/onset/`: today's onset detectors are spectral-flux/energy/SuperFlux
    based (`audio/onset/spectral_flux.go`, `audio/onset/superflux.go`).
    Hilbert envelope gives a true time-domain onset detector (no STFT) — see
    Bello et al. 2005, "A Tutorial on Onset Detection in Music Signals." Slot
    005 (audio missing) and 167 (audio×signal synergy) both note this.
  - `audio/vibration/`: amplitude-envelope-based fundamental-frequency
    extraction needs Hilbert (Sandberg 2010 method).
  - `audio/pitch/yin.go`, `audio/pitch/mpm.go`: instantaneous-frequency
    refinement of pitch estimates.
  - `acoustics/array/` (proposed in 166): beam-forming weight design needs
    analytic signal for relative phase — Hilbert is a hard prerequisite.
  - `sequence/` (slot 165): no consumer yet, but instantaneous-frequency
    descriptors are a candidate sequence-similarity feature.
  - **EMD/Hilbert-Huang** (132 §2.6): instantaneous-frequency analysis of
    intrinsic mode functions requires Hilbert.
  - Communications: SSB modulation/demodulation (textbook Hilbert
    application).
- **Composition cost ranking** (LOC including tests + golden files):
  - T0 FFT-Hilbert + analytic-signal API: **~80 LOC** core + ~120 LOC tests.
  - T1 Envelope + Instantaneous frequency helpers: **~60 LOC** core + ~80 LOC
    tests.
  - T2 FIR-Hilbert (windowed, Hamming default): **~80 LOC** core + ~100 LOC
    tests. Blocked-soft on Kaiser window (slot 317 — not yet shipped) for the
    sharpest stop-band; Hamming default works day-1.
  - T3 All-pass IIR Hilbert (Schüßler-Steffen): **~150 LOC** core + ~120 LOC
    tests. Coefficient table from Vaidyanathan 1993 = 4 numbers per section ×
    4 sections = 16 floats hardcoded.
  - T4 Half-band filter Hilbert pair: **~120 LOC** + ~100 LOC tests.
- **Day-1 PR is unambiguous: T0 + T1, ~140 LOC.** Pure composition over
  shipped `signal.FFT`/`signal.IFFT`. No new dependencies. Saturates two of
  the three R-pins (envelope-of-cosine, instfreq-of-cosine) on day 1; T0 + T2
  + T3 in week 2 saturates the FFT≡FIR≡IIR R-pin.
- **API surface (proposed).** Place in new file `signal/hilbert.go`:
  ```go
  // Hilbert computes the analytic signal x_a[n] = x[n] + j·H[x][n] via FFT
  // (Marple 1999). Length must be a power of 2. outReal and outImag must
  // have the same length as x. Zero allocations in hot path.
  func Hilbert(x []float64, outReal, outImag []float64)

  // AnalyticEnvelope writes |x_a[n]| (instantaneous amplitude) into out.
  // Composes Hilbert internally with caller-supplied scratch buffer.
  func AnalyticEnvelope(x []float64, scratch, out []float64)

  // InstantaneousPhase writes arg(x_a[n]) into out (unwrapped, range -π..π
  // pre-unwrap). Caller may apply UnwrapPhase if needed.
  func InstantaneousPhase(x []float64, scratch, out []float64)

  // InstantaneousFrequency writes (1/2π)·dφ/dt into out using central
  // differences. Returns radians-per-sample; multiply by sampleRate/(2π) for Hz.
  func InstantaneousFrequency(x []float64, scratch, out []float64)
  ```
  All four functions accept caller-supplied output buffers and target zero
  allocation in steady state — matches the buffer convention established in
  `signal/window.go:15`/`signal/fft.go:49`.
- **Edge cases the implementation must pin.**
  - `len(x) == 1`: analytic signal is `x` itself with zero imaginary part —
    no panic.
  - `len(x) == 2`: degenerate — Hilbert treats `x[0]` and `x[1]` as DC and
    Nyquist; both H values are zero. Document explicitly.
  - Length-mismatch panics: same convention as `signal.FFT` (panic with
    `signal.Hilbert: real and imag slices must have equal length`).
  - Non-pow-2 length: panic — defer Bluestein to slot 132-signal-missing T2.
  - DC bias: H[DC] = 0, so DC component passes through real, imag is zero at
    DC bin. Round-trip preserves DC.
  - Real input only: do not accept complex input — that is a different
    primitive (complex-to-complex Hilbert is `−i · sgn(ω) · X(ω)`, the same
    formula but applied symmetrically).
- **Precision pinning.** FFT-path Hilbert inherits FFT's documented `1e-9`
  precision for 1024-point inputs (`signal/fft.go:45`). Envelope of pure
  cosine should pin to `1e-9` internally, `1e-7` near boundaries (FFT-induced
  Gibbs leakage at the periodic boundary). Instantaneous frequency via
  central difference loses 2 orders of magnitude → pin to `1e-7`.
- **No conflict with other dives.** 316-dive-fir-design (Parks-McClellan)
  enables a sharper FIR Hilbert (T2'), but T2 windowed-FIR ships day-1
  without it. 317-dive-window-functions (Kaiser/DPSS) likewise enhances T2
  but does not gate T0/T1. 337-dive-fft-fractional is orthogonal —
  fractional-delay filters are a separate primitive that compose with
  Hilbert (delay-and-Hilbert beam-formers) but neither blocks the other.

## Concrete recommendations

1. **Day-1 PR (T0 + T1, ~140 LOC, ~250 LOC with tests).** Add
   `signal/hilbert.go` with `Hilbert`, `AnalyticEnvelope`,
   `InstantaneousPhase`, `InstantaneousFrequency`. Marple 1999 algorithm —
   pure composition of existing `signal.FFT` and `signal.IFFT`. Cite
   `Marple, S. L. (1999). "Computing the discrete-time analytic signal via
   FFT." IEEE TSP 47(9): 2600–2603` in doc comment per project rule 4.
2. **Golden files.** `signal/testdata/hilbert/` with 30 vectors covering: (a)
   pure cosine at multiple frequencies (envelope ≡ 1), (b) AM signal
   `(1+0.5cos(ω₁t))·cos(ω₂t)` (envelope = 1+0.5cos(ω₁t)), (c) FM signal
   `cos(ω₀t + β·sin(ω_mt))` (instfreq = ω₀ + βω_m·cos(ω_mt)), (d) DC input
   (envelope = |DC|, phase = 0), (e) impulse (analytic-signal magnitude has
   known closed form involving sinc), (f) IEEE 754 edge cases per project
   rule. Generate via SciPy reference; cross-check via Marple-formula
   first-principles in `math/big` at 256-bit precision per project
   golden-file convention.
3. **Saturate R-MUTUAL-CROSS-VALIDATION 3/3 day-1.** Two pins land with T0+T1
   alone:
   - `TestHilbert_EnvelopeOfCosineIsConstant` (1e-9 tol, mask boundaries).
   - `TestHilbert_InstFreqOfCosineIsOmega` (1e-7 tol).
   - Third R-pin (FFT≡FIR≡IIR) deferred to week 2 with T2+T3.
4. **Week-2 PR (T2 FIR Hilbert, ~80 LOC).** Windowed-FIR Hilbert with
   Hamming default (Kaiser when 317 ships). Adds the third R-pin.
5. **Week-3 PR (T3 All-pass IIR, ~150 LOC).** Schüßler-Steffen 1985 with
   hardcoded coefficient tables from Vaidyanathan 1993. Saturates the FFT≡FIR≡IIR
   3/3 pin and unlocks streaming/real-time consumers (audio/onset
   time-domain detector, audio/vibration envelope tracker, communications
   SSB).
6. **Defer T4 (half-band pair).** Wait for a downsampling consumer to
   materialise; today no consumer needs the LP+Hilbert pair simultaneously.
7. **Update consumer reviews.** Annotate 132-signal-missing §1.7,
   166-synergy-acoustics-signal §A8/B1 (beam-forming), 167-synergy-audio-signal
   (envelope-based onset, AGC), and 005-audio-missing as "T0 unblocks all
   downstream Hilbert-dependent primitives." Cross-link this dive's
   PROGRESS.md line.
8. **Do not wrap SciPy/NumPy at any tier.** Project rule 6: reimplement from
   first principles. SciPy is the cross-language *oracle* for golden files,
   not a runtime dependency.

## Sources

- `C:/limitless/foundation/reality/signal/fft.go:49,101,140,167` — existing
  FFT primitives that T0 composes.
- `C:/limitless/foundation/reality/signal/filter.go:19,54,97,130` — existing
  filter scaffolding; T2/T3 follow the same buffer convention.
- `C:/limitless/foundation/reality/signal/window.go:15,44,76,104` — window
  primitives used by T2 FIR Hilbert.
- `C:/limitless/foundation/reality/signal/signal_test.go` — test pattern to
  mirror (table-driven, golden-file, panic tests).
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/132-signal-missing.md`
  §1.7, §2.6, §2.7 — prior identification of the gap.
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/133-signal-sota.md`
  L22, L398 — SciPy parity audit naming Hilbert.
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/166-synergy-acoustics-signal.md`
  L3, L14, L374 — beam-forming/RIR-envelope dependence on Hilbert.
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/167-synergy-audio-signal.md`
  L227 — audio×signal synergy review naming Hilbert as a signal-side gap.
- `C:/limitless/foundation/reality/audio/onset/spectral_flux.go`,
  `audio/onset/superflux.go`, `audio/onset/energy.go` — current onset
  detectors that would benefit from a time-domain Hilbert envelope detector.
- `C:/limitless/foundation/reality/audio/pitch/yin.go`, `audio/pitch/mpm.go` —
  pitch estimators that would gain from instantaneous-frequency refinement.
- Marple, S. L. (1999). "Computing the discrete-time analytic signal via
  FFT." IEEE Trans. Sig. Proc. 47(9): 2600–2603. (Algorithm for T0.)
- Schüßler, H. W., Steffen, P. (1985). "Halfband filters and Hilbert
  transformers." Circuits, Systems, and Signal Processing 17(2): 137–164.
  (Algorithm for T3.)
- Vaidyanathan, P. P. (1993). "Multirate Systems and Filter Banks."
  Prentice-Hall. §4.6 (half-band filters), Appendix coefficient tables.
  (Algorithm for T4 + T3 design tables.)
- Mitra, S. K. (2001). "Digital Signal Processing: A Computer-Based
  Approach." McGraw-Hill. §10.5 (Hilbert transformer FIR design via
  windowing).
- Sandberg, J. et al. (2010). "Hilbert transform-based amplitude envelope
  for speech analysis." (Reference for envelope-based audio applications.)
- Bello, J.P. et al. (2005). "A Tutorial on Onset Detection in Music
  Signals." IEEE TSAP 13(5): 1035–1047. (Hilbert-envelope onset detection.)
- SciPy `scipy.signal.hilbert` documentation and source — cross-language
  golden-file oracle (literally Marple 1999 implementation).
- Huang, N. E. et al. (1998). "The empirical mode decomposition and the
  Hilbert spectrum for nonlinear and non-stationary time series analysis."
  Proc. R. Soc. London A 454: 903–995. (Hilbert-Huang transform — downstream
  consumer in 132-signal-missing §2.6.)
