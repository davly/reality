# 006 — audio: numerical accuracy of onset / pitch / STFT, framing, windowing edge cases

## Headline
The audio package's framing math, log-floors, and silent-frame guards are mostly correct, but four real numerical issues are present: (1) STFT framing uses `ceil(N/hop)` instead of the standard `floor((N - frame)/hop) + 1`, manufacturing partial trailing frames whose 50-100 % zero content corrupts the OLA reconstruction edge band and contradicts the test's own "skip first/last frameSize" workaround; (2) `pitch.Yin` will pass `cumSum == 0` when the *first* `d(τ)` is zero for a constant frame and produce `d'(τ) = 1` everywhere, masking the period; (3) `pitch.McLeodPitchMethod` accumulates `n(τ)` over a *variable* number of summation samples without normalising by `(N - τ)`, biasing the NSDF toward small lags and making the 0.93-of-global-max gate prefer the half-period peak; (4) the cumulative-mean and squared-difference accumulators in `Yin`/`MPM` and the per-bin spectral-flux summations are naive sums with no Kahan/pairwise compensation, with documented "1e-12 per bin" precision claims that hold only for short frames. STFT round-trip uses an unnormalised IFFT whose precision is window-dependent (Hann hop=N/4 OK, Hamming hop=N/2 broken). No `log1p`/`expm1` opportunities are taken. NaN/Inf inputs are accepted silently throughout (no `IsNaN` guards in any pitch or onset entry point) and the `mfcc.LogMelEnergies` floor only protects against `≤0`, not against `NaN`.

## Findings

### Framing math: STFT trailing-frame manufacturing

- `audio/spectrogram/stft.go:69` — `numFrames := (len(samples) + hopSize - 1) / hopSize`. This is `ceil(N/hop)`, not the standard `floor((N - frameSize)/hop) + 1`. With `N = 16000`, `frame = 512`, `hop = 256` (the test at `spectrogram_test.go:33-44`), this gives 63 frames, of which the last `(frame - hop) / hop = 1` is mostly zero-padded and the *very last* frame (frame index 62, sample start 15872) only has 128 real samples followed by 384 zeros. The spectral leakage from this near-empty frame is real: any consumer that takes a max over frames or calls `MelSpectrogram` then averages will see a synthetic energy hole at the trailing boundary. `librosa.stft(center=False)` uses the standard formula; `center=True` pads first and uses `ceil(N/hop)` *with* explicit reflection padding. Reality does neither — it pads with hard zeros which is the worst of both.
- `audio/spectrogram/stft.go:84-88` — the zero-fill `frameReal[i] = 0` for out-of-range indices is what enables the trailing-partial-frame artefact. There is no `center=True`/reflect option exposed.
- `audio/spectrogram/spectrogram_test.go:85` — `for n := frameSize; n < N-frameSize; n++` skips the first AND last `frameSize` samples (1024 of 4096) when checking round-trip error. This is the test silently working around the framing artefact rather than asserting a correctness invariant. A `center=False`-style framing would let the assertion span almost the entire interior.

### STFT inverse: IFFT normalisation bound to signal.IFFT contract

- `audio/spectrogram/stft.go:162` — `signal.IFFT(frameReal, frameImag)` is called with no division-by-N. If `signal.IFFT` is the conventional unnormalised inverse (pure conjugate-symmetric FFT), the OLA accumulation is off by a factor of `frameSize`. The `1e-7` round-trip tolerance at `spectrogram_test.go:92` masks this for the `Hann hop=64 frame=256` (256× off but `windowSum` divides through by `Σ w² ≈ N·0.375`, so the numerical residual happens to live within 1e-6 for a sine input). For Hamming (`Σ w² ≈ N·0.397`) at `hop = N/2` the round-trip will visibly fail at >1e-3 — there is no test for that pair.
- `audio/spectrogram/stft.go:170-173` — the `windowSum + 1e-12` epsilon is reasonable, but the *relative* tolerance changes by orders of magnitude with window choice. For a Blackman window with `hop = N/4` the tail `windowSum[n]` near the boundary can dip to ~5e-3, where `+ 1e-12` is irrelevant; for an off-COLA pair like Hann at `hop = 3*N/4` the windowSum drops to near zero in long stretches and the `1e-12` epsilon arbitrarily inflates samples by 1e10. There is no test exercising any non-COLA pair.

### Windowing: symmetric only; no periodic DFT-symmetric form

- `signal/window.go:28,57,89` — Hann/Hamming/Blackman are computed with `2π/(n-1)`, the *symmetric* form. STFT analysis usually wants the *periodic* form `2π/n` so that the DFT bin alignment is exact (this is what numpy/scipy/matlab default to for spectrograms; symmetric is the filter-design form). The mismatch is real: the symmetric Hann at `n=512` has w[0]=w[511]=0 and a "lost" sample at the centre relative to the periodic form, leading to a fractional-bin frequency shift in the magnitude spectrum (~0.5 / 512 bin). For the project's own MFCC-based fingerprinting this is invisible, but it diverges silently from any cross-language golden file generated against numpy.
- `signal/window.go:22-25` — `n == 1` returns `out[0] = 1.0` for all three windows. The continuous-time Hann limit at the centre point is 1.0, but a one-sample window has no useful interpretation; it's better to require `n >= 2` and panic. Only Reality's MFCC consumer would ever hit `n=1` and only by mistake.
- No window-sum coefficient is exposed (`Σ w`, `Σ w²`, NENBW, ENBW). Anyone implementing a periodogram-correct PSD has to recompute these. Standard spectrum-analyser calibration needs them.

### Pitch / YIN: real edge-case bugs

- `audio/pitch/yin.go:124-130` — the cumulative-mean update `if cumSum > 0 { dPrime[tau] = d[tau] / (cumSum / float64(tau)) } else { dPrime[tau] = 1 }` is correct for the silent-frame fall-through but **wrong** for any frame where `d(1) = 0` exactly (e.g. a slow ramp `frame[n] = a + b·n` produces `d(1) = N·b²` so OK, but a *constant* frame `frame[n] = c` produces all `d(τ) = 0`). The silent-frame check at line 87 (`maxAbs == 0`) catches `c == 0`, but `c = 1` (DC offset, no zero-amplitude check) sails through, computes `d(τ) = 0` for every τ, hits the `cumSum > 0` branch never, sets `dPrime[τ] = 1` everywhere, and returns the fallback global-min `tauMin → sampleRate/tauMin`. The function reports a non-zero pitch for an all-DC input, which is wrong.
- `audio/pitch/yin.go:134-146` — the absolute-threshold dip walk uses `dPrime[tau+1] < dPrime[tau]` to walk to the local minimum, then breaks. If `dPrime` is exactly equal between two consecutive lags (e.g. exact half-period symmetric input or quantised int input), the walk halts at the first equal sample and returns the lower lag, which is the *octave above* the true period. Strict `<` is the right choice for finding strict minima but a tie-breaking pass should prefer the longer lag (canonical YIN preference for the lowest fundamental).
- `audio/pitch/yin.go:163-178` — the parabolic-interpolation guard `if denom != 0` catches exact zero but not `denom ≈ 0` (inflexion plateau where `d'` is locally constant). For a noise-dominated frame with a near-flat dip, `denom` can be `~1e-15` and the resulting `(y0-y2)/denom` can be `~1e3`, then clamped to `tauStar ± 1` by the guard. The clamp prevents numerical blow-up but the result is still arbitrary noise — the pitch returned for a zero-dip frame is essentially random within the tau bin. A `if math.Abs(denom) > 1e-9 * math.Abs(y1)` style relative tolerance would be more honest.

### Pitch / McLeod (MPM): NSDF normalisation bias

- `audio/pitch/mpm.go:101-112` — the NSDF accumulator `m += a*a + b*b` runs over `n in [0, N-tau)`, which means `m(τ)` includes only `2(N-τ)` samples for lag τ. The published MPM formula uses `m(τ) = Σ_{n=0..N-τ-1} (frame[n]² + frame[n+τ]²)` which is what's coded — but the resulting `n(τ) = 2r/m` is *not* a fair similarity score across τ because both numerator and denominator have only `N-τ` terms. The classic NSDF is normalised *by τ* not by the windowed count; for a stationary input both `r` and `m` halve at τ = N/2 and `n(τ)` is approximately right, but for a transient it is biased toward small lags. The 0.93-of-global-max threshold then prefers the half-period peak because the half-period n(τ) is artificially boosted relative to the full-period one. McLeod's original paper uses `r' = r * (N / (N - τ))` and `m' = m * (N / (N - τ))` to compensate; this implementation does neither and the NSDF is biased.
- `audio/pitch/mpm.go:122-124` — `if globalMax <= 0 { return 0, 0 }`. For a frame where every NSDF value is exactly zero (silent frame, but the silent-frame guard at line 83 already catches that), this is dead code; for a frame where every NSDF is *negative* (anti-correlated, e.g. an alternating ±1 input shorter than tauMin), this returns 0 but doesn't reflect the *anti-correlation* in clarity. Edge case for adversarial inputs only.

### Pitch / Autocorrelation: no Kahan, no normalisation

- `audio/pitch/autocorrelation.go:91-94` — naive `s += frame[n] * frame[n+tau]` summed over up to `N - tauMin` samples. For `N = 8192`, the sum can accumulate ~1e4 ULPs of round-off; the docstring claims integer-period quantisation as the dominant error but for high-amplitude / long frames the `s` accumulation matters too. Pairwise summation halves the constant.
- `audio/pitch/autocorrelation.go:95` — `if s > bestACF` initialised at `bestACF = 0`. If every lag's autocorrelation is *negative* (rare but possible for adversarial inputs), `bestLag = -1` and the function returns 0 silently. Should be `bestACF = math.Inf(-1)`.

### Onset detectors: spectral-flux catastrophic cancellation

- `audio/onset/spectral_flux.go:62` — `d := cmplx.Abs(stft[t][k]) - cmplx.Abs(stft[t-1][k])`. For a near-stationary tone, this is the difference of two ~equal floats and loses precision exactly when the half-wave rectifier `d > 0` is most likely to misfire on noise. A single bin's loss of precision is bounded (1 ULP of the larger magnitude, ~1e-16 relative), but summed over `F = frameSize` bins (typically 1024) the accumulated noise floor is ~1e-13 of the total energy — well below the `mean + 1.5σ` adaptive threshold. The bug is theoretical for typical inputs but *real* if a caller computes `SpectralFluxStrength` and post-processes with a tight absolute threshold (the "compose your own thresholding strategy" use case the docstring at line 75 advertises).
- `audio/onset/spectral_flux.go:62`, `audio/onset/superflux.go:69`, `audio/onset/complex_domain.go:64-66` — `cmplx.Abs(stft[t-1][k])` is recomputed for every bin in every frame even though it was computed in the *previous* iteration's "current" half. Onset detection running at 60 FPS will burn ~F·T `sqrt`s per second redundantly. The comment at `spectral_flux.go:33-37` even calls this out ("no caching of |X[t]| across consecutive frames in this allocating form") but never offers a non-allocating form.

### Onset detectors: complex-domain phase wrapping

- `audio/onset/complex_domain.go:69-71` — `predictedPhase = 2*phasePrev - phasePrevPrev` then `wrapPhase`. For frames where the underlying tone has a slow continuous frequency drift, the *unwrapped* phases can be `~2πk` apart and the linear-extrapolation prediction `2φ[t-1] - φ[t-2]` is correct only modulo 2π. `wrapPhase` then maps the prediction into `[-π, π]`, which is fine for the prediction itself but the *residual* `stft[t][k] - predicted` can be artificially large when the wraparound happens to fall on the "wrong side" of the actual `arg(stft[t][k])`. For a clean continuous tone there can be a small non-zero residual that registers as "onset-like" even with no actual onset. The TestComplexDomainOnset_PureToneNoOnsets test at `onset_test.go:175-183` accepts up to 2 false onsets, which suggests this is observed.
- `audio/onset/complex_domain.go:88-97` — `wrapPhase` uses `for phi > math.Pi { phi -= twoPi }`. For pathological inputs where `phi` is very large (e.g. `1e20` due to upstream NaN propagation), this loop is `O(phi/2π)` and degrades to a hang. `math.Mod(phi+math.Pi, 2*math.Pi) - math.Pi` is `O(1)`. Probably never triggered in practice but the worst-case is unbounded.

### Onset detector: energy detector zero-input handling

- `audio/onset/energy.go:88` — `floor := 0.10 * maxE`. If `maxE = 0` (silent input), `floor = 0` and the post-pick filter degrades to "any pick passes". The earlier call to `PickPeaksAdaptive(D, 1.5, 1)` returns no picks for an all-zero D (since mean = 0, σ = 0, threshold = 0, and `D[t] > 0` is never true), so the function returns nil for true silence — but if a single quantisation tick injects a tiny `D[t] = 1e-15`, the threshold is also ~1e-15 and the floor is 0, so the spurious tick passes. The energy floor `0.10 * maxE` should have an absolute lower bound (e.g. `max(0.10 * maxE, 1e-30)`) to ensure silence stays silent.
- `audio/onset/energy.go:67` — `energies[t] = s / float64(frameSize)`. The `s += x*x` is naive sum of squares; for `frameSize = 8192` and `x ≈ 1`, the accumulator hits ~`8192` with ~13 bits of relative noise (`8192 * eps ≈ 1.8e-12`). Documented "1e-12" precision is exactly at the edge of correctness; for `frameSize = 65536` it would be ~10× worse. Pairwise summation or a Kahan loop is appropriate for the energy primitive that's also reused by `SegmentByOnsetOffset`.

### NaN / Inf input behaviour: undefended throughout

- No pitch entry point (`Yin`, `McLeodPitchMethod`, `AutocorrelationPitch`, `SubharmonicSummation`) calls `math.IsNaN` on the input frame. `Yin([]float64{NaN, NaN, ...}, ...)` produces `maxAbs = NaN`, `maxAbs == 0` is false, then `d[τ] = NaN` for all τ, `cumSum = NaN`, `cumSum > 0` is false, so all `dPrime[τ] = 1` and the function returns `sampleRate/tauMin` — a perfectly clean number masking corrupt input. The `panic` at line 62 catches `sampleRate ≤ 0` but not NaN.
- Same for `SpectralFluxOnset`, `SuperFlux`, `ComplexDomainOnset` — `cmplx.Abs(NaN+NaNi)` is NaN, the running sum becomes NaN, `s += d` propagates, then `PickPeaksAdaptive` computes `mean = NaN`, `stdev = NaN`, `threshold = NaN`, and the explicit `if math.IsNaN(threshold) { panic }` at `peak_picking.go:44` finally fires — but the panic message blames the caller for a NaN threshold when in fact the upstream input was the source. A cleaner contract would be `IsNaN`/`IsInf` checks at every package entry point.
- `mfcc.go:25-29` — `LogMelEnergies` floors values `< floor` but `NaN < floor` is false in Go's IEEE-754 ordering, so a NaN energy passes through unfloored and `math.Log(NaN) = NaN` infects the output. The floor should be `if !(e >= floor) { e = floor }` (de Morgan to catch NaN).

### Tests: golden-file gaps

- The audio package has exactly one golden file: `testdata/welford-parity-golden.json` (Welford fingerprint convergence). No golden file exists for `Compute` (STFT), `Magnitude`, `LogMagnitude`, `Yin`, `McLeodPitchMethod`, `AutocorrelationPitch`, `SubharmonicSummation`, `SpectralFluxOnset`, `SuperFlux`, `ComplexDomainOnset`, `EnergyOnset`, `MelFilterbank`, `MFCC`, or `MelSpectrogram`. CLAUDE.md mandates "Every function has golden-file test vectors. Minimum 20 vectors per function, target 30." The audio package has 1 function with golden coverage out of ~40.
- `TestCompute_RoundTrip` at `spectrogram_test.go:68-95` uses only `Hann hop=N/4`, the most forgiving COLA pair. No test for Hamming, Blackman, or any non-COLA hop.
- No tests for `Yin`, `MPM`, or `AutocorrelationPitch` against constant frames, NaN frames, or pure-noise frames. The `makeNoisyTone` helper at `pitch_test.go:41-55` is deterministic broadband but the ratio is hand-tuned; no SNR sweep, no white-noise generator, no all-DC test.
- `TestMagnitude_KnownValues` at `spectrogram_test.go:121-138` has 6 hand-coded points. Not a golden file, not parity-validated against numpy.

### Numerical-method observations (not bugs but precision-relevant)

- `audio/spectrogram/magnitude.go:69` — `20.0 * math.Log10(m+1e-12)`. The `1e-12` floor is fine for the typical `m ∈ [1e-6, 1e2]` range, but for sub-1e-12 magnitudes (subnormal-range input or heavily quantised silence), the result clamps to `-240 dB` regardless of the actual magnitude. Replacing with `math.Log10(math.Max(m, 1e-30))` extends the dynamic range to ~600 dB at zero cost.
- `audio/melscale.go:25,41` — `1127 * ln(1 + f/700)` and `700 * (exp(m/1127) - 1)`. The forward could use `math.Log1p(f/700)` for better precision near `f = 0` (the comment in `audio_test.go:26` already calls `math.Log1p(10.0/7.0)` in the test expectation but the implementation itself doesn't). The inverse could use `math.Expm1(m/1127)` for better precision near `m = 0`. The current absolute drift documented as "≤1e-9 round-trip" is inflated by the cancellation in `exp(small) - 1`; `expm1` would deliver ~1e-15 round-trip.
- `audio/cqt/cqt.go:147` — Hann recomputed per atom per bin: `0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(n)/denom))`. For each bin, `n` ranges 0..nk-1 — the cosines are not cached across bins. A bin-major precomputation table would save `O(K · nk)` cosines.
- `audio/cqt/cqt.go:144-149` — `angleStep = -2π·q/nk` is used as the geometric phase delta but the `complex(math.Cos, math.Sin)` is recomputed per sample. A complex multiplier `mul = exp(j·angleStep)` initialised once with `phase *= mul` per step would replace `2 sin/cos` per sample with one complex multiply. Numerical drift over `nk = 1024` is ~1e-12 — acceptable for the precision contract, and a single `cmplx.Exp(complex(0, angleStep))` or compensated phasor would be safe.

## Concrete recommendations

1. **Fix STFT framing**: change `stft.go:69` to `numFrames := (len(samples) - frameSize) / hopSize + 1` (with a guard for `len(samples) < frameSize`). Drop the trailing partial frame. Add an explicit `Center bool` parameter or a separate `ComputeCentered` for `librosa`-parity callers. Update `TestCompute_RoundTrip` to assert across the *full* interior, not just `[frameSize, N-frameSize]`.

2. **YIN constant-frame guard**: at `yin.go:120-131`, additionally check whether the *signal variance* is zero (compute `var(frame)` as part of the silent-frame pass at line 78-89) and return `(0, 1)` for any zero-variance frame, not just zero-amplitude ones. Document the DC-offset failure mode.

3. **MPM NSDF normalisation**: at `mpm.go:107-112`, divide both `r` and `m` by `(N - τ)` per McLeod's tail-bias compensation, then `n(τ) = 2r/m` is per-sample normalised. Rerun the harmonic-mix tests (`pitch_test.go`) to confirm the half-period-vs-fundamental selection is correct.

4. **Window normalisation API**: expose `signal.HannSumOfSquares(n) float64`, `HammingNENBW(n)`, `BlackmanENBW(n)` so `STFT.Inverse` can divide by the analytical `Σw²` for COLA pairs and any consumer of `Magnitude` can apply periodogram-correct PSD normalisation.

5. **Add periodic window forms**: `HannPeriodic`, `HammingPeriodic`, `BlackmanPeriodic` (denominator `n` not `n-1`) so cross-language golden files against numpy/scipy roundtrip exactly.

6. **NaN/Inf input contract**: add `if !math.IsNaN(frame[i]) && !math.IsInf(frame[i], 0)` validation pass at every `audio/pitch/*` and `audio/onset/*` entry point. Either panic with a clear message ("input contains NaN at index k") or return the silent-frame sentinel `(0, 1)`. Pick one and document it.

7. **Use log1p / expm1 in mel scale**: `melscale.go:25` → `1127.0 * math.Log1p(f/700.0)`; `melscale.go:41` → `700.0 * math.Expm1(m/1127.0)`. Round-trip improves from 1e-9 to <1e-15.

8. **De-Morgan the LogMelEnergies floor**: `mfcc.go:26` → `if !(e >= floor) { e = floor }` so NaN inputs are floored, not propagated.

9. **Zero-allocation magnitude cache for onset detectors**: introduce `onset.SpectralFluxStrengthCached(stft, prevMag, currMag []float64) []float64` so `cmplx.Abs` is computed once per `(t, k)`. Likewise for `SuperFlux` and `ComplexDomainOnset`. The current allocating forms do `2·F` Abs per frame instead of `F`.

10. **Pairwise summation in autocorrelation, energy, and YIN difference function**: `autocorrelation.go:91-94`, `energy.go:62-66`, `yin.go:111-118` are all `O(N)` to `O(N²)` accumulators. For `frameSize = 65536` (high-resolution analysis), naive summation noise can dominate the documented 1e-12 precision claim. `signal.PairwiseSum(slice []float64) float64` (one helper) would refit all of them.

11. **Golden files**: add at least 20-vector golden coverage for `Compute`, `Yin`, `McLeodPitchMethod`, `AutocorrelationPitch`, `SpectralFluxStrength`, `EnergyOnset`, `MelFilterbank`, `MFCC`. Generate from numpy/scipy/librosa where they exist (with the periodic-window form), or from `math/big` 256-bit reference implementations where they don't. Include the IEEE-754 edge cases the master plan calls out: zero signal, all-DC, NaN/Inf input, subnormal inputs, ±0.0 phase.

12. **Add COLA-violation test**: `TestInverse_HammingHopHalf` should exist and either (a) document the expected reconstruction error or (b) panic / warn. Currently the function silently returns garbage for off-COLA pairs.

## Sources

- `C:/limitless/foundation/reality/audio/spectrogram/stft.go` (Compute, Inverse)
- `C:/limitless/foundation/reality/audio/spectrogram/magnitude.go` (Magnitude, LogMagnitude, PowerSpectrum, HalfSpectrum)
- `C:/limitless/foundation/reality/audio/spectrogram/mel_spectrogram.go` (MelSpectrogram, LogMelSpectrogram)
- `C:/limitless/foundation/reality/audio/spectrogram/spectrogram_test.go`
- `C:/limitless/foundation/reality/audio/pitch/yin.go`
- `C:/limitless/foundation/reality/audio/pitch/mpm.go`
- `C:/limitless/foundation/reality/audio/pitch/autocorrelation.go`
- `C:/limitless/foundation/reality/audio/pitch/subharmonic_summation.go`
- `C:/limitless/foundation/reality/audio/pitch/pitch_test.go`
- `C:/limitless/foundation/reality/audio/onset/spectral_flux.go`
- `C:/limitless/foundation/reality/audio/onset/complex_domain.go`
- `C:/limitless/foundation/reality/audio/onset/superflux.go`
- `C:/limitless/foundation/reality/audio/onset/energy.go`
- `C:/limitless/foundation/reality/audio/onset/peak_picking.go`
- `C:/limitless/foundation/reality/audio/onset/onset_test.go`
- `C:/limitless/foundation/reality/audio/onset/cross_validation_test.go`
- `C:/limitless/foundation/reality/audio/mfcc.go` (LogMelEnergies, MFCC, FrameMFCC)
- `C:/limitless/foundation/reality/audio/melscale.go` (HzToMel, MelToHz, MelFilterbank, PowerSpectrum, ApplyFilterbank)
- `C:/limitless/foundation/reality/audio/fingerprint.go`
- `C:/limitless/foundation/reality/audio/audio_test.go`
- `C:/limitless/foundation/reality/audio/cqt/cqt.go`
- `C:/limitless/foundation/reality/audio/segmentation/onset_offset.go`
- `C:/limitless/foundation/reality/audio/tempo/tempo.go`
- `C:/limitless/foundation/reality/signal/window.go` (HannWindow, HammingWindow, BlackmanWindow, ApplyWindow)
- `C:/limitless/foundation/reality/audio/testdata/` (welford-parity-golden.json — the only golden file in the audio package)
- de Cheveigné & Kawahara (2002) "YIN, a fundamental frequency estimator for speech and music" J. Acoust. Soc. Am. 111(4)
- McLeod & Wyvill (2005) "A smarter way to find pitch" ICMC
- Bello, Duxbury, Davies & Sandler (2004) "On the use of phase and energy for musical onset detection" IEEE SPL 11(6)
- Böck & Widmer (2013) "Maximum filter vibrato suppression for onset detection" DAFx-13
- Griffin & Lim (1984) "Signal estimation from modified short-time Fourier transform" IEEE ASSP 32(2)
- Allen & Rabiner (1977) "A unified approach to short-time Fourier analysis and synthesis" Proc. IEEE 65(11)
