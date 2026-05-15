# 340 — dive-room-image-source (Allen-Berkley 1979 / fractional-delay / hybrid Sabine tail / ray-tracing audit)

## Headline
The image-source method is **wholly absent** from `reality/` (zero hits on `ImageSource|Shoebox|RIR|reverberation|fractional` across all `*.go`); the cheapest day-1 PR is `acoustics/room/image_source_shoebox.go` + `acoustics/room/sinc_fractional_delay.go` (~250 LOC) — Allen-Berkley 1979 enumeration over `(2L+1)^3 * 8` mirror sources for a shoebox, deposited via 21-tap Hann-windowed-sinc onto a sample grid, composing the existing `acoustics.SoundSpeed` and `signal.Convolve`; saturates 3/3 R-MUTUAL on (a) order-0 = `d/c` direct path, (b) Sabine ≡ Eyring at low absorption, (c) image-source-shoebox ≡ ray-trace-shoebox at low order.

## Findings

### Existing surface (verified by grep)
- `acoustics/acoustics.go` — 197 LOC, 9 scalar closed-forms only. `SabineRT60` (line 101) is the **only** room-acoustics primitive in the entire repo.
- `acoustics/acoustics.go:29` `SoundSpeed(γ,R,T,M)` — substrate for the `c` constant in `delay = d/c`.
- `signal/filter.go:19` `Convolve(signal, kernel, out)` — direct O(NM) convolution; the right kernel for stamping a windowed-sinc fractional-delay impulse onto an RIR buffer (kernel widths ~21 taps).
- `signal/fft.go:49,101` `FFT/IFFT` — substrate for FFT-convolve once an RIR is built.
- `signal/window.go:15,44,76` `Hann/Hamming/Blackman` — window-source for windowed-sinc fractional-delay.
- `geometry/curves.go:15` 3-D `Vec3` arithmetic via `LinearInterpolate`, but **no `Vec3` struct exposed yet**; image-source loops can use raw `[3]float64` until slot 166 lands its `Vec3` keystone.

### Confirmed gaps (zero hits in production code)
- `ImageSource`, `RoomImpulseResponse`, `RIR`, `ShoeboxRoom`, `Allen.*Berkley` → **0 matches** anywhere in `*.go`.
- `FractionalDelay`, `SincInterp`, `Lagrange`, `Farrow`, `WindowedSinc`, `Sinc(` → **0 matches in production** (only review docs in `reviews/overnight-400/`). Slot 318-dive-resampling already flagged this gap with a 350-LOC plan.
- `EyringRT60`, `Schroeder`, `Millington`, `Hybrid`, `RayTracing`, `BorishPolyhedra` → **0 matches**. Slot 003 and slot 166 already flagged.
- `Cepstrum`, `MinimumPhase`, `Hilbert` → **0 matches** (slot 338 plans Hilbert; slot 339 plans minphase). Out of scope here.

### Numerical-accuracy corner of the algorithm (the assigned focus)
- **Image count grows as `(2L+1)^3 · 8`** for shoebox at order ≤ L (Allen-Berkley 1979 §I, eq. 2). At L=10 this is `9261·8 = 74,088` images; at L=20, `~553,000`; at L=30, `~1.83M`. Each contributes one stamped sinc — direct loop is feasible to L≈30 in single-threaded Go (~ms).
- **Round-off at high order is dominated by distance computation, not reflection-coefficient products**. `d = ||r_image − r_listener||²` involves three squarings of values up to `~2L·max(Lx,Ly,Lz)`; for L=30, room=10 m, `d² ~ 360,000 m²`, then `sqrt`. `float64` mantissa carries 52 bits → relative ε ~2.2e-16 on `d`, so absolute distance error at L=30 is ~1e-10 m, equivalent to ~3e-13 s — **well below sample period at 48 kHz (2.08e-5 s)**. Conclusion: at audio sample rates, image-source positional round-off is negligible up to L≈100. The dominant accuracy issue is **NOT round-off** but **time-quantisation onto the sample grid** (next bullet).
- **Time-quantisation is the real problem.** Each image source's delay `τ_k = d_k/c` is generally non-integer in samples. Naïve "round to nearest sample" introduces broadband error (Peterson 1986 *J. Audio Eng. Soc.* 34(11)) that is audible as a metallic comb-filter colouration. Allen-Berkley's original 1979 paper (eq. 6) addresses this via a 4-tap low-pass FIR; modern implementations use Hann- or Kaiser-windowed-sinc (length 21–41 taps, fractional-delay phase `δ = τ·fs − round(τ·fs)`, evaluate `h_k[n] = window[n] · sinc(n − N/2 − δ)`). pyroomacoustics 0.10.0 uses windowed-sinc with a `sinc_lut_granularity` parameter; that is the canonical implementation.
- **Reflection-coefficient products `Π β_i^{n_i}` underflow to 0 around order 200** for `β=0.7` (`0.7^200 ≈ 1e-31`), so contributions disappear before round-off matters; for `β=0.95` cutoff is order ~1400. Practical truncation rule: stop when `Π β / d < 1e-6` (Allen-Berkley §III).
- **Eyring ≡ Sabine in α→0 limit** (both yield `T60 = 0.161V/A`): `T_Eyring = 0.161V / (-S·ln(1-ᾱ)) → 0.161V/(Sᾱ) = T_Sabine` as `ᾱ→0` since `-ln(1-x) → x`. R-MUTUAL pin (b): the two formulas must agree to 1e-10 at `ᾱ=0.001` (saturated 3/3 with `MillingtonSette` if shipped).
- **Image-source ≡ ray-trace at low order in a shoebox**: launching `M` rays uniformly on the unit sphere from source, each tracing ≤K specular bounces, must converge (in histogram-binned RIR) to the deterministic image-source RIR as `M→∞`. R-MUTUAL pin (c): `||RIR_IS − RIR_RT||_1 < ε(M)` with `ε(M) ~ 1/√M` per central limit; saturates the 3/3.
- **Late-reverberation transition.** Hybrid algorithms (pyroomacoustics, Wayverb, Treble) switch from image-source to a stochastic / Sabine-Eyring tail at the **mixing time** `t_m ≈ √V · 1e-3 s` (Polack 1992) — typically 30–50 ms. Image-source dominates `t < t_m`, exponential-decay statistical tail dominates `t > t_m`. Lehmann-Johansson 2008 (JASA 124(1):269) provides closed-form decay-curve prediction so the tail can be synthesised analytically without running image-source past order ~20.
- **Borish 1984 polyhedra extension.** For non-shoebox geometry, image positions are still well-defined (mirror across each face's plane) but **most image sources are inaudible** — they violate the audibility test (the path from image back to listener must pass through every reflecting wall in order). Borish §III provides an O(N·k) audibility check; without it, naïve enumeration over arbitrary polyhedra is wasted compute. This is the canonical step from "shoebox toy" to "concert hall production".

### Cross-validation pin opportunities (R-MUTUAL 3/3)
| Pin | Path A | Path B | Tolerance |
|---|---|---|---|
| (a) | `ImageSourceRIR` order=0 in shoebox | `δ[round(d/c · fs)] / d` analytic direct path | ≤1 sample bin, amplitude ≤1e-12 |
| (b) | `SabineRT60(V, A)` | `EyringRT60(V, S, ᾱ)` evaluated at `ᾱ=1e-4` | ≤1e-8 absolute (taylor-expansion match) |
| (c) | `ImageSourceRIR(L=2, shoebox)` early-time energy | `RayTraceRIR(M=10⁵ rays, K=2 bounces)` early-time energy histogram | ≤5% in 1 ms bins |
| (d) | `SchroederRT60(ImageSourceRIR(...))` (backward integration) | `SabineRT60(V,A)` of same room | ≤10% (Sabine is approximate; Eyring ≤3%) |
| (e) | `ImageSourceRIR` + Lehmann-Johansson 2008 tail predictor | direct extension of image-source to L=50 | EDC slope match ≤0.5 dB |

Pin (a)+(b) saturate 3/3 by themselves and are ~30 LOC of test code each.

### Cross-package fanout (consumers)
- **Pistachio audio engine**: needs RIR for room-aware playback at 60 FPS. Today: zero path. With `acoustics/room/ImageSource` + `signal.Convolve`, Pistachio can synthesise per-room reverb without leaving reality.
- **AR/VR audio (Howler)**: every spatialisation pipeline needs a deterministic RIR per (source, listener) pose. Image-source is the canonical baseline; HRTF (slot 339) is the perceptual layer on top.
- **ASR robust-to-reverb training**: standard data augmentation convolves clean speech with image-source RIRs at varied `T60` (e.g. Ko-Peddinti-Povey-Khudanpur 2017). Without `acoustics/room`, every consumer reinvents.
- **Architectural acoustics tooling**: Schroeder-integration of an image-source RIR ≡ Sabine RT60 ≡ Eyring RT60 in low-α limit — this trio is the textbook design loop.
- **`audio/spectrogram/`** consumes any RIR via `signal.Convolve` already; no API change needed downstream.

### Recommended placement
A new sub-package **`acoustics/room/`** (the placement already proposed by slot 166-§A0). Keeps the `acoustics/acoustics.go` root file as scalar closed-forms only (architectural cleanliness). One file per algorithm:
- `acoustics/room/types.go` — `Vec3`, `ShoeboxRoom`, `RIR{Sample []float64; SampleRate float64}`.
- `acoustics/room/image_source_shoebox.go` — Allen-Berkley.
- `acoustics/room/sinc_fractional_delay.go` — windowed-sinc deposit.
- `acoustics/room/eyring.go` — `EyringRT60`, `MillingtonSetteRT60`.
- `acoustics/room/schroeder.go` — backward integration to RT60/EDT/C50/C80.
- `acoustics/room/hybrid_tail.go` — Lehmann-Johansson 2008 statistical tail.
- `acoustics/room/raytrace.go` — Krokstad-Strom-Sorsdal 1968.
- `acoustics/room/borish_polyhedron.go` — arbitrary polyhedra with audibility check.

## Concrete recommendations

1. **T0 — `acoustics/room/image_source_shoebox.go` (~150 LOC).** Core Allen-Berkley 1979.

   Signature:
   ```
   ImageSourceShoeboxRIR(
       room ShoeboxRoom, src, lis Vec3, c, fs float64,
       maxOrder int, beta [6]float64, rir []float64,
   )
   ```
   Loop over `(nx, ny, nz)` in `[-L..L]³` and 8 source-mirror parities `(qx, qy, qz) ∈ {0,1}³`; for each, position `r_img = (-1)^q · src + 2n·room`, distance `d = |r_img − lis|`, delay `τ = d/c`, attenuation `att = Π β_i^{|n_i|+|n_i−q_i|} / d`, deposit `att · h_sinc(τ·fs)` into `rir`. Unit: `1/distance`. Truncate when `att < 1e-6`. Sources: Allen-Berkley 1979 eqs. (2),(6); Lehmann-Johansson 2008. Day-1.

2. **T1 — `acoustics/room/sinc_fractional_delay.go` (~80 LOC).** Windowed-sinc fractional-delay deposit.

   `DepositSinc(rir []float64, idx float64, amp float64, halfWidth int, fs float64)` evaluates `amp · hann(n) · sinc(n − halfWidth − δ)` for `n ∈ [0, 2·halfWidth]` and adds to `rir[round(idx) − halfWidth + n]`. `δ = idx − round(idx)`. Composes existing `signal/window.go HannWindow`. **Composes the `Sinc(x)` primitive flagged by slot 318 — the cleanest landing for `Sinc` in the entire repo**. Day-1.

3. **T2 — `acoustics/room/eyring.go` (~60 LOC) + `acoustics/room/schroeder.go` (~60 LOC).** Closes the design loop.

   `EyringRT60(V, S float64, alpha float64) = 0.161 * V / (-S * math.Log(1.0 - alpha))`; for non-uniform `alpha`, use `mean(α_i · S_i / S)` per Eyring 1930. `SchroederRT60(rir []float64, fs float64) float64` does `EDC[k] = Σ_{j≥k} rir[j]²` (reverse cumulative), fits a line to `10·log10(EDC)` in the −5 dB to −35 dB band, returns `60 / |slope|·fs` per Schroeder 1965. **R-MUTUAL pin (d) saturated by composing T0+T2.** Day-1 if T0 lands.

4. **T3 — `acoustics/room/hybrid_tail.go` (~120 LOC).** Lehmann-Johansson 2008 closed-form energy-decay tail. Compute image-source up to mixing time `t_m ≈ √V · 1e-3 s`, then synthesise the diffuse tail as filtered velvet noise (Karjalainen-Järveläinen-Välimäki 2007) with envelope from Lehmann-Johansson eq. (5). Avoids running image-source to order 30+ where compute cost explodes. **R-MUTUAL pin (e) saturated by composing T0+T2+T3.** Day-2.

5. **T4 — `acoustics/room/raytrace.go` (~250 LOC).** Krokstad-Strom-Sorsdal 1968 stochastic ray tracing. Launch M rays from source (uniform on sphere via Marsaglia 1972, or Sobol low-discrepancy via slot 263 if shipped); for each, intersect with room walls, specularly reflect with energy `|·|² · (1−α)`, deposit Dirac at each listener-intersection time. Histograms in 1 ms bins for late tail. **R-MUTUAL pin (c) at low order.** Composes nothing prior; adds a useful primitive for arbitrary geometry. Day-3+.

6. **T5 — `acoustics/room/borish_polyhedron.go` (~250 LOC).** Borish 1984 audibility check + plane-mirror image positions for arbitrary convex polyhedra. Each candidate image must pass: (i) reflection-point lies inside the reflecting face; (ii) every intermediate reflection face must lie between the prior image and the listener. O(N_faces · order) per image. Out-of-scope-of-day-1 but the canonical route to "concert hall" simulation. Day-4+.

7. **T6 — Day-2 cleanup: ambisonic encoding of late reverb (~150 LOC).** Encode the diffuse tail via real spherical harmonics (Daniel 2003 ACN/SN3D normalisation) to N=3 order — produces 16 ambisonic channels per RIR for downstream binaural decoding. Soft-blocked by the `RealSH(l,m,θ,φ)` primitive flagged by slot 003-acoustics-sota and slot 339-§T4. Day-3+ alongside HRTF interpolation.

### Day-1 deliverable
**T0 + T1 + T2 (Eyring) ≈ 290 LOC + 60 golden vectors** (20 per algorithm). Adds the first room-acoustics impulse-response primitive in the entire repo, saturates R-MUTUAL pins (a)+(b)+(d), and exports the missing `Sinc(x)` primitive that simultaneously unblocks slots 312/316/318. This is the single highest-ROI day-1 PR identifiable from this slot.

### Cautions
- **Avoid wrapping pyroomacoustics.** Reality's "reimplement from first principles" rule (CLAUDE.md §6) is non-negotiable. Allen-Berkley 1979 §IV publishes the FORTRAN source; transliteration to idiomatic Go ~150 LOC.
- **Allocation discipline.** RIR is preallocated by caller (`rir []float64`); image-source loop uses zero scratch. Pistachio's 60 FPS rule demands this.
- **Sample-rate parity with `signal.FFT`.** `signal.FFT` is sample-rate-agnostic; `acoustics/room` must carry `sampleRate` explicitly through every signature. Mirror the `audio/spectrogram` convention.
- **Don't overload `acoustics.SabineRT60`.** Keep the existing 1-arg form; add `acoustics/room.EyringRT60` etc. with structured `ShoeboxRoom` input.
- **Numerical truncation rule.** Stop the image-source loop when `Π β / d < 1e-6` (≈ −120 dB) — well below 16-bit noise floor. Never use `maxOrder` alone (it admits inaudible sources at high distance).

## Sources
### Repo files (verified)
- `acoustics/acoustics.go` (197 LOC, line 89-103 `SabineRT60` is the only room primitive)
- `signal/filter.go:19` `Convolve` (RIR convolution substrate)
- `signal/fft.go:49,101` `FFT/IFFT` (FFT-convolve substrate)
- `signal/window.go:15` `HannWindow` (windowed-sinc kernel substrate)
- `reviews/overnight-400/agents/003-acoustics-sota.md` (broader SOTA context)
- `reviews/overnight-400/agents/166-synergy-acoustics-signal.md` (12-primitive `acoustics/room/` plan; this slot deepens the image-source angle)
- `reviews/overnight-400/agents/318-dive-resampling.md` (sinc / fractional-delay gap; T1 here is the smallest landing site)
- `reviews/overnight-400/agents/339-dive-acoustic-hrtf.md` (sister slot; `acoustics/hrtf/` placement precedent)
- `reviews/overnight-400/agents/316-dive-fir-design.md`, `312-dive-bilinear-bias.md` (other consumers of the missing `Sinc` primitive)

### External sources (web-searched)
- Allen, J.B. & Berkley, D.A. (1979). "Image method for efficiently simulating small-room acoustics." *JASA* 65(4):943–950. https://pubs.aip.org/asa/jasa/article/65/4/943/765693 — canonical algorithm with FORTRAN listing.
- Borish, J. (1984). "Extension of the image model to arbitrary polyhedra." *JASA* 75(6):1827–1836. https://pubs.aip.org/asa/jasa/article/75/6/1827/769678 — non-shoebox extension + audibility test.
- Krokstad, A., Strom, S. & Sorsdal, S. (1968). "Calculating the acoustical room response by the use of a ray tracing technique." *J. Sound Vib.* 8(1):118–125 — stochastic ray-tracing baseline.
- Lehmann, E.A. & Johansson, A.M. (2008). "Prediction of energy decay in room impulse responses simulated with an image-source model." *JASA* 124(1):269–277. https://www.semanticscholar.org/paper/70ddba8ba6fb04e754745f0af2531947d2e609bc — closed-form EDC for hybrid tail (T3).
- Lee, H. & Lee, B-H. (1988). "An efficient algorithm for the image model technique." *Applied Acoustics* 24(2):87–115 — order-recursive enumeration that prunes redundant computations.
- Peterson, P.M. (1986). "Simulating the response of multiple microphones to a single acoustic source in a reverberant room." *JASA* 80(5):1527–1529 — fractional-delay LP-FIR for sub-sample image deposit.
- Schroeder, M.R. (1965). "New method of measuring reverberation time." *JASA* 37(3):409–412 — backward-integration EDC → RT60.
- Eyring, C.F. (1930). "Reverberation time in 'dead' rooms." *JASA* 1(2):217–241 — Eyring formula; reduces to Sabine as α→0.
- Polack, J-D. (1992). "Modifying chambers to play billiards: the foundations of reverberation theory." *Acustica* 76 — mixing-time `t_m ≈ √V · 1e-3 s`.
- Smith, J.O. *Physical Audio Signal Processing*. CCRMA online — windowed-sinc fractional-delay derivation.
- Vorländer, M. (2008). *Auralization: Fundamentals of Acoustics, Modelling, Simulation, Algorithms and Acoustic Virtual Reality*. Springer — textbook treatment of image-source + ray-tracing + hybrid.
- pyroomacoustics 0.10.0 (LCAV/EPFL May 2025). https://github.com/LCAV/pyroomacoustics — reference implementation: ISM with `sinc_lut_granularity` LUT, hybrid ISM+RT, Sabine/Eyring/Millington T60 estimators.
- Wayverb (Reuben Thomas). https://reuk.github.io/wayverb/image_source.html — discussion of exponential complexity and audibility check.
- Treble Technologies. https://docs.treble.tech/geometrical-solver/image-source-method — modern production implementation notes.
