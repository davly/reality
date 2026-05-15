# 002 — acoustics: canonical algorithms missing

## Headline
The current `acoustics` package is a 197-line "Acoustics 101" cheat-sheet (8 scalar formulas) — it is missing essentially every algorithm a real acoustics user expects, including the entire impulse-response analysis pipeline (Schroeder/EDC/RT60 from IR, C50/C80, STI), the entire room-simulation pipeline (image-source, ray-tracing, FDTD, FDN reverb), the entire standards-compliant metering pipeline (BS.1770/R128 LUFS, octave/third-octave per IEC 61260, fast/slow SLM integrators, B/C/D/Z weighting), and all spatial/binaural primitives (HRTF, beamforming, DOA).

## What exists today
`acoustics/acoustics.go` (8 functions, all scalar, no vectors, no IR, no spectra):
- `SoundSpeed(γ,R,T,M)`, `SoundIntensity(P,r)` (point source)
- `DecibelSPL(p,pRef)`, `DecibelFromIntensity(I,IRef)`
- `SabineRT60(V,A)` (single coefficient, no Eyring/Millington/Norris-Eyring/Fitzroy/Arau-Puchades variants)
- `DopplerShift(f0,vs,vr,c)` (1-D only, no Doppler factor for arbitrary geometry, no relativistic)
- `ResonantFrequency(L,n,c)` (open-open pipe only — no closed-end, no Helmholtz, no membrane/plate, no rectangular-room mode)
- `AWeighting(f)` (A only — no B, C, D, Z, ITU-R 468, RLB)

No reverb, no IR, no FFT, no filterbank, no buffer/streaming primitives, no array signal processing, no nonlinear, no environmental absorption, no transmission loss.

## Missing canonical algorithms

### Tier 1: high-impact / commonly expected

**Impulse-response analysis (ISO 3382-1/2 — the foundation of measurement acoustics)**
- **Schroeder backward integration** — `EDC(h) = ∫_t^∞ h²(τ)dτ`, the canonical energy-decay curve. Every RT60 measurement built since 1965 starts here. ISO 3382-2 Annex B mandates it.
- **T20 / T30 / EDT** — least-squares fit of the EDC over [-5,-25] dB, [-5,-35] dB, and [0,-10] dB respectively. T20/T30 are *the* reverberation-time standard; Sabine is for design estimation only.
- **C50 / C80 clarity** — `10·log10(∫₀^τ h²/∫_τ^∞ h²)` with τ ∈ {50ms, 80ms}. Speech and music intelligibility metric in ISO 3382-1.
- **D50 definition** — `∫₀^50ms h² / ∫₀^∞ h²`, speech-clarity dual to C50.
- **Center time (Ts)** — `∫t·h²/∫h²`, balance-point IR descriptor (ISO 3382-1).
- **STI / STIPA / RASTI** — modulation transfer function over 7 octave bands × 14 modulation freqs (IEC 60268-16). The standards-mandated speech-intelligibility metric for PA, evac, courtroom, classroom.
- **MTF (modulation transfer function)** from IR — required intermediate for STI and many room-quality metrics.
- **Lateral fraction (LF, LFC), IACC** — spatial-impression metrics (ISO 3382-1) — needs B-format or binaural IR but the math is canonical.
- **G (sound strength)** — `10·log10(∫h²/∫h_ref²)` with reference at 10 m free field.

**Room simulation — there is NOTHING here today**
- **Image-source method (ISM)** for rectangular rooms (Allen-Berkley 1979) — closed-form, the workhorse of every reverb-aware ML training pipeline.
- **General-polyhedral ISM** with visibility checking — needed for non-shoebox rooms.
- **Hybrid ISM + ray tracing** — pyroomacoustics' "RT" mode; ISM for early reflections, stochastic ray tracing for late tail with per-band scattering coefficients.
- **FDTD wave solver** — 1-D, 2-D, and 3-D leapfrog Yee-grid; canonical PDE-accurate room simulator (Botteldooren, Savioja, Kowalczyk).
- **FDN (Feedback Delay Network) reverb** — Jot-Chaigne 1991, Hadamard/Householder mixing matrices; the standard "modeled" reverb.
- **Schroeder reverberator** (1962) — N comb + M allpass; historically and pedagogically essential.
- **Moorer reverberator** (1979) — Schroeder + early-reflection tap delay line + LP-filtered combs.

**Frequency / loudness analysis (the standards stack)**
- **Octave & 1/3-octave filterbank** — IEC 61260-1:2014 base-10 or base-2, Class 0/1/2 tolerance masks; either IIR (Butterworth design) or FFT-bin energy summation. Required for *every* environmental-noise measurement.
- **B, C, D, Z weighting** — IEC 61672-1; same family as the existing `AWeighting`. Z (flat) and C (low-bass) are mandatory for SLMs.
- **ITU-R 468 / CCIR-468** weighting — broadcast noise.
- **RLB / K-weighting** — pre-filter for BS.1770.
- **ITU-R BS.1770-4 loudness** — K-filter (pre-emph + RLB shelving) + 400 ms gated mean → LKFS/LUFS. Every streaming platform requires it.
- **EBU R128 metering** — momentary (400 ms), short-term (3 s), integrated (gated, -10 LU relative gate), Loudness Range (LRA, 10–95 percentile), True Peak (oversampled).
- **True-peak meter** — 4× oversampled inter-sample peak (ITU-R BS.1770-4 Annex 2).
- **SLM integrators** — Fast (125 ms), Slow (1 s), Impulse (35 ms attack / 1.5 s decay) per IEC 61672 — the time-weighting `Leq`, `LAeq,T`, `LAFmax`, `LCpeak`, `LE` (sound exposure), `LAE` (SEL).

**Psychoacoustic scales (used everywhere downstream)**
- **Bark scale** — Zwicker-Terhardt critical bands (24 bands).
- **Mel scale** — Slaney/HTK conventions.
- **ERB / ERB-rate (Cambridge)** — `21.4·log10(0.00437·f+1)` and Glasberg-Moore filter widths.
- **Loudness in sones / phons** — ISO 532-1 (Zwicker) and ISO 532-2 (Moore-Glasberg).
- **Equal-loudness contours** — ISO 226:2003.
- **Masking thresholds** — simultaneous and temporal (Moore-Glasberg, Johnston).

### Tier 2: moderately useful

**Modal / waveguide analysis**
- **Rectangular-room modes** — `f_{lmn} = (c/2)·√((l/Lx)² + (m/Ly)² + (n/Lz)²)`; classify axial/tangential/oblique; mode density and Schroeder frequency `f_s = 2000·√(T60/V)`.
- **Helmholtz resonator** — `f₀ = (c/2π)·√(A/(V·L_eff))` with end correction.
- **Membrane / plate / panel absorber** resonance.
- **Closed-pipe and closed-open pipe** (the existing `ResonantFrequency` only handles open-open).
- **Rayleigh / Stokes attenuation** in air and water.
- **Atmospheric absorption per ISO 9613-1** — frequency-dependent absorption from T, RH, p₀.
- **Outdoor sound propagation per ISO 9613-2** — geometric divergence + atmospheric + ground + barriers.

**Binaural / spatial primitives**
- **HRTF interpolation** — bilinear / spherical-harmonic / VBAP across SOFA-format datasets.
- **ITD / ILD** computation from binaural IR (Woodworth, threshold detection, cross-correlation).
- **HRIR ↔ HRTF** transforms (FFT pair plus minimum-phase reconstruction).
- **VBAP** (vector-based amplitude panning, Pulkki 1997) — N-channel speaker rig panning.
- **Ambisonics encode/decode** — N3D/SN3D normalization, ACN ordering, basic 1st/3rd-order encoding and energy-preserving decoders.
- **B-format → binaural** virtualizer.

**Array / beamforming (DSP-pure, no hardware)**
- **Delay-and-sum beamformer** — frequency- and time-domain.
- **MVDR / Capon** beamformer.
- **LCMV** with linear constraints.
- **GSC (Generalized Sidelobe Canceller)**.
- **MUSIC** DOA — eigen-decomposition of spatial covariance.
- **ESPRIT** DOA.
- **SRP-PHAT** — steered response power with phase transform; mic-array localizer.
- **GCC-PHAT** TDOA estimator.

**Reverb / IR processing**
- **Convolution reverb** — direct, FFT, partitioned-block (Gardner 1995 zero-latency), uniform partitioned (Garcia 2002).
- **Allpass and comb filter** primitives (the building blocks of all algo reverb).
- **Velvet noise** — sparse-signed-impulse late-reverb generator (Karjalainen-Järveläinen).
- **IR truncation, fade-in/fade-out, normalization, denoising via Schroeder pre-mean subtraction** (Lundeby).
- **ESS (exponential sine sweep)** generation + inverse-filter deconvolution (Farina 2000) — the canonical IR-measurement method.

### Tier 3: niche but cool

**Nonlinear acoustics (named in the master-plan line)**
- **Burgers equation** solver (1-D nonlinear, viscous).
- **Westervelt equation** — second-order nonlinear wave; HIFU and parametric-array modeling.
- **Kuznetsov equation** — full second-order nonlinear, includes diffusion.
- **KZK equation** (Khokhlov-Zabolotskaya-Kuznetsov) — parabolic approximation; medical-ultrasound canonical model.
- **Thuras-Jenkins-Olson distortion** — open-air nonlinear distortion of high-SPL tones.

**Underwater / atmospheric**
- **Mackenzie / Chen-Millero / UNESCO** speed-of-sound-in-seawater equations.
- **Thorp / François-Garrison** seawater attenuation.
- **SOFAR / convergence-zone ray tracing** (BELLHOP-style, even just 2-D).

**Aeroacoustics**
- **Lighthill stress tensor** scalar form.
- **Curle / Ffowcs Williams-Hawkings** integrals (analytical kernels).

**Misc canonical**
- **Rayleigh integral** for piston-in-baffle radiation (closed-form on-axis, numerical off-axis).
- **King's integral** for circular piston near-field.
- **Mass law** transmission loss `TL = 20·log10(m·f) − 47` and double-leaf extensions.
- **Sabine vs Eyring vs Millington-Sette vs Fitzroy vs Arau-Puchades** RT formulas (single coefficient generalizations).
- **Bolt-Beranek-Newman room-shape ratio** chart parameters.

## Concrete recommendations

1. **Add `acoustics/ir.go`** — pure-numeric IR analyzers; signature pattern `ImpulseResponseMetrics(h []float64, fs float64) struct{ EDC, T20, T30, EDT, C50, C80, D50, Ts, G float64 }`. Schroeder integration is one loop; T-fit is `linalg.LinearLeastSquares`. Zero new deps.

2. **Add `acoustics/weighting.go`** — generalize the existing IEC-61672 implementation. Sketch:
   ```go
   type Weighting int
   const (WeightZ Weighting = iota; WeightA; WeightB; WeightC; WeightD; WeightITU468)
   func WeightingDB(w Weighting, f float64) float64
   func WeightingFilterBiquads(w Weighting, fs float64) []signal.Biquad // for time-domain
   ```

3. **Add `acoustics/loudness.go` (BS.1770 / R128)** — K-filter biquads, 400 ms gated mean, momentary/short/integrated/LRA/true-peak. Pure float64 streaming pipeline, allocates only on `New`.

4. **Add `acoustics/filterbank.go`** — IEC 61260 octave & 1/3-octave designer (Butterworth biquad cascade) plus FFT-bin energy summation variant. Tabulate the 31 base-10 nominal bands.

5. **Add `acoustics/modes.go`** — rectangular room modes with axial/tangential/oblique classification, Schroeder frequency, mode-density estimate; Helmholtz; closed-pipe.

6. **Add `acoustics/reverb.go`** — Schroeder reverberator, FDN-N (4/8/16) with Hadamard mix, Moorer; allpass + comb building blocks. Output: `Process(in, out []float64)`, no per-sample alloc.

7. **Add `acoustics/conv.go`** — partitioned-block convolution (Gardner zero-latency variant) using the existing `signal` FFT. This unlocks convolution reverb and HRTF rendering.

8. **Add `acoustics/room_sim.go`** — image-source method for shoebox rooms with per-band absorption; `Simulate(room, source, mic, fs, maxOrder) []float64` returning the IR. Defer ray tracing / FDTD to v0.12.

9. **Add `acoustics/scales.go`** — Bark, Mel (HTK + Slaney), ERB, ERB-rate; equal-loudness contours from the ISO 226 table. One-liners that the `audio` package will consume directly for MFCC/Bark filterbank.

10. **Add `acoustics/atmosphere.go`** — ISO 9613-1 atmospheric absorption (function of f, T, RH, p₀) + ISO 9613-2 outdoor propagation (geometric + atmospheric + ground + barriers). Currently no way to compute distance-vs-dB at all in this library.

11. **Add `acoustics/sweep.go`** — Farina exponential sine sweep generator + matched inverse filter; the deconvolution path is one FFT multiply. Pairs with item 7.

12. **Generalize `ResonantFrequency`** into `PipeOpenOpen / PipeClosedOpen / PipeClosedClosed` and add `HelmholtzResonator(V, A_neck, L_neck, c)` with end correction.

13. **Add `Eyring`, `MillingtonSette`, `Fitzroy`, `ArauPuchades`** RT-60 variants alongside `SabineRT60`. One-line additions; cite ISO 3382-1.

14. **Add a basic `Beamform` subpackage**: `DelayAndSum`, `MVDR`, `GCCPhat`, `SRPPhat`. The DSP is FFT + matrix ops — both already live in `signal` + `linalg`.

15. **Tier-3 deferral** — Westervelt/Kuznetsov/KZK PDE solvers belong in a future `nonlinear_acoustics` package once `pde` exists; do not block v0.11 on them. Same for FDTD-3D.

## Sources

Repo files
- `C:\limitless\foundation\reality\acoustics\acoustics.go`
- `C:\limitless\foundation\reality\acoustics\acoustics_test.go`
- `C:\limitless\foundation\reality\acoustics\acoustics_edge_test.go`
- `C:\limitless\foundation\reality\acoustics\testdata\acoustics\` (only `a_weighting`, `decibel_spl`, `doppler`, `sound_speed` golden files)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\001-acoustics-numerics.md` (companion audit)

Web
- pyroomacoustics 0.10 docs (image-source + ray-tracing + DOA + beamforming): https://pyroomacoustics.readthedocs.io/
- python-acoustics `acoustics/room.py` (Schroeder, T20/T30, C50/C80 reference impl): https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/room.py
- AcousPlan — Schroeder integration method explainer (ISO 3382-2 Annex B): https://acousplan.com/learn/what-is-schroeder-method
- ODEON — STI calculation application note (IEC 60268-16): https://odeon.dk/pdf/Application_Note_SpeechTransmissionIndex.pdf
- BrechtDeMan/loudness.py (BS.1770 / R128 reference): https://github.com/BrechtDeMan/loudness.py
- IEC 61672-1:2013 (A/B/C/D/Z weighting + Fast/Slow/Impulse), IEC 61260-1:2014 (octave & fractional-octave filters), ISO 3382-1/2 (room acoustics), ISO 9613-1/2 (atmospheric & outdoor), ISO 226:2003 (equal-loudness), ISO 532-1/2 (loudness in sones)
