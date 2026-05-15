# 003 — acoustics: state-of-the-art comparison

## Headline
Reality's `acoustics` package (197 LOC of textbook formulas) sits roughly four research generations behind 2025 SOTA — which now centers on differentiable FDN reverberators, neural acoustic fields (NAF/NeRAF), GPU-accelerated FDTD, and SOFA/HRTF-driven binaural rendering — but most of the *math* underneath SOTA (spherical harmonics, FDN matrix design, Schroeder integration, partitioned convolution, BTM diffraction) is plausibly in-scope for a zero-dependency math library.

## Reality's current acoustics scope

Single file `acoustics/acoustics.go` (197 lines, 7 functions):

- `SoundSpeed` — Laplace formula c = √(γRT/M)
- `SoundIntensity` — inverse-square 1/(4πr²)
- `DecibelSPL`, `DecibelFromIntensity` — 20·log₁₀ and 10·log₁₀ conversions
- `SabineRT60` — T60 = 0.161·V/A
- `DopplerShift` — scalar, line-of-flight only
- `ResonantFrequency` — open-open pipe harmonic
- `WaveLength` — λ = c/f
- `AWeighting` — IEC 61672-1 analytic curve

That is it. No impulse responses, no rooms, no spatial audio, no filters, no integration of energy decay, no measurement metrics.

## SOTA in 2025-2026

### pyroomacoustics (LCAV / EPFL) — `0.10.0`, May 2025
The de-facto Python baseline. Current capabilities:

- C++-backed **image source method (ISM)** for arbitrary convex/non-convex polyhedral rooms with frequency-band absorption per surface.
- **Hybrid ISM + ray tracing** simulator: ISM for early reflections, RT for late tail and diffuse scattering. Tracks diffraction implicitly via diffuse rain.
- Per-octave-band absorption + scattering coefficients (Sabine + Eyring + Millington-Sette T60 estimators).
- **Beamforming stack**: Delay-and-sum, MVDR, MUSIC, ESPRIT, SRP-PHAT, generalized cross-correlation (GCC-PHAT) for DOA.
- **STFT-domain BSS**: ILRMA, AuxIVA, FastMNMF.
- TorchAudio's `torchaudio.functional.simulate_rir_ism` is a port of pyroomacoustics' ISM, now a reference Python implementation (see [pytorch/audio#2624](https://github.com/pytorch/audio/issues/2624)).

### libroom / RLR-Audio-Propagation / SoundSpaces 2.0 (Meta)
- **Bidirectional ray tracer** powering Habitat-Sim audio-visual embodied-AI research; supports moving listeners and continuous spatial sampling.
- Frequency-dependent material absorption & scattering, air absorption, binaural HRTF spatialization, reverberation tail synthesis.
- Public reference for "fast geometric acoustics good enough for ML training" ([sound-spaces](https://github.com/facebookresearch/sound-spaces), [arXiv 2206.08312](https://arxiv.org/abs/2206.08312)).

### SOFA / HRTF stack (AES69-2022, SOFA Toolbox 2.2)
- **AES69-2022** standardizes the SOFA NetCDF container for HRTFs, BRIRs, DRIRs, and directional source/receiver responses ([sofaconventions.org](https://www.sofaconventions.org/mediawiki/index.php/SOFA_(Spatially_Oriented_Format_for_Acoustics))).
- 2025 datasets: SS2 (78 humans + 3 mannequins), COAT, Hearpiece, OlHeaD — all use SOFA convention v2.1+.
- **libspatialaudio 0.4** (2025) — 3rd-order ambisonics (16ch) with custom HRTF binauralization, unified renderer for HOA + objects + speaker arrays ([Kempf 2025](https://jbkempf.com/blog/2025/libspatialaudio-0.4/)).
- **SHroom** (2026) — open-source Python lib that projects ISM contributions onto SH basis for composable binaural / spherical-array decoding with real-time head rotation ([arXiv 2603.27342](https://arxiv.org/abs/2603.27342)).
- Magnitude Least Squares (MagLS) and Masked-Magnitude-LS decoding are the new reference for low-order binaural reproduction ([arXiv 2501.18224](https://arxiv.org/html/2501.18224)).

### Differentiable acoustics
- **DiffFDN / Differentiable Feedback Delay Networks** (DAFx 2023, JASA 2024): every delay length, mixing matrix, and attenuation filter is a learnable parameter optimized via SGD on a multi-resolution spectral or perceptually-weighted time-domain loss ([arXiv 2402.11216](https://arxiv.org/html/2402.11216v2), [Springer 2024](https://link.springer.com/article/10.1186/s13636-024-00371-5)).
- **Differentiable Grouped FDN** (Aug 2025) — couples multiple FDNs to model coupled-volume acoustics ([arXiv 2508.06686](https://arxiv.org/html/2508.06686)).
- **Misuka** — differentiable room renderer using Time-Resolved Path Replay Backpropagation (PRB), gradients flow through material props, source/receiver positions, and geometry ([github.com/misuka-renderer](https://github.com/misuka-renderer/misuka)).
- **Differentiable Image-Source Model** (2023) — analytic ∂ISM/∂(absorption, geometry) for room optimization tasks.
- These all compose with PyTorch/JAX autograd and are how 2025 papers fit reverberators to real measurements end-to-end.

### Neural acoustic fields
- **NAF** (NeurIPS 2022) — encode RIR in time-frequency domain, query continuously over (source, listener) positions.
- **NACF** (2023) — multi-scale energy-decay loss, monotonic-attenuation prior; uses temporal correlation module.
- **AV-NeRF** — fuses radiance and acoustic fields jointly.
- **NeRAF** (NeurIPS 2024) — state-of-the-art on SoundSpaces benchmark, +22% C50 / +28% EDT vs NACF, conditions an acoustic NeRF on a 3D voxel grid sampled from a radiance field ([arXiv 2405.18213](https://arxiv.org/abs/2405.18213)).
- **Acoustic Volume Rendering** (NeurIPS 2024) — volume-render acoustic energy along rays in a continuous neural field.
- **Neural Acoustic Multipole Splatting** (2025) — Gaussian-splatting analog for RIR synthesis ([arXiv 2509.17410](https://arxiv.org/html/2509.17410)).
- **Real Acoustic Fields** (Meta) — paired audio-visual room dataset benchmark.

### Real-time auralization
- **Partitioned (uniform/non-uniform) convolution** is the engine: long FIR HRIRs/RIRs split into short blocks, FFT-multiplied, overlap-added; non-uniform partitioning hides early-block latency.
- 2024-2025 push: **GPU-implemented uniform-partitioned convolution** for high-channel-count HOA + binaural with feedback cancellation in live worship-space auralization ([arXiv 2509.04390](https://arxiv.org/html/2509.04390v1)).
- Hybrid VBAP + Ambisonics: VBAP for direct/early, 2nd-order HOA for diffuse tail — 14ch convolution per source.
- **PFFDTD** ([github.com/bsxfun/pffdtd](https://github.com/bsxfun/pffdtd)) — 3D FDTD on CUDA; 1s of 7500 m³ at 1650 Hz in 18 min vs 5h CPU. Real-time auralization feasible up to ~1.5 kHz on commodity GPUs.

### Other ecosystem
- **Treble Tech** — commercial hybrid wave-based + geometric solver (image-source + Discontinuous Galerkin) targeting full-spectrum auralization.
- **soundtools / openAcoustics** — niche; mostly thin wrappers around librosa or pyroomacoustics, no novel math.
- **HARP** (Nov 2024) — large higher-order Ambisonic RIR dataset for ML training ([arXiv 2411.14207](https://arxiv.org/html/2411.14207v2)).

## Math reality is missing for SOTA acoustics

| Math area | Used in |
|---|---|
| Schroeder backward integration | All RT60/EDC measurement (ISO 3382-2 Annex B) |
| Partitioned FFT convolution (uniform & non-uniform) | Every real-time auralization engine |
| Real spherical harmonics (Ambisonics ACN/SN3D, condon-shortley conventions) | libspatialaudio, SHroom, HOA decoding |
| Magnitude Least Squares solve (complex LS w/ magnitude constraints) | MagLS binaural decoding |
| Feedback Delay Network design: mixing-matrix unitarity (Householder, Hadamard, paraunitary), Jot-style absorption-filter design | All FDN reverberators including DiffFDN |
| ISM image-source enumeration over polyhedra + visibility tests | pyroomacoustics, TorchAudio, Treble |
| Bidirectional / Monte-Carlo path tracing of acoustic energy with frequency-band absorption | RLR-Audio, SoundSpaces 2.0 |
| Biot-Tolstoy-Medwin edge diffraction integral; UTD coefficients | Geometric-acoustic engines with diffraction |
| 2nd-/4th-order FDTD stencils for the wave equation, PML / locally-reacting impedance boundaries | PFFDTD, Treble |
| Air absorption per ISO 9613-1 (frequency-, humidity-, temperature-dependent α) | All accurate IR engines |
| B/C/D/Z weighting + IEC 61260 fractional-octave filterbank | Standards-grade SLM, RT60 measurement |
| Bilinear/MZT IIR design, all-pass diffusers, Schroeder/Moorer reverb topology | Classical algorithmic reverb |
| Steered-Response Power, MVDR, GCC-PHAT, MUSIC | DOA estimation, beamforming |
| Vector-base amplitude panning (VBAP), DBAP | Spatial-audio panners |
| Williams/Driscoll-Healy spherical-harmonic transform, SOFA-style interpolation (RBF, barycentric on a unit sphere) | HRTF interpolation between measured directions |

## Concrete recommendations (rank by reality-fit)

Filter: must be pure math, zero-dependency, golden-file testable, no I/O, no audio backend.

1. **Schroeder backward integration → EDC + T20/T30/EDT** (ISO 3382 Annex B). Pure numeric integration over a `[]float64` IR. ~30 LOC. Foundational and widely cited. (Already flagged by 002.)
2. **Real spherical-harmonic basis evaluation** (ACN order, SN3D normalization) up to arbitrary order N. Pure recurrences; the matrix-of-basis-values is the building block for *every* ambisonics encoder/decoder/HRTF interpolator. Numerical gold-file friendly.
3. **Partitioned FFT convolution kernel** (uniform first, optional non-uniform later). Reuses existing `signal/fft`. The math is just block scheduling + overlap-add; zero allocations achievable.
4. **FDN matrix designers**: Householder, Hadamard, and paraunitary mixing-matrix generators + Jot-style attenuation-filter coefficients from a target T60(f) curve. Pure linear algebra; reuses `linalg`. Enables both static reverbs and (optionally) the differentiable variants downstream consumers may want.
5. **IEC 61260 fractional-octave filter design** (1/1, 1/3, 1/6, 1/12 octave). Pole/zero placement only — gives reality the standards-compliant filterbank that 002 noted is missing.
6. **Air absorption coefficient α(f, T, RH, p)** per ISO 9613-1. Closed-form analytic; ~20 LOC, lots of decimal places to validate. Used by every serious RIR engine.
7. **Image-source enumeration for shoebox rooms** (axis-aligned). Closed-form: 8·N³ images, distances, attenuation, optional per-band absorption. Doesn't need polyhedral geometry yet; covers >80% of teaching/benchmarking use.
8. **Magnitude Least Squares solve** (`min ||Ax - b||` with phase free on `b`): a small QR + power-iteration kernel. Right-sized for `linalg`; powers MagLS binaural decoders.
9. **VBAP weights** for arbitrary triangulated speaker meshes. Pure trig; the standard panner everywhere.
10. **B/C/Z weighting curves + standardized SLM time-integrators (LAeq, LAFmax, LAFmin)**. Trivial extensions of existing `AWeighting`.
11. **BTM edge-diffraction integral** (single edge, line-source secondary-source formulation). Costlier; mostly an `integrate-along-edge` quadrature problem — fits `calculus`. Worth it because it differentiates "real" geometric acoustics from "shoebox demo".
12. **Beamforming primitives**: GCC-PHAT, SRP-PHAT, MUSIC. All FFT + linear-algebra; nothing exotic.
13. **Differentiable IR forward models** (image-source w.r.t. absorption coeffs, FDN w.r.t. matrix entries) — only after `autodiff` integration is settled. Aligns reality with the 2024-2026 research direction.

Out of scope (require I/O, neural nets, or graphics): SOFA file parsing (NetCDF I/O), neural acoustic fields (training infra), GPU FDTD (CUDA dep), Habitat/SoundSpaces visual coupling.

## Sources

- pyroomacoustics: <https://github.com/LCAV/pyroomacoustics>, <https://pyroomacoustics.readthedocs.io/>, <https://arxiv.org/abs/1710.04196>
- TorchAudio RIR sim issue: <https://github.com/pytorch/audio/issues/2624>
- SOFA / AES69: <https://www.sofaconventions.org/mediawiki/index.php/SOFA_(Spatially_Oriented_Format_for_Acoustics)>
- libspatialaudio 0.4 release: <https://jbkempf.com/blog/2025/libspatialaudio-0.4/>
- SHroom: <https://arxiv.org/abs/2603.27342>
- MagLS / Masked-Magnitude-LS binaural: <https://arxiv.org/html/2501.18224>
- DiffFDN — Efficient Optimization of FDNs: <https://arxiv.org/html/2402.11216v2>
- Differentiable FDN with learnable delay lines: <https://link.springer.com/article/10.1186/s13636-024-00371-5>, <https://arxiv.org/html/2404.00082>
- Differentiable Grouped FDN (coupled volumes, 2025): <https://arxiv.org/html/2508.06686>
- Misuka differentiable acoustic renderer: <https://github.com/misuka-renderer/misuka>
- NAF (Luo & Du): <https://openreview.net/pdf?id=D21DRzkZbSB>
- NACF: <https://arxiv.org/abs/2309.15977>
- NeRAF (NeurIPS 2024): <https://arxiv.org/abs/2405.18213>, <https://openreview.net/forum?id=njvSBvtiwp>
- Acoustic Volume Rendering (NeurIPS 2024): <https://neurips.cc/virtual/2024/poster/94712>
- Neural Acoustic Multipole Splatting (2025): <https://arxiv.org/html/2509.17410>
- Real Acoustic Fields (Meta): <https://facebookresearch.github.io/real-acoustic-fields/>
- SoundSpaces 2.0: <https://arxiv.org/abs/2206.08312>, <https://github.com/facebookresearch/sound-spaces>
- Partitioned convolution (Wefers thesis): <https://publications.rwth-aachen.de/record/466561/files/466561.pdf>
- GPU partitioned-convolution auralization (2025): <https://arxiv.org/html/2509.04390v1>
- PFFDTD GPU FDTD: <https://github.com/bsxfun/pffdtd>
- Image-source method 2025 review: <https://pubs.aip.org/asa/jasa/article/158/5/R9/3371770/The-image-source-method>
- Hybrid ISM + diffusion + DG (2024): <https://www.sciencedirect.com/science/article/pii/S0003682X24002196>
- ISO 3382 / Schroeder integration: <https://www.iso.org/standard/36201.html>, <https://acousplan.com/learn/what-is-schroeder-method>
- HARP HOA RIR dataset: <https://arxiv.org/html/2411.14207v2>
- Edge-diffraction (BTM) thesis: <https://pixl.cs.princeton.edu/pubs/_2009_AIE/calamia-phd-thesis.pdf>
