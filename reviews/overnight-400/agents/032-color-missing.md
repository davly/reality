# 032 — color: missing color spaces, appearance models, CATs, illuminants, gamuts, utilities

**Agent:** 032 / 400
**Topic:** color-missing — enumerate canonical color spaces and tools NOT yet in `color/`
**Date:** 2026-05-07
**Confirmed by 031:** package contents are `spaces.go` (sRGB↔linear, linear-RGB↔XYZ-D65, XYZ↔Lab, RGB↔HSV — **5 spaces, not the 8 CLAUDE.md claims**), `difference.go` (DE76, DE2000), `adapt.go` (Bradford only), `spectral.go` (Planck blackbody → XYZ via CIE 1931 2° observer, Reinhard tonemap). Confirmed missing: WCAG contrast (no `0.05` constant in package), DE94/DECMC, gamut mapping, Lab↔LCH, CAT02/CAT16, all primaries other than sRGB-D65.

## Verified gap

`color/` exposes **9 conversion functions, 2 difference metrics, 1 chromatic adaptation, 1 spectral integrator, 1 tonemap.** Total surface ≈ 13 callable APIs.

Cross-reference vs `colour-science` (Mansencal et al, ~30 spaces, 11 CATs, 7 CAMs, 14 ΔE formulas, 12 illuminants × 3 observers, ~25 RGB primaries, ~8 gamut-mapping algorithms, OpenColorIO config compatibility), `colorio` (Schlömer), `Colorio.jl`, `OpenColorIO` v2.4, `babelcolor`, the `color-rs` ecosystem, ICC.org spec v4.4, CSS Color Module 4/5 (CR 2024-2025-era status), ITU-R BT.2100-2 (HDR), and Material 3's HCT (Google 2021): **`color/` covers ~7% of the canonical surface.** The seven gap-classes below each carry citation-grounded items.

This agent enumerates the missing surface in three tiers. Tier 1 = textbook canonical or modern-must-ship (CSS-spec-mandated, ICC-mandated, BT.2100-mandated), each ≤120 LOC, golden-file-testable. Tier 2 = high-leverage modern additions and second-rank historical spaces. Tier 3 = research-grade, large coordination scope, or zero-dep-incompatible.

Out of scope (031 owns): gamut mapping (`ClipToSRGB`, `LCHChromaClipToSRGB`, `IsInSRGBGamut`), DE94, DECMC, WCAG 2.x contrast, Lab↔LCH wrappers, CAT02/CAT16 added to existing Bradford framework, golden-file backfill for the 8 already-shipping functions. This agent assumes those land per 031 and adds the *catalog* surface on top.

---

## Tier 1 — must ship (modern canonical, CSS/ICC/BT.2100 mandated, ≤120 LOC each)

These are the spaces, primaries, illuminants, CATs, and ΔE formulas that any 2026-era color library must carry to be credible. Every entry is paper-anchored, golden-file-testable, and zero-dep.

### T1 — Modern perceptual spaces (post-2015)

| # | Item | Reference | Why Tier 1 |
|---|---|---|---|
| T1.S1 | **OKLab** (XYZ-D65 ↔ OKLab) — 3×3 LMS matrix `M1`, cube-root nonlinearity, 3×3 LMS-to-Lab matrix `M2` | Ottosson, B. "A perceptual color space for image processing" (2020), bottosson.github.io | **CSS Color Module 4 mandates `oklab()` and `oklch()`** (W3C CR 2024+). Adopted by Figma, Adobe, Google Material 3 docs, every modern design tool. Numerically trivial — two matrix multiplies + cube root. Must-ship. ~25 LOC. |
| T1.S2 | **OKLch** (polar OKLab) — `C = √(a²+b²); H = atan2(b,a)` | Ottosson 2020 | Polar form needed for CSS-mandated chroma clipping in `oklch()`. Pairs with 031's `LCHChromaClipToSRGB`. ~10 LOC. |
| T1.S3 | **HCT (Hue, Chroma, Tone)** — Material 3's color system: hue+chroma from CAM16, tone from L\* (CIELab). Round-trip via Newton's method on chroma | Google Material 3 (2021), `material-color-utilities` | The dominant *application* color space of 2021-2026 (Material 3 dynamic theming, Android 12+ Monet). Requires CAM16 substrate (T1.S6). ~80 LOC including Newton solver. |
| T1.S4 | **ICtCp** (XYZ ↔ LMS via Hunt-Pointer-Estevez ↔ ICtCp) — PQ or HLG nonlinearity on LMS | ITU-R BT.2100-2 (2018), Dolby (2016) | **The HDR-color-difference space** mandated by BT.2100 for HDR10/HDR10+/Dolby Vision. Used by every HDR-aware video pipeline (Netflix, Apple, Microsoft). ~40 LOC + PQ EOTF (T1.S5). |
| T1.S5 | **PQ (SMPTE ST 2084) and HLG (BT.2100) EOTFs** — perceptual quantizer + hybrid log-gamma | SMPTE ST 2084:2014; ITU-R BT.2100-2 | Substrate for ICtCp and BT.2100 RGB→linear. Both are 1-line (PQ) / 2-branch (HLG) closed-form transfer functions. Must-ship for HDR. ~20 LOC each. |
| T1.S6 | **CAM16** color appearance model (forward + inverse) — chromatic adaptation, Hunt-Pointer-Estevez LMS, post-adaptation nonlinearity, opponent dimensions A/a/b, attributes J/Q/C/M/s/h/H | Li, Li, Wang, Luo, Pointer, Cui, Melgosa, Brill, Pointer (Color Research and Application 2017) | Successor to CIECAM02; **canonical CAM as of 2017**, citation backbone for HCT (T1.S3), ZCAM, all post-2017 ΔE-CAM formulas. The 2017 paper resolved CAM02's H-quadrature singularity. Must-ship. ~150 LOC (it's the largest single primitive in the report). |
| T1.S7 | **CAM16-UCS** — uniform color space derived from CAM16 attributes: J' = (1+100c1)·J/(1+c1·J); a',b' from M and h | Luo-Cui-Li (CRA 2006 for CAM02-UCS), extended to CAM16 in Li et al 2017 | The default ΔE space for CAM16; perceptually uniform for ΔE_CAM16 calculations. ~15 LOC layered on T1.S6. |
| T1.S8 | **JzAzBz** (XYZ-D65 ↔ JzAzBz) — perceptual quantizer + 3×3 matrices + Pq-style nonlinearity | Safdar, Cui, Kim, Luo "Perceptually uniform color space for image signals including high dynamic and wide color gamut" (Optics Express 2017) | The HDR-perceptual space; outperforms ICtCp on hue-uniformity per the original paper. Sister to ICtCp (Tier 1) — ship both. ~30 LOC. |
| T1.S9 | **IPT** (Ebner-Fairchild) — XYZ-D65 → LMS (Hunt-Pointer-Estevez) → power 0.43 → 3×3 → IPT | Ebner-Fairchild "Development and Testing of a Color Space (IPT) with Improved Hue Uniformity" (CIC 1998) | Direct predecessor of ICtCp; the canonical hue-uniform color space pre-2016. Still cited by every CAM/UCS/ΔE paper. ~25 LOC. |
| T1.S10 | **ipt-Dolby** (modified IPT, asymmetric scaling for HDR) | Dolby internal docs, referenced in BT.2100 history | The bridge from IPT to ICtCp; included for reproducibility of pre-BT.2100 HDR research. ~15 LOC layered on T1.S9. |
| T1.S11 | **LMS cone responses** — Hunt-Pointer-Estevez D65, Smith-Pokorny, Stockman-Sharpe (2° and 10°) | Hunt-Pointer-Estevez 1958/1971; Smith-Pokorny 1975; Stockman-Sharpe 2000 | Building block for CAM16, IPT, ICtCp, Bradford. Currently Bradford embeds its own LMS matrix; pulling LMS to a top-level export with named matrices unlocks every modern CAM/CAT. ~30 LOC (matrices + named exports). |
| T1.S12 | **LMS_2deg vs LMS_10deg observers** — Stockman-Sharpe 2000 cone fundamentals at both observer-field-of-view standards | CIE 170-1:2006, CIE 170-2:2015 | The 10° observer is the modern preferred standard for color matching beyond ~4° fields (CIE 170-1:2006 formal recommendation). Currently `cieObserver` in `spectral.go` is 2° only. Add `cieObserver10` table + observer-aware `BlackbodyToXYZ(T, observer)`. ~80 LOC (data tables + dispatch). |

### T1 — Older but canonical spaces

| # | Item | Reference | Why Tier 1 |
|---|---|---|---|
| T1.S13 | **CIE 1976 L\*u\*v\*** (LUV) — XYZ → u',v' chromaticity → L\*u\*v\* | CIE 15:2004 §8.2 | The **other** CIE 1976 perceptual space; sister to Lab (which `color/` already has). LUV is preferred for additive-color (display) work; Lab for subtractive (print). Must-ship as part of the CIE 1976 baseline. ~25 LOC. |
| T1.S14 | **CIE 1976 u'v' chromaticity** (u' = 4X/(X+15Y+3Z), v' = 9Y/...) | CIE 15:2004 §8.2 | Substrate for LUV (T1.S13), CCT (T1.U6), MacAdam ellipses. ~10 LOC. |
| T1.S15 | **CIE 1931 xy chromaticity** (x = X/(X+Y+Z), y = Y/...) | CIE S 014-1/E:2006 | The most-used chromaticity diagram in the world. 5 LOC. Currently absent (`BradfordAdapt` accepts `xy` arguments but there's no exported `XYZToxy` / `xyToXYZ`). |
| T1.S16 | **xyY ↔ XYZ** | CIE S 014-1/E:2006 | The chromaticity-plus-luminance triple; canonical "I have an xy and a Y, give me XYZ" form. 8 LOC. |
| T1.S17 | **HSL** (hue, saturation, lightness) | Joblove-Greenberg "Color Spaces for Computer Graphics" (SIGGRAPH 1978) | HTML/CSS canonical. `color/` has HSV but not HSL. ~25 LOC each direction. |
| T1.S18 | **HWB** (hue, whiteness, blackness) | Smith-Lyons "HWB - A More Intuitive Hue-Based Color Model" (JGT 1996) | **CSS Color Module 4 mandates `hwb()`**. Trivial layered on HSV. ~10 LOC. |
| T1.S19 | **CMYK** (naive 4-channel + per-channel optional under-color removal/black generation) | classical | Not perceptually correct without an ICC profile, but the **subtractive primary representation everyone needs**. Document as "naive non-ICC; for accurate print color use a CMYK ICC profile (out of scope: zero-dep)". ~15 LOC. |
| T1.S20 | **YIQ** (NTSC, FCC 1953) — analog NTSC luma+chroma | FCC 1953; SMPTE 170M | Heritage/legacy NTSC; trivial 3×3. ~10 LOC. |
| T1.S21 | **YUV** (PAL/SECAM analog) | ITU-R BT.470-6 | Analog PAL; trivial 3×3. ~10 LOC. |
| T1.S22 | **YPbPr** (analog component) and **YCbCr** (digital component) — **with selectable BT.601 / BT.709 / BT.2020 / BT.2100 coefficients** | ITU-R BT.601-7, BT.709-6, BT.2020-2, BT.2100-2 | The four canonical Y'CbCr matrix sets. Currently zero. Every video codec, every HDMI sink, every JPEG reader needs BT.601 (SDR-SD) and BT.709 (SDR-HD); HDR needs BT.2020/BT.2100. ~20 LOC + 4 named matrix sets (limited-range vs full-range = ×2 = 8 sets). |
| T1.S23 | **Hunter Lab** (NTSC heritage L = 100√(Y/Yn), a = Ka·(X/Xn − Y/Yn)/√(Y/Yn), b = Kb·(Y/Yn − Z/Zn)/√(Y/Yn)) | Hunter, R.S. (1948) | The pre-CIE-1976 perceptual space; still embedded in many spectrophotometer firmware (X-Rite, HunterLab Inc.). ~15 LOC. |

### T1 — RGB primaries / gamuts

The current `LinearRGBToXYZ` hard-codes the sRGB-D65 matrix in function-body literals. The Tier-1 ask is to refactor primaries to a named-matrix registry so every gamut is one constant away.

| # | Gamut | Reference | Why Tier 1 |
|---|---|---|---|
| T1.G1 | **sRGB-D65** (already implemented as literals) | IEC 61966-2-1 | Refactor to `var SRGBToXYZD65 = [3][3]float64{...}` exported. |
| T1.G2 | **Display P3** (sRGB transfer function on DCI-P3 primaries, D65 white) | Apple TN2257 (2015), CSS Color 4 | **CSS Color Module 4 mandates `display-p3`**. Default wide-gamut on every Apple device since 2015, every Pixel since Pixel 4. Must-ship. ~5 LOC (just a matrix + same sRGB transfer). |
| T1.G3 | **DCI-P3** (theatrical D63 white, gamma 2.6) | SMPTE EG 432-1, SMPTE RP 431-2 | The cinema standard; distinct from Display P3 by white point and TRC. ~5 LOC. |
| T1.G4 | **Adobe RGB (1998)** (gamma ~2.2 on Adobe primaries, D65) | Adobe Systems (1998) | The default print-publishing wide gamut; still the default in Photoshop's "Adobe RGB workspace". ~5 LOC. |
| T1.G5 | **ProPhoto RGB / ROMM-RGB** (gamma 1.8, D50 white, larger-than-visible primaries) | Spaulding-Woolfe-Giorgianni "Reference Input/Output Medium Metric RGB Color Encodings" (PICS 2000); ANSI/I3A IT10.7666:2002 | Photographer's archival space; encloses all visible colors. Must-ship for any RAW-photo pipeline. ~5 LOC. |
| T1.G6 | **Rec. 709** (sRGB primaries, ITU-R 709 transfer = ~γ2.4 with toe) | ITU-R BT.709-6 | HDTV broadcast; transfer differs from sRGB (steeper toe, no straight-line for v ≥ 0.018). ~10 LOC including TRC. |
| T1.G7 | **Rec. 2020** (BT.2020 primaries, BT.2020 transfer) | ITU-R BT.2020-2 | UHDTV / 4K broadcast; the de-facto wide gamut in TV land. ~5 LOC + TRC. |
| T1.G8 | **Rec. 2100** (BT.2020 primaries with PQ or HLG transfer, D65) | ITU-R BT.2100-2 | HDR broadcast / streaming. ~5 LOC layered on T1.S5 (PQ/HLG). |
| T1.G9 | **ACEScg** (AP1 primaries, linear, D60-ish white) | AMPAS S-2014-004; ACES 1.0+ | The CG-rendering working space for film/VFX. ~5 LOC. |
| T1.G10 | **ACES2065-1** (AP0 primaries, linear, D60) | AMPAS S-2008-001 | The ACES interchange / archival space; AP0 encloses all visible colors. ~5 LOC. |
| T1.G11 | **ACEScc** (AP1 primaries, log encoding) | AMPAS S-2014-003 | The ACES grading-log space. ~10 LOC including the log encoding. |
| T1.G12 | **ACEScct** (AP1 primaries, log with toe) | AMPAS S-2016-001 | ACES grading-log with a toe; current ASC default. ~10 LOC. |

### T1 — Standard illuminants and white points

Currently white points are passed as `(x, y)` arguments; there's no named-illuminant registry.

| # | Illuminant set | Reference | Why Tier 1 |
|---|---|---|---|
| T1.I1 | **D-illuminants D50, D55, D65, D75, D93** (XYZ + xy at 2° and 10°) | CIE 15:2004 Tables T.1, T.2 | The complete D-series; each is a named canonical white. Currently zero in code. ~10 LOC for a `map[string][2]float64` chromaticity table + `[3]float64` XYZ. |
| T1.I2 | **CIE A** (incandescent, T ≈ 2856K), **B** (deprecated direct sunlight), **C** (deprecated daylight) | CIE 15:2004 | The pre-D-series CIE illuminants; A is still standard for incandescent emulation. ~5 LOC additive. |
| T1.I3 | **F-series F1-F12** (fluorescent illuminants, three subgroups: standard fluorescent, broadband, narrow-band) | CIE 15:2004 §3.1.4 | Required for accurate office/retail lighting reproduction. ~30 LOC of static data tables. |
| T1.I4 | **LED-B series LED-B1..B5, LED-BH1, LED-RGB1, LED-V1, LED-V2** | CIE 15:2018 §A.2 | The 2018-CIE LED illuminant tables; required for any modern lighting metamerism work. ~25 LOC of static data. |
| T1.I5 | **Equal-energy E** (X=Y=Z=1) | CIE 15:2004 | The mathematical ideal; useful as a sanity check. 1 LOC. |

### T1 — Conversion utilities (web research targets per topic prompt)

| # | Utility | Reference | Why Tier 1 |
|---|---|---|---|
| T1.U1 | **xy ↔ u'v'** (Yxy ↔ YuvP) | CIE 15:2004 §8.2 | Substrate for LUV, CCT, MacAdam. ~10 LOC. |
| T1.U2 | **xy ↔ uv** (CIE 1960 UCS, used by McCamy and Robertson CCT methods) | CIE 1960; deprecated 1976 but **still required for CCT calculations** | One of the two CCT methods uses 1960 uv; document the historical reason. ~5 LOC. |
| T1.U3 | **CCT (correlated color temperature) from xy** — three methods: (a) **McCamy's cubic** (one-line closed form, ±50K accuracy), (b) **Robertson** (linear-interpolation table on 31 isotemperature lines, ±10K accuracy), (c) **Ohno 2014** (parabolic interpolation on planckian locus tabulation, ±0.1K accuracy with optional Δuv) | McCamy (CRA 1992), Robertson (J Opt Soc Am 1968), Ohno (LEUKOS 2014) | Universal need: every camera AWB, every monitor calibration, every astrophotography stack needs xy→CCT. Currently absent despite `BlackbodyToXYZ(T)` being in the package — i.e., the forward direction exists, the inverse doesn't. ~80 LOC for all three. |
| T1.U4 | **Planckian locus** parametric `(x(T), y(T))` from blackbody integration (or Krystek's polynomial approximation `x(T) = polynomial; y(T) = polynomial`) | Krystek "An algorithm to calculate correlated colour temperature" (CRA 1985); CIE 15:2004 | Substrate for T1.U3 Robertson/Ohno methods and for plotting on the chromaticity diagram. ~20 LOC for the polynomial; ~50 LOC for the integration-based version (composes existing `BlackbodyToXYZ`). |
| T1.U5 | **Δuv from CCT** (signed perpendicular distance from Planckian locus in uv-1960) | ANSI C78.377; Ohno 2014 | Indoor-lighting industry standard; quantifies "how green/magenta is this nominally-warm-white LED". ~15 LOC layered on T1.U4. |
| T1.U6 | **Spectral power distribution → XYZ** — generalised `IntegrateSPD(spd []float64, wavelengths []float64, observer Observer) (X, Y, Z float64)` | CIE 15:2004 §7 | Currently `BlackbodyToXYZ` is the *only* SPD→XYZ; abstract the integration loop and let the user pass any SPD (including measured spectroradiometer output, F-illuminants, LED SPDs). ~30 LOC refactor of existing code. |
| T1.U7 | **Reflectance × illuminant → XYZ** — extension of T1.U6 for reflectance integration: `IntegrateReflectance(refl, illuminant, observer)` | CIE 15:2004 §7.1.1 | Required for any printed-color or paint-chip simulation; substrate for metamerism index. ~15 LOC layered on T1.U6. |
| T1.U8 | **WCAG 2.x contrast ratio** `(L1 + 0.05) / (L2 + 0.05)` with relative-luminance from sRGB | W3C WCAG 2.0/2.1/2.2 | 031-F2 (one-liner). Listed here only because it composes into T1.U9 (APCA) below as the "old" contrast formula. |
| T1.U9 | **APCA (WCAG 3 / Accessible Perceptual Contrast Algorithm)** — replaces WCAG 2 with a perceptually-corrected `Lc` value (range −108 to +106) accounting for polarity (light-on-dark vs dark-on-light), font-weight/size lookup tables | Lambert, A. "APCA — Accessible Perceptual Contrast Algorithm" (Myndex 2019-2025); WCAG 3 Working Draft 2024 | The intended successor to WCAG 2.x contrast; already in WCAG 3 working draft. Math is closed-form ~15 LOC; the *lookup tables* for "is Lc 60 readable for 14pt 400-weight" are the bulky part (~30 LOC of tables). Total ~50 LOC. |
| T1.U10 | **ΔE_CMC (l:c)** — pre-CIEDE2000 textile/cosmetics standard with selectable lightness:chroma weighting (1:1 graphic arts, 2:1 acceptability) | Clarke-McDonald-Rigg "Modification to the JPC79 colour-difference formula" (J Soc Dyers Col 1984); ISO 105-J03 | 031-F5; named here for completeness of the ΔE family. ~30 LOC. |
| T1.U11 | **ΔE94** — predecessor to DE2000, still required for certain ISO 12647 print conformance reports | CIE 116-1995 | 031-F5; named here. ~25 LOC. Two parameter sets: graphic arts (kL=1, K1=0.045, K2=0.015), textiles (kL=2, K1=0.048, K2=0.014). |
| T1.U12 | **ΔE2000 inverse / "find Lab2 such that ΔE2000(Lab1, Lab2) = D for fixed direction"** — root-finding via bracket+bisect on the perceptual offset vector | not a single citation; common practice in MacAdam-ellipse simulation | Used to render JND ellipses, perceptual gradients. ~40 LOC. Optional Tier-1 (more often Tier-2). |
| T1.U13 | **ΔE_BFD** — Bradford pre-DE94 formula | Luo-Rigg "BFD(l:c) colour-difference formula" (J Soc Dyers Col 1987) | Historical; still cited in textile literature. ~20 LOC. |
| T1.U14 | **ΔE_ITP** — ICtCp difference: `720 · √((I1−I2)² + 0.25(Ct1−Ct2)² + (Cp1−Cp2)²)` | ITU-R BT.2124 (2019) | The HDR-aware ΔE formula recommended by BT.2124. Trivial once ICtCp (T1.S4) ships. ~5 LOC. |
| T1.U15 | **ΔE_Jz / ΔE_JzAzBz** — Euclidean in JzAzBz | Safdar 2017 | Trivial once JzAzBz (T1.S8) ships. ~5 LOC. |
| T1.U16 | **ΔE_CAM16** — Euclidean in CAM16-UCS | Li et al 2017 | Trivial once CAM16-UCS (T1.S7) ships. ~5 LOC. |

### T1 — Chromatic adaptation transforms (CATs)

031-F3 / F15 already calls for CAT02, CAT16. Adding the rest of the canonical CAT family:

| # | CAT | Reference | Why Tier 1 |
|---|---|---|---|
| T1.C1 | **CAT02** (CIECAM02) — `[0.7328, 0.4296, −0.1624; −0.7036, 1.6975, 0.0061; 0.0030, 0.0136, 0.9834]` | CIE 159:2004 (CIECAM02) | ICC-mandated since 2002. 031-F15 calls for this. ~30 LOC layered on Bradford refactor. |
| T1.C2 | **CAT16** — `[0.401288, 0.650173, −0.051461; −0.250268, 1.204414, 0.045854; −0.002079, 0.048952, 0.953127]` | Li et al 2017 (CAM16) | The companion CAT to CAM16; corrects CAT02's numerical issues at the gamut boundary. ~30 LOC. |
| T1.C3 | **Sharp** CAT — `[1.2694, −0.0988, −0.1706; −0.8364, 1.8006, 0.0357; 0.0297, −0.0315, 1.0018]` | Finlayson-Süsstrunk "Spectral Sharpening and the Bradford Transform" (CIE Expert Symposium 2000) | An alternative to Bradford with better whitepoint adaptation. ~5 LOC additive (just the matrix + inverse). |
| T1.C4 | **von Kries** CAT — diagonal scaling on Hunt-Pointer-Estevez LMS | von Kries 1902; modernised in Süsstrunk-Holm-Finlayson "Chromatic Adaptation Performance of Different RGB Sensors" (IS&T 2001) | The original "scale each cone by white-point ratio" CAT. Educational baseline + still used in some commercial sensors. ~10 LOC. |
| T1.C5 | **XYZ scaling** ("wrong von Kries") — diagonal scaling directly in XYZ, no LMS rotation | Süsstrunk-Holm-Finlayson 2001 | The naive baseline that always fails; ship it as a documented "do not use this in production" reference. ~5 LOC. |

### T1 — RGB transfer functions (TRCs)

| # | TRC | Reference | Why Tier 1 |
|---|---|---|---|
| T1.T1 | **sRGB** (already implemented) | IEC 61966-2-1 | — |
| T1.T2 | **Pure gamma 1.8 (ProPhoto), 2.2 (Adobe), 2.4 (BT.1886/Rec.709 idealised), 2.6 (DCI-P3)** | various | Trivial `pow` wrappers; ship with named exports `Gamma18`, `Gamma22`, `Gamma24`, `Gamma26`. ~10 LOC. |
| T1.T3 | **BT.709 / BT.2020 OETF** (the broadcast-standard piecewise: linear toe + power) | ITU-R BT.709-6 §1.2; BT.2020-2 §3 | Distinct from sRGB; toe slope and exponent differ. ~15 LOC. |
| T1.T4 | **BT.1886 EOTF** (display-side for Rec.709 broadcast) | ITU-R BT.1886 (2011) | Distinct from BT.709 OETF; pure gamma 2.4 with black-level offset. ~10 LOC. |
| T1.T5 | **Log-C, S-Log3, V-Log, RED-Log3G10, ARRI LogC4, Cineon log** (camera log encodings) | Arri, Sony, Panasonic, RED, Kodak datasheets | Required for any RAW/cinema pipeline. Each is closed-form ~5-10 LOC. Ship at least Log-C v3 + S-Log3 + Cineon as the three most-used. |
| T1.T6 | **PQ (ST 2084) inverse EOTF / OETF** | SMPTE ST 2084:2014 | Substrate for PQ (T1.S5) bidirectional. ~10 LOC additive (already counted under T1.S5). |

**Tier-1 total surface:** 23 spaces + 5 chromatic adaptations (CATs) + 12 gamuts/primary sets + 5 illuminant sets (~50 named illuminants) + 16 conversion/utility tools + 6 TRC families ≈ **~67 named items, ~1,400 LOC, ~67 golden-file JSONs.** All citation-grounded against CIE 15:2004, CIE 159:2004, IEC 61966-2-1, ITU-R BT.601/709/2020/2100/2124, SMPTE ST 2084 / EG 432-1, AMPAS S-2008-001 / S-2014-003 / S-2014-004 / S-2016-001, Ottosson 2020, Safdar 2017, Ebner-Fairchild 1998, Li et al 2017, McCamy 1992, Robertson 1968, Ohno 2014, W3C CSS Color Module 4. All single-author overnight. Many entries are 5-10 LOC each (just a matrix or transfer function); the heavy hitters are CAM16 (~150 LOC), HCT-Newton-solve (~80 LOC), 10° observer tables (~80 LOC), CCT three-method (~80 LOC), APCA (~50 LOC).

---

## Tier 2 — should ship (modern, validated, fills a real gap)

| # | Item | Reference | Type | Why Tier 2 |
|---|---|---|---|---|
| T2.S1 | **CAM02 / CIECAM02** (forward + inverse + UCS) | CIE 159:2004; Luo-Cui-Li (CRA 2006) | CAM | The 2002-CIE appearance model; **superseded by CAM16 (T1.S6) but still required** for any consumer following pre-2017 ICC profiles. ~150 LOC; large overlap with T1.S6 substrate. Defer relative to CAM16 since CAM16 is the canonical default in 2026. |
| T2.S2 | **ZCAM** — Safdar 2021 successor to JzAzBz with explicit appearance attributes (Jz, Mz, Hz, Cz, Sz, Vz, Wz, Kz) | Safdar, Hardeberg, Luo "ZCAM, a colour appearance model based on a high dynamic range uniform colour space" (Optics Express 2021) | CAM | The HDR-CAM successor; treats JzAzBz as its UCS substrate. ~120 LOC layered on JzAzBz. |
| T2.S3 | **Hunt model** (1991) | Hunt "Revised colour-appearance model for related and unrelated colours" (CRA 1991) | CAM | The historical predecessor to CAM02; cited by every CAM paper. ~200 LOC; defer unless explicitly needed. |
| T2.S4 | **RLAB / Nayatani / Guth ATD** | Fairchild-Berns 1993; Nayatani et al 1990; Guth 1991 | CAM | The 1990s appearance models; cited but rarely used in production 2026. Defer. |
| T2.S5 | **Munsell renotation** (xyY ↔ Munsell HVC notation, e.g., 5R 4/14) | Newhall-Nickerson-Judd "Final Report of the OSA Subcommittee on the Spacing of the Munsell Colors" (J Opt Soc Am 1943); ASTM D1535 | space | Lookup-table-based (the official renotation is 2,734 sample points + interpolation); ~150 LOC including the table or ~30 LOC if we ship just the chromaticity-to-Munsell-hue analytical approximation. |
| T2.S6 | **NCS (Natural Colour System)** — Hering-opponent perceptual notation (e.g., "S 2030-Y90R") | NCS Colour AB / SS 19102:2004 | space | Swedish/Scandinavian-design industry standard; proprietary specification (NCS Colour AB licenses the conversion data). Document as "out of scope: requires licensed conversion tables." Defer. |
| T2.S7 | **CIELab "improved" variants** — DIN99, DIN99o, DIN99d (extended Lab with improved hue uniformity) | DIN 6176:2001 (DIN99); Cui-Luo-Rigg "DIN99 Colour Difference Formula" (CRA 2002) | space | The German-industry alternative to CIEDE2000; analytically simpler but slightly less accurate. ~30 LOC each variant. Defer unless industrial-print consumer arrives. |
| T2.S8 | **CIELCh_uv** (polar form of LUV) | CIE 15:2004 | space | Sister to LCh_ab (Lab polar). ~10 LOC layered on T1.S13. |
| T2.S9 | **OKHSL, OKHSV** (Ottosson's HSL/HSV variants over OKLab) | Ottosson 2023 follow-up posts | space | Adopted by some design tools as a more-perceptual HSL/HSV. ~40 LOC each layered on OKLab. |
| T2.S10 | **xvYCC** (extended-gamut YCbCr; sYCC) | IEC 61966-2-4 | space | Allows out-of-gamut sRGB representation in YCbCr; required for some camcorders. ~10 LOC layered on T1.S22. |
| T2.S11 | **YCoCg / YCoCg-R** (lossless integer-friendly luma-chroma) | Malvar-Sullivan "YCoCg-R: A Color Space with RGB Reversibility and Low Dynamic Range" (JVT 2003) | space | Used in H.264 lossless mode; lossless integer round-trip. ~10 LOC. |
| T2.S12 | **Oklab→sRGB gamut intersection (Ottosson's algorithm)** — analytical-LCh-clip-to-sRGB-cusp via cubic-equation root-finding | Ottosson 2021 follow-up | gamut | Faster than the iterative LCh-chroma-clip 031-F4 calls for; used by Figma. ~80 LOC. |
| T2.S13 | **Out-of-gamut detection / SGCK (Saturation/Hue-preserving) gamut mapping** | CIE 156:2004 §7 | gamut | The sister algorithm to chroma-clip in CIE 156:2004's gamut-mapping recommendation. ~80 LOC. |
| T2.S14 | **HPMINDE (Hue-Preserving Minimum-ΔE)** gamut mapping | CIE 156:2004 §6 | gamut | The other CIE 156:2004 reference algorithm. ~80 LOC. |
| T2.S15 | **SDR↔HDR tone mapping operators** beyond Reinhard: **Hable Uncharted-2 filmic, ACES RRT/ODT (linear→sRGB), AgX (Troy Sobotka 2022), Lottes 2016, Reinhard-Jodie, Khronos PBR Neutral** | various | tonemap | The 2026 default in Blender/Godot/Pistachio is **AgX**; reality currently ships only Reinhard. ~30 LOC each; ~6 operators × 30 = 180 LOC. |
| T2.S16 | **Inverse tonemapping (SDR→HDR)** — Banterle, Mantiuk, Eilertsen DeepHDR (post-2017) | Banterle-Ledda-Debattista-Chalmers "Inverse Tone Mapping" (CGI 2006) | tonemap | Defer; specialist. |
| T2.S17 | **Spectral upsampling — Smits 1999 / Meng 2015 / Jakob-Hanika 2019** (RGB → reflectance spectrum) | Smits "An RGB-to-Spectrum Conversion for Reflectances" (J Graph Tools 1999); Jakob-Hanika "A Low-Dimensional Function Space for Efficient Spectral Upsampling" (EG 2019) | spectral | Required for spectral renderers (Mitsuba 3, PBRT v4); ~80 LOC for Smits, ~150 LOC for Jakob-Hanika (3-coeff sigmoid moment fit). |
| T2.S18 | **CRI (Color Rendering Index Ra)** and **TM-30-20 (Rf, Rg + 99 CES)** | CIE 13.3:1995; IES TM-30-20 | spectral metric | Indoor-lighting industry standards. ~100 LOC for CRI, ~250 LOC for TM-30. |
| T2.S19 | **MacAdam ellipses (1942 / 1949)** as static lookup table | MacAdam "Visual Sensitivities to Color Differences in Daylight" (J Opt Soc Am 1942) | perceptual | Substrate for any "JND visualisation"; the historical baseline ΔE. ~25 LOC of hardcoded ellipse parameters. |
| T2.S20 | **Whiteness indices**: CIE Whiteness W, Tint Tw, Berger BR, Stensby, Hunter | CIE 15:2004 §9.5 | metric | Textile/paper-industry standards. ~40 LOC. |
| T2.S21 | **Yellowness indices**: ASTM E313 (D65 / C illuminants), DIN 6167 | ASTM E313-20 | metric | Plastics/paint-industry standards. ~20 LOC. |
| T2.S22 | **Color constancy / illuminant estimation**: Grey-World, White-Patch, Shades-of-Grey, Grey-Edge, Grey-Pixel | Buchsbaum 1980 (Grey-World); Land-McCann 1971 (Retinex); van de Weijer-Gevers-Gijsenij 2007 (Grey-Edge) | algorithm | Camera AWB primitives. ~30 LOC each; ~5 algos × 30 = 150 LOC. |
| T2.S23 | **Metamerism index**: CIE special metamerism index (illuminant change) | CIE 15:2004 §10 | metric | "Two materials matching under illuminant A — do they match under illuminant B?" Industry-standard textile QA. ~50 LOC. |

**Tier-2 total:** ~23 items ≈ **~1,800 LOC** (Munsell tables + AgX + spectral upsample dominate). Several need cross-package coordination (`prob` for some statistics, `signal` for spectral upsample basis functions, `optim` for Newton-solves in HCT/inverse-CAM16).

---

## Tier 3 — research-grade, large coordination scope, or zero-dep-incompatible

| # | Item | Reference | Why Tier 3 |
|---|---|---|---|
| T3.1 | **ICC profile parsing** (v2 / v4 / v5) — full lutAtoBType/lutBtoAType/parametricCurveType reader, B2A/A2B forward+inverse, named-color, devicelink, abstract profiles | ICC.1:2010 (v4.3), ICC.1:2022 (v5) | The big one. ICC profiles are ubiquitous (every JPEG/PNG/TIFF/HEIC has one). **Full parsing is ~5,000-8,000 LOC** and **breaks zero-dep** unless we hand-roll a tag-table parser. Recommended path: ship a lookup-table-only profile evaluator (multidimensional interpolation: tetrahedral, prism, trilinear; ~400 LOC) that *consumes* a pre-parsed `[]float64` LUT, and let consumers parse the binary ICC tags via a separate non-`reality` package. This is the only way to honour both CLAUDE.md rule 2 (zero deps) and the standard. |
| T3.2 | **OpenColorIO (OCIO) config compatibility** — color-space transform graphs, role lookups, file-format parsing (.ocio YAML) | OCIO Configuration Working Group (Sony Imageworks 2003+, ASWF 2020+) | Same scope as T3.1. Out of zero-dep scope. |
| T3.3 | **Gamut hull computation** (Mesh of Lab/CAM16-UCS surface for any RGB primary set) | Morovic-Luo "The fundamentals of gamut mapping: A survey" (J Imaging Sci Tech 2001) | Substrate for advanced gamut mapping (T2.S13/S14); ~300 LOC; needs convex-hull (`geometry/convex_hull` exists per CLAUDE.md). Cross-package coordination. |
| T3.4 | **Spectral rendering pipeline** (radiance×reflectance integration with Hero-Wavelength, Wyman-Sloan-Shirley simple analytic CMFs, color-matched importance sampling) | Wilkie-Nawaz-Droske-Weidlich-Hanika "Hero Wavelength Spectral Sampling" (CGF 2014) | Specialist; defer to a future `spectral/` or `render/` package. |
| T3.5 | **Color-difference for materials** (BRDF/SBRDF metrics; ΔE for spatially-varying patterns) | Pellacini-Ferwerda-Greenberg "Toward a Psychophysically-Based Light Reflection Model" (SIGGRAPH 2000) | Specialist. |
| T3.6 | **iCAM06 / iCAM-fr** (image-appearance models with spatial filtering) | Kuang-Johnson-Fairchild "iCAM06: A refined image appearance model for HDR image rendering" (J Vis Comm 2007) | Cross-package: needs `signal/` 2D filter primitives. ~400 LOC. |
| T3.7 | **CRI alternatives**: TM-30 advanced metrics (Rf,h_j, Rcs,h_j color-vector graphics, CES groupings); CQS (Color Quality Scale, Davis-Ohno NIST 2010); FCI (Feeling-of-Contrast Index) | NIST internal reports | Specialist. |
| T3.8 | **Color appearance under chromatic adaptation incomplete** (Hunt-Pointer-Estevez D65→D_observer with adaptation luminance and surround dependence) — what CAM16 does internally, exposed as a primitive | Hunt 1995 | Already inside CAM16; defer surfacing it. |
| T3.9 | **Color vision deficiency simulation** (protanopia, deuteranopia, tritanopia, achromatopsia) — Brettel-Viénot-Mollon 1997 / Machado-Oliveira-Fernandes 2009 / Vienot 1999 | Brettel et al 1997; Machado et al "A Physiologically-based Model for Simulation of Color Vision Deficiency" (TVCG 2009) | Specialist; ~100 LOC each; ~3 algorithms × 3 deficiencies = 9 variants ≈ ~300 LOC. Pinning Tier 3 because consumer interest is concentrated in accessibility tooling (could be Tier 2 if Pistachio/Pulse have an a11y mode). |
| T3.10 | **Chromatic-adaptation-aware ΔE** (CAM16-UCS with explicit viewing conditions: surround, luminance, adaptation) | Li et al 2017 | Specialist; falls out of T1.S6 implementation. |
| T3.11 | **ProPhoto-RGB cone-based gamut compression** (ACES Reference Gamut Compression v1.0 — ASC ACES 2.0 implementation) | ACES "Reference Gamut Compression v1.0" (ASC 2022) | Specialist; cinema-pipeline-only. ~150 LOC. |
| T3.12 | **MICR (Multi-Illumination Color Rendition)** and **TM-30 sub-metrics R_t, R_s, etc.** | IES TM-30-20 Annex E | Specialist. |

---

## Cross-package coordination notes

- **`calculus/` Simpson:** `BlackbodyToXYZ` already wants Simpson per 031-F17. Once `IntegrateSPD` (T1.U6) refactors to a generic SPD integrator, every spectral-integration consumer (T1.U7 reflectance, T1.S22 / T2.S17 spectral upsample, T2.S18 CRI/TM-30) lands on the same Simpson primitive. Coordinate with calculus team to ensure `Simpson(f, a, b, n)` accepts a tabulated-data variant.
- **`linalg/`:** Every CAT (T1.C1-C5), every CAM (T1.S6, T2.S1-S4), every primary-matrix (T1.G1-G12), and every illuminant transform composes 3×3 matrix multiply, inverse, and chain-of-matrices. Currently `color/` open-codes 3×3 multiplies inline (e.g., `LinearRGBToXYZ` lines 65-67). Recommend exporting `var SRGBToXYZD65 [3][3]float64`, `var XYZD65ToSRGB [3][3]float64`, etc, and using a tiny internal `mat3Mul` helper. ~30 LOC refactor unblocks all 67 Tier-1 items.
- **`signal/` FFT:** T2.S17 spectral upsampling (Jakob-Hanika moment fit) and T3.6 iCAM06 spatial filtering need 1D/2D FFT primitives that `signal/` already exposes per CLAUDE.md.
- **`optim/`:** T1.S3 HCT requires a Newton-Raphson solve on chroma; T1.U3 Ohno CCT uses parabolic (Brent-style) interpolation; T1.U12 ΔE2000 inverse uses a bracket+bisect. All compose `optim` primitives that already exist (Newton, Brent — per CLAUDE.md). Coordinate signature.
- **`prob/`:** T2.S22 color-constancy (Grey-World etc.) uses simple statistics (mean, percentile, robust mean) — `prob/` likely already has these.
- **`crypto/`:** No coordination.
- **`graph/`:** T3.3 gamut-hull mesh + T3.6 iCAM06 spatial graph could compose `graph` primitives (BFS/DFS for region-growing on a Lab volume) — defer until T2/T3 lands.
- **`geometry/`:** T3.3 explicitly needs `geometry/convex_hull` (per CLAUDE.md, exists). Also `geometry/SDF` could host gamut-boundary signed-distance for fast in-gamut tests.
- **`testutil/`:** Every Tier-1 item is golden-file-testable. Reference data sources (citation-grounded):
  - **OKLab/OKLch:** Ottosson's reference test vectors at https://bottosson.github.io/posts/oklab/ (the article ships JS reference impl).
  - **CAM16:** Li et al 2017 paper Tables 4-6 contain ~30 tabulated test triples.
  - **JzAzBz:** Safdar 2017 paper Table 4.
  - **ICtCp:** ITU-R BT.2124 Annex 1 test vectors.
  - **CCT:** Robertson 1968 Table I (31 isotemperature lines).
  - **Display P3 / ACES:** AMPAS test images; ACEScct/ACEScg are matrix-defined so any RGB triple is its own check.
  - **Munsell:** Newhall-Nickerson-Judd 1943 renotation file (publicly available).
  - **APCA:** Myndex test suite at https://github.com/Myndex/SAPC-APCA.

---

## Recommended commit ordering (highest-leverage first)

1. **T1.S15 / T1.S16 (xyY ↔ XYZ ↔ xy chromaticity)** — 13 LOC. Blocks every CCT, every Planckian utility, every illuminant lookup. Should land first.
2. **T1.G1 refactor (export named matrices) + T1.G2-G5 (Display P3, DCI-P3, AdobeRGB, ProPhoto)** — ~50 LOC of pure additions. Closes the "no wide-gamut RGB" gap and is a 5-line-per-gamut commit pattern after the refactor.
3. **T1.S1 / T1.S2 (OKLab + OKLch)** — ~35 LOC. **CSS Color 4 mandate**, design-tool default in 2026, must-ship.
4. **T1.S17 / T1.S18 (HSL + HWB)** — ~35 LOC. CSS-mandated, trivial layered on existing HSV.
5. **T1.S13 / T1.S14 (CIE LUV + u'v')** — ~35 LOC. The other CIE 1976 perceptual space; closes the CIE 1976 baseline.
6. **T1.I1-I5 (named illuminants registry)** — ~50 LOC. Static data; unblocks every CAT-aware function.
7. **T1.C1 / T1.C2 / T1.C3 / T1.C4 / T1.C5 (CAT02, CAT16, Sharp, von Kries, XYZ-scaling)** — coordinate with 031-F15 Bradford refactor; ~100 LOC after refactor. Closes the chromatic adaptation surface.
8. **T1.U3 (CCT three methods: McCamy + Robertson + Ohno)** — ~80 LOC. Universal AWB/lighting need. Composes T1.S15.
9. **T1.U6 / T1.U7 (generic SPD→XYZ + reflectance integration)** — ~45 LOC. Refactors `BlackbodyToXYZ` from special-case to general-case; unblocks F-illuminants, LED illuminants, measured spectra.
10. **T1.S22 (YCbCr with BT.601/709/2020/2100 matrices)** — ~30 LOC + named matrix sets. Every video pipeline needs this.
11. **T1.S6 / T1.S7 (CAM16 + CAM16-UCS)** — ~165 LOC. The keystone modern CAM. Blocks T1.S3 (HCT) and the CAM-aware ΔE family (T1.U16).
12. **T1.S3 (HCT)** — ~80 LOC. Material 3 default; Newton-solve on chroma above CAM16.
13. **T1.S4 / T1.S5 (ICtCp + PQ/HLG)** — ~70 LOC. BT.2100 mandate; opens HDR pipeline.
14. **T1.S8 / T1.S9 / T1.S10 (JzAzBz + IPT + ipt-Dolby)** — ~70 LOC. The HDR-perceptual + IPT predecessor surface.
15. **T1.U14 / T1.U15 / T1.U16 (ΔE_ITP + ΔE_Jz + ΔE_CAM16)** — ~15 LOC. Trivial once T1.S4/S8/S6 ship.
16. **T1.U9 (APCA)** — ~50 LOC. WCAG 3 successor.
17. **T1.S19-S23 (CMYK, YIQ, YUV, YPbPr, Hunter Lab)** — ~70 LOC. The legacy completeness surface.
18. **T1.S11 / T1.S12 + T1.G6-G12 (LMS observers + remaining gamuts: Rec.709/2020/2100/ACEScg/ACES2065-1/ACEScc/ACEScct)** — ~150 LOC. Closes the gamut and observer surface.
19. **T1.T2-T6 (TRC family: gamma exponents, BT.709/2020 OETF, BT.1886, log encodings)** — ~80 LOC. Closes the transfer-function surface.
20. **Tier 2 in topic-priority order** — AgX tonemap (T2.S15) is the single highest-impact Tier-2 item (Blender/Godot 2024+ default); spectral upsampling (T2.S17) for any spectral-renderer consumer; CRI/TM-30 (T2.S18) for lighting industry; OKHSL (T2.S9) for design tools.
21. **Tier 3** — opens scope question: does `reality/color` ship an ICC profile evaluator? Recommend yes (LUT-only, ~400 LOC, does not break zero-dep), but explicitly defer the binary-tag parser to a sibling package outside `reality/`.

**Total Tier 1 surface:** ~1,400 LOC + ~67 golden-file JSONs (≥20 vectors each per CLAUDE.md). All citation-grounded against CIE / ITU-R / SMPTE / IEC / AMPAS / W3C / NIST / peer-reviewed primary literature. All backwards-compatible (no existing function changes signature; the `LinearRGBToXYZ` inline matrix can stay or move behind the named export with an aliased call). All language-agnostic for Python/C++/C# port-target validation.

---

## Topic-prompt coverage check

| Topic-prompt item | Tier | Where addressed |
|---|---|---|
| OKLab | 1 | T1.S1 |
| OKLch | 1 | T1.S2 |
| HCT | 1 | T1.S3 |
| ICtCp | 1 | T1.S4 |
| ITP / Lp+a+b (= ICtCp alternate naming) | 1 | T1.S4 (same primitive) + T1.U14 |
| JzAzBz | 1 | T1.S8 |
| CAM16-UCS | 1 | T1.S7 |
| IPT (Ebner-Fairchild 1998) | 1 | T1.S9 |
| ipt-Dolby | 1 | T1.S10 |
| LMS cone responses | 1 | T1.S11 |
| LMS_2deg vs LMS_10deg observers | 1 | T1.S12 |
| CIE 1976 L\*u\*v\* | 1 | T1.S13 |
| CIE 1976 u'v' chromaticity | 1 | T1.S14 |
| CIE 1931 xy chromaticity | 1 | T1.S15 |
| Hunter Lab | 1 | T1.S23 |
| YIQ | 1 | T1.S20 |
| YUV | 1 | T1.S21 |
| YPbPr / YCbCr (BT.601 / BT.709 / BT.2020) | 1 | T1.S22 |
| Munsell renotation | 2 | T2.S5 |
| NCS | 2 | T2.S6 (licensing caveat) |
| CAM02 / CAM02-UCS | 2 | T2.S1 |
| CAM16 / CAM16-UCS | 1 | T1.S6 / T1.S7 |
| ZCAM | 2 | T2.S2 |
| Hunt model | 2 | T2.S3 |
| Bradford CAT | present | (already in `adapt.go`) |
| CAT02 | 1 | T1.C1 |
| CAT16 | 1 | T1.C2 |
| Sharp | 1 | T1.C3 |
| von Kries | 1 | T1.C4 |
| D-illuminants D50/D55/D65/D75/D93 | 1 | T1.I1 |
| CIE A, B, C | 1 | T1.I2 |
| F-illuminants F1-F12 | 1 | T1.I3 |
| LED-B series LED-B1..B5 | 1 | T1.I4 |
| sRGB / AdobeRGB / ProPhotoRGB / DCI-P3 / Display P3 / BT.2020 / ACEScg / ACEScc / ACES2065-1 | 1 | T1.G1-G12 |
| Rec.709 vs Rec.2020 vs Rec.2100 | 1 | T1.G6 / T1.G7 / T1.G8 |
| xy ↔ uv | 1 | T1.U1 / T1.U2 |
| CCT from xy | 1 | T1.U3 (McCamy + Robertson + Ohno) |
| Planckian locus | 1 | T1.U4 |
| Spectral power distribution → XYZ | 1 | T1.U6 |
| Munsell notation lookup | 2 | T2.S5 |
| WCAG 2.x contrast | (031) | 031-F2 — owned by 031, named here as T1.U8 for completeness |
| WCAG 3 / APCA contrast | 1 | T1.U9 |
| ΔE CMC l:c | (031) | 031-F5 — owned by 031, named here as T1.U10 |
| ΔE BFD | 1 | T1.U13 |
| ΔE94 | (031) | 031-F5 — owned by 031, named here as T1.U11 |
| ΔE2000 inverse | 1 | T1.U12 |
| ICC profile parsing | 3 | T3.1 (LUT-only path; binary tag parser deferred outside `reality/`) |

Every named topic-prompt item placed; nothing dropped silently. Items shared with 031 are marked as such and not double-counted in the LOC totals.

---

## Bottom line

`color/` is currently **~7% complete relative to the 2026 canonical color-science surface**. The fix-set is bounded: Tier 1 is ~67 named items / ~1,400 LOC / ~67 golden files, single-author overnight if matrix-export refactor (T1.G1) lands first to unblock the 5-LOC-per-gamut commit pattern. Eight items are CSS Color 4 / BT.2100 / ICC mandates that no credible 2026 color library can ship without (OKLab, OKLch, HSL, HWB, Display P3, CAM16, ICtCp, PQ). Twelve items are 5-line matrix drops (the gamuts and CATs) that effectively cost only the golden-file-vector authoring time. The single largest commit is CAM16 (~150 LOC); the second-largest is the F-illuminants and LED illuminants data tables (~80 LOC of static numbers from CIE 15:2018). After Tier 1, `color/` would be at parity with `colour-science` for everything except ICC profile parsing (T3.1, deferred outside `reality/` for zero-dep compliance) and OCIO config compatibility (T3.2, same).
