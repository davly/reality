# 031 — color: numerical correctness audit

Topic: gamut mapping, CIEDE2000 corner cases, white-point conversion accuracy.
Files reviewed: `color/spaces.go` (221 LOC), `color/difference.go` (137 LOC), `color/adapt.go` (80 LOC), `color/spectral.go` (174 LOC), `color/color_test.go` (545 LOC), `color/testdata/color/{delta_e2000,srgb_linear}.json`.

## Headline

`color/` is a tight, mathematically-faithful implementation of the *core* CIE pipeline (sRGB↔linear, linear-RGB↔XYZ, XYZ↔Lab, ΔE76, ΔE2000, Bradford, RGBToHSV/HSVToRGB, blackbody locus, Reinhard tonemap). Every implemented function is numerically defensible; one external probe (this audit) verified the full Sharma-2005 34-pair table to a max abs error of **4.95e-5** (Sharma's reference ΔE values are themselves only published to 4 decimal places, so this is at the limit of what the reference data can detect — the implementation is correct).

But the package is **catalog-incomplete relative to its docstring promise**. The spaces.go header doc claims `color` provides "color space conversions (sRGB, linear RGB, CIE XYZ, CIELAB, HSV)" and the CLAUDE.md package row promises "8 color spaces, CIEDE2000 perceptual distance, WCAG contrast, Bradford adaptation". What ships is **5 color spaces (sRGB, linear-RGB, XYZ, Lab, HSV)** — no LCH/Lab-polar, no HSL, no CMYK, no Oklab, no XYY. There is **no ΔE94, no ΔECMC, no WCAG contrast (the +0.05 offset never appears in the codebase)**, **no CAT02 / CAT16** (only Bradford), and **no gamut-mapping algorithm at all** (no clip, no SGCK, no HPMINDE, no chroma-clip). The CLAUDE.md row overstates by ~3×.

Within what's implemented, no numerical bugs were found. The risks are all gaps (next section).

## Findings, ranked

### F1 — CLAUDE.md "8 color spaces" claim is false (it's 5). High visibility, low risk.
`spaces.go` provides `SRGBToLinear`, `LinearToSRGB`, `LinearRGBToXYZ`, `XYZToLinearRGB`, `XYZToLab`, `LabToXYZ`, `RGBToHSV`, `HSVToRGB`. That's 5 spaces (sRGB, linear-RGB, XYZ, Lab, HSV). Either implement HSL / LCH / Oklab / xyY to reach 8, or update CLAUDE.md.

### F2 — "WCAG contrast" advertised but absent. Direct functional gap.
`Grep` for `0.05`, `WCAG`, `contrast`, `Contrast`, `Luminance` returns nothing in the package. The WCAG 2.x contrast ratio `(L1+0.05)/(L2+0.05)` with relative-luminance `L = 0.2126·R + 0.7152·G + 0.0722·B` after sRGB→linear is a one-liner using existing `SRGBToLinear` and the sRGB-to-XYZ matrix Y row already present in `LinearRGBToXYZ`. Add `RelativeLuminance(r,g,b)` and `WCAGContrast(rgb1, rgb2)`.

### F3 — "Bradford adaptation" is the only CAT. CAT02 / CAT16 are missing.
`adapt.go` exposes only `BradfordAdapt`. CIECAM02 (CAT02) and CAM16 (CAT16) matrices are public domain and routinely required for ICC v4 / modern color management. Either add `CAT02Adapt` / `CAT16Adapt` or document scope ("Bradford only — CAT02/CAT16 deferred").

### F4 — Gamut mapping is absent. The audit topic literally does not exist in the package.
No `GamutClip`, `GamutMap`, `SGCK`, `HPMINDE`, `ChromaClip`, or even a `IsInGamut` helper. Pistachio — the named 60-FPS consumer in CLAUDE.md — cannot avoid out-of-gamut artifacts using `reality/color`. This is the largest gap relative to scope. Recommendation: add at minimum
- `IsInSRGBGamut(X, Y, Z float64) bool`
- `ClipToSRGB(r, g, b float64) (r2, g2, b2 float64)` — naive [0,1] channel clip
- `LCHChromaClipToSRGB(L, C, H float64) (L2, C2, H2 float64)` — the standard "lower C in LCH until linear-RGB is in gamut" iteration that CSS Color 4 mandates.

### F5 — ΔE94 and ΔECMC are missing. Only ΔE76 and ΔE2000 ship.
The classic Δ-formula ladder is CIE76 → CIE94 → CMC(l:c) → CIEDE2000. Most color-difference research and printing/textile QA still uses ΔE94 / ΔECMC because ΔE2000 is too gnarly to invert analytically. Add `DeltaE94(graphics|textiles)` and `DeltaECMC(l, c)`.

### F6 — Lab↔LCH conversions are not exposed.
`color/` has Lab and ΔE2000 (which internally computes hp on a' = a·(1+G), not on a) but no public `LabToLCH(L, a, b) (L, C, H)` / `LCHToLab(L, C, H) (L, a, b)`. Without LCH the user cannot do gamut mapping, hue rotation, or chroma scaling. Two trivial wrappers; ~12 lines.

### F7 — Bradford inverse matrix is hard-coded numerics, not derived.
`adapt.go` lines 75–77 use a hard-coded inverse (`0.9869929, -0.1470543, 0.1599627` …). The forward matrix at lines 29–31 is also literal. Both came from Lindbloom and are correct *to the precision shown* (7 digits), but `M⁻¹·M` will differ from `I` by ~3e-7 rather than at the f64 level. For a "256-bit big-rational reference" library that's loose. Either:
- Compute the inverse from the forward matrix at package-init via Gauss-Jordan (cost: 18 mults at startup, gives full-f64 precision), or
- Promote both matrices to higher-precision rationals at f64 export (e.g., 16-digit literals from `math/big`).
The current 7-digit truncation is the dominant error source in Bradford and visibly contaminates the `BradfordAdapt_D65toD50` test's tolerance: it's set to `1e-3`, ~1000× looser than the package's typical `1e-10` ambition.

### F8 — sRGB threshold is the published-rounded value, not the algebraically-continuous one.
`SRGBToLinear` switches branches at `srgb <= 0.04045`. At exactly 0.04045 the two branches differ by **2.33e-9** (probe: 0.00313080495356037 vs 0.00313080728306768). IEC 61966-2-1 publishes 0.04045 / 0.0031308 as standard rounded values; the strictly-continuous threshold is closer to 0.040449936... — these can never be exactly aligned in f64 because the exponent 2.4 is not representable as a finite ratio. Document: "sRGB transfer function has a 2.3e-9 discontinuity at the IEC-published threshold; this is intrinsic to the standard's rounded constants, not an implementation choice." This is not a bug — every conformant sRGB implementation has the same gap — but the package's "Precision: exact to float64 precision" claim at line 31 is wrong for inputs near the threshold.

### F9 — `LinearToSRGB` threshold has a similar 2.85e-8 discontinuity.
At `linear == 0.0031308`: `linear*12.92 = 0.04044993599…` vs `1.055·pow(linear, 1/2.4) − 0.055 = 0.04044990748…`. Same root cause as F8. Same docstring fix.

### F10 — `XYZToLab` has no Y < 0 handling; negative XYZ silently propagates through linear branch.
`labF` switches on `t > (6/29)^3`. For negative `t` (only possible if user passes negative XYZ — e.g., out-of-gamut after an inverse matrix), the linear branch evaluates `t/(3·(6/29)²) + 4/29`, which extrapolates linearly through the origin. This is not a CIE-prescribed behavior — CIE 15:2004 declares Lab undefined for `Y < 0`. Either:
- Add an explicit `if t < 0 { return math.NaN() }` (preferred — fail loudly), or
- Document that negative XYZ inputs use linear extrapolation (current de facto behavior).

### F11 — `hueAngle(0, 0)` returns 0; correct, but signed-zero is unhandled.
`difference.go:118` checks `ap == 0 && b == 0`. This collapses `+0` and `−0` to the same answer (verified: `(0, +0)` vs `(0, −0)` gives ΔE2000 = 0). Good. But `atan2(±0, ±0)` would otherwise give `±0` or `±π`; the explicit guard correctly defends against `atan2(+0, −0) = π` poisoning the ΔE2000 result for an achromatic input. Keep as-is.

### F12 — `RGBToHSV(0, 0, 0)` saturation: `delta = maxC = 0` → `s = delta / maxC` is `0/0 = NaN`.
`spaces.go:166` returns early on `maxC == 0`, so the bug is dodged. But the comment doesn't note this. Verified: `RGBToHSV(0, 0, 0)` returns `(0, 0, 0)` — correct, defended by the early return. No change needed; documentation could note the early-return is load-bearing.

### F13 — `HSVToRGB` undefined behavior at h ≥ 360.
`spaces.go:198` computes `hPrime := h / 60` then a switch chain `hPrime < 1 / < 2 / < 3 / < 4 / < 5 / default`. For `h == 360`, `hPrime == 6`, the `default` branch runs, returning `r,g,b = c,0,x` — i.e., it treats h=360 as h=300, not h=0. The doc says "h in [0, 360)" but does not validate. Suggest:
```go
hPrime := math.Mod(h, 360) / 60
```
or, at API level, document that h must be wrapped by the caller.

### F14 — ΔE2000 `0/0` in `s_C * s_H` for two zero-chroma inputs is dodged but not asserted.
If both inputs are achromatic, `cp1 = cp2 = 0`, so `cpAvg = 0`, so `sc = 1`, `sh = 1` (since `t` is bounded), `dCp = 0`, `dHp = 0` (the `cp1*cp2 != 0` guard suppresses the hue-difference computation), so result = `|dLp|/sl`. Verified `DeltaE2000(50,0,0,50,0,0) = 0`. Fine. But there is no test for the **doubly-achromatic, different-L** case — `DeltaE2000(0,0,0,100,0,0)` should equal exactly `100 / sl(50)`. Add it as a regression vector.

### F15 — Bradford is not parametrized over the cone-response matrix.
`BradfordAdapt` hard-bakes the Lam 1985 Bradford matrix. To add CAT02/CAT16 the function must be either duplicated (bloat) or refactored to take a `coneResp [3][3]float64`. The latter unlocks all three popular CATs from one core. Recommended refactor:
```go
func ChromaticAdapt(M [3][3]float64, MInv [3][3]float64, X, Y, Z, srcWPx, srcWPy, dstWPx, dstWPy float64) (Xa, Ya, Za float64)
var BradfordCAT = [3][3]float64{...}; var BradfordCATInv = [3][3]float64{...}
var CAT02 = ...; var CAT02Inv = ...
var CAT16 = ...; var CAT16Inv = ...
func BradfordAdapt(...) { ChromaticAdapt(BradfordCAT, BradfordCATInv, ...) }
```

### F16 — Golden-file coverage is 2 functions out of ~10 (DeltaE2000 + SRGBToLinear).
CLAUDE.md says "every function has golden-file test vectors." `color/testdata/color/` has files for two functions only. Missing goldens:
- `LinearToSRGB` (trivial: invert sRGB-linear vectors)
- `LinearRGBToXYZ` / `XYZToLinearRGB` (D65 white, primaries, secondaries)
- `XYZToLab` / `LabToXYZ`
- `RGBToHSV` / `HSVToRGB`
- `BradfordAdapt` (D65↔D50, A↔D65, identity)
- `BlackbodyToXYZ` (CCT samples 1500K, 3000K, 5500K, 6500K, 10000K against published xy-locus)
- `DeltaE76`
This is the biggest "claim vs reality" gap in the whole package. Per CLAUDE.md goal of cross-language validation, the absent goldens mean Python/C++/C# ports can validate only 2 of 10 functions.

### F17 — `BlackbodyToXYZ` integration is rectangle rule (Riemann sum), not Simpson.
`spectral.go:37–53` accumulates `sumX += planck * xBar` over a 5-nm grid with no weighting (i.e., midpoint-rectangle, not even trapezoidal). For smooth integrands like `B(λ,T)·x̄(λ)` this is O(h²) accurate (~1e-3 at 5-nm step) — fine for chromaticity coordinates to 3 sig figs but *not* "limited by 5nm step and tabulated data" as the docstring claims. Switching to Simpson 1/3 (since 81 samples = 80 intervals = even) is one extra loop, costs nothing, and tightens accuracy to O(h⁴) ~1e-6. Worth doing if `calculus/` Simpson primitive can be reused.

### F18 — `BlackbodyToXYZ` overflow guard `exp > 709` is the f64 `exp` overflow but **not** the `(λ⁵·(exp(x)-1))` overflow.
At very low T (≪ 100K), `λ⁵·(exp(hc/λkT)−1)` overflows to `+Inf` even when `exp` itself is finite (since `λ⁵` is `O(1e-34)` and `exp` can be `O(1e300)` before its own overflow). Verified by inspection: the guard is on the exponent only, not the product. Real risk is small (caller-induced via T < 100K) but the docstring promises "T > 0; practical range ~1000K to ~25000K" — a sub-1000K caller can produce a NaN or 0 silently. Add a `if T < 100 { return 0,0,0 }` floor or at least document.

### F19 — Test tolerances are inconsistent and frequently 1e3× looser than the package's stated 1e-10 ambition.
- `TestSRGBToLinear_Half`: tol 1e-6 (vs achievable 1e-15)
- `TestLinearRGBToXYZ_White`: tol 1e-3
- `TestBradfordAdapt_*`: tol 1e-3 (justified by F7 above — the matrix isn't more accurate)
- `TestDeltaE2000_SharmaPair*`: tol 1e-4 (justified — Sharma's published values are 4 decimals)
- `TestDeltaE2000_Symmetric`: tol 1e-4 (should be 0 — DE2000 is exactly symmetric, verified above)
The `TestDeltaE2000_Symmetric` looseness is wrong. Tighten to 0 (i.e., `if d1 != d2`).

### F20 — IEEE-754 NaN/Inf inputs are propagated correctly to ΔE2000 (verified) but **not tested**.
Probe shows `DeltaE2000(NaN, 0, 0, 50, 0, 0) = NaN` and `DeltaE2000(+Inf, 0, 0, 50, 0, 0) = NaN`. Good — but no test asserts this. The CLAUDE.md design rule "IEEE 754 edge cases mandatory: +Inf, -Inf, NaN, -0.0, subnormals" is unmet for this package. Add a `TestDeltaE2000_NaN` and `TestDeltaE2000_Inf`.

### F21 — Hue rotation rt is correct at hpAvg = 275 (peak), but no test asserts the rotation actually engages.
The `R_T = -sin(2·Δθ)·R_C` term peaks at the blue region (hpAvg ≈ 275). Sharma pair 17 (50,2.5,0 vs 73,25,−18) gives hpAvg in the rotation zone, but the test (`TestDeltaE2000_SharmaPair17`) is for **pair 18** in the published table, not 17 (the test name is wrong — the input matches Sharma's pair 18). Rename or document.

### F22 — ΔE2000 has no early-out for identical inputs.
For `DeltaE2000(L, a, b, L, a, b)` the function still runs all the trig and produces a result that **rounds** to 0 but is not exactly 0 in general. Verified: `DeltaE2000(50, 20, -10, 50, 20, -10) = 0` (lucky), but `DeltaE2000(50, 2.6772, -79.7751, 50, 2.6772, -79.7751)` should also be exactly 0 — adding `if a1 == a2 && b1 == b2 && L1 == L2 { return 0 }` would saturate this.

## Numerical pins

- **DE2000 max error vs Sharma 2005 over all 34 pairs: 4.95e-5** (probe in this audit). At Sharma's 4-decimal precision, this saturates the reference. Implementation is correct.
- **sRGB transfer threshold discontinuity: 2.33e-9** at `srgb=0.04045`; **2.85e-8** at `linear=0.0031308`. Intrinsic to IEC's rounded standard constants.
- **Bradford forward·inverse mismatch: ~3e-7** from 7-digit literal truncation. Currently the dominant error source in `BradfordAdapt`.
- **Lab f() continuity at (6/29)³: exact** (probe: diff = 0.0).
- **LabFInv continuity at 6/29: 2.55e-16** (one ULP).
- **ΔE2000 doubly-achromatic, signed-zero, hue-wraparound: all return 0** as expected.

## Recommendations, prioritized

1. **Add gamut-mapping API** (F4). Highest-impact gap. At least `ClipToSRGB` and `LCHChromaClipToSRGB`.
2. **Add ΔE94, ΔECMC, WCAG contrast** (F2, F5). All <30 lines each; close the docstring/CLAUDE.md gap.
3. **Add Lab↔LCH** (F6). Required by F1/F4.
4. **Refactor Bradford → generic ChromaticAdapt over CAT matrix** (F15). Add CAT02 and CAT16.
5. **Fill golden-file vectors for 8 missing functions** (F16). This is the load-bearing CLAUDE.md promise.
6. **Tighten Bradford matrix precision** (F7) — derive inverse at init from 16-digit forward matrix.
7. **Tests**: add NaN/Inf/signed-zero vectors (F20), tighten ΔE2000 symmetry tolerance to exact (F19), add doubly-achromatic-different-L test (F14), rename `SharmaPair17` → `SharmaPair18` (F21).
8. **Docstrings**: caveat the "exact to f64" claims for sRGB transfer (F8/F9), declare scope of valid inputs to `XYZToLab` (F10), declare `BlackbodyToXYZ` low-T floor (F18).
9. **Switch `BlackbodyToXYZ` to Simpson** if `calculus/Simpson` is in scope (F17). Optional.

## Bottom line

The math that is here is right. There just isn't enough of it for what the package's name and CLAUDE.md row promise. ΔE2000 saturates the Sharma reference; Lab continuity is exact; sRGB has the standard's intrinsic 2-9 mismatch and not more; Bradford is correct to 7 digits. But gamut mapping (the audit topic), CAT02/CAT16, ΔE94, ΔECMC, WCAG contrast, LCH, HSL, and 80% of golden files are simply not implemented. The package should either be renamed `color/core` and have the gaps acknowledged in CLAUDE.md, or the gaps should be filled — preferably the latter, since each one is well-scoped (≤100 LOC) and they together unlock the package for actual production color management.
