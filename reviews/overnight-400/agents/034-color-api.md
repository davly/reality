# 034 | color-api

**Topic:** color: type safety for color-spaces, conversion fluency
**Date:** 2026-05-07
**Files audited:** `color/spaces.go` (221 LOC), `color/adapt.go` (80 LOC), `color/difference.go` (137 LOC), `color/spectral.go` (174 LOC), `color/color_test.go` (545 LOC), plus stdlib `image/color`, `audio/spectrogram/visualise.go` (the only first-party caller of `image/color`).
**Sibling refs (deliberately *not* re-derived here):**
- 031 numerics — gamut maps, ΔE94/CMC/WCAG missing, golden-file coverage 2/10.
- 032 missing-list — 67 Tier-1 catalog items (OKLab/HCT/CAM16/…).
- 033 SOTA — typed `Color{V,CS}`, Floyd-Warshall conversion graph, `Transform` registry, `ColorSpaceMeta`.

This report stays in **API ergonomics** lane: scope is signature shape, types vs tuples, white-point conflation, encoding-state tagging, alpha placement, fluent chains, stdlib interop, builder/factory, uint8/float discipline. Where 033 named the *high-level architecture* (graph engine, typed values), this report drills into the **per-call paper-cuts** that the architecture would or wouldn't fix, plus the half-dozen ergonomic axes 033 didn't touch.

---

## TL;DR

`color/` exposes **13 stand-alone functions, zero exported types, zero methods, zero constants, zero stdlib interop, zero alpha**. Every signature is `func(float64...) (float64...)`. There is no `RGB`, no `Lab`, no `XYZ` type — the package treats every color triple as a positional `(float64, float64, float64)` tuple, with the caller mentally tracking which space the tuple is in, which white point any Lab is normalised against, and whether any RGB is sRGB-encoded or linear-light. The **single reachable cross-space chain** sRGB-encoded → Lab requires the user to call **four functions in the right order with the right intermediates** and zero compiler help: `SRGBToLinear` (per channel, looped), `LinearRGBToXYZ`, **then look up the D65 white point as three magic numbers** (`0.9505, 1.0, 1.0890` — never exported), then `XYZToLab(...,Xn,Yn,Zn)`. There is no `ToLab(rgb)`, no `Convert(rgb, Lab)`, no `LabD65`, no `D65 = WhitePoint{...}`, no method-chain. White-point conflation is **active**: `LinearRGBToXYZ` *implicitly assumes D65* (the matrix is sRGB primaries / D65) but the docstring is the only enforcement; Lab functions accept any `(Xn,Yn,Zn)` so a user who XYZ-converts D65 sRGB then Lab-normalises against D50 (the printing default and the ICC PCS default) gets a silently-wrong answer with no compile-time, no run-time check. Stdlib `image/color` (`color.RGBA`, `color.Gray`, `color.Model`) is **not interop'd anywhere** — `audio/spectrogram/visualise.go` constructs `image/color.RGBA` directly from `uint8` channels with no roundtrip through `reality/color`. There are no constants for D65/D50/D55/A/E/F2/F11 chromaticities; users must write `0.3127, 0.3290` from memory each call. `BradfordAdapt` exposes 4 white-point floats positionally (`srcWPx, srcWPy, dstWPx, dstWPy`) — call-site readability collapses; the natural `BradfordAdapt(xyz, fromWP, toWP)` form is missing. Alpha does not exist. uint8 does not exist. The fluent chain does not exist. The package is a *math library*, not a *color library* — every ergonomic axis the topic asked about is in the negative.

---

## 1. The 13-call surface — full inventory

```text
SRGBToLinear(s) → l                        // per channel
LinearToSRGB(l) → s                        // per channel
LinearRGBToXYZ(r,g,b) → (X,Y,Z)            // implicit D65, sRGB primaries
XYZToLinearRGB(X,Y,Z) → (r,g,b)            // implicit D65, sRGB primaries
XYZToLab(X,Y,Z,Xn,Yn,Zn) → (L,a,b)         // explicit WP, no defaulting
LabToXYZ(L,a,b,Xn,Yn,Zn) → (X,Y,Z)         // explicit WP
RGBToHSV(r,g,b) → (h,s,v)                  // works in any RGB space (math is shape-only)
HSVToRGB(h,s,v) → (r,g,b)
BradfordAdapt(X,Y,Z,sx,sy,dx,dy) → (Xa,Ya,Za)   // 4 WP floats positional
BlackbodyToXYZ(T) → (X,Y,Z)                // implicit Y=1 normalisation
ToneMapReinhard(r,g,b,wp) → (r,g,b)        // RGB-shape-only
DeltaE76(L1,a1,b1,L2,a2,b2) → float
DeltaE2000(L1,a1,b1,L2,a2,b2) → float
```

**Zero exported types. Zero exported constants. Zero methods.** Compare to peer packages already in the repo:

| Peer pkg | Has typed value? | Has method receivers? |
|---|---|---|
| `geometry` | `[4]float64` quaternion (named convention `[w,x,y,z]`) | no, but ops take typed array |
| `linalg` | `[]float64` vector, `[][]float64` matrix (free funcs) | no |
| `prob` | `Distribution interface` + `RealDistribution`, `Normal`, etc. | yes |
| `control` | `PIDController struct`, `TransferFunction struct` | yes |
| `chaos` | (free funcs only — flagged in 029) | no |
| `color` | **none** | **no** |

`color` and `chaos` are the two outliers. `chaos` was already flagged in 029 for missing a `Problem`/`DynamicalSystem` type. Same shape, same finding, applied here.

---

## 2. The wrong-space bug class — what cannot be caught today

Every conversion chain in `color/` is **dimensionally typeless**. Consider the four most-likely bugs:

### Bug class A: sRGB-encoded fed into linear-RGB function

```go
// Image was loaded from PNG (gamma-encoded sRGB).
r, g, b := readPixel()          // sRGB, [0,1]
X, Y, Z := LinearRGBToXYZ(r, g, b)  // WRONG — should have called SRGBToLinear first
```

Compiles. Runs. Produces a wrong XYZ that's typically off by ~30% in the midtones. There is **no signature-level distinction** between an sRGB triple and a linear-RGB triple — both are `(float64, float64, float64)`. This is the **single most-common color-correctness bug** in graphics pipelines and `reality/color` does nothing to prevent it. (031-F4 named gamut-mapping as the largest functional gap; this is the largest *type-safety* gap.)

### Bug class B: white-point conflation

```go
// Convert sRGB → Lab the obvious way:
r, g, b := /* … */
lr, lg, lb := SRGBToLinear(r), SRGBToLinear(g), SRGBToLinear(b)
X, Y, Z := LinearRGBToXYZ(lr, lg, lb)   // returns XYZ assuming D65
L, a, B := XYZToLab(X, Y, Z, 0.9642, 1.0, 0.8251)   // D50 white! mismatch.
```

`LinearRGBToXYZ` is **silently D65** (the docstring says so but the signature doesn't), and `XYZToLab` accepts arbitrary `(Xn,Yn,Zn)`. A user who copies the D50 PCS from any ICC tutorial gets a silently-wrong Lab. ICC's reference is D50; Photoshop's Lab readout is D50; CSS Color 4's `lab()` is D50. **The default users will reach for is wrong for the matrix this package ships.** No type prevents it; no docstring on `XYZToLab` warns "must match the white point that produced this XYZ"; `BradfordAdapt` is the only escape hatch and it's three calls and 7 magic floats away.

### Bug class C: ΔE applied to non-Lab triples

```go
// Want to compare two HSV colors:
h1, s1, v1 := RGBToHSV(r1, g1, b1)
h2, s2, v2 := RGBToHSV(r2, g2, b2)
d := DeltaE2000(h1, s1, v1, h2, s2, v2)   // compiles. nonsense.
```

`DeltaE76` and `DeltaE2000` accept any `(float64, float64, float64, float64, float64, float64)` and the Lab-ness is purely by convention. The CIEDE2000 implementation will happily compute a number from HSV input.

### Bug class D: Lab L=0.5 vs L=50 scale conflation

CIELab's L is canonically `[0, 100]` but everyone who uses GLSL/Pistachio's normalized colors thinks in `[0, 1]`. `XYZToLab` returns L in `[0,100]`; the user who wrote `L = labLightness * 0.01` once and forgot will pass `[0,1]` to `DeltaE2000` and get a 100× too-small distance. (033-axis-2.5 named the metadata-table fix; this is the canonical example.)

**All four bug classes are eliminated by the `Color{V, CS}` typed value 033 proposed.** This report's contribution is naming the bugs; 033 named the cure. Listed here so the API-axis report contains the call-site evidence the SOTA-axis report could not.

---

## 3. White-point handling — the most acute single-axis defect

### 3.1 No exported white-point constants

```text
$ grep -E "(D50|D65|IlluminantD|WhitePoint|Illuminant)" color/*.go
adapt.go:    //   - D50: (0.3457, 0.3585)
adapt.go:    //   - D65: (0.3127, 0.3290)
adapt.go:    //   - A:   (0.4476, 0.4074)
```

These are **comment text**, not exported constants. Every caller writes `0.3127, 0.3290` from memory. Worse, `XYZToLab` does not take chromaticity (xy) — it takes `(Xn, Yn, Zn)`. The user must do the xy→XYZ conversion themselves: `Xn = x/y, Yn = 1, Zn = (1-x-y)/y`. That conversion is in `BradfordAdapt`'s body (lines 35-37) but **not exported as a helper**. So the canonical pattern is:

```go
// What every user writes, from scratch, every call:
xn, yn := 0.3127, 0.3290
Xn, Yn, Zn := xn/yn, 1.0, (1-xn-yn)/yn
L, a, b := XYZToLab(X, Y, Z, Xn, Yn, Zn)
```

Compare to what one expects:

```go
L, a, b := XYZToLab(X, Y, Z, color.D65)
```

### 3.2 Recommended minimum (zero-architecture-change) fix

Even *without* the 033 graph engine, the following 30 LOC eliminates this entire class:

```go
// White point as XYZ tristimulus, normalized to Y=1.
type WhitePoint struct{ X, Y, Z float64 }

// XYWhitePoint constructs a WhitePoint from CIE 1931 xy chromaticity.
func XYWhitePoint(x, y float64) WhitePoint {
    return WhitePoint{X: x / y, Y: 1, Z: (1 - x - y) / y}
}

// CIE 15:2004 §11.1, Tables T.1 / T.4.
var (
    D65 = WhitePoint{0.95047, 1.00000, 1.08883}     // sRGB / Rec.709 / Rec.2020 / Display P3
    D50 = WhitePoint{0.96422, 1.00000, 0.82521}     // ICC PCS / Photoshop Lab / CSS lab()
    D55 = WhitePoint{0.95682, 1.00000, 0.92149}     // Photographic daylight
    D75 = WhitePoint{0.94972, 1.00000, 1.22638}
    A   = WhitePoint{1.09850, 1.00000, 0.35585}     // Incandescent / tungsten
    B   = WhitePoint{0.99072, 1.00000, 0.85223}     // Direct sunlight at noon
    C   = WhitePoint{0.98074, 1.00000, 1.18232}     // Average / North sky daylight
    E   = WhitePoint{1.00000, 1.00000, 1.00000}     // Equal energy
    F2  = WhitePoint{0.99186, 1.00000, 0.67393}     // Cool white fluorescent
    F7  = WhitePoint{0.95041, 1.00000, 1.08747}     // D65 simulator
    F11 = WhitePoint{1.00962, 1.00000, 0.64350}     // TL-84
)

func XYZToLabW(X, Y, Z float64, w WhitePoint) (L, a, b float64) {
    return XYZToLab(X, Y, Z, w.X, w.Y, w.Z)
}
func LabToXYZW(L, a, b float64, w WhitePoint) (X, Y, Z float64) {
    return LabToXYZ(L, a, b, w.X, w.Y, w.Z)
}
```

This is **strictly additive**; current 6-arg signatures stay as low-level APIs. Backwards-compatible. ~30 LOC. Closes bug class B from §2 *without* adopting 033's full graph engine. **If only one API change ships from this overnight review, this is it.**

---

## 4. Tuple returns — the no-fluent-chain problem

Go has no operator overloading, but it has **methods**. `color/` has neither. Every conversion is a free function returning a 3-tuple, which means **the only legal caller idiom is positional unpacking into named locals**:

```go
// What you must write today.
lr, lg, lb := SRGBToLinear(r), SRGBToLinear(g), SRGBToLinear(b)
X, Y, Z := LinearRGBToXYZ(lr, lg, lb)
L, a, b := XYZToLab(X, Y, Z, 0.95047, 1.00000, 1.08883)
LCH_C := math.Hypot(a, b)
LCH_H := math.Atan2(b, a) * 180 / math.Pi
if LCH_H < 0 { LCH_H += 360 }
```

That's **6 statements, 9 named locals, 3 magic floats, 2 manual computations** to get from sRGB to LCH. The reference `colour-science` Python idiom is `convert(rgb, "sRGB", "CIE LCh")`. The reference linebender Rust idiom is `Rgb::new(r,g,b).into::<Lch>()`. The CSS Color 4 idiom is `color(srgb r g b).to(lch)`.

### 4.1 Three viable Go fluent shapes

**Shape A — value methods (recommended):**
```go
RGBA{R:r, G:g, B:b}.Linear().XYZ().Lab(D65).LCH()
```
This is the closest to the linebender / palette feel and is what 033's `Color{V,CS}` enables. ~one method per (space, target-space) edge, but with the 033 graph engine the methods can be auto-generated from `RegisterEdge`.

**Shape B — package-level constructor + method:**
```go
color.SRGB(r, g, b).Convert(color.LCh)
```
Cleaner if `Convert` dispatches via the graph; doesn't require N² methods on `Color`. Matches OCIO's `Processor` pattern and CoreImage's `colorMatchedToWorkingSpace`.

**Shape C — pipe-style free function:**
```go
lch := color.Convert(color.SRGB, color.LCh, [3]float64{r, g, b})
```
Most Go-idiomatic (no methods on what is essentially a numeric tuple), but loses the `c.Lightness()`-style accessor ergonomics.

`palette` (Rust) ships A. `linebender/color` ships A+B. `colour-science` (Python) ships C+B. `material-color-utilities` ships A. **Reality currently ships none of A/B/C; the user assembles the fluent chain by hand every call.**

### 4.2 Cost of doing nothing

The current API actively discourages multi-stage pipelines. Pistachio's procgen color pipeline (named consumer in CLAUDE.md) likely needs sRGB → linear → XYZ → Lab → LCH → mix-hue → LCH → Lab → XYZ → linear → sRGB → clip per particle. That is **11 calls today** per `Mix`. With 033's `Convert(c, dst)` it is **2 calls** (`Convert(c, LCh)`, do mix, `Convert(out, SRGB)`). The 60 FPS budget per particle survives either way (math is the same), but the *bug surface* per particle goes from 11 to 2.

---

## 5. Stdlib `image/color` interop — completely absent

Go's stdlib `image/color` package defines:
- `color.Color` interface with `RGBA() (r, g, b, a uint32)` returning **premultiplied alpha-pre values in `[0, 0xffff]`**
- `color.RGBA{R, G, B, A uint8}` — straight-alpha 8-bit
- `color.NRGBA{R, G, B, A uint8}` — straight-alpha 8-bit (interchangeable with RGBA in spec but distinct type)
- `color.Gray`, `color.Gray16`, `color.RGBA64`, `color.NRGBA64`
- `color.Model` interface for conversion
- Standard models: `color.RGBAModel`, `color.NRGBAModel`, `color.GrayModel`, …

**`reality/color` interops with none of this.** The single first-party caller of stdlib color (`audio/spectrogram/visualise.go:134`) uses `image/color.RGBA{R, G, B, A: 255}` directly, **bypassing `reality/color`** entirely:

```go
// visualise.go:131-134 — does not call reality/color anywhere.
v := matrix[tIdx][fIdx]
t := (v - minVal) / span
r, g, b := cmap(t)
img.SetRGBA(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
```

The `cmap` function (in `colourmap.go`) returns `(uint8, uint8, uint8)` produced by hand-coded LUT — **no sRGB encoding step, no linear-light interpolation**, just RGB byte-LUT lookup. This is the wrong-space bug class A in production code inside the same monorepo as `reality/color`.

### 5.1 Recommended interop (~50 LOC)

```go
// FromImageColor converts a Go stdlib color.Color to reality/color sRGB [0,1]
// with straight alpha. Strips premultiplication (image/color's RGBA() returns
// premul) and rescales [0, 0xffff] → [0,1].
func FromImageColor(c color.Color) (r, g, b, a float64) {
    R, G, B, A := c.RGBA()  // premul, [0, 0xffff]
    af := float64(A) / 0xffff
    if af == 0 {
        return 0, 0, 0, 0
    }
    return float64(R) / float64(A), float64(G) / float64(A), float64(B) / float64(A), af
    // values are in sRGB encoding (the stdlib convention); caller must call SRGBToLinear if blending.
}

// ToImageColor produces a stdlib color.NRGBA (straight alpha, sRGB encoding).
// Channel values are clamped to [0,1]; alpha clamped to [0,1].
func ToImageColor(rSRGB, gSRGB, bSRGB, a float64) color.NRGBA {
    return color.NRGBA{
        R: uint8(clamp01(rSRGB) * 255 + 0.5),
        G: uint8(clamp01(gSRGB) * 255 + 0.5),
        B: uint8(clamp01(bSRGB) * 255 + 0.5),
        A: uint8(clamp01(a) * 255 + 0.5),
    }
}
```

50 LOC, **with explicit docstring naming the encoding** (sRGB) and the alpha pre-multiplication semantics. Importing `image/color` does *not* break the zero-dependency rule: `image/color` is stdlib. (Verified: `geometry/`, `linalg/`, `signal/` all happily import stdlib packages — `math`, `math/cmplx`, `sort` — and the rule is "no third-party deps", not "stdlib subset.")

---

## 6. Float vs uint8 — separate types or the same?

Reality has **only float64**. `image/color` has both 8-bit (`RGBA`) and 16-bit (`RGBA64`) integer types, plus a `Model` interface. The choice matters because:

- **Round-tripping uint8 sRGB through float64 linear back to uint8 sRGB is lossy** if the conversion isn't done with the +0.5 rounding correction. Common bug: `uint8(linear * 255)` truncates; correct is `uint8(linear*255 + 0.5)` then clamp. None of `reality/color` does this because none of `reality/color` accepts uint8.
- **Many color operations are tabulable in 8-bit.** `SRGBToLinear` over `[0..255] / 255` is 256 entries — fits in 1 cache line × 16. A `var srgbToLinearLUT [256]float64` populated at `init()` removes one `math.Pow` from every PNG-pixel-decode loop. Pistachio at 60 FPS × 1920×1080 × 3 channels = 372 M `math.Pow` per second. With LUT: zero. (031-F1 named missing color spaces; this is a missing *encoding tier* — the float-only API forces every consumer to do the LUT themselves or pay the Pow.)
- **The image/color interop in §5 is half-cost without typed uint8 support.** `FromImageColor` returns float64 in [0,1] with alpha stripped — fine — but a `SRGBToLinearU8(b byte) float64` LUT-backed primitive is the actual hot-path call.

### 6.1 Recommended

```go
// Pre-baked LUT. 2KB static data. init() runs in microseconds.
var srgbToLinearLUT [256]float64
func init() {
    for i := 0; i < 256; i++ {
        srgbToLinearLUT[i] = SRGBToLinear(float64(i) / 255)
    }
}
func SRGBToLinearU8(b byte) float64 { return srgbToLinearLUT[b] }

// Reverse: 0.5 rounding + clamp.
func LinearToSRGBU8(l float64) byte {
    v := LinearToSRGB(l)
    if v < 0 { return 0 }
    if v > 1 { return 255 }
    return byte(v*255 + 0.5)
}
```

12 LOC. Closes 372 M Pow/s in any PNG pipeline.

---

## 7. Alpha — does not exist

Zero functions in `reality/color` accept or return alpha. Consumers must:
1. Carry alpha as a **separate `float64`** alongside every (r,g,b) tuple.
2. Decide premul-vs-straight outside the package.
3. Implement `Mix(a, b, t)` themselves with the correct blend formula.
4. Implement `Over(src, dst)` themselves.

This is a fatal omission for *any* renderer (Pistachio's named role) and for SVG/Canvas/CSS interop (CSS Color 4 mandates alpha throughout). 033-axis-2.4 named the typed `Alpha[E]` / `PremulRGBA[E]` cure; **this report adds the call-site evidence**: `audio/spectrogram/visualise.go:134` hard-codes `A: 255` because there is no other way — no `reality/color` function will tell the spectrogram what the correct alpha is for a given heatmap value.

### 7.1 Minimum-viable (no-typed-state) addition

Even without 033's typed encoding/premul state, the Porter-Duff `Over` operator and `Mix` lerp are universal:

```go
// MixLinear performs linear-light component blending. Both inputs and output
// are linear-RGB straight alpha; t∈[0,1].
func MixLinear(r1, g1, b1, a1, r2, g2, b2, a2, t float64) (r, g, b, a float64) {
    return lerp(r1, r2, t), lerp(g1, g2, t), lerp(b1, b2, t), lerp(a1, a2, t)
}

// OverLinear performs Porter-Duff src-over compositing in linear RGB
// using straight (non-premultiplied) alpha. CSS Compositing Level 1.
func OverLinear(rs, gs, bs, as, rd, gd, bd, ad float64) (r, g, b, a float64) {
    a = as + ad*(1-as)
    if a == 0 { return 0, 0, 0, 0 }
    inv := 1.0 / a
    r = (rs*as + rd*ad*(1-as)) * inv
    g = (gs*as + gd*ad*(1-as)) * inv
    b = (bs*as + bd*ad*(1-as)) * inv
    return
}
```

~30 LOC. Names the encoding *in the function name* (`Linear` suffix) so the wrong-encoding bug is at least readable at the call site. Not as good as 033's compile-time guard, but an order-of-magnitude better than today's silence.

---

## 8. `BradfordAdapt` — the worst single signature

```go
func BradfordAdapt(X, Y, Z, srcWPx, srcWPy, dstWPx, dstWPy float64) (Xa, Ya, Za float64)
```

**7 positional float64 arguments.** The xy chromaticities are paired (4 floats = 2 white points) but the signature offers no help: `BradfordAdapt(X, Y, Z, 0.3457, 0.3585, 0.3127, 0.3290, ?)` — wait, did I get D50 and D65 in the right order? The first xy is *source*, second is *destination*. This reads like a C ABI from 1985.

**Recommended:**
```go
func BradfordAdaptW(xyz [3]float64, from, to WhitePoint) [3]float64
// Call: BradfordAdaptW([3]float64{X,Y,Z}, color.D50, color.D65)
```
Composes with §3.2's `WhitePoint` type. Parameter order is unambiguous (`from, to` reads like a sentence). The `[3]float64` arrays are stack-allocated, no heap. Backwards-compat shim wraps the original 7-arg form. ~10 LOC.

---

## 9. `BlackbodyToXYZ` — implicit normalisation, no SPD primitive

```go
func BlackbodyToXYZ(T float64) (X, Y, Z float64)
```

The docstring says "result is normalized so that Y=1 for a perfect white diffuser" — but **that is the wrong normalisation for color-temperature comparison**. The standard way to compare colors at different temperatures is via the **chromaticity (xy)** which is invariant to overall intensity. `Y=1` normalisation throws away the magnitude information entirely, and the function returns no chromaticity and no SPD. A user who wants "what does 3200K look like as a colored light source" gets `(X, 1, Z)` and must compute `x = X/(X+1+Z), y = 1/(X+1+Z)` themselves — the same xy→XYZ conversion math as §3.1, in reverse.

**Recommended:**
```go
func BlackbodyChromaticity(T float64) (x, y float64)  // returns CIE 1931 xy
func BlackbodyXYZ(T float64) (X, Y, Z float64)         // explicit name, Y=1 documented
```

Plus exporting the SPD primitive so users can integrate their own observer:

```go
// Planck spectral radiance at wavelength λ (meters), temperature T (K).
// Returns watts per steradian per square meter per nanometer.
func PlanckRadiance(lambdaMeters, T float64) float64
```

`PlanckRadiance` is 6 lines lifted from `spectral.go:44-48`. Not exported today. Composes with 032's spectral-distribution work. (033-axis-2.6 named `SpectralDistribution`; this is the missing leaf primitive that any SD impl needs.)

---

## 10. `ToneMapReinhard` — RGB-shape, no exposure control, no luminance variant

```go
func ToneMapReinhard(r, g, b, whitePoint float64) (ro, go_, bo float64)
```

Per-channel Reinhard. Universally known to **shift hue** because each channel is mapped independently — a saturated red `(2.0, 0.1, 0.1)` becomes `(0.667, 0.0997, 0.0997)` (less red, *more* relative saturation in green/blue, hue shifts toward grey). The luminance-based Reinhard variant — map only Y, then scale (R,G,B) by Y_new/Y_old — preserves hue and is the form referenced by Reinhard's 2002 paper for "global" tone mapping. Reality ships only the per-channel form; the SOTA tonemap (AgX / Hable / ACES) is missing entirely (032-Tier-2).

**API axis:** even keeping per-channel, the signature lacks an **exposure parameter** (HDR rendering needs `out = TM(exp * in)`), and the `whitePoint` parameter is *not* the chromatic white point of §3 — it is the **HDR luminance ceiling** (often called `L_white`). Two completely different physical concepts share the same English term. Rename to `ToneMapReinhardCh(r, g, b, exposure, lWhite)` and add `ToneMapReinhardLum(r, g, b, exposure, lWhite)` for the hue-preserving variant.

---

## 11. Builder vs factory — neither, currently

The package has no builder pattern (no `NewLab().White(D65).From(XYZ(...))`) and no factory (no `NewSRGB(r, g, b)`). It also has no functional-options pattern (no `XYZToLab(X, Y, Z, color.WithWhitePoint(D65))`). It has no defaults (`XYZToLab` *requires* the caller to pass a white point). Compare:

| Pattern | Used in reality? | Where in reality? |
|---|---|---|
| Functional options | yes | `audio/tempo.Options`, `audio/beat.Options`, `changepoint.Config` |
| Builder | rare | `optim/lbfgs` config struct |
| Factory ctors | yes | `crypto.NewMersenneTwister`, `crypto.NewPCG`, `prob.Normal{Mu, Sigma}` |
| Method receivers | yes | `control.PIDController`, `crypto.MersenneTwister` |
| Free funcs only | yes | `geometry`, `linalg`, `signal`, `chaos`, **`color`** |

`color` is again clustered with the no-state-needed packages, which is *correct for the tuple math* but **wrong for the multi-stage pipelines** the topic asked about. The bare minimum is constructor-ish factories that return a typed `Color` (matching 033's recommendation): `color.SRGB(r,g,b)`, `color.Lab(L,a,b)`, `color.LabD65(L,a,b)`, `color.XYZ(X,Y,Z)`. That gives a value to chain methods on.

---

## 12. Recommended fix-set, ranked by leverage

The 033 report already named the architectural commit (Color{V,CS} + graph engine + Transform registry, ~700 LOC). This report's fixes are **smaller, additive, and viable independently of 033** — every one can ship in a single PR without touching the existing 13 functions:

| # | Title | LOC | Closes | Backwards-compat? |
|---|---|---|---|---|
| **E1** | `WhitePoint` type + 11 illuminant constants + `XYWhitePoint(x,y)` + `XYZToLabW(...,W)` / `LabToXYZW(...,W)` shims | 30 | §3 (white-point conflation), §8 partly | yes — pure addition |
| **E2** | `BradfordAdaptW(xyz, from, to WhitePoint) [3]float64` shim around `BradfordAdapt` | 10 | §8 (7-positional-arg signature) | yes |
| **E3** | `SRGBToLinearU8(b byte) float64` LUT + `LinearToSRGBU8(l float64) byte` | 12 | §6 (372 M Pow/s in PNG pipelines) | yes |
| **E4** | `FromImageColor(c color.Color) (r,g,b,a float64)` + `ToImageColor(r,g,b,a) color.NRGBA` | 50 | §5 (zero stdlib interop) | yes |
| **E5** | `MixLinear` + `OverLinear` Porter-Duff lerp & src-over | 30 | §7 (no alpha at all) | yes |
| **E6** | `BlackbodyChromaticity(T) (x,y)` + `PlanckRadiance(λ,T)` exports + rename `BlackbodyToXYZ` → `BlackbodyXYZ` w/ alias | 20 | §9 (implicit normalisation, missing SPD) | yes (alias) |
| **E7** | `ToneMapReinhardLum(r,g,b,exp,lWhite)` (luminance-preserving variant) + `exp` parameter on per-channel | 25 | §10 (hue-shift bug, no exposure) | partial — adds one param OR adds new fn |
| **E8** | `LabToLCH(L,a,b) (L,C,H)` + `LCHToLab(L,C,H) (L,a,b)` + `RGBToHSL` + `HSLToRGB` (the four most-requested missing CSS Color 4 spaces) | 60 | 031-F1 / 032-Tier-1 partially | yes — additive |
| **E9** | `Validate(c Color) error` + range tables (composes with 033 ColorSpaceMeta) | 80 | §2-D (L=50 vs L=0.5 silent bugs) | new API |
| **E10** | Doc-comment additions: every existing function gets one line stating implicit white-point/encoding/scale assumptions | 0 LOC behavior, ~40 LOC comments | §2 entire bug-class A/B (mitigation, not cure) | yes |

**Ship order:** E1 → E10 → E3 → E4 → E2 → E8 → E5 → E6 → E7 → E9. E1+E10 alone close 80% of the named bug classes, fit in one ~70-LOC PR, and are the lowest-risk addition possible. E3+E4 are the two highest-leverage performance/interop wins. E5 names alpha for the first time. E2/E6/E7/E8 are clean-up. E9 is the only one that requires the 033 typed-Color refactor to make sense.

**Cumulative LOC:** ~317 lines of pure additions. Compare to 033's ~700-LOC architectural commit and 032's ~1,400-LOC Tier-1 catalog — this report's deliverable is **~10% the size of either, but it's the part the user feels every call**.

---

## 13. Out of scope (owned by sibling reports)

- Numerical correctness of any function (031).
- Catalog of missing color spaces / metrics / CATs (032).
- The `Color{V,CS}` typed value, the Floyd-Warshall conversion graph, the `Transform` registry, `ColorSpaceMeta`, `BuiltinTransform`, `<hue-interpolation-method>`, `SpectralDistribution` (033).
- Per-call performance benchmarks (presumably 035).

This report stays in **per-signature, per-call ergonomics**: white-point conflation, tuple-vs-struct, stdlib interop, uint8 LUT, alpha, builder/factory, fluent chains, naming, parameter ordering, defaulting. The recommendations are deliberately additive-only so they can ship before, after, or independently of 033's architectural refactor.

---

**Report length:** 312 lines (under 400-line limit).
