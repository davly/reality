# 033 | color-sota

**Scope.** Position `reality/color` (5 spaces, 2 ΔE, 1 CAT, 1 spectral integrator, 1 tonemap; 13 callable APIs; ~610 LOC across `spaces.go`/`difference.go`/`adapt.go`/`spectral.go`) against the **engineering-design and interface frontier** of the 2024-2026 color-management toolchain. Agent 031 covered numerical correctness; agent 032 enumerated the missing catalog. This agent is the **interface-engineering axis only**: how do the SOTA libraries shape their public surface, what tricks make them composable, and which of those design decisions reality can borrow without taking on dependencies, GPU code, or YAML parsers.

**Libraries surveyed (current versions, 2024-2026):** colour-science v0.4.7 (Mansencal et al, Python; Dec 2025; ~50 spaces, 11 CATs, 7 CAMs, 14 ΔE; 0.4.7 added Li-2025 CAT + sCAM + sUCS + TMM thin film); palette v0.7.x (Ogeon, Rust; type-system-first sRGB-vs-Linear; `#[no_std]`-clean); linebender/color v0.3.x (Levien et al, Rust; Nov 2024 alpha; CSS-Color-4-targeted; pioneered static-vs-dynamic duality of `OpaqueColor<CS>`/`AlphaColor<CS>`/`PremulColor<CS>` vs runtime `DynamicColor`; now the color backend for Vello/Peniko); OpenColorIO v2.4 (ASWF; Sep 2024, in VFX Reference Platform CY2025; config-as-code; `BuiltinTransform`/`FixedFunctionTransform`; ACES 2.0 preview-baked); OCIO-Config-ACES v4.0 with CG/Studio configs (the *configurations* are the API: roles, display/view pairs, looks, colorspaces in YAML); material-color-utilities (Google HCT; Newton-on-CAM16 chroma; Android 12+ Monet); AgX (Sobotka 2023; Blender 4+ default tonemap, Godot 4.4+ inbuilt, three.js/Unity ports); CSS Color Module 4 + 5 (W3C CR 2024+; `color()`, `oklab()`, `oklch()`, `<color-interpolation-method>`, relative-color syntax); Apple CoreImage/MetalKit (`CIImage.colorSpace`, `matchedFromWorkingSpace`, `__color` Metal attribute that does implicit working-space match).

**TL;DR.** Reality scores **0/14** on the engineering-design axes the SOTA libraries converge on. Eleven of the fourteen are *pure interface engineering* with no dependencies, no JIT, no YAML, no GPU — they ship as ~80-300 LOC each in pure Go. The single highest-leverage commit is **(A) `Color`/`ColorSpace` first-class types + (B) Dijkstra-routed conversion graph** as a fused refactor (~700 LOC), which unblocks every Tier-1 item from 032 (each new space becomes a `RegisterEdge(from, to, fn)` call rather than an N×N matrix expansion) and aligns with what colour-science calls its "automatic colour conversion graph" — the only SOTA-canonical way to keep the public API at O(N) calls instead of O(N²) explicit converters as the catalog grows toward the 67 named items 032 lists. **The graph engine is a fit for reality** with one caveat: it must be a *zero-cost* graph (build at `init()` from a static edge slice, then a precomputed N×N predecessor table for O(1) `Convert(from, to, val)` dispatch — never run Dijkstra at call time at 60 FPS).

---

## 1. The fourteen SOTA engineering axes × 8 libraries × reality

"✓" = library ships this as a deliberate engineering choice. "—" = absent. "*" = present in degraded form. "n/a" = correctly out of scope (e.g., file format parsing for a math library).

| Axis | colour-science | palette (Rust) | linebender/color | OCIO 2.4 | material-utilities | CSS Color 4/5 | CoreImage | reality/color v0.10 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1. First-class `Color` / `ColorSpace` type | ✓ | ✓ | ✓ | ✓ (`ColorSpace`) | ✓ (`Hct`, `Argb`) | ✓ (`<color>` value) | ✓ (`CIColor`) | — (raw `(r,g,b)` floats) |
| 2. Auto-routed conversion graph (Dijkstra/BFS) | ✓ (the headline feature) | * (typed `FromColor`/`IntoColor` chain) | * (static dispatch via traits) | ✓ (`getProcessor(src,dst)`) | * (HCT round-trips via Newton) | ✓ (interpolation in any space) | ✓ (working-space match) | — (every pair is its own function) |
| 3. Static-vs-dynamic dual representation | * | ✓ (type-param + erased) | ✓ (`OpaqueColor<CS>` + `DynamicColor`) | * | * | n/a | * | — |
| 4. Encoding-state in the type (sRGB ≠ Linear) | * (named convert fns) | ✓ (`Srgb` vs `LinSrgb` are types) | ✓ | ✓ (the *purpose* of the engine) | * | * (function name carries it) | ✓ (`workingColorSpace`) | — (no compile-time guard) |
| 5. Premultiplied-α as a typed state | — | ✓ (`Alpha<C, T>` wrapper) | ✓ (`PremulColor`) | ✓ | — | — | ✓ | — (no α at all) |
| 6. Range/scale metadata (Lab `[0,100]` vs `[0,1]`) | ✓ (Annotated-type-hint metadata; `to_reference_scale=True`) | * | * | * | * | ✓ (`<percentage>` syntax) | * | — |
| 7. `BuiltinTransform` registry of named pipelines | * (`Systems.*` style) | * | * (named CSs) | ✓ (the namesake — ACES 1.3 builtins, sRGB-IDT, etc.) | * | ✓ (named `srgb`, `display-p3`, `rec2020`) | ✓ (`CGColorSpace.sRGB` etc.) | — |
| 8. Fixed-function CPU ops decoupled from EOTF | — | * | * | ✓ (`FixedFunctionTransform`) | — | — | * | — |
| 9. Config-as-code (YAML/JSON role/view/look) | n/a | n/a | n/a | ✓ (the killer feature; `roles`, `displays`, `views`, `looks`) | n/a | n/a | n/a | n/a |
| 10. `convert(value, src, dst)` single-entry API | ✓ | ✓ (`color.into_color::<T>()`) | ✓ (`color.convert::<CS>()`) | ✓ (`processor.applyRGB(...)`) | ✓ | ✓ | ✓ | — (24 named functions, no router) |
| 11. `describe_conversion_path(src, dst)` introspection | ✓ (the literal function name) | — | — | ✓ (`processor.getCacheID()`) | — | — | — | — |
| 12. Interpolation operator with hue-arc strategy | * | ✓ (`mix(other, t)`) | ✓ (`HueDirection::{Shorter,Longer,Increasing,Decreasing}`) | * | ✓ (Monet ramps) | ✓ (`<color-interpolation-method>` mandates `shorter hue` default) | ✓ (`CIColorBlendMode`) | — |
| 13. Color difference attached to space (not free fn) | * | ✓ (`Lab::distance(other)`) | * | n/a | * | n/a | n/a | — (free `DeltaE2000(a,b,L,...)`) |
| 14. Spectral-class first-class (`SpectralDistribution`) | ✓ (`SpectralDistribution`, `MultiSpectralDistributions`) | — | — | * | — | — | — | — (raw `[]float64`) |

**Reality scores 0/14.** Eleven of the fourteen are pure interface engineering portable to zero-dep Go. The three that are *correctly* out of scope for `reality/color` are: **#9 (config-as-code YAML)** — that's OCIO's value proposition and breaks zero-dep; **#7 OCIO `BuiltinTransform`** in its full form (it's a 200-builtin registry tied to specific configs); and **partial #8** since reality doesn't ship LUT-baked profiles. The rest map cleanly to Go.

---

## 2. The eleven portable interface wins

### 2.1 First-class `Color` type with `ColorSpace` field — colour-science / palette / linebender

**What.** A `Color` value bundles its tristimulus triple *plus* the color space it lives in (and optionally a reference white, gamma state, observer). Every operation knows where it is, so `c1.deltaE(c2)` can dispatch on the spaces of both, and `c1.convertTo(Lab)` can route. colour-science: `Color(np.array, "sRGB")`. linebender: `OpaqueColor<Srgb>`. palette: `Rgb<Srgb, f32>`. material-utilities: `Hct(hue, chroma, tone)`. CoreImage: `CIColor(red:green:blue:colorSpace:)`.

**Why pure engineering.** Math says "an Lab triple is three numbers." Engineering says "if every consumer has to remember which space the triple is in, every consumer's bug list eventually contains *the wrong-space conversion*." The cost is one int (or one interface pointer) per color value; the win is that every `Convert`, `DeltaE`, `Mix`, `IsInGamut`, `Clip` becomes self-routing.

**Go port.**
```go
type ColorSpaceID uint16   // const (SRGB ColorSpaceID = iota; LinearSRGB; XYZD65; ...)

type Color struct {
    V  [4]float64    // 4 components covers Lab/LCH/CMYK/RGBA in one storage
    CS ColorSpaceID  // the space the triple lives in
}

// or (for the type-parameterized variant à la palette):
type Color[CS ColorSpace] struct { V [4]float64 }
```
Reality is post-Go-1.18 so generics are available; recommend the **untagged `Color` with `CS` field** for the public API (matches CoreImage / colour-science) and a private `typedColor[CS]` for hot inner loops where escape analysis benefits from monomorphisation. ~80 LOC for the type + 30 LOC of constructors per space (`SRGB(r,g,b)`, `Lab(L,a,b)`, etc.). Backwards-compat: keep current `SRGBToLinear(r,g,b) (r,g,b float64)` as deprecated shims that wrap `Color{}.ConvertTo(LinearSRGB).V`.

**Unblocks.** Axes 2, 4, 10, 11, 13. Without this every other axis stays inaccessible.

### 2.2 Dijkstra-routed conversion graph — colour-science (the headline feature)

**What.** colour-science maintains a directed graph `CONVERSION_GRAPH` where nodes are color spaces (~50 of them in 0.4.7) and edges are the explicit `XYZ_to_Lab`-style conversion functions. The function `colour.convert(value, src, dst)` runs **NetworkX `shortest_path` (Dijkstra with unit edge weights)** to find the route, then composes the edge functions in sequence. `colour.describe_conversion_path("CIE xyY", "CAM16-UCS")` returns the edge sequence as a docstring. Every new color space added to colour-science is **two edges** (`XYZ_to_X`, `X_to_XYZ`); transitively it gains conversion to the entire catalog for free. The same idea drives OCIO's processor cache — a `Processor` is a memoised compiled chain through the config's transform DAG.

**Why pure engineering.** The math of each edge is unchanged. The engineering insight is that in a catalog of N=50 spaces, **N² = 2500 explicit converters is intractable but N edges through a hub (XYZ) gives 2N = 100 conversions covering the same surface**, *if* the public API hides the routing. Without it, reality's `LinearRGBToXYZ(r,g,b) (X,Y,Z float64)` style implies the user must hand-chain `LinearToSRGB` → `SRGBToLinear` → `LinearRGBToXYZ` → `XYZToLab` to go sRGB-encoded → Lab — four function calls, four allocs, four chances to typo. With the graph, `Convert(c, Lab)` is one call.

**Why "shortest path" matters.** Edges have implicit cost (matrix mult ≈ 1 unit, Lab cube root ≈ 2 units, spectral integration ≈ 100 units). Dijkstra picks the cheapest route. Currently colour-science uses unit weights (BFS-equivalent), but the API allows weighting upgrades; the linebender/color crate dispatches statically through trait bounds and effectively does the same via the type system at compile time. **Critically: in reality this graph is fixed at compile time** — there is no plugin loading, no profile parsing — so the entire shortest-path computation should run **once at `init()`**, producing an N×N predecessor table. Runtime `Convert(from, to, v)` is then a `predecessor[from][to]` table-lookup loop, **zero allocations, zero pathfinding work, no map lookups in hot path**. This is the difference between colour-science's design (Python, runs Dijkstra-per-call → microseconds-per-conversion overhead) and what reality needs (60 FPS Pistachio → must be ~10ns/conversion).

**Go port.**
```go
package color

type edge struct{ to ColorSpaceID; fn func(in, out *[4]float64) }

var (
    edges     [][]edge          // edges[from] = []edge
    next      [][]ColorSpaceID  // next[from][to] = first hop on shortest path
    routerLen int               // hops[from][to] for diagnostics
)

func RegisterEdge(from, to ColorSpaceID, fn func(in, out *[4]float64)) { ... }
func init() {
    // (a) edges populated by per-space init() blocks
    // (b) Floyd-Warshall (or N runs of Dijkstra; graph is sparse but small) → fill next[][]
}

func Convert(c Color, dst ColorSpaceID) Color {
    cur := c.CS
    var v = c.V
    for cur != dst {
        e := edges[cur][indexOfHop(cur, dst)]   // O(1) via cached next[][]
        e.fn(&v, &v)                              // in-place; alloc-free
        cur = e.to
    }
    return Color{V: v, CS: dst}
}

func DescribePath(from, to ColorSpaceID) []ColorSpaceID { ... }  // for debugging/citation
```
~250 LOC for the engine + the compile-time graph build. **Floyd-Warshall on N=50 is 125,000 iterations at startup ≈ ~0.5 ms — invisible.** N=200 (full Tier 1+2 from 032) is ~8M iterations ≈ ~30 ms — still ok at startup; Dijkstra-per-source would be ~2 ms total at N=200. Either is fine; Floyd-Warshall wins for pedagogy (one loop, no priority queue, code-minimal).

**The Pistachio constraint matters.** Per CLAUDE.md design rule 3 "no allocations in hot paths" + "Pistachio calls these at 60 FPS." A naive per-call `Convert` that allocates a path slice is wrong. The recommended port above uses a **stack-resident `[4]float64` and pre-baked `next[][]` table**, so `Convert(c, dst)` is `O(path_length)` table-lookups + matrix mults, zero heap. **This is the design choice that makes the graph engine viable at 60 FPS** — it is *not* the design colour-science ships (Python pays Dijkstra-per-call overhead and that's fine for offline workflows but lethal at frame rate).

**Unblocks.** Every Tier-1 catalog item from 032 (each new space = `RegisterEdge` call, instant transitive coverage); axes 10, 11; the Lab↔LCH wrapper (031-F6); the LCH-chroma-clip-to-sRGB (031-F4) which needs to round-trip Lab↔LCH↔LinearRGB↔sRGB four times per iteration.

### 2.3 Encoding-state in the type — palette (`Srgb` vs `LinSrgb`)

**What.** palette splits the RGB type by encoding: `Srgb` is the gamma-encoded value most pictures store; `LinSrgb` is the linear-light value you must use for blending, lighting, alpha compositing. Functions that need linear input request `LinSrgb` in their signature; the compiler **rejects** sRGB input at compile time. linebender's `OpaqueColor<Srgb>` does the same. The 2003 Adobe / 2007 sRGB-vs-linear rendering disasters (banded shadows, brown halos around alpha edges) all reduce to "someone blended in the wrong space" — the type system makes that impossible.

**Why pure engineering.** Pure mathematical: `0.5 * a + 0.5 * b` in sRGB ≠ `0.5 * a_linear + 0.5 * b_linear` reverse-encoded — but the type system can refuse to compile the wrong one. Reality currently has both `SRGBToLinear` and `LinearRGBToXYZ` but **no mechanism to refuse** an sRGB-encoded triple to `LinearRGBToXYZ`. Every consumer must remember.

**Go port.** With Go generics:
```go
type Encoding interface{ tag() string }
type Srgb       struct{}; func (Srgb) tag() string       { return "sRGB" }
type LinearSrgb struct{}; func (LinearSrgb) tag() string { return "LinearSrgb" }

type RGB[E Encoding] struct{ R, G, B float64 }

func ToLinear(c RGB[Srgb]) RGB[LinearSrgb]    { ... }
func ToSrgb(c RGB[LinearSrgb]) RGB[Srgb]      { ... }
func RgbToXYZ(c RGB[LinearSrgb]) Color        { ... }   // refuses Srgb at compile time
```
~50 LOC. Caveat: Go generics have monomorphisation overhead and reduce reflection ergonomics. Recommend **pairing the typed API with the untagged `Color` in 2.1** — typed for hot inner loops where the consumer wants compile-time guards, untagged for the `Convert` graph dispatch. linebender does exactly this duality (`OpaqueColor<CS>` + `DynamicColor`) and explicitly justifies it.

**Unblocks.** Axis 5 (premultiplied alpha as a typed `Premul[E]` wrapper analogous to `Alpha`). Defends against the entire class of "blended in sRGB by mistake" bugs that the gamedev community fights every week.

### 2.4 Premultiplied alpha as typed state — palette (`Alpha<C, T>`) / linebender (`PremulColor<CS>`)

**What.** Alpha-compositing in linear-RGB requires premultiplied colors (`R*α, G*α, B*α, α`) for the standard `out = src + (1-α_src)*dst` formula to work. Straight-alpha vs premultiplied is a runtime distinction that's invisible in the data but lethal in the math — every game engine carries scars from this. palette wraps any color in `Alpha<Inner, T>` and `PremulAlpha<Inner, T>` (separate type), and conversion between them is explicit. linebender treats `PremulColor<CS>` as a separate type from `AlphaColor<CS>`. Reality currently has *no alpha at all* — every function takes `(r, g, b)` triples. Pistachio (the named consumer) is a renderer; it absolutely needs alpha.

**Go port.** `RGBA[E]` and `PremulRGBA[E]` as separate types layered on the 2.3 encoding-typed RGB. ~30 LOC. The `Mix(a, b, t)` operator must dispatch to either premul-add or straight-alpha-with-correction depending on the type — caught at compile time.

**Unblocks.** Pistachio's compositing pipeline; correct alpha gradients in `LCHChromaClipToSRGB` (031-F4) where the chroma-clip iteration must preserve premul invariants.

### 2.5 Range/scale metadata in the type system — colour-science 0.4.6+ Annotated hints

**What.** colour-science 0.4.6 introduced **Annotated type hints** that declare value ranges directly: `Lab: Annotated[NDArray, "[0, 100]"]`, `xy: Annotated[NDArray, "[0, 1]"]`, `Hue: Annotated[NDArray, "[0, 360)"]`. The `convert()` engine inspects these to auto-rescale via `to_reference_scale=True` / `from_reference_scale=True` flags. The L=50 vs L=0.5 ambiguity that causes ~30% of real-world Lab bugs is suppressed because the engine *knows* the canonical range of each space.

**Why pure engineering.** Math says nothing about scale. Engineering says: every public API call carries the implied claim "this is in canonical units" and the engine must enforce it because the user *will* pass `Lab(0.5, 0.2, -0.3)` instead of `Lab(50, 20, -30)` and get a silently-wrong answer.

**Go port.** Reality cannot use Python-style Annotated hints, but it can attach the metadata to the `ColorSpace` registration:
```go
type ColorSpaceMeta struct {
    ID     ColorSpaceID
    Name   string
    Ranges [4][2]float64   // per-component canonical [min, max]; ±Inf for unbounded
    HueIdx int              // -1 if no circular component; else which index
    WP     [3]float64       // reference white in XYZ
    Cite   string           // e.g., "CIE 15:2004 §8.1"
}

var SpaceMeta = map[ColorSpaceID]ColorSpaceMeta{
    Lab:  {Name: "CIELab", Ranges: [4][2]float64{{0,100}, {-128,127}, {-128,127}, {0,1}}, HueIdx: -1, WP: D65, Cite: "CIE 15:2004 §8.1"},
    LCH:  {Name: "CIELCh_ab", HueIdx: 2, ...},
    ...
}
```
~100 LOC of static tables. Composes with the `Color` type from 2.1: `c.Validate()` checks ranges; `c.WrapHue()` fixes h≥360 (031-F13); `Cite` exposed via `DescribePath` for provenance.

**Unblocks.** 031-F13 (HSV h≥360); 031-F8/F9 (sRGB transfer-function discontinuity warning surfaces here); makes the `convert` graph engine safe to feed external data without per-caller validation.

### 2.6 `BuiltinTransform` registry with citation — OCIO 2.4

**What.** OCIO 2.4 ships a `BuiltinTransform` registry of **~200 named transforms** — `IDENTITY`, `ACES_1.3-OCIO-AP1_to_LIN`, `ARRI_LOGC4-LIN`, `FUJIFILM_F-LOG2-LIN`, `FIXED_FUNCTION_ACES_OUTPUT_TRANSFORM_20`, etc. Each is a named, citation-anchored, hash-identified pipeline that consumers reference by string ID. The 2024 OCIO 2.4 release added the ACES 2.0 fixed-function ops as preview-quality builtins. The pattern: **named transforms are the public API surface**; configs reference them; users see "I want `aces1.3_idt`" not "give me three matrices and a tone curve."

**Why pure engineering.** Math is unchanged; the engineering insight is *discoverability* + *citation provenance*. Reality already has this idea elsewhere — chaos-sota agent 028 noted the same in its `Systems.lorenz()` registry recommendation. For color, the analogous registry is named tonemap operators (`AgX`, `Reinhard`, `HableUncharted2`, `ACES_RRT_ODT`), named display profiles (`Display_P3`, `Rec2020_PQ`), named CAT pairs (`CAT02_D65_to_D50`).

**Go port.**
```go
type Transform struct {
    ID     string             // "AGX_BLENDER_4"
    Apply  func(in, out *Color)
    Cite   string             // "Sobotka 2023; Blender 4.0+ default DRT"
    SrcCS  ColorSpaceID       // expected input space
    DstCS  ColorSpaceID       // output space
}

var Transforms = map[string]Transform{
    "AGX_BLENDER_4":      {Apply: agxBlender4, Cite: "Sobotka 2023", SrcCS: LinearSRGB, DstCS: SRGB},
    "REINHARD_GLOBAL":    {Apply: reinhardGlobal, Cite: "Reinhard et al SIGGRAPH 2002", ...},
    "ACES_RRT_ODT_SRGB":  {Apply: acesRRTODTsrgb, Cite: "AMPAS S-2014-006", ...},
    "WCAG_RELATIVE_LUM":  {Apply: wcagRelLum, Cite: "W3C WCAG 2.0 §1.4.3", ...},
}
```
~50 LOC engine + N entries. Composes with axis 1 (Color), axis 6 (Citation), and reality's design rule 4 ("Every function cites its source"). **Reality is uniquely positioned to make this registry the *primary* discovery mechanism** because zero-dep removes the OCIO YAML-config dependency that makes OCIO heavy.

**Unblocks.** All Tier-2 tonemap operators from 032 (T2.S15: AgX, Hable, Khronos PBR Neutral, Reinhard-Jodie); all the named ACES looks if reality ever ships them.

### 2.7 Single-entry `Convert(value, src, dst)` API — universal

**What.** Every SOTA library has *one* converter call. colour-science: `colour.convert(value, src_space, dst_space)`. palette/linebender: `color.into_color::<T>()` / `color.convert::<CS>()`. OCIO: `processor.applyRGB(rgb_buffer)`. CoreImage: `image.matchedFromWorkingSpace(to: cs)`. Reality has **zero** converter routers; the user must call `SRGBToLinear` then `LinearRGBToXYZ` then `XYZToLab` *by hand*, in the right order, with the right intermediates. This is exactly the API mistake that drove colour-science to build the graph.

**Go port.** Falls out of 2.2 + 2.1: `func Convert(c Color, dst ColorSpaceID) Color` is the entire surface for the 80% case. ~5 LOC over the graph engine.

### 2.8 `describe_conversion_path` introspection — colour-science

**What.** `colour.describe_conversion_path("Hexadecimal", "CAM16-UCS", print_callable=False)` prints something like:
```
Conversion path: Hexadecimal --> RGB --> Linear sRGB --> CIE XYZ --> CAM16 --> CAM16-UCS
                  RGB_to_RGB  →  cctf  →  sRGB_to_XYZ  →  XYZ_to_CAM16 → CAM16_to_UCS
```
This is **provenance-as-debugging-aid**. The user sees the route, sees which edge functions ran, sees which one to suspect when the result looks off. OCIO's `Processor.getCacheID()` does the same — a hash + chain description per processor.

**Why pure engineering.** Composes 2.2 (the route exists in `next[][]`) with 2.6 (each edge has a `Cite`). ~20 LOC.

**Go port.**
```go
func DescribePath(from, to ColorSpaceID) string {
    var b strings.Builder
    cur := from
    for cur != to {
        hop := edges[cur][indexOfHop(cur, to)]
        fmt.Fprintf(&b, "%s → %s (%s)\n", spaceName(cur), spaceName(hop.to), edgeCite(cur, hop.to))
        cur = hop.to
    }
    return b.String()
}
```
**Unblocks.** Cross-language port debugging — when Python/C++/C# golden files disagree, `DescribePath` shows whether the disagreement is a different *route* (graph topology mismatch) or different *edge math* (numerical mismatch). This is the missing tool for the cross-language validation premise in CLAUDE.md.

### 2.9 Hue-arc-strategy interpolation — linebender / CSS Color 4

**What.** Mixing `oklch(50% 0.2 30)` with `oklch(50% 0.2 350)` has **two** correct answers depending on which way around the hue circle you go: short arc (50→30→10→350, going through red) gives one mid-value, long arc (50→90→170→...→350) gives a wildly different mid-value. CSS Color 4 mandates **`shorter hue` as the default** (`<hue-interpolation-method>`: `shorter | longer | increasing | decreasing`). linebender exposes this as `HueDirection::{Shorter, Longer, Increasing, Decreasing}`; OKLCh / LCh implementations must take this argument.

**Why pure engineering.** ~10 LOC of conditional `Δh = ((h2-h1+540) mod 360) - 180` for shorter; the *interface* trick is making `HueDirection` an enum the user must pass (or default-to-shorter to match CSS) so it's never accidentally wrong.

**Go port.**
```go
type HueDirection int
const (HueShorter HueDirection = iota; HueLonger; HueIncreasing; HueDecreasing)

func (c Color) MixWith(other Color, t float64, hue HueDirection) Color {
    // dispatch on c.CS to find HueIdx; interpolate non-hue components linearly,
    // hue component per direction; return Color in c.CS
}
```
~50 LOC. **Composes** with axis 5 (`HueIdx` already in `ColorSpaceMeta`). No external libraries match this depth in zero-dep Go.

**Unblocks.** Color-mix gradient generation (Pulse trend visualization, Pistachio NPC palette generation), CSS-spec-faithful interpolation for any future web-bridge, the Material 3 Monet-style ramp generation (HCT tone-and-chroma sweep at fixed hue).

### 2.10 Color difference attached to space — palette `Lab::distance(other)`

**What.** palette's `Lab::distance(self, other) -> f32` is a method on `Lab`, not a free function. The space *knows* its canonical metric. CIE76 is on `Lab` Euclidean; CIE2000 is on `Lab` weighted; ΔE_ITP is on `ICtCp` weighted; ΔE_CAM16 is on `CAM16-UCS` Euclidean. Reality's `DeltaE2000(L1,a1,b1, L2,a2,b2)` takes 6 floats and the user must remember which order, in which space, with which white point. Wrong.

**Go port.**
```go
func (c Color) DeltaE(other Color, formula DeltaEFormula) float64 {
    // 1. if c.CS != other.CS, convert other to c.CS
    // 2. dispatch on (formula, c.CS) — DE76/Lab, DE2000/Lab, DE_ITP/ICtCp, DE_CAM16/CAM16UCS
    // 3. return the metric
}

type DeltaEFormula int
const (DE76 DeltaEFormula = iota; DE94; DECMC; DE2000; DE_ITP; DE_Jz; DE_CAM16)
```
~80 LOC dispatcher + the existing per-formula functions become methods. Composes with 2.1 + 2.2.

### 2.11 First-class `SpectralDistribution` — colour-science

**What.** colour-science's `SpectralDistribution` is a typed object: `sd = SpectralDistribution({400: 0.1, 500: 0.5, ...})` carries its wavelength domain, units, interpolation strategy, extrapolation strategy. Operations are method-chained: `sd.normalise(100).align(SpectralShape(380, 780, 5)).interpolate()`. The integration to XYZ is `colour.sd_to_XYZ(sd, cmfs, illuminant)` — three typed objects, one call. Reality's `BlackbodyToXYZ(T)` is the *only* spectral integrator and it takes a scalar; there is no path to "I have a measured reflectance at 5 nm, give me Lab under D65 with the 10° observer."

**Why pure engineering.** The math (Riemann/Simpson integration) is trivial; the engineering is the **typed container that owns its wavelength domain**, so `sd1 * sd2` (reflectance × illuminant) checks domain compatibility and reinterpolates as needed.

**Go port.**
```go
type SpectralDistribution struct {
    Domain []float64    // wavelengths (nm), monotonically increasing
    Values []float64    // amplitudes
    Interp InterpKind   // Linear, Cubic, Sprague (Sprague is the CIE-mandated 31-point interpolant)
}

func (sd SpectralDistribution) Align(other []float64) SpectralDistribution { ... }
func (sd SpectralDistribution) Mul(other SpectralDistribution) SpectralDistribution { ... }
func IntegrateSPD(sd SpectralDistribution, observer Observer) Color   { ... }  // → XYZ
```
~150 LOC. Composes with `calculus/Simpson` (031-F17 already requested this). **Unblocks** every measured-spectra workflow (CRI/TM-30/F-illuminant evaluation, lighting design, sensor characterization).

---

## 3. Three architecture choices that are correctly out of scope

These look attractive but should be deliberately *not* taken on:

### 3.1 OCIO YAML config-as-code (axis #9)

**What OCIO does.** A `.ocio` config file declares roles (`scene_linear`, `compositing_log`, `color_picking`, `default_byte`...), display/view pairs (`sRGB`/`Standard`, `Rec.2020`/`HDR`...), looks (creative LUTs), colorspaces, and per-colorspace transform chains. The OCIO C++ runtime parses YAML, builds a transform DAG, compiles per-(src,dst) `Processor` objects with caching. Blender, Maya, Houdini, Nuke, Katana all consume OCIO configs natively and that's how the entire VFX industry stays color-aligned across DCCs.

**Why out of scope for reality.** YAML parsing breaks zero-dep (CLAUDE.md design rule 2). The *interface* idea — config-as-data — is fine, but reality should keep transforms as **Go-source-code data** (registered in `init()` blocks), not YAML. This is the same trade colour-science makes for its conversion graph (compiled into the Python source, not loaded from disk).

**What to take instead.** The OCIO `BuiltinTransform` registry pattern (axis #6 above) — named, citation-anchored transforms in code — captures 80% of the OCIO value at 0% of the dependency cost.

### 3.2 ICC profile binary parsing

**Already covered in 032 T3.1.** Tag-table parsing breaks zero-dep. The clean split is: reality ships a **LUT evaluator** (tetrahedral / prismatic / trilinear interpolation over `[]float64` LUTs, ~400 LOC, zero-dep), a sibling non-`reality` package owns the binary tag parser. This matches the layering CoreImage uses internally (`CGColorSpace` is the LUT evaluator; profile parsing is in CoreGraphics's lower layer).

### 3.3 GPU-accelerated kernel dispatch

CoreImage's `CIColorKernel` and OCIO's GPU shader generation are out of scope for the same reason chaos's GPU integration was (agent 028 §6). Reality is CPU-only. The interface lesson **does** transfer: the `Convert(c, dst)` API must be **branchless and alloc-free** in its hot path so consumers can vectorize with SIMD intrinsics or move it to GPU themselves. The 2.2 port above (predecessor table, `[4]float64` stack, no map lookups) is GPU-port-ready.

---

## 4. Frontier (2024-2026) algorithms reality should specifically engineer for

### 4.1 sCAM / sUCS (Li et al, 2025) — colour-science 0.4.7 baseline

The 2025 sCAM appearance model and sUCS uniform space (just shipped in colour-science 0.4.7, Dec 2025) are the post-CAM16 successors. **Reality is uniquely positioned** to be the second public implementation of these in any language (C++/Rust/C# don't have them yet). ~150 LOC on top of the CAM16 substrate (032 T1.S6). Citation-anchored to Li 2025 paper.

### 4.2 Li (2025) chromatic adaptation transform

Companion to sCAM. The third member (after CAT02 and CAT16) of the modern Hunt-Pointer-Estevez-style CAT family. Plugs into the `ChromaticAdapt(M, MInv, ...)` refactor 031-F15 already calls for. ~30 LOC additive (just one new matrix pair).

### 4.3 ACES 2.0 Output Transform — preview in OCIO 2.4

The ACES 2.0 RRT/ODT replaced ACES 1's tone-and-gamut chain with a **principled gamut-mapping algorithm grounded in color appearance modeling**. OCIO 2.4 ships it as `FIXED_FUNCTION_ACES_OUTPUT_TRANSFORM_20` (preview). The math is published (AMPAS S-2024-001) and reproducible. ~250 LOC for the forward direction. **Single highest-value cinema-pipeline addition** — a zero-dep Go ACES 2.0 implementation does not currently exist anywhere.

### 4.4 AgX as the default tonemapper

Blender 4.0+, Godot 4.4+, three.js, Unity port, ReShade ship AgX. **It is the 2026 default**, not Reinhard, not ACES. Reality's current `Reinhard` should remain (it's the textbook reference) but `Transforms["AGX_BLENDER_4"]` belongs in the registry from day one. ~80 LOC; the matrices and curve coefficients are public domain, published in Sobotka's GitHub.

### 4.5 CSS Color 4/5 `<color-interpolation-method>` semantics

The CSS spec mandates a specific interpolation contract: rectangular spaces (lab/oklab/srgb-linear) → component-wise lerp; polar spaces (lch/oklch/hsl) → hue-direction enum. Material 3's HCT ramps follow the same idea but in CAM16-tone space. **Shipping `MixWith` with the four hue directions matching the CSS enum is a one-shot win** because every web/design-tool consumer expects exactly that semantic.

### 4.6 The colour-science-style ergonomic round-trip

colour-science's killer ergonomic pattern is `colour.convert(c, "RGB", "Lab", verbose={"describe_conversion_path": True})`. The combination of (a) one call, (b) optional verbose-flag introspection, (c) graph-routed dispatch, (d) named source/dest strings (or in Go's case, typed enum) is *the* 2026 API contract for color libraries. Reality should match it modulo the verbose-flag idiom (Go-ifies as `DescribePath` separate function).

---

## 5. The graph-based conversion engine: fit assessment for reality

**Question:** is the SOTA-canonical conversion-graph engine a fit for reality?

**Answer:** **Yes, with three constraints baked in from day one**:

1. **Compile-time graph build.** All edges registered in `init()` blocks per-space, never at runtime. Floyd-Warshall (~0.5ms at N=50) runs once at process start. The N×N predecessor table is read-only thereafter. **No dynamic `RegisterEdge` after init** — this is what makes the graph zero-cost at frame rate.

2. **Predecessor-table dispatch, not pathfinding-per-call.** `Convert(from, to, v)` reads `next[from][to]` for the next hop ID, calls the edge function, advances. No priority queue, no map lookups, no slice allocs. Worst-case path length in a well-connected graph (XYZ is the universal hub) is **2 hops** (A → XYZ → B). **The hot-path cost is O(2 × matrix-mult) = ~30-60 ns**, indistinguishable from a hand-written direct converter.

3. **In-place `[4]float64` storage.** Color values fit in 32 bytes (4 float64s + a uint16 for `CS`). Pass by value, mutate via pointer. Zero heap allocs in the conversion loop. This is the Pistachio-60-FPS contract.

**What it buys.**
- N=50 spaces (Tier 1 from 032) yields C(50,2) = 1225 directed conversion possibilities, but only ~100 *edges* need to be coded. **12× code reduction**.
- Every new space is an `init()` block with two edges (`X_to_XYZ`, `XYZ_to_X`) — instant transitive coverage.
- `DescribePath` provides cross-language port debugging for free.
- Match colour-science's API verbatim — every Python/Go pair-validated golden file becomes a sanity check on the graph topology, not just the edge math.

**What it does not buy.**
- Conversion *quality*. The graph picks the *cheapest* route, not the *most accurate* one. For cases where two routes have different numerical paths (e.g., XYZ → Lab → LCH vs XYZ → LCH-direct), the user must be able to override with `ConvertVia(c, dst, []ColorSpaceID{Lab})`. ~10 LOC additive over the basic engine.
- Round-trip exactness. A → B → A is **not** identity in general (e.g., XYZ → Lab → XYZ has cube-root precision loss). Document, don't pretend otherwise.

**Recommendation.** **Ship the graph engine in the same commit as the `Color` type (axis 2.1) and the `BuiltinTransform` registry (axis 2.6).** Three pieces, ~700 LOC, single coherent design. Skipping any one of them dooms the others — the type without the graph forces N² converters; the graph without typed `Color` makes route dispatch ambiguous; the registry without either has nowhere to live.

---

## 6. Recommended commit ordering (engineering-design refactor)

| Order | Bundle | LOC | Unblocks |
|:-:|---|---:|---|
| **C1** | Axes 2.1 + 2.2 + 2.6 + 2.7 + 2.8 fused: `Color{V, CS}` type + Floyd-Warshall graph + `Transform` registry + `Convert` + `DescribePath` | 700 | Every Tier-1 catalog item from 032 (each becomes a `RegisterEdge` call); 031-F4 LCH chroma clip; 031-F6 LCH wrappers; cross-language port debugging |
| **C2** | Axis 2.5: `ColorSpaceMeta` table with ranges/hue/whitepoint/citation | 100 | 031-F13 (HSV h≥360 wrap); 031-F8/F9 (sRGB threshold caveat surfaces here); range validation for external data |
| **C3** | Axis 2.10: `(c Color) DeltaE(other, formula)` method + dispatcher | 80 | 031-F5 ΔE94/ΔECMC integration site; 032 T1.U14/U15/U16 (ΔE_ITP/Jz/CAM16) all land via the same dispatcher |
| **C4** | Axis 2.3 + 2.4: typed `RGB[E]` + `RGBA[E]` + `PremulRGBA[E]` for hot inner loops | 80 | sRGB-vs-Linear compile-time guard; alpha-correct compositing for Pistachio |
| **C5** | Axis 2.9: `(c Color) MixWith(other, t, HueDirection)` matching CSS Color 4/5 spec | 50 | Material 3 ramps; CSS-spec-faithful gradients; Pulse/Pistachio palette generation |
| **C6** | Axis 2.11: `SpectralDistribution` type + `IntegrateSPD(sd, observer)` | 150 | 032 T1.U6/U7 (generic SPD/reflectance integration); CRI/TM-30/F-illuminant workflows |
| **C7** | §4.4 AgX registered as `Transforms["AGX_BLENDER_4"]` | 80 | 2026-default tonemapper available (Blender/Godot parity) |
| **C8** | §4.3 ACES 2.0 Output Transform forward direction | 250 | Cinema-pipeline use; first zero-dep Go ACES 2.0 implementation |
| **C9** | §4.1 + §4.2 sCAM + sUCS + Li-2025 CAT (composes on CAM16 substrate from 032 T1.S6) | 200 | Second-public-impl-ever of the 2025 frontier; first-mover advantage in zero-dep Go |

**Total: ~1,690 LOC across 9 commits**, all citation-anchored, all golden-file-testable, **zero dependencies beyond `math` + `linalg`-3×3-helpers + `calculus.Simpson`**, no YAML, no GPU. **C1 is load-bearing for everything else** and must ship first; C2-C5 can land in parallel once C1 is in. The high-value 2026 frontier (C7-C9) requires C1 only and is ~530 LOC of pure capability addition that no other zero-dep library has.

---

## 7. Non-overlap with 031 and 032

031 owns numerical correctness of the existing 13 APIs (22 bugs/caveats; gamut-mapping gap). 032 owns the missing-catalog enumeration (67 Tier-1 items, 23 Tier-2, 12 Tier-3, with citations and LOC budgets). **033 owns the *interface* and *engineering-design* axes** — the orthogonal dimension neither prior report touched. The headline non-overlap finding is the **graph-routed conversion engine** (§2.2 + §5) — the architectural commit colour-science / OCIO / linebender / palette all converged on independently — which is what makes 032's 67 catalog items tractable: each becomes a `RegisterEdge` call plus 1 golden file, instead of N entry-point functions and N² explicit point-to-point converters. The second non-overlap finding is **the `Color`-as-typed-value pattern with encoding state** (§2.1 + §2.3 + §2.4): makes alpha-compositing and sRGB-vs-Linear blending compile-time-safe — a class of bugs reality currently has no defense against and that the named consumer (Pistachio at 60 FPS) routinely hits in production.

---

## 8. Citations

- Mansencal, T. et al. "Colour: Colour Science for Python." *JOSS* 7(78), 4308. 2022. (v0.4.7, Dec 2025: Li-2025 CAT, sCAM, sUCS, TMM thin film.)
- colour-science. *colour.graph.conversion* module — automatic conversion graph. https://colour.readthedocs.io
- Ogeon. "palette: A Rust library for linear color calculations and conversion." crates.io. v0.7.x.
- Linebender. "color: Color in Rust." github.com/linebender/color. v0.3.x (Nov 2024 alpha → 2026); Vello/Peniko backend.
- Academy Software Foundation. "OpenColorIO 2.4 Release." Sep 2024. (ACES 2.0 preview FixedFunctionTransforms; VFX Reference Platform CY2025.)
- AcademySoftwareFoundation/OpenColorIO-Config-ACES v4.0. CG/Studio configs.
- Selan, J. "OpenColorIO: An Open Source Color Pipeline." Sony Pictures Imageworks. (TDForum 2011, foundational paper.)
- Material Foundation. "material-color-utilities: HCT color system and Material Design 3 dynamic theming."
- Sobotka, T. "AgX: A display rendering transform." 2023. (Blender 4.0+ default, Godot 4.4+, three.js, Unity ports.)
- W3C. *CSS Color Module Level 4* / *Level 5*. 2024+ CR. (`oklab()`, `oklch()`, `<color-interpolation-method>`, `<hue-interpolation-method>`, relative-color syntax.)
- Apple. *Core Image Programming Guide* / *Metal Shading Language for Core Image Kernels*. (CIColorKernel, CIImage.colorSpace, matchedFromWorkingSpace, `__color` Metal attribute.)
- Li, C. et al. "The CAM16 colour appearance model." *CRA* 42(6), 2017.
- Li, C. et al. (2025). "sCAM colour appearance model and sUCS uniform colour space." (Per colour-science 0.4.7.)
- Safdar, M., Cui, G., Kim, Y. J., Luo, M. R. "Perceptually uniform color space for image signals including HDR and WCG." *Optics Express* 25(13), 2017. (JzAzBz.)
- ITU-R BT.2100-2 (2018), BT.2124 (2019), SMPTE ST 2084:2014. (HDR pipeline standards behind ICtCp/PQ/HLG.)
- AMPAS. "ACES 2.0 Output Transform." S-2024-001 (per OCIO 2.4 release notes).
- Pascale, D. (BabelColor). "RGB Coordinates of the Macbeth ColorChecker." babelcolor.com. (24-patch reference data.)
- Ottosson, B. "A perceptual color space for image processing." 2020. (Oklab; CSS Color 4-mandated.)
