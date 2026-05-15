# 144 | topology-api — API ergonomics of `topology/persistent`

**Scope.** API shape, not algorithms. 141 owns numerics, 142 owns
missing primitives, 143 owns SOTA tricks. Strictly the **public
surface of `topology/persistent/`** — how a v2 author or v1 consumer
trips on type names, signatures, struct fields. Yardstick: sibling
`prob/`'s `Distribution` interface pattern
(`prob/distribution.go:27-35`).

**Files read.** `topology/persistent/{doc,vr,barcode,bottleneck,
errors,persistent_test}.go` (1378 LOC); `prob/distribution.go` for
the interface idiom; `prob/distributions.go` for the free-function +
struct-method duality.

**v1 public surface (eleven symbols):** `Simplex`,
`Simplex.{Dim,Equal}`, `Filtration{Simplices,Times}`,
`Filtration.Len`, `Bar{Dim,Birth,Death}`,
`Bar.{Persistence,IsEssential}`, `VietorisRipsComplex(points,maxRadius,
maxDim)→(Filtration,error)`, `ComputeBarcode(filtration,maxDim)→
([]Bar,error)`, `BottleneckDistance(d1,d2,dim)→float64`, four `Err*`
sentinels. **No interfaces, no type aliases, no hooks.**

---

## 0. Headline

The v1 surface is **a one-pass minimal API that hard-codes every choice
into a positional argument or a struct field**. No injection seam for:
filtration kind, distance function, coefficient field, output-diagram
representation, error-reporting style. Six findings, ranked by **how
hard each bites the v2 author who adds Čech / α / witness / Z_p /
cohomology**:

1. **`VietorisRipsComplex` is named for the *only* filtration that
   exists.** When a Čech complex ships (`142 §T1.1`) the symbol either
   becomes vestigial (renamed → `Filtration` alongside `CechComplex`,
   `AlphaComplex` — breaking every consumer) or stays a fossil. Compare
   `prob.Distribution`: the abstraction is named for the abstraction,
   not for the first concrete type. **The v2-friendly rename happens
   pre-1.0 or never.** Ship a `FiltrationBuilder` interface (`§2`)
   *before* the second filtration kind lands, not after.

2. **`maxDim int` is passed twice — once to `VietorisRipsComplex`, once
   to `ComputeBarcode` — same name, same `{0, 1}` validation in two
   places.** `vr.go:175-177` and `barcode.go:61-63` are duplicate
   validators. The filtration carries enough metadata to know its own
   maxDim, but `Filtration` has no `MaxDim` field, so `ComputeBarcode`
   must re-validate. v2 will add `Coefficients FieldMod` and the
   barcode dimension cap can drift from the filtration's natural
   skeleton dimension. **Fix: `Filtration.MaxDim` field +
   `ComputeBarcode(filtration)` (no second arg).** See `§3`.

3. **`Bar.Death = math.Inf(+1)` is a magic-value sentinel for
   "essential class".** Every consumer site must remember
   `IsEssential()` before subtracting Birth from Death. Nothing in the
   type system stops `b.Death - b.Birth`. **Fix: `Bar.Death *float64`
   (nil = essential).** Idiomatic Go (`database/sql.NullFloat64`
   pattern). See `§4`.

4. **`pairwiseDistanceMatrix` (`vr.go:197-216`) hard-codes Euclidean
   L² with no extension hook.** Three of the package's own named
   consumers (`doc.go:30-41`) want non-Euclidean: RubberDuck wants
   Mahalanobis-corrected correlation distance `√(2(1−ρ_ij))` from a
   correlation matrix that already exists; Witness wants cosine on
   attention embeddings; Insights blast-radius and Tether import-graph
   want graph-shortest-path on a service/import graph (input is *not*
   a point cloud). **Fix: ship `Metric` callback + a
   `VietorisRipsFromDistances(d [][]float64, ...)` entry point.** See
   `§5`.

5. **`BottleneckDistance` returns a bare `float64` — no error.** Every
   other heavy public function returns `(T, error)`. Doc says "if d1
   and d2 disagree on essential-bar count … returns +Inf" — second
   magic-value sentinel in the same package. Worse: +Inf encodes two
   distinct conditions (essential mismatch *and* a legitimately giant
   finite bottleneck). A consumer asserting `result < threshold`
   cannot tell which fired. **Fix: `(float64, error)` with
   `ErrEssentialMismatch`.** See `§6`.

6. **No `Diagram` type wraps `[]Bar`.** `[]Bar` is the
   persistence-diagram representation throughout, forcing every
   per-dimension-query consumer to re-filter inline (`bottleneck.go:
   93-102` does it three times in a single function). **Fix: a
   `Diagram` struct** with `BarsByDim`, `EssentialByDim`, `MaxDim`,
   `MaxPersistence` — ~40 LOC, removes five duplicated loops. The
   `prob.Distribution` interface centralised PDF/CDF for exactly this
   reason. See `§7`.

**Single highest-leverage fix: #1 — rename and abstract before the
second filtration kind lands.** Every v2 plan in 142 (Čech / α /
witness / sparse-Rips / cubical / flag) will *force* this; doing it
now costs ~80 LOC and zero v1 behaviour change. Doing it after the
second filtration ships breaks every consumer call site.

---

## 1. Sibling pattern: `prob.Distribution`

`prob/distribution.go:27-35` ships a `Distribution` interface (`PDF` +
`CDF`); concrete `BetaDist` / `NormalDist` / `ExponentialDist` /
`UniformDist` implement it; free functions (`BetaPDF`, `NormalCDF`)
serve the zero-allocation hot path; constructors (`NewBetaDist`)
validate and return `nil` on invalid input; polymorphic helpers
(`KLDivergenceNumerical(p, q Distribution, ...)`) consume the
interface. The docstring (`:13-15`) calls this *"a Type 2 innovation
from Wave 1 cross-pollination: Haskell's Distribution typeclass and
C#'s IDistribution interface both converged on this shape
independently."*

`topology/persistent` has **no analogue** — it sits at the
"free-functions-only, no interface" stage `prob` was at pre-wave.
**What is the `Distribution` of topology?** Answer: at least three
interfaces (`FiltrationBuilder`, `Metric`, `CoefficientField`) plus
one new struct (`Diagram`).

---

## 2. Naming policy: name the abstraction, not the first concrete

When 142 §T1.1 lands, the v1 naming produces:

```go
VietorisRipsComplex(points, maxRadius, maxDim)
CechComplex(points, maxRadius, maxDim)
AlphaComplex(points, maxRadius, maxDim)         // gates on geometry/Delaunay
WitnessComplex(landmarks, witnesses, maxRadius, maxDim)
SparseRipsComplex(points, maxRadius, maxDim, ratio)  // 1 extra arg
```

Five top-level constructors with nearly identical positional args.
This is the anti-pattern `prob.Distribution` mitigated by lifting
polymorphism to an interface.

The API-shape fix: a `FiltrationBuilder` interface with `Build()
(Filtration, error)`; a `RipsBuilder` struct (`Points`, optional
`Distances` taking precedence, optional `Metric` with Euclidean
default, `MaxRadius`, `MaxDim`) implementing it; plus a convenience
`VietorisRips(points, maxRadius, maxDim) → RipsBuilder{...}.Build()`
that preserves v1 behaviour (alias `VietorisRipsComplex` →
`VietorisRips` for one release). The Builder seam is what every v2
filtration plugs into. **Cost: ~30 LOC scaffold + alias. Saves:
every v2 filtration ~50 LOC of boilerplate, zero break in consumer
code.**

---

## 3. `MaxDim` lives on the filtration, not the algorithm

`maxDim` appears in three places, validated in two:

- `VietorisRipsComplex(points, maxRadius, maxDim)` controls k-skeleton
  built. Validator at `vr.go:175-177`.
- `ComputeBarcode(filtration, maxDim)` controls bar dimensions
  returned. Validator at `barcode.go:61-63`.
- `BottleneckDistance(d1, d2, dimension)` — third param also `int`,
  *singular* dimension to filter to (different semantics from the
  other two but same type/name pattern).

`vr.go:175-177` and `barcode.go:61-63` are **identical
{0,1}-validation, two implementations, two `ErrInvalidMaxDim`
returns**.

Fix: (a) add `MaxDim int` to `Filtration`, set once at construction;
(b) `ComputeBarcode(filtration Filtration) ([]Bar, error)` — no
second arg; reads from filtration; (c) optional
`ComputeBarcodeWith(filtration, BarcodeOptions{Dim: ..., Field: ...,
Cohomology: bool})` for advanced consumers. **Cost: ~10 LOC. Saves:
maxDim-mismatch bugs in v2** (e.g. today vr.go:123 writes triangles
"even at maxDim = 1"; the corresponding "even at maxDim = 0"
asymmetry surfaces only when the discipline forces the question).

---

## 4. `Bar.Death = +Inf` for essential classes — the sentinel-float trap

`barcode.go:13-17`: `Bar { Dim int; Birth, Death float64 }`, with
`+Inf` Death meaning "essential". Three Go-idiomatic alternatives:
(1) `Bar.Death *float64` (nil = essential; stdlib pattern per
`sql.NullFloat64`; ~5 LOC; **recommended**); (2) private `Death` +
`Death() (float64, bool)` method (forces discriminator but breaks
struct-literal init that `persistent_test.go` does extensively;
**too invasive**); (3) discriminated-union via interface
(over-engineered for a sort-hot-path struct). The +Inf sentinel was
likely chosen because it threads through `BottleneckDistance`
cleanly, but that's a separate issue (`§6`). **Cost: ~5 LOC + test
rewrite of `Bar{..., Death: math.Inf(1)}` → `Bar{..., Death: nil}`.
Saves: every future consumer's "oh-I-forgot-IsEssential" bug.**

---

## 5. Metric is hard-coded — `Metric` callback **and** distance-matrix entry

`vr.go:91` calls `pairwiseDistanceMatrix(points)` (Euclidean L², no
hook). The single largest API-rigidity issue — three of the package's
own named consumers (`doc.go:30-41`) want non-Euclidean:

- **RubberDuck** — Mahalanobis-corrected `√(2(1−ρ_ij))` from a
  correlation matrix that already exists. Today must reconstruct
  embedded points to satisfy VR's signature.
- **Witness** — cosine on attention-vector embeddings.
- **Insights blast-radius / Tether import-graph** — shortest-path on
  a service/import graph (input is *not* a point cloud).

Ship two entries: `type Metric func(p, q []float64) float64`,
`VietorisRipsWith(points, metric, maxRadius, maxDim)`, and
`VietorisRipsFromDistances(dist, maxRadius, maxDim)` (square,
symmetric, zero-diagonal). `FromDistances` is what Ripser / GUDHI /
scikit-tda all expose. Existing `pairwiseDistanceMatrix` becomes the
body of the no-metric `VietorisRips`, which then calls
`FromDistances` on the freshly-computed matrix. Also dovetails with
**141 §1 twist/clearing** — both are matrix-side optimisations
benefiting from matrix exposure. **Cost: ~40 LOC. Saves: 3 named
consumers ~50-150 LOC each of point-cloud reconstruction; opens
validation against Ripser test fixtures (distance matrices
natively).**

---

## 6. `BottleneckDistance` → `(float64, error)`

`bottleneck.go:50` returns bare `float64`. Failure modes today —
essential-bar count mismatch (`bottleneck.go:55-59`); easy v2
additions: negative-persistence bar, NaN coordinate — all encoded as
`+Inf`. **Lossy:** a giant-but-finite bottleneck (`d_B = 1e10`,
perfectly valid) is indistinguishable from "malformed input". Fix
sibling-consistent: `(float64, error)` with `ErrEssentialMismatch`.
The `+Inf` semantic was inherited from C# RubberDuck (per
`doc.go:115-116`); C# lacks multi-return so it *had to* fold
"undefined" into a magic value. **Reality has multi-return; use it.**
Also enables a `BottleneckOptions` (e.g. `{P: 2}` for p-Wasserstein
per 142 §T1.4) without breaking the signature. **Cost: ~5 LOC + caller
fix. Saves: ambiguity bug the day a real consumer gets `1e10` back.**

---

## 7. `[]Bar` → `Diagram` wrapper

The persistence diagram is **the** output of TDA. Today it's an
unwrapped `[]Bar`. `bottleneck.go:51-52` and `:55-56` call
`filterFinite` and `countEssential` on every `BottleneckDistance`
call, each scanning the slice from scratch.

A `Diagram` struct (Bars + lazy-built `byDim` map) lifts this with
methods: `NewDiagram`, `BarsByDim`, `FiniteByDim`, `EssentialByDim`,
`MaxDim`, `MaxPersistence`, `Len`, `IsEmpty`. `ComputeBarcode`
returns `(Diagram, error)`. `BottleneckDistance` takes `Diagram`
instead of `[]Bar`. Internal use caches `byDim` after first build;
today's repeated linear filters become amortised O(1).

**Direct application of `prob.Distribution`'s lesson:** identify the
canonical output type, give it a method-rich API. `prob` did it via
interface; `topology` should do it via struct (Go has no obvious
interface for "persistence diagram" — the operations *are* the
struct). **Cost: ~80 LOC. Saves: ~30 LOC duplicated filter loops in
`bottleneck.go` + every consumer site, plus opens v2
`Diagram.Vectorise()` / `Landscape()` / `Image()` per 142 §T1.5.**

---

## 8. Smaller ergonomics issues

**8.1.** `Simplex []int` not `[]uint32`. Indices are non-negative by
construction; `[]int` lets a consumer pass `Simplex{-1, 2}`, and
`appendInt` (`barcode.go:197-216`) handles a negative case it should
never see. `[]uint32` is 50% smaller on 64-bit (matrix-reduction
columns are cache-resident). **Defer to pre-1.0 ratchet** — touches
every consumer.

**8.2.** `Simplex.Equal` (`vr.go:23-24`) assumes both operands are
pre-sorted. A consumer hand-rolling `Simplex{2, 1, 3}` silently gets
`false` against the sorted equivalent. Either ship `NewSimplex(verts
...int)` that sorts on construction, or have `Equal` sort-and-compare.
Option (a) also rejects duplicate-vertex `{1, 1, 2}`.

**8.3.** `simplexKey` (`barcode.go:170-195`) is internal but reinvents
allocation-free int-to-string. The v2 author who adds Mapper /
persistence-on-graphs *will* want to key simplices in a hash. Promote
to `Simplex.Key() string` exported.

**8.4.** Error sentinels lack `%w` wrapping discipline. `errors.go`
ships four `errors.New`; call sites return them bare. Works today;
the moment v2 wants "input clipping discarded N triangles", needs
`fmt.Errorf("...: %w", ...)` and the consumer side needs `errors.Is`.
**Forward-compatible doc fix:** add `// See errors.Is for matching.`
per sentinel. ~4 LOC.

**8.5.** `BottleneckDistance(d1, d2, dimension int)` — singular
`dimension`; the other two entries take `maxDim int` (maximum).
Different semantic roles, same type/name. **Rename →
`BottleneckDistance(d1, d2, atDim int)`** OR fold into
`BottleneckOptions{Dim: ...}` once the `(_, error)` promotion of `§6`
lands.

**8.6.** `Filtration.Len()` exists; `Filtration.Empty()` does not.
Tiny asymmetry; ~3 LOC.

**8.7.** Coefficient field is hardcoded F_2 with zero abstraction
surface for Z_p. Doc at `barcode.go:32-58` says "over F_2"; `symDiff`
(`barcode.go:254-280`) is the F_2 column-add. v2 wants Z_3 / Z_5
(142 §T2.6, required for Klein bottle H_1). Forward-compatible shape:
a `CoefficientField` interface — but **F_2 is a bit-vector, Z_p with
p > 2 is a (row, coefficient) sparse list**, so the data structure
diverges. **Recommend: do NOT abstract today.** Ship as
`ComputeBarcode(filtration) (Diagram, error)` and add
`ComputeBarcodeMod(filtration, p int) (Diagram, error)` when Z_p
lands. The "mod" suffix is the sibling-package convention (`prob` has
`PoissonPMF` not `Poisson` — types in the name).

---

## 9. Concrete v2-friendly API after `§§ 1–8`

```go
type Filtration struct {
    Simplices []Simplex
    Times     []float64
    MaxDim    int                       // §3
}
func (f Filtration) Len() int
func (f Filtration) Empty() bool        // §8.6

type FiltrationBuilder interface { Build() (Filtration, error) }   // §2

type Metric func(p, q []float64) float64                           // §5

type RipsBuilder struct {
    Points    [][]float64
    Distances [][]float64               // alternative to Points
    Metric    Metric                    // optional; default Euclidean
    MaxRadius float64
    MaxDim    int
}
func (b RipsBuilder) Build() (Filtration, error)

func VietorisRips(points [][]float64, maxRadius float64, maxDim int) (Filtration, error)
func VietorisRipsWith(points [][]float64, metric Metric, maxRadius float64,
    maxDim int) (Filtration, error)
func VietorisRipsFromDistances(dist [][]float64, maxRadius float64,
    maxDim int) (Filtration, error)

type Bar struct {
    Dim   int
    Birth float64
    Death *float64                      // §4: nil = essential
}
func (b Bar) Persistence() float64
func (b Bar) IsEssential() bool

type Diagram struct { Bars []Bar; ... }                            // §7
func NewDiagram(bars []Bar) Diagram
func (d Diagram) BarsByDim(dim int) []Bar
func (d Diagram) FiniteByDim(dim int) []Bar
func (d Diagram) EssentialByDim(dim int) []Bar
func (d Diagram) MaxDim() int
func (d Diagram) MaxPersistence(dim int) float64
func (d Diagram) Len() int

func ComputeBarcode(filtration Filtration) (Diagram, error)        // §3
func BottleneckDistance(d1, d2 Diagram, atDim int) (float64, error) // §6
```

**Net public-symbol delta:** +3 types (`Metric`, `RipsBuilder`,
`FiltrationBuilder`), +1 type (`Diagram`), 0 broken signatures
(today's `VietorisRipsComplex` aliases to `VietorisRips` for one
release). **Total ~280 LOC across vr/barcode/bottleneck + test
updates.** Smallest API change that makes v2 (142's tier-1) drop in
without further breakage. Largest immediate ergonomic win:
`Diagram` (`§7`). Largest v2-future-proofing win: `RipsBuilder`
(`§§ 2, 5`).

---

## 10. Two-line summary

`topology/persistent` v1 ships 11 public symbols that hard-code every
choice — filtration kind in the function name (`VietorisRipsComplex`),
metric in `pairwiseDistanceMatrix`, coefficient field in `symDiff`,
"essential" as a `+Inf` sentinel on `Bar.Death`, "undefined
bottleneck" as a `+Inf` return — with no `Diagram` wrapper, no
`Metric` callback, no `FiltrationBuilder` interface, and no shared
idiom with `prob.Distribution`. Six landable ergonomics fixes (~280
LOC total) — `RipsBuilder` + `Metric` callback +
`VietorisRipsFromDistances` entry, wrap `[]Bar` in a `Diagram`, move
`MaxDim` onto `Filtration`, change `Bar.Death` to `*float64`, promote
`BottleneckDistance` to `(float64, error)`, rename to `VietorisRips`
with alias — unblock every Tier-1 v2 addition listed in 142 without
breaking a single existing consumer call site.
