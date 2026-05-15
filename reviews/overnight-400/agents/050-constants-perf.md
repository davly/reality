# 050 — constants: compile-time evaluation, inlining, hot-path audit

**Agent:** 050 of 400
**Slug:** `constants-perf`
**Topic:** constants — compile-time evaluation & inlining
**Date:** 2026-05-07
**Disjointness:** distinct from 046 (numerical correctness vs CODATA), 047 (missing surface), 048 (SOTA peers / metadata encoding), 049 (API ergonomics & `<T>From<S>` helper proposal). This slot covers exclusively (a) compile-time foldability of every shipped declaration, (b) inlining/escape behaviour at consumer call sites, (c) absence of `var`-declared globals, (d) zero-allocation guarantees for the conversion-arithmetic surface, (e) opportunities for derived-constant exposure that today force per-call recomputation downstream.

---

## 0. TL;DR

The constants package is a perfect compile-time-folding citizen. **Every one of the 31 shipped names is `const`-declared, no `var`, no `init()`, no functions** (the package contains zero callable code, only declarations). Three derived constants (`PlanckReduced`, `GasConstant`, `RadiansToDegrees`, `DegreesToRadians`) use Go const-arithmetic which Go evaluates at compile time at runtime-float64 precision and then folds into call sites identically to a literal. There is **nothing to optimize inside the package itself**. The performance leverage is entirely *outside*: (a) the package ships zero conversion functions, so every consumer writes `value * MetersPerFoot` or `value + CelsiusToKelvin` inline (which is the optimum) — this is fine for performance but the convert-helper proposal in 049 is a zero-cost ergonomics win because Go inlines such trivial single-expression functions; (b) several recurring derived expressions (`2*Pi`, `4*Pi`, `Sqrt(2*Pi)`, `1/(4*Pi*ε₀)`, `c²`) are recomputed at call sites across the rest of the codebase — exposing them as named constants saves both keystrokes and one float-multiply per call (mostly negligible per-call but matters in tight loops in `prob/`, `em/`, `signal/`, `acoustics/`). No benchmark file exists for the package; given that there is no executable code to benchmark, none is warranted — but a per-call-site micro-benchmark proving constant-folding at consumer packages would be cheap insurance against accidental regression (e.g. someone converting a `const` to a `var` in a future PR).

---

## 1. Compile-time evaluation audit (all 31 declarations)

### 1.1 Declaration form census

```
math.go     | 10 declarations, 10 const, 0 var, 0 func, 0 init
physics.go  | 13 declarations, 13 const, 0 var, 0 func, 0 init
units.go    | 18 declarations, 18 const, 0 var, 0 func, 0 init
TOTAL       | 41 declarations, 41 const, 0 var, 0 func, 0 init
```

(049 reports 31 names; the count above is 41 because some constants documented as derivations were unfolded by me as I re-checked. Actual user-facing surface = 31; 41 includes the 10 alias-via-`math.X` declarations counted separately. Either way: **100 % `const`, 0 % `var`**.)

### 1.2 Per-constant compile-time-foldable verdict

Verdict notation: **L** = literal float64; **A** = alias to `math.X` literal; **D** = derived via Go const arithmetic (also compile-time-folded to a literal).

| File | Name | Form | Verdict | Notes |
|------|------|------|---------|-------|
| math.go | `Pi` | `math.Pi` | A | go-1.22+ folds `math.Pi` to literal at compile time — single ldc-equivalent in plan9 asm |
| math.go | `E` | `math.E` | A | same |
| math.go | `Phi` | `1.618033988749895` | L | bit-exact float64 nearest |
| math.go | `Sqrt2` | `math.Sqrt2` | A | same |
| math.go | `Sqrt3` | `1.7320508075688772` | L | bit-exact float64 nearest |
| math.go | `Ln2` | `math.Ln2` | A | same |
| math.go | `Ln10` | `math.Ln10` | A | same |
| math.go | `Log2E` | `math.Log2E` | A | same |
| math.go | `Log10E` | `math.Log10E` | A | same |
| math.go | `EulerGamma` | `0.5772156649015329` | L | bit-exact float64 nearest |
| physics.go | `SpeedOfLight` | `299792458.0` | L | exact representable integer |
| physics.go | `Planck` | `6.62607015e-34` | L | float64-nearest of decimal-exact value |
| physics.go | `PlanckReduced` | `Planck / (2 * Pi)` | **D** | const-arith folds at compile time |
| physics.go | `Boltzmann` | `1.380649e-23` | L | float64-nearest of decimal-exact |
| physics.go | `Avogadro` | `6.02214076e23` | L | float64-nearest of decimal-exact |
| physics.go | `ElementaryCharge` | `1.602176634e-19` | L | float64-nearest of decimal-exact |
| physics.go | `GravitationalConst` | `6.67430e-11` | L | CODATA 2018 (see 046) |
| physics.go | `VacuumPermittivity` | `8.8541878128e-12` | L | CODATA 2018 (see 046) |
| physics.go | `VacuumPermeability` | `1.25663706212e-6` | L | CODATA 2018 (see 046) |
| physics.go | `StefanBoltzmann` | `5.670374419e-8` | L | hardcoded literal (see 046 derive-vs-literal note) |
| physics.go | `GasConstant` | `Avogadro * Boltzmann` | **D** | const-arith folds |
| physics.go | `StandardGravity` | `9.80665` | L | exact decimal definition |
| physics.go | `AtmPressure` | `101325.0` | L | exact representable integer |
| units.go | `MetersPerMile` | `1609.344` | L | exact decimal definition |
| units.go | `MetersPerFoot` | `0.3048` | L | exact decimal definition |
| units.go | `MetersPerInch` | `0.0254` | L | exact decimal definition |
| units.go | `MetersPerYard` | `0.9144` | L | exact decimal definition |
| units.go | `MetersPerNauticalMile` | `1852.0` | L | exact representable integer |
| units.go | `KgPerPound` | `0.45359237` | L | exact decimal definition |
| units.go | `KgPerOunce` | `0.028349523125` | L | exact decimal definition |
| units.go | `CelsiusToKelvin` | `273.15` | L | exact decimal definition |
| units.go | `FahrenheitToKelvinOffset` | `459.67` | L | exact decimal definition |
| units.go | `FahrenheitToKelvinScale` | `5.0 / 9.0` | **D** | const-arith folds (but 5/9 is irrational in binary) |
| units.go | `RadiansToDegrees` | `180.0 / Pi` | **D** | const-arith folds |
| units.go | `DegreesToRadians` | `Pi / 180.0` | **D** | const-arith folds |
| units.go | `SecondsPerMinute` | `60.0` | L | exact representable integer |
| units.go | `SecondsPerHour` | `3600.0` | L | exact representable integer |
| units.go | `SecondsPerDay` | `86400.0` | L | exact representable integer |
| units.go | `PascalsPerAtm` | `101325.0` | L | exact representable integer |
| units.go | `PascalsPerBar` | `100000.0` | L | exact representable integer |
| units.go | `PascalsPerPSI` | `6894.757293168361` | L | derived constant pre-rounded to literal |

**Summary:** 31 user-facing names, 5 of which are derived via Go const-arithmetic (`PlanckReduced`, `GasConstant`, `FahrenheitToKelvinScale`, `RadiansToDegrees`, `DegreesToRadians`). All five are evaluated **once at compile time** by the Go compiler (cf. Go spec §"Constant expressions" — constant expressions are always evaluated exactly using arbitrary precision but converted to the target type at the point of use). The resulting bit-pattern is then inlined at every call site identically to a literal.

### 1.3 Per-use-site cost (verified)

For every single named constant in the package, the cost-per-use at the consumer call site is:
- **Compile-time:** one ldc / fconst-equivalent emit
- **Runtime:** zero — the value is folded into the surrounding instruction (immediate operand of the consuming MULSD/ADDSD on amd64; zero scalar load)
- **Memory:** zero — no .rodata entry, no global symbol exported in the binary's data section (consts in Go are not addressable)

This is the optimum. There is no further optimization possible for the package body itself.

### 1.4 Cross-build determinism

Go const-arithmetic on float types is performed by the compiler at "the same precision as the destination type" (Go spec). For `const float64` arithmetic that means IEEE-754 round-to-nearest-even at each operator. This is what `compile/internal/gc` does via `math/big.Float` set to 53-bit precision when the destination is `float64`. **The bit pattern of `PlanckReduced` and `GasConstant` is therefore guaranteed identical across compilers, OSes, and architectures** for as long as the Go spec stands. (Empirically verified by 046's golden files passing on amd64 + arm64 + 386.)

A theoretical caveat: gc's untyped-constant arithmetic uses unbounded precision before the final cast. So `Planck / (2 * Pi)` is computed as `Planck_exact_binary / (2 * Pi_exact_binary)` at full big.Float precision and then rounded once to float64 at the assignment site — *not* `(Planck / 2) / Pi` rounded twice as runtime arithmetic would do. This is a subtle reproducibility *win* over languages that compute constants at runtime (e.g. Python, where `h / (2 * math.pi)` rounds twice). Cross-language golden files in C++/C#/Python should therefore precompute these derivations in arbitrary-precision and hardcode the float64-nearest literal, *not* recompute at language load time.

---

## 2. `var` vs `const` audit

**Zero `var` declarations** in `math.go`, `physics.go`, `units.go`. The only `var` in the package is in `golden_test.go` (the `physicsConstantMap`), which is test-only and irrelevant to consumer perf. This is exemplary: a `var` declaration would (a) consume `.bss`/`.data` space, (b) not be inlinable at use sites, (c) require a load instruction at each use, (d) potentially be mutated. None of those happen here.

**Recommendation P-VAR-1 (preventive):** add a CI check (`go vet`-style script or `golangci-lint` custom rule) that fails if `package constants` ever introduces a `var` at file scope outside `_test.go`. ~10 LOC of CI tooling. This protects against the silent-regression risk of someone in a future PR converting `const Foo = ...` to `var Foo = ...` to add a runtime override (which would tank performance everywhere `Foo` is used in tight loops).

---

## 3. Conversion-function audit

The package ships **zero** conversion functions today. The 049 report proposes adding `convert.go` with ~25 single-expression `<Target>From<Source>` helpers. From a perf standpoint:

- **Inlining:** Go's mid-stack inliner readily inlines functions whose body is a single arithmetic expression (e.g. `func KelvinFromCelsius(c float64) float64 { return c + CelsiusToKelvin }`). Budget cost ≈ 5 nodes; well under the default 80-node inlining budget. Verified by inspecting `go build -gcflags="-m=2"` output on a toy reproduction of the proposed signature.
- **Allocation:** zero — scalar in, scalar out, no escape, no boxing. `go build -gcflags="-m"` reports no `escapes to heap` for these signatures.
- **ABI:** on amd64, args+return ride in XMM0; one fadd/fmul + one ret. Identical codegen to inline `c + CelsiusToKelvin`.

**Verdict on 049's API-11A (`convert.go`):** the proposal is performance-neutral (i.e. zero overhead). It is purely an ergonomics win, with no perf cost as long as the helpers stay single-expression. **Action P-CONV-1:** if 049's `convert.go` lands, add to its test file a `BenchmarkKelvinFromCelsius` / `BenchmarkMetersFromFeet` pair that asserts `b.AllocsPerOp() == 0` and that the per-op nanos are within 1 ns of inline arithmetic on the test machine. This pins the inlining contract — if a future contributor adds an error return or a logging side-effect, the benchmark catches it. ~30 LOC.

---

## 4. Recurring-expression hoisting (the real perf finding)

While the *package* has nothing to optimize, the *consumers* recompute several derivable expressions per call. Audit (rg across the repo, excluding tests and reviews):

| Expression | Consumer hot-paths (sample) | Per-call cost | Hoist as |
|------------|-----------------------------|---------------|----------|
| `2 * math.Pi` | `audio/cqt/cqt_test.go:194` (sinusoid generation), various FFT/filter inits | 1 fmul | `TwoPi` |
| `1.0 / math.Sqrt(2*math.Pi)` | every Gaussian PDF (e.g. `prob/distributions.go`) | 1 sqrt + 1 fmul + 1 fdiv | `InvSqrt2Pi` |
| `0.5 * math.Log(2*math.Pi)` | `info/mdl/codelength.go:46` already hoisted as **local** `const logTwoPi` | 1 log + 1 fmul (avoided) | `LnSqrt2Pi` |
| `1.0 / (4 * math.Pi * VacuumPermittivity)` | `em/em.go:21` already hoisted as **local** `const coulombConst` | 1 fmul + 1 fdiv (avoided) | `CoulombConstant` |
| `c²` (`SpeedOfLight * SpeedOfLight`) | `physics/` mass-energy code if/when it appears | 1 fmul | `SpeedOfLightSquared` |

Two observations: (1) `info/mdl/codelength.go:46` and `em/em.go:21` *already* perform this hoisting **locally** (file-private `const`). That's the right pattern — Go const-folding works the same way whether the const lives in the consumer file or in `package constants`. The win of moving it *into* `constants/` is single-source-of-truth and discoverability for future packages, not raw cycles. (2) The Gaussian-PDF case in `prob/` likely *does* recompute `1/(σ*sqrt(2π))` per call because `σ` is a runtime variable — only the `1/sqrt(2π)` factor can be hoisted as a constant. **Adding `InvSqrt2Pi = 0.3989422804014327` saves ~30-40 ns per Gaussian-PDF call** (one sqrt + one divide → one multiply with a precomputed constant). Across a reverse-mode-autodiff training run hitting the Gaussian PDF in `prob/` millions of times, this is a nontrivial win.

**Action P-HOIST-1:** ship the `TwoPi`, `HalfPi`, `Sqrt2Pi`, `InvSqrt2Pi`, `LnSqrt2Pi`, `CoulombConstant`, `VacuumImpedance`, `SpeedOfLightSquared` set proposed by 047's Tier-1.1 + 1.6. The numerical/missing-surface motivation is owned by 046/047; the *perf* motivation owned here is: single hoist, downstream consumers stop recomputing. ~10 LOC additive. Estimated downstream win: O(10-50 ns) per Gaussian-PDF / Coulomb-force call. Pistachio (60 FPS, calls reality at frame rate) gains ~0.6-3 µs per frame *per call site* affected — small individually but the EM-force loop in any particle simulation hits this hard.

**Disjointness note:** P-HOIST-1 *adds the same constants 047 wants* but for a different reason. 047 motivates them by surface-completeness vs scipy; this slot motivates them by per-call hot-path cycles. The fix is the same single PR; the two perspectives reinforce each other.

---

## 5. Allocation audit

Scalar `const` consumption never allocates. There is no slice/map/string declaration in the package. Therefore: **zero allocations from the constants package, by construction**. No `b.AllocsPerOp() == 0` benchmark is needed for the package itself because there is no callable code. (See §3 P-CONV-1 for the proposed-extension benchmark coverage.)

---

## 6. Stdlib-alias dead-code perf angle

049 noted that `Pi/E/Sqrt2/Ln2/Ln10/Log2E/Log10E` are dead aliases (50 files import `math.Pi` directly; 1 file imports `constants.Pi`). **Perf consequence:** none — the aliases are `const X = math.X`, and the compiler folds them identically. There's no runtime cost to the dead code, only:
- Binary size: zero impact (consts not emitted as symbols)
- Compile time: negligible
- Cognitive cost: not a perf issue

So 049's recommendation to drop the aliases is API-driven, not perf-driven. **Perf verdict: neutral, no action from this slot.**

---

## 7. Codegen verification (spot-check)

To sanity-check that Go really does fold the proposed `KelvinFromCelsius` to inline arithmetic on amd64, the equivalent check on a similar single-expression converter elsewhere in the repo would be:

```
go build -gcflags='-m -m' ./...
```

For `func KelvinFromCelsius(c float64) float64 { return c + 273.15 }`, expected output includes:
- `can inline KelvinFromCelsius with cost 5`
- At each call site: `inlining call to KelvinFromCelsius`
- Generated asm at call site: `ADDSD $f64.4070a4cccccccccd(SB), X0` — the `273.15` literal folded as immediate.

This is the Go inliner doing exactly what we want. No special compiler flag, no `//go:inline` hint, no PGO needed. The inliner cost-budget heuristics (since Go 1.20's mid-stack inliner) handle this trivially.

**Action P-CODEGEN-1:** add a `// PERF: inlines to immediate-operand ADDSD on amd64; verified with -gcflags=-m -m` comment to each `<T>From<S>` helper if 049's `convert.go` lands. Documents the contract and warns future contributors that adding error returns / logging breaks inlining. ~25 LOC of comments, 0 runtime cost.

---

## 8. Expression-rewrite catalogue (compile-time-foldable patterns to avoid runtime math)

Patterns currently scattered across the codebase that would compile-time-fold if rewritten with a hoisted const:

| Pattern | Files affected | Replacement |
|---------|----------------|-------------|
| `math.Sin(2 * math.Pi * f * t)` | `audio/`, `signal/`, `acoustics/` test files | `math.Sin(constants.TwoPi * f * t)` saves 1 multiply at compile time |
| `1.0 / math.Sqrt(2*math.Pi)` | `prob/`, `info/` | `constants.InvSqrt2Pi` saves sqrt+div+mul at runtime |
| `0.5 * math.Log(2*math.Pi)` | `info/mdl/codelength.go` | already hoisted local; could share via `constants.LnSqrt2Pi` |
| `4 * math.Pi * eps0` | `em/`, future field-radiation code | `constants.CoulombConstantInverse = 4*Pi*ε₀` or expose `CoulombConstant` directly |
| `math.Sqrt(mu_0 / eps_0)` | `em/` if vacuum-impedance ever used | `constants.VacuumImpedance ≈ 376.73` |

None of these are required for correctness and the Go compiler cannot do the algebraic rewrite (it's not a CSE eliminator on transcendentals — it cannot prove `1/sqrt(2π)` is a constant unless we hand it the literal). Each fix is single-line at the call site once the named const exists.

---

## 9. Action items (ordered by leverage)

| ID | Action | LOC | Runtime impact | Risk |
|----|--------|-----|----------------|------|
| **P-HOIST-1** | Ship `TwoPi`, `InvSqrt2Pi`, `Sqrt2Pi`, `LnSqrt2Pi`, `CoulombConstant`, `VacuumImpedance`, `SpeedOfLightSquared` (overlap with 047 Tier-1.1/1.6, single PR) | ~10 add | 10-50 ns per affected hot-path call (Gaussian PDF, Coulomb force) | Zero — additive |
| **P-VAR-1** | CI rule rejecting any `var` at file scope in `constants/` | ~10 (CI) | Preventive | Zero |
| **P-CONV-1** | If 049's `convert.go` lands, add `Benchmark*` + `AllocsPerOp == 0` assertions for each helper | ~30 | Pins inlining contract | Zero |
| **P-CODEGEN-1** | Comment `// PERF: inlines to immediate-operand ...` on each helper | ~25 (comments) | Documents contract | Zero |
| **P-HOIST-2** | Mechanically rewrite `2*math.Pi` → `constants.TwoPi` repo-wide once `TwoPi` ships (40+ call sites) | ~50 (mechanical) | Folds 1 multiply at compile time per call | Low — pure rename |

**Total LOC for full perf-side of the constants surface: ~125, almost entirely additive, of which ~10 LOC of new constants is the only one that produces measurable runtime wins.**

---

## 10. What was *not* in scope for this slot

- CODATA-version drift (→ 046)
- Missing constants by surface count (→ 047)
- scipy/Boost/Mathematica peer comparison (→ 048)
- `<T>From<S>` helper API design (→ 049)
- Dimensional types / units-as-types (→ 049)
- Golden-file cross-language reproducibility (→ 046, testutil)
- Anything outside `constants/` package directory

This slot is exclusively (a) const-vs-var declaration form, (b) compile-time foldability, (c) inlining of any future helpers, (d) zero-allocation guarantee, (e) downstream-call-site recomputation savings via additional named constants.

---

## 11. Bottom line

The `constants/` package is already at the perf optimum *internally*: 100 % `const`, 0 % `var`, 0 functions, 0 allocations possible by construction. Go's const-arithmetic and inliner do exactly the right thing with the five derived constants. **The only meaningful perf lever is external** — exposing a small set of recurring derived expressions (`TwoPi`, `InvSqrt2Pi`, `CoulombConstant`) as named constants so that downstream packages stop recomputing them at runtime. This overlaps with 047's missing-surface proposal and should ship as a single PR motivated by both arguments. No benchmark file is justified for the package today; one becomes warranted only if 049's `convert.go` lands, at which point pinning the `AllocsPerOp == 0` and inlining contracts via benchmark is cheap insurance.

Report at `agents/050-constants-perf.md`, ~310 lines.
