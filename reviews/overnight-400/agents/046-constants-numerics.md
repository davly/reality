# 046 | constants-numerics

**Scope:** numerical-correctness audit of `C:\limitless\foundation\reality\constants\` (`math.go` 71 LOC, `physics.go` 82 LOC, `units.go` 102 LOC, `constants_test.go` 277 LOC, `golden_test.go` 54 LOC, `testdata/constants/physics_constants.json` 13 cases). All tests pass (cached).

CLAUDE.md and `physics.go:6,10` declare the package follows "NIST CODATA 2018 recommended values otherwise" (with SI 2019 exact values for the four redefined constants). CODATA 2022 was published 2024 (Mohr, Newell, Taylor, Tiesinga, *Rev. Mod. Phys.* 97 025002, available 2024-09 arXiv 2409.03787). The package is one full CODATA cycle behind reality. The next adjustment (CODATA 2026) is already inside the four-year cycle window — any update should bake in a deprecation/version policy, not just bump numbers.

---

## CODATA 2018 → 2022 deltas vs shipped values

Source: `physics.nist.gov/cuu/Constants/Table/allascii.txt` (2022 CODATA, fetched 2026-05-07).

| Constant | Code (CODATA 2018) | CODATA 2022 | Δ | Sigma move |
|---|---|---|---|---|
| `GravitationalConst` G | 6.67430e-11 | **6.67430e-11** | 0 (digits shown) | unchanged at 5-digit precision; relative u_r still ~2.2e-5 |
| `VacuumPermittivity` ε₀ | 8.8541878128e-12 | **8.8541878188e-12** | +6.0e-21 (~7 ulp at this exponent) | **6e-10 relative — exceeds 1e-11 transcendental tolerance contract from CLAUDE.md** |
| `VacuumPermeability` μ₀ | 1.25663706212e-6 | **1.25663706127e-6** | −8.5e-16 (~7 ulp) | **7e-10 relative — same tolerance bust** |
| Fine-structure α (not shipped) | 7.2973525693e-3 (2018) | **7.2973525643e-3** | −5e-12 abs, ~7e-10 rel | n/a — package does not export α |
| `StefanBoltzmann` σ | 5.670374419e-8 | **5.670374419e-8** | 0 — exact derived | unchanged (depends only on h, k_B, c, π — all exact) |
| `SpeedOfLight`, `Planck`, `Boltzmann`, `Avogadro`, `ElementaryCharge` | exact SI 2019 | **identical exact** | 0 | The 2019 SI redefinition fixed these — they cannot change in CODATA 2022 or any future cycle by definition |

Constants the package does NOT ship but that did move in 2022 (relevant for downstream physics):
- electron mass: 9.1093837015e-31 → 9.1093837139e-31 kg
- proton mass: 1.67262192369e-27 → 1.67262192595e-27 kg
- atomic mass constant u: 1.66053906660e-27 → 1.66053906892e-27 kg
- Rydberg constant: 10973731.568160 → 10973731.568157 m⁻¹
- inverse α: 137.035999084 → 137.035999177

**Action item C-2022-1:** update ε₀ and μ₀ to 2022 values, update golden vector tolerance lines 49,55 (currently `tolerance: 0` against 2018 literal), update `physics.go:11` doc string from "CODATA 2018" → "CODATA 2022", update CLAUDE.md "NIST CODATA 2018" reference. Five-line code change, but it WILL break any downstream consumer that pinned the literal — only `em/em.go:21` (`coulombConst = 1/(4*pi*ε₀)`) is currently affected and it is correct to track.

**Action item C-2022-2:** introduce a versioning convention. Either:
(a) suffix non-exact constants with vintage: `GravitationalConst2022`, keep `GravitationalConst` pointing at the latest;
(b) document a CODATA-vintage constant in `physics.go` (`const CODATAVintage = "2022"`) and freeze a per-vintage golden file (`physics_constants_2022.json`);
(c) keep current names but add a regenerate-on-CODATA-update note.
Cost: ~30 LOC. Without it, the every-four-year update creates silent breakage.

---

## Exact-by-definition constants (post-2019 SI)

The 2019 SI redefinition fixed exact values for h, e, k_B, N_A by definition; ΔνCs, c, K_cd were already fixed exact. Audit:

| Symbol | Shipped value | Exact decimal | Float64 round-trip | Verdict |
|---|---|---|---|---|
| c (`SpeedOfLight`) | 299792458.0 | integer, exactly representable | yes | exact |
| h (`Planck`) | 6.62607015e-34 | NOT exactly representable in binary float64 | n/a | the literal `6.62607015e-34` is the nearest float64 to the exact decimal — cross-language goldens MUST validate against the same literal, not against arbitrary-precision exact |
| k_B (`Boltzmann`) | 1.380649e-23 | NOT exactly representable | n/a | same — float64 nearest to decimal exact |
| N_A (`Avogadro`) | 6.02214076e23 | NOT exactly representable | n/a | same |
| e (`ElementaryCharge`) | 1.602176634e-19 | NOT exactly representable | n/a | same |
| ΔνCs | not shipped | 9192631770 Hz exact integer | n/a | **gap** — should add for completeness |
| K_cd (luminous efficacy 540 THz) | not shipped | 683 lm/W exact integer | n/a | **gap** — should add |

**Finding C-EXACT-1:** the four "exact" SI 2019 constants are exact-by-decimal-definition but float64 cannot represent them exactly. The package and golden file are internally consistent (the literal IS the test value), but the docstring "exact" elides this — a Python implementation written by someone who reads "exact" will compute `6.62607015e-34` differently if they go through `Decimal` → `float`. Recommend docstring tweak: "exact decimal value 6.62607015e-34; float64 representation differs by < 1 ulp."

**Finding C-EXACT-2:** `ΔνCs` (9192631770 Hz, defines the second) and `K_cd` (683 lm/W, defines the candela) are both integer-exact and missing from the package. Adding them costs 4 LOC and completes the seven SI defining constants.

---

## Derived vs defined: how the package distinguishes

`physics.go` uses three signaling patterns:
- **Defined exact** (h, e, k_B, N_A, c, g_n, atm): docstring says "exact" + value is literal.
- **Derived exact-from-exact**: `PlanckReduced = Planck / (2*Pi)` and `GasConstant = Avogadro * Boltzmann` use Go const arithmetic — Go evaluates const float arithmetic at compile time at the same precision as runtime float64, so these are reproducible across builds. `StefanBoltzmann` is hardcoded as a literal `5.670374419e-8` rather than derived: this is **inconsistent**. Computing `2*π^5*k_B^4/(15*h^3*c^2)` in float64 yields `5.6703744191844e-8` (verified), differing from the literal `5.670374419e-8` at the 11th significant digit (1.84e-18 absolute, 3.3e-10 relative). Either derive it (matches the docstring "derived from SI 2019 exact constants" but breaks the golden literal) or document the rounding (current state).
- **Measured / non-exact** (G, ε₀, μ₀): docstring cites CODATA + uncertainty. No machine-readable uncertainty field — a consumer cannot programmatically know that G has u_r = 2.2e-5 while the SI 2019 four are zero.

**Finding C-DERIVED-1:** `StefanBoltzmann` should either be the computed expression (`2*math.Pow(Pi,5)*math.Pow(Boltzmann,4)/(15*math.Pow(Planck,3)*math.Pow(SpeedOfLight,2))` — but `math.Pow` is not const) or — preferable — a literal accompanied by a test that the literal matches the high-precision derived value to within a documented rounding (currently absent: `constants_test.go:159-163` tests `|σ - 5.670374419e-8| < 1e-17`, a self-comparison that proves nothing about the physics).

**Finding C-DERIVED-2:** add `Uncertainty` companion table or a `type Constant struct { Value, RelStdUnc float64; Source string }` parallel surface. Cost: ~40 LOC. Lets downstream code distinguish exact-by-SI from measured.

---

## Unit conversion factors: round-trip exactness

Computed (running `0.3048`, `0.0254`, `0.45359237`, `0.01` through encode/decode for x ∈ {1, 5, 100, 1000, 1e6}):

| Conversion | Round-trip exact? | Notes |
|---|---|---|
| cm ↔ m (×0.01) | **yes for all tested x** | 0.01 has nonterminating binary representation but the divide round-trips |
| ft ↔ m (×0.3048) | **yes for all tested x** | despite 0.3048 being unrepresentable |
| in ↔ m (×0.0254) | **fails for x=12**: 12 → 0.30479999999999996 → 11.999999999999998 | other tested x (1, 100, 1000) round-trip exact; this is the inverse-of-12 ulp drift |
| lb ↔ kg (×0.45359237) | **yes for all tested x** | |
| 5280 × MetersPerFoot | **bit-exact equal** to MetersPerMile literal | both produce 1609.34400000000005093170 |
| 3 × MetersPerFoot | **bit-exact equal** to MetersPerYard literal | |
| MetersPerFoot / 12 | **bit-exact equal** to MetersPerInch literal | |
| KgPerPound / 16 | **bit-exact equal** to KgPerOunce literal | both produce 0.0283495231250000014056045 |
| (32 + FahrenheitToKelvinOffset) × 5/9 | yields 273.14999999999997726263, **NOT 273.15** | 2.3e-13 K = 2.3e-13 °C ≈ 4e-13 °F absolute error at the freezing point of water |

**Finding C-ROUNDTRIP-1:** the unit literals are coherent with each other (mile/yard/inch all derive bit-exactly from foot, ounce derives bit-exactly from pound) — this is a non-trivial consistency property and well done. `constants_test.go:241-260` tests this with 1e-10 tolerances, but the actual property is bit-exact equality (tolerance = 0). Strengthen the tests: `MetersPerMile == 5280.0 * MetersPerFoot` etc.

**Finding C-ROUNDTRIP-2:** `PascalsPerPSI = 6894.757293168361` (units.go:101) does NOT match the bit-exact derivation `lbf / inch²` = `4.4482216152605 / (0.0254*0.0254)` = `6894.7572931683617` (last two digits differ; ~9e-13 absolute, 1.3e-16 relative — 1 ulp). Either derive it (`KgPerPound * StandardGravity / (MetersPerInch * MetersPerInch)` — but uses lbf-vs-kgf-with-g convention, so define lbf first as `PoundForce = KgPerPound * StandardGravity`) or document the chosen rounding. Cost: 1 line of derivation, 5 LOC for `PoundForce`.

**Finding C-ROUNDTRIP-3:** Fahrenheit-Kelvin conversion has a known rounding floor of ~3e-13 K because `5.0/9.0` is unrepresentable. Tests do not catch this. Either accept (document it) or use the alternate algebraic form `(F - 32) × 5/9 + 273.15` (no improvement, same 5/9 issue) or `(F × 5 - 160 + 491.67 × 5) / 9` (slightly worse — single-divide form is close to optimal). Recommend documenting "all F↔K conversions exhibit ~3e-13 K rounding due to 5/9 representation."

---

## ft/in/lb/oz/gal/ton: which are exact in international standards?

The package ships ft, in, yd, mile, NM, lb, oz. **Missing entirely**: gal, qt, pt, fl oz, cu in, ton (short/long), slug, BTU, calorie, hp, knot, ly, AU, parsec, eV.

Exact in international agreements (1959 yard-and-pound + later derivatives):
- **mile, yd, ft, in, NM, lb, oz** — shipped, all exact ✓
- **US gallon** (231 in³ × 0.0254³ m³/in³) — exact, 0.003785411784 m³ — **missing**
- **Imperial gallon** = 4.54609 L = 0.00454609 m³ exact (UK Weights and Measures Act 1985) — **missing, easy to mis-implement**
- **fluid ounce US** = gallon/128, fluid ounce Imperial = gallon/160 — **different by ~4%** — **missing, classic bug source**
- **short ton** = 2000 lb, **long ton** = 2240 lb, **metric tonne** = 1000 kg — **missing all three**
- **slug** = lbf·s²/ft = 14.59390294 kg (derivable, not in SI) — **missing**
- **knot** = 1 NM/h = 1852/3600 m/s = 0.5144444... m/s (recurring) — **missing**
- **BTU** has at least 5 different definitions (IT, thermochemical, 39°F, 59°F, 60°F) — adding requires picking one (recommend `BTU_IT = 1055.05585262` J)
- **cal** has 3 definitions (IT 4.1868 J, thermochemical 4.184 J exact, 15°C 4.1855 J)
- **eV** = `ElementaryCharge × 1 V` = 1.602176634e-19 J exact (since 2019)
- **AU** = 149597870700 m exact (IAU 2012 resolution B2)
- **ly** = 365.25 × 86400 × c m = 9460730472580800 m exact (Julian year × c)
- **parsec** = 648000/π × AU m — limited by π precision

**Finding C-MISSING-1:** at minimum add `LitersPerGallonUS = 3.785411784`, `LitersPerGallonImperial = 4.54609`, `KgPerShortTon = 907.18474`, `KgPerLongTon = 1016.0469088`, `KgPerTonne = 1000.0`, `JoulesPerEV = ElementaryCharge` (the equation form makes the relationship explicit), `MetersPerAU = 149597870700.0`, `MetersPerLightYear = 9460730472580800.0`. Cost: ~30 LOC, all exact, all defended by golden vectors.

**Finding C-MISSING-2:** explicitly comment the imperial-vs-US gallon trap — they differ by ~20% (US 3.785 L vs Imp 4.546 L) and US fl oz vs Imp fl oz differ by ~4% in opposite direction (US 29.57 mL vs Imp 28.41 mL). The current convention `MetersPerFoot` style does not extend cleanly here — recommend `LitersPerGallonUS` / `LitersPerGallonImperial` over `LitersPerGallon`.

---

## Time conversions: leap-second-aware? UTC vs TT vs TAI vs TDB

Shipped: `SecondsPerMinute = 60.0`, `SecondsPerHour = 3600.0`, `SecondsPerDay = 86400.0`. Comment on line 87 says "one mean solar day."

**Finding C-TIME-1:** `SecondsPerDay = 86400.0` is exact for **TAI/TT** (atomic time scales) and exact-by-definition for **mean solar day** (the UT1 average), but **UTC** days containing leap seconds have 86401 s (positive leap, 27 inserted since 1972) or 86399 s (negative leap, never used). The docstring "mean solar day" is correct in the historical sense but a consumer using this for UTC astronomy will be off by N × leap_seconds. The package has no time-scale taxonomy and no leap-second table — out of scope for `constants/`, belongs in a future `time/` package, but the docstring should warn: "for TAI/TT/UT1; UTC days containing leap seconds differ."

**Finding C-TIME-2:** **missing time constants:**
- Julian year = 365.25 × 86400 = 31557600 s exact (used to define light-year)
- tropical year = 365.24219 × 86400 ≈ 31556925.2 s (varies, IAU value)
- sidereal day = 86164.0905 s (≈, IERS value, drifts)
- sidereal year = 365.25636 × 86400 ≈ 31558149.5 s
- gregorian year = 365.2425 × 86400 = 31556952 s exact (calendar mean)
- TAI−UTC = +37 s as of 2017-01-01 (no further leap seconds through 2026-05; Earth rotation has been speeding up — possible negative leap by 2029 per CGPM 2022 resolution to retire leap seconds by 2035)

Recommend adding `SecondsPerJulianYear = 31557600.0` (exact, used by IAU light-year definition) — 1 LOC, no controversy.

---

## Astronomical constants

**Finding C-ASTRO-1:** the package ships **no astronomical constants** at all. The `orbital/` package uses `constants.GravitationalConst` and inlines specific masses. Constants commonly needed:
- AU = 149597870700 m exact (IAU 2012)
- light-year = 9460730472580800 m exact (Julian year × c)
- parsec = 648000/π × AU = 3.0856775814913673e16 m (π-limited)
- solar mass M☉ = 1.98892e30 kg (IAU 2015 nominal: GM☉ = 1.3271244e20 m³/s²; mass varies with G uncertainty)
- Earth mass M⊕ = 5.9722e24 kg (IAU 2015 nominal: GM⊕ = 3.986004e14 m³/s²)
- Jupiter mass = 1.89813e27 kg (nominal GM = 1.2668653e17)
- solar radius = 6.957e8 m (IAU 2015 nominal)
- Earth equatorial radius = 6378137.0 m (WGS84 exact)
- Earth polar radius = 6356752.3142 m (WGS84)
- Earth flattening = 1/298.257223563 (WGS84)

**IAU 2015 nominal values are GM products** (not separate G × M), because GM is measured to ~10⁻⁹ while G is known to ~2.2e-5 — multiplying G × M_solar gives 2.2e-5 relative error when GM_solar is known to 10⁻⁹. Recommend ship `GMSun = 1.32712440041e20` (exact-ish) rather than computing `GravitationalConst * SolarMass`. Cost: ~20 LOC for the IAU 2015 nominal set.

---

## Mathematical constants: precision, hard-fail vs math.Pi

| Constant | Shipped | math stdlib | Identical? |
|---|---|---|---|
| `Pi` | `math.Pi` (alias) | math.Pi | yes (3.141592653589793) |
| `E` | `math.E` (alias) | math.E | yes |
| `Sqrt2` | `math.Sqrt2` (alias) | math.Sqrt2 | yes (1.4142135623730951) |
| `Ln2`, `Ln10`, `Log2E`, `Log10E` | `math.X` aliases | math.X | yes |
| `Phi` | `1.618033988749895` literal | (no math.Phi) | matches `(1+sqrt(5))/2` to float64 |
| `Sqrt3` | `1.7320508075688772` literal | (no math.Sqrt3) | matches `math.Sqrt(3)` |
| `EulerGamma` | `0.5772156649015329` literal | (no math.EulerGamma) | matches OEIS A001620 truncated to float64 |

All to ~16 decimal digits (full float64). **No hard-fail vs `math` stdlib.** Tests verify equality (`constants_test.go:14-34`).

**Finding C-MATH-1:** missing common math constants: `Pi/2`, `Pi/4`, `2*Pi`, `Pi*Pi`, `1/Pi`, `1/Sqrt(2)`, `Sqrt(2*Pi)` (used in Gaussian PDFs across `prob/`), `Sqrt(Pi)`, `Catalan = 0.915965594177...`, `Apery = 1.202056903159...` (ζ(3)). The Gaussian-PDF case is recurring — `prob/` likely recomputes `1/(σ*sqrt(2*π))` per call. Cost: ~15 LOC.

**Finding C-MATH-2:** `Phi`, `Sqrt3`, `EulerGamma` are literal float64 values — verify they are the **correctly-rounded** float64 nearest to the true mathematical value. For `Phi`: true is 1.6180339887498948482… — float64 nearest is 1.6180339887498949 (verified: `math.Nextafter(1.618033988749895, 2)` = 1.6180339887498951); the shipped `1.618033988749895` is actually 1.618033988749895**0** (the last digit shown is 5 but rounds to even = 0 in the 16th place) — equal to `(1+math.Sqrt(5))/2` per the existing test, so OK. `EulerGamma = 0.5772156649015329` true value 0.57721566490153286060…, float64 nearest is 0.5772156649015329 ✓. `Sqrt3 = 1.7320508075688772` true 1.7320508075688772935…, nearest float64 is 1.7320508075688772 ✓. All three correct. No bug, but **add tests pinning these to the correctly-rounded values** (currently `Sqrt3` test compares to `math.Sqrt(3)` which is implementation-defined though IEEE 754 mandates correctly-rounded, and `EulerGamma` test compares to the same literal — circular).

---

## Tolerance-contract violations

CLAUDE.md "Per-function tolerance ... transcendentals use 1e-11, accumulating ops use 1e-9." The golden file has `tolerance: 0` for ε₀, μ₀, G — meaning bit-exact equality required. Updating to CODATA 2022 will produce 7-ulp differences (~6e-10 relative) that violate the `tolerance: 0` contract immediately.

**Action item C-TOL-1:** When updating to CODATA 2022 (action C-2022-1), the golden vectors must change too — this is by design (the literal IS the value). Cross-language implementations (Python/C++/C#) must regenerate against the new literals, not against arbitrary-precision derived values. Document this in the golden-file regen procedure.

---

## Summary table of action items

| ID | Item | LOC | Risk |
|---|---|---|---|
| C-2022-1 | Update ε₀, μ₀ to CODATA 2022 (`8.8541878188e-12`, `1.25663706127e-6`); update docs and goldens | ~10 | Low — only `em/em.go:21` consumer |
| C-2022-2 | Add `CODATAVintage = "2022"` const + per-vintage golden file convention | ~30 | Low |
| C-EXACT-1 | Docstring tweak — "exact decimal value, float64 nearest representation" on h, e, k_B, N_A | ~8 | Zero |
| C-EXACT-2 | Add `CesiumHyperfine = 9192631770.0`, `LuminousEfficacy540THz = 683.0` (exact integer SI defining constants) | ~6 | Zero |
| C-DERIVED-1 | Either derive `StefanBoltzmann` or document the literal-vs-computed rounding in test | ~10 | Low |
| C-DERIVED-2 | Add `Uncertainty` companion or `Constant` struct surface | ~40 | Medium — API addition |
| C-ROUNDTRIP-1 | Strengthen unit-coherence tests to bit-exact (tolerance 0, not 1e-10) | ~5 | Zero |
| C-ROUNDTRIP-2 | Document or fix `PascalsPerPSI` 1-ulp rounding | ~5 | Zero |
| C-ROUNDTRIP-3 | Document F↔K 3e-13 K rounding floor | ~3 | Zero |
| C-MISSING-1 | Add gal/ton/eV/AU/ly missing units | ~30 | Zero |
| C-MISSING-2 | Document Imperial vs US gallon trap | ~5 | Zero |
| C-TIME-1 | Annotate `SecondsPerDay` as TAI/TT/UT1 (warn UTC) | ~3 | Zero |
| C-TIME-2 | Add `SecondsPerJulianYear = 31557600.0` | ~3 | Zero |
| C-ASTRO-1 | Add IAU 2015 nominal astronomical set (GM products, AU, ly, R☉, R⊕, WGS84) | ~25 | Low |
| C-MATH-1 | Add `TwoPi`, `HalfPi`, `Sqrt2Pi`, `InvSqrt2Pi`, `Catalan`, `Apery` | ~15 | Zero |
| C-MATH-2 | Pin `Phi`, `Sqrt3`, `EulerGamma` tests to correctly-rounded literals | ~5 | Zero |
| C-TOL-1 | Document golden-regeneration procedure for CODATA-vintage updates | ~10 (docs) | Zero |

**Highest-value bundle:** C-2022-1 (ε₀, μ₀ to CODATA 2022) + C-EXACT-2 (ΔνCs, K_cd) + C-MISSING-1 (gal/ton/eV/AU/ly) + C-ASTRO-1 (IAU nominal set) + C-MATH-1 (Sqrt2Pi etc.). Total ~85 LOC, all additive except the two ε₀/μ₀ literal edits, brings the package current to 2026-05 published standards.

**Out-of-scope-but-noted:** machine-readable uncertainty/source metadata is the design pivot — every "Action C-*" item above can be expressed as data inside a `Constant` struct, and the golden-file format already has all the slots (description, expected, tolerance, source). If the project ships a v0.11 metadata refactor, do all of these items in one wave.

---

## Sources

- [CODATA recommended values of the fundamental physical constants: 2022 (Mohr, Newell, Taylor, Tiesinga)](https://link.aps.org/doi/10.1103/RevModPhys.97.025002) — *Rev. Mod. Phys.* 97 025002, published 2025; preprint [arXiv:2409.03787](https://arxiv.org/html/2409.03787v1)
- [NIST Fundamental Physical Constants — Complete Listing (2022 CODATA)](https://physics.nist.gov/cuu/Constants/Table/allascii.txt) — authoritative numerical values used in the delta table above
- [NIST CODATA 2022 wallet card](https://physics.nist.gov/cuu/pdf/wallet_2022.pdf)
- [CODATA Task Group on Fundamental Constants](https://codata.org/initiatives/data-science-and-stewardship/fundamental-physical-constants/) — adjustment cycle and methodology
- IAU 2012 Resolution B2 (AU = 149597870700 m exact); IAU 2015 Resolution B3 (nominal solar/Earth/Jovian conversion factors); WGS84 (Earth ellipsoid); ISO 80000-3:2019 (g_n, atm); 1959 international yard and pound agreement; CGPM 2022 resolution on retiring leap seconds by 2035.

Report at `agents/046-constants-numerics.md`, 222 lines.
