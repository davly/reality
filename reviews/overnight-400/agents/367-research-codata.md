# 367 — research-codata (CODATA 2022 / IAU / ITRF / BIPM alignment audit)

## Headline
Reality `constants/physics.go` self-declares CODATA 2018; CODATA 2022 (NIST/CODATA-TGFC, *Rev. Mod. Phys.* 97 025002, 2024) only moves two shipped values — ε₀ and μ₀ at the 7-ulp level — but reality lacks any IAU 2015, ITRF2020, or BIPM defining-constants surface entirely (ΔνCs, K_cd, AU, parsec, GM_⊙, R_⊕ all absent).

## Context (no duplication of slot 046/047)
Slot 046 (`agents/046-constants-numerics.md`) already enumerated CODATA 2018→2022 deltas at the float-literal level and proposed action items C-2022-1/2. Slot 047 enumerated additive missing constants. This slot 367 adds: (a) verification of those deltas against current NIST authoritative tables (fetched 2026-05-09), (b) IAU 2015 Resolution B3 nominal-values cross-check, (c) ITRF2020 / WGS84 / GRS80 alignment, (d) BIPM SI-2019 defining-constants completeness.

## Constants audit (existing vs CODATA 2022 / IAU 2015 / BIPM)

| Constant | Reality file:line | Reality value | Authoritative 2022/2015 | Drift? |
|---|---|---|---|---|
| `SpeedOfLight` c | `physics.go:16` | 299792458.0 | 299 792 458 (BIPM SI 2019, exact) | none — defining |
| `Planck` h | `physics.go:20` | 6.62607015e-34 | 6.626 070 15 × 10⁻³⁴ J·Hz⁻¹ (BIPM SI 2019, exact) | none — defining |
| `Boltzmann` k_B | `physics.go:31` | 1.380649e-23 | 1.380 649 × 10⁻²³ J/K (BIPM SI 2019, exact) | none — defining |
| `Avogadro` N_A | `physics.go:35` | 6.02214076e23 | 6.022 140 76 × 10²³ mol⁻¹ (BIPM SI 2019, exact) | none — defining |
| `ElementaryCharge` e | `physics.go:39` | 1.602176634e-19 | 1.602 176 634 × 10⁻¹⁹ C (BIPM SI 2019, exact) | none — defining |
| `GravitationalConst` G | `physics.go:45` | 6.67430e-11 | 6.674 30(15) × 10⁻¹¹ m³ kg⁻¹ s⁻² (CODATA 2022) | none at 5-digit precision; u_r still 2.2 × 10⁻⁵ |
| `VacuumPermittivity` ε₀ | `physics.go:52` | 8.8541878128e-12 | **8.854 187 8188(14) × 10⁻¹² F/m** (CODATA 2022) | **DRIFT: +6 × 10⁻²¹ abs, ~7 × 10⁻¹⁰ rel — exceeds 1 × 10⁻¹¹ CLAUDE.md tolerance contract** |
| `VacuumPermeability` μ₀ | `physics.go:59` | 1.25663706212e-6 | **1.256 637 061 27(20) × 10⁻⁶ H/m** (CODATA 2022) | **DRIFT: −8.5 × 10⁻¹⁶ abs, ~7 × 10⁻¹⁰ rel — exceeds tolerance contract** |
| `StefanBoltzmann` σ | `physics.go:65` | 5.670374419e-8 | 5.670 374 419 × 10⁻⁸ W m⁻² K⁻⁴ (derived-exact, unchanged) | none |
| `GasConstant` R | `physics.go:71` | N_A·k_B (const-arith) | 8.314 462 618… (derived-exact) | none |
| `StandardGravity` g_n | `physics.go:77` | 9.80665 | 9.806 65 m/s² (CGPM 1901, ISO 80000-3:2019, exact) | none |
| `AtmPressure` 1 atm | `physics.go:81` | 101325.0 | 101 325 Pa (ISO 80000-3:2019, exact) | none |
| Math constants | `math.go:23-70` | float64 of π, e, φ, √2, √3, ln 2, ln 10, log₂e, log₁₀e, γ | mathematical, no committee | none — exact-to-float64 by construction |
| Length/mass/time/pressure unit factors | `units.go:18-101` | mile, ft, in, yd, NM, lb, oz, °C, °F, rad/deg, s/h/d, atm, bar, psi | NIST SP811, ISO 80000-3:2019, IHO 1929, intl yard-and-pound 1959 | none |

**Net drift surface: 2 of 31 shipped constants** (ε₀, μ₀). Both at the same ~7 × 10⁻¹⁰ relative level, both consequence of the post-2019 SI re-derivation through fine-structure α (which itself moved 7 × 10⁻¹⁰ between 2018 and 2022).

## CODATA 2022 deltas reality does NOT ship (downstream impact)

These were not in scope for slot 046 and matter for any future expansion (electron/proton mass, atomic masses surface) — see slot 047 §Tier-1.4.

| Constant | CODATA 2018 | CODATA 2022 | Δ_rel |
|---|---|---|---|
| α (fine-structure) | 7.2973525693e-3 | 7.2973525643e-3 | −7 × 10⁻¹⁰ |
| α⁻¹ | 137.035999084 | 137.035999177 | +6.8 × 10⁻¹⁰ |
| m_e (electron) | 9.1093837015e-31 kg | 9.1093837139e-31 kg | +1.4 × 10⁻⁹ |
| m_p (proton) | 1.67262192369e-27 kg | 1.67262192595e-27 kg | +1.4 × 10⁻⁹ |
| u (atomic mass) | 1.66053906660e-27 kg | 1.66053906892e-27 kg | +1.4 × 10⁻⁹ |
| R_∞ (Rydberg) | 10973731.568160 m⁻¹ | 10973731.568157 m⁻¹ | −2.7 × 10⁻¹³ |
| cR_∞ (Rydberg freq) | 3.2898419602508e15 Hz | 3.2898419602500e15 Hz | −2.4 × 10⁻¹³ |

Source: NIST `physics.nist.gov/cuu/Constants/Table/allascii.txt` (CODATA 2022 release, May 2024) cross-checked against Mohr et al. *Rev. Mod. Phys.* 97 025002 (Apr 2025).

## IAU 2015 Resolution B3 — coverage gap (entirely missing)

Reality ships zero astronomical constants. IAU 2015 Resolution B3 (Mamajek et al. arXiv 1510.07674; Prša et al. *AJ* 152 41, 2016) defines the canonical nominal-conversion set as exact SI values (not measured CBEs):

| IAU 2015 nominal | Defined value | Reality? |
|---|---|---|
| Astronomical unit (au) | 149 597 870 700 m (IAU 2012 Res B2, exact) | absent |
| Parsec | 648000/π × au = 3.085677581491×10¹⁶ m (π-limited) | absent |
| Light-year | 9.4607304725808×10¹⁵ m (Julian year × c, derivable-exact) | absent |
| Nominal solar radius R_⊙ᴺ | 6.957 × 10⁸ m (exact) | absent |
| Nominal solar irradiance S_⊙ᴺ | 1361 W/m² (exact) | absent |
| Nominal solar luminosity L_⊙ᴺ | 3.828 × 10²⁶ W (exact) | absent |
| Nominal solar effective temp T_eff⊙ᴺ | 5772 K (exact) | absent |
| Nominal solar mass parameter (GM)_⊙ᴺ | 1.3271244 × 10²⁰ m³/s² (exact) | absent |
| Nominal terrestrial equatorial radius R_eE^N | 6.3781 × 10⁶ m (zero-tide, IERS-derived, exact) | absent |
| Nominal terrestrial polar radius R_pE^N | 6.3568 × 10⁶ m (exact) | absent |
| Nominal terrestrial mass parameter (GM)_E^N | 3.986004 × 10¹⁴ m³/s² (exact) | absent |
| Jovian equatorial radius R_eJ^N | 7.1492 × 10⁷ m (exact) | absent |
| Jovian mass parameter (GM)_J^N | 1.2668653 × 10¹⁷ m³/s² (exact) | absent |

**Critical IAU design point** (slot 046 §107-orbital also flagged): IAU 2015 ships **GM products not separate G and M**, because GM is measured to ~10⁻⁹ while G is known only to 2.2 × 10⁻⁵. Multiplying `GravitationalConst × SolarMass` discards 4-5 digits of precision. Any future `orbital/`-facing constants surface MUST be GM-first, not (G, M)-pair.

## ITRF2020 / WGS84 / GRS80 — coverage gap

Reality ships zero geodetic constants. Reference: IERS / IGN ITRF2020 release (epoch 2015.0), GPS-aligned via WGS 84 (G2296)=ITRF2020 since 2024-01-07 (NGA earth-info).

| Geodetic | Value | Source | Reality? |
|---|---|---|---|
| GRS80/WGS84 semi-major axis a | 6 378 137.0 m (exact) | GRS80, ITRF2020 | absent (slot 107 already requested) |
| GRS80 inverse flattening 1/f | 298.257222101 | GRS80 | absent |
| WGS84 inverse flattening 1/f | 298.257223563 | WGS84 (differs from GRS80 at 8th digit) | absent |
| WGS84 angular velocity ω | 7.292115 × 10⁻⁵ rad/s | WGS84 | absent |
| WGS84 GM | 3.986004418 × 10¹⁴ m³/s² | WGS84 (differs from IAU GM_E^N at 7th digit) | absent |
| J₂ (Earth dynamic form factor) | 1.0826267 × 10⁻³ | EGM2008 | absent (slot 107 requested) |

ITRF2020 vs ITRF2014 is a frame-realisation update (station coordinates and velocities), not a defining-constants change — no impact on the constants package per se.

## BIPM SI 2019 — defining constants completeness

The 2019 SI redefinition fixed seven defining constants (BIPM SI Brochure 9th ed., 2019). Reality coverage:

| BIPM defining constant | Defined value | Reality? |
|---|---|---|
| ΔνCs (Cs hyperfine transition) | 9 192 631 770 Hz | **absent** (slot 046 C-EXACT-2) |
| c (speed of light) | 299 792 458 m/s | `physics.go:16` |
| h (Planck) | 6.626 070 15 × 10⁻³⁴ J·s | `physics.go:20` |
| e (elementary charge) | 1.602 176 634 × 10⁻¹⁹ C | `physics.go:39` |
| k (Boltzmann) | 1.380 649 × 10⁻²³ J/K | `physics.go:31` |
| N_A (Avogadro) | 6.022 140 76 × 10²³ mol⁻¹ | `physics.go:35` |
| K_cd (luminous efficacy 540 THz) | 683 lm/W | **absent** (slot 046 C-EXACT-2) |

5 of 7 defining constants shipped; ΔνCs and K_cd absent. Both are integer-exact (4 LOC each).

## Concrete recommendations

1. **Update ε₀** (`physics.go:52`): `8.8541878128e-12` → `8.8541878188e-12`. Bump golden vector at `testdata/constants/physics_constants.json` for the matching slot. Update docstring "CODATA 2018" → "CODATA 2022 (NIST/Mohr et al. 2024)".

2. **Update μ₀** (`physics.go:59`): `1.25663706212e-6` → `1.25663706127e-6`. Same golden + docstring touchups.

3. **Update package-doc** (`math.go:11`, `physics.go:11`, CLAUDE.md): "NIST CODATA 2018" → "NIST CODATA 2022 (Mohr, Newell, Taylor, Tiesinga, *Rev. Mod. Phys.* 97 025002, 2025)". Total: 3 strings.

4. **Adopt slot 046 C-2022-2 versioning** before the bump: add `const CODATAVintage = "2022"` to `physics.go` and freeze a per-vintage golden file. The four-year CODATA cycle means the next adjustment lands ~2027 — a vintage marker prevents silent breakage.

5. **Add the two missing BIPM defining constants** (slot 046 C-EXACT-2): `DeltaNuCs = 9192631770.0` (Hz, exact) and `LumEfficacy540THz = 683.0` (lm/W, exact). 4 LOC, completes the 7-constant SI defining set.

6. **Consider IAU 2015 Resolution B3 nominal-set surface as a separate `astro` block** in `physics.go` (or a new `constants/astro.go`): au, parsec, ly, R_⊙^N, S_⊙^N, L_⊙^N, T_eff⊙^N, (GM)_⊙^N, R_eE^N, R_pE^N, (GM)_E^N. ~40 LOC. **Ship as GM products, not (G, M) pairs.** This unlocks slot 107-orbital additive requests with proper precision.

7. **Defer ITRF2020 / WGS84 geodetic constants** to a separate `constants/geodetic.go` if and when a `geodesy/` package is justified (slot 107 stages this). They are ellipsoid-realisation specific and arguably belong with the package that uses them, not in `constants/`.

8. **Cross-reference**: slot 047 (constants-missing) Tier-1 fully captures items 5–7 above as additive. Slot 046 (constants-numerics) C-2022-1/2 captures items 1–4. This slot 367 supplies the external-reference-traceable evidence (NIST allascii fetch 2026-05-09, IAU Res B3, BIPM SI Brochure 9th ed.) that those slots' recommendations are still authoritative as of today.

## Cross-links

- Slot **046** (`agents/046-constants-numerics.md`): float-level CODATA 2018→2022 delta table, action items C-2022-1/2, C-EXACT-1/2, C-DERIVED-1/2.
- Slot **047** (`agents/047-constants-missing.md`): additive enumeration of ~140 missing constants Tier-1, including IAU/parsec/ly/Planck-set.
- Slot **048** (`agents/048-constants-sota.md`): peer-package comparison (scipy.constants, Boost.Units, Mathematica `PhysicalConstants[]`).
- Slot **049** (`agents/049-constants-api.md`): API ergonomics including `SolarMass`/`PlanckMass` naming.
- Slot **107** (`agents/107-orbital-missing.md`): explicit downstream request for `J2Earth`, `MuSun`, `MuMoon`, `EarthRadiusEquatorial`, `EarthRadiusPolar`, `EarthFlattening`, `SolarConstant`.

## Sources

- NIST CODATA 2022 wallet card — `https://physics.nist.gov/cuu/pdf/wallet_2022.pdf` (May 2024 release)
- NIST `physics.nist.gov/cuu/Constants/Table/allascii.txt` (CODATA 2022 ASCII table)
- Mohr, Newell, Taylor, Tiesinga — "CODATA recommended values of the fundamental physical constants: 2022" *Rev. Mod. Phys.* 97 025002 (2025) — `https://link.aps.org/doi/10.1103/RevModPhys.97.025002` and arXiv 2409.03787
- NIST individual values: G `https://physics.nist.gov/cgi-bin/cuu/Value?bg`, ε₀ `https://physics.nist.gov/cgi-bin/cuu/Value?eqep0`, μ₀ `https://physics.nist.gov/cgi-bin/cuu/Value?mu0`
- IAU 2015 Resolution B3 — Mamajek et al. arXiv 1510.07674; Prša et al. "Nominal Values for Selected Solar and Planetary Quantities: IAU 2015 Resolution B3" *AJ* 152 41 (2016) — `https://iopscience.iop.org/article/10.3847/0004-6256/152/2/41`
- IAU 2012 Resolution B2 — astronomical unit defined as 149 597 870 700 m exact
- BIPM SI Brochure, 9th ed. (2019) — `https://www.bipm.org/en/publications/si-brochure` — defines the seven defining constants
- IERS / ITRF2020 — `https://itrf.ign.fr/en/solutions/ITRF2020`; EPSG:9988 / EPSG:9989; NGA WGS 84 (G2296) realisation document
- ISO 80000-3:2019 (space and time quantities) — defines g_n = 9.80665, atm = 101325 Pa, °C–K offset 273.15 exact
- Slot 046 (`reviews/overnight-400/agents/046-constants-numerics.md`) and slot 047 (`reviews/overnight-400/agents/047-constants-missing.md`)
