# 047 | constants-missing

Strictly-additive enumeration of constants/unit-systems NOT in
`C:\limitless\foundation\reality\constants\` (math.go 10 + physics.go 11
+ units.go 10 = **31 total**). Sibling 046 covered numerical correctness
of shipped values (CODATA 2022 deltas, ULP audits) and noted in passing:
CODATA 2022 update, ΔνCs, K_cd, AU, ly, IAU GMs, gal/ton/eV/BTU, Sqrt2Pi.

Reference: scipy.constants (CODATA 2022, ~300), NIST allascii.txt (350),
IAU 2012/2015 nominal, Mathematica `PhysicalConstants[]`, Boost.Units,
Planck/Stoney/Hartree/geometrized literature.

**TL;DR.** Reality ships 31; canonical 2026 surface is ~180-220 named
+ 6-8 unit-system families. **~14-17% coverage.** Biggest deposit:
CODATA particle-property block (~55 entries). Most-impactful internal
gap: `FineStructure = 7.2973525643e-3` — `em/em.go:21` recomputes
`1/(4πε₀)` instead. Most-impactful natural-unit gap: full Planck set
(zero shipped; 4 foundational + 9 derived ≈ 20 LOC).

---

## Tier 1 — high-impact (additive ~140 constants, ~250 LOC)

### 1.1 Mathematical constants (~30 LOC, all const-arithmetic or correctly-rounded literals)

Pi-family (recurring across `prob/`, `signal/`, `acoustics/`, `chaos/`):
`TwoPi`, `HalfPi`, `QuarterPi`, `PiSquared`, `Sqrt2Pi ≈ 2.5066282746310002`
(Gaussian PDF norm — `prob/distributions.go` recomputes per call),
`InvSqrt2Pi ≈ 0.3989422804014327`, `SqrtPi ≈ 1.7724538509055159`
(gamma(1/2), erf), `SqrtHalfPi ≈ 1.2533141373155003`,
`LnSqrt2Pi ≈ 0.9189385332046728`, `InvPi`, `Inv2Pi`.

Standalone transcendentals: `Catalan = 0.9159655941772190` (G; lattice
paths), `Apery = 1.2020569031595943` (ζ(3); QED loop integrals),
`OmegaConstant = 0.5671432904097839` (Ω·e^Ω = 1; Lambert W(1)),
`eGamma = 1.7810724179901979` (Mertens' theorem), `LnPi = 1.1447298858494002`,
`Log10_2 = 0.30102999566398120` (compression bits/digit; DSP octave/decade),
`MeisselMertens = 0.2614972128476428`, `BrunsConstant = 1.902160583104`
(twin primes), `Khinchin = 2.6854520010653064`, `GlaisherKinkelin = 1.2824271291006226`.

Numerical-analysis: `SqrtMachineEps ≈ 1.4901161193847656e-8`
(central-difference step), `CubeRootMachineEps ≈ 6.055454452393343e-6`
(Ridders), `LogMaxFloat64 ≈ 709.7827128933840` (log-sum-exp overflow),
`LogMinFloat64 ≈ -708.3964185322641`.

### 1.2 Complete the SI-defining-constant set (~4 LOC)

- `CesiumHyperfine = 9192631770.0` Hz (ΔνCs; defines the second; integer exact)
- `LuminousEfficacy540THz = 683.0` lm/W (K_cd; defines the candela; integer exact)

(`SpeedOfLight`, `Planck`, `ElementaryCharge`, `Boltzmann`, `Avogadro`
already shipped — these two close out 7 of 7.)

### 1.3 CODATA particle masses (~25 LOC, all CODATA-2022 literals)

Leptons: `ElectronMass = 9.1093837139e-31` kg (was 9.1093837015e-31 in
2018), `MuonMass = 1.883531627e-28`, `TauMass = 3.16754e-27`.

Baryons/nucleons: `ProtonMass = 1.67262192595e-27`,
`NeutronMass = 1.67492750056e-27`, `DeuteronMass = 3.3435837768e-27`,
`TritonMass = 5.0073567512e-27`, `HelionMass = 5.0064127862e-27` (³He),
`AlphaMass = 6.6446573450e-27` (⁴He).

Atomic mass: `AtomicMassConstant = 1.66053906892e-27` kg (Da, CODATA 2022;
was 1.66053906660e-27 in 2018), `AtomicMassMeV = 931.49410372` MeV/c².

Mass ratios (dimensionless): `ProtonElectronRatio = 1836.152673426`,
`MuonElectronRatio = 206.7682827`, `NeutronProtonRatio = 1.00137841946`,
`ElectronProtonMassRatio = 5.446170214889e-4`.

### 1.4 Hartree atomic-units block (~50 LOC)

The standard system of computational chemistry/QED (Gaussian, NWChem,
ORCA, Psi4 native). Reality has zero.

Foundational: `BohrRadius = 5.29177210544e-11` m (a₀),
`HartreeEnergy = 4.3597447222060e-18` J (E_h ≈ 27.211 eV),
`HartreeEnergyEv = 27.211386245981` eV,
`RydbergConstant = 10973731.568157` m⁻¹ (R∞; CODATA 2022; was
10973731.568160 in 2018), `RydbergFrequency = 3.2898419602500e15` Hz,
`RydbergEnergy = 2.1798723611030e-18` J,
**`FineStructure = 7.2973525643e-3`** (α; CODATA 2022; was
7.2973525693e-3 in 2018 — the single most-cited dimensionless constant),
`InverseFineStructure = 137.035999177` (1/α — literature usually quotes
inverse).

Characteristic atomic scales:
`ClassicalElectronRadius = 2.8179403205e-15` m (r_e = α²·a₀),
`ComptonWavelength = 2.42631023538e-12` m (λ_C electron = h/m_ec),
`ReducedComptonWavelength = 3.8615926744e-13` m (λ_C/2π),
`ProtonComptonWavelength = 1.32140985360e-15` m,
`ThomsonCrossSection = 6.6524587051e-29` m² (σ_T = 8πr_e²/3).

Magnetic moments: `BohrMagneton = 9.2740100657e-24` J/T (μ_B = eħ/2m_e),
`NuclearMagneton = 5.0507837393e-27` J/T (μ_N = eħ/2m_p),
`ElectronMagneticMoment = -9.2847646917e-24` J/T (sign matters),
`ProtonMagneticMoment = 1.41060679545e-26` J/T,
`NeutronMagneticMoment = -9.6623653e-27` J/T.

g-factors (dimensionless): `ElectronGFactor = -2.00231930436092`
(QED-tested to 12 digits), `MuonGFactor = -2.00233184123` (Fermilab g-2),
`ProtonGFactor = 5.5856946893`, `NeutronGFactor = -3.82608552`.

### 1.5 Stefan-Boltzmann / Wien / radiation block (~10 LOC)

Currently shipped: `StefanBoltzmann`. Missing the rest:

- `WienDisplacement = 2.897771955e-3` m·K (b; λ_max·T = b)
- `WienFrequency = 5.878925757e10` Hz/K (b'; ν_max/T = b')
- `FirstRadiationConstant = 3.741771852e-16` W·m² (c₁ = 2πhc²)
- `FirstRadiationConstantSpectral = 1.191042972e-16` W·m²/sr (c₁L = 2hc²)
- `SecondRadiationConstant = 1.438776877e-2` m·K (c₂ = hc/k_B)

### 1.6 EM / quantum-condensed-matter (~12 LOC)

- `FaradayConstant = 96485.33212` C/mol (F = N_A·e; exact-from-exact)
- `MagneticFluxQuantum = 2.067833848e-15` Wb (Φ₀ = h/2e; QHE/Josephson)
- `JosephsonConstant = 4.835978484e14` Hz/V (K_J = 2e/h)
- `VonKlitzingConstant = 25812.80745` Ω (R_K = h/e²; quantum Hall)
- `ConductanceQuantum = 7.748091729e-5` S (G₀ = 2e²/h)
- `CoulombConstant = 8.9875517873681764e9` N·m²/C² (k_e = 1/(4πε₀);
  **currently inlined in `em/em.go:21` — extract!**)
- `VacuumImpedance = 376.730313412` Ω (Z₀ = μ₀c ≈ 120π Ω)

### 1.7 Astronomical — IAU 2012/2015 nominal (~30 LOC; noted by 046)

Distance: `MetersPerAU = 149597870700.0` (IAU 2012 B2; integer exact),
`MetersPerLightYear = 9460730472580800.0` (Julian year × c; integer exact),
`MetersPerParsec = 3.0856775814913673e16` (648000/π × AU; π-limited),
`MetersPerKiloparsec`, `MetersPerMegaparsec`,
`SolarRadiusNominal = 6.957e8` (R_⊙^N; IAU 2015 B3),
`EarthEquatorialRadiusNominal = 6.3781e6`, `EarthPolarRadiusNominal = 6.3568e6`,
`EarthEquatorialRadiusWGS84 = 6378137.0` (a; integer exact),
`EarthPolarRadiusWGS84 = 6356752.3142` (b),
`EarthFlatteningWGS84 = 1.0/298.257223563` (1/f),
`JupiterEquatorialRadiusNominal = 7.1492e7`.

GM products (IAU 2015 ships GM, **not** G × M, because GM is measured to
~10⁻⁹ while G to ~2.2×10⁻⁵; multiplying loses 4-5 orders of precision):
`GMSunNominal = 1.32712440041e20` m³/s², `GMEarthNominal = 3.986004e14`,
`GMJupiterNominal = 1.2668653e17`.

Cosmology (vintage-tracked): `H0Planck2018 = 67.4` km/s/Mpc,
`H0SH0ES2022 = 73.04` (the 5σ Hubble tension),
`CMBTemperature = 2.7255` K (Fixsen 2009 ApJ 707 916),
`CriticalDensity = 9.47e-27` kg/m³ (at H₀ = 67.4).

Time/epoch: `SecondsPerJulianYear = 31557600.0` (365.25 × 86400; exact;
defines ly), `SecondsPerJulianCentury = 3155760000.0`,
`SecondsPerGregorianYear = 31556952.0` (365.2425 × 86400; integer exact),
`SecondsPerTropicalYear = 31556925.2`, `SecondsPerSiderealYear = 31558149.5`,
`SecondsPerSiderealDay = 86164.0905`, `JulianDateJ2000 = 2451545.0`.

### 1.8 Energy/power/force/volume/mass/pressure/speed unit conversions (~80 LOC)

Energy: `JoulesPerEv = 1.602176634e-19` (= e × 1V, exact since SI 2019),
`JoulesPerErg = 1e-7` (CGS, integer exact), `JoulesPerCal_IT = 4.1868`
(IT cal, exact), `JoulesPerCal_th = 4.184` (thermochemical, exact; food
"Calorie" = kcal_th), `JoulesPerKcal = 4184.0`,
`JoulesPerBTU_IT = 1055.05585262` (ISO default), `JoulesPerBTU_th = 1054.350`,
`JoulesPerKWh = 3600000.0` (integer exact),
`JoulesPerFootPound = 1.3558179483314004`,
`JoulesPerHartree = 4.3597447222060e-18`, `EvPerHartree = 27.211386245981`.

Power: `WattsPerHorsepowerMech = 745.6998715822702` (imperial; 550 ft·lbf/s),
`WattsPerHorsepowerMetric = 735.49875` (PS/cv; 75 kgf·m/s),
`WattsPerHorsepowerElectric = 746.0` (IEEE),
`WattsPerBTUPerHour = 0.29307107017222222` (HVAC).

Force: `NewtonsPerPoundForce = 4.4482216152605` (lbf; exact),
`NewtonsPerKgf = 9.80665` (= StandardGravity exact),
`NewtonsPerDyne = 1e-5` (CGS), `KgPerSlug = 14.59390294`.

Volume (gallon trap noted by 046):
`LitersPerGallonUS = 3.785411784` (= 231 in³ exact),
`LitersPerGallonImperial = 4.54609` (UK W&M Act 1985; exact;
**~20% larger than US gal**),
`LitersPerQuartUS = 0.946352946`, `LitersPerPintUS = 0.473176473`,
`LitersPerFluidOunceUS = 0.0295735295625` (= US gal/128),
`LitersPerFluidOunceImperial = 0.0284130625` (= Imp gal/160; ~4%
**smaller** than US fl oz — opposite-direction trap),
`LitersPerCubicInch = 0.016387064`, `LitersPerCubicFoot = 28.316846592`,
`LitersPerBarrelOil = 158.987294928` (= 42 US gal),
`LitersPerBarrelUS = 119.240471196` (= 31.5 US gal).

Mass: `KgPerShortTon = 907.18474` (US, 2000 lb),
`KgPerLongTon = 1016.0469088` (UK, 2240 lb), `KgPerTonne = 1000.0`,
`KgPerStone = 6.35029318` (UK, 14 lb), `KgPerGrain = 6.479891e-5`
(pharmacy/ammo; lb/7000), `KgPerCarat = 2e-4` (jewellery, integer exact),
`KgPerAtomicMass = 1.66053906892e-27` (= AtomicMassConstant; particle bridge).

Pressure: `PascalsPerMillibar = 100.0`, `PascalsPerMmHg = 133.322387415`,
`PascalsPerTorr = 101325.0/760.0` (atm/760, exact-derived; differs from
mmHg by rounding), `PascalsPerInHg = 3386.389` (US weather),
`PascalsPerKsi = 6894757.293168361` (engineering; 1000 psi).

Speed: `MetersPerSecondPerKnot = 0.5144444444444444` (= NM/h = 1852/3600;
exact rational, recurring), `MetersPerSecondPerMph = 0.44704`
(= mile/3600), `MetersPerSecondPerKmh = 1.0/3.6`.

### 1.9 SI prefix + binary-prefix tables (~30 LOC)

scipy.constants ships these; Reality ships zero. CGPM 2022 added
quetta/ronna/quecto/ronto (first new SI prefixes since 1991 — Reality
being a 2026 package should ship them):

```go
const (
    Quetta=1e30; Ronna=1e27; Yotta=1e24; Zetta=1e21; Exa=1e18; Peta=1e15
    Tera=1e12;  Giga=1e9;   Mega=1e6;   Kilo=1e3;   Hecto=1e2; Deca=1e1
    Deci=1e-1;  Centi=1e-2; Milli=1e-3; Micro=1e-6; Nano=1e-9; Pico=1e-12
    Femto=1e-15; Atto=1e-18; Zepto=1e-21; Yocto=1e-24; Ronto=1e-27; Quecto=1e-30
)
// IEC 60027-2 / ISO/IEC 80000-13 binary prefixes
const (
    Kibi=1024; Mebi=1024*Kibi; Gibi=1024*Mebi; Tebi=1024*Gibi
    Pebi=1024*Tebi; Exbi=1024*Pebi; Zebi=1024*Exbi; Yobi=1024*Zebi
)
```

---

## Tier 2 — moderately useful (additive ~50 constants, ~80 LOC)

### 2.1 Planck units (the natural-units gap; foundational 4 + 9 derived = ~20 LOC)

The single most-impactful natural-unit gap; currently zero shipped. Each
is a one-line const-arithmetic derivation analogous to `PlanckReduced`.

```go
// Planck units: c = G = ħ = k_B = 1
const (
    PlanckLength      = 1.616255e-35  // ℓ_P = √(ħG/c³)  m
    PlanckMass        = 2.176434e-8   // m_P = √(ħc/G)   kg
    PlanckTime        = 5.391247e-44  // t_P = ℓ_P/c     s
    PlanckTemperature = 1.416784e32   // T_P = m_Pc²/k_B K
    PlanckEnergy      = 1.956082e9    // E_P = m_Pc²     J  (≈ 1.22e19 GeV)
    PlanckCharge      = 1.875546e-18  // q_P = √(4πε₀ħc) C
    PlanckMomentum    = 6.524786      // p_P = m_Pc      kg·m/s
    PlanckForce       = 1.210256e44   // F_P = E_P/ℓ_P   N
    PlanckPower       = 3.628254e52   // P_P = E_P/t_P   W
    PlanckDensity     = 5.155500e96   // ρ_P = m_P/ℓ_P³  kg/m³
    PlanckPressure    = 4.633090e113  // F_P/ℓ_P²        Pa
    PlanckArea        = 2.612270e-70  // ℓ_P²            m²
    PlanckVolume      = 4.222111e-105 // ℓ_P³            m³
)
```

The four foundational (length/mass/time/temperature) are must-haves;
the rest are convenience derivations.

### 2.2 Stoney units (~6 LOC)

Older than Planck (Stoney 1881 vs Planck 1899); normalises G, c, e,
1/(4πε₀). Present in Boost.Units / Mathematica.
`StoneyLength = 1.380681e-36` (smaller than Planck length by √α),
`StoneyMass = 1.859210e-9`, `StoneyTime = 4.605375e-45`,
`StoneyEnergy = 1.671099e8`.

### 2.3 Geometrized GR conversion factors (~6 LOC)

GR convention G = c = 1. Conversion factors useful even without a full
unit-system framework: `GeometrizedMassToMeters = G/c² ≈ 7.4256e-28` m/kg
(M_⊙ → 1.477 km = R_Schwarzschild/2),
`GeometrizedTimeToMeters = c ≈ 2.998e8` (1 s = 1 light-second m),
`GeometrizedEnergyToMeters = G/c⁴ ≈ 8.262e-45` m/J.

### 2.4 Heaviside-Lorentz / CGS-Gaussian / CGS-EMU/ESU conversion factors (~10 LOC)

Full type-tagged unit-system framework belongs in a separate `units/`
package; the cross-system numerical conversions are appropriate here:
`StatCoulombsPerCoulomb = 2.99792458e9` (CGS-Gaussian; numerically 10×c),
`GaussPerTesla = 1e4` (CGS-EMU; integer exact),
`OerstedPerAmpPerMeter = 4*Pi/1000`, `MaxwellPerWeber = 1e8` (integer exact).

### 2.5 Pi convenience (~6 LOC)

`Tau = 2*Pi ≈ 6.283185307179586` (modern circle-constant name; ship
alongside `TwoPi` cross-referenced), `PiCubed`, `PiToTheFourth`,
`PiToTheFifth` (Stefan-Boltzmann), `Sqrt5` (Phi/Lucas/Fibonacci formula).

### 2.6 Particle-physics couplings + boson masses (~12 LOC, vintage-tracked)

`WeakCouplingFermi = 1.1663787e-5` GeV⁻² (G_F/(ħc)³),
`WeakMixingAngleSinSq = 0.23121` (sin²θ_W; CODATA 2022),
`StrongCouplingMz = 0.1180` (α_s(M_Z); PDG 2024),
`HiggsVacuumExpectation = 246.219651` GeV,
`WBosonMass = 80.3692` GeV/c² (CDF 2022 / PDG average),
`ZBosonMass = 91.1880`, `HiggsBosonMass = 125.20`, `TopQuarkMass = 172.69`.

(Larger uncertainty ~0.1% than CODATA atomic ~10⁻⁹; needs vintage marker.)

### 2.7 Time/epoch (~10 LOC)

`JulianDateUnixEpoch = 2440587.5`, `MJDOffset = 2400000.5`,
`B1950Epoch = 2433282.4235`, `GregorianYearDays = 365.2425`,
`JulianYearDays = 365.25`, `TropicalYearDays = 365.24219`,
`SiderealYearDays = 365.25636`, `AnomalisticYearDays = 365.259636`
(perihelion return), `DraconicYearDays = 346.620075883` (eclipse year).

---

## Tier 3 — nice-to-have / specialty (additive ~30 constants)

### 3.1 Specialty physical/chemistry (~10 LOC)

`LoschmidtConstant = 2.686780111e25` m⁻³ (n₀ at STP),
`MolarVolumeIdealGas = 22.41396954e-3` m³/mol (V_m at IUPAC STP since
1982 — T=273.15K, p=100kPa), `MolarVolumeIdealGasOldSTP = 22.413962e-3`
(legacy at 1 atm), `SackurTetrodeConstant = -1.16487052149`,
`MolarMassCarbon12 = 12.0000000e-3` kg/mol (defined Da pre-2019),
`MolarPlanck = N_A × h ≈ 3.990312712e-10` J·s/mol.

### 3.2 Math curios (~6 LOC)

`PaperFoldingConstant = 0.8507361882`,
`RamanujanSoldner = 1.4513692348833810502839` (μ; non-trivial root of
log-integral), `Plastic = 1.3247179572447460260` (ρ; real root of
x³ = x+1; silver-ratio cousin).

### 3.3 Conway / Chaitin (~4 LOC; topic-prompt requested)

`Conway = 1.303577269034296...` — limit ratio of look-and-say (flag in
docstring as a curio). Chaitin's Ω is uncomputable so it has no float64
representation — document this as deliberate omission.

### 3.4 Specialty unit conversions (~12 LOC)

`MetersPerAngstrom = 1e-10` (X-ray crystal), `MetersPerFermi = 1e-15`
(nuclear), `MetersPerMicron = 1e-6`,
`MetersPerSurveyFoot = 1200.0/3937.0` (US survey foot — differs from
international foot by 2 ppm; deprecated 2023 but in legacy GIS data),
`RadiansPerArcMinute = Pi/(180*60)`, `RadiansPerArcSecond = Pi/(180*3600)`,
`RadiansPerMilArmy = 2*Pi/6400` (NATO), `RadiansPerMilUSSR = 2*Pi/6000`
(Warsaw Pact), `RadiansPerMilDanish = 2*Pi/6300` (Danish/Swedish).

---

## Unit-system gaps (out of scope, noted)

Reality is firmly camp (a) constants-only (scipy.constants model), not
(b) type-tagged framework (Boost.Units / Frink / Wolfram `Quantity[]`).
Cross-system **conversion factors** as scalar constants remain in scope
and are covered above.

| System | Set to 1 | Status |
|---|---|---|
| SI base 7 | seven base units | shipped |
| Hartree atomic | ħ = m_e = e = 4πε₀ = 1 | missing — see 1.4 |
| Planck | c = G = ħ = k_B = 1 | missing — see 2.1 |
| Stoney | G = c = e = 1/(4πε₀) = 1 | missing — see 2.2 |
| Geometrized (GR) | G = c = 1 | missing — see 2.3 |
| Heaviside-Lorentz | rationalised CGS, c = 1 (HEP) | missing — see 2.4 |
| CGS-Gaussian / EMU / ESU | unrationalised CGS variants | missing — see 2.4 |
| Imperial vs US customary | distinct volume (gal, fl oz) | partial; distinguish needed |
| HEP natural | ħ = c = 1, energy in eV/MeV/GeV | partial via 1.8 |

---

## Summary

Tier 1 ~250 LOC: 1.1 math (30), 1.2 SI defining ΔνCs/K_cd (4), 1.3 CODATA
particle masses (25), 1.4 Hartree atomic incl. α/R∞/a₀/E_h (50),
1.5 Wien/radiation (10), 1.6 EM/condensed-matter incl. CoulombConst
extract (12), 1.7 IAU astronomical (30), 1.8 energy/power/volume/etc.
unit conversions (80), 1.9 SI/binary prefix tables (30).

Tier 2 ~80 LOC: 2.1 Planck (20), 2.2 Stoney (6), 2.3 geometrized GR (6),
2.4 Heaviside-Lorentz/CGS (10), 2.5 Tau/Pi powers/Sqrt5 (6), 2.6 SM
couplings + boson masses (12), 2.7 time/epoch (10).

Tier 3 ~30 LOC: 3.1 Loschmidt/molar/Sackur-Tetrode (10), 3.2 math curios
Plastic/Ramanujan-Soldner (6), 3.3 Conway+Chaitin doc (4), 3.4 specialty
units Å/fermi/mils/arcsec/survey-foot (12).

**Grand total:** ~373 LOC strictly additive. Fits existing `math.go`/
`physics.go`/`units.go` plus two new files (`atomic.go` for 1.4,
`astro.go` for 1.7).

**Highest-value PR bundle:** Tiers 1.1-1.9 + 2.1 ≈ **~290 LOC**, lifts
package from ~14% → ~75% of scipy.constants surface plus foundational
Planck/Hartree blocks. Critical single-line wins: `FineStructure`,
`BohrRadius`, `RydbergConstant`, `BohrMagneton`, `ElectronMass`,
`ProtonMass`, `MetersPerAU`, `MetersPerLightYear`, `PlanckLength`,
`PlanckMass`, `JoulesPerEv`, `LitersPerGallonUS`.

**Out-of-scope but noted:** type-tagged unit-system framework
(Boost.Units / Wolfram `Quantity[]` model) — design pivot for a separate
`units/` package proposal.

---

## Sources

- [scipy.constants (CODATA 2022)](https://docs.scipy.org/doc/scipy/reference/constants.html); [NIST allascii.txt](https://physics.nist.gov/cuu/Constants/Table/allascii.txt); [CODATA 2022 RMP 97 025002](https://link.aps.org/doi/10.1103/RevModPhys.97.025002), preprint [arXiv:2409.03787](https://arxiv.org/html/2409.03787v1)
- [Wikipedia Planck units](https://en.wikipedia.org/wiki/Planck_units), [Natural units](https://en.wikipedia.org/wiki/Natural_units); [Boost.Units](https://www.boost.org/doc/libs/release/doc/html/boost_units.html); [Wolfram PhysicalConstants[]](https://reference.wolfram.com/language/ref/PhysicalConstants.html)
- IAU 2012 B2 (AU exact); IAU 2015 B3 (nominal solar/Earth/Jovian); WGS84; ISO 80000-3:2019; 1959 international yard-and-pound; UK W&M Act 1985 (Imp gallon); CGPM 2022 (quetta/ronna/quecto/ronto); IEC 60027-2 (binary prefixes); PDG 2024; Fixsen 2009 ApJ 707 916 (T_CMB = 2.7255 K)

Report at `agents/047-constants-missing.md`.
