package constants

// Unit conversion constants — exact definitions from NIST and ISO standards.
//
// Naming convention: <Target>Per<Source> means multiply a value in <Source>
// units by this constant to get <Target> units. For example:
//
//	meters := feet * MetersPerFoot
//
// All values are exact conversions defined by international agreement,
// not measured values.

// --- Length conversions ---

// MetersPerMile is the number of meters in one international mile.
// Source: international yard and pound agreement (1959).
// 1 mile = 5280 feet = 5280 * 0.3048 m = 1609.344 m (exact).
const MetersPerMile = 1609.344

// MetersPerFoot is the number of meters in one international foot.
// Source: international yard and pound agreement (1959).
// 1 foot = 0.3048 m (exact).
const MetersPerFoot = 0.3048

// MetersPerInch is the number of meters in one inch.
// Source: derived from international foot. 1 inch = 0.0254 m (exact).
const MetersPerInch = 0.0254

// MetersPerYard is the number of meters in one international yard.
// Source: international yard and pound agreement (1959).
// 1 yard = 0.9144 m (exact).
const MetersPerYard = 0.9144

// MetersPerNauticalMile is the number of meters in one nautical mile.
// Source: International Hydrographic Organization (1929).
// 1 nautical mile = 1852 m (exact).
const MetersPerNauticalMile = 1852.0

// --- Mass conversions ---

// KgPerPound is the number of kilograms in one avoirdupois pound.
// Source: international yard and pound agreement (1959).
// 1 lb = 0.45359237 kg (exact).
const KgPerPound = 0.45359237

// KgPerOunce is the number of kilograms in one avoirdupois ounce.
// Source: derived from pound. 1 oz = 1/16 lb = 0.028349523125 kg (exact).
const KgPerOunce = 0.028349523125

// --- Temperature conversions ---

// CelsiusToKelvin is the offset to convert Celsius to Kelvin.
// Source: SI definition. T(K) = T(C) + 273.15 (exact).
// Usage: kelvin := celsius + CelsiusToKelvin
const CelsiusToKelvin = 273.15

// FahrenheitToKelvinOffset is the offset component for Fahrenheit to Kelvin.
// T(K) = (T(F) + 459.67) * 5/9. This constant is the additive offset.
// Source: derived from exact Celsius/Fahrenheit relationship.
const FahrenheitToKelvinOffset = 459.67

// FahrenheitToKelvinScale is the scale factor for Fahrenheit to Kelvin.
// T(K) = (T(F) + 459.67) * FahrenheitToKelvinScale.
// Source: exact definition. 5/9.
const FahrenheitToKelvinScale = 5.0 / 9.0

// --- Angle conversions ---

// RadiansToDegrees converts radians to degrees.
// Source: mathematical definition. 1 rad = 180/pi degrees.
// Precision: limited by float64 representation of pi.
const RadiansToDegrees = 180.0 / Pi

// DegreesToRadians converts degrees to radians.
// Source: mathematical definition. 1 degree = pi/180 radians.
// Precision: limited by float64 representation of pi.
const DegreesToRadians = Pi / 180.0

// --- Time conversions ---

// SecondsPerMinute is the number of seconds in one minute (exact).
const SecondsPerMinute = 60.0

// SecondsPerHour is the number of seconds in one hour (exact).
const SecondsPerHour = 3600.0

// SecondsPerDay is the number of seconds in one mean solar day (exact).
const SecondsPerDay = 86400.0

// --- Pressure conversions ---

// PascalsPerAtm is the number of pascals in one standard atmosphere.
// Source: ISO 80000-3:2019 exact definition. 1 atm = 101325 Pa.
const PascalsPerAtm = 101325.0

// PascalsPerBar is the number of pascals in one bar (exact).
const PascalsPerBar = 100000.0

// PascalsPerPSI is the number of pascals in one pound per square inch.
// Source: derived from exact definitions. 1 psi = 6894.757293168... Pa.
const PascalsPerPSI = 6894.757293168361
