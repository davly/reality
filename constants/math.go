// Package constants provides fundamental mathematical, physical, and unit
// conversion constants to full float64 precision. Every constant documents
// its source and known precision.
//
// These constants are the single source of truth for the Reality library.
// All domain packages (linalg, calculus, physics, etc.) import from here
// rather than hardcoding values.
//
// Mathematical constants are exact to float64 precision (53-bit mantissa,
// ~15.95 decimal digits). Physics constants use SI 2019 exact definitions
// where available, and NIST CODATA 2018 recommended values otherwise.
package constants

import "math"

// Mathematical constants — exact to float64 precision.
// Source: standard mathematical definitions, values match Go's math package
// where corresponding constants exist.

// Pi is the ratio of a circle's circumference to its diameter.
// Source: mathematical definition, pi = 3.14159265358979323846...
// Precision: exact to float64 (matches math.Pi).
const Pi = math.Pi // 3.141592653589793

// E is Euler's number, the base of the natural logarithm.
// Source: mathematical definition, e = lim(n->inf) (1 + 1/n)^n
// Precision: exact to float64 (matches math.E).
const E = math.E // 2.718281828459045

// Phi is the golden ratio, (1 + sqrt(5)) / 2.
// Source: mathematical definition, phi = 1.61803398874989484820...
// Precision: exact to float64.
const Phi = 1.618033988749895

// Sqrt2 is the square root of 2.
// Source: mathematical definition, sqrt(2) = 1.41421356237309504880...
// Precision: exact to float64 (matches math.Sqrt2).
const Sqrt2 = math.Sqrt2 // 1.4142135623730951

// Sqrt3 is the square root of 3.
// Source: mathematical definition, sqrt(3) = 1.73205080756887729352...
// Precision: exact to float64.
const Sqrt3 = 1.7320508075688772

// Ln2 is the natural logarithm of 2.
// Source: mathematical definition, ln(2) = 0.69314718055994530941...
// Precision: exact to float64 (matches math.Ln2).
const Ln2 = math.Ln2 // 0.6931471805599453

// Ln10 is the natural logarithm of 10.
// Source: mathematical definition, ln(10) = 2.30258509299404568401...
// Precision: exact to float64 (matches math.Ln10).
const Ln10 = math.Ln10 // 2.302585092994046

// Log2E is the base-2 logarithm of e.
// Source: mathematical definition, log2(e) = 1/ln(2) = 1.44269504088896340735...
// Precision: exact to float64 (matches math.Log2E).
const Log2E = math.Log2E // 1.4426950408889634

// Log10E is the base-10 logarithm of e.
// Source: mathematical definition, log10(e) = 1/ln(10) = 0.43429448190325182765...
// Precision: exact to float64 (matches math.Log10E).
const Log10E = math.Log10E // 0.4342944819032518

// EulerGamma is the Euler-Mascheroni constant, the limiting difference
// between the harmonic series and the natural logarithm.
// Source: mathematical definition, gamma = lim(n->inf) (sum(1/k, k=1..n) - ln(n))
// = 0.57721566490153286060...
// Precision: exact to float64.
const EulerGamma = 0.5772156649015329
