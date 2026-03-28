# Reality

Universal truth encoded in code. Pure math, physics, and constants with zero external dependencies.

## Overview

Reality is the foundational math and science library for the Limitless ecosystem. It provides deterministic, pure functions validated against cross-language golden-file test vectors (JSON).

**Module:** `github.com/davly/reality`
**License:** MIT
**Go version:** 1.24+
**External dependencies:** None

## Packages

| Package | Description |
|---------|-------------|
| `constants` | Mathematical, physical, and unit conversion constants (SI 2019, NIST CODATA 2018) |
| `testutil` | Golden-file test infrastructure for cross-language validation |

## Building

```bash
# Verify the module compiles
go build ./...
```

## Testing

```bash
# Run all tests
go test ./...

# Run with verbose output
go test -v ./...

# Run only golden-file tests
go test -run TestGolden ./...

# Run tests for a specific package
go test ./constants/
go test ./testutil/
```

## Golden-File Test Vectors

Golden files are JSON documents in `testdata/` that define expected inputs and outputs for every function. The same JSON files are used by Go, Python, C++, and C# implementations to ensure cross-language consistency.

Format:
```json
{
  "function": "Package.Function",
  "cases": [
    {
      "description": "human-readable description",
      "inputs": {"param": 1.0},
      "expected": 2.0,
      "tolerance": 1e-15
    }
  ]
}
```

Tolerance is per-case, not global. Exact constants use tolerance 0. Iterative algorithms may use wider tolerances.

## Constants Reference

### Mathematical Constants
Pi, E, Phi (golden ratio), Sqrt2, Sqrt3, Ln2, Ln10, Log2E, Log10E, EulerGamma

### Physical Constants (SI 2019 / NIST CODATA 2018)
SpeedOfLight, Planck, PlanckReduced, Boltzmann, Avogadro, ElementaryCharge, GravitationalConst, VacuumPermittivity, VacuumPermeability, StefanBoltzmann, GasConstant, StandardGravity, AtmPressure

### Unit Conversions
Length (meters per mile/foot/inch/yard/nautical mile), mass (kg per pound/ounce), temperature (Celsius/Fahrenheit to Kelvin), angle (radians/degrees), time (seconds per minute/hour/day), pressure (pascals per atm/bar/PSI)

## Design Rules

1. **Zero dependencies.** Only Go standard library.
2. **Golden files are the proof.** Every function has cross-language test vectors.
3. **Every constant cites its source.** SI 2019, NIST CODATA 2018, or ISO standards.
4. **Pure functions only.** No global state, no goroutines, numbers in / numbers out.
