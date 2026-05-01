// Package vibration provides primitives for mechanical vibration analysis:
// fundamental-frequency detection from a windowed FFT and harmonic-energy
// ratio (the fraction of total band-power that lies in narrow bands around
// integer multiples of a known fundamental).
//
// Mechanical signals carry harmonic structure that is more deterministic
// than biological vocalisations: a bearing has a known fundamental
// frequency tied to (RPM × ball count); a fan blade has a fundamental at
// (RPM × blade count). These primitives complement reality/audio's
// MFCC + Welford fingerprint primitives by giving consumers a physics-
// grounded Layer-0 path that does not require statistical convergence.
//
// All functions are deterministic, use only the Go standard library and
// reality/signal (FFT), and target zero allocations in hot paths. Frame-
// level operations write into caller-provided output slices.
//
// This package is the substrate-extraction outcome of the 2026-05-01
// Dipstick reference forge — the FundamentalHz / HarmonicEnergyRatio
// primitives originally lived at flagships/dipstick/reference/forge/
// vibration.go and were promoted here once the SHARED-ENGINE-DUAL-BRAND
// R-pattern reached its first instantiated consumer (Dipstick); the
// pattern goes to 2/3 once Fleetworks Torque (FleetWorks fleet sister)
// wires the same primitives, and to 3/3 once a third consumer (e.g.
// Pigeonhole's wing-beat-frequency detection or Howler's purr-rate
// fundamental) lands. See:
//
//   - flagships/dipstick/docs/INSIGHTS.md §3 — SHARED-ENGINE-DUAL-BRAND
//   - LimitlessGodfather/reviews/NEW_FLAGSHIPS_COHORT_2026-05-01.md §13.I
//
// Reference: standard DSP — Smith 1997, "Scientist and Engineer's Guide
// to Digital Signal Processing" §11; standard machine-condition monitoring
// — Cempel & Tabaszewski 2007 "Multidimensional condition monitoring of
// machines in non-stationary operation".
//
// Consumed by:
//   - flagships/dipstick (mechanical failure modes — substrate pioneer)
//   - flagships/fleetworks-torque (commercial fleet sister, planned)
//
// Cross-substrate parity: golden test vectors generated from this
// package will become the canonical reference. Native ports (Kotlin
// Android, Swift iOS) are validated against the golden vectors at
// ≤1e-9 closed-form precision and ≤1e-7 quadrature precision, matching
// the reality/audio + reality/prob/conformal substrate-shipped-first /
// consumers-port-against-golden pattern.
package vibration
