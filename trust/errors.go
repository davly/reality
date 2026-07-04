package trust

import "errors"

// ErrInvalidOpinion is returned when an Opinion's masses are out of range —
// any of b, d, u outside [0,1], base rate a outside [0,1], or the additivity
// law b+d+u = 1 violated beyond floating-point tolerance.
var ErrInvalidOpinion = errors.New("trust: opinion masses must lie in [0,1] with b+d+u = 1 and base rate a in [0,1]")

// ErrNegativeEvidence is returned when OpinionFromEvidence is given a
// negative positive/negative evidence count. Evidence tallies are
// non-negative by definition.
var ErrNegativeEvidence = errors.New("trust: evidence counts r and s must be non-negative")

// ErrInvalidMass is returned when a MassFunction is malformed: a negative
// mass, a mass on the empty set, a focal element referencing a frame index
// outside the declared frame size, or masses that do not sum to 1 within
// tolerance.
var ErrInvalidMass = errors.New("trust: mass function must assign non-negative masses summing to 1 over non-empty subsets of the frame")

// ErrFrameMismatch is returned when two MassFunctions defined over frames of
// different sizes are combined — Dempster/Yager combination requires a shared
// frame of discernment.
var ErrFrameMismatch = errors.New("trust: mass functions must share the same frame size to be combined")

// ErrTotalConflict is returned by DempsterCombine when the conflict
// coefficient K equals 1 (the two bodies of evidence are wholly
// contradictory): Dempster's normalisation divides by 1-K = 0 and the
// combined mass is undefined. The conflict K is still returned so the caller
// can see the total contradiction rather than a divide-by-zero.
var ErrTotalConflict = errors.New("trust: total conflict (K=1) — Dempster's combined mass is undefined")
