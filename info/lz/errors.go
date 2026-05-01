package lz

import "errors"

// ErrTooShort is returned when the input symbol sequence has fewer than
// LZ76MinSymbols entries.  RubberDuck's reference implementation
// surfaces "too short" via a nullable return; the Go API uses a typed
// error so callers explicitly handle the degenerate-input case.
//
// The 10-symbol floor is the RubberDuck-imposed empirical minimum
// below which the asymptotic n / log_A(n) normalisation produces
// uninformative noise; preserved here for cross-substrate parity.
var ErrTooShort = errors.New("lz: symbol sequence must have at least 10 entries")

// ErrTooManyNaN is returned by ComplexityFromReturns when the input
// return series has more than 10% non-finite (NaN/+Inf/-Inf) entries.
// This mirrors the RubberDuck `ComplexityFromReturns` short-circuit
// that returns null when the symbolisation surface would be dominated
// by filtered values; preserved for cross-substrate parity.
var ErrTooManyNaN = errors.New("lz: too many non-finite values in input")

// ErrInvalidWindow is returned by RollingComplexity when windowSize
// is below LZ76MinSymbols or stepSize is non-positive.  The rolling-
// window machinery requires a per-window sample count that itself
// satisfies the LZ76 minimum-length floor.
var ErrInvalidWindow = errors.New("lz: invalid windowSize or stepSize")
