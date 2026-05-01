package mdl

import "errors"

// ErrEmptyCounts is returned by NMLMultinomial when the supplied
// count vector is empty (zero categories).  The Kontkanen-Myllymäki
// 2007 NML recursion is undefined for k = 0; callers receive a
// typed signal rather than a quiet NaN.
var ErrEmptyCounts = errors.New("mdl: count vector must be non-empty")

// ErrNegativeInput is returned when an integer or count input is
// negative (e.g. negative successes / trials supplied to
// NMLBernoulli, or negative entry in the multinomial count vector).
// The NML formulation is undefined for negative observations.
var ErrNegativeInput = errors.New("mdl: input must be non-negative")

// ErrInvalidTrials is returned by NMLBernoulli when successes >
// trials, or when trials is zero (the codelength of zero
// observations is degenerate — the empty model trivially compresses
// the empty string to zero bits).
var ErrInvalidTrials = errors.New("mdl: invalid (successes, trials) pair")

// ErrInvalidUniversalInt is returned by UniversalIntegerCodeLength
// when n is non-positive.  Rissanen's universal prior is defined
// for positive integers n >= 1; the codelength of n = 0 or negative
// is undefined.
var ErrInvalidUniversalInt = errors.New("mdl: universal integer code requires n >= 1")

// ErrEmptyModelList is returned by SelectMDL when the supplied
// model-codelength slice is empty.  Argmin over the empty set is
// undefined.
var ErrEmptyModelList = errors.New("mdl: model list must be non-empty")

// ErrNonFiniteCodeLength is returned by SelectMDL when one or more
// supplied codelengths are non-finite (NaN, ±Inf).  A non-finite
// codelength signals an upstream numerical-instability bug; rather
// than silently propagating NaN through the argmin, the function
// surfaces the typed signal.
var ErrNonFiniteCodeLength = errors.New("mdl: non-finite codelength in model list")
