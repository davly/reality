package hrp

import "errors"

// ErrEmptyMatrix is returned when a matrix argument is nil or has zero
// rows. A distance/correlation/covariance matrix with no assets has no
// HRP problem to solve; callers receive a typed signal rather than a
// silent empty result.
var ErrEmptyMatrix = errors.New("hrp: matrix must be non-empty")

// ErrNotSquare is returned when a matrix is ragged or non-square: some
// row's length differs from the number of rows. Correlation, distance,
// and covariance matrices are all n-by-n by definition.
var ErrNotSquare = errors.New("hrp: matrix must be square")

// ErrDimensionMismatch is returned by HRPWeights when the covariance and
// correlation matrices have different dimensions. HRP clusters on corr
// and sizes on cov; the two must describe the same asset universe.
var ErrDimensionMismatch = errors.New("hrp: covariance and correlation matrices must have equal dimension")

// ErrLinkageMismatch is returned by QuasiDiagonalize when the supplied
// linkage does not describe a valid dendrogram over n leaves: the step
// count is not n-1, or a cluster id references a non-existent merge. A
// well-formed single-linkage run over n leaves always yields exactly
// n-1 steps with cluster ids in [n, 2n-2].
var ErrLinkageMismatch = errors.New("hrp: linkage does not describe a valid dendrogram over n leaves")

// ErrIndexOutOfRange is returned by RecursiveBisection when the leaf
// order references a covariance index outside [0, len(cov)) or repeats
// an index. The order must be a permutation of a subset of asset indices
// with no duplicates.
var ErrIndexOutOfRange = errors.New("hrp: leaf order contains an out-of-range or duplicate index")
