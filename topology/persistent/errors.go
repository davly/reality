package persistent

import "errors"

// ErrEmptyPoints is returned when a point cloud passed to
// VietorisRipsComplex is empty or nil.  A persistent-homology
// computation requires at least one point; the empty filtration is
// degenerate (no simplices, no bars, no diagram).  Reject early so
// callers receive a typed signal instead of a quiet zero diagram.
var ErrEmptyPoints = errors.New("persistent: points must be non-empty")

// ErrInvalidMaxDim is returned when maxDim is negative.  Negative
// homological dimensions are not defined.  The Phase-A scope of this
// package supports maxDim in {0, 1}; callers that pass maxDim >= 2
// receive the same error because higher-dimensional persistent
// homology requires the matrix-reduction path against the full
// boundary of the (maxDim+1)-skeleton, which is deferred to v2 along
// with persistent cohomology, landscape, Wasserstein, and Mapper.
var ErrInvalidMaxDim = errors.New("persistent: maxDim must be 0 or 1 in v1")

// ErrInconsistentDim is returned when the points slice contains rows
// of differing length, or rows of length zero.  Vietoris-Rips uses an
// L^2 (Euclidean) metric on R^d; a ragged input has no well-defined
// dimension, so we reject the slice rather than silently coercing.
var ErrInconsistentDim = errors.New("persistent: every point must have the same positive dimension")

// ErrInvalidMaxRadius is returned when maxRadius is non-finite or
// negative.  The Vietoris-Rips filtration is parameterised by a
// non-negative scale parameter; +Inf is not a valid radius cap because
// the simplicial complex would then be the full (n-1)-simplex on n
// points, blowing up combinatorially.  Callers that genuinely want
// "no cap" should pass the diameter of the point cloud (or any
// sufficiently large finite number).
var ErrInvalidMaxRadius = errors.New("persistent: maxRadius must be a non-negative finite number")
