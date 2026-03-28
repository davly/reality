// Package prob provides probability and statistics primitives extracted from
// the Oracle calibrated prediction engine and the aicore calibration package.
//
// Every function is pure, deterministic, and uses only the Go standard library.
// No external dependencies. All numerical routines cite their mathematical
// provenance and document their valid input ranges, precision guarantees, and
// failure modes.
//
// Extracted from:
//
//	aicore/oraclemath  (Bayesian updating, scoring, aggregation)
//	aicore/calibration (ECE/MCE, reliability diagrams, isotonic regression)
package prob

// PredictionOutcome records a single forecasted probability paired with its
// binary outcome. This is the fundamental unit for calibration analysis.
//
// Predicted must be in [0, 1]. Actual must be 0.0 (false) or 1.0 (true).
type PredictionOutcome struct {
	Predicted float64 // forecasted probability [0, 1]
	Actual    float64 // binary outcome: 0.0 or 1.0
}

// CalibrationPoint is an (x, y) observation for isotonic regression.
// X is the raw score (e.g., predicted confidence) and Y is the observed
// outcome (e.g., 0 or 1, or an observed rate for binned data).
type CalibrationPoint struct {
	X float64 // raw score / predicted confidence
	Y float64 // observed outcome / rate
}

// DiagramBucket holds one bin of a reliability diagram.
//
// A reliability diagram partitions predictions into equal-width bins by their
// predicted probability, then computes the mean predicted and mean actual
// (observed) rate within each bin. A perfectly calibrated forecaster has
// MeanPredicted == MeanActual for every bucket.
type DiagramBucket struct {
	MeanPredicted float64 // average predicted probability in this bin
	MeanActual    float64 // observed outcome rate in this bin
	Count         int     // number of predictions in this bin
}
