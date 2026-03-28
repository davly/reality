package prob

import (
	"math"
	"sort"
)

// MinProb is the minimum allowed probability. Probabilities are clamped to
// [MinProb, MaxProb] to prevent log(0) or division-by-zero in log-odds
// conversions. This encodes the epistemic principle that absolute certainty
// (0% or 100%) is never warranted.
const MinProb = 0.01

// MaxProb is the maximum allowed probability. See MinProb.
const MaxProb = 0.99

// ---------------------------------------------------------------------------
// Clamping & conversion
// ---------------------------------------------------------------------------

// ClampProbability clamps p to the interval [MinProb, MaxProb].
//
// Formula: max(MinProb, min(MaxProb, p))
// Valid range: any float64 (including NaN — returns MinProb)
// Precision: exact
// Reference: standard numerical safeguard for log-odds computation
func ClampProbability(p float64) float64 {
	return math.Max(MinProb, math.Min(MaxProb, p))
}

// ProbToLogOdds converts a probability to log-odds (logit function).
//
// Formula: log(p / (1 - p))
// Valid range: p in (0, 1); input is clamped to [MinProb, MaxProb]
// Precision: limited by float64 log; ~15 significant digits
// Reference: logit function, standard in Bayesian forecasting
func ProbToLogOdds(p float64) float64 {
	p = ClampProbability(p)
	return math.Log(p / (1 - p))
}

// LogOddsToProb converts log-odds back to a probability (logistic function).
//
// Formula: 1 / (1 + exp(-logOdds))
// Valid range: any float64 log-odds; result clamped to [MinProb, MaxProb]
// Precision: limited by float64 exp; ~15 significant digits
// Reference: logistic (sigmoid) function
func LogOddsToProb(logOdds float64) float64 {
	return ClampProbability(1.0 / (1.0 + math.Exp(-logOdds)))
}

// ---------------------------------------------------------------------------
// Bayesian updating
// ---------------------------------------------------------------------------

// BayesianUpdate applies Bayes' theorem in log-odds space for numerical stability.
//
// Formula: posterior = logistic(logit(prior) + log(likelihoodRatio))
// Valid range: prior in (0,1), likelihoodRatio > 0
// Precision: ~15 significant digits (float64)
// Failure mode: if likelihoodRatio <= 0, returns prior unchanged
// Reference: Bayes' theorem via log-odds; see Jaynes (2003) "Probability Theory"
func BayesianUpdate(prior, likelihoodRatio float64) float64 {
	if likelihoodRatio <= 0 {
		return prior
	}
	priorLogOdds := ProbToLogOdds(prior)
	posteriorLogOdds := priorLogOdds + math.Log(likelihoodRatio)
	return LogOddsToProb(posteriorLogOdds)
}

// BayesianUpdateChain applies a sequence of Bayesian updates in order.
// Each likelihood ratio is applied sequentially to the running posterior.
//
// Formula: fold BayesianUpdate over likelihoodRatios
// Valid range: prior in (0,1), each lr > 0
// Precision: accumulated float64 error across chain length
// Reference: sequential Bayesian inference
func BayesianUpdateChain(prior float64, likelihoodRatios []float64) float64 {
	p := prior
	for _, lr := range likelihoodRatios {
		p = BayesianUpdate(p, lr)
	}
	return p
}

// ---------------------------------------------------------------------------
// Scoring rules
// ---------------------------------------------------------------------------

// BrierScore computes the Brier score for a single prediction.
//
// Formula: (predicted - actual)^2
// Valid range: predicted in [0,1], actual in {0, 1}
// Output range: [0.0 (perfect), 0.25 (random at 0.5), 1.0 (maximally wrong)]
// Precision: exact (single multiply)
// Reference: Brier, G.W. (1950) "Verification of Forecasts Expressed in Terms of Probability"
func BrierScore(predicted, actual float64) float64 {
	diff := predicted - actual
	return diff * diff
}

// BrierScoreBatch computes the mean Brier score for a batch of predictions.
// Returns 0 if the slices are empty or have mismatched lengths.
//
// Formula: (1/n) * sum_i (predictions[i] - actuals[i])^2
// Precision: accumulated float64 summation error
// Reference: Brier (1950)
func BrierScoreBatch(predictions, actuals []float64) float64 {
	if len(predictions) == 0 || len(predictions) != len(actuals) {
		return 0
	}
	sum := 0.0
	for i := range predictions {
		sum += BrierScore(predictions[i], actuals[i])
	}
	return sum / float64(len(predictions))
}

// LogLoss computes the logarithmic loss (cross-entropy) for a single prediction.
// The predicted value is clamped to [MinProb, MaxProb] before computation to
// prevent log(0).
//
// Formula: -(actual * log(predicted) + (1-actual) * log(1-predicted))
// Simplified: -log(p) if actual >= 0.5, -log(1-p) otherwise
// Valid range: predicted in (0,1), actual in {0, 1}
// Output range: [0.01005.. (perfect), +inf theoretical; clamped to finite]
// Precision: limited by float64 log
// Reference: standard cross-entropy loss
func LogLoss(predicted, actual float64) float64 {
	p := ClampProbability(predicted)
	if actual >= 0.5 {
		return -math.Log(p)
	}
	return -math.Log(1.0 - p)
}

// LogLossBatch computes the mean log-loss for a batch of predictions.
// Returns 0 if the slices are empty or have mismatched lengths.
//
// Formula: (1/n) * sum_i LogLoss(predictions[i], actuals[i])
// Precision: accumulated float64 summation and log error
// Reference: standard cross-entropy loss
func LogLossBatch(predictions, actuals []float64) float64 {
	if len(predictions) == 0 || len(predictions) != len(actuals) {
		return 0
	}
	sum := 0.0
	for i := range predictions {
		sum += LogLoss(predictions[i], actuals[i])
	}
	return sum / float64(len(predictions))
}

// ---------------------------------------------------------------------------
// Probability aggregation
// ---------------------------------------------------------------------------

// LogOddsPool aggregates multiple probability estimates using a weighted
// log-odds linear pool. This respects the geometry of probability space
// better than a simple arithmetic average.
//
// Formula: logistic( sum(w_i * logit(p_i)) / sum(w_i) )
// Valid range: each p_i in (0,1), weights >= 0
// If weights is nil or empty, equal weights are used.
// Returns 0.5 if no valid inputs (empty slice or all weights <= 0).
// Precision: limited by float64 log/exp
// Reference: log-odds linear opinion pool; Satopaa et al. (2014)
func LogOddsPool(probabilities, weights []float64) float64 {
	if len(probabilities) == 0 {
		return 0.5
	}
	if len(weights) == 0 {
		weights = make([]float64, len(probabilities))
		for i := range weights {
			weights[i] = 1.0
		}
	}

	totalWeight := 0.0
	weightedLogOdds := 0.0

	for i, p := range probabilities {
		w := 1.0
		if i < len(weights) {
			w = weights[i]
		}
		if w <= 0 {
			continue
		}
		totalWeight += w
		weightedLogOdds += w * ProbToLogOdds(p)
	}

	if totalWeight == 0 {
		return 0.5
	}

	return LogOddsToProb(weightedLogOdds / totalWeight)
}

// WilsonConfidenceInterval computes the Wilson score interval for a binomial
// proportion given a number of observations.
//
// Formula:
//
//	centre = (p + z^2/(2n)) / (1 + z^2/n)
//	margin = z * sqrt((p(1-p) + z^2/(4n)) / n) / (1 + z^2/n)
//	interval = [centre - margin, centre + margin]
//
// Valid range: p in [0,1], n > 0, z > 0
// If n <= 0, returns a wide interval [p-0.3, p+0.3] clamped to [MinProb, MaxProb].
// If z <= 0, defaults to z = 1.96 (95% confidence).
// Precision: limited by float64 sqrt
// Reference: Wilson, E.B. (1927) "Probable Inference, the Law of Succession,
// and Statistical Inference"
func WilsonConfidenceInterval(p float64, n int, z float64) (low, high float64) {
	if n <= 0 {
		return ClampProbability(p - 0.3), ClampProbability(p + 0.3)
	}
	if z <= 0 {
		z = 1.96
	}

	nf := float64(n)
	z2 := z * z
	denominator := 1.0 + z2/nf
	centre := (p + z2/(2.0*nf)) / denominator
	margin := (z * math.Sqrt((p*(1.0-p)+z2/(4.0*nf))/nf)) / denominator

	return ClampProbability(centre - margin), ClampProbability(centre + margin)
}

// SimpleAverage computes the arithmetic mean of values.
// Returns 0.5 if the slice is empty. Result is clamped to [MinProb, MaxProb].
//
// Formula: (1/n) * sum(values)
// Precision: accumulated float64 summation error
func SimpleAverage(values []float64) float64 {
	if len(values) == 0 {
		return 0.5
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return ClampProbability(sum / float64(len(values)))
}

// WeightedAverage computes a weighted arithmetic mean of values.
// Returns 0.5 if the slice is empty or all weights are zero/negative.
// If weights is shorter than values, missing weights default to 1.0.
//
// Formula: sum(w_i * v_i) / sum(w_i)
// Precision: accumulated float64 summation error
func WeightedAverage(values, weights []float64) float64 {
	if len(values) == 0 {
		return 0.5
	}

	totalWeight := 0.0
	weightedSum := 0.0

	for i, v := range values {
		w := 1.0
		if i < len(weights) {
			w = weights[i]
		}
		if w <= 0 {
			continue
		}
		totalWeight += w
		weightedSum += w * v
	}

	if totalWeight == 0 {
		return 0.5
	}

	return ClampProbability(weightedSum / totalWeight)
}

// Median computes the median of values.
// Returns 0.5 if the slice is empty. Result is clamped to [MinProb, MaxProb].
// The input slice is not modified (a copy is sorted internally).
//
// Formula: middle value (odd n) or average of two middle values (even n)
// Precision: exact for odd n; one addition + division for even n
func Median(values []float64) float64 {
	if len(values) == 0 {
		return 0.5
	}

	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	n := len(sorted)
	if n%2 == 0 {
		return ClampProbability((sorted[n/2-1] + sorted[n/2]) / 2.0)
	}
	return ClampProbability(sorted[n/2])
}

// TrimmedMean computes a trimmed (truncated) mean by removing the top and
// bottom fractions before averaging.
//
// trimFraction: fraction to trim from each end (e.g., 0.1 trims 10% from each end).
// Returns 0.5 if the slice is empty. Result is clamped to [MinProb, MaxProb].
// If trimFraction < 0 or >= 0.5, no trimming is applied.
// The input slice is not modified (a copy is sorted internally).
//
// Formula: mean of values[k..n-k] where k = floor(n * trimFraction)
// Precision: accumulated float64 summation error
// Reference: Wilcox, R.R. (2012) "Introduction to Robust Estimation and
// Hypothesis Testing"
func TrimmedMean(values []float64, trimFraction float64) float64 {
	if len(values) == 0 {
		return 0.5
	}
	if trimFraction < 0 || trimFraction >= 0.5 {
		trimFraction = 0
	}

	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	trimCount := int(math.Floor(float64(len(sorted)) * trimFraction))
	if trimCount*2 >= len(sorted) {
		trimCount = 0
	}

	trimmed := sorted[trimCount : len(sorted)-trimCount]
	if len(trimmed) == 0 {
		return 0.5
	}

	sum := 0.0
	for _, v := range trimmed {
		sum += v
	}
	return ClampProbability(sum / float64(len(trimmed)))
}

// ---------------------------------------------------------------------------
// Calibration metrics
// ---------------------------------------------------------------------------

// ExpectedCalibrationError computes the expected calibration error (ECE),
// which is the weighted mean absolute calibration error across equal-width
// probability bins.
//
// Formula: sum_b (|B_b|/N) * |meanPredicted_b - meanActual_b|
// where B_b is the set of predictions in bucket b, N is total predictions.
// Valid range: numBuckets >= 1 (defaults to 10 if < 1)
// Output range: [0 (perfect calibration), ~1 (worst)]
// Returns 0 if predictions is empty.
// Precision: limited by binning granularity
// Reference: Naeini, Cooper, Hauskrecht (2015) "Obtaining Well Calibrated
// Probabilities Using Bayesian Binning into Quantiles"
func ExpectedCalibrationError(predictions []PredictionOutcome, numBuckets int) float64 {
	buckets := ReliabilityDiagram(predictions, numBuckets)
	total := 0
	for _, b := range buckets {
		total += b.Count
	}
	if total == 0 {
		return 0
	}
	var ece float64
	for _, b := range buckets {
		if b.Count > 0 {
			ece += float64(b.Count) / float64(total) * math.Abs(b.MeanPredicted-b.MeanActual)
		}
	}
	return ece
}

// MaximumCalibrationError returns the maximum bucket calibration error (MCE).
// This is the worst-case miscalibration across all non-empty bins.
//
// Formula: max_b |meanPredicted_b - meanActual_b| for non-empty buckets
// Valid range: numBuckets >= 1 (defaults to 10 if < 1)
// Returns 0 if predictions is empty.
// Reference: Naeini et al. (2015)
func MaximumCalibrationError(predictions []PredictionOutcome, numBuckets int) float64 {
	buckets := ReliabilityDiagram(predictions, numBuckets)
	var mce float64
	for _, b := range buckets {
		if b.Count > 0 {
			err := math.Abs(b.MeanPredicted - b.MeanActual)
			if err > mce {
				mce = err
			}
		}
	}
	return mce
}

// ReliabilityDiagram computes a bucketed reliability diagram from predictions.
// Predictions are binned into numBuckets equal-width bins by their Predicted
// value. Within each bin, the mean predicted probability and mean observed
// outcome rate are computed.
//
// numBuckets must be >= 1; defaults to 10 if < 1.
// Returns one DiagramBucket per bin (including empty bins with Count == 0).
//
// Reference: DeGroot, Fienberg (1983) "The Comparison and Evaluation of
// Forecasters"
func ReliabilityDiagram(predictions []PredictionOutcome, numBuckets int) []DiagramBucket {
	if numBuckets < 1 {
		numBuckets = 10
	}

	buckets := make([]DiagramBucket, numBuckets)
	width := 1.0 / float64(numBuckets)

	for _, p := range predictions {
		idx := int(p.Predicted / width)
		if idx >= numBuckets {
			idx = numBuckets - 1
		}
		if idx < 0 {
			idx = 0
		}
		buckets[idx].MeanPredicted += p.Predicted
		buckets[idx].MeanActual += p.Actual
		buckets[idx].Count++
	}

	for i := range buckets {
		if buckets[i].Count > 0 {
			buckets[i].MeanPredicted /= float64(buckets[i].Count)
			buckets[i].MeanActual /= float64(buckets[i].Count)
		}
	}

	return buckets
}

// IsotonicRegression applies the Pool-Adjacent-Violators (PAV) algorithm
// to produce a monotonically non-decreasing calibration mapping.
//
// The input points must be sorted by X ascending. The algorithm merges
// adjacent blocks whenever monotonicity is violated, replacing their Y
// values with the block average. This produces the optimal (in a
// least-squares sense) monotone approximation.
//
// Returns a new slice of CalibrationPoint with the same X values but
// adjusted Y values that are monotonically non-decreasing. Returns nil
// if points is empty. The input slice is not modified.
//
// Time complexity: O(n), single pass with stack-based merging.
// Reference: Barlow, Bartholomew, Bremner, Brunk (1972) "Statistical
// Inference Under Order Restrictions"
func IsotonicRegression(points []CalibrationPoint) []CalibrationPoint {
	n := len(points)
	if n == 0 {
		return nil
	}

	// Copy to avoid mutating input.
	result := make([]CalibrationPoint, n)
	copy(result, points)

	// Pool-Adjacent-Violators: merge adjacent blocks when monotonicity violated.
	type block struct {
		sum   float64
		count int
	}
	blocks := make([]block, 0, n)
	for _, p := range result {
		blocks = append(blocks, block{sum: p.Y, count: 1})
		// Merge while the last two blocks violate monotonicity.
		for len(blocks) >= 2 {
			last := blocks[len(blocks)-1]
			prev := blocks[len(blocks)-2]
			if prev.sum/float64(prev.count) > last.sum/float64(last.count) {
				// Merge: pool into previous block.
				blocks[len(blocks)-2] = block{
					sum:   prev.sum + last.sum,
					count: prev.count + last.count,
				}
				blocks = blocks[:len(blocks)-1]
			} else {
				break
			}
		}
	}

	// Expand blocks back to per-point values.
	idx := 0
	for _, b := range blocks {
		val := b.sum / float64(b.count)
		for j := 0; j < b.count; j++ {
			result[idx].Y = val
			idx++
		}
	}

	return result
}
