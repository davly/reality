package prob

import (
	"errors"
	"math"
)

// ---------------------------------------------------------------------------
// Time-series forecasting primitives.
//
// Every function is pure, deterministic, and uses only the Go standard
// library. No external dependencies.
//
// Consumers:
//   - Pulse:     predictive monitoring (exponential smoothing, ARIMA)
//   - Oracle:    time-series prediction calibration (Holt linear)
//   - Horizon:   personal analytics forecasting (all three)
//   - RubberDuck: financial time-series (ARIMA, exponential smoothing)
// ---------------------------------------------------------------------------

// ExponentialSmoothing applies simple exponential smoothing (SES) to data
// and writes the smoothed values into out. The out slice must be at least
// as long as data.
//
// Formula:
//
//	s[0] = data[0]
//	s[t] = alpha * data[t] + (1 - alpha) * s[t-1]
//
// Valid range: alpha in (0, 1], len(data) >= 1, len(out) >= len(data)
// Precision: exact (multiplications and additions only)
// Failure mode: panics if out is shorter than data (caller responsibility)
// Reference: Brown, R.G. (1956) "Exponential Smoothing for Predicting Demand"
func ExponentialSmoothing(data []float64, alpha float64, out []float64) {
	if len(data) == 0 {
		return
	}
	if alpha <= 0 {
		alpha = 0.01
	}
	if alpha > 1 {
		alpha = 1.0
	}
	out[0] = data[0]
	for t := 1; t < len(data); t++ {
		out[t] = alpha*data[t] + (1-alpha)*out[t-1]
	}
}

// HoltLinear applies Holt's linear trend method (double exponential smoothing)
// and produces a forecast of length horizon appended to the smoothed series.
// The output slice out must have capacity for len(data) + horizon elements.
//
// Formula:
//
//	level[0] = data[0]
//	trend[0] = data[1] - data[0]  (if len(data) >= 2, else 0)
//	level[t] = alpha * data[t] + (1 - alpha) * (level[t-1] + trend[t-1])
//	trend[t] = beta * (level[t] - level[t-1]) + (1 - beta) * trend[t-1]
//	forecast[h] = level[n-1] + h * trend[n-1]    for h = 1..horizon
//
// Valid range: alpha in (0, 1], beta in (0, 1], len(data) >= 1, horizon >= 0
//
//	len(out) >= len(data) + horizon
//
// Precision: exact (multiplications and additions only)
// Reference: Holt, C.C. (1957) "Forecasting Seasonals and Trends by
// Exponentially Weighted Moving Averages"
func HoltLinear(data []float64, alpha, beta float64, horizon int, out []float64) {
	n := len(data)
	if n == 0 {
		return
	}
	if alpha <= 0 {
		alpha = 0.01
	}
	if alpha > 1 {
		alpha = 1.0
	}
	if beta <= 0 {
		beta = 0.01
	}
	if beta > 1 {
		beta = 1.0
	}
	if horizon < 0 {
		horizon = 0
	}

	level := data[0]
	trend := 0.0
	if n >= 2 {
		trend = data[1] - data[0]
	}
	out[0] = level

	for t := 1; t < n; t++ {
		prevLevel := level
		level = alpha*data[t] + (1-alpha)*(prevLevel+trend)
		trend = beta*(level-prevLevel) + (1-beta)*trend
		out[t] = level
	}

	// Forecast beyond observed data.
	for h := 1; h <= horizon; h++ {
		out[n-1+h] = level + float64(h)*trend
	}
}

// ARIMA fits a simplified ARIMA(p, d, q) model to data and returns the
// estimated AR and MA coefficients. The model first differences the series
// d times, then fits AR(p) coefficients using the Yule-Walker method, and
// MA(q) coefficients from the residual autocorrelations.
//
// Returns a coefficient slice of length p + q:
//
//	coefficients[0..p-1] are AR coefficients (phi_1 .. phi_p)
//	coefficients[p..p+q-1] are MA coefficients (theta_1 .. theta_q)
//
// Valid range: p >= 0, d >= 0, q >= 0, len(data) > p + d + 1
// Failure mode: returns error if data is too short or parameters invalid
// Precision: limited by autocorrelation estimation and Levinson-Durbin solver
// Reference: Box, G.E.P. & Jenkins, G.M. (1970) "Time Series Analysis:
// Forecasting and Control"
func ARIMA(data []float64, p, d, q int) ([]float64, error) {
	if p < 0 || d < 0 || q < 0 {
		return nil, errors.New("prob.ARIMA: p, d, q must be non-negative")
	}
	if p == 0 && q == 0 {
		return []float64{}, nil
	}

	// Step 1: Differencing. Apply d-th order differencing.
	series := make([]float64, len(data))
	copy(series, data)
	for i := 0; i < d; i++ {
		if len(series) < 2 {
			return nil, errors.New("prob.ARIMA: data too short after differencing")
		}
		diff := make([]float64, len(series)-1)
		for j := 0; j < len(diff); j++ {
			diff[j] = series[j+1] - series[j]
		}
		series = diff
	}

	n := len(series)
	if n <= p {
		return nil, errors.New("prob.ARIMA: differenced series too short for AR order p")
	}

	// Step 2: Compute mean and center the series.
	mean := 0.0
	for _, v := range series {
		mean += v
	}
	mean /= float64(n)
	centered := make([]float64, n)
	for i, v := range series {
		centered[i] = v - mean
	}

	// Step 3: Compute autocorrelations up to max(p, q) lags.
	maxLag := p
	if q > maxLag {
		maxLag = q
	}
	if maxLag == 0 {
		return []float64{}, nil
	}

	// Autocorrelation at lag k.
	autocorr := make([]float64, maxLag+1)
	for k := 0; k <= maxLag; k++ {
		sum := 0.0
		for i := k; i < n; i++ {
			sum += centered[i] * centered[i-k]
		}
		autocorr[k] = sum / float64(n)
	}

	coefficients := make([]float64, p+q)

	// Step 4: Fit AR(p) using Levinson-Durbin recursion.
	if p > 0 && autocorr[0] > 0 {
		ar := levinsonDurbin(autocorr, p)
		copy(coefficients[:p], ar)
	}

	// Step 5: Fit MA(q) from residual autocorrelations.
	if q > 0 && autocorr[0] > 0 {
		// Compute residuals from AR fit.
		residuals := make([]float64, n)
		for t := 0; t < n; t++ {
			predicted := 0.0
			for j := 0; j < p; j++ {
				if t-j-1 >= 0 {
					predicted += coefficients[j] * centered[t-j-1]
				}
			}
			residuals[t] = centered[t] - predicted
		}

		// Autocorrelation of residuals.
		resVar := 0.0
		for _, r := range residuals {
			resVar += r * r
		}
		resVar /= float64(n)

		if resVar > 0 {
			for k := 1; k <= q; k++ {
				sum := 0.0
				for i := k; i < n; i++ {
					sum += residuals[i] * residuals[i-k]
				}
				// MA coefficient estimated from residual autocorrelation.
				coefficients[p+k-1] = sum / (float64(n) * resVar)
			}
		}
	}

	return coefficients, nil
}

// levinsonDurbin solves the Yule-Walker equations using Levinson-Durbin
// recursion to obtain AR coefficients from autocorrelations.
//
// autocorr[0] is the variance (lag 0), autocorr[1..p] are lag autocorrelations.
// Returns the p AR coefficients.
//
// Reference: Levinson, N. (1946); Durbin, J. (1960)
func levinsonDurbin(autocorr []float64, p int) []float64 {
	if p == 0 || autocorr[0] == 0 {
		return make([]float64, p)
	}

	a := make([]float64, p)
	e := autocorr[0]

	for i := 0; i < p; i++ {
		// Compute reflection coefficient.
		lambda := autocorr[i+1]
		for j := 0; j < i; j++ {
			lambda -= a[j] * autocorr[i-j]
		}
		lambda /= e

		// Clamp to prevent divergence.
		if lambda > 1 {
			lambda = 1
		}
		if lambda < -1 {
			lambda = -1
		}

		// Update coefficients.
		newA := make([]float64, p)
		for j := 0; j < i; j++ {
			newA[j] = a[j] - lambda*a[i-1-j]
		}
		newA[i] = lambda
		copy(a, newA)

		e *= (1 - lambda*lambda)
		if e <= 0 {
			break
		}
	}

	// Negate to match standard AR convention: x[t] = sum(phi_i * x[t-i]) + eps.
	// Levinson-Durbin gives coefficients for the prediction filter, which is
	// what we want (phi_i), so no negation needed.
	result := make([]float64, p)
	copy(result, a)
	return result
}

// arimaAutocovariance computes sample autocovariance at lag k.
// Not exported; used internally by ARIMA.
func arimaAutocovariance(data []float64, mean float64, k int) float64 {
	n := len(data)
	if k >= n || k < 0 {
		return 0
	}
	sum := 0.0
	for i := k; i < n; i++ {
		sum += (data[i] - mean) * (data[i-k] - mean)
	}
	return sum / float64(n)
}

// Sigmoid returns the logistic sigmoid function: 1 / (1 + exp(-x)).
// Used internally by logistic regression and other functions.
//
// Formula: 1 / (1 + exp(-x))
// Valid range: all float64
// Output range: (0, 1)
// Precision: ~15 significant digits (float64)
func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1.0 / (1.0 + math.Exp(-x))
	}
	// For negative x, use the equivalent form to avoid overflow in exp(|x|).
	ex := math.Exp(x)
	return ex / (1.0 + ex)
}
