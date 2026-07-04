package statespace

import "math"

// FilterResult holds the output of a forward Kalman pass over a series. The
// predicted quantities (before the measurement update at each step) are
// retained because the Rauch-Tung-Striebel smoother needs them.
//
// All per-step matrices are flat row-major n×n slices stored back to back:
// step t occupies indices [t*n*n : (t+1)*n*n]. Means are stored back to back
// as length-n vectors.
type FilterResult struct {
	N int // state dimension
	T int // number of steps

	// FilteredMean[t] (length n) is E[x_t | z_{1:t}]; FilteredCov[t] its cov.
	FilteredMean []float64 // T*n
	FilteredCov  []float64 // T*n*n

	// PredictedMean[t] (length n) is E[x_t | z_{1:t-1}] (the prior at step t
	// after the time update); PredictedCov[t] its covariance. For t=0 these
	// equal the supplied initial mean/cov.
	PredictedMean []float64 // T*n
	PredictedCov  []float64 // T*n*n

	// LogLikelihood is the total Gaussian log-likelihood of the series,
	// sum_t log N(z_t; H x_{t|t-1}, S_t) — the prediction-error decomposition.
	LogLikelihood float64
}

// Filter runs the Kalman filter forward over an m-dimensional observation
// series. obs is a flat row-major T×m slice (observation t at [t*m:(t+1)*m]).
// x0/P0 are the initial state mean (length n) and covariance (n×n). F, Q are
// the n×n transition and process-noise matrices; H is the m×n observation
// matrix; R the m×m observation-noise covariance.
//
// The state at step t is defined so that the prediction for step 0 is the
// prior (x0, P0) itself (no time update precedes the first observation), and a
// time update is applied between steps. Returns ErrEmptyData for T==0 and
// ErrDimension / ErrSingular on the underlying primitives' failures.
func Filter(obs, x0, P0, F, Q, H, R []float64, n, m int) (FilterResult, error) {
	if len(obs) == 0 {
		return FilterResult{}, ErrEmptyData
	}
	if len(obs)%m != 0 {
		return FilterResult{}, ErrDimension
	}
	T := len(obs) / m
	if len(x0) != n || len(P0) != n*n || len(F) != n*n || len(Q) != n*n ||
		len(H) != m*n || len(R) != m*m {
		return FilterResult{}, ErrDimension
	}

	res := FilterResult{
		N: n, T: T,
		FilteredMean:  make([]float64, T*n),
		FilteredCov:   make([]float64, T*n*n),
		PredictedMean: make([]float64, T*n),
		PredictedCov:  make([]float64, T*n*n),
	}

	// Scratch buffers reused each step.
	xPred := make([]float64, n)
	PPred := make([]float64, n*n)
	xPost := make([]float64, n)
	PPost := make([]float64, n*n)
	v := make([]float64, m)
	S := make([]float64, m*m)

	// Step 0 prediction = prior.
	copy(xPred, x0)
	copy(PPred, P0)

	for t := 0; t < T; t++ {
		copy(res.PredictedMean[t*n:(t+1)*n], xPred)
		copy(res.PredictedCov[t*n*n:(t+1)*n*n], PPred)

		z := obs[t*m : (t+1)*m]
		ll, err := KalmanUpdate(xPred, PPred, z, H, R, n, m, xPost, PPost, v, S)
		if err != nil {
			return FilterResult{}, err
		}
		res.LogLikelihood += ll
		copy(res.FilteredMean[t*n:(t+1)*n], xPost)
		copy(res.FilteredCov[t*n*n:(t+1)*n*n], PPost)

		// Time update for the next step (skip after the last observation).
		if t < T-1 {
			if err := KalmanPredict(xPost, PPost, F, Q, n, xPred, PPred); err != nil {
				return FilterResult{}, err
			}
		}
	}
	return res, nil
}

// SmoothResult holds the output of the RTS backward pass.
type SmoothResult struct {
	N int
	T int
	// SmoothedMean[t] (length n) is E[x_t | z_{1:T}]; SmoothedCov[t] its cov.
	SmoothedMean []float64 // T*n
	SmoothedCov  []float64 // T*n*n
}

// RTSSmooth runs the Rauch-Tung-Striebel fixed-interval smoother over the
// output of Filter. The transition matrix F must be the same one used in the
// forward pass. The recursion is, backward from T-1:
//
//	C_t   = P_{t|t} Fᵀ (P_{t+1|t})⁻¹                 (smoother gain)
//	x_{t|T} = x_{t|t} + C_t (x_{t+1|T} - x_{t+1|t})
//	P_{t|T} = P_{t|t} + C_t (P_{t+1|T} - P_{t+1|t}) C_tᵀ
//
// where x_{t+1|t} / P_{t+1|t} are the *predicted* quantities Filter stored.
// The last smoothed step equals the last filtered step by construction.
// Returns ErrSingular if a predicted covariance is singular.
func RTSSmooth(fr FilterResult, F []float64) (SmoothResult, error) {
	n, T := fr.N, fr.T
	if len(F) != n*n {
		return SmoothResult{}, ErrDimension
	}
	sr := SmoothResult{
		N: n, T: T,
		SmoothedMean: make([]float64, T*n),
		SmoothedCov:  make([]float64, T*n*n),
	}
	if T == 0 {
		return sr, nil
	}
	// Terminal: smoothed = filtered.
	copy(sr.SmoothedMean[(T-1)*n:T*n], fr.FilteredMean[(T-1)*n:T*n])
	copy(sr.SmoothedCov[(T-1)*n*n:T*n*n], fr.FilteredCov[(T-1)*n*n:T*n*n])

	predInv := make([]float64, n*n)
	C := make([]float64, n*n)
	pFt := make([]float64, n*n)
	tmpV := make([]float64, n)
	tmpV2 := make([]float64, n)
	tmpM := make([]float64, n*n)
	tmpM2 := make([]float64, n*n)

	for t := T - 2; t >= 0; t-- {
		Pf := fr.FilteredCov[t*n*n : (t+1)*n*n]
		xf := fr.FilteredMean[t*n : (t+1)*n]
		// Predicted at t+1 (the prior that generated step t+1).
		Ppred := fr.PredictedCov[(t+1)*n*n : (t+2)*n*n]
		xpred := fr.PredictedMean[(t+1)*n : (t+2)*n]

		if err := invMat(Ppred, predInv, n); err != nil {
			return SmoothResult{}, err
		}
		// C = Pf Fᵀ predInv.
		matMulT(Pf, F, pFt, n, n, n) // Pf Fᵀ
		matMul(pFt, predInv, C, n, n, n)

		// x_{t|T} = xf + C (x_{t+1|T} - xpred).
		xs1 := sr.SmoothedMean[(t+1)*n : (t+2)*n]
		for i := 0; i < n; i++ {
			tmpV[i] = xs1[i] - xpred[i]
		}
		matVec(C, tmpV, tmpV2, n, n)
		xsOut := sr.SmoothedMean[t*n : (t+1)*n]
		for i := 0; i < n; i++ {
			xsOut[i] = xf[i] + tmpV2[i]
		}

		// P_{t|T} = Pf + C (P_{t+1|T} - Ppred) Cᵀ.
		Ps1 := sr.SmoothedCov[(t+1)*n*n : (t+2)*n*n]
		for i := 0; i < n*n; i++ {
			tmpM[i] = Ps1[i] - Ppred[i]
		}
		matMul(C, tmpM, tmpM2, n, n, n) // C (dP)
		PsOut := sr.SmoothedCov[t*n*n : (t+1)*n*n]
		matMulT(tmpM2, C, PsOut, n, n, n) // C (dP) Cᵀ
		for i := 0; i < n*n; i++ {
			PsOut[i] += Pf[i]
		}
		symmetrise(PsOut, n)
	}
	return sr, nil
}

// ---------------------------------------------------------------------------
// Univariate local-level model (Durbin & Koopman 2012, §2).
//
//	y_t     = mu_t + eps_t,   eps_t ~ N(0, R)   (observation noise)
//	mu_{t+1} = mu_t + eta_t,  eta_t ~ N(0, Q)   (random-walk level)
//
// This is the F=H=1 special case. It mirrors RubberDuck's scalar
// KalmanFilter.Filter(observations, Q, R, x0, P0) so RubberDuck.Reality can
// pin its port directly against these vectors.
// ---------------------------------------------------------------------------

// LocalLevelState is one step of the scalar local-level filter/smoother.
type LocalLevelState struct {
	// PredMean / PredVar: level mean and variance before seeing y_t.
	PredMean, PredVar float64
	// Mean / Var: filtered (or, from Smooth, smoothed) level and variance.
	Mean, Var float64
}

// LocalLevelFilter runs the scalar Kalman filter for the local-level model
// over series y with process variance q and observation variance r, starting
// from level mean x0 and variance p0. It returns one LocalLevelState per
// observation and the total log-likelihood. This is the closed-form scalar
// path (no matrix inverse), so it holds to machine precision.
func LocalLevelFilter(y []float64, q, r, x0, p0 float64) ([]LocalLevelState, float64, error) {
	if len(y) == 0 {
		return nil, 0, ErrEmptyData
	}
	out := make([]LocalLevelState, len(y))
	predMean, predVar := x0, p0
	logLik := 0.0
	for t, yt := range y {
		f := predVar + r            // innovation variance
		v := yt - predMean          // innovation
		k := predVar / f            // Kalman gain
		mean := predMean + k*v      // filtered mean
		variance := predVar * r / f // filtered variance = (1-k) predVar
		logLik += -0.5 * (log2Pi + math.Log(f) + v*v/f)
		out[t] = LocalLevelState{
			PredMean: predMean, PredVar: predVar,
			Mean: mean, Var: variance,
		}
		// Time update.
		predMean = mean
		predVar = variance + q
	}
	return out, logLik, nil
}

// LocalLevelSmooth appends the RTS-smoothed level and variance to a filtered
// series (produced by LocalLevelFilter with the same q). The returned states
// have Mean/Var overwritten with the smoothed values.
func LocalLevelSmooth(filtered []LocalLevelState, q float64) []LocalLevelState {
	n := len(filtered)
	out := make([]LocalLevelState, n)
	copy(out, filtered)
	if n == 0 {
		return out
	}
	// Terminal smoothed = terminal filtered (already in place).
	for t := n - 2; t >= 0; t-- {
		predVarNext := filtered[t].Var + q // = predVar at t+1
		c := filtered[t].Var / predVarNext // smoother gain
		out[t].Mean = filtered[t].Mean + c*(out[t+1].Mean-filtered[t].Mean)
		out[t].Var = filtered[t].Var + c*c*(out[t+1].Var-predVarNext)
	}
	return out
}

// LocalLevelSteadyState returns the steady-state (limiting) filtered variance
// pInf and Kalman gain kInf of the local-level model with process variance q
// and observation variance r. The Riccati fixed point P = P·r/(P+r) + q solves
// to P² - qP - qr = 0, so
//
//	pInf = (q + sqrt(q² + 4 q r)) / 2,   kInf = pInf / (pInf + r).
//
// q must be > 0 and r must be > 0.
func LocalLevelSteadyState(q, r float64) (pInf, kInf float64) {
	pInf = 0.5 * (q + math.Sqrt(q*q+4*q*r))
	kInf = pInf / (pInf + r)
	return pInf, kInf
}
