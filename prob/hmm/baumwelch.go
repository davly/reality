package hmm

import "math"

// BaumWelchResult holds the outcome of an EM fit.
type BaumWelchResult struct {
	// Model is the re-estimated model.
	Model Model
	// LogLikHistory[k] is the log-likelihood P(obs | model) BEFORE the k-th
	// re-estimation step (so LogLikHistory[0] is the likelihood of the initial
	// model). It is non-decreasing by the EM guarantee.
	LogLikHistory []float64
	// Iterations is the number of re-estimation steps actually performed.
	Iterations int
	// Converged is true if the loop stopped on the tolerance rather than the
	// iteration cap.
	Converged bool
}

// BaumWelch re-estimates a discrete HMM from a single observation sequence by
// the Baum-Welch (EM) algorithm (Rabiner 1989 §III.C). It is deterministic
// given the initial model init: the same init, maxIter and tol always produce
// the same fit. The loop stops when the increase in log-likelihood between
// successive iterations falls below tol, or after maxIter re-estimation steps,
// whichever comes first. maxIter must be > 0; tol must be >= 0.
//
// The re-estimation formulae are the standard expectations:
//
//	Pi'_i   = gamma_1(i)
//	A'_{ij} = sum_{t=1}^{T-1} xi_t(i,j) / sum_{t=1}^{T-1} gamma_t(i)
//	B'_j(k) = sum_{t: o_t=k} gamma_t(j) / sum_{t=1}^{T} gamma_t(j)
//
// where gamma and xi are the smoothed state and transition posteriors from the
// Forward-Backward pass, all computed in log space.
func BaumWelch(init Model, obs []int, maxIter int, tol float64) (BaumWelchResult, error) {
	if err := init.Validate(); err != nil {
		return BaumWelchResult{}, err
	}
	if len(obs) == 0 {
		return BaumWelchResult{}, ErrEmpty
	}
	if maxIter <= 0 || tol < 0 {
		return BaumWelchResult{}, ErrShape
	}
	for _, o := range obs {
		if o < 0 || o >= init.M {
			return BaumWelchResult{}, ErrShape
		}
	}

	N, M, T := init.N, init.M, len(obs)

	// Working copy of the parameters.
	cur := Model{
		N: N, M: M,
		Pi: append([]float64(nil), init.Pi...),
		A:  append([]float64(nil), init.A...),
		B:  append([]float64(nil), init.B...),
	}

	res := BaumWelchResult{}
	prevLL := math.Inf(-1)

	for iter := 0; iter < maxIter; iter++ {
		logB, err := cur.logEmissions(obs)
		if err != nil {
			return BaumWelchResult{}, err
		}
		logPi := cur.logPi()
		logA := cur.logA()

		logAlpha, ll := ForwardLog(logPi, logA, logB, N, T)
		logBeta := BackwardLog(logA, logB, N, T)
		res.LogLikHistory = append(res.LogLikHistory, ll)

		// Convergence check on the likelihood improvement.
		if iter > 0 && ll-prevLL < tol {
			res.Converged = true
			res.Iterations = iter
			res.Model = cur
			return res, nil
		}
		prevLL = ll

		// gamma[t*N+i] = exp(logAlpha + logBeta - ll).
		gamma := make([]float64, T*N)
		for t := 0; t < T; t++ {
			for i := 0; i < N; i++ {
				gamma[t*N+i] = math.Exp(logAlpha[t*N+i] + logBeta[t*N+i] - ll)
			}
		}

		// Expected transition counts sum_t xi_t(i,j).
		xiSum := make([]float64, N*N)
		for t := 0; t < T-1; t++ {
			for i := 0; i < N; i++ {
				for j := 0; j < N; j++ {
					lx := logAlpha[t*N+i] + logA[i*N+j] + logB[(t+1)*N+j] + logBeta[(t+1)*N+j] - ll
					xiSum[i*N+j] += math.Exp(lx)
				}
			}
		}

		// gammaSumToTminus1[i] = sum_{t=0}^{T-2} gamma_t(i) (transition denom);
		// gammaSumAll[i]       = sum_{t=0}^{T-1} gamma_t(i) (emission denom).
		gammaTrans := make([]float64, N)
		gammaAll := make([]float64, N)
		for i := 0; i < N; i++ {
			for t := 0; t < T; t++ {
				gammaAll[i] += gamma[t*N+i]
				if t < T-1 {
					gammaTrans[i] += gamma[t*N+i]
				}
			}
		}

		next := Model{
			N: N, M: M,
			Pi: make([]float64, N),
			A:  make([]float64, N*N),
			B:  make([]float64, N*M),
		}
		// Pi' = gamma_0.
		for i := 0; i < N; i++ {
			next.Pi[i] = gamma[i]
		}
		// A'.
		for i := 0; i < N; i++ {
			if gammaTrans[i] > 0 {
				for j := 0; j < N; j++ {
					next.A[i*N+j] = xiSum[i*N+j] / gammaTrans[i]
				}
			} else {
				// Unvisited state: keep the prior row to stay stochastic.
				copy(next.A[i*N:(i+1)*N], cur.A[i*N:(i+1)*N])
			}
		}
		// B'.
		for j := 0; j < N; j++ {
			if gammaAll[j] > 0 {
				for t := 0; t < T; t++ {
					next.B[j*M+obs[t]] += gamma[t*N+j]
				}
				for k := 0; k < M; k++ {
					next.B[j*M+k] /= gammaAll[j]
				}
			} else {
				copy(next.B[j*M:(j+1)*M], cur.B[j*M:(j+1)*M])
			}
		}

		cur = next
		res.Iterations = iter + 1
	}

	// Reached the iteration cap: record the final model's likelihood too so the
	// caller sees the post-last-step value.
	logB, err := cur.logEmissions(obs)
	if err != nil {
		return BaumWelchResult{}, err
	}
	_, ll := ForwardLog(cur.logPi(), cur.logA(), logB, N, T)
	res.LogLikHistory = append(res.LogLikHistory, ll)
	res.Model = cur
	res.Converged = false
	return res, nil
}
