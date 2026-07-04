package hmm

import (
	"errors"
	"math"
)

// ErrShape is returned when a model's Pi/A/B slices do not match N and M, or
// when an observation index is out of range.
var ErrShape = errors.New("hmm: model or observation shape mismatch")

// ErrEmpty is returned when an observation sequence is empty.
var ErrEmpty = errors.New("hmm: observation sequence must be non-empty")

// Model is a discrete (categorical-emission) hidden Markov model. All matrices
// are flat row-major.
type Model struct {
	N  int       // number of hidden states
	M  int       // number of observation symbols
	Pi []float64 // length N, initial state distribution
	A  []float64 // N*N, A[i*N+j] = P(state j at t+1 | state i at t)
	B  []float64 // N*M, B[i*M+k] = P(symbol k | state i)
}

// Validate checks the shape and (loosely) the stochastic structure of m.
func (m Model) Validate() error {
	if m.N <= 0 || m.M <= 0 {
		return ErrShape
	}
	if len(m.Pi) != m.N || len(m.A) != m.N*m.N || len(m.B) != m.N*m.M {
		return ErrShape
	}
	return nil
}

// logEmissions builds the T×N emission log-likelihood matrix for a discrete
// model and observation indices: logB[t*N+i] = log B[i][obs[t]].
func (m Model) logEmissions(obs []int) ([]float64, error) {
	T := len(obs)
	logB := make([]float64, T*m.N)
	for t, o := range obs {
		if o < 0 || o >= m.M {
			return nil, ErrShape
		}
		for i := 0; i < m.N; i++ {
			logB[t*m.N+i] = math.Log(m.B[i*m.M+o])
		}
	}
	return logB, nil
}

func (m Model) logPi() []float64 {
	out := make([]float64, m.N)
	for i, p := range m.Pi {
		out[i] = math.Log(p)
	}
	return out
}

func (m Model) logA() []float64 {
	out := make([]float64, len(m.A))
	for i, p := range m.A {
		out[i] = math.Log(p)
	}
	return out
}

// logSumExp returns log(sum_i exp(x_i)) computed stably by subtracting the
// maximum. An all -Inf input (every path impossible) returns -Inf.
func logSumExp(x []float64) float64 {
	mx := math.Inf(-1)
	for _, v := range x {
		if v > mx {
			mx = v
		}
	}
	if math.IsInf(mx, -1) {
		return math.Inf(-1)
	}
	s := 0.0
	for _, v := range x {
		s += math.Exp(v - mx)
	}
	return mx + math.Log(s)
}

// ---------------------------------------------------------------------------
// Emission-agnostic log-space core. logB is a T×N matrix (row-major) of
// emission log-likelihoods logB[t*N+i] = log P(o_t | state i). logPi is
// length N; logA is N×N row-major. These serve both the discrete wrappers
// below and Gaussian-emission consumers that build logB from a Normal log-pdf.
// ---------------------------------------------------------------------------

// ForwardLog runs the log-space Forward pass. It returns the T×N matrix of
// log-forward variables logAlpha[t*N+i] = log P(o_1..o_t, state_t = i) and the
// total log-likelihood logLik = log P(o_1..o_T).
func ForwardLog(logPi, logA, logB []float64, N, T int) (logAlpha []float64, logLik float64) {
	logAlpha = make([]float64, T*N)
	for i := 0; i < N; i++ {
		logAlpha[i] = logPi[i] + logB[i]
	}
	tmp := make([]float64, N)
	for t := 1; t < T; t++ {
		for j := 0; j < N; j++ {
			for i := 0; i < N; i++ {
				tmp[i] = logAlpha[(t-1)*N+i] + logA[i*N+j]
			}
			logAlpha[t*N+j] = logSumExp(tmp) + logB[t*N+j]
		}
	}
	last := logAlpha[(T-1)*N : T*N]
	return logAlpha, logSumExp(last)
}

// BackwardLog runs the log-space Backward pass. It returns the T×N matrix
// logBeta[t*N+i] = log P(o_{t+1}..o_T | state_t = i). logBeta[T-1][*] = 0.
func BackwardLog(logA, logB []float64, N, T int) []float64 {
	logBeta := make([]float64, T*N)
	// last row already zero.
	tmp := make([]float64, N)
	for t := T - 2; t >= 0; t-- {
		for i := 0; i < N; i++ {
			for j := 0; j < N; j++ {
				tmp[j] = logA[i*N+j] + logB[(t+1)*N+j] + logBeta[(t+1)*N+j]
			}
			logBeta[t*N+i] = logSumExp(tmp)
		}
	}
	return logBeta
}

// ViterbiLog runs the log-space Viterbi decoder. It returns the most likely
// state path (length T) and its joint log-probability
// log P(best path, o_1..o_T). Argmax ties resolve to the lowest state index.
func ViterbiLog(logPi, logA, logB []float64, N, T int) (path []int, logProb float64) {
	logDelta := make([]float64, T*N)
	psi := make([]int, T*N)
	for i := 0; i < N; i++ {
		logDelta[i] = logPi[i] + logB[i]
	}
	for t := 1; t < T; t++ {
		for j := 0; j < N; j++ {
			bestVal := math.Inf(-1)
			bestArg := 0
			for i := 0; i < N; i++ {
				v := logDelta[(t-1)*N+i] + logA[i*N+j]
				if v > bestVal {
					bestVal = v
					bestArg = i
				}
			}
			logDelta[t*N+j] = bestVal + logB[t*N+j]
			psi[t*N+j] = bestArg
		}
	}
	// Terminate.
	bestVal := math.Inf(-1)
	bestArg := 0
	for i := 0; i < N; i++ {
		if v := logDelta[(T-1)*N+i]; v > bestVal {
			bestVal = v
			bestArg = i
		}
	}
	path = make([]int, T)
	path[T-1] = bestArg
	for t := T - 1; t > 0; t-- {
		path[t-1] = psi[t*N+path[t]]
	}
	return path, bestVal
}

// ---------------------------------------------------------------------------
// Discrete-model wrappers.
// ---------------------------------------------------------------------------

// Forward returns the total log-likelihood log P(obs | m) and the T×N
// log-forward matrix for a discrete model.
func Forward(m Model, obs []int) (logAlpha []float64, logLik float64, err error) {
	if err = m.Validate(); err != nil {
		return nil, 0, err
	}
	if len(obs) == 0 {
		return nil, 0, ErrEmpty
	}
	logB, err := m.logEmissions(obs)
	if err != nil {
		return nil, 0, err
	}
	a, ll := ForwardLog(m.logPi(), m.logA(), logB, m.N, len(obs))
	return a, ll, nil
}

// Backward returns the T×N log-backward matrix for a discrete model.
func Backward(m Model, obs []int) ([]float64, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if len(obs) == 0 {
		return nil, ErrEmpty
	}
	logB, err := m.logEmissions(obs)
	if err != nil {
		return nil, err
	}
	return BackwardLog(m.logA(), logB, m.N, len(obs)), nil
}

// Viterbi returns the most likely state path and its joint log-probability for
// a discrete model.
func Viterbi(m Model, obs []int) (path []int, logProb float64, err error) {
	if err = m.Validate(); err != nil {
		return nil, 0, err
	}
	if len(obs) == 0 {
		return nil, 0, ErrEmpty
	}
	logB, err := m.logEmissions(obs)
	if err != nil {
		return nil, 0, err
	}
	p, lp := ViterbiLog(m.logPi(), m.logA(), logB, m.N, len(obs))
	return p, lp, nil
}

// Posterior returns the smoothed state posteriors gamma[t*N+i] =
// P(state_t = i | obs, m) (each row sums to 1) and the log-likelihood.
func Posterior(m Model, obs []int) (gamma []float64, logLik float64, err error) {
	if err = m.Validate(); err != nil {
		return nil, 0, err
	}
	if len(obs) == 0 {
		return nil, 0, ErrEmpty
	}
	logB, err := m.logEmissions(obs)
	if err != nil {
		return nil, 0, err
	}
	N, T := m.N, len(obs)
	logAlpha, ll := ForwardLog(m.logPi(), m.logA(), logB, N, T)
	logBeta := BackwardLog(m.logA(), logB, N, T)
	gamma = make([]float64, T*N)
	for t := 0; t < T; t++ {
		for i := 0; i < N; i++ {
			gamma[t*N+i] = math.Exp(logAlpha[t*N+i] + logBeta[t*N+i] - ll)
		}
	}
	return gamma, ll, nil
}
