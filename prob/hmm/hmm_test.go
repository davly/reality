package hmm

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func loadJSON(t *testing.T, rel string, dst any) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "hmm", rel))
	if err != nil {
		t.Fatalf("read %s: %v", rel, err)
	}
	if err := json.Unmarshal(data, dst); err != nil {
		t.Fatalf("parse %s: %v", rel, err)
	}
}

func assertClose(t *testing.T, label string, got, want, tol float64) {
	t.Helper()
	if d := math.Abs(got - want); d > tol {
		t.Errorf("%s = %.12g, want %.12g (diff %.3e > %.3e)", label, got, want, d, tol)
	}
}

func assertSlice(t *testing.T, label string, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", label, len(got), len(want))
	}
	for i := range want {
		assertClose(t, label, got[i], want[i], tol)
	}
}

// ---------------------------------------------------------------------------
// Golden: Rabiner forward / viterbi / posterior.
// ---------------------------------------------------------------------------

type fvGolden struct {
	Inputs struct {
		N, M int
		Pi   []float64
		A    []float64
		B    []float64
		Obs  []int `json:"obs"`
	} `json:"inputs"`
	Expected struct {
		ForwardLoglik  float64   `json:"forward_loglik"`
		ForwardProb    float64   `json:"forward_prob"`
		ViterbiPath    []int     `json:"viterbi_path"`
		ViterbiLogprob float64   `json:"viterbi_logprob"`
		ViterbiProb    float64   `json:"viterbi_prob"`
		Gamma          []float64 `json:"gamma"`
	} `json:"expected"`
	Tolerance float64 `json:"tolerance"`
}

func (g fvGolden) model() Model {
	return Model{N: g.Inputs.N, M: g.Inputs.M, Pi: g.Inputs.Pi, A: g.Inputs.A, B: g.Inputs.B}
}

func TestGoldenRabinerForwardViterbi(t *testing.T) {
	var g fvGolden
	loadJSON(t, "rabiner_forward_viterbi.json", &g)
	m := g.model()
	obs := g.Inputs.Obs
	tol := g.Tolerance

	_, ll, err := Forward(m, obs)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, "forward_loglik", ll, g.Expected.ForwardLoglik, tol)
	assertClose(t, "forward_prob", math.Exp(ll), g.Expected.ForwardProb, tol)

	path, lp, err := Viterbi(m, obs)
	if err != nil {
		t.Fatal(err)
	}
	if len(path) != len(g.Expected.ViterbiPath) {
		t.Fatalf("viterbi path length %d, want %d", len(path), len(g.Expected.ViterbiPath))
	}
	for i := range path {
		if path[i] != g.Expected.ViterbiPath[i] {
			t.Errorf("viterbi path[%d] = %d, want %d", i, path[i], g.Expected.ViterbiPath[i])
		}
	}
	assertClose(t, "viterbi_logprob", lp, g.Expected.ViterbiLogprob, tol)
	assertClose(t, "viterbi_prob", math.Exp(lp), g.Expected.ViterbiProb, tol)

	gamma, _, err := Posterior(m, obs)
	if err != nil {
		t.Fatal(err)
	}
	assertSlice(t, "gamma", gamma, g.Expected.Gamma, tol)
}

// ---------------------------------------------------------------------------
// Golden: Baum-Welch re-estimation contract.
// ---------------------------------------------------------------------------

type bwGolden struct {
	Inputs struct {
		N, M    int
		InitPi  []float64 `json:"init_Pi"`
		InitA   []float64 `json:"init_A"`
		InitB   []float64 `json:"init_B"`
		Obs     []int     `json:"obs"`
		MaxIter int       `json:"maxIter"`
		Tol     float64   `json:"tol"`
	} `json:"inputs"`
	Expected struct {
		Iterations  int       `json:"iterations"`
		Converged   bool      `json:"converged"`
		Pi          []float64 `json:"Pi"`
		A           []float64 `json:"A"`
		B           []float64 `json:"B"`
		FinalLoglik float64   `json:"final_loglik"`
	} `json:"expected"`
	Tolerance float64 `json:"tolerance"`
}

func TestGoldenBaumWelch(t *testing.T) {
	var g bwGolden
	loadJSON(t, "baumwelch_reestimation.json", &g)
	init := Model{N: g.Inputs.N, M: g.Inputs.M, Pi: g.Inputs.InitPi, A: g.Inputs.InitA, B: g.Inputs.InitB}
	res, err := BaumWelch(init, g.Inputs.Obs, g.Inputs.MaxIter, g.Inputs.Tol)
	if err != nil {
		t.Fatal(err)
	}
	if res.Iterations != g.Expected.Iterations {
		t.Errorf("iterations = %d, want %d", res.Iterations, g.Expected.Iterations)
	}
	if res.Converged != g.Expected.Converged {
		t.Errorf("converged = %v, want %v", res.Converged, g.Expected.Converged)
	}
	tol := g.Tolerance
	assertSlice(t, "Pi", res.Model.Pi, g.Expected.Pi, tol)
	assertSlice(t, "A", res.Model.A, g.Expected.A, tol)
	assertSlice(t, "B", res.Model.B, g.Expected.B, tol)
	last := res.LogLikHistory[len(res.LogLikHistory)-1]
	assertClose(t, "final_loglik", last, g.Expected.FinalLoglik, tol)
}

// ---------------------------------------------------------------------------
// Independent anchors.
// ---------------------------------------------------------------------------

// bruteForceForward sums P(path, obs) over all N^T state paths — an O(N^T)
// oracle independent of the O(T N^2) Forward recursion.
func bruteForceForward(m Model, obs []int) float64 {
	N, T := m.N, len(obs)
	total := 0.0
	path := make([]int, T)
	var rec func(t int)
	rec = func(t int) {
		if t == T {
			p := m.Pi[path[0]] * m.B[path[0]*m.M+obs[0]]
			for s := 1; s < T; s++ {
				p *= m.A[path[s-1]*N+path[s]] * m.B[path[s]*m.M+obs[s]]
			}
			total += p
			return
		}
		for i := 0; i < N; i++ {
			path[t] = i
			rec(t + 1)
		}
	}
	rec(0)
	return total
}

// bruteForceViterbi returns the max P(path, obs) and an argmax path over all
// N^T paths.
func bruteForceViterbi(m Model, obs []int) (float64, []int) {
	N, T := m.N, len(obs)
	best := -1.0
	var bestPath []int
	path := make([]int, T)
	var rec func(t int)
	rec = func(t int) {
		if t == T {
			p := m.Pi[path[0]] * m.B[path[0]*m.M+obs[0]]
			for s := 1; s < T; s++ {
				p *= m.A[path[s-1]*N+path[s]] * m.B[path[s]*m.M+obs[s]]
			}
			if p > best {
				best = p
				bestPath = append([]int(nil), path...)
			}
			return
		}
		for i := 0; i < N; i++ {
			path[t] = i
			rec(t + 1)
		}
	}
	rec(0)
	return best, bestPath
}

// TestForwardBruteForceParity pins Forward against exhaustive path enumeration
// across several models and sequences.
func TestForwardBruteForceParity(t *testing.T) {
	models := []Model{
		{N: 2, M: 3, Pi: []float64{0.6, 0.4}, A: []float64{0.7, 0.3, 0.4, 0.6}, B: []float64{0.5, 0.4, 0.1, 0.1, 0.3, 0.6}},
		{N: 3, M: 2, Pi: []float64{0.2, 0.5, 0.3}, A: []float64{0.5, 0.3, 0.2, 0.1, 0.6, 0.3, 0.25, 0.25, 0.5}, B: []float64{0.9, 0.1, 0.5, 0.5, 0.2, 0.8}},
	}
	seqs := [][]int{{0, 1, 2, 1, 0}, {1, 0, 1, 1, 0, 1}}
	for mi, m := range models {
		for _, obs := range seqs {
			ok := true
			for _, o := range obs {
				if o >= m.M {
					ok = false
				}
			}
			if !ok {
				continue
			}
			_, ll, err := Forward(m, obs)
			if err != nil {
				t.Fatal(err)
			}
			want := bruteForceForward(m, obs)
			if d := math.Abs(math.Exp(ll) - want); d > 1e-12 {
				t.Errorf("model %d obs %v: forward %.12f != brute %.12f", mi, obs, math.Exp(ll), want)
			}
		}
	}
}

// TestViterbiBruteForceParity pins Viterbi's best log-probability against the
// exhaustive max over all paths.
func TestViterbiBruteForceParity(t *testing.T) {
	m := Model{N: 3, M: 2, Pi: []float64{0.2, 0.5, 0.3}, A: []float64{0.5, 0.3, 0.2, 0.1, 0.6, 0.3, 0.25, 0.25, 0.5}, B: []float64{0.9, 0.1, 0.5, 0.5, 0.2, 0.8}}
	for _, obs := range [][]int{{0, 1, 1, 0, 1}, {1, 1, 0, 0, 1, 0}} {
		_, lp, err := Viterbi(m, obs)
		if err != nil {
			t.Fatal(err)
		}
		want, _ := bruteForceViterbi(m, obs)
		if d := math.Abs(math.Exp(lp) - want); d > 1e-12 {
			t.Errorf("obs %v: viterbi %.12f != brute %.12f", obs, math.Exp(lp), want)
		}
	}
}

// TestForwardBackwardConsistency: the likelihood computed from the backward
// pass must equal the forward likelihood, and every gamma row must sum to 1.
func TestForwardBackwardConsistency(t *testing.T) {
	m := Model{N: 2, M: 3, Pi: []float64{0.6, 0.4}, A: []float64{0.7, 0.3, 0.4, 0.6}, B: []float64{0.5, 0.4, 0.1, 0.1, 0.3, 0.6}}
	obs := []int{0, 1, 2, 2, 1, 0}
	_, llF, err := Forward(m, obs)
	if err != nil {
		t.Fatal(err)
	}
	logBeta, err := Backward(m, obs)
	if err != nil {
		t.Fatal(err)
	}
	// P(O) via backward: logsumexp_i(logPi_i + logB_i(o_0) + logBeta_0(i)).
	logPi := m.logPi()
	logB, _ := m.logEmissions(obs)
	tmp := make([]float64, m.N)
	for i := 0; i < m.N; i++ {
		tmp[i] = logPi[i] + logB[i] + logBeta[i]
	}
	llB := logSumExp(tmp)
	assertClose(t, "forward-vs-backward loglik", llB, llF, 1e-12)

	gamma, _, err := Posterior(m, obs)
	if err != nil {
		t.Fatal(err)
	}
	for tt := 0; tt < len(obs); tt++ {
		s := 0.0
		for i := 0; i < m.N; i++ {
			s += gamma[tt*m.N+i]
		}
		assertClose(t, "gamma row sum", s, 1.0, 1e-12)
	}
}

// TestBaumWelchMonotone: the EM log-likelihood is non-decreasing every step.
func TestBaumWelchMonotone(t *testing.T) {
	init := Model{N: 2, M: 2, Pi: []float64{0.5, 0.5}, A: []float64{0.6, 0.4, 0.4, 0.6}, B: []float64{0.7, 0.3, 0.3, 0.7}}
	seq := []int{0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1}
	res, err := BaumWelch(init, seq, 50, 1e-12)
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i < len(res.LogLikHistory); i++ {
		if res.LogLikHistory[i] < res.LogLikHistory[i-1]-1e-10 {
			t.Errorf("log-likelihood decreased at step %d: %.12f -> %.12f", i, res.LogLikHistory[i-1], res.LogLikHistory[i])
		}
	}
}

// TestBaumWelchDeterministic: identical inputs yield identical fits.
func TestBaumWelchDeterministic(t *testing.T) {
	init := Model{N: 2, M: 2, Pi: []float64{0.5, 0.5}, A: []float64{0.6, 0.4, 0.4, 0.6}, B: []float64{0.7, 0.3, 0.3, 0.7}}
	seq := []int{0, 0, 1, 1, 0, 1, 0, 0, 1, 1}
	a, _ := BaumWelch(init, seq, 7, 0)
	b, _ := BaumWelch(init, seq, 7, 0)
	assertSlice(t, "Pi determinism", a.Model.Pi, b.Model.Pi, 0)
	assertSlice(t, "A determinism", a.Model.A, b.Model.A, 0)
	assertSlice(t, "B determinism", a.Model.B, b.Model.B, 0)
}

// TestBaumWelchStochasticRows: fitted Pi/A/B rows stay valid distributions.
func TestBaumWelchStochasticRows(t *testing.T) {
	init := Model{N: 2, M: 2, Pi: []float64{0.5, 0.5}, A: []float64{0.6, 0.4, 0.4, 0.6}, B: []float64{0.7, 0.3, 0.3, 0.7}}
	seq := []int{0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1}
	res, err := BaumWelch(init, seq, 10, 0)
	if err != nil {
		t.Fatal(err)
	}
	m := res.Model
	s := 0.0
	for _, p := range m.Pi {
		if p < -1e-12 {
			t.Errorf("negative Pi %v", p)
		}
		s += p
	}
	assertClose(t, "Pi sum", s, 1.0, 1e-9)
	for i := 0; i < m.N; i++ {
		rs := 0.0
		for j := 0; j < m.N; j++ {
			rs += m.A[i*m.N+j]
		}
		assertClose(t, "A row sum", rs, 1.0, 1e-9)
		bs := 0.0
		for k := 0; k < m.M; k++ {
			bs += m.B[i*m.M+k]
		}
		assertClose(t, "B row sum", bs, 1.0, 1e-9)
	}
}

// TestErrorGuards checks shape / empty guards.
func TestErrorGuards(t *testing.T) {
	m := Model{N: 2, M: 2, Pi: []float64{0.5, 0.5}, A: []float64{0.6, 0.4, 0.4, 0.6}, B: []float64{0.7, 0.3, 0.3, 0.7}}
	if _, _, err := Forward(m, nil); err != ErrEmpty {
		t.Errorf("empty obs: got %v", err)
	}
	if _, _, err := Forward(m, []int{0, 5}); err != ErrShape {
		t.Errorf("bad symbol: got %v", err)
	}
	bad := Model{N: 2, M: 2, Pi: []float64{1}, A: m.A, B: m.B}
	if _, _, err := Forward(bad, []int{0}); err != ErrShape {
		t.Errorf("bad Pi len: got %v", err)
	}
}
