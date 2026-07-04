package statespace

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// ---------------------------------------------------------------------------
// Golden-file loaders. The state-space golden files carry several output
// vectors per case, so they use a bespoke schema (richer than the scalar
// testutil.GoldenFile) with a typed loader here. They remain a plain-JSON,
// language-neutral contract that any port (RubberDuck.Reality, the C++/Python
// mirrors) reproduces byte-for-byte within tolerance.
// ---------------------------------------------------------------------------

type localLevelGolden struct {
	Function string `json:"function"`
	Source   string `json:"source"`
	Inputs   struct {
		Y  []float64 `json:"y"`
		Q  float64   `json:"q"`
		R  float64   `json:"r"`
		X0 float64   `json:"x0"`
		P0 float64   `json:"p0"`
	} `json:"inputs"`
	Expected struct {
		Loglik       float64   `json:"loglik"`
		FilteredMean []float64 `json:"filtered_mean"`
		FilteredVar  []float64 `json:"filtered_var"`
		SmoothedMean []float64 `json:"smoothed_mean"`
		SmoothedVar  []float64 `json:"smoothed_var"`
	} `json:"expected"`
	Tolerance float64 `json:"tolerance"`
}

type cvGolden struct {
	Function string `json:"function"`
	Source   string `json:"source"`
	Inputs   struct {
		Positions []float64 `json:"positions"`
		F         []float64 `json:"F"`
		H         []float64 `json:"H"`
		Q         []float64 `json:"Q"`
		R         []float64 `json:"R"`
		X0        []float64 `json:"x0"`
		P0        []float64 `json:"P0"`
		N         int       `json:"n"`
		M         int       `json:"m"`
	} `json:"inputs"`
	Expected struct {
		Loglik       float64   `json:"loglik"`
		FilteredMean []float64 `json:"filtered_mean"`
		FilteredCov  []float64 `json:"filtered_cov"`
		SmoothedMean []float64 `json:"smoothed_mean"`
		SmoothedCov  []float64 `json:"smoothed_cov"`
	} `json:"expected"`
	Tolerance float64 `json:"tolerance"`
}

func loadJSON(t *testing.T, rel string, dst any) {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "statespace", rel))
	if err != nil {
		t.Fatalf("read %s: %v", rel, err)
	}
	if err := json.Unmarshal(data, dst); err != nil {
		t.Fatalf("parse %s: %v", rel, err)
	}
}

func assertSlice(t *testing.T, label string, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", label, len(got), len(want))
	}
	for i := range want {
		if d := math.Abs(got[i] - want[i]); d > tol {
			t.Errorf("%s[%d] = %.12g, want %.12g (diff %.3e > %.3e)", label, i, got[i], want[i], d, tol)
		}
	}
}

// TestGoldenLocalLevelNile pins the scalar local-level filter and RTS smoother
// against the Durbin-Koopman Nile golden vectors.
func TestGoldenLocalLevelNile(t *testing.T) {
	var g localLevelGolden
	loadJSON(t, "local_level_nile.json", &g)

	filt, ll, err := LocalLevelFilter(g.Inputs.Y, g.Inputs.Q, g.Inputs.R, g.Inputs.X0, g.Inputs.P0)
	if err != nil {
		t.Fatal(err)
	}
	sm := LocalLevelSmooth(filt, g.Inputs.Q)

	if d := math.Abs(ll - g.Expected.Loglik); d > g.Tolerance {
		t.Errorf("loglik = %.12f, want %.12f (diff %.3e)", ll, g.Expected.Loglik, d)
	}
	fm := make([]float64, len(filt))
	fv := make([]float64, len(filt))
	smm := make([]float64, len(sm))
	smv := make([]float64, len(sm))
	for i := range filt {
		fm[i] = filt[i].Mean
		fv[i] = filt[i].Var
		smm[i] = sm[i].Mean
		smv[i] = sm[i].Var
	}
	assertSlice(t, "filtered_mean", fm, g.Expected.FilteredMean, g.Tolerance)
	assertSlice(t, "filtered_var", fv, g.Expected.FilteredVar, g.Tolerance)
	assertSlice(t, "smoothed_mean", smm, g.Expected.SmoothedMean, g.Tolerance)
	assertSlice(t, "smoothed_var", smv, g.Expected.SmoothedVar, g.Tolerance)
}

// TestGoldenConstantVelocity2D pins the full multivariate Kalman filter + RTS
// smoother against the 2-D constant-velocity golden vectors.
func TestGoldenConstantVelocity2D(t *testing.T) {
	var g cvGolden
	loadJSON(t, "constant_velocity_2d.json", &g)
	in := g.Inputs

	fr, err := Filter(in.Positions, in.X0, in.P0, in.F, in.Q, in.H, in.R, in.N, in.M)
	if err != nil {
		t.Fatal(err)
	}
	sm, err := RTSSmooth(fr, in.F)
	if err != nil {
		t.Fatal(err)
	}
	if d := math.Abs(fr.LogLikelihood - g.Expected.Loglik); d > g.Tolerance {
		t.Errorf("loglik = %.10f, want %.10f (diff %.3e)", fr.LogLikelihood, g.Expected.Loglik, d)
	}
	assertSlice(t, "filtered_mean", fr.FilteredMean, g.Expected.FilteredMean, g.Tolerance)
	assertSlice(t, "filtered_cov", fr.FilteredCov, g.Expected.FilteredCov, g.Tolerance)
	assertSlice(t, "smoothed_mean", sm.SmoothedMean, g.Expected.SmoothedMean, g.Tolerance)
	assertSlice(t, "smoothed_cov", sm.SmoothedCov, g.Expected.SmoothedCov, g.Tolerance)
}

// ---------------------------------------------------------------------------
// Independent anchors — validate the golden numbers against math the golden
// files cannot fake.
// ---------------------------------------------------------------------------

// TestSteadyStateMatchesConvergedFilter is the closed-form anchor: the Riccati
// fixed point P_inf = (q + sqrt(q^2 + 4qr))/2 must equal the variance the
// local-level filter converges to after many constant observations.
func TestSteadyStateMatchesConvergedFilter(t *testing.T) {
	cases := []struct{ q, r float64 }{
		{1469.1, 15099.0}, {1.0, 1.0}, {0.001, 1.0}, {5.0, 0.5},
	}
	for _, c := range cases {
		pInf, kInf := LocalLevelSteadyState(c.q, c.r)
		// Fixed-point identity: P_inf^2 - q P_inf - q r = 0.
		resid := pInf*pInf - c.q*pInf - c.q*c.r
		if math.Abs(resid) > 1e-6*(1+pInf*pInf) {
			t.Errorf("q=%v r=%v: Riccati residual %.3e", c.q, c.r, resid)
		}
		// Converge the filter over a long constant series.
		y := make([]float64, 2000)
		for i := range y {
			y[i] = 100.0
		}
		filt, _, err := LocalLevelFilter(y, c.q, c.r, 100.0, 1e9)
		if err != nil {
			t.Fatal(err)
		}
		last := filt[len(filt)-1]
		predVar := last.Var + c.q // predicted variance -> P_inf
		if d := math.Abs(predVar - pInf); d > 1e-6 {
			t.Errorf("q=%v r=%v: converged predVar %.9f != pInf %.9f", c.q, c.r, predVar, pInf)
		}
		gain := last.Var / (last.Var + c.q + c.r) // one-step gain at convergence
		_ = gain
		if kInf <= 0 || kInf >= 1 {
			t.Errorf("q=%v r=%v: kInf %.6f out of (0,1)", c.q, c.r, kInf)
		}
	}
}

// TestScalarMatrixParity is the cross-implementation anchor: the general
// multivariate Filter with n=m=1 must reproduce the scalar LocalLevelFilter to
// machine precision (they are the same recursion via different code paths).
func TestScalarMatrixParity(t *testing.T) {
	y := []float64{1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140}
	q, r, x0, p0 := 1469.1, 15099.0, 1120.0, 1e7
	filtS, llS, err := LocalLevelFilter(y, q, r, x0, p0)
	if err != nil {
		t.Fatal(err)
	}
	obs := append([]float64(nil), y...)
	fr, err := Filter(obs, []float64{x0}, []float64{p0}, []float64{1}, []float64{q}, []float64{1}, []float64{r}, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if d := math.Abs(llS - fr.LogLikelihood); d > 1e-9 {
		t.Errorf("loglik parity diff %.3e", d)
	}
	for t2 := range y {
		if d := math.Abs(filtS[t2].Mean - fr.FilteredMean[t2]); d > 1e-9 {
			t.Errorf("mean parity[%d] diff %.3e", t2, d)
		}
		if d := math.Abs(filtS[t2].Var - fr.FilteredCov[t2]); d > 1e-9 {
			t.Errorf("var parity[%d] diff %.3e", t2, d)
		}
	}
}

// TestSmootherTerminalIdentity: the last smoothed estimate equals the last
// filtered estimate (the RTS backward recursion has no future to borrow).
func TestSmootherTerminalIdentity(t *testing.T) {
	pos := []float64{1.0, 2.1, 2.9, 4.2, 5.0, 5.8, 7.3}
	F := []float64{1, 1, 0, 1}
	fr, err := Filter(pos, []float64{0, 0}, []float64{1, 0, 0, 1}, F, []float64{0.1, 0, 0, 0.1}, []float64{1, 0}, []float64{1.0}, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	sm, err := RTSSmooth(fr, F)
	if err != nil {
		t.Fatal(err)
	}
	T, n := fr.T, fr.N
	for j := 0; j < n; j++ {
		fMean := fr.FilteredMean[(T-1)*n+j]
		sMean := sm.SmoothedMean[(T-1)*n+j]
		if math.Abs(fMean-sMean) > 1e-12 {
			t.Errorf("terminal mean[%d]: filt %.12f != smooth %.12f", j, fMean, sMean)
		}
	}
	for k := 0; k < n*n; k++ {
		fc := fr.FilteredCov[(T-1)*n*n+k]
		sc := sm.SmoothedCov[(T-1)*n*n+k]
		if math.Abs(fc-sc) > 1e-12 {
			t.Errorf("terminal cov[%d]: filt %.12f != smooth %.12f", k, fc, sc)
		}
	}
}

// TestSmoothingReducesVariance: the smoothed marginal variance (diagonal of
// the smoothed covariance) is never larger than the filtered marginal variance
// — conditioning on more data cannot increase uncertainty.
func TestSmoothingReducesVariance(t *testing.T) {
	y := []float64{1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140}
	q, r := 1469.1, 15099.0
	filt, _, err := LocalLevelFilter(y, q, r, 1120, 1e7)
	if err != nil {
		t.Fatal(err)
	}
	sm := LocalLevelSmooth(filt, q)
	for t2 := range y {
		if sm[t2].Var > filt[t2].Var+1e-9 {
			t.Errorf("t=%d: smoothed var %.6f > filtered var %.6f", t2, sm[t2].Var, filt[t2].Var)
		}
	}
}

// TestKalmanUpdateHandRecursion checks a single scalar update against a fully
// hand-computed step: x=0, P=1, z=2, H=1, R=1 => S=2, K=0.5, xpost=1, Ppost=0.5,
// loglik = -0.5(log2pi + log2 + 2).
func TestKalmanUpdateHandRecursion(t *testing.T) {
	xOut := make([]float64, 1)
	POut := make([]float64, 1)
	v := make([]float64, 1)
	S := make([]float64, 1)
	ll, err := KalmanUpdate([]float64{0}, []float64{1}, []float64{2}, []float64{1}, []float64{1}, 1, 1, xOut, POut, v, S)
	if err != nil {
		t.Fatal(err)
	}
	wantLL := -0.5 * (log2Pi + math.Log(2) + 2.0)
	checks := []struct {
		name      string
		got, want float64
	}{
		{"xpost", xOut[0], 1.0},
		{"Ppost", POut[0], 0.5},
		{"innovation", v[0], 2.0},
		{"S", S[0], 2.0},
		{"loglik", ll, wantLL},
	}
	for _, c := range checks {
		if math.Abs(c.got-c.want) > 1e-12 {
			t.Errorf("%s = %.12f, want %.12f", c.name, c.got, c.want)
		}
	}
}

// TestMatrixInverseIdentity checks the Gauss-Jordan inverse against A A^-1 = I.
func TestMatrixInverseIdentity(t *testing.T) {
	A := []float64{4, 3, 6, 3} // 2x2, det = 12-18 = -6
	inv := make([]float64, 4)
	if err := invMat(A, inv, 2); err != nil {
		t.Fatal(err)
	}
	prod := make([]float64, 4)
	matMul(A, inv, prod, 2, 2, 2)
	want := []float64{1, 0, 0, 1}
	assertSlice(t, "A*Ainv", prod, want, 1e-12)
}

// TestSingularInnovation: a zero observation covariance with zero state
// covariance makes S singular and must error, not panic.
func TestSingularInnovation(t *testing.T) {
	xOut := make([]float64, 1)
	POut := make([]float64, 1)
	v := make([]float64, 1)
	S := make([]float64, 1)
	_, err := KalmanUpdate([]float64{0}, []float64{0}, []float64{1}, []float64{1}, []float64{0}, 1, 1, xOut, POut, v, S)
	if err != ErrSingular {
		t.Errorf("expected ErrSingular, got %v", err)
	}
}

// TestDimensionGuards: wrong buffer lengths error rather than corrupt memory.
func TestDimensionGuards(t *testing.T) {
	if err := KalmanPredict([]float64{0}, []float64{1}, []float64{1}, []float64{1}, 1, make([]float64, 2), make([]float64, 1)); err != ErrDimension {
		t.Errorf("KalmanPredict bad xOut: got %v", err)
	}
	if _, err := Filter([]float64{}, []float64{0}, []float64{1}, []float64{1}, []float64{1}, []float64{1}, []float64{1}, 1, 1); err != ErrEmptyData {
		t.Errorf("Filter empty: got %v", err)
	}
}
