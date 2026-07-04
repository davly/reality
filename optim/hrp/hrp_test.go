package hrp

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// goldenCase mirrors one entry of the HRP golden JSON. The vectors were
// produced by an independent Python reference implementation (see the
// w34-hrp build commit) and cross-checked against a by-hand derivation,
// following Lopez de Prado (2016). The intermediate stages (distance,
// linkage, leaf order) are pinned as well as the final weights, because
// HRP's silent-failure modes live between the stages.
type goldenCase struct {
	Description     string        `json:"description"`
	Note            string        `json:"note"`
	Corr            [][]float64   `json:"corr"`
	Cov             [][]float64   `json:"cov"`
	ExpectedDist    [][]float64   `json:"expected_dist"`
	ExpectedLinkage []LinkageStep `json:"expected_linkage"`
	ExpectedOrder   []int         `json:"expected_order"`
	ExpectedWeights []float64     `json:"expected_weights"`
	Tolerance       float64       `json:"tolerance"`
}

type goldenFile struct {
	Function string       `json:"function"`
	Source   string       `json:"source"`
	Cases    []goldenCase `json:"cases"`
}

func loadGolden(t *testing.T, name string) goldenFile {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read golden %s: %v", name, err)
	}
	var gf goldenFile
	if err := json.Unmarshal(data, &gf); err != nil {
		t.Fatalf("parse golden %s: %v", name, err)
	}
	if len(gf.Cases) == 0 {
		t.Fatalf("golden %s has no cases", name)
	}
	return gf
}

func approx(t *testing.T, label string, got, want, tol float64) {
	t.Helper()
	if math.IsNaN(want) && math.IsNaN(got) {
		return
	}
	if d := math.Abs(got - want); d > tol {
		t.Errorf("%s: got %.17g want %.17g (|diff| %.3g > tol %.3g)", label, got, want, d, tol)
	}
}

// TestGolden_FullPipeline drives every stage against the published
// worked-example golden vectors and asserts distance, linkage, leaf
// order, and final weights independently.
func TestGolden_FullPipeline(t *testing.T) {
	gf := loadGolden(t, "hrp_golden.json")
	for _, tc := range gf.Cases {
		tc := tc
		t.Run(tc.Description, func(t *testing.T) {
			n := len(tc.Corr)

			// Stage 1: correlation distance.
			dist, err := CorrelationDistance(tc.Corr)
			if err != nil {
				t.Fatalf("CorrelationDistance: %v", err)
			}
			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					approx(t, "dist", dist[i][j], tc.ExpectedDist[i][j], tc.Tolerance)
				}
			}

			// Stages 2a/2b + 3 via the whole pipeline.
			w, err := HRPWeights(tc.Cov, tc.Corr)
			if err != nil {
				t.Fatalf("HRPWeights: %v", err)
			}
			if len(w) != len(tc.ExpectedWeights) {
				t.Fatalf("weights length: got %d want %d", len(w), len(tc.ExpectedWeights))
			}
			for i := range w {
				approx(t, "weight", w[i], tc.ExpectedWeights[i], tc.Tolerance)
			}

			// Intermediate linkage + order (skip degenerate n<=1).
			if n >= 2 {
				lk, err := SingleLinkage(dist)
				if err != nil {
					t.Fatalf("SingleLinkage: %v", err)
				}
				if len(lk) != len(tc.ExpectedLinkage) {
					t.Fatalf("linkage steps: got %d want %d", len(lk), len(tc.ExpectedLinkage))
				}
				for i, step := range lk {
					exp := tc.ExpectedLinkage[i]
					if step.A != exp.A || step.B != exp.B || step.Size != exp.Size {
						t.Errorf("linkage[%d]: got {A:%d B:%d Size:%d} want {A:%d B:%d Size:%d}",
							i, step.A, step.B, step.Size, exp.A, exp.B, exp.Size)
					}
					approx(t, "linkage.Dist", step.Dist, exp.Dist, tc.Tolerance)
				}
				order, err := QuasiDiagonalize(lk, n)
				if err != nil {
					t.Fatalf("QuasiDiagonalize: %v", err)
				}
				if len(order) != len(tc.ExpectedOrder) {
					t.Fatalf("order length: got %d want %d", len(order), len(tc.ExpectedOrder))
				}
				for i := range order {
					if order[i] != tc.ExpectedOrder[i] {
						t.Errorf("order[%d]: got %d want %d", i, order[i], tc.ExpectedOrder[i])
					}
				}
			}
		})
	}
}

// TestGolden_TieBreak pins the deterministic ascending-cluster-id
// tie-break contract: when two candidate pairs share the minimum
// distance, the lexicographically smallest pair merges first.
func TestGolden_TieBreak(t *testing.T) {
	gf := loadGolden(t, "hrp_tiebreak_golden.json")
	for _, tc := range gf.Cases {
		tc := tc
		t.Run(tc.Description, func(t *testing.T) {
			dist, err := CorrelationDistance(tc.Corr)
			if err != nil {
				t.Fatalf("CorrelationDistance: %v", err)
			}
			lk, err := SingleLinkage(dist)
			if err != nil {
				t.Fatalf("SingleLinkage: %v", err)
			}
			for i, step := range lk {
				exp := tc.ExpectedLinkage[i]
				if step.A != exp.A || step.B != exp.B {
					t.Errorf("tie-break linkage[%d]: got (%d,%d) want (%d,%d) — tie-break not deterministic",
						i, step.A, step.B, exp.A, exp.B)
				}
			}
			order, err := QuasiDiagonalize(lk, len(tc.Corr))
			if err != nil {
				t.Fatalf("QuasiDiagonalize: %v", err)
			}
			for i := range order {
				if order[i] != tc.ExpectedOrder[i] {
					t.Errorf("tie-break order[%d]: got %d want %d", i, order[i], tc.ExpectedOrder[i])
				}
			}
			w, err := HRPWeights(tc.Cov, tc.Corr)
			if err != nil {
				t.Fatalf("HRPWeights: %v", err)
			}
			for i := range w {
				approx(t, "tie weight", w[i], tc.ExpectedWeights[i], tc.Tolerance)
			}
		})
	}
}

// TestTieBreak_Determinism runs the same tie case many times to confirm
// the output never varies (the map-iteration nondeterminism that afflicts
// the C# twin cannot occur here because active clusters are kept sorted).
func TestTieBreak_Determinism(t *testing.T) {
	corr := [][]float64{{1, 0.5, 0.2}, {0.5, 1, 0.5}, {0.2, 0.5, 1}}
	dist, err := CorrelationDistance(corr)
	if err != nil {
		t.Fatal(err)
	}
	var first []LinkageStep
	for iter := 0; iter < 200; iter++ {
		lk, err := SingleLinkage(dist)
		if err != nil {
			t.Fatal(err)
		}
		if first == nil {
			first = lk
			continue
		}
		for i := range lk {
			if lk[i] != first[i] {
				t.Fatalf("nondeterministic linkage at iter %d step %d: %+v vs %+v", iter, i, lk[i], first[i])
			}
		}
	}
}

// --- Property tests (independent of the golden vectors) ---

func TestProperties_SumToOneAndNonNegative(t *testing.T) {
	corr := [][]float64{
		{1, 0.3, 0.1, 0.6, 0.05},
		{0.3, 1, 0.2, 0.25, 0.1},
		{0.1, 0.2, 1, 0.15, 0.7},
		{0.6, 0.25, 0.15, 1, 0.2},
		{0.05, 0.1, 0.7, 0.2, 1},
	}
	sig := []float64{0.10, 0.22, 0.35, 0.15, 0.30}
	n := len(sig)
	cov := make([][]float64, n)
	for i := 0; i < n; i++ {
		cov[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			cov[i][j] = corr[i][j] * sig[i] * sig[j]
		}
	}
	w, err := HRPWeights(cov, corr)
	if err != nil {
		t.Fatal(err)
	}
	sum := 0.0
	for _, x := range w {
		if x < 0 {
			t.Errorf("negative weight %v", x)
		}
		sum += x
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Errorf("weights sum to %.17g, want 1", sum)
	}
}

// TestScaleInvariance confirms the documented claim that scaling every
// distance by a positive constant (the de-Prado-vs-RubberDuck 2x factor)
// leaves clustering, ordering, and weights unchanged.
func TestScaleInvariance(t *testing.T) {
	corr := [][]float64{{1, 0.7, 0.2, 0.1}, {0.7, 1, 0.15, 0.05}, {0.2, 0.15, 1, 0.6}, {0.1, 0.05, 0.6, 1}}
	dist, _ := CorrelationDistance(corr)
	scaled := make([][]float64, len(dist))
	for i := range dist {
		scaled[i] = make([]float64, len(dist))
		for j := range dist[i] {
			scaled[i][j] = 2 * dist[i][j]
		}
	}
	l1, _ := SingleLinkage(dist)
	l2, _ := SingleLinkage(scaled)
	o1, _ := QuasiDiagonalize(l1, len(corr))
	o2, _ := QuasiDiagonalize(l2, len(corr))
	for i := range o1 {
		if o1[i] != o2[i] {
			t.Fatalf("order changed under 2x scale at %d: %v vs %v", i, o1, o2)
		}
	}
	for i := range l1 {
		if l1[i].A != l2[i].A || l1[i].B != l2[i].B {
			t.Fatalf("merge pair changed under scale at step %d", i)
		}
		if math.Abs(2*l1[i].Dist-l2[i].Dist) > 1e-12 {
			t.Fatalf("scaled distance mismatch at step %d: %v vs %v", i, 2*l1[i].Dist, l2[i].Dist)
		}
	}
}

func TestCorrelationDistance_KnownValues(t *testing.T) {
	corr := [][]float64{{1, -1, 0}, {-1, 1, 0}, {0, 0, 1}}
	d, err := CorrelationDistance(corr)
	if err != nil {
		t.Fatal(err)
	}
	// rho=1 -> 0; rho=-1 -> sqrt(1)=1; rho=0 -> sqrt(1/2).
	approx(t, "d[0][0]", d[0][0], 0, 1e-15)
	approx(t, "d[0][1]", d[0][1], 1, 1e-15)
	approx(t, "d[0][2]", d[0][2], math.Sqrt(0.5), 1e-15)
}

func TestCorrelationDistance_ClampsOutOfRange(t *testing.T) {
	// A cleaned correlation matrix can carry entries a few ulps outside
	// [-1,1]; the clamp must prevent a spurious NaN.
	corr := [][]float64{{1.0000000002, 0.5}, {0.5, 1}}
	d, err := CorrelationDistance(corr)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(d[0][0]) || d[0][0] != 0 {
		t.Errorf("clamp failed: d[0][0]=%v", d[0][0])
	}
}

func TestHRPWeights_SingleAsset(t *testing.T) {
	w, err := HRPWeights([][]float64{{0.09}}, [][]float64{{1}})
	if err != nil {
		t.Fatal(err)
	}
	if len(w) != 1 || w[0] != 1 {
		t.Errorf("single asset: got %v want [1]", w)
	}
}

func TestHRPWeights_LowVolGetsMoreWeight(t *testing.T) {
	// Two near-uncorrelated assets, asset 0 much lower variance -> higher weight.
	corr := [][]float64{{1, 0.1}, {0.1, 1}}
	cov := [][]float64{{0.01, 0.001}, {0.001, 0.25}}
	w, err := HRPWeights(cov, corr)
	if err != nil {
		t.Fatal(err)
	}
	if !(w[0] > w[1]) {
		t.Errorf("low-vol asset should get more weight: %v", w)
	}
}

// --- Error-path tests ---

func TestErrors(t *testing.T) {
	if _, err := CorrelationDistance(nil); err != ErrEmptyMatrix {
		t.Errorf("empty corr: got %v want ErrEmptyMatrix", err)
	}
	if _, err := CorrelationDistance([][]float64{{1, 0}, {0}}); err != ErrNotSquare {
		t.Errorf("ragged corr: got %v want ErrNotSquare", err)
	}
	if _, err := SingleLinkage(nil); err != ErrEmptyMatrix {
		t.Errorf("empty dist: got %v want ErrEmptyMatrix", err)
	}
	if _, err := HRPWeights([][]float64{{1}}, [][]float64{{1}, {1}}); err != ErrDimensionMismatch {
		t.Errorf("dim mismatch: got %v want ErrDimensionMismatch", err)
	}
	if _, err := HRPWeights(nil, nil); err != ErrEmptyMatrix {
		t.Errorf("empty pipeline: got %v want ErrEmptyMatrix", err)
	}
	// QuasiDiagonalize with wrong step count.
	if _, err := QuasiDiagonalize([]LinkageStep{{A: 0, B: 1, Dist: 1, Size: 2}}, 4); err != ErrLinkageMismatch {
		t.Errorf("bad linkage: got %v want ErrLinkageMismatch", err)
	}
	// RecursiveBisection with out-of-range index.
	if _, err := RecursiveBisection([][]float64{{1}}, []int{0, 5}); err != ErrIndexOutOfRange {
		t.Errorf("oob order: got %v want ErrIndexOutOfRange", err)
	}
	// RecursiveBisection with duplicate index.
	if _, err := RecursiveBisection([][]float64{{1, 0}, {0, 1}}, []int{0, 0}); err != ErrIndexOutOfRange {
		t.Errorf("dup order: got %v want ErrIndexOutOfRange", err)
	}
}

func TestSingleLinkage_SingleAsset(t *testing.T) {
	lk, err := SingleLinkage([][]float64{{0}})
	if err != nil {
		t.Fatal(err)
	}
	if len(lk) != 0 {
		t.Errorf("single asset linkage: got %d steps want 0", len(lk))
	}
	order, err := QuasiDiagonalize(lk, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(order) != 1 || order[0] != 0 {
		t.Errorf("single asset order: got %v want [0]", order)
	}
}

// TestZeroVarianceFloor confirms a constant-return (zero-variance) leg
// does not blow up the inverse-variance reciprocals.
func TestZeroVarianceFloor(t *testing.T) {
	corr := [][]float64{{1, 0.1}, {0.1, 1}}
	cov := [][]float64{{0, 0}, {0, 0.04}}
	w, err := HRPWeights(cov, corr)
	if err != nil {
		t.Fatal(err)
	}
	sum := w[0] + w[1]
	if math.IsNaN(sum) || math.IsInf(sum, 0) || math.Abs(sum-1) > 1e-9 {
		t.Errorf("zero-variance floor failed: weights %v", w)
	}
}
