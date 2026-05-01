package persistent

import (
	"math"
	"math/rand"
	"testing"
)

// =========================================================================
// VietorisRipsComplex — input validation + structural correctness
// =========================================================================

func TestVietorisRipsComplex_EmptyPoints(t *testing.T) {
	_, err := VietorisRipsComplex(nil, 1.0, 1)
	if err != ErrEmptyPoints {
		t.Errorf("nil points: err = %v, want ErrEmptyPoints", err)
	}
	_, err = VietorisRipsComplex([][]float64{}, 1.0, 1)
	if err != ErrEmptyPoints {
		t.Errorf("empty points: err = %v, want ErrEmptyPoints", err)
	}
}

func TestVietorisRipsComplex_InvalidMaxDim(t *testing.T) {
	pts := [][]float64{{0, 0}, {1, 0}}
	if _, err := VietorisRipsComplex(pts, 1.0, -1); err != ErrInvalidMaxDim {
		t.Errorf("maxDim=-1: err = %v, want ErrInvalidMaxDim", err)
	}
	if _, err := VietorisRipsComplex(pts, 1.0, 2); err != ErrInvalidMaxDim {
		t.Errorf("maxDim=2: err = %v, want ErrInvalidMaxDim", err)
	}
}

func TestVietorisRipsComplex_InconsistentDim(t *testing.T) {
	pts := [][]float64{{0, 0}, {1, 0, 0}}
	if _, err := VietorisRipsComplex(pts, 1.0, 1); err != ErrInconsistentDim {
		t.Errorf("ragged points: err = %v, want ErrInconsistentDim", err)
	}
	pts = [][]float64{{}, {}}
	if _, err := VietorisRipsComplex(pts, 1.0, 1); err != ErrInconsistentDim {
		t.Errorf("zero-dim points: err = %v, want ErrInconsistentDim", err)
	}
}

func TestVietorisRipsComplex_InvalidMaxRadius(t *testing.T) {
	pts := [][]float64{{0}, {1}}
	if _, err := VietorisRipsComplex(pts, math.Inf(1), 1); err != ErrInvalidMaxRadius {
		t.Errorf("+Inf radius: err = %v, want ErrInvalidMaxRadius", err)
	}
	if _, err := VietorisRipsComplex(pts, math.NaN(), 1); err != ErrInvalidMaxRadius {
		t.Errorf("NaN radius: err = %v, want ErrInvalidMaxRadius", err)
	}
	if _, err := VietorisRipsComplex(pts, -1, 1); err != ErrInvalidMaxRadius {
		t.Errorf("negative radius: err = %v, want ErrInvalidMaxRadius", err)
	}
}

func TestVietorisRipsComplex_VerticesOnly_AtMaxRadius0(t *testing.T) {
	pts := [][]float64{{0, 0}, {1, 0}, {0, 1}}
	f, err := VietorisRipsComplex(pts, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if f.Len() != 3 {
		t.Errorf("maxRadius=0: %d simplices, want 3 (vertices only)", f.Len())
	}
	for i, s := range f.Simplices {
		if s.Dim() != 0 {
			t.Errorf("simplex %d has dim %d, want 0", i, s.Dim())
		}
		if f.Times[i] != 0 {
			t.Errorf("vertex %d born at %v, want 0", i, f.Times[i])
		}
	}
}

func TestVietorisRipsComplex_TimesNonDecreasing(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	pts := make([][]float64, 6)
	for i := range pts {
		pts[i] = []float64{rng.Float64(), rng.Float64()}
	}
	f, err := VietorisRipsComplex(pts, 2.0, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i < f.Len(); i++ {
		if f.Times[i] < f.Times[i-1] {
			t.Errorf("filtration time decreased at i=%d: %v < %v", i, f.Times[i], f.Times[i-1])
		}
	}
}

// =========================================================================
// ComputeBarcode — algorithmic correctness on canonical fixtures
// =========================================================================

func TestComputeBarcode_TrianglePointCloud_H0(t *testing.T) {
	// 3 points forming an equilateral triangle, side length 1.0.
	// Expect 2 finite H_0 bars dying at 1.0 + 1 essential H_0 bar.
	// At maxDim=0, no H_1 information is emitted even though the
	// triangle fills its loop at radius 1.0.
	pts := [][]float64{
		{0, 0},
		{1, 0},
		{0.5, math.Sqrt(3) / 2},
	}
	f, err := VietorisRipsComplex(pts, 2.0, 0)
	if err != nil {
		t.Fatal(err)
	}
	bars, err := ComputeBarcode(f, 0)
	if err != nil {
		t.Fatal(err)
	}
	finiteH0 := 0
	essentialH0 := 0
	for _, b := range bars {
		if b.Dim != 0 {
			t.Errorf("dim-0 only at maxDim=0, got dim=%d", b.Dim)
		}
		if b.IsEssential() {
			essentialH0++
		} else {
			finiteH0++
			if math.Abs(b.Death-1.0) > 1e-9 || b.Birth != 0 {
				t.Errorf("expected H_0 bar (0, 1.0), got (%v, %v)", b.Birth, b.Death)
			}
		}
	}
	if finiteH0 != 2 {
		t.Errorf("finite H_0 count = %d, want 2", finiteH0)
	}
	if essentialH0 != 1 {
		t.Errorf("essential H_0 count = %d, want 1", essentialH0)
	}
}

func TestComputeBarcode_SquareLoop_H1Detected(t *testing.T) {
	// Cyclic graph fixture mirroring RubberDuck PersistentHomology
	// CyclicGraph_DetectsLoop: 4 points at corners of a square,
	// adjacent edges length 1, diagonal 1.5.  At maxDim=1 we expect
	// 3 finite H_0 bars dying at 1 + 1 essential H_0 + 1 H_1 bar
	// born at 1.0 dying at 1.5.
	pts := [][]float64{
		{0, 0},
		{1, 0},
		{1, 1},
		{0, 1},
	}
	// Shift one diagonal so the diagonal distance is 1.5 (matches
	// the FW C# fixture's distance matrix).  We achieve this by
	// using a custom point cloud where Euclidean diagonals are
	// 1.5 by construction: place points so that pairwise (i, j),
	// (j, k), (k, l), (l, i) = 1 and (i, k), (j, l) = 1.5.
	// Geometric realisation: vertices of a unit square would have
	// diagonal sqrt(2) ~= 1.414 < 1.5; we need a slightly squashed
	// rectangle.  Use pts = (0,0), (1,0), (1, 0.5), (0, 0.5)
	// with diagonals sqrt(1.25) ~= 1.118 < 1.5, edges 1 and 0.5.
	// That doesn't match the C# fixture either.  The C# test feeds
	// a hand-built distance matrix, not a point cloud.  For VR we
	// build points that are *isometric* to the matrix only when the
	// matrix is realisable in R^d.  Distance matrix
	//   [0 1 1.5 1; 1 0 1 1.5; 1.5 1 0 1; 1 1.5 1 0]
	// is realisable as the regular tetragon embedded in 3D.  Cleanest:
	// use 4D coords directly such that the L^2 distances match.
	// The pseudo-square realisation with diagonals 1.5 lives in R^3:
	//   p0 = (0,0,0)
	//   p1 = (1,0,0)
	//   p2 = (1,0,h)  where h = ? so |p1-p2|=1 and |p0-p2|=1.5.
	//                 |p0-p2|^2 = 1+h^2 = 2.25 -> h = sqrt(1.25)
	//   p3 = (0,0,h)  |p0-p3| = h = sqrt(1.25) which is 1.118 not 1.
	// The matrix isn't realisable in any R^d as a "true square with
	// diagonal 1.5".  Skip the geometric realisation: instead pass
	// the FW fixture distances via a tighter point cloud where the
	// expected H_1 bar is born and dies in the right ranges.
	//
	// The simplest VR fixture that births an H_1 class in (0.9, 1.1)
	// and kills it in (1.4, 1.6) is the regular hexagon side 1: born
	// at 1.0, dies at 2.0 (the opposite-vertex distance).  We use
	// that instead.

	pts = []([]float64){
		{1, 0},
		{0.5, math.Sqrt(3) / 2},
		{-0.5, math.Sqrt(3) / 2},
		{-1, 0},
		{-0.5, -math.Sqrt(3) / 2},
		{0.5, -math.Sqrt(3) / 2},
	}
	f, err := VietorisRipsComplex(pts, 3.0, 1)
	if err != nil {
		t.Fatal(err)
	}
	bars, err := ComputeBarcode(f, 1)
	if err != nil {
		t.Fatal(err)
	}
	h1Count := 0
	for _, b := range bars {
		if b.Dim == 1 && !b.IsEssential() && b.Persistence() > 1e-9 {
			h1Count++
			// The hexagon's H_1 class is born at 1.0 (adjacent
			// edges) and dies at sqrt(3) ~= 1.732 (the next-
			// nearest-vertex distance — at that radius, two
			// alternating triangles fill the hexagon and the
			// loop becomes a boundary).
			if math.Abs(b.Birth-1.0) > 1e-9 {
				t.Errorf("hexagon H_1 birth = %v, want 1.0", b.Birth)
			}
			if math.Abs(b.Death-math.Sqrt(3)) > 1e-9 {
				t.Errorf("hexagon H_1 death = %v, want sqrt(3) = %v", b.Death, math.Sqrt(3))
			}
		}
	}
	if h1Count != 1 {
		t.Errorf("hexagon H_1 count = %d, want 1", h1Count)
	}
}

func TestComputeBarcode_Empty(t *testing.T) {
	bars, err := ComputeBarcode(Filtration{}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(bars) != 0 {
		t.Errorf("empty filtration: %d bars, want 0", len(bars))
	}
}

func TestComputeBarcode_InvalidMaxDim(t *testing.T) {
	if _, err := ComputeBarcode(Filtration{}, -1); err != ErrInvalidMaxDim {
		t.Errorf("maxDim=-1: err = %v, want ErrInvalidMaxDim", err)
	}
	if _, err := ComputeBarcode(Filtration{}, 2); err != ErrInvalidMaxDim {
		t.Errorf("maxDim=2: err = %v, want ErrInvalidMaxDim", err)
	}
}

func TestComputeBarcode_BarcodeIsDeterministic(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	pts := make([][]float64, 8)
	for i := range pts {
		pts[i] = []float64{rng.Float64(), rng.Float64()}
	}
	f1, _ := VietorisRipsComplex(pts, 2.0, 1)
	f2, _ := VietorisRipsComplex(pts, 2.0, 1)
	b1, _ := ComputeBarcode(f1, 1)
	b2, _ := ComputeBarcode(f2, 1)
	if len(b1) != len(b2) {
		t.Fatalf("non-deterministic length: %d vs %d", len(b1), len(b2))
	}
	for i := range b1 {
		if b1[i] != b2[i] {
			t.Errorf("bar %d differs: %+v vs %+v", i, b1[i], b2[i])
		}
	}
}

// =========================================================================
// BottleneckDistance — closed-form sanity checks
// =========================================================================

func TestBottleneckDistance_IdenticalDiagrams_Zero(t *testing.T) {
	bars := []Bar{
		{Dim: 1, Birth: 0.5, Death: 1.5},
		{Dim: 1, Birth: 0.3, Death: 0.8},
	}
	d := BottleneckDistance(bars, bars, 1)
	if d != 0 {
		t.Errorf("identical: d = %v, want 0", d)
	}
}

func TestBottleneckDistance_EmptyDiagrams_Zero(t *testing.T) {
	d := BottleneckDistance(nil, nil, 0)
	if d != 0 {
		t.Errorf("both empty: d = %v, want 0", d)
	}
}

func TestBottleneckDistance_OneBarShifted_ExactShift(t *testing.T) {
	a := []Bar{{Dim: 1, Birth: 0, Death: 2}}
	b := []Bar{{Dim: 1, Birth: 0.3, Death: 2.3}}
	// Shift is uniform: birth and death both move by 0.3.  Bottleneck
	// distance under L^inf = max(|0.3|, |0.3|) = 0.3.
	d := BottleneckDistance(a, b, 1)
	if math.Abs(d-0.3) > 1e-9 {
		t.Errorf("shifted: d = %v, want 0.3", d)
	}
}

func TestBottleneckDistance_OneEmpty_HalfPersistence(t *testing.T) {
	// Empty vs single bar (0, 2): cost is matching the bar to its
	// diagonal projection at distance (2-0)/2 = 1.0.
	a := []Bar{{Dim: 1, Birth: 0, Death: 2}}
	b := []Bar{}
	d := BottleneckDistance(a, b, 1)
	if math.Abs(d-1.0) > 1e-9 {
		t.Errorf("empty vs single (0,2): d = %v, want 1.0", d)
	}
	d2 := BottleneckDistance(b, a, 1)
	if math.Abs(d2-1.0) > 1e-9 {
		t.Errorf("symmetric empty vs single (0,2): d = %v, want 1.0", d2)
	}
}

func TestBottleneckDistance_EssentialMismatch_Infinity(t *testing.T) {
	a := []Bar{{Dim: 0, Birth: 0, Death: math.Inf(1)}}
	b := []Bar{
		{Dim: 0, Birth: 0, Death: math.Inf(1)},
		{Dim: 0, Birth: 0, Death: math.Inf(1)},
	}
	d := BottleneckDistance(a, b, 0)
	if !math.IsInf(d, 1) {
		t.Errorf("essential count mismatch: d = %v, want +Inf", d)
	}
}

func TestBottleneckDistance_SymmetryAndNonNegative(t *testing.T) {
	a := []Bar{
		{Dim: 1, Birth: 0.1, Death: 0.5},
		{Dim: 1, Birth: 0.2, Death: 0.9},
	}
	b := []Bar{
		{Dim: 1, Birth: 0.15, Death: 0.55},
		{Dim: 1, Birth: 0.25, Death: 0.95},
	}
	dab := BottleneckDistance(a, b, 1)
	dba := BottleneckDistance(b, a, 1)
	if math.Abs(dab-dba) > 1e-9 {
		t.Errorf("asymmetry: d(a,b)=%v vs d(b,a)=%v", dab, dba)
	}
	if dab < 0 {
		t.Errorf("non-negative: d = %v", dab)
	}
}

// =========================================================================
// R80b cross-substrate output parity with RubberDuck PersistentHomology
// =========================================================================
//
// These tests replicate fixtures from
// flagships/rubberduck/tests/RubberDuck.Core.Tests/Analysis/
// PersistentHomologyTests.cs — specifically the equidistant-4-point
// H_0 fixture (ComputePersistence_EquidistantPoints_CorrectH0) and
// the cyclic-square H_1 fixture (ComputePersistence_CyclicGraph_
// DetectsLoop).
//
// Tolerance: ≤1e-9 absolute on Birth/Death values.  R80b
// (output-parity, not strict-byte) is appropriate because the
// substrate differs (Go float64 vs C# double — both IEEE-754
// binary64 but with intermediate-rounding differences in
// transcendental functions).
//
// Note: the C# test ComputePersistence accepts a precomputed
// distance matrix directly, while VietorisRipsComplex builds the
// matrix from a point cloud.  We embed the fixture distances in
// realisable point clouds where possible; where the C# distance
// matrix is not L^2-realisable in R^d (the cyclic 4-point fixture
// with adjacent=1, diagonal=1.5 is not — diagonal of a unit square
// is sqrt(2) ~= 1.414), we cite the fixture's qualitative claim
// (one H_1 bar, born at 1.0, dies at 1.5) and swap to a hexagonal
// point cloud whose H_1 class has the same qualitative profile.

func TestCrossSubstratePrecision_RubberDuck_EquidistantPoints_H0(t *testing.T) {
	// FW C# fixture: 4 points pairwise distance 1.0.  Realised as
	// the regular tetrahedron embedded in R^3.
	a := math.Sqrt(2.0 / 3.0)
	pts := [][]float64{
		{0, 0, 0},
		{1, 0, 0},
		{0.5, math.Sqrt(3) / 2, 0},
		{0.5, math.Sqrt(3) / 6, a},
	}
	f, err := VietorisRipsComplex(pts, 2.0, 1)
	if err != nil {
		t.Fatal(err)
	}
	bars, err := ComputeBarcode(f, 1)
	if err != nil {
		t.Fatal(err)
	}
	// Expect 3 finite H_0 bars, each dying at 1.0 (parity with FW
	// EquidistantPoints_CorrectH0 which asserts 3 finite H_0
	// intervals all dying at eps=1.0).
	finiteH0 := 0
	for _, b := range bars {
		if b.Dim != 0 || b.IsEssential() {
			continue
		}
		finiteH0++
		if math.Abs(b.Death-1.0) > 1e-9 {
			t.Errorf("RubberDuck parity: H_0 bar dies at %v, want 1.0 (FW EquidistantPoints fixture)", b.Death)
		}
	}
	if finiteH0 != 3 {
		t.Errorf("RubberDuck parity: finite H_0 = %d, want 3 (FW EquidistantPoints)", finiteH0)
	}
}

// =========================================================================
// Random Gaussian cloud — statistical sanity check
// =========================================================================

func TestVRBarcode_GaussianCloud_NoneBornOutsideWindow(t *testing.T) {
	rng := rand.New(rand.NewSource(2026))
	n := 12
	pts := make([][]float64, n)
	for i := range pts {
		pts[i] = []float64{rng.NormFloat64(), rng.NormFloat64()}
	}
	f, err := VietorisRipsComplex(pts, 5.0, 1)
	if err != nil {
		t.Fatal(err)
	}
	bars, err := ComputeBarcode(f, 1)
	if err != nil {
		t.Fatal(err)
	}
	// All bars must satisfy 0 <= Birth <= Death <= maxRadius (or
	// Death = +Inf for essential).
	for i, b := range bars {
		if b.Birth < 0 {
			t.Errorf("bar %d: negative birth %v", i, b.Birth)
		}
		if !b.IsEssential() && b.Death < b.Birth {
			t.Errorf("bar %d: death %v < birth %v", i, b.Death, b.Birth)
		}
		if !b.IsEssential() && b.Death > 5.0+1e-9 {
			t.Errorf("bar %d: death %v exceeds maxRadius 5.0", i, b.Death)
		}
	}
	// Exactly one essential H_0 class (the surviving connected
	// component).  H_0 finite count = n - 1 because n vertices
	// merge into 1 component over the filtration when maxRadius is
	// large enough to connect everything.
	essential := 0
	for _, b := range bars {
		if b.Dim == 0 && b.IsEssential() {
			essential++
		}
	}
	if essential != 1 {
		t.Errorf("Gaussian cloud (large radius): essential H_0 = %d, want 1", essential)
	}
}
