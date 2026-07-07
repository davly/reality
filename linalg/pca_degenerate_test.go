package linalg

import (
	"math"
	"testing"
)

// Regression for the PCA degenerate-eigenvalue fix. With repeated eigenvalues
// (isotropic data) inverse iteration converged every component to the same
// direction; Gram-Schmidt then collapsed the residual to ~1e-16 and the
// re-normalization blew it back up to +/- the previous component, so PCA
// returned NON-orthogonal components (dot = -1). Deflation now yields a proper
// orthonormal basis of the eigenspace.

func dot(a []float64, ai int, b []float64, bi int, nf int) float64 {
	s := 0.0
	for i := 0; i < nf; i++ {
		s += a[ai*nf+i] * b[bi*nf+i]
	}
	return s
}

func TestPCA_Degenerate2D_Orthogonal(t *testing.T) {
	// 8 points on a unit circle -> isotropic 2x2 covariance (eigenvalue x2).
	data := make([]float64, 16)
	for k := 0; k < 8; k++ {
		ang := float64(k) * math.Pi / 4
		data[2*k] = math.Cos(ang)
		data[2*k+1] = math.Sin(ang)
	}
	comp := make([]float64, 4)
	expl := make([]float64, 2)
	PCA(data, 8, 2, 2, comp, expl)

	if d := dot(comp, 0, comp, 1, 2); math.Abs(d) > 1e-6 {
		t.Errorf("isotropic PCA: dot(c0,c1)=%.6f (was -1 pre-fix); c0=[%.4f,%.4f] c1=[%.4f,%.4f]",
			d, comp[0], comp[1], comp[2], comp[3])
	}
	for c := 0; c < 2; c++ {
		if n := dot(comp, c, comp, c, 2); math.Abs(n-1) > 1e-6 {
			t.Errorf("component %d not unit norm: %.6f", c, n)
		}
	}
}

func TestPCA_Degenerate3D_Orthogonal(t *testing.T) {
	// 6 axis points +/-e_i -> isotropic 3x3 covariance (eigenvalue x3).
	data := []float64{
		1, 0, 0, -1, 0, 0,
		0, 1, 0, 0, -1, 0,
		0, 0, 1, 0, 0, -1,
	}
	comp := make([]float64, 9)
	expl := make([]float64, 3)
	PCA(data, 6, 3, 3, comp, expl)

	for a := 0; a < 3; a++ {
		for b := a + 1; b < 3; b++ {
			if d := dot(comp, a, comp, b, 3); math.Abs(d) > 1e-6 {
				t.Errorf("isotropic 3D PCA: dot(c%d,c%d)=%.6f; want ~0", a, b, d)
			}
		}
		if n := dot(comp, a, comp, a, 3); math.Abs(n-1) > 1e-6 {
			t.Errorf("component %d not unit norm: %.6f", a, n)
		}
	}
}
