package prob

import (
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

// ---------------------------------------------------------------------------
// NormalPDF
// ---------------------------------------------------------------------------

func TestNormalPDF(t *testing.T) {
	tests := []struct {
		name  string
		x     float64
		mu    float64
		sigma float64
		want  float64
		tol   float64
	}{
		{"peak at mean", 0, 0, 1, 0.3989422804014327, 1e-15},
		{"at x=1", 1, 0, 1, 0.24197072451914337, 1e-15},
		{"symmetry: PDF(-1) = PDF(1)", -1, 0, 1, 0.24197072451914337, 1e-15},
		{"at x=2", 2, 0, 1, 0.05399096651318806, 1e-15},
		{"at x=3 (tail)", 3, 0, 1, 0.0044318484119380075, 1e-15},
		{"mu=5, sigma=2 at peak", 5, 5, 2, 0.19947114020071635, 1e-15},
		{"mu=10, sigma=0.5 at peak", 10, 10, 0.5, 0.7978845608028654, 1e-14},
		{"sigma<=0 returns NaN", 0, 0, 0, math.NaN(), 0},
		{"negative sigma returns NaN", 0, 0, -1, math.NaN(), 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NormalPDF(tt.x, tt.mu, tt.sigma)
			if math.IsNaN(tt.want) {
				if !math.IsNaN(got) {
					t.Errorf("NormalPDF(%v, %v, %v) = %v, want NaN",
						tt.x, tt.mu, tt.sigma, got)
				}
				return
			}
			if math.Abs(got-tt.want) > tt.tol {
				t.Errorf("NormalPDF(%v, %v, %v) = %v, want %v (tol %v)",
					tt.x, tt.mu, tt.sigma, got, tt.want, tt.tol)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// NormalCDF
// ---------------------------------------------------------------------------

func TestNormalCDF(t *testing.T) {
	tests := []struct {
		name  string
		x     float64
		mu    float64
		sigma float64
		want  float64
		tol   float64
	}{
		{"0.5 at mean", 0, 0, 1, 0.5, 1e-15},
		{"~0.8413 at mu+sigma", 1, 0, 1, 0.8413447460685429, 1e-12},
		{"~0.1587 at mu-sigma", -1, 0, 1, 0.15865525393145707, 1e-12},
		{"~0.9772 at mu+2sigma", 2, 0, 1, 0.9772498680518208, 1e-12},
		{"~0.0228 at mu-2sigma", -2, 0, 1, 0.02275013194817921, 1e-12},
		{"~0.9987 at mu+3sigma", 3, 0, 1, 0.9986501019683699, 1e-12},
		{"symmetry: CDF(-x) + CDF(x) = 1", 1.5, 0, 1, 0, 0}, // checked below
		{"sigma<=0 returns NaN", 0, 0, 0, math.NaN(), 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NormalCDF(tt.x, tt.mu, tt.sigma)
			if math.IsNaN(tt.want) {
				if !math.IsNaN(got) {
					t.Errorf("NormalCDF(%v, %v, %v) = %v, want NaN",
						tt.x, tt.mu, tt.sigma, got)
				}
				return
			}
			if tt.name == "symmetry: CDF(-x) + CDF(x) = 1" {
				sum := NormalCDF(1.5, 0, 1) + NormalCDF(-1.5, 0, 1)
				if math.Abs(sum-1.0) > 1e-14 {
					t.Errorf("CDF(1.5) + CDF(-1.5) = %v, want 1.0", sum)
				}
				return
			}
			if math.Abs(got-tt.want) > tt.tol {
				t.Errorf("NormalCDF(%v, %v, %v) = %v, want %v",
					tt.x, tt.mu, tt.sigma, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// NormalQuantile
// ---------------------------------------------------------------------------

func TestNormalQuantile(t *testing.T) {
	tests := []struct {
		name  string
		p     float64
		mu    float64
		sigma float64
		want  float64
		tol   float64
	}{
		{"median (p=0.5)", 0.5, 0, 1, 0, 1e-8},
		{"p=0.8413 ~ x=1", 0.8413447460685429, 0, 1, 1.0, 1e-6},
		{"p=0.9772 ~ x=2", 0.9772498680518208, 0, 1, 2.0, 1e-6},
		{"p=0.025 ~ x=-1.96", 0.025, 0, 1, -1.959963984540054, 1e-6},
		{"p=0.975 ~ x=1.96", 0.975, 0, 1, 1.959963984540054, 1e-6},
		{"non-standard: mu=10, sigma=3", 0.5, 10, 3, 10.0, 1e-8},
		{"sigma<=0 returns NaN", 0.5, 0, 0, math.NaN(), 0},
		{"p<=0 returns NaN", 0, 0, 1, math.NaN(), 0},
		{"p>=1 returns NaN", 1, 0, 1, math.NaN(), 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NormalQuantile(tt.p, tt.mu, tt.sigma)
			if math.IsNaN(tt.want) {
				if !math.IsNaN(got) {
					t.Errorf("NormalQuantile(%v, %v, %v) = %v, want NaN",
						tt.p, tt.mu, tt.sigma, got)
				}
				return
			}
			if math.Abs(got-tt.want) > tt.tol {
				t.Errorf("NormalQuantile(%v, %v, %v) = %v, want %v (tol %v)",
					tt.p, tt.mu, tt.sigma, got, tt.want, tt.tol)
			}
		})
	}
}

func TestNormalQuantileCDFRoundtrip(t *testing.T) {
	// Quantile(CDF(x)) should return x (approximately).
	values := []float64{-3, -2, -1, 0, 1, 2, 3}
	for _, x := range values {
		p := NormalCDF(x, 0, 1)
		got := NormalQuantile(p, 0, 1)
		if math.Abs(got-x) > 1e-6 {
			t.Errorf("roundtrip(%v): CDF -> Quantile = %v", x, got)
		}
	}
}

// ---------------------------------------------------------------------------
// ExponentialPDF / ExponentialCDF
// ---------------------------------------------------------------------------

func TestExponentialPDF(t *testing.T) {
	tests := []struct {
		name   string
		x      float64
		lambda float64
		want   float64
		tol    float64
	}{
		{"at x=0", 0, 1, 1.0, 1e-15},
		{"at x=1, lambda=1", 1, 1, 0.36787944117144233, 1e-14},
		{"at x=2, lambda=0.5", 2, 0.5, 0.5 * math.Exp(-1.0), 1e-14},
		{"x<0 returns 0", -1, 1, 0, 1e-15},
		{"lambda<=0 returns NaN", 0, 0, math.NaN(), 0},
		{"lambda=2 at x=0", 0, 2, 2.0, 1e-15},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ExponentialPDF(tt.x, tt.lambda)
			if math.IsNaN(tt.want) {
				if !math.IsNaN(got) {
					t.Errorf("ExponentialPDF(%v, %v) = %v, want NaN", tt.x, tt.lambda, got)
				}
				return
			}
			if math.Abs(got-tt.want) > tt.tol {
				t.Errorf("ExponentialPDF(%v, %v) = %v, want %v",
					tt.x, tt.lambda, got, tt.want)
			}
		})
	}
}

func TestExponentialCDF(t *testing.T) {
	tests := []struct {
		name   string
		x      float64
		lambda float64
		want   float64
		tol    float64
	}{
		{"at x=0", 0, 1, 0, 1e-15},
		{"at x=1, lambda=1", 1, 1, 1 - math.Exp(-1.0), 1e-14},
		{"at x=2, lambda=0.5", 2, 0.5, 1 - math.Exp(-1.0), 1e-14},
		{"x<0 returns 0", -1, 1, 0, 1e-15},
		{"lambda<=0 returns NaN", 0, 0, math.NaN(), 0},
		{"large x approaches 1", 100, 1, 1.0, 1e-10},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ExponentialCDF(tt.x, tt.lambda)
			if math.IsNaN(tt.want) {
				if !math.IsNaN(got) {
					t.Errorf("ExponentialCDF(%v, %v) = %v, want NaN", tt.x, tt.lambda, got)
				}
				return
			}
			if math.Abs(got-tt.want) > tt.tol {
				t.Errorf("ExponentialCDF(%v, %v) = %v, want %v",
					tt.x, tt.lambda, got, tt.want)
			}
		})
	}
}

func TestExponentialMemoryless(t *testing.T) {
	// Memoryless property: P(X > s+t | X > s) = P(X > t)
	// i.e., (1 - CDF(s+t)) / (1 - CDF(s)) = 1 - CDF(t)
	lambda := 2.0
	s, dt := 1.0, 0.5
	lhs := (1 - ExponentialCDF(s+dt, lambda)) / (1 - ExponentialCDF(s, lambda))
	rhs := 1 - ExponentialCDF(dt, lambda)
	if math.Abs(lhs-rhs) > 1e-14 {
		t.Errorf("memoryless property: LHS=%v, RHS=%v", lhs, rhs)
	}
}

// ---------------------------------------------------------------------------
// UniformPDF / UniformCDF
// ---------------------------------------------------------------------------

func TestUniformPDF(t *testing.T) {
	tests := []struct {
		name string
		x    float64
		a    float64
		b    float64
		want float64
	}{
		{"in range", 0.5, 0, 1, 1.0},
		{"below range", -0.1, 0, 1, 0},
		{"above range", 1.1, 0, 1, 0},
		{"at boundary a", 0, 0, 1, 1.0},
		{"at boundary b", 1, 0, 1, 1.0},
		{"wider range", 5, 0, 10, 0.1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := UniformPDF(tt.x, tt.a, tt.b)
			if math.Abs(got-tt.want) > 1e-15 {
				t.Errorf("UniformPDF(%v, %v, %v) = %v, want %v",
					tt.x, tt.a, tt.b, got, tt.want)
			}
		})
	}
}

func TestUniformPDFInvalid(t *testing.T) {
	got := UniformPDF(0.5, 1, 0) // a >= b
	if !math.IsNaN(got) {
		t.Errorf("UniformPDF with a>=b should be NaN, got %v", got)
	}
}

func TestUniformCDF(t *testing.T) {
	tests := []struct {
		name string
		x    float64
		a    float64
		b    float64
		want float64
	}{
		{"below range", -1, 0, 1, 0},
		{"at a", 0, 0, 1, 0},
		{"midpoint", 0.5, 0, 1, 0.5},
		{"at b", 1, 0, 1, 1.0},
		{"above range", 2, 0, 1, 1.0},
		{"quarter", 0.25, 0, 1, 0.25},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := UniformCDF(tt.x, tt.a, tt.b)
			if math.Abs(got-tt.want) > 1e-15 {
				t.Errorf("UniformCDF(%v, %v, %v) = %v, want %v",
					tt.x, tt.a, tt.b, got, tt.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// BetaPDF
// ---------------------------------------------------------------------------

func TestBetaPDF(t *testing.T) {
	tests := []struct {
		name  string
		x     float64
		alpha float64
		beta  float64
		want  float64
		tol   float64
	}{
		{"uniform (1,1) at 0.5", 0.5, 1, 1, 1.0, 1e-15},
		{"symmetric (2,2) at 0.5", 0.5, 2, 2, 1.5, 1e-14},
		{"(2,5) at 0.2", 0.2, 2, 5, 2.4576, 1e-10},
		{"(5,1) at 0.8", 0.8, 5, 1, 2.048, 1e-12},
		{"U-shaped (0.5,0.5) at 0.5", 0.5, 0.5, 0.5, 0.6366197723675814, 1e-12},
		{"(2,2) at x=0 boundary", 0.0, 2, 2, 0, 1e-15},
		{"(2,2) at x=1 boundary", 1.0, 2, 2, 0, 1e-15},
		{"(10,10) at 0.5 (peaked)", 0.5, 10, 10, 3.523941040039057, 1e-8},
		{"alpha<=0 returns NaN", 0.5, 0, 1, math.NaN(), 0},
		{"beta<=0 returns NaN", 0.5, 1, 0, math.NaN(), 0},
		{"x<0 returns 0", -0.1, 2, 2, 0, 1e-15},
		{"x>1 returns 0", 1.1, 2, 2, 0, 1e-15},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BetaPDF(tt.x, tt.alpha, tt.beta)
			if math.IsNaN(tt.want) {
				if !math.IsNaN(got) {
					t.Errorf("BetaPDF(%v, %v, %v) = %v, want NaN",
						tt.x, tt.alpha, tt.beta, got)
				}
				return
			}
			if math.Abs(got-tt.want) > tt.tol {
				t.Errorf("BetaPDF(%v, %v, %v) = %v, want %v (tol %v)",
					tt.x, tt.alpha, tt.beta, got, tt.want, tt.tol)
			}
		})
	}
}

func TestBetaPDFUShapedAtBoundaries(t *testing.T) {
	// For alpha < 1 and beta < 1, PDF -> +Inf at x=0 and x=1.
	if !math.IsInf(BetaPDF(0, 0.5, 0.5), 1) {
		t.Error("BetaPDF(0, 0.5, 0.5) should be +Inf")
	}
	if !math.IsInf(BetaPDF(1, 0.5, 0.5), 1) {
		t.Error("BetaPDF(1, 0.5, 0.5) should be +Inf")
	}
}

// ---------------------------------------------------------------------------
// BetaCDF
// ---------------------------------------------------------------------------

func TestBetaCDF(t *testing.T) {
	tests := []struct {
		name  string
		x     float64
		alpha float64
		beta  float64
		want  float64
		tol   float64
	}{
		{"0 at x=0", 0, 2, 2, 0, 1e-15},
		{"1 at x=1", 1, 2, 2, 1, 1e-15},
		{"0.5 at x=0.5 for symmetric", 0.5, 2, 2, 0.5, 1e-10},
		{"uniform (1,1) at 0.5", 0.5, 1, 1, 0.5, 1e-10},
		{"uniform (1,1) at 0.25", 0.25, 1, 1, 0.25, 1e-10},
		{"(5,1) at 0.5", 0.5, 5, 1, 0.03125, 1e-10},
		{"(1,5) at 0.5", 0.5, 1, 5, 0.96875, 1e-10},
		{"alpha<=0 returns NaN", 0.5, 0, 1, math.NaN(), 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BetaCDF(tt.x, tt.alpha, tt.beta)
			if math.IsNaN(tt.want) {
				if !math.IsNaN(got) {
					t.Errorf("BetaCDF(%v, %v, %v) = %v, want NaN",
						tt.x, tt.alpha, tt.beta, got)
				}
				return
			}
			if math.Abs(got-tt.want) > tt.tol {
				t.Errorf("BetaCDF(%v, %v, %v) = %v, want %v (tol %v)",
					tt.x, tt.alpha, tt.beta, got, tt.want, tt.tol)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// PoissonPMF
// ---------------------------------------------------------------------------

func TestPoissonPMF(t *testing.T) {
	tests := []struct {
		name   string
		k      int
		lambda float64
		want   float64
		tol    float64
	}{
		{"k=0, lambda=1", 0, 1, 0.36787944117144233, 1e-14},
		{"k=1, lambda=1", 1, 1, 0.36787944117144233, 1e-14},
		{"k=2, lambda=1", 2, 1, 0.18393972058572114, 1e-14},
		{"k=5, lambda=5 (mode)", 5, 5, 0.17546736976785068, 1e-12},
		{"k=4, lambda=5", 4, 5, 0.17546736976785068, 1e-12},
		{"k=0, lambda=5", 0, 5, 0.006737946999085467, 1e-14},
		{"k<0 returns 0", -1, 1, 0, 1e-15},
		{"lambda<=0 returns NaN", 0, 0, math.NaN(), 0},
		{"k=10, lambda=3", 10, 3, 0.0008101511794681914, 1e-12},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PoissonPMF(tt.k, tt.lambda)
			if math.IsNaN(tt.want) {
				if !math.IsNaN(got) {
					t.Errorf("PoissonPMF(%v, %v) = %v, want NaN", tt.k, tt.lambda, got)
				}
				return
			}
			if math.Abs(got-tt.want) > tt.tol {
				t.Errorf("PoissonPMF(%v, %v) = %v, want %v (tol %v)",
					tt.k, tt.lambda, got, tt.want, tt.tol)
			}
		})
	}
}

func TestPoissonPMFSumsToOne(t *testing.T) {
	// Sum of PMF for k=0..50 with lambda=5 should be ~1.
	lambda := 5.0
	sum := 0.0
	for k := 0; k <= 50; k++ {
		sum += PoissonPMF(k, lambda)
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("sum of PoissonPMF(k=0..50, lambda=5) = %v, want ~1.0", sum)
	}
}

func TestPoissonPMFModeAtFloorLambda(t *testing.T) {
	// For non-integer lambda, mode is at floor(lambda).
	lambda := 4.7
	modeK := int(math.Floor(lambda))
	modePMF := PoissonPMF(modeK, lambda)

	// PMF at mode should be >= PMF at mode-1 and mode+1.
	if modePMF < PoissonPMF(modeK-1, lambda) {
		t.Errorf("PMF at mode (%v) < PMF at mode-1", modePMF)
	}
	if modePMF < PoissonPMF(modeK+1, lambda) {
		t.Errorf("PMF at mode (%v) < PMF at mode+1", modePMF)
	}
}

// ---------------------------------------------------------------------------
// PoissonCDF
// ---------------------------------------------------------------------------

func TestPoissonCDF(t *testing.T) {
	tests := []struct {
		name   string
		k      int
		lambda float64
		want   float64
		tol    float64
	}{
		{"k=0, lambda=1", 0, 1, 0.36787944117144233, 1e-10},
		{"k=1, lambda=1", 1, 1, 0.7357588823428847, 1e-10},
		{"k=5, lambda=2", 5, 2, 0.9834363915193856, 1e-8},
		{"k<0 returns 0", -1, 1, 0, 1e-15},
		{"lambda<=0 returns NaN", 0, 0, math.NaN(), 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PoissonCDF(tt.k, tt.lambda)
			if math.IsNaN(tt.want) {
				if !math.IsNaN(got) {
					t.Errorf("PoissonCDF(%v, %v) = %v, want NaN", tt.k, tt.lambda, got)
				}
				return
			}
			if math.Abs(got-tt.want) > tt.tol {
				t.Errorf("PoissonCDF(%v, %v) = %v, want %v (tol %v)",
					tt.k, tt.lambda, got, tt.want, tt.tol)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// BinomialPMF
// ---------------------------------------------------------------------------

func TestBinomialPMF(t *testing.T) {
	tests := []struct {
		name string
		k    int
		n    int
		p    float64
		want float64
		tol  float64
	}{
		{"k=0, n=10, p=0.5", 0, 10, 0.5, 0.0009765625, 1e-12},
		{"k=5, n=10, p=0.5 (peak)", 5, 10, 0.5, 0.24609375, 1e-10},
		{"k=10, n=10, p=0.5", 10, 10, 0.5, 0.0009765625, 1e-12},
		{"k=3, n=10, p=0.3", 3, 10, 0.3, 0.26682793200, 1e-8},
		{"k<0 returns 0", -1, 10, 0.5, 0, 1e-15},
		{"k>n returns 0", 11, 10, 0.5, 0, 1e-15},
		{"p=0, k=0", 0, 10, 0, 1, 1e-15},
		{"p=0, k=5", 5, 10, 0, 0, 1e-15},
		{"p=1, k=n", 10, 10, 1, 1, 1e-15},
		{"p=1, k!=n", 5, 10, 1, 0, 1e-15},
		{"p<0 returns NaN", 0, 10, -0.1, math.NaN(), 0},
		{"p>1 returns NaN", 0, 10, 1.1, math.NaN(), 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BinomialPMF(tt.k, tt.n, tt.p)
			if math.IsNaN(tt.want) {
				if !math.IsNaN(got) {
					t.Errorf("BinomialPMF(%v, %v, %v) = %v, want NaN",
						tt.k, tt.n, tt.p, got)
				}
				return
			}
			if math.Abs(got-tt.want) > tt.tol {
				t.Errorf("BinomialPMF(%v, %v, %v) = %v, want %v (tol %v)",
					tt.k, tt.n, tt.p, got, tt.want, tt.tol)
			}
		})
	}
}

func TestBinomialPMFSumsToOne(t *testing.T) {
	n, p := 20, 0.3
	sum := 0.0
	for k := 0; k <= n; k++ {
		sum += BinomialPMF(k, n, p)
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("sum of BinomialPMF(k=0..20, n=20, p=0.3) = %v, want ~1.0", sum)
	}
}

func TestBinomialPMFPeakAtNP(t *testing.T) {
	// Peak should be at or near n*p.
	n, p := 20, 0.3
	peakK := int(math.Round(float64(n) * p))
	peakPMF := BinomialPMF(peakK, n, p)

	// Check that neighbors are not larger.
	if peakK > 0 && BinomialPMF(peakK-1, n, p) > peakPMF+1e-14 {
		t.Errorf("BinomialPMF at k=%d is larger than at peak k=%d", peakK-1, peakK)
	}
	if peakK < n && BinomialPMF(peakK+1, n, p) > peakPMF+1e-14 {
		t.Errorf("BinomialPMF at k=%d is larger than at peak k=%d", peakK+1, peakK)
	}
}

// ---------------------------------------------------------------------------
// BinomialCDF
// ---------------------------------------------------------------------------

func TestBinomialCDF(t *testing.T) {
	tests := []struct {
		name string
		k    int
		n    int
		p    float64
		want float64
		tol  float64
	}{
		{"k<0 returns 0", -1, 10, 0.5, 0, 1e-15},
		{"k>=n returns 1", 10, 10, 0.5, 1, 1e-15},
		{"k=4, n=10, p=0.5", 4, 10, 0.5, 0.376953125, 1e-8},
		{"k=5, n=10, p=0.5", 5, 10, 0.5, 0.623046875, 1e-8},
		{"p<0 returns NaN", 5, 10, -0.1, math.NaN(), 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BinomialCDF(tt.k, tt.n, tt.p)
			if math.IsNaN(tt.want) {
				if !math.IsNaN(got) {
					t.Errorf("BinomialCDF(%v, %v, %v) = %v, want NaN",
						tt.k, tt.n, tt.p, got)
				}
				return
			}
			if math.Abs(got-tt.want) > tt.tol {
				t.Errorf("BinomialCDF(%v, %v, %v) = %v, want %v (tol %v)",
					tt.k, tt.n, tt.p, got, tt.want, tt.tol)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// RegularizedBetaInc (mathutil.go)
// ---------------------------------------------------------------------------

func TestRegularizedBetaInc(t *testing.T) {
	tests := []struct {
		name string
		x    float64
		a    float64
		b    float64
		want float64
		tol  float64
	}{
		{"I_0(1,1) = 0", 0, 1, 1, 0, 1e-15},
		{"I_1(1,1) = 1", 1, 1, 1, 1, 1e-15},
		{"I_0.5(1,1) = 0.5 (uniform)", 0.5, 1, 1, 0.5, 1e-12},
		{"I_0.5(2,2) = 0.5 (symmetric)", 0.5, 2, 2, 0.5, 1e-10},
		{"I_0.3(2,5)", 0.3, 2, 5, 0.57982500, 1e-4},
		{"invalid x < 0", -0.1, 1, 1, math.NaN(), 0},
		{"invalid a <= 0", 0.5, 0, 1, math.NaN(), 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := RegularizedBetaInc(tt.x, tt.a, tt.b)
			if math.IsNaN(tt.want) {
				if !math.IsNaN(got) {
					t.Errorf("RegularizedBetaInc(%v, %v, %v) = %v, want NaN",
						tt.x, tt.a, tt.b, got)
				}
				return
			}
			if math.Abs(got-tt.want) > tt.tol {
				t.Errorf("RegularizedBetaInc(%v, %v, %v) = %v, want %v (tol %v)",
					tt.x, tt.a, tt.b, got, tt.want, tt.tol)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// TTestOneSample
// ---------------------------------------------------------------------------

func TestTTestOneSample(t *testing.T) {
	// Known dataset: mean=5, want to test if mu0=5 (expect p~1, t~0).
	data := []float64{4, 5, 5, 6, 5, 4, 6, 5, 5, 5}
	tStat, pVal := TTestOneSample(data, 5.0)
	if math.Abs(tStat) > 0.5 {
		t.Errorf("t-stat for data with mean=5, mu0=5 should be ~0, got %v", tStat)
	}
	if pVal < 0.5 {
		t.Errorf("p-value for data matching mu0 should be high, got %v", pVal)
	}

	// Data clearly different from mu0=0.
	data2 := []float64{10, 11, 12, 10, 11, 13, 12, 10, 11, 12}
	tStat2, pVal2 := TTestOneSample(data2, 0)
	if tStat2 < 10 {
		t.Errorf("t-stat should be large for data far from mu0=0, got %v", tStat2)
	}
	if pVal2 > 0.001 {
		t.Errorf("p-value should be very small, got %v", pVal2)
	}

	// Too few data points.
	_, pNaN := TTestOneSample([]float64{1.0}, 0)
	if !math.IsNaN(pNaN) {
		t.Errorf("expected NaN for n < 2, got %v", pNaN)
	}
}

func TestTTestOneSampleKnownValues(t *testing.T) {
	// Dataset with known t-statistic.
	// data = [2, 4, 6, 8, 10], mean = 6, s = sqrt(10), n = 5
	// t = (6 - 5) / (sqrt(10)/sqrt(5)) = 1 / sqrt(2) ~ 0.7071
	data := []float64{2, 4, 6, 8, 10}
	tStat, _ := TTestOneSample(data, 5)
	expected := 1.0 / math.Sqrt(2)
	if math.Abs(tStat-expected) > 1e-10 {
		t.Errorf("t-stat = %v, want %v", tStat, expected)
	}
}

// ---------------------------------------------------------------------------
// TTestTwoSample
// ---------------------------------------------------------------------------

func TestTTestTwoSample(t *testing.T) {
	// Two identical samples -> t ~ 0, p ~ 1.
	same := []float64{1, 2, 3, 4, 5}
	tStat, pVal := TTestTwoSample(same, same)
	if math.Abs(tStat) > 1e-10 {
		t.Errorf("identical samples: t-stat should be ~0, got %v", tStat)
	}
	if pVal < 0.9 {
		t.Errorf("identical samples: p-value should be ~1, got %v", pVal)
	}

	// Two clearly different samples.
	low := []float64{1, 2, 3, 1, 2, 3, 1, 2, 3, 2}
	high := []float64{10, 11, 12, 10, 11, 12, 10, 11, 12, 11}
	tStat2, pVal2 := TTestTwoSample(low, high)
	if math.Abs(tStat2) < 5 {
		t.Errorf("different samples: |t-stat| should be large, got %v", tStat2)
	}
	if pVal2 > 0.001 {
		t.Errorf("different samples: p-value should be very small, got %v", pVal2)
	}

	// Too few data points.
	_, pNaN := TTestTwoSample([]float64{1}, []float64{2, 3})
	if !math.IsNaN(pNaN) {
		t.Errorf("expected NaN for n1 < 2, got %v", pNaN)
	}
}

// ---------------------------------------------------------------------------
// ChiSquaredTest
// ---------------------------------------------------------------------------

func TestChiSquaredTest(t *testing.T) {
	// Observed matches expected perfectly -> chi2 = 0, p ~ 1.
	obs := []float64{50, 50, 50, 50}
	exp := []float64{50, 50, 50, 50}
	chi2, pVal := ChiSquaredTest(obs, exp)
	if chi2 != 0 {
		t.Errorf("perfect match: chi2 should be 0, got %v", chi2)
	}
	if pVal < 0.99 {
		t.Errorf("perfect match: p-value should be ~1, got %v", pVal)
	}

	// Clear deviation: expect uniform, observe heavily skewed.
	obs2 := []float64{90, 10, 10, 10}
	exp2 := []float64{30, 30, 30, 30}
	chi2_2, pVal2 := ChiSquaredTest(obs2, exp2)
	if chi2_2 < 10 {
		t.Errorf("skewed: chi2 should be large, got %v", chi2_2)
	}
	if pVal2 > 0.01 {
		t.Errorf("skewed: p-value should be very small, got %v", pVal2)
	}

	// Mismatched lengths.
	_, pNaN := ChiSquaredTest([]float64{1, 2}, []float64{1})
	if !math.IsNaN(pNaN) {
		t.Errorf("mismatched lengths: expected NaN, got %v", pNaN)
	}

	// Expected value <= 0.
	_, pNaN2 := ChiSquaredTest([]float64{10, 10}, []float64{10, 0})
	if !math.IsNaN(pNaN2) {
		t.Errorf("zero expected: expected NaN, got %v", pNaN2)
	}

	// Too few categories.
	_, pNaN3 := ChiSquaredTest([]float64{10}, []float64{10})
	if !math.IsNaN(pNaN3) {
		t.Errorf("single category: expected NaN, got %v", pNaN3)
	}
}

func TestChiSquaredTestKnownValue(t *testing.T) {
	// Known chi-squared: observed = [20, 30, 50], expected = [30, 30, 40]
	// chi2 = (20-30)^2/30 + (30-30)^2/30 + (50-40)^2/40
	//      = 100/30 + 0 + 100/40 = 3.333... + 2.5 = 5.833...
	obs := []float64{20, 30, 50}
	exp := []float64{30, 30, 40}
	chi2, _ := ChiSquaredTest(obs, exp)
	want := 100.0/30.0 + 0 + 100.0/40.0
	if math.Abs(chi2-want) > 1e-12 {
		t.Errorf("chi2 = %v, want %v", chi2, want)
	}
}

// ---------------------------------------------------------------------------
// Golden-file tests
// ---------------------------------------------------------------------------

func TestGoldenNormalPDFCDF(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/prob/normal_pdf_cdf.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			x := testutil.InputFloat64(t, tc, "x")
			mu := testutil.InputFloat64(t, tc, "mu")
			sigma := testutil.InputFloat64(t, tc, "sigma")

			// Determine which function from the "fn" field.
			fnVal, ok := tc.Inputs["fn"]
			if !ok {
				t.Fatal("missing 'fn' field in inputs")
			}
			fn, ok := fnVal.(string)
			if !ok {
				t.Fatal("'fn' field must be a string")
			}

			var got float64
			switch fn {
			case "pdf":
				got = NormalPDF(x, mu, sigma)
			case "cdf":
				got = NormalCDF(x, mu, sigma)
			default:
				t.Fatalf("unknown fn: %s", fn)
			}
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

func TestGoldenBetaPDF(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/prob/beta_pdf.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			x := testutil.InputFloat64(t, tc, "x")
			alpha := testutil.InputFloat64(t, tc, "alpha")
			beta := testutil.InputFloat64(t, tc, "beta")
			got := BetaPDF(x, alpha, beta)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ---------------------------------------------------------------------------
// LogGamma / Erfc wrappers
// ---------------------------------------------------------------------------

func TestLogGamma(t *testing.T) {
	// lgamma(1) = 0, lgamma(2) = 0, lgamma(5) = ln(24)
	tests := []struct {
		x    float64
		want float64
		tol  float64
	}{
		{1, 0, 1e-15},
		{2, 0, 1e-15},
		{5, math.Log(24), 1e-12},
		{0.5, math.Log(math.Pi) / 2, 1e-12}, // lgamma(0.5) = ln(sqrt(pi))
	}
	for _, tt := range tests {
		got := LogGamma(tt.x)
		if math.Abs(got-tt.want) > tt.tol {
			t.Errorf("LogGamma(%v) = %v, want %v", tt.x, got, tt.want)
		}
	}
}

func TestErfc(t *testing.T) {
	// erfc(0) = 1, erfc(inf) = 0
	if Erfc(0) != 1 {
		t.Errorf("Erfc(0) = %v, want 1", Erfc(0))
	}
	if Erfc(math.Inf(1)) != 0 {
		t.Errorf("Erfc(+Inf) = %v, want 0", Erfc(math.Inf(1)))
	}
	// erfc(1) ~ 0.15729920705
	if math.Abs(Erfc(1)-0.15729920705028513) > 1e-14 {
		t.Errorf("Erfc(1) = %v, want ~0.15730", Erfc(1))
	}
}
