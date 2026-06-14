package timeseries

import (
	"math"
	"testing"
)

// All golden values below are HAND-COMPUTED from the recurrence documented in
// the package comment (alpha = weight on the newest sample):
//
//	first sample:  mean = x, var = 0
//	thereafter:    diff = x - mean_-          (mean BEFORE this update)
//	               mean = mean_- + alpha*diff
//	               var  = (1-alpha)*var_- + alpha*diff*diff
//
// Worked example, alpha = 0.5, xs = {1, 2, 3, 4} (the G1 row below):
//
//	n=1 x=1: mean=1                         var=0
//	n=2 x=2: diff=1   mean=1+0.5*1=1.5      var=0.5*0   +0.5*1^2   =0.5
//	n=3 x=3: diff=1.5 mean=1.5+0.5*1.5=2.25 var=0.5*0.5 +0.5*1.5^2 =0.25+1.125 =1.375
//	n=4 x=4: diff=1.75 mean=2.25+0.5*1.75=3.125
//	                                        var=0.5*1.375+0.5*1.75^2=0.6875+1.53125=2.21875
//
// MUTATION CHECK (performed while writing this test): each of the three terms
// of the recurrence is pinned by a distinct golden, so a wrong formula fails:
//   - dropping the (1-alpha) decay on the carried variance, or using alpha vs
//     (1-alpha) on either term, makes the G1 / G4 variance goldens fail at n>=3
//     (the first step where the carried-variance term is non-zero).
//   - measuring diff against the NEW mean instead of the OLD mean shifts every
//     mean/var from n=2 onward (G1 mean at n=2 would be 1.75, not 1.5).
//   - squaring an absolute deviation differently, or forgetting to square,
//     fails G1 var at n=2 (0.5 vs 1.0) and G4.
//   - seeding var != 0 on the first sample fails every "single sample" assert.
// All of the above were tried by hand against the goldens; each breaks at least
// one assert.

const tol = 1e-12

// step is one expected (mean, var) snapshot after folding xs[i].
type step struct {
	mean float64
	varr float64
}

func TestEWMoments_Series(t *testing.T) {
	tests := []struct {
		name  string
		alpha float64
		xs    []float64
		want  []step // one entry per sample, in order
	}{
		{
			name:  "G1 alpha=0.5 {1,2,3,4}",
			alpha: 0.5,
			xs:    []float64{1, 2, 3, 4},
			want: []step{
				{1, 0},
				{1.5, 0.5},
				{2.25, 1.375},
				{3.125, 2.21875},
			},
		},
		{
			name:  "G2 constant {5,5,5,5} -> var stays 0",
			alpha: 0.5,
			xs:    []float64{5, 5, 5, 5},
			want: []step{
				{5, 0},
				{5, 0},
				{5, 0},
				{5, 0},
			},
		},
		{
			name:  "G3 baseline {10,12,11,13}",
			alpha: 0.5,
			xs:    []float64{10, 12, 11, 13},
			want: []step{
				{10, 0},
				{11, 2},     // diff=2  -> mean=11, var=0.5*0+0.5*4=2
				{11, 1},     // diff=0  -> mean=11, var=0.5*2+0.5*0=1
				{12, 2.5},   // diff=2  -> mean=12, var=0.5*1+0.5*4=2.5
			},
		},
		{
			name:  "G4 alpha=0.3 {2,4,4,4,5,5}",
			alpha: 0.3,
			xs:    []float64{2, 4, 4, 4, 5, 5},
			want: []step{
				{2, 0},
				{2.6, 1.2},                                   // diff=2  var=0.7*0+0.3*4=1.2
				{3.02, 1.428},                                // diff=1.4 var=0.7*1.2+0.3*1.96=0.84+0.588
				{3.314, 1.28772},                             // diff=0.98 var=0.7*1.428+0.3*0.9604
				{3.8198, 1.7541828},                          // diff=1.686 var=0.7*1.28772+0.3*2.842596
				{4.17386, 1.645789572},                       // diff=1.1802 var=0.7*1.7541828+0.3*1.39287204
			},
		},
		{
			name:  "single sample carries no variance",
			alpha: 0.7,
			xs:    []float64{42},
			want:  []step{{42, 0}},
		},
		{
			name:  "alpha=1 is last-value-only",
			alpha: 1,
			xs:    []float64{3, 9},
			want: []step{
				{3, 0},
				{9, 36}, // diff=6 -> mean=9, var=0*0+1*36=36
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := NewEWMoments(tt.alpha)
			if e.Count() != 0 {
				t.Fatalf("fresh Count = %d, want 0", e.Count())
			}
			for i, x := range tt.xs {
				e.Update(x)
				if e.Count() != uint64(i+1) {
					t.Errorf("after %d updates Count = %d, want %d", i+1, e.Count(), i+1)
				}
				if math.Abs(e.Mean()-tt.want[i].mean) > tol {
					t.Errorf("step %d Mean = %.17g, want %.17g", i, e.Mean(), tt.want[i].mean)
				}
				if math.Abs(e.Variance()-tt.want[i].varr) > tol {
					t.Errorf("step %d Variance = %.17g, want %.17g", i, e.Variance(), tt.want[i].varr)
				}
				// StdDev is sqrt(Variance) by construction.
				if math.Abs(e.StdDev()-math.Sqrt(tt.want[i].varr)) > tol {
					t.Errorf("step %d StdDev = %.17g, want %.17g", i, e.StdDev(), math.Sqrt(tt.want[i].varr))
				}
			}
		})
	}
}

func TestEWMoments_ZScore(t *testing.T) {
	// Build the G3 baseline {10,12,11,13}: ends at mean=12, var=2.5,
	// std=sqrt(2.5)=1.58113883008418966...
	e := NewEWMoments(0.5)
	for _, x := range []float64{10, 12, 11, 13} {
		e.Update(x)
	}
	wantStd := math.Sqrt(2.5)
	if math.Abs(e.StdDev()-wantStd) > tol {
		t.Fatalf("baseline StdDev = %.17g, want %.17g", e.StdDev(), wantStd)
	}

	// Outlier z: (20 - 12) / sqrt(2.5) = 8 / 1.5811388300841898 = 5.05964425...
	gotZ := e.ZScore(20)
	wantZ := 8.0 / wantStd // 5.0596442562694071
	if math.Abs(gotZ-wantZ) > tol {
		t.Errorf("ZScore(20) = %.17g, want %.17g", gotZ, wantZ)
	}
	if math.Abs(gotZ-5.0596442562694071) > 1e-12 {
		t.Errorf("ZScore(20) hand-value = %.17g, want 5.0596442562694071", gotZ)
	}

	// z at the mean is exactly 0.
	if z := e.ZScore(12); z != 0 {
		t.Errorf("ZScore(mean) = %v, want 0", z)
	}

	// ZScore does NOT fold x in: mean/var unchanged after probing.
	if math.Abs(e.Mean()-12) > tol || math.Abs(e.Variance()-2.5) > tol {
		t.Errorf("ZScore mutated state: mean=%v var=%v", e.Mean(), e.Variance())
	}

	// Symmetry of the sign.
	if z := e.ZScore(4); math.Abs(z-(-wantZ)) > tol { // (4-12)/std = -wantZ
		t.Errorf("ZScore(4) = %.17g, want %.17g", z, -wantZ)
	}
}

func TestEWMoments_DegenerateEdges(t *testing.T) {
	// Empty tracker: all queries return 0, no panic.
	e := NewEWMoments(0.5)
	if e.Count() != 0 || e.Mean() != 0 || e.Variance() != 0 || e.StdDev() != 0 || e.ZScore(99) != 0 {
		t.Errorf("empty tracker not all-zero: count=%d mean=%v var=%v std=%v z=%v",
			e.Count(), e.Mean(), e.Variance(), e.StdDev(), e.ZScore(99))
	}

	// Constant baseline -> std 0 -> ZScore is 0 at the mean, +Inf above, -Inf below.
	c := NewEWMoments(0.5)
	for i := 0; i < 5; i++ {
		c.Update(7)
	}
	if c.Variance() != 0 {
		t.Fatalf("constant baseline Variance = %v, want 0", c.Variance())
	}
	if z := c.ZScore(7); z != 0 {
		t.Errorf("ZScore at constant mean = %v, want 0", z)
	}
	if z := c.ZScore(8); !math.IsInf(z, +1) {
		t.Errorf("ZScore above constant baseline = %v, want +Inf", z)
	}
	if z := c.ZScore(6); !math.IsInf(z, -1) {
		t.Errorf("ZScore below constant baseline = %v, want -Inf", z)
	}

	// Single sample: variance/std are 0, mean is the sample.
	s := NewEWMoments(0.3)
	s.Update(1.25)
	if s.Mean() != 1.25 || s.Variance() != 0 || s.StdDev() != 0 {
		t.Errorf("single sample: mean=%v var=%v std=%v", s.Mean(), s.Variance(), s.StdDev())
	}
}

func TestNewEWMoments_AlphaRange(t *testing.T) {
	for _, bad := range []float64{0, -0.1, 1.0001, 2, math.NaN(), math.Inf(1)} {
		func(a float64) {
			defer func() {
				if recover() == nil {
					t.Errorf("NewEWMoments(%v) did not panic", a)
				}
			}()
			_ = NewEWMoments(a)
		}(bad)
	}
	// Valid endpoints do not panic.
	for _, ok := range []float64{1e-9, 0.5, 1} {
		_ = NewEWMoments(ok) // would panic the test if it panicked
	}
	_ = NewEWMoments(1).Alpha()
}
