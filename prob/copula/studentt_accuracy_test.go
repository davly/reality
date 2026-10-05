package copula

import (
	"math"
	"testing"

	"github.com/davly/reality/prob"
)

// studentTGrid holds Student-t quantiles x with P(T <= x) = p for df degrees of
// freedom, computed independently of this repository with mpmath at 60 digits
// (the tail probability of the exact float64 p is inverted by bisection on the
// regularized incomplete beta function until the CDF at x reproduces it to
// 1e-40 relative; df = 1 and df = 2 are cross-checked against their closed
// forms tan(pi (p - 1/2)) and (2p - 1) / sqrt(2 p (1 - p))).
var studentTGrid = []struct {
	df, p, x float64
}{
	{1.0, 1e-12, -318309886183.79065},
	{1.0, 1e-06, -318309.8861827435},
	{1.0, 0.01, -31.820515953773956},
	{1.0, 0.1, -3.077683537175253},
	{1.0, 0.5, 0.0},
	{1.0, 0.9, 3.077683537175254},
	{1.0, 0.99, 31.82051595377393},
	{1.0, 0.999999, 318309.88617359026},
	{2.0, 1e-12, -707106.7811854868},
	{2.0, 1e-06, -707.1057205259339},
	{2.0, 0.01, -6.964556734283274},
	{2.0, 0.1, -1.8856180831641267},
	{2.0, 0.5, 0.0},
	{2.0, 0.9, 1.885618083164127},
	{2.0, 0.99, 6.964556734283271},
	{2.0, 0.999999, 707.1057205157671},
	{3.0, 1e-12, -10331.108244292485},
	{3.0, 1e-06, -103.29946778041935},
	{3.0, 0.01, -4.5407028585681335},
	{3.0, 0.1, -1.63774435369621},
	{3.0, 0.5, 0.0},
	{3.0, 0.9, 1.6377443536962104},
	{3.0, 0.99, 4.540702858568132},
	{3.0, 0.999999, 103.29946777942897},
	{5.0, 1e-12, -393.95695957760375},
	{5.0, 1e-06, -24.771029720515944},
	{5.0, 0.01, -3.3649299989072188},
	{5.0, 0.1, -1.475884048824481},
	{5.0, 0.5, 0.0},
	{5.0, 0.9, 1.4758840488244813},
	{5.0, 0.99, 3.364929998907218},
	{5.0, 0.999999, 24.77102972037249},
	{10.0, 1e-12, -40.5320961786626},
	{10.0, 1e-06, -9.75199549094058},
	{10.0, 0.01, -2.763769458112696},
	{10.0, 0.1, -1.3721836411103356},
	{10.0, 0.5, 0.0},
	{10.0, 0.9, 1.3721836411103359},
	{10.0, 0.99, 2.7637694581126957},
	{10.0, 0.999999, 9.751995490909854},
	{30.0, 1e-12, -11.397217523311411},
	{30.0, 1e-06, -5.871117120418883},
	{30.0, 0.01, -2.4572615424005915},
	{30.0, 0.1, -1.3104150253913955},
	{30.0, 0.5, 0.0},
	{30.0, 0.9, 1.3104150253913958},
	{30.0, 0.99, 2.457261542400591},
	{30.0, 0.999999, 5.871117120408623},
	{100.0, 1e-12, -8.025825594493254},
	{100.0, 1e-06, -5.048830877228346},
	{100.0, 0.01, -2.364217366238482},
	{100.0, 0.1, -1.290074761346516},
	{100.0, 0.5, 0.0},
	{100.0, 0.9, 1.290074761346516},
	{100.0, 0.99, 2.3642173662384818},
	{100.0, 0.999999, 5.048830877221447},
	{10000.0, 1e-12, -7.0433716020557755},
	{10000.0, 1e-06, -4.756229685056779},
	{10000.0, 0.01, -2.3267208386694755},
	{10000.0, 0.1, -1.2816362297304775},
	{10000.0, 0.5, 0.0},
	{10000.0, 0.9, 1.2816362297304777},
	{10000.0, 0.99, 2.3267208386694755},
	{10000.0, 0.999999, 4.756229685050958},
}

// relErr is the relative error of got against want; absolute where want is 0.
func relErr(got, want float64) float64 {
	if math.IsNaN(got) {
		return math.Inf(1)
	}
	if want == 0 {
		return math.Abs(got)
	}
	return math.Abs(got-want) / math.Abs(want)
}

// StudentTQuantile in this package is deprecated in favour of
// prob.StudentTQuantile on the grounds that the latter is at least as accurate.
// This pins that claim on the measured grid: df in {1, 2, 3, 5, 10, 30, 100,
// 1e4} by p in {1e-12, 1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1 - 1e-6}. Measured, the
// prob version is more accurate at all 64 points: relative error 5e-16 to 8e-13
// for 0.01 <= p <= 0.99, at most 8.4e-11 at p = 1e-6 and at most 3.3e-5 at
// p = 1e-12 (this package: 2e-11 to 5e-9, up to 1.1e-5 and up to 0.44), and
// exactly 0 at p = 0.5.
func TestStudentTQuantile_ProbIsAtLeastAsAccurate(t *testing.T) {
	const probBound = 1e-4 // worst measured 3.3e-5 (df = 1, p = 1e-12)
	for _, c := range studentTGrid {
		eProb := relErr(prob.StudentTQuantile(c.p, c.df), c.x)
		eHere := relErr(StudentTQuantile(c.p, c.df), c.x)
		if eProb > eHere {
			t.Errorf("df=%v p=%v: prob.StudentTQuantile error %.3g exceeds this package's %.3g", c.df, c.p, eProb, eHere)
		}
		if eProb > probBound {
			t.Errorf("df=%v p=%v: prob.StudentTQuantile error %.3g exceeds %g", c.df, c.p, eProb, probBound)
		}
	}
}
