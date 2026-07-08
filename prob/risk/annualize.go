package risk

import "math"

// TradingDaysPerYear is the conventional number of trading days in a calendar
// year for US/UK equity markets, and the constant this package uses to scale
// daily statistics to an annual horizon. It is exported (rather than a magic
// literal buried in a formula) precisely because the "252" convention is a
// choice: some desks use 250 or 260, and a consumer that annualizes with a
// different calendar must be able to see and override the number rather than
// inherit an invisible one. It is the direct antidote to the "z-score with an
// invented standard deviation" class of hidden-constant defect.
const TradingDaysPerYear = 252

// Other common annualization bases, provided for the same
// convention-visibility reason. Pass whichever matches the sampling frequency
// of the returns being annualized.
const (
	MonthsPerYear   = 12
	WeeksPerYear    = 52
	QuartersPerYear = 4
)

// AnnualizeReturn scales a per-period MEAN simple return to an annual simple
// return by GEOMETRIC compounding:
//
//	annual = (1 + perPeriodMean)^periodsPerYear - 1
//
// This is the compounding-consistent convention: reinvesting a mean return of
// perPeriodMean each period for periodsPerYear periods multiplies capital by
// (1+perPeriodMean)^periodsPerYear. It is NOT the arithmetic approximation
// perPeriodMean*periodsPerYear, which overstates the annual figure whenever
// perPeriodMean != 0 (a caller who deliberately wants the arithmetic form
// should compute it explicitly, so the choice is visible).
//
// Valid range: periodsPerYear >= 1; 1+perPeriodMean must be > 0 (a per-period
// loss of -100% or worse wipes out capital and has no real geometric annual
// equivalent). Returns NaN otherwise.
// Precision: one math.Pow; ~15 significant digits.
func AnnualizeReturn(perPeriodMean float64, periodsPerYear int) float64 {
	if periodsPerYear < 1 {
		return math.NaN()
	}
	growth := 1.0 + perPeriodMean
	if growth <= 0 {
		return math.NaN()
	}
	return math.Pow(growth, float64(periodsPerYear)) - 1.0
}

// AnnualizeVolatility scales a per-period return standard deviation to an
// annual standard deviation by the square-root-of-time rule:
//
//	annual = perPeriodStdDev * sqrt(periodsPerYear)
//
// The square-root-of-time scaling assumes returns are serially UNCORRELATED
// and identically distributed across periods; under positive autocorrelation
// it understates and under mean reversion it overstates true annual
// volatility. That assumption is the convention this function pins — a
// consumer whose returns are autocorrelated should scale differently and
// knowingly.
//
// Valid range: periodsPerYear >= 1; perPeriodStdDev >= 0. Returns NaN
// otherwise.
// Precision: one math.Sqrt and one multiply; ~15 significant digits.
func AnnualizeVolatility(perPeriodStdDev float64, periodsPerYear int) float64 {
	if periodsPerYear < 1 || perPeriodStdDev < 0 || math.IsNaN(perPeriodStdDev) {
		return math.NaN()
	}
	return perPeriodStdDev * math.Sqrt(float64(periodsPerYear))
}
