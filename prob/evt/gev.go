package evt

import "math"

// eulerMascheroni is Euler's constant gamma, used in the Gumbel (xi -> 0)
// limit of the GEV mean and of the L-moment estimator.
const eulerMascheroni = 0.5772156649015328606065120900824024310421

// GEVParams holds the three parameters of the Generalized Extreme Value
// distribution in the von Mises / Coles (2001) parameterisation:
//
//	Mu    - location
//	Sigma - scale (> 0)
//	Xi    - shape (tail index).  Xi > 0 Frechet (heavy), Xi == 0 Gumbel
//	        (light), Xi < 0 Weibull (bounded upper tail).
type GEVParams struct {
	Mu    float64
	Sigma float64
	Xi    float64
}

// GEVKind names the three limiting types of the extremal-types theorem
// (Fisher-Tippett-Gnedenko).
type GEVKind int

const (
	Gumbel  GEVKind = iota // Xi == 0    (light tail, unbounded)
	Frechet                // Xi > 0     (heavy tail, unbounded)
	Weibull                // Xi < 0     (short tail, upper bounded)
)

// Kind classifies the GEV family from the shape parameter, with a dead band
// of half-width tol around zero mapped to Gumbel.  tol <= 0 is treated as an
// exact Xi == 0 test.
func (p GEVParams) Kind(tol float64) GEVKind {
	if tol < 0 {
		tol = 0
	}
	switch {
	case p.Xi > tol:
		return Frechet
	case p.Xi < -tol:
		return Weibull
	default:
		return Gumbel
	}
}

// gevReduced returns t = 1 + Xi*(x-Mu)/Sigma, the argument of the GEV
// generator, together with the standardised deviate z = (x-Mu)/Sigma.
func (p GEVParams) gevReduced(x float64) (t, z float64) {
	z = (x - p.Mu) / p.Sigma
	t = 1 + p.Xi*z
	return t, z
}

// GEVCDF is the GEV cumulative distribution function
//
//	F(x) = exp{ -[1 + xi (x-mu)/sigma]^(-1/xi) },   1 + xi (x-mu)/sigma > 0
//
// with the Gumbel limit (xi == 0) handled explicitly:
//
//	F(x) = exp{ -exp(-(x-mu)/sigma) }.
//
// Outside the support the CDF is 0 (lower tail, xi > 0) or 1 (upper tail,
// xi < 0), the correct limits of the exp(-t^{-1/xi}) form.
//
// Reference: Coles (2001) eq. (3.2); Jenkinson (1955).
// Precision: exact to machine eps away from the support boundary; the xi != 0
// branch uses log1p/expm1 to stay stable as xi -> 0.
func GEVCDF(x float64, p GEVParams) float64 {
	if p.Sigma <= 0 || math.IsNaN(p.Sigma) {
		return math.NaN()
	}
	if p.Xi == 0 {
		z := (x - p.Mu) / p.Sigma
		return math.Exp(-math.Exp(-z))
	}
	t, _ := p.gevReduced(x)
	if t <= 0 {
		if p.Xi > 0 {
			return 0
		}
		return 1
	}
	// t^{-1/xi} = exp( (-1/xi) * log t ); stable near xi = 0.
	return math.Exp(-math.Exp((-1 / p.Xi) * math.Log(t)))
}

// GEVPDF is the GEV probability density function
//
//	f(x) = (1/sigma) t^{-1-1/xi} exp(-t^{-1/xi}),   t = 1 + xi (x-mu)/sigma > 0
//
// with the Gumbel limit (xi == 0):
//
//	f(x) = (1/sigma) exp(-z) exp(-exp(-z)),   z = (x-mu)/sigma.
//
// Density is 0 outside the support.
//
// Reference: Coles (2001) eq. (3.3).
// Precision: exact to machine eps in the interior of the support.
func GEVPDF(x float64, p GEVParams) float64 {
	if p.Sigma <= 0 || math.IsNaN(p.Sigma) {
		return math.NaN()
	}
	if p.Xi == 0 {
		z := (x - p.Mu) / p.Sigma
		ez := math.Exp(-z)
		return ez * math.Exp(-ez) / p.Sigma
	}
	t, _ := p.gevReduced(x)
	if t <= 0 {
		return 0
	}
	logT := math.Log(t)
	power := -1 / p.Xi
	tPow := math.Exp(power * logT) // t^{-1/xi}
	// f = (1/sigma) exp( (power-1) log t - t^{-1/xi} )
	return math.Exp((power-1)*logT-tPow) / p.Sigma
}

// GEVQuantile is the inverse CDF (return-level map): the value x with
// F(x) = pr for pr in (0,1),
//
//	x = mu + (sigma/xi) [ (-ln pr)^{-xi} - 1 ],   xi != 0
//	x = mu - sigma ln(-ln pr),                    xi == 0 (Gumbel).
//
// The xi != 0 branch uses expm1 so it degrades smoothly to the Gumbel form
// as xi -> 0.
//
// Reference: Coles (2001) eq. (3.4).
// Precision: exact to machine eps for pr in (0,1); returns +/-Inf at the
// endpoints per the distribution's support.
func GEVQuantile(pr float64, p GEVParams) float64 {
	if p.Sigma <= 0 || math.IsNaN(p.Sigma) || math.IsNaN(pr) || pr < 0 || pr > 1 {
		return math.NaN()
	}
	if pr == 0 { // lower endpoint of the support
		if p.Xi > 0 {
			return p.Mu - p.Sigma/p.Xi
		}
		return math.Inf(-1)
	}
	if pr == 1 { // upper endpoint of the support
		if p.Xi < 0 {
			return p.Mu - p.Sigma/p.Xi
		}
		return math.Inf(1)
	}
	y := -math.Log(pr) // y > 0
	if p.Xi == 0 {
		return p.Mu - p.Sigma*math.Log(y)
	}
	// (sigma/xi)( y^{-xi} - 1 ) = (sigma/xi) expm1(-xi ln y)
	return p.Mu + (p.Sigma/p.Xi)*math.Expm1(-p.Xi*math.Log(y))
}

// GEVReturnLevel is the block-maxima return level: the level exceeded on
// average once every T blocks, i.e. the quantile at probability 1 - 1/T.
//
//	z_T = GEVQuantile(1 - 1/T, p).
//
// T must be > 1.  Reference: Coles (2001) eq. (3.4) / §3.3.3.
func GEVReturnLevel(T float64, p GEVParams) float64 {
	if T <= 1 {
		return math.NaN()
	}
	return GEVQuantile(1-1/T, p)
}
