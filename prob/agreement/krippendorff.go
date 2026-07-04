package agreement

import (
	"math"
	"sort"
)

// Metric selects Krippendorff's alpha difference function: how far apart
// two category values c and k are taken to be for the purpose of weighting
// disagreement.
//
// Reference: Krippendorff, K. (2011). Computing Krippendorff's
// Alpha-Reliability, §D.
type Metric int

const (
	// Nominal: categories either match or they don't.
	// delta^2(c,k) = 0 if c == k, else 1.
	Nominal Metric = iota
	// Ordinal: distance depends on how many category ranks lie between c
	// and k, weighted by how often those ranks were actually used — not on
	// the raw numeric gap between the values.
	// delta^2(c,k) = (sum_{g=c}^{k} n_g - (n_c+n_k)/2)^2, summing over
	// every distinct category rank from c to k inclusive, where n_g is
	// that rank's total pairable-value count (its coincidence-matrix
	// marginal).
	Ordinal
	// Interval: distance is the raw numeric gap.
	// delta^2(c,k) = (c-k)^2.
	Interval
)

// KrippendorffAlpha returns Krippendorff's alpha reliability coefficient
// for m >= 2 raters coding N units, natively tolerating missing data:
// data[r][u] is rater r's value for unit u, or math.NaN() if rater r did
// not code unit u. A unit coded by fewer than 2 raters contributes
// nothing — per Krippendorff's definition, a lone value cannot be paired
// with anything and is excluded even from the marginals, not merely from
// the pairing.
//
//	Do    = (1/n)       * sum_ck o_ck         * delta^2(c,k)
//	De    = (1/(n*(n-1))) * sum_ck n_c*n_k    * delta^2(c,k)
//	alpha = 1 - Do/De
//
// where o_ck is the observed coincidence count for the (c,k) value pair
// across all countable units, n_c is category c's marginal pairable count,
// and n is the total number of pairable values (n_c summed over
// categories).
//
// alpha = 1 is perfect reliability, alpha = 0 is chance-level agreement,
// and alpha < 0 indicates systematic disagreement.
//
// Returns ErrTooFewRaters if len(data) < 2 or every unit has fewer than 2
// non-NaN raters, ErrEmptyInput if data has zero units, ErrRaggedRows if
// raters' rows have differing lengths, and ErrDegenerateChanceAgreement if
// the chance-expected disagreement De is 0.
//
// # Golden vectors
//
// Krippendorff, K. (2011). Computing Krippendorff's Alpha-Reliability —
// the 4-observer/12-unit worked example with missing data (the same
// dataset appears in Hayes & Krippendorff (2007) and as the R
// `irr::kripp.alpha` reference example; "." denotes a missing rating):
//
//	Units:        1 2 3 4 5 6 7 8 9 10 11 12
//	Observer A:   1 2 3 3 2 1 4 1 2  .  .  .
//	Observer B:   1 2 3 3 2 2 4 1 2  5  .  3
//	Observer C:   . 3 3 3 2 3 4 2 2  5  1  .
//	Observer D:   1 2 3 3 2 4 4 1 2  5  1  .
//
//	Nominal:  alpha = 113/152       ~= 0.743421  (published: 0.743)
//	Ordinal:  alpha = 108577/133160 ~= 0.815385  (published: 0.815)
//	Interval: alpha = 951/1120      ~= 0.849107  (published: 0.849)
//
// (unit 12 has only one coder and is excluded, leaving 11 countable units
// and 40 pairable values — exactly as the source reports.)
//
// The same tutorial's smaller, no-missing-data examples are also golden
// vectors (see krippendorff_test.go): 2-observer nominal, 5 categories,
// alpha = 155/224 ~= 0.691964 (published: 0.692); 2-observer binary
// nominal, alpha = 2/21 ~= 0.095238 (published: 0.095).
//
// Reference: Krippendorff, K. (2011). Computing Krippendorff's
// Alpha-Reliability.
// https://www.asc.upenn.edu/sites/default/files/2021-03/Computing%20Krippendorff's%20Alpha-Reliability.pdf
func KrippendorffAlpha(data [][]float64, metric Metric) (float64, error) {
	m := len(data)
	if m < 2 {
		return 0, ErrTooFewRaters
	}
	nUnits := len(data[0])
	if nUnits == 0 {
		return 0, ErrEmptyInput
	}
	for _, row := range data {
		if len(row) != nUnits {
			return 0, ErrRaggedRows
		}
	}

	// Coincidence matrix, keyed by distinct observed category values.
	coincidence := make(map[float64]map[float64]float64)
	add := func(c, k, amount float64) {
		if coincidence[c] == nil {
			coincidence[c] = make(map[float64]float64)
		}
		coincidence[c][k] += amount
	}

	for u := 0; u < nUnits; u++ {
		var vals []float64
		for r := 0; r < m; r++ {
			v := data[r][u]
			if !math.IsNaN(v) {
				vals = append(vals, v)
			}
		}
		mu := len(vals)
		if mu < 2 {
			continue // lone or unrated unit: excluded even from marginals
		}
		denom := float64(mu - 1)
		for p := 0; p < mu; p++ {
			for q := 0; q < mu; q++ {
				if p == q {
					continue
				}
				add(vals[p], vals[q], 1/denom)
			}
		}
	}

	if len(coincidence) == 0 {
		return 0, ErrTooFewRaters
	}

	categories := make([]float64, 0, len(coincidence))
	for c := range coincidence {
		categories = append(categories, c)
	}
	sort.Float64s(categories)

	marginal := make(map[float64]float64, len(categories))
	var n float64
	for _, c := range categories {
		var total float64
		for _, k := range categories {
			total += coincidence[c][k]
		}
		marginal[c] = total
		n += total
	}
	if n < 2 {
		return 0, ErrTooFewRaters
	}

	delta2 := deltaSquared(metric, categories, marginal)

	var do, de float64
	for _, c := range categories {
		for _, k := range categories {
			d2 := delta2(c, k)
			if d2 == 0 {
				continue
			}
			do += coincidence[c][k] * d2
			de += marginal[c] * marginal[k] * d2
		}
	}
	do /= n
	de /= n * (n - 1)

	if de == 0 {
		return 0, ErrDegenerateChanceAgreement
	}
	return 1 - do/de, nil
}

// deltaSquared returns the pairwise squared-distance function for the
// given metric, closing over the sorted category list and marginal counts
// that Ordinal needs.
func deltaSquared(metric Metric, categories []float64, marginal map[float64]float64) func(c, k float64) float64 {
	switch metric {
	case Interval:
		return func(c, k float64) float64 {
			d := c - k
			return d * d
		}
	case Ordinal:
		return func(c, k float64) float64 {
			if c == k {
				return 0
			}
			lo, hi := c, k
			if lo > hi {
				lo, hi = hi, lo
			}
			var sum float64
			for _, g := range categories {
				if g >= lo && g <= hi {
					sum += marginal[g]
				}
			}
			d := sum - (marginal[c]+marginal[k])/2
			return d * d
		}
	default: // Nominal
		return func(c, k float64) float64 {
			if c == k {
				return 0
			}
			return 1
		}
	}
}
