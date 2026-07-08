package agreement

import "math/big"

// McNemarExact returns the EXACT two-sided p-value of McNemar's test for
// marginal homogeneity in paired binary data, given only the two DISCORDANT
// cell counts of the 2x2 paired table:
//
//	                subject 2 = 1   subject 2 = 0
//	subject 1 = 1        a               b
//	subject 1 = 0        c               d
//
// b = number of pairs where subject 1 scored 1 and subject 2 scored 0, and
// c = number of pairs where subject 1 scored 0 and subject 2 scored 1. Only
// the discordant pairs (b, c) carry information about a difference in the two
// marginal rates; the concordant cells a and d are irrelevant and are not
// arguments. Use DiscordantCounts to derive (b, c) from raw paired 0/1
// slices.
//
// # The exact conditional test
//
// Condition on the number of discordant pairs n = b + c. Under the null
// hypothesis of marginal homogeneity, each discordant pair is independently
// and equally likely to fall in cell b or cell c, so b is distributed
// Binomial(n, 1/2). The exact two-sided p-value doubles the smaller tail of
// that symmetric binomial and caps the result at 1:
//
//	p = min(1, 2 * sum_{i=0}^{m} C(n,i) / 2^n),   m = min(b, c)
//
// The whole probability is a dyadic rational, so this implementation sums the
// binomial coefficients with math/big integers and forms the tail as an exact
// big.Rat, converting to float64 only once at the end. The result is
// therefore correct to within a single float64 rounding (<= 0.5 ulp) for any
// b, c that fit in an int — there is no cumulative floating-point error and no
// large-n underflow of 2^-n, unlike a naive PMF summation.
//
// If b == c the two tails are equal and each is >= 1/2, so 2 * tail >= 1 and
// the p-value is exactly 1. If b + c == 0 there are no discordant pairs and
// hence no evidence against marginal homogeneity, so the p-value is 1 by
// convention. A negative count returns ErrNegativeCount.
//
// Golden vector — Fagerland, Lydersen & Laake (2013), the airway
// hyper-responsiveness matched-pairs data (Bentur et al. 2009), discordant
// counts (b, c) = (1, 7), n = 8:
//
//	tail = C(8,0) + C(8,1) = 1 + 8 = 9
//	p    = 2 * 9 / 2^8 = 18 / 256 = 0.0703125
//
// which is the exact conditional p = 0.070 reported there.
//
// References:
//   - McNemar, Q. (1947). Note on the sampling error of the difference
//     between correlated proportions or percentages. Psychometrika 12(2):
//     153-157.
//   - Fagerland, M.W., Lydersen, S., Laake, P. (2013). The McNemar test for
//     binary matched-pairs data: mid-p and asymptotic are better than exact
//     conditional. BMC Medical Research Methodology 13:91.
func McNemarExact(b, c int) (float64, error) {
	if b < 0 || c < 0 {
		return 0, ErrNegativeCount
	}
	n := b + c
	if n == 0 {
		return 1, nil
	}
	m := b
	if c < b {
		m = c
	}
	// tail = sum_{i=0}^{m} C(n,i), as an exact big integer.
	tail := new(big.Int)
	for i := 0; i <= m; i++ {
		tail.Add(tail, binomialBig(n, i))
	}
	// p = 2 * tail / 2^n, exact rational, capped at 1.
	num := new(big.Int).Lsh(tail, 1)                // 2 * tail
	den := new(big.Int).Lsh(big.NewInt(1), uint(n)) // 2^n
	p := new(big.Rat).SetFrac(num, den)
	if p.Cmp(big.NewRat(1, 1)) > 0 {
		return 1, nil
	}
	f, _ := p.Float64()
	return f, nil
}

// McNemarMidP returns the two-sided MID-P value of McNemar's test on the
// discordant counts (b, c). The mid-p correction counts only HALF of the
// probability mass at the observed boundary point, which removes the
// conservatism of the exact conditional test (whose actual type-I error rate
// sits well below the nominal level) while remaining valid:
//
//	p_mid = p_exact - C(n,m) / 2^n,   n = b + c,  m = min(b, c)
//
// i.e. two-sided exact minus one whole point mass at the observed tail point
// (removing half of it from each of the two symmetric tails). The result is
// clamped to [0, 1]. Fagerland, Lydersen & Laake (2013) recommend the mid-p
// test over both the exact conditional and the asymptotic chi-square tests
// for binary matched-pairs data.
//
// Golden vector — same Fagerland et al. (2013) data, (b, c) = (1, 7), n = 8:
//
//	p_exact = 18/256 = 0.0703125
//	C(8,1)/2^8 = 8/256 = 0.03125
//	p_mid = 0.0703125 - 0.03125 = 0.0390625
//
// which is the mid-p = 0.039 reported there.
//
// Reference: Fagerland, M.W., Lydersen, S., Laake, P. (2013). BMC Medical
// Research Methodology 13:91.
func McNemarMidP(b, c int) (float64, error) {
	if b < 0 || c < 0 {
		return 0, ErrNegativeCount
	}
	n := b + c
	if n == 0 {
		return 1, nil
	}
	exact, err := McNemarExact(b, c)
	if err != nil {
		return 0, err
	}
	m := b
	if c < b {
		m = c
	}
	// point = C(n,m) / 2^n, exact rational.
	den := new(big.Int).Lsh(big.NewInt(1), uint(n))
	point, _ := new(big.Rat).SetFrac(binomialBig(n, m), den).Float64()
	mid := exact - point
	if mid < 0 {
		return 0, nil
	}
	if mid > 1 {
		return 1, nil
	}
	return mid, nil
}

// DiscordantCounts extracts McNemar's discordant cell counts (b, c) from two
// positionally-paired binary rating slices. Each entry must be 0 or 1 (e.g.
// incorrect/correct for a per-item eval, or absent/present for a matched
// pair). b counts pairs with x = 1, y = 0; c counts pairs with x = 0, y = 1.
// The concordant pairs (both 0 or both 1) are discarded because McNemar's
// test conditions them away.
//
// Returns ErrLengthMismatch if the slices differ in length, ErrEmptyInput if
// they are empty, and ErrNotBinary if any entry is neither 0 nor 1. The
// returned (b, c) feed directly into McNemarExact / McNemarMidP.
func DiscordantCounts(x, y []int) (b, c int, err error) {
	if len(x) != len(y) {
		return 0, 0, ErrLengthMismatch
	}
	if len(x) == 0 {
		return 0, 0, ErrEmptyInput
	}
	for i := range x {
		if (x[i] != 0 && x[i] != 1) || (y[i] != 0 && y[i] != 1) {
			return 0, 0, ErrNotBinary
		}
		switch {
		case x[i] == 1 && y[i] == 0:
			b++
		case x[i] == 0 && y[i] == 1:
			c++
		}
	}
	return b, c, nil
}

// binomialBig returns C(n, k) as an exact big.Int using the multiplicative
// recurrence C(n,k) = C(n,k-1) * (n-k+1) / k, which stays integral at every
// step. It exploits the symmetry C(n,k) = C(n,n-k) to minimise the number of
// multiplications. Zero-dependency: math/big is Go stdlib.
func binomialBig(n, k int) *big.Int {
	if k < 0 || k > n {
		return big.NewInt(0)
	}
	if k > n-k {
		k = n - k
	}
	result := big.NewInt(1)
	tmp := new(big.Int)
	for i := 0; i < k; i++ {
		result.Mul(result, tmp.SetInt64(int64(n-i)))
		result.Div(result, tmp.SetInt64(int64(i+1)))
	}
	return result
}
