// Package combinatorics provides classical combinatorial functions:
// counting (factorial, binomial, Catalan, Fibonacci, derangements) and
// generation (permutations, combinations, lexicographic next, random subsets).
//
// All counting functions use float64 to support large values (exact up to
// about 170! for factorial). Generation functions produce concrete slices.
// Zero external dependencies.
package combinatorics

import (
	"math"
	"math/bits"
)

// ---------------------------------------------------------------------------
// Counting Functions
// ---------------------------------------------------------------------------

// Factorial returns n! as a float64: the float64 nearest to the exact integer
// n! for 0 <= n <= 170, and +Inf for n > 170 (171! exceeds math.MaxFloat64).
// Returns 1 for n <= 0 (by convention 0! = 1).
//
// Formula: n! = 1 * 2 * 3 * ... * n
// Valid range: n >= 0 (negative n returns 1)
// Precision: correctly rounded (round to nearest, ties to even) for every
// n <= 170, and exact for n <= 22, where n! is representable. The values come
// from a table of the exact integer products, so the result does not depend
// on the platform's floating-point evaluation.
// Reference: fundamental counting principle; Knuth, TAOCP vol. 1
func Factorial(n int) float64 {
	if n <= 0 {
		return 1
	}
	if n < len(factorialTable) {
		return factorialTable[n]
	}
	return math.Inf(1)
}

// factorialTable[n] is the float64 nearest to n! (ties to even), for
// 0 <= n <= 170. Each entry is the shortest decimal that parses to that
// float64; the tests re-derive all of them from exact math/big products.
var factorialTable = [171]float64{
	1,                       // 0!
	1,                       // 1!
	2,                       // 2!
	6,                       // 3!
	24,                      // 4!
	120,                     // 5!
	720,                     // 6!
	5040,                    // 7!
	40320,                   // 8!
	362880,                  // 9!
	3628800,                 // 10!
	39916800,                // 11!
	479001600,               // 12!
	6227020800,              // 13!
	87178291200,             // 14!
	1307674368000,           // 15!
	20922789888000,          // 16!
	355687428096000,         // 17!
	6402373705728000,        // 18!
	1.21645100408832e+17,    // 19!
	2.43290200817664e+18,    // 20!
	5.109094217170944e+19,   // 21!
	1.1240007277776077e+21,  // 22!
	2.585201673888498e+22,   // 23!
	6.204484017332394e+23,   // 24!
	1.5511210043330986e+25,  // 25!
	4.0329146112660565e+26,  // 26!
	1.0888869450418352e+28,  // 27!
	3.0488834461171387e+29,  // 28!
	8.841761993739702e+30,   // 29!
	2.6525285981219107e+32,  // 30!
	8.222838654177922e+33,   // 31!
	2.631308369336935e+35,   // 32!
	8.683317618811886e+36,   // 33!
	2.9523279903960416e+38,  // 34!
	1.0333147966386145e+40,  // 35!
	3.7199332678990125e+41,  // 36!
	1.3763753091226346e+43,  // 37!
	5.230226174666011e+44,   // 38!
	2.0397882081197444e+46,  // 39!
	8.159152832478977e+47,   // 40!
	3.345252661316381e+49,   // 41!
	1.40500611775288e+51,    // 42!
	6.041526306337383e+52,   // 43!
	2.658271574788449e+54,   // 44!
	1.1962222086548019e+56,  // 45!
	5.502622159812089e+57,   // 46!
	2.5862324151116818e+59,  // 47!
	1.2413915592536073e+61,  // 48!
	6.082818640342675e+62,   // 49!
	3.0414093201713376e+64,  // 50!
	1.5511187532873822e+66,  // 51!
	8.065817517094388e+67,   // 52!
	4.2748832840600255e+69,  // 53!
	2.308436973392414e+71,   // 54!
	1.2696403353658276e+73,  // 55!
	7.109985878048635e+74,   // 56!
	4.0526919504877214e+76,  // 57!
	2.3505613312828785e+78,  // 58!
	1.3868311854568984e+80,  // 59!
	8.32098711274139e+81,    // 60!
	5.075802138772248e+83,   // 61!
	3.146997326038794e+85,   // 62!
	1.98260831540444e+87,    // 63!
	1.2688693218588417e+89,  // 64!
	8.247650592082472e+90,   // 65!
	5.443449390774431e+92,   // 66!
	3.647111091818868e+94,   // 67!
	2.4800355424368305e+96,  // 68!
	1.711224524281413e+98,   // 69!
	1.1978571669969892e+100, // 70!
	8.504785885678623e+101,  // 71!
	6.1234458376886085e+103, // 72!
	4.4701154615126844e+105, // 73!
	3.307885441519386e+107,  // 74!
	2.48091408113954e+109,   // 75!
	1.8854947016660504e+111, // 76!
	1.4518309202828587e+113, // 77!
	1.1324281178206297e+115, // 78!
	8.946182130782976e+116,  // 79!
	7.156945704626381e+118,  // 80!
	5.797126020747368e+120,  // 81!
	4.753643337012842e+122,  // 82!
	3.945523969720659e+124,  // 83!
	3.314240134565353e+126,  // 84!
	2.81710411438055e+128,   // 85!
	2.4227095383672734e+130, // 86!
	2.107757298379528e+132,  // 87!
	1.8548264225739844e+134, // 88!
	1.650795516090846e+136,  // 89!
	1.4857159644817615e+138, // 90!
	1.352001527678403e+140,  // 91!
	1.2438414054641308e+142, // 92!
	1.1567725070816416e+144, // 93!
	1.087366156656743e+146,  // 94!
	1.032997848823906e+148,  // 95!
	9.916779348709496e+149,  // 96!
	9.619275968248212e+151,  // 97!
	9.426890448883248e+153,  // 98!
	9.332621544394415e+155,  // 99!
	9.332621544394415e+157,  // 100!
	9.42594775983836e+159,   // 101!
	9.614466715035127e+161,  // 102!
	9.90290071648618e+163,   // 103!
	1.0299016745145628e+166, // 104!
	1.081396758240291e+168,  // 105!
	1.1462805637347084e+170, // 106!
	1.226520203196138e+172,  // 107!
	1.324641819451829e+174,  // 108!
	1.4438595832024937e+176, // 109!
	1.588245541522743e+178,  // 110!
	1.7629525510902446e+180, // 111!
	1.974506857221074e+182,  // 112!
	2.2311927486598138e+184, // 113!
	2.5435597334721877e+186, // 114!
	2.925093693493016e+188,  // 115!
	3.393108684451898e+190,  // 116!
	3.969937160808721e+192,  // 117!
	4.684525849754291e+194,  // 118!
	5.574585761207606e+196,  // 119!
	6.689502913449127e+198,  // 120!
	8.094298525273444e+200,  // 121!
	9.875044200833601e+202,  // 122!
	1.214630436702533e+205,  // 123!
	1.506141741511141e+207,  // 124!
	1.882677176888926e+209,  // 125!
	2.372173242880047e+211,  // 126!
	3.0126600184576594e+213, // 127!
	3.856204823625804e+215,  // 128!
	4.974504222477287e+217,  // 129!
	6.466855489220474e+219,  // 130!
	8.47158069087882e+221,   // 131!
	1.1182486511960043e+224, // 132!
	1.4872707060906857e+226, // 133!
	1.9929427461615188e+228, // 134!
	2.6904727073180504e+230, // 135!
	3.659042881952549e+232,  // 136!
	5.012888748274992e+234,  // 137!
	6.917786472619489e+236,  // 138!
	9.615723196941089e+238,  // 139!
	1.3462012475717526e+241, // 140!
	1.898143759076171e+243,  // 141!
	2.695364137888163e+245,  // 142!
	3.854370717180073e+247,  // 143!
	5.5502938327393044e+249, // 144!
	8.047926057471992e+251,  // 145!
	1.1749972043909107e+254, // 146!
	1.727245890454639e+256,  // 147!
	2.5563239178728654e+258, // 148!
	3.80892263763057e+260,   // 149!
	5.713383956445855e+262,  // 150!
	8.62720977423324e+264,   // 151!
	1.3113358856834524e+267, // 152!
	2.0063439050956823e+269, // 153!
	3.0897696138473508e+271, // 154!
	4.789142901463394e+273,  // 155!
	7.471062926282894e+275,  // 156!
	1.1729568794264145e+278, // 157!
	1.853271869493735e+280,  // 158!
	2.9467022724950384e+282, // 159!
	4.7147236359920616e+284, // 160!
	7.590705053947219e+286,  // 161!
	1.2296942187394494e+289, // 162!
	2.0044015765453026e+291, // 163!
	3.287218585534296e+293,  // 164!
	5.423910666131589e+295,  // 165!
	9.003691705778438e+297,  // 166!
	1.503616514864999e+300,  // 167!
	2.5260757449731984e+302, // 168!
	4.269068009004705e+304,  // 169!
	7.257415615307999e+306,  // 170!
}

// BinomialCoeff returns C(n,k) = n! / (k! * (n-k)!), the number of k-element
// subsets of an n-element set. Returns 0 if k < 0 or k > n.
//
// Formula: C(n,k) = prod_{i=1..k} (n-k+i) / i with k replaced by min(k, n-k),
// evaluated in exact integer arithmetic: every partial product C(n-k+i, i)
// is an integer, so each division is exact.
// Valid range: 0 <= k <= n
// Precision: correctly rounded: the result is the float64 nearest to the
// exact integer C(n,k) (ties to even), and therefore exact whenever C(n,k) is
// representable, in particular whenever C(n,k) <= 2^53. Returns +Inf when the
// exact value rounds beyond math.MaxFloat64 (C(1029,514) is finite,
// C(1030,515) is +Inf).
// Cost: O(min(k, n-k)) word operations on a fixed-size integer, without
// allocation. The running product never decreases and at least doubles with
// each factor, so the work stops after at most 1025 factors once it passes
// 2^1024, where every result is +Inf.
// Reference: Knuth, TAOCP vol. 1, Section 1.2.6.
func BinomialCoeff(n, k int) float64 {
	if n < 0 || k < 0 || k > n {
		return 0
	}
	nn, kk := uint64(n), uint64(k)
	// Symmetry: C(n,k) = C(n, n-k). Use the smaller k for fewer terms.
	kk = min(kk, nn-kk)
	var x exactUint
	x.setOne()
	for i := uint64(1); i <= kk; {
		// Group consecutive factors while both products fit in 64 bits:
		// C(n-k+j, j) = C(n-k+i-1, i-1) * prod_{t=i..j} (n-k+t) / prod_{t=i..j} t
		// is an integer for every j, so one exact division covers a group.
		num, den := nn-kk+i, i
		for i++; i <= kk; i++ {
			hn, ln := bits.Mul64(num, nn-kk+i)
			hd, ld := bits.Mul64(den, i)
			if hn != 0 || hd != 0 {
				break
			}
			num, den = ln, ld
		}
		x.mulWord(num)
		x.divExact(den)
		if x.bitLen() > 1024 {
			return math.Inf(1)
		}
	}
	return x.nearestFloat64()
}

// Permutations returns P(n,k) = n! / (n-k)!, the number of k-permutations
// of n elements. Returns 0 if k < 0 or k > n.
//
// Formula: n * (n-1) * ... * (n-k+1), evaluated in exact integer arithmetic
// Valid range: 0 <= k <= n
// Precision: correctly rounded: the result is the float64 nearest to the
// exact integer P(n,k) (ties to even), and therefore exact whenever P(n,k) is
// representable. Returns +Inf when the exact value rounds beyond
// math.MaxFloat64; P(n,n) equals Factorial(n).
// Cost: O(k) word operations on a fixed-size integer, without allocation;
// the first j factors multiply to at least j!, so the work stops within 171
// factors once the product passes 2^1024, where every result is +Inf.
// Reference: fundamental counting principle
func Permutations(n, k int) float64 {
	if n < 0 || k < 0 || k > n {
		return 0
	}
	nn, kk := uint64(n), uint64(k)
	var x exactUint
	x.setOne()
	for i := uint64(0); i < kk; {
		// Group consecutive factors while their product fits in 64 bits.
		f := nn - i
		for i++; i < kk; i++ {
			hi, lo := bits.Mul64(f, nn-i)
			if hi != 0 {
				break
			}
			f = lo
		}
		x.mulWord(f)
		if x.bitLen() > 1024 {
			return math.Inf(1)
		}
	}
	return x.nearestFloat64()
}

// exactUint is a positive integer of up to 17 little-endian 64-bit words:
// any value below 2^1024 multiplied by one 64-bit factor fits. BinomialCoeff
// and Permutations use it to form exact results without allocating.
type exactUint struct {
	w [17]uint64
	n int // words in use; w[n-1] != 0
}

func (x *exactUint) setOne() {
	x.w[0], x.n = 1, 1
}

func (x *exactUint) bitLen() int {
	return (x.n-1)*64 + bits.Len64(x.w[x.n-1])
}

// mulWord sets x = x * m for m >= 1. The caller keeps x below 2^1024
// beforehand, so the product fits in 17 words.
func (x *exactUint) mulWord(m uint64) {
	var carry uint64
	for i := 0; i < x.n; i++ {
		hi, lo := bits.Mul64(x.w[i], m)
		var c uint64
		x.w[i], c = bits.Add64(lo, carry, 0)
		carry = hi + c // cannot overflow: hi <= 2^64 - 2
	}
	if carry != 0 {
		x.w[x.n] = carry
		x.n++
	}
}

// divExact sets x = x / d for a d >= 1 that divides x exactly. The power of
// two in d is removed by a shift and the odd part by Hensel (exact) division:
// multiplying by the inverse of the odd part modulo 2^64, from the least
// significant word up, with no hardware division (Jebelean, 1993).
func (x *exactUint) divExact(d uint64) {
	if s := uint(bits.TrailingZeros64(d)); s > 0 {
		for i := 0; i < x.n-1; i++ {
			x.w[i] = x.w[i]>>s | x.w[i+1]<<(64-s)
		}
		x.w[x.n-1] >>= s
		d >>= s
	}
	if d > 1 {
		// Newton's iteration for the inverse modulo 2^64. inv = d is correct
		// to 3 bits (d*d = 1 mod 8 for odd d); each step doubles that.
		inv := d
		for i := 0; i < 5; i++ {
			inv *= 2 - d*inv
		}
		var borrow uint64
		for i := 0; i < x.n; i++ {
			s, b := bits.Sub64(x.w[i], borrow, 0)
			q := s * inv // q*d = s (mod 2^64)
			x.w[i] = q
			hi, _ := bits.Mul64(q, d)
			borrow = hi + b // hi <= d-1, so this cannot overflow
		}
	}
	for x.n > 1 && x.w[x.n-1] == 0 {
		x.n--
	}
}

// nearestFloat64 returns the float64 nearest to x (ties to even), or +Inf
// when x rounds beyond math.MaxFloat64. The top 64 bits are converted with
// the bits below them folded into the lowest bit (a sticky bit), which
// leaves the rounding decision of the conversion exactly as for x itself;
// the scaling by a power of two is then exact.
func (x *exactUint) nearestFloat64() float64 {
	if x.n == 1 {
		return float64(x.w[0]) // correctly rounded conversion
	}
	top := x.w[x.n-1]
	lz := uint(bits.LeadingZeros64(top))
	m := top<<lz | x.w[x.n-2]>>(64-lz) // the 64 most significant bits
	sticky := x.w[x.n-2]<<lz != 0
	for i := 0; i < x.n-2 && !sticky; i++ {
		sticky = x.w[i] != 0
	}
	if sticky {
		m |= 1
	}
	return math.Ldexp(float64(m), (x.n-1)*64-int(lz))
}

// CatalanNumber returns the nth Catalan number C_n = C(2n,n) / (n+1).
// Returns 1 for n <= 0 (C_0 = 1).
//
// Formula: C_n = C(2n, n) / (n + 1) = (2n)! / ((n+1)! * n!)
// Valid range: n >= 0
// Precision: exact for small n; float64 rounding for large n
// Reference: Stanley, R.P. "Catalan Numbers" (2015)
func CatalanNumber(n int) float64 {
	if n <= 0 {
		return 1
	}
	return BinomialCoeff(2*n, n) / float64(n+1)
}

// FibonacciNumber returns the nth Fibonacci number F_n using matrix
// exponentiation in O(log n) time. F_0 = 0, F_1 = 1, F_n = F_{n-1} + F_{n-2}.
// Returns 0 for n <= 0.
//
// Method: matrix exponentiation of [[1,1],[1,0]]^n.
// Valid range: n >= 0; exact for n <= 93 (F_93 = 12200160415121876738,
// which is the largest Fibonacci number fitting in uint64).
// Precision: exact (integer arithmetic, no floating point)
// Reference: Knuth, TAOCP vol. 1, Section 1.2.8
func FibonacciNumber(n int) uint64 {
	if n <= 0 {
		return 0
	}
	if n == 1 || n == 2 {
		return 1
	}

	// Matrix [[a,b],[c,d]] represents [[F_{k+1}, F_k], [F_k, F_{k-1}]].
	// We exponentiate [[1,1],[1,0]] by n using repeated squaring.
	var (
		// Result matrix (identity)
		ra, rb, rc, rd uint64 = 1, 0, 0, 1
		// Base matrix
		ba, bb, bc, bd uint64 = 1, 1, 1, 0
	)

	exp := n
	for exp > 0 {
		if exp%2 == 1 {
			// result = result * base
			na := ra*ba + rb*bc
			nb := ra*bb + rb*bd
			nc := rc*ba + rd*bc
			nd := rc*bb + rd*bd
			ra, rb, rc, rd = na, nb, nc, nd
		}
		// base = base * base
		na := ba*ba + bb*bc
		nb := ba*bb + bb*bd
		nc := bc*ba + bd*bc
		nd := bc*bb + bd*bd
		ba, bb, bc, bd = na, nb, nc, nd
		exp /= 2
	}

	return rb // F_n is in position (0,1) of the result matrix
}

// StirlingFirst returns the (unsigned) Stirling number of the first kind,
// |s(n, k)|, which counts the number of permutations of n elements with
// exactly k disjoint cycles.
//
// Recurrence: |s(n, k)| = (n-1) * |s(n-1, k)| + |s(n-1, k-1)|
// Base cases: |s(0, 0)| = 1, |s(n, 0)| = 0 for n > 0, |s(0, k)| = 0 for k > 0
// Valid range: n >= 0, k >= 0; returns 0 if k > n or k < 0 or n < 0
// Precision: exact for small n (float64 mantissa limits for large n)
// Reference: Knuth, TAOCP vol. 1, Section 1.2.6; Graham, Knuth & Patashnik,
// "Concrete Mathematics", Chapter 6
func StirlingFirst(n, k int) float64 {
	if k < 0 || n < 0 || k > n {
		return 0
	}
	if n == 0 && k == 0 {
		return 1
	}
	if n == 0 || k == 0 {
		return 0
	}
	// Use iterative DP to avoid stack depth issues.
	// We only need the previous row.
	prev := make([]float64, k+1)
	prev[0] = 1 // |s(0, 0)| = 1

	for i := 1; i <= n; i++ {
		curr := make([]float64, k+1)
		for j := 1; j <= k && j <= i; j++ {
			curr[j] = float64(i-1)*prev[j] + prev[j-1]
		}
		prev = curr
	}
	return prev[k]
}

// StirlingSecond returns the Stirling number of the second kind, S(n, k),
// which counts the number of ways to partition a set of n elements into
// exactly k non-empty subsets.
//
// Recurrence: S(n, k) = k * S(n-1, k) + S(n-1, k-1)
// Base cases: S(0, 0) = 1, S(n, 0) = 0 for n > 0, S(0, k) = 0 for k > 0
// Valid range: n >= 0, k >= 0; returns 0 if k > n or k < 0 or n < 0
// Precision: exact for small n (float64 mantissa limits for large n)
// Reference: Knuth, TAOCP vol. 1, Section 1.2.6; Stanley, "Enumerative
// Combinatorics", Vol. 1
func StirlingSecond(n, k int) float64 {
	if k < 0 || n < 0 || k > n {
		return 0
	}
	if n == 0 && k == 0 {
		return 1
	}
	if n == 0 || k == 0 {
		return 0
	}
	// Iterative DP.
	prev := make([]float64, k+1)
	prev[0] = 1 // S(0, 0) = 1

	for i := 1; i <= n; i++ {
		curr := make([]float64, k+1)
		for j := 1; j <= k && j <= i; j++ {
			curr[j] = float64(j)*prev[j] + prev[j-1]
		}
		prev = curr
	}
	return prev[k]
}

// BellNumber returns B_n, the nth Bell number, which counts the total
// number of ways to partition a set of n elements into non-empty subsets.
//
// B_n = sum_{k=0}^{n} S(n, k), where S(n, k) are Stirling numbers of
// the second kind.
//
// Equivalently, computed via the Bell triangle for efficiency:
//
//	B[0] = 1
//	B[i][0] = B[i-1][i-1]    (wrap from end of previous row)
//	B[i][j] = B[i][j-1] + B[i-1][j-1]
//
// Valid range: n >= 0; returns 1 for n <= 0 (B_0 = 1)
// Precision: exact for small n (float64 mantissa limits for large n)
// Reference: Bell, E.T. (1934) "Exponential Numbers"; Rota, G.-C. (1964)
// "The Number of Partitions of a Set"
func BellNumber(n int) float64 {
	if n <= 0 {
		return 1
	}
	// Bell triangle computation.
	// Row i has i+1 entries. We only need two rows at a time.
	prev := []float64{1} // Row 0: [1]

	for i := 1; i <= n; i++ {
		curr := make([]float64, i+1)
		curr[0] = prev[len(prev)-1] // wrap from end of previous row
		for j := 1; j <= i; j++ {
			curr[j] = curr[j-1] + prev[j-1]
		}
		prev = curr
	}
	return prev[0] // B_n is the first element of row n (which is wrapped from end)
}

// IntegerPartitions returns the number of ways to write n as a sum of
// positive integers, disregarding order. For example, 4 = 4 = 3+1 = 2+2
// = 2+1+1 = 1+1+1+1, so IntegerPartitions(4) = 5.
//
// Uses dynamic programming with the recurrence:
//
//	p(n, k) = p(n, k-1) + p(n-k, k)
//
// where p(n, k) is the number of partitions of n using parts of size at most k.
//
// Valid range: n >= 0; returns 1 for n == 0 (empty partition), 0 for n < 0
// Precision: exact for moderate n (float64 mantissa limits for large n)
// Time complexity: O(n^2).
// Reference: Hardy, G.H. & Ramanujan, S. (1918) "Asymptotic Formulae in
// Combinatory Analysis"; Andrews, G.E. (1976) "The Theory of Partitions"
func IntegerPartitions(n int) float64 {
	if n < 0 {
		return 0
	}
	if n == 0 {
		return 1
	}

	// dp[j] = number of partitions of j using parts 1..i
	dp := make([]float64, n+1)
	dp[0] = 1

	for i := 1; i <= n; i++ {
		for j := i; j <= n; j++ {
			dp[j] += dp[j-i]
		}
	}
	return dp[n]
}

// DerangementCount returns !n, the number of derangements (permutations with
// no fixed points) of n elements. Returns 1 for n == 0, 0 for n == 1.
//
// Formula: !n = n! * sum_{k=0}^{n} (-1)^k / k!
// Equivalently: !n = round(n! / e) for n >= 1
// Valid range: n >= 0
// Precision: exact via rounding for moderate n; float64 limits for very large n
// Reference: Euler (1751); see also Knuth TAOCP vol. 1
func DerangementCount(n int) float64 {
	if n <= 0 {
		return 1
	}
	if n == 1 {
		return 0
	}
	// !n = round(n! / e) for n >= 1
	return math.Round(Factorial(n) / math.E)
}
