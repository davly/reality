package crypto

// ---------------------------------------------------------------------------
// Mersenne Twister (MT19937-64)
// ---------------------------------------------------------------------------

// MersenneTwister implements the 64-bit Mersenne Twister PRNG (MT19937-64).
// It produces a deterministic sequence of pseudo-random uint64 values for a
// given seed. The period is 2^19937 - 1.
//
// This is a faithful reimplementation of the reference C code by Takuji
// Nishimura and Makoto Matsumoto.
//
// Reference: Matsumoto & Nishimura, "Mersenne Twister: A 623-Dimensionally
// Equidistributed Uniform Pseudo-Random Number Generator" (1998).
type MersenneTwister struct {
	mt    [312]uint64
	index int
}

// Mersenne Twister 64-bit constants (from the reference implementation).
const (
	mtN         = 312
	mtM         = 156
	mtMatrixA   = uint64(0xB5026F5AA96619E9)
	mtUpperMask = uint64(0xFFFFFFFF80000000) // most significant 33 bits
	mtLowerMask = uint64(0x7FFFFFFF)         // least significant 31 bits
)

// NewMersenneTwister creates a new MT19937-64 generator seeded with the
// given value. The same seed always produces the same sequence.
func NewMersenneTwister(seed uint64) *MersenneTwister {
	mt := &MersenneTwister{index: mtN}
	mt.mt[0] = seed
	for i := 1; i < mtN; i++ {
		mt.mt[i] = 6364136223846793005*(mt.mt[i-1]^(mt.mt[i-1]>>62)) + uint64(i)
	}
	return mt
}

// Uint64 generates the next pseudo-random uint64 in the sequence.
func (mt *MersenneTwister) Uint64() uint64 {
	if mt.index >= mtN {
		mt.twist()
	}

	y := mt.mt[mt.index]
	mt.index++

	// Tempering.
	y ^= (y >> 29) & 0x5555555555555555
	y ^= (y << 17) & 0x71D67FFFEDA60000
	y ^= (y << 37) & 0xFFF7EEE000000000
	y ^= y >> 43

	return y
}

// Float64 returns a pseudo-random float64 in [0, 1) by dividing Uint64
// output by 2^64.
func (mt *MersenneTwister) Float64() float64 {
	return float64(mt.Uint64()>>11) / float64(uint64(1)<<53)
}

// twist generates the next mtN values in the state array.
func (mt *MersenneTwister) twist() {
	mag01 := [2]uint64{0, mtMatrixA}

	for i := 0; i < mtN-mtM; i++ {
		y := (mt.mt[i] & mtUpperMask) | (mt.mt[i+1] & mtLowerMask)
		mt.mt[i] = mt.mt[i+mtM] ^ (y >> 1) ^ mag01[y&1]
	}
	for i := mtN - mtM; i < mtN-1; i++ {
		y := (mt.mt[i] & mtUpperMask) | (mt.mt[i+1] & mtLowerMask)
		mt.mt[i] = mt.mt[i+(mtM-mtN)] ^ (y >> 1) ^ mag01[y&1]
	}
	y := (mt.mt[mtN-1] & mtUpperMask) | (mt.mt[0] & mtLowerMask)
	mt.mt[mtN-1] = mt.mt[mtM-1] ^ (y >> 1) ^ mag01[y&1]

	mt.index = 0
}

// ---------------------------------------------------------------------------
// PCG (Permuted Congruential Generator)
// ---------------------------------------------------------------------------

// PCG implements the PCG-XSH-RR variant — a 64-bit state / 32-bit output
// PRNG with excellent statistical quality and small state. Fully
// deterministic for a given (seed, seq) pair.
//
// Reference: Melissa O'Neill, "PCG: A Family of Simple Fast Space-Efficient
// Statistically Good Algorithms for Random Number Generation" (2014).
type PCG struct {
	state uint64
	inc   uint64
}

// NewPCG creates a new PCG-XSH-RR generator. The seed determines the
// initial state; seq selects which of 2^63 independent sequences to use.
// The same (seed, seq) pair always produces the same sequence.
func NewPCG(seed, seq uint64) *PCG {
	p := &PCG{
		state: 0,
		inc:   (seq << 1) | 1, // inc must be odd
	}
	// Advance state once with seed mixed in.
	p.state += seed
	p.Uint32() // warm up: advance past initial state
	return p
}

// Uint32 generates the next pseudo-random uint32 in the sequence.
func (p *PCG) Uint32() uint32 {
	oldState := p.state
	// Advance internal state (LCG step).
	p.state = oldState*6364136223846793005 + p.inc

	// XSH-RR output function.
	xorShifted := uint32(((oldState >> 18) ^ oldState) >> 27)
	rot := uint32(oldState >> 59)
	return (xorShifted >> rot) | (xorShifted << ((-rot) & 31))
}

// Float64 returns a pseudo-random float64 in [0, 1).
func (p *PCG) Float64() float64 {
	return float64(p.Uint32()) / float64(uint64(1)<<32)
}

// ---------------------------------------------------------------------------
// xoshiro256** (xoshiro256starstar)
// ---------------------------------------------------------------------------

// Xoshiro256 implements the xoshiro256** PRNG — a 256-bit state / 64-bit
// output generator with excellent speed and statistical properties. Fully
// deterministic for a given seed.
//
// Reference: Blackman & Vigna, "Scrambled Linear Pseudorandom Number
// Generators" (2021). https://prng.di.unimi.it/
type Xoshiro256 struct {
	s [4]uint64
}

// NewXoshiro256 creates a new xoshiro256** generator. The seed is expanded
// into a 256-bit state using SplitMix64 (as recommended by the authors).
// The same seed always produces the same sequence.
func NewXoshiro256(seed uint64) *Xoshiro256 {
	x := &Xoshiro256{}
	// Use SplitMix64 to expand the seed into 4 state words.
	x.s[0] = splitmix64(&seed)
	x.s[1] = splitmix64(&seed)
	x.s[2] = splitmix64(&seed)
	x.s[3] = splitmix64(&seed)
	return x
}

// Uint64 generates the next pseudo-random uint64 in the sequence.
func (x *Xoshiro256) Uint64() uint64 {
	result := rotl64(x.s[1]*5, 7) * 9

	t := x.s[1] << 17

	x.s[2] ^= x.s[0]
	x.s[3] ^= x.s[1]
	x.s[1] ^= x.s[2]
	x.s[0] ^= x.s[3]

	x.s[2] ^= t
	x.s[3] = rotl64(x.s[3], 45)

	return result
}

// Float64 returns a pseudo-random float64 in [0, 1).
func (x *Xoshiro256) Float64() float64 {
	return float64(x.Uint64()>>11) / float64(uint64(1)<<53)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// splitmix64 is a simple 64-bit PRNG used to seed other generators.
// It modifies the state in-place and returns the output value.
// Reference: Vigna, "An experimental exploration of Marsaglia's xorshift
// generators, scrambled" (2017).
func splitmix64(state *uint64) uint64 {
	*state += 0x9E3779B97F4A7C15
	z := *state
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// rotl64 performs a 64-bit left rotation.
func rotl64(x uint64, k uint) uint64 {
	return (x << k) | (x >> (64 - k))
}
