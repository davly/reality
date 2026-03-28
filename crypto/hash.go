package crypto

// ---------------------------------------------------------------------------
// FNV-1a hash functions
// ---------------------------------------------------------------------------

// FNV offset basis and prime constants for 32-bit and 64-bit FNV-1a.
// Reference: Fowler-Noll-Vo hash, http://www.isthe.com/chongo/tech/comp/fnv/
const (
	fnv32OffsetBasis = uint32(2166136261)
	fnv32Prime       = uint32(16777619)
	fnv64OffsetBasis = uint64(14695981039346656037)
	fnv64Prime       = uint64(1099511628211)
)

// FNV1a32 computes the FNV-1a 32-bit hash of data. FNV-1a is a non-
// cryptographic hash function with excellent distribution for hash tables.
// The "1a" variant XORs before multiplying (better avalanche than FNV-1).
//
// Formula: for each byte b: hash = (hash XOR b) * FNV_prime
// Time complexity: O(len(data))
// Reference: Fowler, Noll, Vo (1991); IETF draft-eastlake-fnv-17
func FNV1a32(data []byte) uint32 {
	h := fnv32OffsetBasis
	for _, b := range data {
		h ^= uint32(b)
		h *= fnv32Prime
	}
	return h
}

// FNV1a64 computes the FNV-1a 64-bit hash of data. Same algorithm as
// FNV1a32 but with 64-bit offset basis and prime.
//
// Formula: for each byte b: hash = (hash XOR b) * FNV_prime
// Time complexity: O(len(data))
// Reference: Fowler, Noll, Vo (1991); IETF draft-eastlake-fnv-17
func FNV1a64(data []byte) uint64 {
	h := fnv64OffsetBasis
	for _, b := range data {
		h ^= uint64(b)
		h *= fnv64Prime
	}
	return h
}

// ---------------------------------------------------------------------------
// MurmurHash3
// ---------------------------------------------------------------------------

// MurmurHash3_32 computes the MurmurHash3 32-bit hash of data with the
// given seed. MurmurHash3 is a non-cryptographic hash function with
// excellent distribution, low collision rate, and high throughput.
//
// This implements the x86_32 variant from the reference C++ source.
//
// Time complexity: O(len(data))
// Reference: Austin Appleby (2011), https://github.com/aappleby/smhasher
func MurmurHash3_32(data []byte, seed uint32) uint32 {
	const (
		c1 = uint32(0xcc9e2d51)
		c2 = uint32(0x1b873593)
	)

	h := seed
	length := len(data)
	nblocks := length / 4

	// Body: process 4-byte blocks.
	for i := 0; i < nblocks; i++ {
		k := uint32(data[i*4]) |
			uint32(data[i*4+1])<<8 |
			uint32(data[i*4+2])<<16 |
			uint32(data[i*4+3])<<24

		k *= c1
		k = rotl32(k, 15)
		k *= c2

		h ^= k
		h = rotl32(h, 13)
		h = h*5 + 0xe6546b64
	}

	// Tail: handle remaining bytes.
	tail := data[nblocks*4:]
	var k1 uint32
	switch len(tail) {
	case 3:
		k1 ^= uint32(tail[2]) << 16
		fallthrough
	case 2:
		k1 ^= uint32(tail[1]) << 8
		fallthrough
	case 1:
		k1 ^= uint32(tail[0])
		k1 *= c1
		k1 = rotl32(k1, 15)
		k1 *= c2
		h ^= k1
	}

	// Finalization mix (fmix32).
	h ^= uint32(length)
	h = fmix32(h)

	return h
}

// ---------------------------------------------------------------------------
// Consistent hashing
// ---------------------------------------------------------------------------

// ConsistentHash implements Google's Jump Consistent Hash algorithm.
// Given a key and the number of buckets, it returns a bucket in [0, numBuckets).
//
// The algorithm has two important properties:
//   1. Uniform distribution: keys are evenly spread across buckets.
//   2. Monotonicity: when numBuckets increases, keys only move to the new
//      bucket (never between existing buckets).
//
// Returns 0 if numBuckets <= 0.
//
// Time complexity: O(ln(numBuckets))
// Reference: Lamping & Veach, "A Fast, Minimal Memory, Consistent Hash
// Algorithm" (Google, 2014)
func ConsistentHash(key uint64, numBuckets int) int {
	if numBuckets <= 0 {
		return 0
	}

	var b int64 = -1
	var j int64 = 0

	for j < int64(numBuckets) {
		b = j
		key = key*2862933555777941757 + 1
		j = int64(float64(b+1) * (float64(int64(1)<<31) / float64((key>>33)+1)))
	}

	return int(b)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// rotl32 performs a 32-bit left rotation.
func rotl32(x uint32, r uint8) uint32 {
	return (x << r) | (x >> (32 - r))
}

// fmix32 is the MurmurHash3 finalization mix for 32-bit values.
func fmix32(h uint32) uint32 {
	h ^= h >> 16
	h *= 0x85ebca6b
	h ^= h >> 13
	h *= 0xc2b2ae35
	h ^= h >> 16
	return h
}
