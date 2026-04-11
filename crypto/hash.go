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
// Structural hashing — FNV-1a with observation shape fingerprint
// ---------------------------------------------------------------------------

// SituationHashWithStructure computes a situation hash that incorporates
// both the observation content and its structural fingerprint. The
// structural component captures the "shape" of the observation: how many
// fields are present, their types, and their ordering.
//
// This is a cross-pollination innovation from the blind build: the layout
// hash distinguishes observations that have the same data but different
// structure (e.g., flat vs nested, sparse vs dense). Two observations with
// identical content but different shapes will produce different hashes.
//
// Algorithm:
//  1. Hash the content via FNV-1a 64-bit.
//  2. Hash the structural descriptor via FNV-1a 64-bit.
//  3. Combine using XOR-fold: (contentHash * fnv64Prime) ^ structHash
//
// The XOR-fold preserves FNV-1a's avalanche properties while mixing the
// two independent hash channels.
//
// Time complexity: O(len(content) + len(structure))
// Reference: FNV-1a (Fowler, Noll, Vo); structural hashing from
// blind-build layout-hash innovation.
func SituationHashWithStructure(content []byte, structure []byte) uint64 {
	contentHash := FNV1a64(content)
	structHash := FNV1a64(structure)
	return (contentHash * fnv64Prime) ^ structHash
}

// StructuralDescriptor builds a byte descriptor for a flat key-value
// observation. The descriptor encodes the count of fields and the length
// of each key, producing a compact structural fingerprint.
//
// For example, an observation with keys ["name", "age", "score"] produces
// a descriptor like [3, 4, 3, 5] (count=3, key lengths 4, 3, 5).
//
// This is intentionally simple: it captures shape without leaking content.
// More sophisticated structural descriptors (nested, typed) can be built
// on top by callers.
func StructuralDescriptor(keys []string) []byte {
	desc := make([]byte, 0, 1+len(keys))
	// Encode field count (capped at 255).
	count := len(keys)
	if count > 255 {
		count = 255
	}
	desc = append(desc, byte(count))
	// Encode each key length (capped at 255).
	for i := 0; i < count; i++ {
		kl := len(keys[i])
		if kl > 255 {
			kl = 255
		}
		desc = append(desc, byte(kl))
	}
	return desc
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
