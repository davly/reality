package compression

// RunLengthEncode compresses data using run-length encoding. Consecutive
// identical bytes are stored as (count, value) pairs. The count byte
// represents the number of repetitions (1-255). Runs longer than 255 are
// split into multiple pairs.
//
// Format: [count1, value1, count2, value2, ...]
// Valid range: any byte slice (including empty)
// Output: encoded byte slice; length <= 2*len(data), often much shorter
// Precision: exact (lossless)
// Reference: Golomb (1966); widely used in BMP, TIFF, PCX formats
func RunLengthEncode(data []byte) []byte {
	if len(data) == 0 {
		return nil
	}
	// Pre-allocate conservatively. Worst case is 2x (no runs).
	encoded := make([]byte, 0, len(data))
	i := 0
	for i < len(data) {
		val := data[i]
		count := 1
		for i+count < len(data) && data[i+count] == val && count < 255 {
			count++
		}
		encoded = append(encoded, byte(count), val)
		i += count
	}
	return encoded
}

// RunLengthDecode decompresses run-length encoded data back to the original
// byte sequence. The input must be a valid RLE stream of (count, value) pairs.
//
// Format: input is [count1, value1, count2, value2, ...]
// Valid range: len(encoded) must be even; each count >= 1
// Output: original byte slice
// Precision: exact (lossless)
// Failure mode: returns nil if encoded is nil or has odd length
func RunLengthDecode(encoded []byte) []byte {
	if len(encoded) == 0 {
		return nil
	}
	if len(encoded)%2 != 0 {
		return nil // Invalid: must be (count, value) pairs.
	}
	// Compute total output length first to allocate once.
	total := 0
	for i := 0; i < len(encoded); i += 2 {
		total += int(encoded[i])
	}
	decoded := make([]byte, 0, total)
	for i := 0; i < len(encoded); i += 2 {
		count := int(encoded[i])
		val := encoded[i+1]
		for j := 0; j < count; j++ {
			decoded = append(decoded, val)
		}
	}
	return decoded
}

// DeltaEncode computes the first difference of a sequence. The first element
// is stored as-is; subsequent elements store the difference from the previous
// value. This is optimal for slowly-changing sequences (e.g., timestamps,
// monotonic counters) where deltas fit in fewer bits.
//
// Formula: out[0] = data[0]; out[i] = data[i] - data[i-1] for i > 0
// Valid range: any int64 slice (including empty)
// Output: same length as input
// Precision: exact (lossless, assuming no int64 overflow in differences)
// Reference: standard delta coding; used in Gorilla time series compression
func DeltaEncode(data []int64) []int64 {
	if len(data) == 0 {
		return nil
	}
	encoded := make([]int64, len(data))
	encoded[0] = data[0]
	for i := 1; i < len(data); i++ {
		encoded[i] = data[i] - data[i-1]
	}
	return encoded
}

// DeltaDecode reconstructs the original sequence from delta-encoded data
// by computing the running prefix sum.
//
// Formula: out[0] = encoded[0]; out[i] = out[i-1] + encoded[i] for i > 0
// Valid range: any int64 slice (including empty)
// Output: original sequence before delta encoding
// Precision: exact (lossless, assuming no int64 overflow)
// Reference: inverse of DeltaEncode
func DeltaDecode(encoded []int64) []int64 {
	if len(encoded) == 0 {
		return nil
	}
	decoded := make([]int64, len(encoded))
	decoded[0] = encoded[0]
	for i := 1; i < len(encoded); i++ {
		decoded[i] = decoded[i-1] + encoded[i]
	}
	return decoded
}
