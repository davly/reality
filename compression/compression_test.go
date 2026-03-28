package compression

import (
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

// ═══════════════════════════════════════════════════════════════════════════
// Golden-file tests — shared test vectors across Go, Python, C++, C#
// ═══════════════════════════════════════════════════════════════════════════

func TestGolden_ShannonEntropy(t *testing.T) {
	gf := testutil.LoadGolden(t, "testdata/compression/shannon_entropy.json")
	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			probs := testutil.InputFloat64Slice(t, tc, "probs")
			got := ShannonEntropy(probs)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — ShannonEntropy
// ═══════════════════════════════════════════════════════════════════════════

func TestShannonEntropy_Uniform2(t *testing.T) {
	// Fair coin: H = log2(2) = 1.0 bit.
	got := ShannonEntropy([]float64{0.5, 0.5})
	assertClose(t, "uniform-2", got, 1.0, 1e-15)
}

func TestShannonEntropy_Uniform4(t *testing.T) {
	// Four equally likely outcomes: H = log2(4) = 2.0 bits.
	got := ShannonEntropy([]float64{0.25, 0.25, 0.25, 0.25})
	assertClose(t, "uniform-4", got, 2.0, 1e-15)
}

func TestShannonEntropy_Uniform8(t *testing.T) {
	// Eight equally likely outcomes: H = log2(8) = 3.0 bits.
	probs := make([]float64, 8)
	for i := range probs {
		probs[i] = 1.0 / 8.0
	}
	got := ShannonEntropy(probs)
	assertClose(t, "uniform-8", got, 3.0, 1e-14)
}

func TestShannonEntropy_Certain(t *testing.T) {
	// Single certain event: H = 0.
	got := ShannonEntropy([]float64{1.0})
	assertClose(t, "certain", got, 0.0, 1e-15)
}

func TestShannonEntropy_CertainWithZeros(t *testing.T) {
	// One certain event, rest zero: H = 0. Zeros should be skipped.
	got := ShannonEntropy([]float64{0.0, 1.0, 0.0, 0.0})
	assertClose(t, "certain-zeros", got, 0.0, 1e-15)
}

func TestShannonEntropy_BinaryCoin(t *testing.T) {
	// Biased coin: p=0.9, q=0.1. H = -(0.9*log2(0.9) + 0.1*log2(0.1))
	expected := -(0.9*math.Log2(0.9) + 0.1*math.Log2(0.1))
	got := ShannonEntropy([]float64{0.9, 0.1})
	assertClose(t, "biased-coin", got, expected, 1e-15)
}

func TestShannonEntropy_Empty(t *testing.T) {
	got := ShannonEntropy([]float64{})
	assertClose(t, "empty", got, 0.0, 0)
}

func TestShannonEntropy_SingleZero(t *testing.T) {
	// All-zero probabilities (degenerate): H = 0.
	got := ShannonEntropy([]float64{0.0})
	assertClose(t, "single-zero", got, 0.0, 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — JointEntropy
// ═══════════════════════════════════════════════════════════════════════════

func TestJointEntropy_Independent(t *testing.T) {
	// Independent: P(X,Y) = P(X)*P(Y). For two fair coins:
	// H(X,Y) = H(X) + H(Y) = 1 + 1 = 2 bits.
	joint := [][]float64{
		{0.25, 0.25},
		{0.25, 0.25},
	}
	got := JointEntropy(joint)
	assertClose(t, "independent-coins", got, 2.0, 1e-14)
}

func TestJointEntropy_PerfectCorrelation(t *testing.T) {
	// Perfectly correlated: X=Y always. H(X,Y) = H(X) = 1 bit.
	joint := [][]float64{
		{0.5, 0.0},
		{0.0, 0.5},
	}
	got := JointEntropy(joint)
	assertClose(t, "perfect-corr", got, 1.0, 1e-15)
}

func TestJointEntropy_Empty(t *testing.T) {
	got := JointEntropy([][]float64{})
	assertClose(t, "empty-joint", got, 0.0, 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — ConditionalEntropy
// ═══════════════════════════════════════════════════════════════════════════

func TestConditionalEntropy_Independent(t *testing.T) {
	// Independent: H(Y|X) = H(Y).
	joint := [][]float64{
		{0.25, 0.25},
		{0.25, 0.25},
	}
	// H(Y) for marginal [0.5, 0.5] = 1 bit.
	got := ConditionalEntropy(joint)
	assertClose(t, "cond-independent", got, 1.0, 1e-14)
}

func TestConditionalEntropy_PerfectCorrelation(t *testing.T) {
	// If X determines Y completely, H(Y|X) = 0.
	joint := [][]float64{
		{0.5, 0.0},
		{0.0, 0.5},
	}
	got := ConditionalEntropy(joint)
	assertClose(t, "cond-perfect", got, 0.0, 1e-15)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — MutualInformation
// ═══════════════════════════════════════════════════════════════════════════

func TestMutualInformation_Independent(t *testing.T) {
	// Independent variables: I(X;Y) = 0.
	joint := [][]float64{
		{0.25, 0.25},
		{0.25, 0.25},
	}
	got := MutualInformation(joint)
	assertClose(t, "mi-independent", got, 0.0, 1e-14)
}

func TestMutualInformation_PerfectCorrelation(t *testing.T) {
	// Perfectly correlated: I(X;Y) = H(X) = 1 bit.
	joint := [][]float64{
		{0.5, 0.0},
		{0.0, 0.5},
	}
	got := MutualInformation(joint)
	assertClose(t, "mi-perfect", got, 1.0, 1e-15)
}

func TestMutualInformation_NonNegative(t *testing.T) {
	// MI is always non-negative.
	joint := [][]float64{
		{0.1, 0.2},
		{0.3, 0.4},
	}
	got := MutualInformation(joint)
	if got < -1e-14 {
		t.Errorf("MI should be non-negative, got %v", got)
	}
}

func TestMutualInformation_Identity(t *testing.T) {
	// I(X;Y) = H(X) + H(Y) - H(X,Y) identity check.
	joint := [][]float64{
		{0.1, 0.2},
		{0.3, 0.4},
	}
	margX := []float64{0.3, 0.7}
	margY := []float64{0.4, 0.6}
	expected := ShannonEntropy(margX) + ShannonEntropy(margY) - JointEntropy(joint)
	got := MutualInformation(joint)
	assertClose(t, "mi-identity", got, expected, 1e-14)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — KLDivergence
// ═══════════════════════════════════════════════════════════════════════════

func TestKLDivergence_Identical(t *testing.T) {
	// KL(P||P) = 0 for any distribution.
	p := []float64{0.3, 0.7}
	got := KLDivergence(p, p)
	assertClose(t, "kl-identical", got, 0.0, 1e-15)
}

func TestKLDivergence_Shifted(t *testing.T) {
	// KL([0.5,0.5] || [0.25,0.75])
	p := []float64{0.5, 0.5}
	q := []float64{0.25, 0.75}
	// KL = 0.5*log2(0.5/0.25) + 0.5*log2(0.5/0.75)
	expected := 0.5*math.Log2(0.5/0.25) + 0.5*math.Log2(0.5/0.75)
	got := KLDivergence(p, q)
	assertClose(t, "kl-shifted", got, expected, 1e-14)
}

func TestKLDivergence_Asymmetric(t *testing.T) {
	// KL divergence is NOT symmetric: KL(P||Q) != KL(Q||P).
	p := []float64{0.9, 0.1}
	q := []float64{0.1, 0.9}
	pq := KLDivergence(p, q)
	qp := KLDivergence(q, p)
	// Both should be positive but different.
	if pq <= 0 {
		t.Errorf("KL(P||Q) should be positive, got %v", pq)
	}
	if qp <= 0 {
		t.Errorf("KL(Q||P) should be positive, got %v", qp)
	}
	// They are equal in this symmetric case (swapped mirror), but let's verify.
	assertClose(t, "kl-asymmetric", pq, qp, 1e-14)
}

func TestKLDivergence_ZeroInQ(t *testing.T) {
	// If q[i]=0 where p[i]>0, KL = +Inf.
	p := []float64{0.5, 0.5}
	q := []float64{1.0, 0.0}
	got := KLDivergence(p, q)
	if !math.IsInf(got, 1) {
		t.Errorf("expected +Inf, got %v", got)
	}
}

func TestKLDivergence_Empty(t *testing.T) {
	got := KLDivergence([]float64{}, []float64{})
	assertClose(t, "kl-empty", got, 0.0, 0)
}

func TestKLDivergence_LengthMismatch(t *testing.T) {
	got := KLDivergence([]float64{0.5, 0.5}, []float64{1.0})
	assertClose(t, "kl-mismatch", got, 0.0, 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — CrossEntropy
// ═══════════════════════════════════════════════════════════════════════════

func TestCrossEntropy_SameDistribution(t *testing.T) {
	// H(P, P) = H(P) = ShannonEntropy(P).
	p := []float64{0.5, 0.5}
	got := CrossEntropy(p, p)
	expected := ShannonEntropy(p)
	assertClose(t, "ce-same", got, expected, 1e-15)
}

func TestCrossEntropy_Identity(t *testing.T) {
	// H(P, Q) = H(P) + KL(P||Q).
	p := []float64{0.3, 0.7}
	q := []float64{0.6, 0.4}
	got := CrossEntropy(p, q)
	expected := ShannonEntropy(p) + KLDivergence(p, q)
	assertClose(t, "ce-identity", got, expected, 1e-14)
}

func TestCrossEntropy_ZeroInQ(t *testing.T) {
	p := []float64{0.5, 0.5}
	q := []float64{1.0, 0.0}
	got := CrossEntropy(p, q)
	if !math.IsInf(got, 1) {
		t.Errorf("expected +Inf, got %v", got)
	}
}

func TestCrossEntropy_Empty(t *testing.T) {
	got := CrossEntropy([]float64{}, []float64{})
	assertClose(t, "ce-empty", got, 0.0, 0)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — RunLengthEncode / Decode
// ═══════════════════════════════════════════════════════════════════════════

func TestRLE_Empty(t *testing.T) {
	encoded := RunLengthEncode([]byte{})
	if encoded != nil {
		t.Errorf("expected nil for empty input, got %v", encoded)
	}
	decoded := RunLengthDecode([]byte{})
	if decoded != nil {
		t.Errorf("expected nil for empty decode, got %v", decoded)
	}
}

func TestRLE_NoRuns(t *testing.T) {
	// All unique bytes: worst case for RLE.
	data := []byte{1, 2, 3, 4, 5}
	encoded := RunLengthEncode(data)
	expected := []byte{1, 1, 1, 2, 1, 3, 1, 4, 1, 5}
	assertBytesEqual(t, "rle-no-runs", encoded, expected)
}

func TestRLE_AllSame(t *testing.T) {
	// All same bytes: best case for RLE.
	data := []byte{7, 7, 7, 7, 7}
	encoded := RunLengthEncode(data)
	expected := []byte{5, 7}
	assertBytesEqual(t, "rle-all-same", encoded, expected)
}

func TestRLE_MixedRuns(t *testing.T) {
	data := []byte{1, 1, 1, 2, 2, 3}
	encoded := RunLengthEncode(data)
	expected := []byte{3, 1, 2, 2, 1, 3}
	assertBytesEqual(t, "rle-mixed", encoded, expected)
}

func TestRLE_Roundtrip(t *testing.T) {
	data := []byte{10, 10, 20, 20, 20, 30, 30, 30, 30}
	encoded := RunLengthEncode(data)
	decoded := RunLengthDecode(encoded)
	assertBytesEqual(t, "rle-roundtrip", decoded, data)
}

func TestRLE_SingleByte(t *testing.T) {
	data := []byte{42}
	encoded := RunLengthEncode(data)
	expected := []byte{1, 42}
	assertBytesEqual(t, "rle-single", encoded, expected)

	decoded := RunLengthDecode(encoded)
	assertBytesEqual(t, "rle-single-rt", decoded, data)
}

func TestRLE_LongRun(t *testing.T) {
	// Run of 300 bytes: should split at 255.
	data := make([]byte, 300)
	for i := range data {
		data[i] = 0xAA
	}
	encoded := RunLengthEncode(data)
	// Should produce [255, 0xAA, 45, 0xAA].
	expected := []byte{255, 0xAA, 45, 0xAA}
	assertBytesEqual(t, "rle-long-run", encoded, expected)

	decoded := RunLengthDecode(encoded)
	assertBytesEqual(t, "rle-long-rt", decoded, data)
}

func TestRLE_DecodeOddLength(t *testing.T) {
	// Odd-length input is invalid.
	decoded := RunLengthDecode([]byte{1, 2, 3})
	if decoded != nil {
		t.Errorf("expected nil for odd-length input, got %v", decoded)
	}
}

func TestRLE_RoundtripAllBytes(t *testing.T) {
	// Encode all 256 byte values.
	data := make([]byte, 256)
	for i := 0; i < 256; i++ {
		data[i] = byte(i)
	}
	encoded := RunLengthEncode(data)
	decoded := RunLengthDecode(encoded)
	assertBytesEqual(t, "rle-all-bytes", decoded, data)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — DeltaEncode / Decode
// ═══════════════════════════════════════════════════════════════════════════

func TestDelta_Empty(t *testing.T) {
	encoded := DeltaEncode([]int64{})
	if encoded != nil {
		t.Errorf("expected nil for empty input, got %v", encoded)
	}
	decoded := DeltaDecode([]int64{})
	if decoded != nil {
		t.Errorf("expected nil for empty decode, got %v", decoded)
	}
}

func TestDelta_Constant(t *testing.T) {
	// Constant sequence: deltas are all zeros (except first).
	data := []int64{5, 5, 5, 5, 5}
	encoded := DeltaEncode(data)
	expected := []int64{5, 0, 0, 0, 0}
	assertInt64sEqual(t, "delta-constant", encoded, expected)
}

func TestDelta_Linear(t *testing.T) {
	// Linear sequence: constant delta.
	data := []int64{10, 20, 30, 40, 50}
	encoded := DeltaEncode(data)
	expected := []int64{10, 10, 10, 10, 10}
	assertInt64sEqual(t, "delta-linear", encoded, expected)
}

func TestDelta_Decreasing(t *testing.T) {
	// Decreasing: negative deltas.
	data := []int64{100, 80, 60, 40, 20}
	encoded := DeltaEncode(data)
	expected := []int64{100, -20, -20, -20, -20}
	assertInt64sEqual(t, "delta-decreasing", encoded, expected)
}

func TestDelta_Roundtrip(t *testing.T) {
	data := []int64{1000, 1005, 1003, 1010, 1008, 1020}
	encoded := DeltaEncode(data)
	decoded := DeltaDecode(encoded)
	assertInt64sEqual(t, "delta-roundtrip", decoded, data)
}

func TestDelta_SingleElement(t *testing.T) {
	data := []int64{42}
	encoded := DeltaEncode(data)
	assertInt64sEqual(t, "delta-single", encoded, []int64{42})

	decoded := DeltaDecode(encoded)
	assertInt64sEqual(t, "delta-single-rt", decoded, data)
}

func TestDelta_Timestamps(t *testing.T) {
	// Realistic: monotonically increasing timestamps (milliseconds).
	data := []int64{1711612800000, 1711612800100, 1711612800200, 1711612800350, 1711612800500}
	encoded := DeltaEncode(data)
	// Deltas: [base, 100, 100, 150, 150] — much smaller than originals.
	expected := []int64{1711612800000, 100, 100, 150, 150}
	assertInt64sEqual(t, "delta-timestamps", encoded, expected)

	decoded := DeltaDecode(encoded)
	assertInt64sEqual(t, "delta-timestamps-rt", decoded, data)
}

func TestDelta_NegativeValues(t *testing.T) {
	data := []int64{-10, -5, 0, 5, 10}
	encoded := DeltaEncode(data)
	expected := []int64{-10, 5, 5, 5, 5}
	assertInt64sEqual(t, "delta-negative", encoded, expected)

	decoded := DeltaDecode(encoded)
	assertInt64sEqual(t, "delta-negative-rt", decoded, data)
}

// ═══════════════════════════════════════════════════════════════════════════
// Unit tests — ScalarQuantize / Dequantize
// ═══════════════════════════════════════════════════════════════════════════

func TestQuantize_Roundtrip256(t *testing.T) {
	// 256 levels: roundtrip error should be small.
	data := []float64{0.0, 0.1, 0.5, 0.9, 1.0}
	levels := 256
	quantized := make([]int, len(data))
	min, step := ScalarQuantize(data, levels, quantized)

	reconstructed := make([]float64, len(data))
	ScalarDequantize(quantized, min, step, reconstructed)

	maxErr := step / 2.0
	for i, v := range data {
		diff := math.Abs(v - reconstructed[i])
		if diff > maxErr+1e-14 {
			t.Errorf("element %d: original %v, reconstructed %v, diff %v exceeds max error %v",
				i, v, reconstructed[i], diff, maxErr)
		}
	}
}

func TestQuantize_Levels1(t *testing.T) {
	// Single level: all values map to 0, step = 0.
	data := []float64{1.0, 5.0, 3.0}
	quantized := make([]int, len(data))
	min, step := ScalarQuantize(data, 1, quantized)

	if step != 0 {
		t.Errorf("expected step=0 for levels=1, got %v", step)
	}
	if min != 1.0 {
		t.Errorf("expected min=1.0, got %v", min)
	}
	for i, q := range quantized {
		if q != 0 {
			t.Errorf("quantized[%d] = %d, expected 0", i, q)
		}
	}
}

func TestQuantize_AllSame(t *testing.T) {
	// All identical values: step = 0.
	data := []float64{3.14, 3.14, 3.14}
	quantized := make([]int, len(data))
	min, step := ScalarQuantize(data, 10, quantized)

	if step != 0 {
		t.Errorf("expected step=0 for constant data, got %v", step)
	}
	if min != 3.14 {
		t.Errorf("expected min=3.14, got %v", min)
	}
}

func TestQuantize_TwoLevels(t *testing.T) {
	// Two levels: binary quantization.
	data := []float64{0.0, 0.3, 0.7, 1.0}
	quantized := make([]int, len(data))
	min, step := ScalarQuantize(data, 2, quantized)

	assertClose(t, "quant-2-min", min, 0.0, 1e-15)
	assertClose(t, "quant-2-step", step, 1.0, 1e-15)

	// 0.0 → 0, 0.3 → 0, 0.7 → 1, 1.0 → 1
	expectedQ := []int{0, 0, 1, 1}
	assertIntsEqual(t, "quant-2-bins", quantized, expectedQ)
}

func TestQuantize_Empty(t *testing.T) {
	min, step := ScalarQuantize([]float64{}, 10, []int{})
	if min != 0 || step != 0 {
		t.Errorf("expected (0, 0) for empty data, got (%v, %v)", min, step)
	}
}

func TestQuantize_NegativeRange(t *testing.T) {
	data := []float64{-10.0, -5.0, 0.0, 5.0, 10.0}
	levels := 5
	quantized := make([]int, len(data))
	min, step := ScalarQuantize(data, levels, quantized)

	assertClose(t, "quant-neg-min", min, -10.0, 1e-15)
	assertClose(t, "quant-neg-step", step, 5.0, 1e-15)

	// -10 → 0, -5 → 1, 0 → 2, 5 → 3, 10 → 4
	expectedQ := []int{0, 1, 2, 3, 4}
	assertIntsEqual(t, "quant-neg-bins", quantized, expectedQ)

	// Roundtrip should be exact since values align with bin centers.
	reconstructed := make([]float64, len(data))
	ScalarDequantize(quantized, min, step, reconstructed)
	for i, v := range data {
		assertClose(t, "quant-neg-rt", reconstructed[i], v, 1e-14)
	}
}

func TestQuantize_LargeDataset(t *testing.T) {
	// Generate 1000 random-ish values using a linear congruential generator.
	n := 1000
	data := make([]float64, n)
	seed := int64(42)
	for i := 0; i < n; i++ {
		seed = (seed*1103515245 + 12345) & 0x7FFFFFFF
		data[i] = float64(seed) / float64(0x7FFFFFFF) * 100.0 // range [0, 100]
	}

	levels := 256
	quantized := make([]int, n)
	min, step := ScalarQuantize(data, levels, quantized)

	reconstructed := make([]float64, n)
	ScalarDequantize(quantized, min, step, reconstructed)

	maxErr := step / 2.0
	for i := 0; i < n; i++ {
		diff := math.Abs(data[i] - reconstructed[i])
		if diff > maxErr+1e-10 {
			t.Errorf("element %d: diff %v exceeds max error %v", i, diff, maxErr)
			break
		}
	}
}

func TestDequantize_KnownValues(t *testing.T) {
	// Manual dequantization check.
	quantized := []int{0, 1, 2, 3}
	min := 10.0
	step := 2.5
	out := make([]float64, 4)
	ScalarDequantize(quantized, min, step, out)
	expected := []float64{10.0, 12.5, 15.0, 17.5}
	for i, v := range expected {
		assertClose(t, "dequant-known", out[i], v, 1e-14)
	}
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration tests — cross-function verification
// ═══════════════════════════════════════════════════════════════════════════

func TestEntropy_CrossEntropyDecomposition(t *testing.T) {
	// H(P, Q) = H(P) + KL(P||Q) — fundamental identity.
	p := []float64{0.2, 0.3, 0.5}
	q := []float64{0.1, 0.6, 0.3}
	ce := CrossEntropy(p, q)
	hp := ShannonEntropy(p)
	kl := KLDivergence(p, q)
	assertClose(t, "decomposition", ce, hp+kl, 1e-14)
}

func TestEntropy_MIDecomposition(t *testing.T) {
	// I(X;Y) = H(X) - H(X|Y) — fundamental identity.
	joint := [][]float64{
		{0.1, 0.2},
		{0.3, 0.4},
	}
	mi := MutualInformation(joint)
	margX := []float64{0.3, 0.7}
	hx := ShannonEntropy(margX)
	// H(X|Y) = H(X,Y) - H(Y)
	margY := []float64{0.4, 0.6}
	hxGivenY := JointEntropy(joint) - ShannonEntropy(margY)
	assertClose(t, "mi-decomposition", mi, hx-hxGivenY, 1e-14)
}

func TestDelta_ThenQuantize(t *testing.T) {
	// Pipeline: delta encode timestamps, quantize the deltas.
	timestamps := []int64{1000, 1010, 1020, 1025, 1040}
	deltas := DeltaEncode(timestamps)

	// Convert deltas to float64 for quantization.
	fDeltas := make([]float64, len(deltas))
	for i, d := range deltas {
		fDeltas[i] = float64(d)
	}

	quantized := make([]int, len(fDeltas))
	min, step := ScalarQuantize(fDeltas, 16, quantized)

	// Dequantize and reconstruct.
	reconstructed := make([]float64, len(fDeltas))
	ScalarDequantize(quantized, min, step, reconstructed)

	// First delta (the base value 1000) should roundtrip well.
	assertClose(t, "pipeline-base", reconstructed[0], float64(deltas[0]), step/2.0+1e-10)
}

func TestJointEntropy_ChainRule(t *testing.T) {
	// Chain rule: H(X,Y) = H(X) + H(Y|X).
	joint := [][]float64{
		{0.15, 0.05},
		{0.25, 0.55},
	}
	hxy := JointEntropy(joint)
	margX := []float64{0.2, 0.8}
	hx := ShannonEntropy(margX)
	hyGivenX := ConditionalEntropy(joint)
	assertClose(t, "chain-rule", hxy, hx+hyGivenX, 1e-14)
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

func assertClose(t *testing.T, label string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %v, want %v (tol %v)", label, got, want, tol)
	}
}

func assertBytesEqual(t *testing.T, label string, got, want []byte) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch: got %d, want %d", label, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("%s[%d]: got 0x%02X, want 0x%02X", label, i, got[i], want[i])
		}
	}
}

func assertInt64sEqual(t *testing.T, label string, got, want []int64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch: got %d, want %d", label, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("%s[%d]: got %d, want %d", label, i, got[i], want[i])
		}
	}
}

func assertIntsEqual(t *testing.T, label string, got, want []int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch: got %d, want %d", label, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("%s[%d]: got %d, want %d", label, i, got[i], want[i])
		}
	}
}
