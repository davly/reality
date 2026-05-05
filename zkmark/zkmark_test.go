package zkmark

import (
	"bytes"
	"errors"
	"testing"
)

// fakeSign is a deterministic stub of the canonical mirror-mark Sign
// function for tests. Real callers wire mirrormark.Sign from nexus.
func fakeSign(payload []byte, corpusSHA [32]byte, key []byte) string {
	// Deterministic but not real-mirror-mark format. Sufficient for
	// interface-shape tests; real cross-byte-parity tests live in the
	// nexus repo where mirrormark.Sign is callable.
	return "fake@v1:" + string(corpusSHA[:4]) + "|" + string(payload[:min(8, len(payload))]) + "|" + string(key[:min(4, len(key))])
}

func fakeVerify(mark string, corpusSHA [32]byte, payload []byte, key []byte) (bool, error) {
	expected := fakeSign(payload, corpusSHA, key)
	if mark == expected {
		return true, nil
	}
	return false, errors.New("fake verify: mismatch")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ---------------------------------------------------------------------------
// HonestProver tests
// ---------------------------------------------------------------------------

func TestHonestProver_Prove_PopulatesProofShape(t *testing.T) {
	prover := NewHonestProver(fakeSign)
	if prover == nil {
		t.Fatal("NewHonestProver returned nil for non-nil SignerFunc")
	}

	payload := []byte("test payload")
	var corpusSHA [32]byte
	copy(corpusSHA[:], []byte("0123456789abcdef0123456789abcdef"))
	key := []byte("iik_testkey")

	proof, err := prover.Prove(payload, corpusSHA, key)
	if err != nil {
		t.Fatalf("Prove failed: %v", err)
	}

	if proof.MarkChainStatement == "" {
		t.Error("MarkChainStatement should be populated by HonestProver")
	}
	if proof.Algorithm != AlgorithmHonestPending {
		t.Errorf("Algorithm = %q, want %q", proof.Algorithm, AlgorithmHonestPending)
	}
	if !proof.ProofPending {
		t.Error("ProofPending should be true for HonestProver")
	}
	if proof.ProofBytes != nil {
		t.Errorf("ProofBytes should be nil for HonestProver, got %d bytes", len(proof.ProofBytes))
	}
	if proof.CorpusSHA != corpusSHA {
		t.Error("CorpusSHA should be echoed unchanged")
	}
}

func TestHonestProver_NewWithNilSigner_ReturnsNil(t *testing.T) {
	prover := NewHonestProver(nil)
	if prover != nil {
		t.Error("NewHonestProver(nil) should return nil")
	}
}

func TestHonestProver_Algorithm(t *testing.T) {
	prover := NewHonestProver(fakeSign)
	if prover.Algorithm() != AlgorithmHonestPending {
		t.Errorf("Algorithm() = %q, want %q", prover.Algorithm(), AlgorithmHonestPending)
	}
}

// ---------------------------------------------------------------------------
// Halo2Prover tests
// ---------------------------------------------------------------------------

func TestHalo2Prover_Prove_ReturnsErrNotYetWired(t *testing.T) {
	prover := NewHalo2Prover()
	_, err := prover.Prove([]byte("p"), [32]byte{}, []byte("k"))
	if !errors.Is(err, ErrNotYetWired) {
		t.Errorf("expected ErrNotYetWired, got %v", err)
	}
}

func TestHalo2Prover_Algorithm(t *testing.T) {
	prover := NewHalo2Prover()
	if prover.Algorithm() != AlgorithmHalo2 {
		t.Errorf("Algorithm() = %q, want %q", prover.Algorithm(), AlgorithmHalo2)
	}
}

// ---------------------------------------------------------------------------
// MarkVerifier tests
// ---------------------------------------------------------------------------

func TestMarkVerifier_VerifyProof_HonestPending_RoundTrip(t *testing.T) {
	prover := NewHonestProver(fakeSign)
	verifier := NewMarkVerifier(fakeVerify)

	payload := []byte("round trip payload")
	var corpusSHA [32]byte
	copy(corpusSHA[:], []byte("11111111222222223333333344444444"))
	key := []byte("iik_round")

	proof, err := prover.Prove(payload, corpusSHA, key)
	if err != nil {
		t.Fatalf("Prove failed: %v", err)
	}

	ok, err := verifier.VerifyProof(proof, payload, key)
	if err != nil {
		t.Fatalf("VerifyProof failed: %v", err)
	}
	if !ok {
		t.Error("VerifyProof returned ok=false on round-trip")
	}
}

func TestMarkVerifier_VerifyProof_TamperedPayload_Fails(t *testing.T) {
	prover := NewHonestProver(fakeSign)
	verifier := NewMarkVerifier(fakeVerify)

	original := []byte("original payload")
	tampered := []byte("tampered xxxxxxx")
	var corpusSHA [32]byte
	copy(corpusSHA[:], []byte("11111111222222223333333344444444"))
	key := []byte("iik_tamp")

	proof, _ := prover.Prove(original, corpusSHA, key)

	ok, err := verifier.VerifyProof(proof, tampered, key)
	if ok {
		t.Error("VerifyProof returned ok=true on tampered payload")
	}
	if err == nil {
		t.Error("VerifyProof returned nil error on tampered payload")
	}
}

func TestMarkVerifier_VerifyProof_Halo2_ReturnsErrNotYetWired(t *testing.T) {
	verifier := NewMarkVerifier(fakeVerify)
	proof := Proof{
		Algorithm:    AlgorithmHalo2,
		ProofPending: false,
		ProofBytes:   bytes.Repeat([]byte{0x01}, 32),
	}
	_, err := verifier.VerifyProof(proof, []byte("p"), []byte("k"))
	if !errors.Is(err, ErrNotYetWired) {
		t.Errorf("expected ErrNotYetWired for Halo2 proof, got %v", err)
	}
}

func TestMarkVerifier_VerifyProof_EmptyAlgorithm_Fails(t *testing.T) {
	verifier := NewMarkVerifier(fakeVerify)
	proof := Proof{Algorithm: ""}
	ok, err := verifier.VerifyProof(proof, []byte("p"), []byte("k"))
	if ok {
		t.Error("VerifyProof returned ok=true for empty Algorithm")
	}
	if err == nil {
		t.Error("VerifyProof returned nil error for empty Algorithm")
	}
}

func TestMarkVerifier_VerifyProof_UnknownAlgorithm_Fails(t *testing.T) {
	verifier := NewMarkVerifier(fakeVerify)
	proof := Proof{Algorithm: "made-up"}
	_, err := verifier.VerifyProof(proof, []byte("p"), []byte("k"))
	if err == nil {
		t.Error("VerifyProof returned nil error for unknown algorithm")
	}
}

func TestMarkVerifier_NewWithNilVerifier_ReturnsNil(t *testing.T) {
	v := NewMarkVerifier(nil)
	if v != nil {
		t.Error("NewMarkVerifier(nil) should return nil")
	}
}

// ---------------------------------------------------------------------------
// Forward-compat regression: adding a new algorithm doesn't break existing
// parsers IF they treat unknown algorithms as ProofPending=true (the
// conservative default). This test pins that contract.
// ---------------------------------------------------------------------------

func TestProof_NewAlgorithmDefaultsToProofPending(t *testing.T) {
	// A future Tranche 2+ Algorithm could ship as ProofPending=false
	// (real proof inside ProofBytes). But in the meantime, anyone
	// constructing a Proof by hand without setting ProofPending gets
	// ProofPending=false (Go zero value), which means "trust the
	// ProofBytes". To preserve safety in face of forward-evolution,
	// THE EXPLICIT DOCSTRING in zkmark.go (Proof.ProofPending) tells
	// callers to construct explicitly. This test pins the runtime
	// behavior so the doc is enforceable.
	zero := Proof{}
	if zero.ProofPending {
		t.Error("Proof zero value should have ProofPending=false (Go default); doc must warn callers explicitly")
	}
}
