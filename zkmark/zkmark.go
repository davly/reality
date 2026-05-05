// Package zkmark — Zero-Knowledge Mirror-Mark substrate.
//
// Tranche 1 (S61): an honest-pending interface that wraps the existing L43
// cold-verifier digest under a [Prover] / [Verifier] / [Proof] contract.
// The Prover's Prove method takes a caller-injected [SignerFunc] (typically
// the canonical mirror-mark Sign function from `nexus/internal/mirrormark`)
// and returns a [Proof] with `Algorithm = "honest-pending"` and `Proof = nil`.
// A regulator running cold-verify --zk gets the same byte-verifiable mark
// they get today, plus a structurally-distinct envelope that signals the
// Halo2 backend is pending.
//
// Tranche 2 (S62-S63, conditional on Marc dry-run encouragement): a real
// Halo2 [Prover] implementation that returns Proof.Bytes populated with a
// circuit-output proof. The interface is forward-compatible — Tranche 2
// swaps the implementation, not the call site.
//
// Cross-substrate parity (R80b): the Prover/Verifier interfaces translate
// directly to C# (Func<byte[], byte[8], byte[], string> for SignerFunc).
// A C# port at Limitless.Nexus.Client/Forge/ZkMark/ will share the
// honest-pending semantics byte-for-byte.
//
// Honest-gate compliance (R128): every public surface here ships with a
// stub-aware error sentinel. Halo2Prover.Prove returns ErrNotYetWired
// instead of silently producing a fake proof. The cold-verifier --zk
// flag emits proof_pending=true so a downstream parser can branch on
// "real proof available" vs "honest-pending".
//
// Origin: S61 NEW-1 ship per S61_S65_PLAN.md. Closes the "regulator-grade
// audit substrate" leg of the SYNTHESIS top-3 picks.
package zkmark

import (
	"errors"
	"fmt"
)

// ErrNotYetWired is returned by [Halo2Prover.Prove] until the Tranche 2
// Halo2 sidecar lands. Callers should branch on this and fall back to
// the [HonestProver] if available.
var ErrNotYetWired = errors.New("zkmark: Halo2 backend not yet wired (Tranche 2)")

// SignerFunc is the canonical mirror-mark digest signature: given a
// payload, a 32-byte corpus SHA, and a key, return a mirror-mark string
// of the form "lore@v1:<base64url-of-corpus-prefix-and-hmac>".
//
// Caller is responsible for injecting the canonical implementation
// (typically nexus/internal/mirrormark.Sign in the Go ecosystem; the
// equivalent C# Func in the .NET ecosystem). The substrate does NOT
// re-implement the digest — that would risk byte-parity drift across
// the L43 cohort.
type SignerFunc func(payload []byte, corpusSHA [32]byte, key []byte) string

// VerifierFunc is the canonical mirror-mark verify signature: given a
// mark, corpus SHA, payload, and key, return (ok, error). Caller injects
// the canonical implementation (mirrormark.Verify in Go).
type VerifierFunc func(mark string, corpusSHA [32]byte, payload []byte, key []byte) (bool, error)

// Algorithm names. Forward-compatible with future Tranche 2+ algorithms.
const (
	// AlgorithmHonestPending is Tranche 1: the proof is the mirror-mark
	// itself, byte-verifiable via the cold-verifier. No zero-knowledge
	// property; the regulator sees the payload + key + corpus directly.
	AlgorithmHonestPending = "honest-pending"

	// AlgorithmHalo2 is Tranche 2: a real Halo2 circuit proof. The
	// regulator sees the proof bytes only; the payload/key/corpus
	// remain hidden (the zero-knowledge property the substrate is
	// named for).
	AlgorithmHalo2 = "halo2"
)

// Proof is the attestation returned by [Prover.Prove]. The shape is
// forward-compatible across Tranche 1 (HonestProver) and Tranche 2
// (Halo2Prover). A verifier branches on Algorithm + ProofPending:
//
//   - AlgorithmHonestPending + ProofPending=true: re-derive via cold-verifier
//   - AlgorithmHalo2 + ProofPending=false: full Halo2 circuit verification
//
// Future Algorithm values can be added without breaking existing parsers
// because ProofPending defaults to true (safe default: assume the proof
// needs cold-verifier corroboration unless the implementation explicitly
// sets it to false).
type Proof struct {
	// MarkChainStatement is the mirror-mark string that anchors this
	// proof to the L43 chain. ALWAYS populated, regardless of Algorithm.
	// A cold-verifier with the corpus + key can re-derive this byte-
	// for-byte.
	MarkChainStatement string

	// Algorithm names the prover that produced this proof. See the
	// Algorithm* constants. New algorithms can be added in future
	// Tranches without breaking existing parsers.
	Algorithm string

	// ProofPending signals "no real cryptographic proof yet — verify
	// via cold-verifier digest only". Tranche 1 always sets this true;
	// Tranche 2 sets it false once Halo2 is wired.
	//
	// Default zero-value (false) is the SAFER conservative interpretation
	// IF the proof is supposed to be real. To preserve "verify via
	// cold-verifier" semantics in unset cases, callers should construct
	// Proof structs explicitly rather than relying on zero values.
	ProofPending bool

	// ProofBytes carries the algorithm-specific proof payload. Tranche 1
	// (HonestProver) leaves this nil. Tranche 2 (Halo2Prover) populates
	// it with the Halo2 circuit output. Other future algorithms populate
	// it with their respective proof data.
	ProofBytes []byte

	// CorpusSHA echoes the 32-byte corpus SHA the proof was generated
	// against. Surfacing it explicitly lets a verifier short-circuit
	// the corpus walk if it already has the SHA cached.
	CorpusSHA [32]byte
}

// Prover is the interface for ZK Mirror-Mark proof generation.
// Tranche 1 ships [HonestProver]; Tranche 2 will ship Halo2Prover.
type Prover interface {
	// Prove produces a [Proof] over (payload, corpusSHA, key). Pure /
	// deterministic for a given Prover implementation: same inputs,
	// same Proof. No I/O; no network; no clock dependency.
	Prove(payload []byte, corpusSHA [32]byte, key []byte) (Proof, error)

	// Algorithm returns the Algorithm name this Prover produces.
	// Used by callers that want to log / dispatch on the algorithm
	// without invoking Prove.
	Algorithm() string
}

// Verifier is the interface for ZK Mirror-Mark proof verification.
// Tranche 1 ships [MarkVerifier] which delegates to a caller-injected
// VerifierFunc; Tranche 2 will add a Halo2Verifier.
type Verifier interface {
	// VerifyProof checks the proof's integrity against the supplied
	// payload + key. Returns (true, nil) on success; (false, err) on
	// failure with the underlying mirror-mark or Halo2 error preserved
	// in err. Pure / deterministic.
	VerifyProof(proof Proof, payload []byte, key []byte) (bool, error)
}

// ---------------------------------------------------------------------------
// HonestProver — Tranche 1 implementation
// ---------------------------------------------------------------------------

// HonestProver is the Tranche 1 [Prover] implementation. It produces a
// Proof whose MarkChainStatement is a real, byte-verifiable mirror-mark
// (computed by the caller-injected [SignerFunc]) but whose ProofBytes is
// nil. Honest about not being a "real" zero-knowledge prover.
//
// Cold-verifier flow: a regulator with the same corpus + key can re-derive
// MarkChainStatement byte-for-byte using the existing lore-mark-verify
// CLI. The "ZK" of zkmark is structural in Tranche 1, not cryptographic.
type HonestProver struct {
	sign SignerFunc
}

// NewHonestProver constructs an HonestProver wrapping the canonical
// mirror-mark digest. Typical usage:
//
//	import "nexus/internal/mirrormark"
//	prover := zkmark.NewHonestProver(mirrormark.Sign)
//
// Returns nil if sign is nil (caller bug).
func NewHonestProver(sign SignerFunc) *HonestProver {
	if sign == nil {
		return nil
	}
	return &HonestProver{sign: sign}
}

// Prove implements [Prover]. Calls the injected SignerFunc to produce
// a real mirror-mark; wraps it in a Proof envelope with ProofPending=true.
func (h *HonestProver) Prove(payload []byte, corpusSHA [32]byte, key []byte) (Proof, error) {
	if h == nil || h.sign == nil {
		return Proof{}, errors.New("zkmark: HonestProver not initialized (nil SignerFunc)")
	}
	mark := h.sign(payload, corpusSHA, key)
	return Proof{
		MarkChainStatement: mark,
		Algorithm:          AlgorithmHonestPending,
		ProofPending:       true,
		ProofBytes:         nil,
		CorpusSHA:          corpusSHA,
	}, nil
}

// Algorithm implements [Prover].
func (h *HonestProver) Algorithm() string {
	return AlgorithmHonestPending
}

// ---------------------------------------------------------------------------
// Halo2Prover — Tranche 2 stub
// ---------------------------------------------------------------------------

// Halo2Prover is the Tranche 2 stub. Returns [ErrNotYetWired] from Prove
// so callers can detect the not-yet-implemented state and fall back to
// [HonestProver] explicitly.
//
// When Tranche 2 lands, this struct will hold a handle to the Rust
// Halo2 sidecar (likely an IPC channel or FFI binding); Prove will
// produce a real circuit proof; the type signature stays unchanged.
type Halo2Prover struct{}

// NewHalo2Prover returns a Halo2Prover stub. Calling Prove on it
// returns [ErrNotYetWired].
func NewHalo2Prover() *Halo2Prover {
	return &Halo2Prover{}
}

// Prove implements [Prover]. Tranche 1 stub: returns [ErrNotYetWired].
func (h *Halo2Prover) Prove(payload []byte, corpusSHA [32]byte, key []byte) (Proof, error) {
	return Proof{}, ErrNotYetWired
}

// Algorithm implements [Prover].
func (h *Halo2Prover) Algorithm() string {
	return AlgorithmHalo2
}

// ---------------------------------------------------------------------------
// MarkVerifier — Tranche 1 verifier
// ---------------------------------------------------------------------------

// MarkVerifier verifies an honest-pending Proof by delegating to a
// caller-injected [VerifierFunc] (typically nexus/internal/mirrormark.Verify).
// For AlgorithmHalo2 proofs (Tranche 2+), this verifier returns
// [ErrNotYetWired] — a future Halo2Verifier will handle that branch.
type MarkVerifier struct {
	verify VerifierFunc
}

// NewMarkVerifier constructs a MarkVerifier wrapping the canonical
// mirror-mark verify function. Returns nil if verify is nil.
func NewMarkVerifier(verify VerifierFunc) *MarkVerifier {
	if verify == nil {
		return nil
	}
	return &MarkVerifier{verify: verify}
}

// VerifyProof implements [Verifier]. For AlgorithmHonestPending proofs,
// delegates to the injected VerifierFunc. For AlgorithmHalo2 proofs,
// returns [ErrNotYetWired] (Halo2Verifier is Tranche 2). For unknown
// algorithms, returns a wrapped error.
func (m *MarkVerifier) VerifyProof(proof Proof, payload []byte, key []byte) (bool, error) {
	if m == nil || m.verify == nil {
		return false, errors.New("zkmark: MarkVerifier not initialized (nil VerifierFunc)")
	}
	switch proof.Algorithm {
	case AlgorithmHonestPending:
		return m.verify(proof.MarkChainStatement, proof.CorpusSHA, payload, key)
	case AlgorithmHalo2:
		return false, ErrNotYetWired
	case "":
		return false, errors.New("zkmark: Proof.Algorithm is empty (malformed proof)")
	default:
		return false, fmt.Errorf("zkmark: unknown proof algorithm %q", proof.Algorithm)
	}
}
