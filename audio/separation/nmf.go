package separation

import "math"

// Decompose performs Non-negative Matrix Factorisation of a non-negative
// input matrix V into two non-negative factors W and H using the Lee &
// Seung (1999, 2001) multiplicative-update rules with the Frobenius-norm
// objective.
//
// Algorithm (Lee & Seung 2001, Theorem 1 — Frobenius-norm
// multiplicative updates):
//
//	V ≈ W · H
//	V is F × T (frequency bins × time frames)
//	W is F × R (basis matrix — R "atoms")
//	H is R × T (activation matrix)
//
// Update rules (alternating; multiplicative; preserves non-negativity):
//
//	H ← H ⊙ (Wᵀ V) ⊘ (Wᵀ W H + ε)
//	W ← W ⊙ (V Hᵀ) ⊘ (W H Hᵀ + ε)
//
// where ⊙ is element-wise multiplication, ⊘ is element-wise division,
// and ε is a tiny constant (1e-12) preventing division by zero. The
// Frobenius-norm objective ‖V - WH‖² is non-increasing under these
// updates (Lee & Seung 2001 Theorem 1) but the iteration is not
// guaranteed to converge to a global optimum — it finds a local
// minimum.
//
// NMF is well-suited to spectrogram separation because:
//   - Spectrograms (magnitude or power) are intrinsically non-negative.
//   - Repeated patterns (a bird call sung many times, a drum hit) form
//     a low-dimensional basis: few atoms, many activations.
//   - W learns the "what" (per-atom spectral templates) and H learns
//     the "when" (activation timing).
//
// Parameters:
//   - V:          F × T spectrogram (row-major, all entries >= 0)
//   - rank:       R, number of latent atoms (typical: 2-16 for
//     monophonic separation, more for polyphonic)
//   - iterations: number of multiplicative-update passes (typical:
//     50-200; convergence is monotonic but slow)
//
// Returns: (W [F × R], H [R × T]) as row-major matrices wrapped as
// [][]float64. Newly allocated; the caller owns the returned slices.
//
// Initialisation: deterministic absolute-value of a fixed Halton-like
// sequence — pure for reproducibility. Callers wanting stochastic
// initialisation should apply a permutation pre-call.
//
// Valid range: V non-negative; rank >= 1; iterations >= 1.
// Precision: relative reconstruction error ‖V-WH‖_F / ‖V‖_F drops by
// ~50% in the first 20 iterations and asymptotes; convergence
// tolerance is iteration-driven (no early-stop in this implementation).
// Panics on shape violations or empty input.
//
// Reference: Lee, D. D. & Seung, H. S. (1999). "Learning the parts of
// objects by non-negative matrix factorization." Nature 401, 788-791;
// Lee, D. D. & Seung, H. S. (2001). "Algorithms for Non-negative
// Matrix Factorization." Advances in Neural Information Processing
// Systems 13; Smaragdis, P. & Brown, J. C. (2003) "Non-negative matrix
// factorization for polyphonic music transcription" IEEE WASPAA — the
// definitive audio-NMF reference.
//
// Consumed by: pigeonhole (multi-bird-song basis decomposition),
// dipstick (machine-component spectral atoms).
func Decompose(V [][]float64, rank, iterations int) (W, H [][]float64) {
	F := len(V)
	if F < 1 {
		panic("separation.Decompose: V must have at least 1 row")
	}
	T := len(V[0])
	if T < 1 {
		panic("separation.Decompose: V must have at least 1 column")
	}
	for f := 1; f < F; f++ {
		if len(V[f]) != T {
			panic("separation.Decompose: all rows of V must have equal length")
		}
	}
	if rank < 1 {
		panic("separation.Decompose: rank must be >= 1")
	}
	if iterations < 1 {
		panic("separation.Decompose: iterations must be >= 1")
	}
	for f := 0; f < F; f++ {
		for t := 0; t < T; t++ {
			if V[f][t] < 0 {
				panic("separation.Decompose: V must be non-negative")
			}
		}
	}

	// Allocate W (F × R) and H (R × T).
	W = make([][]float64, F)
	for f := 0; f < F; f++ {
		W[f] = make([]float64, rank)
	}
	H = make([][]float64, rank)
	for r := 0; r < rank; r++ {
		H[r] = make([]float64, T)
	}

	// Deterministic init in (0.5 ± 0.5) using a low-discrepancy
	// Halton sequence (van der Corput base 2 / 3).
	for f := 0; f < F; f++ {
		for r := 0; r < rank; r++ {
			W[f][r] = 0.1 + halton(f*rank+r+1, 2)
		}
	}
	for r := 0; r < rank; r++ {
		for t := 0; t < T; t++ {
			H[r][t] = 0.1 + halton(r*T+t+1, 3)
		}
	}

	const eps = 1e-12

	// Scratch buffers for matrix multiplications.
	WtV := make([][]float64, rank)
	for r := 0; r < rank; r++ {
		WtV[r] = make([]float64, T)
	}
	WtW := make([][]float64, rank)
	for r := 0; r < rank; r++ {
		WtW[r] = make([]float64, rank)
	}
	WtWH := make([][]float64, rank)
	for r := 0; r < rank; r++ {
		WtWH[r] = make([]float64, T)
	}
	VHt := make([][]float64, F)
	for f := 0; f < F; f++ {
		VHt[f] = make([]float64, rank)
	}
	HHt := make([][]float64, rank)
	for r := 0; r < rank; r++ {
		HHt[r] = make([]float64, rank)
	}
	WHHt := make([][]float64, F)
	for f := 0; f < F; f++ {
		WHHt[f] = make([]float64, rank)
	}

	for it := 0; it < iterations; it++ {
		// H update: H ← H ⊙ (Wᵀ V) ⊘ (Wᵀ W H + ε)

		// WtV [R × T] = Wᵀ V
		for r := 0; r < rank; r++ {
			for t := 0; t < T; t++ {
				s := 0.0
				for f := 0; f < F; f++ {
					s += W[f][r] * V[f][t]
				}
				WtV[r][t] = s
			}
		}
		// WtW [R × R] = Wᵀ W
		for r1 := 0; r1 < rank; r1++ {
			for r2 := 0; r2 < rank; r2++ {
				s := 0.0
				for f := 0; f < F; f++ {
					s += W[f][r1] * W[f][r2]
				}
				WtW[r1][r2] = s
			}
		}
		// WtWH [R × T] = WtW H
		for r := 0; r < rank; r++ {
			for t := 0; t < T; t++ {
				s := 0.0
				for r2 := 0; r2 < rank; r2++ {
					s += WtW[r][r2] * H[r2][t]
				}
				WtWH[r][t] = s
			}
		}
		// H ← H ⊙ WtV ⊘ (WtWH + ε)
		for r := 0; r < rank; r++ {
			for t := 0; t < T; t++ {
				H[r][t] *= WtV[r][t] / (WtWH[r][t] + eps)
			}
		}

		// W update: W ← W ⊙ (V Hᵀ) ⊘ (W H Hᵀ + ε)

		// VHt [F × R] = V Hᵀ
		for f := 0; f < F; f++ {
			for r := 0; r < rank; r++ {
				s := 0.0
				for t := 0; t < T; t++ {
					s += V[f][t] * H[r][t]
				}
				VHt[f][r] = s
			}
		}
		// HHt [R × R] = H Hᵀ
		for r1 := 0; r1 < rank; r1++ {
			for r2 := 0; r2 < rank; r2++ {
				s := 0.0
				for t := 0; t < T; t++ {
					s += H[r1][t] * H[r2][t]
				}
				HHt[r1][r2] = s
			}
		}
		// WHHt [F × R] = W HHt
		for f := 0; f < F; f++ {
			for r := 0; r < rank; r++ {
				s := 0.0
				for r2 := 0; r2 < rank; r2++ {
					s += W[f][r2] * HHt[r2][r]
				}
				WHHt[f][r] = s
			}
		}
		// W ← W ⊙ VHt ⊘ (WHHt + ε)
		for f := 0; f < F; f++ {
			for r := 0; r < rank; r++ {
				W[f][r] *= VHt[f][r] / (WHHt[f][r] + eps)
			}
		}
	}
	return W, H
}

// halton returns the n-th element of the van der Corput / Halton
// low-discrepancy sequence in base b. n >= 1, returns a value in
// [0, 1). Used as a deterministic non-zero initialiser for NMF
// factor matrices to avoid the trivial all-zero fixed-point of the
// multiplicative-update iteration.
func halton(n, base int) float64 {
	f := 1.0
	r := 0.0
	for n > 0 {
		f /= float64(base)
		r += f * float64(n%base)
		n /= base
	}
	return r
}

// Reconstruct returns W·H as a newly-allocated F × T matrix. Useful
// for inspecting NMF reconstruction quality. Convenience for callers;
// not used by Decompose itself.
//
// Returns: F × T row-major matrix.
// Panics on shape violation.
func Reconstruct(W, H [][]float64) [][]float64 {
	F := len(W)
	if F == 0 {
		panic("separation.Reconstruct: W must have at least 1 row")
	}
	R := len(W[0])
	if R == 0 {
		panic("separation.Reconstruct: W rows must have at least 1 column")
	}
	if len(H) != R {
		panic("separation.Reconstruct: H must have R rows where R = len(W[0])")
	}
	T := len(H[0])
	out := make([][]float64, F)
	for f := 0; f < F; f++ {
		out[f] = make([]float64, T)
		for t := 0; t < T; t++ {
			s := 0.0
			for r := 0; r < R; r++ {
				s += W[f][r] * H[r][t]
			}
			out[f][t] = s
		}
	}
	return out
}

// FrobeniusError returns ‖A - B‖_F (Frobenius norm of the element-wise
// difference). Both matrices must have identical shape. Used as a
// reconstruction-quality metric.
//
// Panics on shape mismatch.
func FrobeniusError(A, B [][]float64) float64 {
	if len(A) != len(B) {
		panic("separation.FrobeniusError: A and B must have equal row count")
	}
	s := 0.0
	for i := 0; i < len(A); i++ {
		if len(A[i]) != len(B[i]) {
			panic("separation.FrobeniusError: A and B rows must have equal length")
		}
		for j := 0; j < len(A[i]); j++ {
			d := A[i][j] - B[i][j]
			s += d * d
		}
	}
	return math.Sqrt(s)
}
