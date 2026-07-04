package statespace

import (
	"errors"
	"math"
)

// ErrDimension is returned when a matrix or buffer has the wrong length for
// the declared dimensions n (state) / m (observation).
var ErrDimension = errors.New("statespace: matrix or buffer dimension mismatch")

// ErrSingular is returned when the innovation covariance S = H P Hᵀ + R is
// numerically singular and cannot be inverted.
var ErrSingular = errors.New("statespace: innovation covariance is singular")

// ErrEmptyData is returned when a filter routine receives an empty series.
var ErrEmptyData = errors.New("statespace: observation series must be non-empty")

// log2Pi is the constant log(2*pi) used in the Gaussian log-likelihood.
const log2Pi = 1.8378770664093454835606594728112

// ---------------------------------------------------------------------------
// Small dense linear-algebra helpers.
//
// All matrices are flat row-major slices: element (i,j) of an r×c matrix M is
// M[i*c+j]. These are intentionally local (not imported from linalg) so the
// package is self-contained and the recursion is auditable in one file. The
// measurement dimension m in a state-space model is small, so the O(m^3)
// Gauss-Jordan inverse is not a bottleneck.
// ---------------------------------------------------------------------------

// matVec computes y = A x, where A is r×c (row-major) and x has length c.
// y must have length r.
func matVec(a, x, y []float64, r, c int) {
	for i := 0; i < r; i++ {
		s := 0.0
		base := i * c
		for j := 0; j < c; j++ {
			s += a[base+j] * x[j]
		}
		y[i] = s
	}
}

// matMul computes C = A B, where A is p×q, B is q×r, C is p×r (all row-major).
func matMul(a, b, c []float64, p, q, r int) {
	for i := 0; i < p; i++ {
		for k := 0; k < r; k++ {
			s := 0.0
			for j := 0; j < q; j++ {
				s += a[i*q+j] * b[j*r+k]
			}
			c[i*r+k] = s
		}
	}
}

// matMulT computes C = A Bᵀ, where A is p×q, B is r×q, C is p×r.
func matMulT(a, b, c []float64, p, q, r int) {
	for i := 0; i < p; i++ {
		for k := 0; k < r; k++ {
			s := 0.0
			for j := 0; j < q; j++ {
				s += a[i*q+j] * b[k*q+j]
			}
			c[i*r+k] = s
		}
	}
}

// invMat inverts a general n×n matrix a (row-major) into out by Gauss-Jordan
// elimination with partial pivoting. Returns ErrSingular if the matrix is
// numerically singular. a is not modified.
func invMat(a, out []float64, n int) error {
	// Working augmented copy [a | I].
	aug := make([]float64, n*2*n)
	w := 2 * n
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			aug[i*w+j] = a[i*n+j]
		}
		aug[i*w+n+i] = 1.0
	}
	for col := 0; col < n; col++ {
		// Partial pivot: largest magnitude in this column at/below the pivot.
		piv := col
		best := math.Abs(aug[col*w+col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(aug[r*w+col]); v > best {
				best = v
				piv = r
			}
		}
		if best < 1e-300 || math.IsNaN(best) {
			return ErrSingular
		}
		if piv != col {
			for j := 0; j < w; j++ {
				aug[col*w+j], aug[piv*w+j] = aug[piv*w+j], aug[col*w+j]
			}
		}
		// Normalise pivot row.
		d := aug[col*w+col]
		for j := 0; j < w; j++ {
			aug[col*w+j] /= d
		}
		// Eliminate the column from every other row.
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := aug[r*w+col]
			if f == 0 {
				continue
			}
			for j := 0; j < w; j++ {
				aug[r*w+j] -= f * aug[col*w+j]
			}
		}
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			out[i*n+j] = aug[i*w+n+j]
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Kalman primitives (output-buffer style).
// ---------------------------------------------------------------------------

// KalmanPredict performs the time update of the Kalman filter for an
// n-dimensional linear-Gaussian state:
//
//	xOut = F x
//	POut = F P Fᵀ + Q
//
// x is the length-n prior mean, P the n×n prior covariance (row-major), F the
// n×n transition matrix, Q the n×n process-noise covariance. Results are
// written into xOut (length n) and POut (length n×n). POut may not alias P.
// Returns ErrDimension if any slice has the wrong length.
func KalmanPredict(x, P, F, Q []float64, n int, xOut, POut []float64) error {
	if len(x) != n || len(P) != n*n || len(F) != n*n || len(Q) != n*n ||
		len(xOut) != n || len(POut) != n*n {
		return ErrDimension
	}
	matVec(F, x, xOut, n, n)
	// POut = F P Fᵀ + Q. Use a scratch FP = F P first.
	fp := make([]float64, n*n)
	matMul(F, P, fp, n, n, n)
	matMulT(fp, F, POut, n, n, n) // (F P) Fᵀ
	for i := range POut {
		POut[i] += Q[i]
	}
	symmetrise(POut, n)
	return nil
}

// KalmanUpdate performs the measurement update for observation z (length m):
//
//	v = z - H x            (innovation)
//	S = H P Hᵀ + R         (innovation covariance)
//	K = P Hᵀ S⁻¹           (Kalman gain)
//	xOut = x + K v
//	POut = (I-KH) P (I-KH)ᵀ + K R Kᵀ   (Joseph-stabilised form)
//
// x is the length-n predicted mean, P the n×n predicted covariance, H the m×n
// observation matrix, R the m×m observation-noise covariance. The updated mean
// and covariance are written to xOut (n) and POut (n×n); the innovation is
// written to vOut (m) and the innovation covariance to SOut (m×m). The Gaussian
// log-likelihood of the observation under the predictive distribution,
// log N(z; H x, S), is returned. Returns ErrSingular if S cannot be inverted.
func KalmanUpdate(x, P, z, H, R []float64, n, m int, xOut, POut, vOut, SOut []float64) (loglik float64, err error) {
	if len(x) != n || len(P) != n*n || len(z) != m || len(H) != m*n || len(R) != m*m ||
		len(xOut) != n || len(POut) != n*n || len(vOut) != m || len(SOut) != m*m {
		return 0, ErrDimension
	}
	// Innovation v = z - H x.
	hx := make([]float64, m)
	matVec(H, x, hx, m, n)
	for i := 0; i < m; i++ {
		vOut[i] = z[i] - hx[i]
	}
	// S = H P Hᵀ + R.
	hp := make([]float64, m*n) // m×n
	matMul(H, P, hp, m, n, n)
	matMulT(hp, H, SOut, m, n, m) // (H P) Hᵀ -> m×m
	for i := range SOut {
		SOut[i] += R[i]
	}
	symmetrise(SOut, m)
	// Sinv.
	sInv := make([]float64, m*m)
	if e := invMat(SOut, sInv, m); e != nil {
		return 0, e
	}
	// Kalman gain K = P Hᵀ S⁻¹  (n×m).
	pHt := make([]float64, n*m)
	matMulT(P, H, pHt, n, n, m) // P Hᵀ -> n×m
	K := make([]float64, n*m)
	matMul(pHt, sInv, K, n, m, m)
	// xOut = x + K v.
	kv := make([]float64, n)
	matVec(K, vOut, kv, n, m)
	for i := 0; i < n; i++ {
		xOut[i] = x[i] + kv[i]
	}
	// Joseph form: POut = A P Aᵀ + K R Kᵀ, A = I - K H  (n×n).
	A := make([]float64, n*n)
	matMul(K, H, A, n, m, n) // K H -> n×n
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				A[i*n+j] = 1.0 - A[i*n+j]
			} else {
				A[i*n+j] = -A[i*n+j]
			}
		}
	}
	ap := make([]float64, n*n)
	matMul(A, P, ap, n, n, n)
	matMulT(ap, A, POut, n, n, n) // A P Aᵀ
	// + K R Kᵀ.
	kr := make([]float64, n*m)
	matMul(K, R, kr, n, m, m)
	krkt := make([]float64, n*n)
	matMulT(kr, K, krkt, n, m, n)
	for i := range POut {
		POut[i] += krkt[i]
	}
	symmetrise(POut, n)
	// Log-likelihood log N(v; 0, S) = -0.5(m log2pi + log|S| + vᵀ S⁻¹ v).
	logDet, ok := logDetFromInverseCheck(SOut, m)
	if !ok {
		return 0, ErrSingular
	}
	quad := 0.0
	for i := 0; i < m; i++ {
		for j := 0; j < m; j++ {
			quad += vOut[i] * sInv[i*m+j] * vOut[j]
		}
	}
	loglik = -0.5 * (float64(m)*log2Pi + logDet + quad)
	return loglik, nil
}

// symmetrise averages M with its transpose in place to damp round-off
// asymmetry in covariance products.
func symmetrise(M []float64, n int) {
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			avg := 0.5 * (M[i*n+j] + M[j*n+i])
			M[i*n+j] = avg
			M[j*n+i] = avg
		}
	}
}

// logDetFromInverseCheck returns log|M| for a symmetric positive-definite M via
// LU (Doolittle) with partial pivoting, and false if M is not positive.
func logDetFromInverseCheck(M []float64, n int) (float64, bool) {
	lu := make([]float64, n*n)
	copy(lu, M)
	logDet := 0.0
	for col := 0; col < n; col++ {
		piv := col
		best := math.Abs(lu[col*n+col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(lu[r*n+col]); v > best {
				best = v
				piv = r
			}
		}
		if best < 1e-300 || math.IsNaN(best) {
			return 0, false
		}
		if piv != col {
			for j := 0; j < n; j++ {
				lu[col*n+j], lu[piv*n+j] = lu[piv*n+j], lu[col*n+j]
			}
		}
		d := lu[col*n+col]
		logDet += math.Log(math.Abs(d))
		for r := col + 1; r < n; r++ {
			f := lu[r*n+col] / d
			for j := col; j < n; j++ {
				lu[r*n+j] -= f * lu[col*n+j]
			}
		}
	}
	return logDet, true
}
