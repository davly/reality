// Package statespace implements the linear-Gaussian state-space model: the
// Kalman filter (Kalman 1960), the Rauch-Tung-Striebel fixed-interval
// smoother (Rauch, Tung & Striebel 1965), and the univariate local-level
// model of Durbin & Koopman (2012).
//
// The model is
//
//	x_t     = F x_{t-1} + w_t,   w_t ~ N(0, Q)   (state / transition)
//	z_t     = H x_t     + v_t,   v_t ~ N(0, R)   (observation)
//
// with x an n-vector, z an m-vector, F an n×n transition matrix, H an m×n
// observation matrix, Q an n×n process-noise covariance and R an m×m
// observation-noise covariance. All matrices are flat row-major slices; the
// dimensions n, m are passed explicitly. The primitives write their results
// into caller-supplied output buffers (the "output-buffer style" used
// throughout Reality's timeseries substrate — see timeseries/garch) so a
// streaming consumer allocates once and reuses.
//
// # Why this exists in Reality
//
// Reality's own CONTEXT.md:424 promised "RubberDuck gets real stochastic
// calculus. Kalman filters, Markov chains, HMM, ARIMA, PCA — all validated
// against golden files with known precision bounds." PCA shipped; Kalman and
// HMM never did (a grep for Kalman/Viterbi/BaumWelch over the repo returned
// zero implementation hits before this package). Meanwhile RubberDuck runs a
// hand-rolled scalar KalmanFilter.cs whose filtered/smoothed state feeds
// RegimeAdaptiveStrategy and RiskMetricsService — a regime posterior that
// gates live strategy behaviour with no reference implementation to pin it
// to. This package is the reference: the golden-file vectors are the
// language-neutral contract every port (including RubberDuck.Reality) must
// reproduce. See CONTEXT.md Phase-3 plan (lines 375-384).
//
// # Primitives
//
//   - KalmanPredict: time update. Predicts the next state mean and
//     covariance, xOut = F x, POut = F P Fᵀ + Q.
//   - KalmanUpdate: measurement update. Corrects the predicted mean and
//     covariance with observation z. Returns the innovation v = z - H x, the
//     innovation covariance S = H P Hᵀ + R (written to a buffer), and the
//     Gaussian log-likelihood contribution log N(z; H x, S). The covariance
//     update uses the Joseph-stabilised form (I-KH) P (I-KH)ᵀ + K R Kᵀ, which
//     stays symmetric positive-semidefinite under round-off.
//   - Filter: runs Predict/Update forward over an m-dimensional observation
//     series, storing the predicted and filtered means and covariances (the
//     RTS smoother needs the predicted quantities) plus the total
//     log-likelihood.
//   - RTSSmooth: the backward Rauch-Tung-Striebel pass. Given the stored
//     filtered and predicted quantities it computes the smoothed means and
//     covariances that condition on the *entire* series.
//   - LocalLevelFilter / LocalLevelSmooth: the univariate random-walk-plus-
//     noise special case (F=H=1) of Durbin & Koopman (2012) §2, mirroring the
//     scalar API RubberDuck's KalmanFilter.cs exposes. LocalLevelSteadyState
//     returns the closed-form steady-state variance and Kalman gain.
//
// # Determinism, allocations and precision
//
// Every operation is deterministic and free of hidden state. The multivariate
// path inverts the m×m innovation covariance by Gauss-Jordan elimination with
// partial pivoting (adequate for the small measurement dimensions state-space
// models use); scalar paths divide directly. Zero non-stdlib dependencies
// (math only). The golden vectors are pinned at 1e-9; the exact hand-derivable
// cases (steady state, 2-step recursions) hold to 1e-12.
//
// # References
//
//   - Kalman R. E. (1960). A New Approach to Linear Filtering and Prediction
//     Problems. Journal of Basic Engineering 82(1):35-45.
//   - Rauch H. E., Tung F. & Striebel C. T. (1965). Maximum Likelihood
//     Estimates of Linear Dynamic Systems. AIAA Journal 3(8):1445-1450.
//   - Durbin J. & Koopman S. J. (2012). Time Series Analysis by State Space
//     Methods, 2nd ed. Oxford University Press. §2 (local level model; the
//     annual Nile river flow at Aswan is the canonical worked series).
//   - Harvey A. C. (1989). Forecasting, Structural Time Series Models and the
//     Kalman Filter. Cambridge University Press.
package statespace
