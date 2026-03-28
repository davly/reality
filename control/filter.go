package control

// LowPassFilter applies a single-pole exponential smoothing low-pass filter.
//
// Definition: out = alpha * current + (1 - alpha) * prev
//
// alpha controls the cutoff: alpha=1 passes current through unchanged (no
// filtering), alpha=0 holds the previous value (infinite smoothing).
// alpha must be in [0, 1]; values outside this range are clamped.
//
// Zero heap allocations. Stateless — caller maintains prev across calls.
//
// Consumers: Sentinel (metric smoothing), Pistachio (camera damping),
// Pulse (alert threshold smoothing).
func LowPassFilter(prev, current, alpha float64) float64 {
	if alpha < 0 {
		alpha = 0
	}
	if alpha > 1 {
		alpha = 1
	}
	return alpha*current + (1-alpha)*prev
}

// HighPassFilter applies a complementary high-pass filter that extracts the
// rapidly-changing component of a signal.
//
// Definition: out = alpha * (prevFiltered + current - prev)
//
// This is the discrete-time first-difference high-pass filter, the complement
// of the exponential low-pass filter. When alpha is close to 1, more of the
// high-frequency content is preserved.
//
// alpha must be in [0, 1]; values outside this range are clamped.
// Zero heap allocations. Stateless — caller maintains prevFiltered and prev.
//
// Consumers: Pistachio (jitter detection), Sentinel (anomaly extraction).
func HighPassFilter(prevFiltered, prev, current, alpha float64) float64 {
	if alpha < 0 {
		alpha = 0
	}
	if alpha > 1 {
		alpha = 1
	}
	return alpha * (prevFiltered + current - prev)
}

// ComplementaryFilter fuses two sensor readings (typically accelerometer and
// gyroscope) using a weighted blend. The accelerometer is trusted for low-
// frequency (steady-state) orientation, while the gyroscope provides high-
// frequency (transient) accuracy.
//
// Definition: out = alpha * (gyro * dt + accel_component) + (1 - alpha) * accel
//
// In practice this is a first-order complementary filter:
//   out = alpha * (prev_angle + gyro * dt) + (1 - alpha) * accel
//
// The caller must maintain the previous angle estimate and pass it as accel's
// complementary value. For a simplified interface:
//
//   angle = ComplementaryFilter(accel_angle, gyro_rate, alpha, dt)
//
// where accel_angle is the angle derived from the accelerometer alone, and
// gyro_rate is the angular velocity from the gyroscope.
//
// alpha controls the trust split: alpha near 1 trusts the gyroscope more,
// alpha near 0 trusts the accelerometer more. Typical values: 0.93–0.98.
//
// alpha must be in [0, 1]; values outside this range are clamped.
// dt must be positive; if dt <= 0, only the accelerometer reading is used.
// Zero heap allocations.
//
// Consumers: Pistachio (IMU sensor fusion for camera orientation).
func ComplementaryFilter(accel, gyro, alpha, dt float64) float64 {
	if alpha < 0 {
		alpha = 0
	}
	if alpha > 1 {
		alpha = 1
	}
	if dt <= 0 {
		return accel
	}
	return alpha*(accel+gyro*dt) + (1-alpha)*accel
}

// RateLimiter constrains how fast a value can change per timestep.
//
// Given the current value and a target value, returns a value that moves
// toward target but does not exceed maxRate * dt change from current.
//
// Definition:
//   delta = target - current
//   if |delta| <= maxRate * dt: return target
//   else: return current + sign(delta) * maxRate * dt
//
// maxRate must be positive (the maximum rate of change per second).
// dt must be positive; if dt <= 0, current is returned unchanged.
// Zero heap allocations.
//
// Consumers: Pistachio (smooth camera transitions), BookaBloke (slot
// availability ramping), Pulse (alert threshold ramping).
func RateLimiter(current, target, maxRate, dt float64) float64 {
	if dt <= 0 || maxRate <= 0 {
		return current
	}
	delta := target - current
	maxDelta := maxRate * dt
	if delta > maxDelta {
		return current + maxDelta
	}
	if delta < -maxDelta {
		return current - maxDelta
	}
	return target
}
