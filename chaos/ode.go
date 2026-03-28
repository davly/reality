// Package chaos provides dynamical systems primitives: ODE solvers, classic
// chaotic and ecological systems, and chaos analysis tools (Lyapunov exponents,
// bifurcation diagrams, recurrence plots).
//
// All functions are deterministic, use only the Go standard library, and target
// zero heap allocations in hot paths (ODE stepping functions write into
// caller-provided slices).
//
// ODE solvers accept derivative functions of the form:
//
//	func(t float64, y []float64, dydt []float64)
//
// where t is the current time, y is the current state vector, and dydt is the
// output slice to write derivatives into (pre-allocated by the caller).
//
// Consumed by: Pistachio (particle/NPC simulation), Pulse (trend modeling),
// Oracle (dynamical prediction), Muse (game physics), Horizon (forecasting).
package chaos

// RK4Step performs a single Runge-Kutta 4th order integration step.
//
// f is the derivative function, t is the current time, y is the current state
// vector, dt is the time step, and out is the pre-allocated output slice for
// the new state. out must have len(out) >= len(y).
//
// The classic RK4 method computes:
//
//	k1 = f(t, y)
//	k2 = f(t + dt/2, y + dt/2 * k1)
//	k3 = f(t + dt/2, y + dt/2 * k2)
//	k4 = f(t + dt, y + dt * k3)
//	y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
//
// RK4Step allocates temporary k-vectors internally. For truly allocation-free
// usage in tight loops, callers should implement the method inline.
func RK4Step(f func(t float64, y []float64, dydt []float64), t float64, y []float64, dt float64, out []float64) {
	n := len(y)
	k1 := make([]float64, n)
	k2 := make([]float64, n)
	k3 := make([]float64, n)
	k4 := make([]float64, n)
	tmp := make([]float64, n)

	// k1 = f(t, y)
	f(t, y, k1)

	// k2 = f(t + dt/2, y + dt/2 * k1)
	for i := 0; i < n; i++ {
		tmp[i] = y[i] + 0.5*dt*k1[i]
	}
	f(t+0.5*dt, tmp, k2)

	// k3 = f(t + dt/2, y + dt/2 * k2)
	for i := 0; i < n; i++ {
		tmp[i] = y[i] + 0.5*dt*k2[i]
	}
	f(t+0.5*dt, tmp, k3)

	// k4 = f(t + dt, y + dt * k3)
	for i := 0; i < n; i++ {
		tmp[i] = y[i] + dt*k3[i]
	}
	f(t+dt, tmp, k4)

	// y_new = y + (dt/6)(k1 + 2*k2 + 2*k3 + k4)
	for i := 0; i < n; i++ {
		out[i] = y[i] + (dt/6.0)*(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i])
	}
}

// EulerStep performs a single forward Euler integration step.
//
// f is the derivative function, t is the current time, y is the current state
// vector, dt is the time step, and out is the pre-allocated output slice for
// the new state. out must have len(out) >= len(y).
//
// Forward Euler: y_new = y + dt * f(t, y)
//
// First-order accurate. Useful as a baseline or for stiff-system comparison.
func EulerStep(f func(t float64, y []float64, dydt []float64), t float64, y []float64, dt float64, out []float64) {
	n := len(y)
	dydt := make([]float64, n)
	f(t, y, dydt)
	for i := 0; i < n; i++ {
		out[i] = y[i] + dt*dydt[i]
	}
}

// SolveODE integrates an ODE system from t0 to tEnd using RK4, returning the
// full trajectory as a slice of state vectors.
//
// f is the derivative function, y0 is the initial state, t0 and tEnd define
// the integration interval, and dt is the time step.
//
// The returned trajectory includes the initial state y0 at index 0. Each
// subsequent entry is the state at t0 + i*dt. The number of entries is
// floor((tEnd - t0) / dt) + 1.
//
// Returns nil if dt <= 0 or tEnd < t0.
func SolveODE(f func(t float64, y []float64, dydt []float64), y0 []float64, t0, tEnd, dt float64) [][]float64 {
	if dt <= 0 || tEnd < t0 {
		return nil
	}

	n := len(y0)
	steps := int((tEnd - t0) / dt)
	trajectory := make([][]float64, 0, steps+1)

	// Copy initial state.
	state := make([]float64, n)
	copy(state, y0)
	row := make([]float64, n)
	copy(row, state)
	trajectory = append(trajectory, row)

	t := t0
	next := make([]float64, n)
	for i := 0; i < steps; i++ {
		RK4Step(f, t, state, dt, next)
		t += dt

		row = make([]float64, n)
		copy(row, next)
		trajectory = append(trajectory, row)

		// Swap state and next to avoid extra copy.
		state, next = next, state
	}

	return trajectory
}
