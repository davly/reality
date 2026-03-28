package chaos

// LorenzSystem returns a derivative function for the Lorenz attractor:
//
//	dx/dt = sigma * (y - x)
//	dy/dt = x * (rho - z) - y
//	dz/dt = x * y - beta * z
//
// Standard parameters: sigma=10, rho=28, beta=8/3.
// State vector: y = [x, y, z].
//
// The Lorenz system is the canonical example of deterministic chaos. Small
// differences in initial conditions lead to exponentially diverging trajectories
// (sensitive dependence on initial conditions, or "the butterfly effect").
//
// Reference: Lorenz, E. N. (1963). "Deterministic nonperiodic flow."
func LorenzSystem(sigma, rho, beta float64) func(t float64, y, dydt []float64) {
	return func(t float64, y, dydt []float64) {
		x, yy, z := y[0], y[1], y[2]
		dydt[0] = sigma * (yy - x)
		dydt[1] = x*(rho-z) - yy
		dydt[2] = x*yy - beta*z
	}
}

// RosslerSystem returns a derivative function for the Rossler attractor:
//
//	dx/dt = -y - z
//	dy/dt = x + a*y
//	dz/dt = b + z*(x - c)
//
// Standard parameters: a=0.2, b=0.2, c=5.7.
// State vector: y = [x, y, z].
//
// The Rossler system produces a simpler chaotic attractor than Lorenz, with a
// single spiral band and a folding mechanism. Useful for studying period-doubling
// routes to chaos.
//
// Reference: Rossler, O. E. (1976). "An equation for continuous chaos."
func RosslerSystem(a, b, c float64) func(t float64, y, dydt []float64) {
	return func(t float64, y, dydt []float64) {
		x, yy, z := y[0], y[1], y[2]
		dydt[0] = -yy - z
		dydt[1] = x + a*yy
		dydt[2] = b + z*(x-c)
	}
}

// LotkaVolterra returns a derivative function for the Lotka-Volterra
// predator-prey model:
//
//	dx/dt = alpha*x - beta*x*y     (prey growth - predation)
//	dy/dt = delta*x*y - gamma*y    (predator growth - death)
//
// Parameters:
//   - alpha: prey birth rate
//   - beta:  predation rate
//   - delta: predator birth efficiency
//   - gamma: predator death rate
//
// State vector: y = [prey, predator].
//
// The system produces oscillating predator-prey populations with a conserved
// quantity (Hamiltonian): H = delta*x - gamma*ln(x) + beta*y - alpha*ln(y).
//
// Reference: Lotka (1925), Volterra (1926).
func LotkaVolterra(alpha, beta, delta, gamma float64) func(t float64, y, dydt []float64) {
	return func(t float64, y, dydt []float64) {
		prey, pred := y[0], y[1]
		dydt[0] = alpha*prey - beta*prey*pred
		dydt[1] = delta*prey*pred - gamma*pred
	}
}

// SIRModel returns a derivative function for the SIR epidemic model:
//
//	dS/dt = -beta * S * I
//	dI/dt = beta * S * I - gamma * I
//	dR/dt = gamma * I
//
// Parameters:
//   - beta:  transmission rate (contact rate * probability of transmission)
//   - gamma: recovery rate (1/gamma = average infectious period)
//
// State vector: y = [S, I, R] where S + I + R = N (constant).
// S, I, R are proportions if normalized to N=1, or absolute counts otherwise.
//
// The basic reproduction number R_0 = beta/gamma determines epidemic behavior:
//   - R_0 > 1: epidemic grows initially
//   - R_0 < 1: epidemic dies out
//
// Reference: Kermack & McKendrick (1927).
func SIRModel(beta, gamma float64) func(t float64, y, dydt []float64) {
	return func(t float64, y, dydt []float64) {
		s, i := y[0], y[1]
		dydt[0] = -beta * s * i
		dydt[1] = beta*s*i - gamma*i
		dydt[2] = gamma * i
	}
}

// VanDerPol returns a derivative function for the Van der Pol oscillator:
//
//	dx/dt = y
//	dy/dt = mu*(1 - x^2)*y - x
//
// The parameter mu controls nonlinearity:
//   - mu = 0: simple harmonic oscillator
//   - mu > 0: limit cycle oscillator with relaxation oscillations
//   - Large mu: strongly nonlinear relaxation oscillations
//
// State vector: y = [x, dx/dt].
//
// Reference: Van der Pol (1926).
func VanDerPol(mu float64) func(t float64, y, dydt []float64) {
	return func(t float64, y, dydt []float64) {
		x, v := y[0], y[1]
		dydt[0] = v
		dydt[1] = mu*(1-x*x)*v - x
	}
}

// LogisticMap computes one iteration of the logistic map:
//
//	x_{n+1} = r * x_n * (1 - x_n)
//
// The logistic map is the simplest mathematical model exhibiting chaos.
//   - r < 1:    x converges to 0
//   - 1 < r < 3: x converges to a fixed point (r-1)/r
//   - 3 < r < 3.57: period-doubling cascade (period 2, 4, 8, ...)
//   - r ≈ 3.57: onset of chaos
//   - r = 4:    fully chaotic, Lyapunov exponent = ln(2)
//
// x must be in [0, 1] and r in [0, 4] for bounded dynamics.
func LogisticMap(r, x float64) float64 {
	return r * x * (1 - x)
}

// GameOfLife advances one step of Conway's Game of Life on a boolean grid.
//
// Rules for each cell:
//  1. Any live cell with 2 or 3 live neighbors survives.
//  2. Any dead cell with exactly 3 live neighbors becomes alive.
//  3. All other cells die or remain dead.
//
// grid is the current state, rows and cols are grid dimensions, and out is the
// pre-allocated output grid. grid and out must both have dimensions [rows][cols].
// Boundary conditions: the grid wraps around (torus topology).
func GameOfLife(grid [][]bool, rows, cols int, out [][]bool) {
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			neighbors := 0
			for dr := -1; dr <= 1; dr++ {
				for dc := -1; dc <= 1; dc++ {
					if dr == 0 && dc == 0 {
						continue
					}
					nr := (r + dr + rows) % rows
					nc := (c + dc + cols) % cols
					if grid[nr][nc] {
						neighbors++
					}
				}
			}
			if grid[r][c] {
				out[r][c] = neighbors == 2 || neighbors == 3
			} else {
				out[r][c] = neighbors == 3
			}
		}
	}
}
