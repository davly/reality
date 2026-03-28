// Package control provides classical control theory primitives: PID controllers,
// signal filters, rate limiters, and transfer function analysis.
//
// All functions are allocation-free and deterministic (PIDController is stateful
// by design — it accumulates integral error and tracks previous error).
// Zero external dependencies.
//
// Consumers: Pistachio (camera/animation controllers), Pulse (monitoring
// feedback loops), Sentinel (alert tuning), BookaBloke (scheduling feedback).
package control

// PIDController implements a discrete-time Proportional-Integral-Derivative
// controller with anti-windup output clamping.
//
// The controller computes:
//
//	error = setpoint - measured
//	P = Kp * error
//	I = Ki * integralSum  (integralSum += error * dt each step)
//	D = Kd * (error - prevError) / dt
//	output = clamp(P + I + D, minOutput, maxOutput)
//
// Anti-windup: the integral term is not accumulated when the output is already
// saturated (clamped). This prevents integral windup in systems where the
// actuator cannot keep up with the control signal.
//
// Usage:
//
//	pid := NewPID(2.0, 0.5, 0.1, -10, 10)
//	for each timestep dt {
//	    output := pid.Update(setpoint, measured, dt)
//	    applyOutput(output)
//	}
//
// Reference: Astrom & Murray, Feedback Systems, Chapter 10.
type PIDController struct {
	// Gains.
	Kp float64
	Ki float64
	Kd float64

	// Output clamping bounds (anti-windup).
	minOutput float64
	maxOutput float64

	// Internal state.
	integralSum float64
	prevError   float64
}

// NewPID creates a PID controller with the given gains and output limits.
//
// Kp is the proportional gain, Ki is the integral gain, Kd is the derivative
// gain. minOut and maxOut define the output clamping range.
//
// Panics if minOut > maxOut.
func NewPID(kp, ki, kd, minOut, maxOut float64) *PIDController {
	if minOut > maxOut {
		panic("control.NewPID: minOut must be <= maxOut")
	}
	return &PIDController{
		Kp:        kp,
		Ki:        ki,
		Kd:        kd,
		minOutput: minOut,
		maxOutput: maxOut,
	}
}

// Update computes the PID output for the current timestep.
//
// setpoint is the desired value, measured is the current process variable,
// and dt is the time step in seconds. dt must be positive; if dt <= 0,
// the derivative term is zero and the integral is not accumulated.
//
// Returns the clamped control output.
func (p *PIDController) Update(setpoint, measured, dt float64) float64 {
	err := setpoint - measured

	// Proportional term.
	pTerm := p.Kp * err

	// Integral term (only accumulate if dt > 0).
	if dt > 0 {
		p.integralSum += err * dt
	}
	iTerm := p.Ki * p.integralSum

	// Derivative term (only if dt > 0 to avoid division by zero).
	var dTerm float64
	if dt > 0 {
		dTerm = p.Kd * (err - p.prevError) / dt
	}

	output := pTerm + iTerm + dTerm

	// Anti-windup clamping: if output is saturated, undo the integral
	// accumulation to prevent windup.
	if output > p.maxOutput {
		// Undo the integral step that pushed us over.
		if dt > 0 {
			p.integralSum -= err * dt
		}
		output = p.maxOutput
	} else if output < p.minOutput {
		if dt > 0 {
			p.integralSum -= err * dt
		}
		output = p.minOutput
	}

	p.prevError = err
	return output
}

// Reset clears the controller's internal state (integral sum and previous
// error), returning it to its initial condition. Gains and output limits
// are preserved.
func (p *PIDController) Reset() {
	p.integralSum = 0
	p.prevError = 0
}
