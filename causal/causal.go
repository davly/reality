// Package causal estimates causal effects from OBSERVATIONAL data using
// Pearl's back-door adjustment. It composes the identification layer in
// reality/graph (graph.BackdoorAdjustmentSet) to first DECIDE which variables
// must be adjusted for, then computes the adjusted average treatment effect
// (ATE) by stratifying the observed data on that adjustment set.
//
// The problem it solves
//
// In observational data the naive association E[Y|X=1] - E[Y|X=0] is generally
// a BIASED estimate of the causal effect of X on Y, because a confounder Z may
// influence both the treatment X and the outcome Y (a "back-door" path
// X <- Z -> Y). The most striking symptom is Simpson's paradox, where the naive
// association can have the OPPOSITE sign of the true causal effect.
//
// The back-door criterion (Pearl) tells us when, and on which set Z, adjustment
// removes that confounding bias: Z is admissible when (1) no node in Z is a
// descendant of the treatment, and (2) Z blocks every back-door path from
// treatment to outcome. Given such a Z, the back-door adjustment formula
// identifies the interventional (causal) mean E[Y | do(X=x)] from purely
// observational quantities:
//
//	E[Y | do(X=x)] = Σ_z  E[Y | X=x, Z=z] · P(Z=z)
//
// and the average treatment effect is
//
//	ATE = E[Y | do(X=1)] - E[Y | do(X=0)]
//	    = Σ_z ( E[Y | X=1, Z=z] - E[Y | X=0, Z=z] ) · P(Z=z).
//
// This package implements exactly that estimator over BINARY variables (values
// 0/1), with the adjustment set obtained — not re-derived — from
// graph.BackdoorAdjustmentSet. When the effect is NOT identifiable by the
// back-door criterion, it refuses to guess and reports NotIdentifiable.
//
// Scope and honesty
//
//   - Identification is delegated to reality/graph. This package contributes the
//     ESTIMATION step (the adjustment-formula arithmetic over data); it does not
//     reimplement d-separation or the back-door search.
//   - Positivity: a stratum z is usable only if it contains BOTH a treated
//     (X=1) and an untreated (X=0) observation. Strata that lack one arm cannot
//     contribute a within-stratum contrast, so they are DROPPED from the
//     weighted sum and the mass they carried is reported (PositivityDroppedMass)
//     so the caller can judge how much of the population the estimate covers.
//     This is the honest "drop and report" handling rather than silently
//     imputing an unobserved arm.
//   - Variables are binary; outcomes are treated as 0/1 so a cell mean is the
//     fraction of Y=1 in that cell (a probability).
//
// Reference:
//
//	Pearl, J. (2009). Causality: Models, Reasoning, and Inference, 2nd ed.
//	Cambridge University Press. (Back-door criterion, Theorem 3.3.2; the
//	adjustment formula, Eq. 3.19.)
package causal

import (
	"errors"
	"sort"
	"strings"

	"github.com/davly/reality/graph"
)

// An Observation is a single record over binary variables: each variable name
// maps to its observed value, which must be 0 or 1. A dataset is []Observation.
type Observation = map[string]int

// Result is the outcome of a back-door ATE estimation.
//
// When Identifiable is false, the effect could not be identified by the
// back-door criterion (graph.BackdoorAdjustmentSet returned ok==false); in that
// case the adjusted fields (BackdoorATE, AdjustedOutcome1, AdjustedOutcome0,
// AdjustmentSet, PositivityDroppedMass) are not meaningful and are left zero.
// NaiveATE is still reported when computable, because it is a property of the
// data alone and does not depend on identification.
type Result struct {
	// Identifiable reports whether the causal effect of treatment on outcome is
	// identifiable from observational data via the back-door criterion in the
	// supplied DAG. If false, no adjusted estimate is produced.
	Identifiable bool

	// BackdoorATE is the back-door-adjusted average treatment effect,
	// Σ_z (E[Y|X=1,Z=z] - E[Y|X=0,Z=z]) · P(Z=z), over strata with positivity
	// support. Equals AdjustedOutcome1 - AdjustedOutcome0. Valid only when
	// Identifiable is true.
	BackdoorATE float64

	// AdjustedOutcome1 is E[Y | do(X=1)] = Σ_z E[Y|X=1,Z=z] · P(Z=z).
	// Valid only when Identifiable is true.
	AdjustedOutcome1 float64

	// AdjustedOutcome0 is E[Y | do(X=0)] = Σ_z E[Y|X=0,Z=z] · P(Z=z).
	// Valid only when Identifiable is true.
	AdjustedOutcome0 float64

	// NaiveATE is the unadjusted association E[Y|X=1] - E[Y|X=0]. This is the
	// confounded estimate; comparing it to BackdoorATE shows the adjustment
	// effect (in Simpson's paradox the two have opposite signs).
	NaiveATE float64

	// AdjustmentSet is the back-door set Z used, as returned by
	// graph.BackdoorAdjustmentSet (sorted). Empty (length 0) means no adjustment
	// was needed (no confounding), in which case BackdoorATE == NaiveATE.
	AdjustmentSet []string

	// PositivityDroppedMass is the fraction of observations [0,1] that fell into
	// strata lacking either a treated or an untreated arm and were therefore
	// excluded from the adjusted sum. Zero means full positivity support. A
	// large value means the adjusted estimate covers only part of the
	// population and should be read with caution.
	PositivityDroppedMass float64
}

// ErrInsufficientData is returned when the naive contrast itself cannot be
// computed because the data lacks any treated or any untreated observation
// (E[Y|X=1] or E[Y|X=0] is undefined). The DAG-level effect may still be
// identifiable, but no estimate can be produced from this data.
var ErrInsufficientData = errors.New("causal: data has no treated (X=1) or no untreated (X=0) observations")

// BackdoorATE estimates the average treatment effect of treatment on outcome
// from observational data via Pearl's back-door adjustment.
//
// It first asks reality/graph for an admissible adjustment set Z using the
// back-door criterion. If none exists (the effect is not identifiable by the
// back-door route), it returns a Result with Identifiable==false and does NOT
// guess an effect — only NaiveATE (a data-only quantity) is filled in. If Z is
// admissible, it computes the adjusted ATE by stratifying on Z:
//
//	ATE = Σ_z ( E[Y|X=1,Z=z] - E[Y|X=0,Z=z] ) · P(Z=z)
//
// summing only over strata that have BOTH arms (positivity); the dropped mass
// is reported in the Result.
//
// edges:     the causal DAG; graph.Edge{U,V} means U -> V (U directly causes V).
// treatment: the binary treatment variable name (the "X" / do-variable).
// outcome:   the binary outcome variable name (the "Y").
// data:      observations over binary variables (values 0/1).
//
// Errors:
//   - ErrInsufficientData if the data contains no X=1 observation or no X=0
//     observation (the naive contrast is undefined). Note: this is checked
//     before identification, since with no contrast at all there is nothing to
//     estimate either way.
//
// Time complexity: O(|data| · |Z|) plus the O(|V|+|E|) identification step.
func BackdoorATE(edges []graph.Edge, treatment, outcome string, data []Observation) (Result, error) {
	// Naive (unadjusted) association — a property of the data alone.
	naive, ok := naiveATE(treatment, outcome, data)
	if !ok {
		return Result{}, ErrInsufficientData
	}

	// Identification: delegate to the graph layer. Do NOT re-derive.
	z, identifiable := graph.BackdoorAdjustmentSet(edges, treatment, outcome)
	if !identifiable {
		// Refuse to guess. Report only the data-only naive association.
		return Result{
			Identifiable: false,
			NaiveATE:     naive,
		}, nil
	}

	// Adjusted estimate via the back-door formula over strata of Z.
	out1, out0, droppedMass := adjustedOutcomes(z, treatment, outcome, data)

	return Result{
		Identifiable:          true,
		BackdoorATE:           out1 - out0,
		AdjustedOutcome1:      out1,
		AdjustedOutcome0:      out0,
		NaiveATE:              naive,
		AdjustmentSet:         z,
		PositivityDroppedMass: droppedMass,
	}, nil
}

// AdjustedOutcome returns the back-door-adjusted expected outcome under the
// intervention do(X=doX):
//
//	E[Y | do(X=doX)] = Σ_z E[Y | X=doX, Z=z] · P(Z=z)
//
// summing over strata of the admissible set Z that have positivity support for
// the requested arm. It composes graph.BackdoorAdjustmentSet for identification.
//
// The boolean result is true when the effect is identifiable (an admissible Z
// exists). When false, the float result is 0 and should be ignored. doX should
// be 0 or 1; any non-zero value is treated as the treated arm (X=1).
//
// Relationship to BackdoorATE: when identifiable and under full positivity,
// AdjustedOutcome(1) - AdjustedOutcome(0) equals the Result.BackdoorATE from
// BackdoorATE. (They can differ only via which strata satisfy positivity for a
// single arm vs. both arms; see the positivity note in the package doc.)
func AdjustedOutcome(edges []graph.Edge, treatment, outcome string, doX int, data []Observation) (float64, bool) {
	z, identifiable := graph.BackdoorAdjustmentSet(edges, treatment, outcome)
	if !identifiable {
		return 0, false
	}
	arm := 0
	if doX != 0 {
		arm = 1
	}
	val, _ := adjustedOutcomeOneArm(z, treatment, outcome, arm, data)
	return val, true
}

// naiveATE computes E[Y|X=1] - E[Y|X=0] from the data. The boolean is false if
// either arm is empty (the contrast is undefined).
func naiveATE(treatment, outcome string, data []Observation) (float64, bool) {
	var sum1, sum0 float64
	var n1, n0 int
	for _, obs := range data {
		x := binary(obs[treatment])
		y := binary(obs[outcome])
		if x == 1 {
			sum1 += float64(y)
			n1++
		} else {
			sum0 += float64(y)
			n0++
		}
	}
	if n1 == 0 || n0 == 0 {
		return 0, false
	}
	return sum1/float64(n1) - sum0/float64(n0), true
}

// stratumCounts holds, for one value-combination z of the adjustment set, the
// per-arm outcome sums and counts and the total stratum size.
type stratumCounts struct {
	sum1, sum0 float64 // Σ Y within X=1 / X=0
	n1, n0     int     // count within X=1 / X=0
	total      int     // n1 + n0 (size of the stratum)
}

// tabulate groups the data into strata keyed by the (sorted) values of the
// adjustment variables z, accumulating per-arm outcome sums/counts. It returns
// the strata map and the total number of observations (the denominator for
// P(Z=z)). The key is a deterministic encoding of z's values for this row.
func tabulate(z []string, treatment, outcome string, data []Observation) (map[string]*stratumCounts, int) {
	// Use the adjustment variables in sorted order for a stable key. z from
	// graph.BackdoorAdjustmentSet is already sorted, but sort defensively so the
	// key is independent of caller ordering.
	zs := append([]string(nil), z...)
	sort.Strings(zs)

	strata := make(map[string]*stratumCounts)
	total := 0
	var b strings.Builder
	for _, obs := range data {
		b.Reset()
		for i, v := range zs {
			if i > 0 {
				b.WriteByte('|')
			}
			b.WriteString(v)
			b.WriteByte('=')
			if binary(obs[v]) == 1 {
				b.WriteByte('1')
			} else {
				b.WriteByte('0')
			}
		}
		key := b.String()
		sc := strata[key]
		if sc == nil {
			sc = &stratumCounts{}
			strata[key] = sc
		}
		y := float64(binary(obs[outcome]))
		if binary(obs[treatment]) == 1 {
			sc.sum1 += y
			sc.n1++
		} else {
			sc.sum0 += y
			sc.n0++
		}
		sc.total++
		total++
	}
	return strata, total
}

// adjustedOutcomes computes BOTH adjusted potential outcomes
// E[Y|do(X=1)] and E[Y|do(X=0)] and the positivity-dropped mass in one pass,
// summing only over strata that have BOTH arms (so the within-stratum contrast
// is defined). Strata missing an arm are dropped and their mass accumulated.
//
// Returns (out1, out0, droppedMass). With no data, all are 0.
func adjustedOutcomes(z []string, treatment, outcome string, data []Observation) (out1, out0, droppedMass float64) {
	strata, total := tabulate(z, treatment, outcome, data)
	if total == 0 {
		return 0, 0, 0
	}
	denom := float64(total)
	for _, sc := range strata {
		if sc.n1 == 0 || sc.n0 == 0 {
			// Positivity violation for this stratum: cannot form a contrast.
			droppedMass += float64(sc.total) / denom
			continue
		}
		pz := float64(sc.total) / denom
		out1 += (sc.sum1 / float64(sc.n1)) * pz
		out0 += (sc.sum0 / float64(sc.n0)) * pz
	}
	return out1, out0, droppedMass
}

// adjustedOutcomeOneArm computes E[Y|do(X=arm)] = Σ_z E[Y|X=arm,Z=z]·P(Z=z)
// over strata that have support for the requested arm only. Returns the adjusted
// outcome and the mass dropped for lack of that arm. arm must be 0 or 1.
func adjustedOutcomeOneArm(z []string, treatment, outcome string, arm int, data []Observation) (float64, float64) {
	strata, total := tabulate(z, treatment, outcome, data)
	if total == 0 {
		return 0, 0
	}
	denom := float64(total)
	var out, droppedMass float64
	for _, sc := range strata {
		var sum float64
		var n int
		if arm == 1 {
			sum, n = sc.sum1, sc.n1
		} else {
			sum, n = sc.sum0, sc.n0
		}
		if n == 0 {
			droppedMass += float64(sc.total) / denom
			continue
		}
		pz := float64(sc.total) / denom
		out += (sum / float64(n)) * pz
	}
	return out, droppedMass
}

// binary coerces an observed value to 0/1: any non-zero value is treated as 1.
// This keeps the estimator robust to inputs that use, e.g., 2 for "present"
// while preserving the binary semantics documented for Observation.
func binary(v int) int {
	if v != 0 {
		return 1
	}
	return 0
}
