package trust

import (
	"math"
	"sort"
)

// MassFunction is a Dempster-Shafer basic probability assignment (BPA) over a
// frame of discernment of FrameSize mutually exclusive, exhaustive elements.
// A subset of the frame is encoded as a bitmask: element i is present iff bit
// i is set, so the full frame Θ is the mask (1<<FrameSize)-1. Masses is the
// mass assigned to each focal subset; unlisted subsets carry zero mass, and
// the empty set (mask 0) MUST carry zero mass. The masses sum to 1.
//
// Unlike a probability distribution, a BPA can commit mass to a NON-singleton
// subset — that is how ignorance is represented explicitly: mass on Θ is mass
// the evidence declines to allocate to any single element.
//
// Reference: Shafer, G. (1976). A Mathematical Theory of Evidence.
type MassFunction struct {
	FrameSize int
	Masses    map[uint]float64
}

// frameMask returns the full-frame Θ bitmask for the given frame size.
func frameMask(frameSize int) uint {
	return (uint(1) << uint(frameSize)) - 1
}

// NewMassFunction constructs and validates a mass function. It copies the
// provided map so later mutation of the caller's map cannot corrupt the BPA.
// It returns ErrInvalidMass if the frame size is non-positive, any mass is
// negative or NaN, any focal set is the empty set or references an element
// outside the frame, or the masses do not sum to 1 within additivityTol.
func NewMassFunction(frameSize int, masses map[uint]float64) (MassFunction, error) {
	if frameSize <= 0 {
		return MassFunction{}, ErrInvalidMass
	}
	full := frameMask(frameSize)
	cp := make(map[uint]float64, len(masses))
	var sum float64
	for set, m := range masses {
		if math.IsNaN(m) || m < 0 {
			return MassFunction{}, ErrInvalidMass
		}
		if set == 0 || set > full {
			// Empty set carries no mass in a normalised BPA; a set with
			// bits outside the frame is not a subset of Θ.
			return MassFunction{}, ErrInvalidMass
		}
		if m == 0 {
			continue // drop explicit zeros; they are the default
		}
		cp[set] = m
		sum += m
	}
	if math.Abs(sum-1) > additivityTol {
		return MassFunction{}, ErrInvalidMass
	}
	return MassFunction{FrameSize: frameSize, Masses: cp}, nil
}

// Belief returns Bel(a), the total mass that necessarily supports the
// hypothesis subset a — the sum of masses of every focal set fully contained
// in a. It is the LOWER probability bound on a.
//
//	Bel(a) = Σ_{b ⊆ a, b ≠ ∅} m(b)
//
// Reference: Shafer (1976), §2.
func (mf MassFunction) Belief(a uint) float64 {
	var bel float64
	for set, m := range mf.Masses {
		if set != 0 && set&a == set { // set ⊆ a
			bel += m
		}
	}
	return bel
}

// Plausibility returns Pl(a), the total mass that could POSSIBLY support the
// hypothesis subset a — the sum of masses of every focal set that intersects
// a. It is the UPPER probability bound on a, and Pl(a) = 1 − Bel(¬a).
//
//	Pl(a) = Σ_{b ∩ a ≠ ∅} m(b)
//
// The interval [Bel(a), Pl(a)] is the evidential range a probability collapses
// away; its width is the uncertainty about a.
//
// Reference: Shafer (1976), §2.
func (mf MassFunction) Plausibility(a uint) float64 {
	var pl float64
	for set, m := range mf.Masses {
		if set&a != 0 {
			pl += m
		}
	}
	return pl
}

// sortedSets returns the focal-set keys in ascending order, for deterministic
// iteration (map ranges are randomised in Go).
func (mf MassFunction) sortedSets() []uint {
	keys := make([]uint, 0, len(mf.Masses))
	for set := range mf.Masses {
		keys = append(keys, set)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return keys
}

// conflictProducts computes, for two mass functions over the same frame, the
// raw combined mass on every non-empty intersection AND the conflict
// coefficient K (the total product mass whose intersection is empty). This is
// the shared core of both DempsterCombine and YagerCombine — the ONLY
// difference between the two rules is what they do with K.
//
//	K = Σ_{a ∩ b = ∅} m1(a)·m2(b)
//
// Deterministic: iterates focal sets in sorted order.
func conflictProducts(m1, m2 MassFunction) (raw map[uint]float64, k float64) {
	raw = make(map[uint]float64)
	for _, a := range m1.sortedSets() {
		for _, b := range m2.sortedSets() {
			inter := a & b
			prod := m1.Masses[a] * m2.Masses[b]
			if inter == 0 {
				k += prod
				continue
			}
			raw[inter] += prod
		}
	}
	return raw, k
}

// DempsterCombine combines two independent bodies of evidence over a shared
// frame with Dempster's rule of combination, returning the combined mass
// function AND the conflict coefficient K as a FIRST-CLASS output.
//
// Dempster's rule redistributes the conflict mass K by normalising every
// combined mass by 1/(1−K):
//
//	m(c) = (1/(1−K)) · Σ_{a ∩ b = c} m1(a)·m2(b)   for c ≠ ∅
//	K    = Σ_{a ∩ b = ∅} m1(a)·m2(b)
//
// The normalisation is exactly the step Zadeh's (1984) counterexample warns
// about: high conflict can be renormalised away into unwarranted certainty
// (see the package test). This function NEVER hides that — K is returned so
// the caller can gate on it (e.g. refuse to trust a combination whose
// K > 0.5). It returns ErrTotalConflict when K = 1 (1−K = 0): the combined
// mass is undefined and the returned K still reports the total contradiction.
//
// Returns ErrFrameMismatch if the two frames differ in size.
//
// References: Dempster (1968); Shafer (1976), §3; Zadeh (1984).
func DempsterCombine(m1, m2 MassFunction) (MassFunction, float64, error) {
	if m1.FrameSize != m2.FrameSize {
		return MassFunction{}, 0, ErrFrameMismatch
	}
	raw, k := conflictProducts(m1, m2)
	oneMinusK := 1 - k
	if oneMinusK <= 0 {
		return MassFunction{}, k, ErrTotalConflict
	}
	combined := make(map[uint]float64, len(raw))
	for set, prod := range raw {
		combined[set] = prod / oneMinusK
	}
	return MassFunction{FrameSize: m1.FrameSize, Masses: combined}, k, nil
}

// YagerCombine combines two bodies of evidence with Yager's (1987) rule: the
// HONEST-degradation alternative to Dempster's. Instead of normalising the
// conflict mass K away, Yager reassigns it to the whole frame Θ (ground
// probability of ignorance), so conflict surfaces as uncertainty rather than
// as spurious certainty:
//
//	q(c) = Σ_{a ∩ b = c} m1(a)·m2(b)   for c ≠ ∅
//	m(c) = q(c)                        for c ≠ Θ
//	m(Θ) = q(Θ) + K
//
// No division is performed, so the result is always defined (there is no
// total-conflict failure), and the masses still sum to 1. K is returned
// alongside so the caller sees exactly how much of the frame-mass came from
// conflict. On the Zadeh counterexample this preserves Bel(tumour)=0.0001
// with Pl=1 instead of minting m(tumour)=1.
//
// Returns ErrFrameMismatch if the two frames differ in size.
//
// Reference: Yager, R. R. (1987). On the Dempster-Shafer framework and new
// combination rules. Information Sciences 41(2): 93-137.
func YagerCombine(m1, m2 MassFunction) (MassFunction, float64, error) {
	if m1.FrameSize != m2.FrameSize {
		return MassFunction{}, 0, ErrFrameMismatch
	}
	raw, k := conflictProducts(m1, m2)
	full := frameMask(m1.FrameSize)
	combined := make(map[uint]float64, len(raw)+1)
	for set, prod := range raw {
		combined[set] = prod
	}
	combined[full] += k // conflict → ignorance
	return MassFunction{FrameSize: m1.FrameSize, Masses: combined}, k, nil
}

// ToBinaryMass expresses a binomial opinion as a Dempster-Shafer mass
// function on the 2-element frame {x, ¬x}: element 0 (mask 1) is x, element 1
// (mask 2) is ¬x, and Θ (mask 3) is the shared ignorance. Belief maps to
// m({x}), disbelief to m({¬x}), and uncertainty to m(Θ) — the exact
// equivalence between binomial opinions and binary BPAs (Jøsang §3.5). Note
// the base rate A has no DS counterpart and is dropped by this direction.
//
// Reference: Jøsang, A. (2016). Subjective Logic, §3.5 (opinion ↔ belief
// function correspondence).
func (o Opinion) ToBinaryMass() MassFunction {
	masses := make(map[uint]float64, 3)
	if o.B > 0 {
		masses[0b01] = o.B
	}
	if o.D > 0 {
		masses[0b10] = o.D
	}
	if o.U > 0 {
		masses[0b11] = o.U
	}
	return MassFunction{FrameSize: 2, Masses: masses}
}

// OpinionFromBinaryMass is the inverse of ToBinaryMass: it reads a mass
// function on a 2-element frame back into a binomial opinion, taking m({x})
// as belief, m({¬x}) as disbelief and m(Θ) as uncertainty. Because a BPA
// carries no base rate, the recovered opinion uses the neutral base rate
// a=0.5; set the field afterwards if a different prior applies. It returns
// ErrFrameMismatch if the mass function is not over a 2-element frame.
//
// Reference: Jøsang, A. (2016). Subjective Logic, §3.5.
func OpinionFromBinaryMass(mf MassFunction) (Opinion, error) {
	if mf.FrameSize != 2 {
		return Opinion{}, ErrFrameMismatch
	}
	return Opinion{
		B: mf.Masses[0b01],
		D: mf.Masses[0b10],
		U: mf.Masses[0b11],
		A: 0.5,
	}, nil
}
