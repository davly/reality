package trust

import "testing"

// TestCumulativeFusion_IsEvidenceAddition reproduces the defining identity of
// Jøsang's cumulative fusion operator (Subjective Logic 2016, §12.3): fusing
// two opinions built from evidence (r1,s1) and (r2,s2) yields exactly the
// opinion built from the SUMMED evidence (r1+r2, s1+s2). This is the golden
// property that makes cumulative fusion the correct answer to "do the engines
// agree, and on how much evidence?" — agreement over more observation is more
// certain.
func TestCumulativeFusion_IsEvidenceAddition(t *testing.T) {
	oA, _ := OpinionFromEvidence(6, 1, 0.5) // denom 9
	oB, _ := OpinionFromEvidence(3, 2, 0.5) // denom 7
	fused := CumulativeFusion(oA, oB)

	// Expected: opinion from (9, 3) → denom 9+3+2 = 14.
	want, _ := OpinionFromEvidence(9, 3, 0.5)
	approx(t, "b", fused.B, want.B, epsAcc)
	approx(t, "d", fused.D, want.D, epsAcc)
	approx(t, "u", fused.U, want.U, epsAcc)
	approx(t, "a", fused.A, 0.5, epsAcc)
	approx(t, "b+d+u", fused.B+fused.D+fused.U, 1.0, epsAcc)

	// Fused uncertainty is strictly SMALLER than either input's — the
	// property flat averaging lacks.
	if fused.U >= oA.U || fused.U >= oB.U {
		t.Errorf("cumulative fusion should reduce uncertainty below both inputs: fused.U=%v oA.U=%v oB.U=%v", fused.U, oA.U, oB.U)
	}
}

// TestCumulativeFusion_VacuousIsIdentity: fusing with a vacuous opinion
// (u=1, no evidence) leaves the informative opinion unchanged — adding zero
// evidence changes nothing.
func TestCumulativeFusion_VacuousIsIdentity(t *testing.T) {
	oA, _ := OpinionFromEvidence(6, 1, 0.5)
	vac := Opinion{B: 0, D: 0, U: 1, A: 0.5}
	fused := CumulativeFusion(oA, vac)
	approx(t, "b", fused.B, oA.B, epsAcc)
	approx(t, "d", fused.D, oA.D, epsAcc)
	approx(t, "u", fused.U, oA.U, epsAcc)
}

// TestCumulativeFusion_BothDogmatic: two absolute opinions average (the
// equal-weight limit) and stay dogmatic.
func TestCumulativeFusion_BothDogmatic(t *testing.T) {
	oA := Opinion{B: 1, D: 0, U: 0, A: 0.5}
	oB := Opinion{B: 0, D: 1, U: 0, A: 0.5}
	fused := CumulativeFusion(oA, oB)
	approx(t, "b", fused.B, 0.5, eps)
	approx(t, "d", fused.D, 0.5, eps)
	approx(t, "u", fused.U, 0, eps)
}

// TestAveragingFusion_Manual reproduces the averaging operator (Jøsang §12.5)
// on a symmetric pair; the fused uncertainty stays between the inputs rather
// than shrinking below both.
func TestAveragingFusion_Manual(t *testing.T) {
	// Two identical opinions: averaging must return the same opinion.
	o, _ := OpinionFromEvidence(6, 1, 0.5)
	fused := AveragingFusion(o, o)
	approx(t, "b", fused.B, o.B, epsAcc)
	approx(t, "d", fused.D, o.D, epsAcc)
	approx(t, "u", fused.U, o.U, epsAcc)

	// Asymmetric: b=(bA·uB+bB·uA)/(uA+uB), u=2uAuB/(uA+uB).
	oA := Opinion{B: 0.8, D: 0.0, U: 0.2, A: 0.5}
	oB := Opinion{B: 0.0, D: 0.6, U: 0.4, A: 0.5}
	f := AveragingFusion(oA, oB)
	sum := oA.U + oB.U // 0.6
	approx(t, "b", f.B, (0.8*oB.U+0.0*oA.U)/sum, epsAcc)
	approx(t, "d", f.D, (0.0*oB.U+0.6*oA.U)/sum, epsAcc)
	approx(t, "u", f.U, 2*oA.U*oB.U/sum, epsAcc)
	approx(t, "b+d+u", f.B+f.D+f.U, 1.0, epsAcc)
}

// TestFuseAll_OrderIndependent: cumulative fusion is associative/commutative,
// so folding in any order gives the same result; the empty slice yields the
// vacuous opinion (u=1), NOT a minted 1.0.
func TestFuseAll_OrderIndependent(t *testing.T) {
	a, _ := OpinionFromEvidence(4, 1, 0.5)
	b, _ := OpinionFromEvidence(2, 3, 0.5)
	c, _ := OpinionFromEvidence(7, 0, 0.5)

	f1 := FuseAll([]Opinion{a, b, c})
	f2 := FuseAll([]Opinion{c, a, b})
	approx(t, "b order-indep", f1.B, f2.B, epsAcc)
	approx(t, "u order-indep", f1.U, f2.U, epsAcc)

	// Equals evidence sum (13, 4).
	want, _ := OpinionFromEvidence(13, 4, 0.5)
	approx(t, "b", f1.B, want.B, epsAcc)
	approx(t, "u", f1.U, want.U, epsAcc)

	empty := FuseAll(nil)
	approx(t, "empty u", empty.U, 1.0, eps)
	if !empty.IsVacuous() {
		t.Errorf("FuseAll(nil) must be vacuous, got %+v", empty)
	}
}

// TestDiscount reproduces trust discounting (Jøsang §14.3). A fully trusted
// source passes through; a distrusted source's assertion decays into
// uncertainty; an untrusted source yields the vacuous opinion.
func TestDiscount(t *testing.T) {
	assertion := Opinion{B: 0.8, D: 0.1, U: 0.1, A: 0.5} // source says x is likely

	// Full trust (p=1): unchanged.
	full := Opinion{B: 1, D: 0, U: 0, A: 0.5}
	d := assertion.Discount(full)
	approx(t, "full.b", d.B, 0.8, epsAcc)
	approx(t, "full.d", d.D, 0.1, epsAcc)
	approx(t, "full.u", d.U, 0.1, epsAcc)

	// No trust (p=0): vacuous.
	none := Opinion{B: 0, D: 1, U: 0, A: 0.5}
	dn := assertion.Discount(none)
	approx(t, "none.b", dn.B, 0, epsAcc)
	approx(t, "none.d", dn.D, 0, epsAcc)
	approx(t, "none.u", dn.U, 1, epsAcc)

	// Half trust (projected p=0.5): belief/disbelief halved, remainder → u.
	half := Opinion{B: 0.5, D: 0.5, U: 0, A: 0.5} // p = 0.5
	dh := assertion.Discount(half)
	approx(t, "half.b", dh.B, 0.4, epsAcc)
	approx(t, "half.d", dh.D, 0.05, epsAcc)
	approx(t, "half.u", dh.U, 1-0.5*(0.8+0.1), epsAcc)
	approx(t, "half b+d+u", dh.B+dh.D+dh.U, 1.0, epsAcc)
}
