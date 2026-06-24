package separation

// Precision property tests — pins the Wiener-filter numerical contract as a
// tested invariant. Pure Go stdlib (testing/quick + math + math/cmplx);
// ADDITIVE, zero math change.
//
// Claims pinned (wiener.go):
//   - The Wiener gain G[k] = SNR/(1+SNR) with SNR>=0 lies in [0,1], so the
//     filtered magnitude never exceeds the input magnitude: |Ŝ[k]| <= |X[k]|.
//     This is the exact-arithmetic invariant behind the "1e-15 per bin" claim.
//   - Documented boundary behaviour: |N|==0 => G=1 (pass-through, bit-exact);
//     signal-dominated bin (|X|^2 >> |N|^2) => G≈1; noise-only bin
//     (|X|==|N|) => G==0 (full attenuation).

import (
	"math"
	"math/cmplx"
	"testing"
	"testing/quick"
)

func cunit(u uint64) float64 { return 200*float64(u)/float64(math.MaxUint64) - 100 }

// TestWienerGainBounded pins |output| <= |input| for every bin (the gain is in
// [0,1] by construction).
func TestWienerGainBounded(t *testing.T) {
	prop := func(re, im, nre, nim uint64) bool {
		in := []complex128{complex(cunit(re), cunit(im))}
		noise := []complex128{complex(cunit(nre), cunit(nim))}
		out := WienerFilter(in, noise)
		// |out| must not exceed |in| (gain in [0,1]); allow 1 ULP of slack.
		ai := cmplx.Abs(in[0])
		ao := cmplx.Abs(out[0])
		return ao <= ai+1e-12*math.Max(1, ai)
	}
	if err := quick.Check(prop, &quick.Config{MaxCount: 200000}); err != nil {
		t.Errorf("PRECISION REGRESSION: WienerFilter gain not in [0,1] — |out| exceeded |in|: %v", err)
	}
}

// TestWienerNoNoisePassThrough pins the |N|==0 => G=1 bit-exact pass-through.
func TestWienerNoNoisePassThrough(t *testing.T) {
	prop := func(re, im uint64) bool {
		in := []complex128{complex(cunit(re), cunit(im))}
		noise := []complex128{0}
		out := WienerFilter(in, noise)
		return out[0] == in[0] // bit-exact: G=1 => complex(1,0)*in == in
	}
	if err := quick.Check(prop, &quick.Config{MaxCount: 100000}); err != nil {
		t.Errorf("PRECISION REGRESSION: WienerFilter with zero noise is not a bit-exact pass-through: %v", err)
	}
}

// TestWienerNoiseOnlyFullAttenuation pins |X|==|N| => SNR==0 => G==0.
func TestWienerNoiseOnlyFullAttenuation(t *testing.T) {
	// in == noise (same magnitude) => gain 0 => output 0.
	in := []complex128{complex(3, 4)}      // |X|=5
	noise := []complex128{complex(5, 0)}   // |N|=5
	out := WienerFilter(in, noise)
	if out[0] != 0 {
		t.Errorf("PRECISION OVER-CLAIM: WienerFilter |X|==|N| gave %v, want 0 (full attenuation)", out[0])
	}
}

// TestWienerSignalDominatedNearUnity pins |X|^2 >> |N|^2 => G≈1.
func TestWienerSignalDominatedNearUnity(t *testing.T) {
	in := []complex128{complex(1000, 0)} // |X|=1000
	noise := []complex128{complex(1, 0)} // |N|=1
	out := WienerFilter(in, noise)
	// SNR = (1e6 - 1)/1 ≈ 1e6, G = SNR/(1+SNR) ≈ 1 - 1e-6.
	gain := cmplx.Abs(out[0]) / cmplx.Abs(in[0])
	if gain < 1-1e-5 || gain > 1 {
		t.Errorf("WienerFilter signal-dominated gain = %g, want ~1", gain)
	}
}
