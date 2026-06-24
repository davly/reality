package conduit

import "testing"

// TestSampleRateFromEnv pins the wiring of the REALITY_CONDUIT_SAMPLE knob that
// the SampleRate doc promised but the code never read (it was a hard const). The
// override now takes effect; invalid / non-positive / unset values keep the
// 10000 default.
func TestSampleRateFromEnv(t *testing.T) {
	t.Setenv("REALITY_CONDUIT_SAMPLE", "250")
	if got := sampleRateFromEnv(); got != 250 {
		t.Errorf("sampleRateFromEnv() with REALITY_CONDUIT_SAMPLE=250 = %d, want 250", got)
	}
	t.Setenv("REALITY_CONDUIT_SAMPLE", "0") // non-positive -> default
	if got := sampleRateFromEnv(); got != 10000 {
		t.Errorf("sampleRateFromEnv() with =0 = %d, want 10000 default", got)
	}
	t.Setenv("REALITY_CONDUIT_SAMPLE", "notanumber")
	if got := sampleRateFromEnv(); got != 10000 {
		t.Errorf("sampleRateFromEnv() with invalid = %d, want 10000 default", got)
	}
}
