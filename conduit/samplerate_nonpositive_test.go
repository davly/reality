package conduit

import (
	"context"
	"testing"
)

// TestEmitSampled_NonPositiveRateIsANoOp pins the guard for an invalid
// sampling rate. Before it, SampleRate = 0 made EmitSampled panic with an
// integer divide by zero, and a negative rate was converted to a huge uint64
// modulus. Both must now be silent no-ops (no panic, no emission).
func TestEmitSampled_NonPositiveRateIsANoOp(t *testing.T) {
	old := SampleRate
	defer func() { SampleRate = old }()
	before := sampleCounter.Load()
	for _, rate := range []int{0, -1, -10000} {
		SampleRate = rate
		EmitSampled(context.Background(), Event{})
	}
	if after := sampleCounter.Load(); after != before {
		t.Fatalf("a non-positive rate must not count or emit: counter moved %d -> %d", before, after)
	}
}
