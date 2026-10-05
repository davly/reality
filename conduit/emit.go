// Package conduit provides a fail-silent, non-blocking HTTP shim that
// publishes ForgeEcosystemEvents to the Conduit bus.
//
// Reality is a pure-math foundation library. Instrumenting every function
// call would flood the bus with semantically empty events, so this shim
// supports both unconditional Emit() (for callers that have a meaningful
// observation) and sampled EmitSampled() (one-in-N) for use inside
// hot-path math primitives. The shim is fire-and-forget: if Conduit is
// down, math primitives are unaffected.
//
// Wave 6.A5 (Session 24) — Conduit-emit shim for the engines + foundation
// layer.
package conduit

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"os"
	"strconv"
	"sync/atomic"
	"time"
)

// DefaultURL is the canonical Conduit ingest endpoint. May be overridden
// by the CONDUIT_URL environment variable.
const DefaultURL = "http://localhost:8200/v1/events"

// SampleRate is the 1-in-N sampling rate for hot-path math primitives. Defaults
// to 10000 and can be tuned by setting the REALITY_CONDUIT_SAMPLE env var to a
// positive integer at process start (invalid or unset values keep the default).
var SampleRate = sampleRateFromEnv()

// sampleRateFromEnv reads the REALITY_CONDUIT_SAMPLE override once at init,
// falling back to the 10000 default for an unset, non-integer, or non-positive
// value.
func sampleRateFromEnv() int {
	if s := os.Getenv("REALITY_CONDUIT_SAMPLE"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			return n
		}
	}
	return 10000
}

// Event is the minimal Conduit ingest payload. Field tags MUST match
// store.ForgeLifecycleEvent in the Conduit repo.
type Event struct {
	SituationHash    uint64  `json:"situation_hash"`
	ProjectID        string  `json:"project_id"`
	Domain           string  `json:"domain"`
	OldStatus        string  `json:"old_status,omitempty"`
	NewStatus        string  `json:"new_status"`
	DominanceRate    float64 `json:"dominance_rate,omitempty"`
	ObservationCount int     `json:"observation_count,omitempty"`
	EventType        string  `json:"event_type,omitempty"`
	Payload          string  `json:"payload,omitempty"`
	Timestamp        string  `json:"timestamp,omitempty"`
}

var sampleCounter atomic.Uint64

// Emit publishes an event to Conduit unconditionally. Non-blocking,
// fail-silent.
func Emit(ctx context.Context, e Event) {
	if e.NewStatus == "" {
		e.NewStatus = "OBSERVING"
	}
	if e.ProjectID == "" {
		e.ProjectID = "reality"
	}
	if e.Timestamp == "" {
		e.Timestamp = time.Now().UTC().Format(time.RFC3339)
	}

	go func() { // #nosec G118 -- fire-and-forget by design: the emit must not be cancelled with the caller's context; it has its own 100 ms timeout
		ctx2, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()

		url := os.Getenv("CONDUIT_URL")
		if url == "" {
			url = DefaultURL
		}

		body, err := json.Marshal(e)
		if err != nil {
			return
		}
		req, err := http.NewRequestWithContext(ctx2, http.MethodPost, url, bytes.NewReader(body)) // #nosec G704 -- the URL is operator configuration (environment), not request input
		if err != nil {
			return
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req) // #nosec G704 -- the URL is operator configuration (environment), not request input
		if err != nil || resp == nil {
			return
		}
		_ = resp.Body.Close()
	}()
}

// EmitSampled publishes the event only on every Nth call (default
// SampleRate). Use this from hot-path math primitives to avoid flooding
// the event bus while still keeping usage observable.
//
// A SampleRate of zero or less disables sampled emission. (Zero used to
// panic with an integer divide by zero, and a negative rate silently became
// a huge modulus.)
func EmitSampled(ctx context.Context, e Event) {
	rate := SampleRate
	if rate <= 0 {
		return
	}
	n := sampleCounter.Add(1)
	if n%uint64(rate) != 0 { // #nosec G115 -- rate > 0 checked above
		return
	}
	e.ObservationCount = int(n) // #nosec G115 -- a call counter; it would need 2^63 calls to overflow
	Emit(ctx, e)
}
