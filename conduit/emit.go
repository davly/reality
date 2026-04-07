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
	"sync/atomic"
	"time"
)

// DefaultURL is the canonical Conduit ingest endpoint. May be overridden
// by the CONDUIT_URL environment variable.
const DefaultURL = "http://localhost:8200/v1/events"

// SampleRate is the default 1-in-N sampling rate for hot-path math
// primitives. Can be tuned by setting the REALITY_CONDUIT_SAMPLE env var.
const SampleRate = 10000

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

	go func() {
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
		req, err := http.NewRequestWithContext(ctx2, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil || resp == nil {
			return
		}
		_ = resp.Body.Close()
	}()
}

// EmitSampled publishes the event only on every Nth call (default
// SampleRate). Use this from hot-path math primitives to avoid flooding
// the Conduit bus while still keeping cross-pollination observable.
func EmitSampled(ctx context.Context, e Event) {
	n := sampleCounter.Add(1)
	if n%SampleRate != 0 {
		return
	}
	e.ObservationCount = int(n)
	Emit(ctx, e)
}
