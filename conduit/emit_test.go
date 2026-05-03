package conduit

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// --- Event marshaling --------------------------------------------------

func TestEvent_MarshalsRequiredFieldsOnly(t *testing.T) {
	// Empty event: omitempty should hide all the optional fields, leaving
	// just situation_hash, project_id, domain, new_status.
	e := Event{
		SituationHash: 12345,
		ProjectID:     "test",
		Domain:        "test-domain",
		NewStatus:     "OBSERVING",
	}

	b, err := json.Marshal(e)
	if err != nil {
		t.Fatalf("Marshal err: %v", err)
	}
	got := string(b)

	wantSubstrings := []string{
		`"situation_hash":12345`,
		`"project_id":"test"`,
		`"domain":"test-domain"`,
		`"new_status":"OBSERVING"`,
	}
	for _, w := range wantSubstrings {
		if !contains(got, w) {
			t.Errorf("want %q in JSON, got %s", w, got)
		}
	}
	notWantSubstrings := []string{
		`"old_status"`,
		`"dominance_rate"`,
		`"observation_count"`,
		`"event_type"`,
		`"payload"`,
		`"timestamp"`,
	}
	for _, nw := range notWantSubstrings {
		if contains(got, nw) {
			t.Errorf("did NOT want %q in JSON (omitempty broken), got %s", nw, got)
		}
	}
}

func TestEvent_OptionalFieldsIncludedWhenSet(t *testing.T) {
	e := Event{
		SituationHash:    1,
		ProjectID:        "x",
		Domain:           "y",
		OldStatus:        "PROVISIONAL",
		NewStatus:        "DOMINATED",
		DominanceRate:    0.7,
		ObservationCount: 100,
		EventType:        "test_event",
		Payload:          `{"k":"v"}`,
		Timestamp:        "2026-05-04T12:00:00Z",
	}

	b, err := json.Marshal(e)
	if err != nil {
		t.Fatalf("Marshal err: %v", err)
	}
	got := string(b)

	wantSubstrings := []string{
		`"old_status":"PROVISIONAL"`,
		`"dominance_rate":0.7`,
		`"observation_count":100`,
		`"event_type":"test_event"`,
		`"payload":"{\"k\":\"v\"}"`,
		`"timestamp":"2026-05-04T12:00:00Z"`,
	}
	for _, w := range wantSubstrings {
		if !contains(got, w) {
			t.Errorf("want %q in JSON, got %s", w, got)
		}
	}
}

// --- Constants ---------------------------------------------------------

func TestDefaultURL_IsCanonicalConduitIngest(t *testing.T) {
	if DefaultURL != "http://localhost:8200/v1/events" {
		t.Errorf("DefaultURL drifted: got %q", DefaultURL)
	}
}

func TestSampleRate_IsTenThousand(t *testing.T) {
	// SampleRate is the default 1-in-N for hot-path emit. Pinned at 10k
	// so a future tweak forces a coordinated update with documentation.
	if SampleRate != 10000 {
		t.Errorf("SampleRate drifted: got %d, want 10000", SampleRate)
	}
}

// --- Emit fire-and-forget round-trip ----------------------------------

func TestEmit_PostsEventToConfiguredURL(t *testing.T) {
	// httptest server captures the POST.
	var (
		mu          sync.Mutex
		gotMethod   string
		gotPath     string
		gotBody     []byte
		gotContent  string
		bodyArrived = make(chan struct{}, 1)
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotContent = r.Header.Get("Content-Type")
		buf := make([]byte, r.ContentLength)
		if r.ContentLength > 0 {
			_, _ = r.Body.Read(buf)
		}
		gotBody = buf
		select {
		case bodyArrived <- struct{}{}:
		default:
		}
		w.WriteHeader(http.StatusAccepted)
	}))
	defer srv.Close()

	t.Setenv("CONDUIT_URL", srv.URL)

	e := Event{
		SituationHash: 999,
		ProjectID:     "reality-test",
		Domain:        "test",
		NewStatus:     "OBSERVING",
	}

	Emit(context.Background(), e)

	// Wait up to 500ms for the goroutine to fire.
	select {
	case <-bodyArrived:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("Emit did not deliver to httptest server within 500ms")
	}

	mu.Lock()
	defer mu.Unlock()

	if gotMethod != http.MethodPost {
		t.Errorf("Method = %q, want POST", gotMethod)
	}
	if gotPath == "" {
		t.Error("Path was empty")
	}
	if gotContent != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", gotContent)
	}
	if !contains(string(gotBody), `"situation_hash":999`) {
		t.Errorf("body did not contain situation_hash; got %s", string(gotBody))
	}
}

func TestEmit_FailureIsSwallowedSilently(t *testing.T) {
	// Set CONDUIT_URL to a port that won't accept connections (high port,
	// nothing listening). Emit must return immediately without blocking
	// the caller.
	t.Setenv("CONDUIT_URL", "http://127.0.0.1:1/v1/events")

	done := make(chan struct{})
	go func() {
		Emit(context.Background(), Event{SituationHash: 1, ProjectID: "p", Domain: "d", NewStatus: "OBSERVING"})
		close(done)
	}()

	// Emit should return well within 50ms regardless of network outcome.
	select {
	case <-done:
	case <-time.After(50 * time.Millisecond):
		t.Error("Emit blocked > 50ms; should return immediately (fire-and-forget)")
	}
}

func TestEmit_DefaultsAppliedWhenFieldsMissing(t *testing.T) {
	// Capture body to verify defaults.
	var capturedBody []byte
	bodyArrived := make(chan struct{}, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		buf := make([]byte, r.ContentLength)
		if r.ContentLength > 0 {
			_, _ = r.Body.Read(buf)
		}
		capturedBody = buf
		select {
		case bodyArrived <- struct{}{}:
		default:
		}
		w.WriteHeader(http.StatusAccepted)
	}))
	defer srv.Close()

	t.Setenv("CONDUIT_URL", srv.URL)

	// Only SituationHash + Domain set; NewStatus + ProjectID + Timestamp
	// should be defaulted.
	Emit(context.Background(), Event{SituationHash: 42, Domain: "x"})

	select {
	case <-bodyArrived:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("Emit did not deliver within 500ms")
	}

	body := string(capturedBody)
	if !contains(body, `"new_status":"OBSERVING"`) {
		t.Errorf("default new_status not applied: %s", body)
	}
	if !contains(body, `"project_id":"reality"`) {
		t.Errorf("default project_id not applied: %s", body)
	}
	if !contains(body, `"timestamp"`) {
		t.Errorf("default timestamp not applied: %s", body)
	}
}

// --- EmitSampled -------------------------------------------------------

func TestEmitSampled_OnlyEmitsEveryNthCall(t *testing.T) {
	var posts atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		posts.Add(1)
		w.WriteHeader(http.StatusAccepted)
	}))
	defer srv.Close()

	t.Setenv("CONDUIT_URL", srv.URL)

	// Reset counter via reflection isn't possible; instead, use the
	// fact that SampleRate is 10000 and call > 10000 times.
	ctx := context.Background()
	for i := 0; i < SampleRate*2; i++ {
		EmitSampled(ctx, Event{SituationHash: uint64(i), Domain: "test"})
	}

	// Wait for any in-flight emits to land.
	time.Sleep(200 * time.Millisecond)

	// Should have fired at most 2-3 times (sample-counter is package-global
	// and may have been incremented by other tests, so be lenient).
	count := posts.Load()
	if count < 1 || count > 5 {
		t.Errorf("EmitSampled posted %d times after %d calls; expected 1-5", count, SampleRate*2)
	}
}

// --- helpers -----------------------------------------------------------

func contains(haystack, needle string) bool {
	if len(needle) == 0 {
		return true
	}
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}
