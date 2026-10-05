package conduit

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"runtime"
	"sync/atomic"
	"testing"
	"time"
)

// TestEmit_DestinationResolvedAtCallTime pins that an event goes to the
// CONDUIT_URL in force when Emit is called. Emit used to read the variable
// inside its goroutine, so a change made right after the call redirected the
// event (and one test's late goroutine delivered into the next test's
// server). With GOMAXPROCS(1) the spawned goroutine cannot run before the
// variable is switched, so the old behaviour fails deterministically.
func TestEmit_DestinationResolvedAtCallTime(t *testing.T) {
	var hitsA, hitsB atomic.Int64
	srvA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hitsA.Add(1)
		w.WriteHeader(http.StatusAccepted)
	}))
	defer srvA.Close()
	srvB := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hitsB.Add(1)
		w.WriteHeader(http.StatusAccepted)
	}))
	defer srvB.Close()

	t.Setenv("CONDUIT_URL", srvA.URL)
	prev := runtime.GOMAXPROCS(1)
	Emit(context.Background(), Event{SituationHash: 7, ProjectID: "call-time", Domain: "d"})
	_ = os.Setenv("CONDUIT_URL", srvB.URL) // switched before the goroutine can run
	runtime.GOMAXPROCS(prev)

	deadline := time.Now().Add(2 * time.Second)
	for hitsA.Load()+hitsB.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if hitsB.Load() != 0 || hitsA.Load() != 1 {
		t.Fatalf("event delivered to A=%d B=%d; it must go to the URL in force at call time (A)", hitsA.Load(), hitsB.Load())
	}
}
