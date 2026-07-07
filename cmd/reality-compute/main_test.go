package main

import (
	"math"
	"testing"
	"time"
)

// buildServer must default the port, set bounded timeouts (gosec G114), and
// wire the fail-closed mux.
func TestBuildServerDefaults(t *testing.T) {
	srv := buildServer("tok", "")
	if srv.Addr != ":"+defaultPort {
		t.Errorf("default port: Addr = %q, want :%s", srv.Addr, defaultPort)
	}
	if srv.ReadHeaderTimeout <= 0 || srv.ReadTimeout <= 0 || srv.WriteTimeout <= 0 || srv.IdleTimeout <= 0 {
		t.Errorf("server must set all timeouts (slow-loris guard): %+v", srv)
	}
	if srv.ReadHeaderTimeout != 5*time.Second {
		t.Errorf("ReadHeaderTimeout = %v, want 5s", srv.ReadHeaderTimeout)
	}
	if srv.Handler == nil {
		t.Errorf("handler must be wired")
	}
}

func TestBuildServerExplicitPort(t *testing.T) {
	srv := buildServer("tok", "12345")
	if srv.Addr != ":12345" {
		t.Errorf("Addr = %q, want :12345", srv.Addr)
	}
}

// jsonSafe must map every non-finite case to a JSON-encodable value and pass
// finite values through unchanged.
func TestJSONSafe(t *testing.T) {
	cases := []struct {
		name string
		in   float64
		want float64
	}{
		{"finite", 42.5, 42.5},
		{"zero", 0, 0},
		{"negative finite", -7.25, -7.25},
		{"+Inf", math.Inf(1), math.MaxFloat64},
		{"-Inf", math.Inf(-1), -math.MaxFloat64},
		{"NaN", math.NaN(), 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := jsonSafe(c.in)
			if got != c.want {
				t.Errorf("jsonSafe(%v) = %v, want %v", c.in, got, c.want)
			}
		})
	}
}
