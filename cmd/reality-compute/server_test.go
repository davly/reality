package main

import (
	"bytes"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/davly/reality/prob/conformal"
)

const testToken = "s3cr3t-shared-token"

// doReq is a small helper: issues a request against a live httptest server and
// returns status + decoded mcpResult.
func doReq(t *testing.T, srv *httptest.Server, method, path, token, userID, body string) (int, mcpResult) {
	t.Helper()
	var rdr io.Reader
	if body != "" {
		rdr = strings.NewReader(body)
	}
	req, err := http.NewRequest(method, srv.URL+path, rdr)
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	if token != "" {
		req.Header.Set(nexusServiceHeader, token)
	}
	if userID != "" {
		req.Header.Set(headerUserID, userID)
	}
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatalf("do request: %v", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var res mcpResult
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &res); err != nil {
			t.Fatalf("decode response (%s): %v", string(raw), err)
		}
	}
	return resp.StatusCode, res
}

// ---------------------------------------------------------------------------
// STEP-1.5: reachability — the /mcp/tools routes answer over a real HTTP
// listener with NO auth cookie, only the service-token header.
// ---------------------------------------------------------------------------

func TestManifestReachable(t *testing.T) {
	srv := httptest.NewServer(newMux(testToken))
	defer srv.Close()

	// Decode the manifest separately (it is not an mcpResult shape).
	req, _ := http.NewRequest(http.MethodGet, srv.URL+"/mcp/tools/", nil)
	req.Header.Set(nexusServiceHeader, testToken)
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatalf("do: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("manifest: want 200, got %d", resp.StatusCode)
	}
	var m manifestResponse
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if len(m.Tools) != 1 || m.Tools[0].Name != toolConformalInterval {
		t.Fatalf("manifest tools = %+v, want exactly [%s]", m.Tools, toolConformalInterval)
	}
	if m.Tools[0].ApprovalRequired {
		t.Errorf("conformal_interval must not require approval (read-only deterministic compute)")
	}
	// The advertised input_schema must be valid JSON (Nexus flattens it).
	var schema map[string]any
	if err := json.Unmarshal(m.Tools[0].InputSchema, &schema); err != nil {
		t.Fatalf("input_schema is not valid JSON: %v", err)
	}
}

// ---------------------------------------------------------------------------
// STEP-1.5: fail-closed — when NEXUS_SERVICE_TOKEN is UNSET (configured token
// is empty), EVERY /mcp/tools route returns 401, even with a token header
// present. This is the "never fail open" invariant.
// ---------------------------------------------------------------------------

func TestUnsetTokenFailsClosed(t *testing.T) {
	srv := httptest.NewServer(newMux("")) // empty configured secret == env unset
	defer srv.Close()

	cases := []struct {
		name, method, path, hdrToken, userID, body string
	}{
		{"manifest no header", http.MethodGet, "/mcp/tools/", "", "", ""},
		{"manifest with header", http.MethodGet, "/mcp/tools/", "anything", "", ""},
		{"invoke no header", http.MethodPost, "/mcp/tools/" + toolConformalInterval, "", "u1", `{"point_estimate":1,"calibration_residuals":[1],"alpha":0.1,"half_life_steps":10}`},
		{"invoke with header + user", http.MethodPost, "/mcp/tools/" + toolConformalInterval, "anything", "u1", `{"point_estimate":1,"calibration_residuals":[1],"alpha":0.1,"half_life_steps":10}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			status, res := doReq(t, srv, c.method, c.path, c.hdrToken, c.userID, c.body)
			if status != http.StatusUnauthorized {
				t.Fatalf("want 401 (fail-closed), got %d (%+v)", status, res)
			}
			if !res.IsError {
				t.Errorf("401 response must have is_error:true")
			}
		})
	}
}

// authOK must NEVER return true for an empty configured secret, regardless of
// what header is presented (unit-level guard on the trust primitive).
func TestAuthOKEmptyConfiguredAlwaysFalse(t *testing.T) {
	for _, presented := range []string{"", "x", testToken, strings.Repeat("z", 256)} {
		req, _ := http.NewRequest(http.MethodGet, "/mcp/tools/", nil)
		if presented != "" {
			req.Header.Set(nexusServiceHeader, presented)
		}
		if authOK(req, "") {
			t.Fatalf("authOK returned true for empty configured secret (presented=%q) — fail-open!", presented)
		}
	}
}

// ---------------------------------------------------------------------------
// Trust boundary: wrong / missing token -> 401 even when a real secret is set.
// ---------------------------------------------------------------------------

func TestWrongTokenRejected(t *testing.T) {
	srv := httptest.NewServer(newMux(testToken))
	defer srv.Close()

	status, res := doReq(t, srv, http.MethodGet, "/mcp/tools/", "wrong-token", "", "")
	if status != http.StatusUnauthorized {
		t.Fatalf("wrong token: want 401, got %d (%+v)", status, res)
	}
	status, _ = doReq(t, srv, http.MethodGet, "/mcp/tools/", "", "", "")
	if status != http.StatusUnauthorized {
		t.Fatalf("missing token: want 401, got %d", status)
	}
}

// ---------------------------------------------------------------------------
// Provenance: a valid token but no X-User-Id on a tool invocation -> 400.
// ---------------------------------------------------------------------------

func TestMissingUserIDRejected(t *testing.T) {
	srv := httptest.NewServer(newMux(testToken))
	defer srv.Close()

	body := `{"point_estimate":1,"calibration_residuals":[1,2,3],"alpha":0.1,"half_life_steps":10}`
	status, res := doReq(t, srv, http.MethodPost, "/mcp/tools/"+toolConformalInterval, testToken, "", body)
	if status != http.StatusBadRequest {
		t.Fatalf("missing X-User-Id: want 400, got %d (%+v)", status, res)
	}
	if !res.IsError {
		t.Errorf("400 must have is_error:true")
	}
}

// ---------------------------------------------------------------------------
// Real-engine: a successful invocation forwards to prob/conformal and returns
// the SAME numbers the engine produces when called directly. This proves the
// producer wraps the real engine, not a stub.
// ---------------------------------------------------------------------------

func TestConformalIntervalRealEngine(t *testing.T) {
	srv := httptest.NewServer(newMux(testToken))
	defer srv.Close()

	point := 100.0
	residuals := []float64{0.5, 1.0, 2.0, 1.5, 3.0, 0.8, 2.2, 1.1, 0.9, 2.7}
	alpha := 0.1
	halfLife := 5

	bodyStruct := conformalIntervalInput{
		PointEstimate:        point,
		CalibrationResiduals: residuals,
		Alpha:                alpha,
		HalfLifeSteps:        halfLife,
	}
	bodyBytes, _ := json.Marshal(bodyStruct)

	status, res := doReq(t, srv, http.MethodPost, "/mcp/tools/"+toolConformalInterval, testToken, "user-42", string(bodyBytes))
	if status != http.StatusOK {
		t.Fatalf("want 200, got %d (%+v)", status, res)
	}
	if res.IsError {
		t.Fatalf("unexpected is_error: %s", res.ErrorMessage)
	}

	var out conformalIntervalOutput
	if err := json.Unmarshal(res.Content, &out); err != nil {
		t.Fatalf("decode content: %v", err)
	}

	// Independently compute the expected values straight from the engine.
	wantLo, wantHi, err := conformal.AdaptiveInterval(point, residuals, alpha, halfLife)
	if err != nil {
		t.Fatalf("engine AdaptiveInterval errored: %v", err)
	}
	wantCov, _, err := conformal.MarginalCoverageBounds(len(residuals), alpha)
	if err != nil {
		t.Fatalf("engine MarginalCoverageBounds errored: %v", err)
	}
	wantN := conformal.EffectiveSampleSize(len(residuals), halfLife)

	if out.Lo != jsonSafe(wantLo) || out.Hi != jsonSafe(wantHi) {
		t.Errorf("interval [%v,%v], engine [%v,%v]", out.Lo, out.Hi, wantLo, wantHi)
	}
	if out.Coverage != wantCov {
		t.Errorf("coverage %v, engine %v", out.Coverage, wantCov)
	}
	if out.EffectiveN != wantN {
		t.Errorf("effective_n %v, engine %v", out.EffectiveN, wantN)
	}
	// Sanity: the interval is symmetric around the point estimate and covers
	// the lower marginal bound.
	if math.Abs((out.Hi-point)-(point-out.Lo)) > 1e-9 {
		t.Errorf("interval not symmetric about point estimate: lo=%v hi=%v point=%v", out.Lo, out.Hi, point)
	}
	if out.Coverage != 1.0-alpha {
		t.Errorf("coverage %v, want guaranteed lower bound %v", out.Coverage, 1.0-alpha)
	}
}

// ---------------------------------------------------------------------------
// Engine domain errors surface as 422 (well-formed JSON, bad numeric args).
// ---------------------------------------------------------------------------

func TestEngineDomainErrors(t *testing.T) {
	srv := httptest.NewServer(newMux(testToken))
	defer srv.Close()

	cases := []struct {
		name, body string
	}{
		{"alpha out of range", `{"point_estimate":1,"calibration_residuals":[1,2],"alpha":1.5,"half_life_steps":10}`},
		{"negative residual", `{"point_estimate":1,"calibration_residuals":[1,-2],"alpha":0.1,"half_life_steps":10}`},
		{"empty residuals", `{"point_estimate":1,"calibration_residuals":[],"alpha":0.1,"half_life_steps":10}`},
		{"non-positive half-life", `{"point_estimate":1,"calibration_residuals":[1,2],"alpha":0.1,"half_life_steps":0}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			status, res := doReq(t, srv, http.MethodPost, "/mcp/tools/"+toolConformalInterval, testToken, "u", c.body)
			if status != http.StatusUnprocessableEntity {
				t.Fatalf("want 422, got %d (%+v)", status, res)
			}
			if !res.IsError || res.ErrorMessage == "" {
				t.Errorf("domain error must carry is_error + message")
			}
		})
	}
}

// Malformed JSON / unknown fields -> 400 (not 422).
func TestMalformedBody(t *testing.T) {
	srv := httptest.NewServer(newMux(testToken))
	defer srv.Close()

	cases := []string{
		`{not json`,
		`{"point_estimate":1,"calibration_residuals":[1],"alpha":0.1,"half_life_steps":10,"surprise":true}`, // unknown field
	}
	for _, body := range cases {
		status, res := doReq(t, srv, http.MethodPost, "/mcp/tools/"+toolConformalInterval, testToken, "u", body)
		if status != http.StatusBadRequest {
			t.Fatalf("malformed body %q: want 400, got %d (%+v)", body, status, res)
		}
	}
}

// A window too small to honour the requested coverage returns +Inf from the
// engine; the producer encodes it as MaxFloat64 (effectively unbounded) and
// still returns 200 with is_error:false.
func TestUnboundedIntervalEncodedFinite(t *testing.T) {
	srv := httptest.NewServer(newMux(testToken))
	defer srv.Close()

	// n=1, alpha=0.1 -> rank target overflows -> AdaptiveQuantile returns +Inf.
	body := `{"point_estimate":5,"calibration_residuals":[2.0],"alpha":0.1,"half_life_steps":3}`
	status, res := doReq(t, srv, http.MethodPost, "/mcp/tools/"+toolConformalInterval, testToken, "u", body)
	if status != http.StatusOK || res.IsError {
		t.Fatalf("want 200 non-error, got %d (%+v)", status, res)
	}
	var out conformalIntervalOutput
	if err := json.Unmarshal(res.Content, &out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out.Hi != math.MaxFloat64 || out.Lo != -math.MaxFloat64 {
		t.Errorf("unbounded interval should encode as +/-MaxFloat64, got lo=%v hi=%v", out.Lo, out.Hi)
	}
	// And the raw response must be valid JSON with no Inf token.
	if strings.Contains(string(res.Content), "Inf") {
		t.Errorf("response contains a literal Inf token (invalid JSON): %s", res.Content)
	}
}

// Unknown tool name under /mcp/tools/ -> 404 (after auth + provenance pass).
func TestUnknownTool(t *testing.T) {
	srv := httptest.NewServer(newMux(testToken))
	defer srv.Close()

	status, res := doReq(t, srv, http.MethodPost, "/mcp/tools/reality.does_not_exist", testToken, "u", `{}`)
	if status != http.StatusNotFound {
		t.Fatalf("unknown tool: want 404, got %d (%+v)", status, res)
	}
}

// Wrong HTTP methods are rejected.
func TestMethodGuards(t *testing.T) {
	srv := httptest.NewServer(newMux(testToken))
	defer srv.Close()

	// Manifest is GET-only.
	status, _ := doReq(t, srv, http.MethodPost, "/mcp/tools/", testToken, "u", `{}`)
	// POST to the exact manifest path is routed to tool-invoke -> unknown tool ""? No:
	// path "/mcp/tools/" exactly => manifest handler => 405 for POST.
	if status != http.StatusMethodNotAllowed {
		t.Fatalf("POST manifest: want 405, got %d", status)
	}
	// Tool invoke is POST-only.
	status, _ = doReq(t, srv, http.MethodGet, "/mcp/tools/"+toolConformalInterval, testToken, "u", "")
	if status != http.StatusMethodNotAllowed {
		t.Fatalf("GET tool: want 405, got %d", status)
	}
}

// Oversized body -> 413.
func TestOversizeBody(t *testing.T) {
	srv := httptest.NewServer(newMux(testToken))
	defer srv.Close()

	big := bytes.Repeat([]byte("a"), maxRequestBytes+10)
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/mcp/tools/"+toolConformalInterval, bytes.NewReader(big))
	req.Header.Set(nexusServiceHeader, testToken)
	req.Header.Set(headerUserID, "u")
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatalf("do: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversize: want 413, got %d", resp.StatusCode)
	}
}
