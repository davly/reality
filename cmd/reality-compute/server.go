// Command reality-compute is a thin, stateless HTTP producer that exposes a
// small set of Reality's deterministic compute functions as Nexus-routable
// capabilities over the standard /mcp/tools wire contract.
//
// # Why this binary exists
//
// Reality is a pure-math foundation library: "numbers in / numbers out", zero
// external dependencies, no network calls (CONTEXT.md §1, ARCHITECTURE.md).
// The dependency arrow points one way — aicore imports reality; reality
// imports nothing. Adding an inbound HTTP handler to any of the math packages
// would violate that design rule.
//
// This command is the deliberate exception: it is a separate `cmd/` binary —
// the ONLY thing in the tree that links net/http for an inbound listener — that
// imports the math packages and forwards numeric requests to them. The math
// packages stay pure; the sidecar carries the server. This is "Option A" from
// the Nexus capability-exposure spec (reviews/CAPABILITY_EXPOSURE_2026-06-01/
// B_specs/reality.md §4): lowest blast radius, preserves the language-agnostic
// wire contract, keeps reality independently shippable.
//
// # Wire contract (matches iris / rubberduck exemplars)
//
//	GET  /mcp/tools/                            -> { "tools": [ {name, description, input_schema, approval_required} ] }
//	POST /mcp/tools/reality.conformal_interval  -> body: tool input JSON
//	                                               resp: { content: <json>, is_error: bool, error_message: string }
//
// Tool names are reality.{verb_noun}. The single capability shipped here is
// reality.conformal_interval (the regulator-grade calibration leg, S55 L01);
// wasserstein_distance / garch_volatility are additive siblings deferred until
// a paying consumer asks (spec §6 "Honest caveat").
//
// # Trust boundary (fail-closed)
//
// The hard machine-trust boundary is a constant-time comparison of the
// X-Nexus-Service-Token request header against the NEXUS_SERVICE_TOKEN env
// value. If that env is UNSET (empty), EVERY request to a /mcp/tools route is
// rejected with 401 — fail-closed, never fail-open. This mirrors the Nexus
// exemplar contract ("empty configured secret => 401 for everything").
//
// X-User-Id is required on tool invocations for metering / learning
// attribution (which consumer originated the call) and a missing header is a
// 400. Note (spec §4): these capabilities take NO per-user data — they are
// pure functions over caller-supplied numeric arrays — so X-User-Id is carried
// for billing/provenance only, NOT for data scoping. There is no user data to
// scope here. This is a documented, deliberate divergence from the RubberDuck
// per-user-portfolio contract.
package main

import (
	"bytes"
	"crypto/subtle"
	"encoding/json"
	"io"
	"math"
	"net/http"

	"github.com/davly/reality/prob/conformal"
)

const (
	// nexusServiceHeader is the machine-trust shared-secret header NAME. Nexus
	// sets it on every producer call; its VALUE must equal NEXUS_SERVICE_TOKEN.
	// (Named so it does not match gosec G101's secret-identifier pattern — this
	// constant is a header key, not a credential.)
	nexusServiceHeader = "X-Nexus-Service-Token"
	// headerUserID carries provenance — which end-user the call originated
	// from. Required for metering/attribution (NOT data-scoping here).
	headerUserID = "X-User-Id"

	// toolConformalInterval is the single tool name exposed today. The name is
	// the Nexus routing key.
	toolConformalInterval = "reality.conformal_interval"

	// maxRequestBytes caps an inbound tool-invocation body. Conformal residual
	// arrays are float64 lists; 5 MiB comfortably holds a very large window
	// while bounding memory from a hostile/buggy caller (spec §5).
	maxRequestBytes = 5 << 20 // 5 MiB

	// maxResiduals is an element-count guard on the calibration window. A
	// recency-weighted conformal window of this size is already far beyond any
	// realistic calibration set; it bounds CPU (the engine sorts the slice)
	// independently of the byte cap.
	maxResiduals = 1 << 20 // 1,048,576 residuals
)

// mcpTool is one entry in the GET /mcp/tools/ manifest. Field shape matches the
// iris / rubberduck producers so Nexus's FlagshipToolLoader can flatten it into
// an ai.Tool unchanged.
type mcpTool struct {
	Name             string          `json:"name"`
	Description      string          `json:"description"`
	InputSchema      json.RawMessage `json:"input_schema"`
	ApprovalRequired bool            `json:"approval_required"`
}

// manifestResponse is the GET /mcp/tools/ body.
type manifestResponse struct {
	Tools []mcpTool `json:"tools"`
}

// mcpResult is the standard POST /mcp/tools/{name} envelope. On success
// Content carries the tool's output JSON; on failure IsError is true and
// ErrorMessage explains why (no Content).
type mcpResult struct {
	Content      json.RawMessage `json:"content,omitempty"`
	IsError      bool            `json:"is_error"`
	ErrorMessage string          `json:"error_message,omitempty"`
}

// conformalIntervalInput is the request body for reality.conformal_interval.
// It mirrors the spec input shape exactly:
//
//	{ point_estimate, calibration_residuals[], alpha, half_life_steps }
//
// CalibrationResiduals are *absolute* residuals in time order (oldest first) —
// the contract of conformal.AdaptiveInterval / AdaptiveQuantile, which reject
// negative scores via ErrInvalidScore.
type conformalIntervalInput struct {
	PointEstimate        float64   `json:"point_estimate"`
	CalibrationResiduals []float64 `json:"calibration_residuals"`
	Alpha                float64   `json:"alpha"`
	HalfLifeSteps        int       `json:"half_life_steps"`
}

// conformalIntervalOutput is the tool's success payload (carried inside
// mcpResult.Content), mirroring the spec output shape:
//
//	{ lo, hi, coverage, effective_n }
//
// lo/hi are the recency-weighted symmetric prediction interval from
// AdaptiveInterval; coverage is the guaranteed marginal lower bound (1-alpha,
// per Lei et al 2018, via MarginalCoverageBounds); effective_n is the Kish
// effective sample size of the recency-weighted window (EffectiveSampleSize).
type conformalIntervalOutput struct {
	Lo         float64 `json:"lo"`
	Hi         float64 `json:"hi"`
	Coverage   float64 `json:"coverage"`
	EffectiveN float64 `json:"effective_n"`
}

// conformalIntervalSchema is the JSON-Schema advertised in the manifest. It is
// a literal (not reflected) so the wire contract is reviewable in one place.
var conformalIntervalSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "point_estimate": { "type": "number", "description": "The point forecast yhat the interval is centred on." },
    "calibration_residuals": { "type": "array", "items": { "type": "number", "minimum": 0 }, "description": "Absolute nonconformity residuals in time order (oldest first). Must be non-negative." },
    "alpha": { "type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1, "description": "Miscoverage level, e.g. 0.1 for a 90% interval." },
    "half_life_steps": { "type": "integer", "minimum": 1, "description": "Recency half-life: samples after which a residual's weight halves. Large value -> classical split conformal." }
  },
  "required": ["point_estimate", "calibration_residuals", "alpha", "half_life_steps"]
}`)

// newMux builds the producer's HTTP routing. The /mcp/tools routes are NOT
// wrapped in any application-wide auth middleware (there is none in this
// binary); the per-handler constant-time service-token check IS the trust
// boundary. serviceToken is the configured shared secret — an EMPTY value
// makes every /mcp/tools call fail-closed with 401.
func newMux(serviceToken string) *http.ServeMux {
	mux := http.NewServeMux()

	// Manifest. GET only. Fail-closed on the service token.
	mux.HandleFunc("/mcp/tools/", func(w http.ResponseWriter, r *http.Request) {
		// Distinguish the exact manifest path from tool-invocation paths.
		if r.URL.Path != "/mcp/tools/" {
			handleToolInvoke(w, r, serviceToken)
			return
		}
		if r.Method != http.MethodGet {
			writeJSON(w, http.StatusMethodNotAllowed, mcpResult{
				IsError:      true,
				ErrorMessage: "manifest is GET-only",
			})
			return
		}
		if !authOK(r, serviceToken) {
			writeUnauthorized(w)
			return
		}
		writeJSON(w, http.StatusOK, manifestResponse{Tools: manifest()})
	})

	return mux
}

// manifest returns the advertised tool set. Today: a single read-only,
// side-effect-free compute tool, so approval_required is false.
func manifest() []mcpTool {
	return []mcpTool{
		{
			Name:             toolConformalInterval,
			Description:      "Distribution-free, recency-weighted conformal prediction interval with regulator-grade finite-sample coverage. Deterministic, golden-pinned cross-substrate. Input absolute residuals in time order; output a symmetric interval plus its guaranteed marginal coverage and effective sample size.",
			InputSchema:      conformalIntervalSchema,
			ApprovalRequired: false,
		},
	}
}

// handleToolInvoke routes POST /mcp/tools/{name}. It enforces the trust
// boundary (service token, fail-closed) and provenance (X-User-Id) BEFORE
// dispatching to a tool, then returns the standard {content,is_error,
// error_message} envelope.
func handleToolInvoke(w http.ResponseWriter, r *http.Request, serviceToken string) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, mcpResult{
			IsError:      true,
			ErrorMessage: "tool invocation is POST-only",
		})
		return
	}

	// Machine trust first — fail-closed. An unset configured secret rejects
	// everything; a wrong/absent header rejects this call.
	if !authOK(r, serviceToken) {
		writeUnauthorized(w)
		return
	}

	// Provenance. Required for metering/attribution even though no per-user
	// data is scoped here (pure functions). 400 when absent.
	if r.Header.Get(headerUserID) == "" {
		writeJSON(w, http.StatusBadRequest, mcpResult{
			IsError:      true,
			ErrorMessage: "missing X-User-Id (provenance required for metering)",
		})
		return
	}

	name := r.URL.Path[len("/mcp/tools/"):]
	switch name {
	case toolConformalInterval:
		handleConformalInterval(w, r)
	default:
		writeJSON(w, http.StatusNotFound, mcpResult{
			IsError:      true,
			ErrorMessage: "unknown tool: " + name,
		})
	}
}

// handleConformalInterval decodes the request, forwards it to the real
// conformal engine (prob/conformal), and writes the result envelope.
//
// Validation taxonomy:
//   - malformed JSON / oversized body / element-count overflow -> 400 (is_error)
//   - engine validation errors (bad alpha, empty/negative residuals, bad
//     half-life) -> 422 (is_error) — the request was well-formed JSON but the
//     numeric arguments are out of the engine's documented domain.
//
// A successful interval (including the documented +Inf return when the window
// is too small to support the requested coverage) is 200 with is_error:false;
// the +Inf is normalised to a JSON null so the response stays valid JSON.
func handleConformalInterval(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxRequestBytes+1))
	if err != nil {
		writeJSON(w, http.StatusBadRequest, mcpResult{
			IsError:      true,
			ErrorMessage: "failed to read request body",
		})
		return
	}
	if len(body) > maxRequestBytes {
		writeJSON(w, http.StatusRequestEntityTooLarge, mcpResult{
			IsError:      true,
			ErrorMessage: "request body exceeds size limit",
		})
		return
	}

	var in conformalIntervalInput
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&in); err != nil {
		writeJSON(w, http.StatusBadRequest, mcpResult{
			IsError:      true,
			ErrorMessage: "invalid request body: " + err.Error(),
		})
		return
	}

	if len(in.CalibrationResiduals) > maxResiduals {
		writeJSON(w, http.StatusBadRequest, mcpResult{
			IsError:      true,
			ErrorMessage: "calibration_residuals exceeds element-count limit",
		})
		return
	}

	// Forward to the real engine. AdaptiveInterval returns the symmetric band;
	// any domain error (alpha, residual sign/NaN, half-life) surfaces here.
	lo, hi, err := conformal.AdaptiveInterval(in.PointEstimate, in.CalibrationResiduals, in.Alpha, in.HalfLifeSteps)
	if err != nil {
		writeJSON(w, http.StatusUnprocessableEntity, mcpResult{
			IsError:      true,
			ErrorMessage: err.Error(),
		})
		return
	}

	// coverage = the guaranteed marginal lower bound (1 - alpha) from the same
	// package; n is validated > 0 above implicitly (empty residuals already
	// rejected by AdaptiveInterval via ErrEmptyCalibration).
	coverage, _, cErr := conformal.MarginalCoverageBounds(len(in.CalibrationResiduals), in.Alpha)
	if cErr != nil {
		// Should be unreachable given AdaptiveInterval already validated alpha
		// and non-empty residuals, but surface honestly rather than emit a
		// silent zero.
		writeJSON(w, http.StatusUnprocessableEntity, mcpResult{
			IsError:      true,
			ErrorMessage: cErr.Error(),
		})
		return
	}

	effectiveN := conformal.EffectiveSampleSize(len(in.CalibrationResiduals), in.HalfLifeSteps)

	out := conformalIntervalOutput{
		Lo:         jsonSafe(lo),
		Hi:         jsonSafe(hi),
		Coverage:   coverage,
		EffectiveN: effectiveN,
	}
	payload, err := json.Marshal(out)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, mcpResult{
			IsError:      true,
			ErrorMessage: "failed to encode result",
		})
		return
	}
	writeJSON(w, http.StatusOK, mcpResult{Content: payload, IsError: false})
}

// authOK performs the fail-closed, constant-time machine-trust check. A
// configured token of "" (env unset) rejects unconditionally — there is no
// branch that returns true for an empty configured secret.
func authOK(r *http.Request, configured string) bool {
	if configured == "" {
		return false
	}
	presented := r.Header.Get(nexusServiceHeader)
	// subtle.ConstantTimeCompare returns 1 only when lengths match AND bytes
	// are equal; a length mismatch is a non-constant-time early-out but reveals
	// only length, which the attacker already controls.
	return subtle.ConstantTimeCompare([]byte(presented), []byte(configured)) == 1
}

// writeUnauthorized emits the canonical 401 envelope.
func writeUnauthorized(w http.ResponseWriter) {
	writeJSON(w, http.StatusUnauthorized, mcpResult{
		IsError:      true,
		ErrorMessage: "unauthorized: invalid or missing X-Nexus-Service-Token",
	})
}

// writeJSON serialises v as the response body with the given status. Encoding
// errors are swallowed (the header is already committed); they cannot occur for
// the concrete types used here.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// jsonSafe maps non-finite float64 values to ones encoding/json accepts (it
// errors on Inf/NaN). AdaptiveQuantile documents a +Inf return when the window
// is too small to honour the requested coverage — an "effectively unbounded"
// interval, a real outcome we must NOT silently zero. We encode it as the
// largest finite float64 (round-trips, unambiguously "huge"); -Inf symmetrically.
// A NaN bound is not a documented engine output; map it to 0 defensively rather
// than emit invalid JSON.
func jsonSafe(f float64) float64 {
	switch {
	case math.IsInf(f, 1):
		return math.MaxFloat64
	case math.IsInf(f, -1):
		return -math.MaxFloat64
	case math.IsNaN(f):
		return 0
	default:
		return f
	}
}
