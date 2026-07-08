# Reality — Nexus Capability Exposure (operator runbook)

**Capability:** `conformal_interval` · **Producer binary:** `cmd/reality-compute` ·
**Shape:** (2) Capable provider, forwarding to a Shape-(1) `/mcp/tools` producer ·
**Status:** producer BUILT (this repo); Nexus-side Go shim DEFERRED (see §4).

Reality is a pure-math foundation library — "numbers in / numbers out", zero deps,
no network (`CONTEXT.md` §1: *"reality imports nothing"*). It is normally consumed by
**in-process import** (`import github.com/davly/reality/...`). Go apps that already
import it should **keep doing so** — adding a network hop to a pure function is a
latency regression. This producer exists for the narrow case the capability-hub
serves: **cross-language consumers** and **metered/audited single-source compute**
that want a calibrated conformal interval over the wire without a Go FFI.

To keep the math packages pure, the HTTP server lives **only** in the
`cmd/reality-compute` binary — it is the single thing in the tree that links an
inbound `net/http` listener. The math packages stay untouched.

---

## 1. What the producer exposes

### `GET /mcp/tools/` — manifest

Returns the tool set Nexus's `FlagshipToolLoader` registers at startup:

```json
{
  "tools": [
    {
      "name": "reality.conformal_interval",
      "description": "Distribution-free, recency-weighted conformal prediction interval with regulator-grade finite-sample coverage...",
      "input_schema": { "type": "object", "properties": { "...": "..." }, "required": ["point_estimate","calibration_residuals","alpha","half_life_steps"] },
      "approval_required": false
    }
  ]
}
```

`approval_required` is `false` — the tool is read-only deterministic compute with no
side effects.

### `POST /mcp/tools/reality.conformal_interval` — invoke

**Request body**

```json
{
  "point_estimate": 100.0,
  "calibration_residuals": [0.5, 1.0, 2.0, 1.5, 3.0],
  "alpha": 0.1,
  "half_life_steps": 5
}
```

- `point_estimate` — the forecast `yhat` the interval is centred on.
- `calibration_residuals` — **absolute** nonconformity residuals in **time order**
  (oldest first), non-negative. (This is the contract of `conformal.AdaptiveInterval`.)
- `alpha` — miscoverage level in `(0,1)` (e.g. `0.1` → a 90% interval).
- `half_life_steps` — recency half-life ≥ 1; the number of samples after which a
  residual's weight halves. A large value converges to classical split-conformal.

**Response body** (standard MCP envelope)

```json
{
  "content": {
    "lo": 97.3,
    "hi": 102.7,
    "coverage": 0.9,
    "effective_n": 4.12
  },
  "is_error": false
}
```

- `lo` / `hi` — the recency-weighted symmetric prediction interval
  (`conformal.AdaptiveInterval`). When the window is too small to honour the requested
  coverage the engine returns an unbounded interval; the producer encodes the bound as
  `±1.7976931348623157e+308` (max finite float64) rather than emit invalid JSON.
- `coverage` — the **guaranteed marginal lower bound** `1 - alpha`
  (`conformal.MarginalCoverageBounds`, Lei et al 2018).
- `effective_n` — the Kish effective sample size of the recency-weighted window
  (`conformal.EffectiveSampleSize`).

**Error envelope** (any failure): `{ "content": null, "is_error": true, "error_message": "..." }`.

Status taxonomy:

| Status | Cause |
|---|---|
| `200` | success (including the documented unbounded-interval case) |
| `400` | malformed JSON / unknown field / oversized body / missing `X-User-Id` |
| `401` | missing/wrong `X-Nexus-Service-Token`, **or service token unset (fail-closed)** |
| `404` | unknown tool name |
| `405` | wrong HTTP method |
| `413` | body exceeds 5 MiB |
| `422` | well-formed JSON but invalid numeric args (bad alpha, negative/empty residuals, non-positive half-life) |

---

## 2. Trust boundary (fail-closed) + provenance

Two headers, never confused:

- **`X-Nexus-Service-Token`** — machine trust (Nexus ↔ this producer). Constant-time
  compared against `NEXUS_SERVICE_TOKEN`. **If `NEXUS_SERVICE_TOKEN` is UNSET, every
  `/mcp/tools` request returns `401`** — fail-closed, never fail-open.
- **`X-User-Id`** — provenance: which end-user originated the call. **Required** (400 if
  absent) for metering / learning attribution.
  - **Deliberate divergence from the RubberDuck contract:** these capabilities take
    **no per-user data** (pure functions over caller-supplied numeric arrays), so
    `X-User-Id` is carried for billing/provenance **only** — it does NOT scope any data.
    There is no user data to scope. The hard fail-closed boundary is the service token.

The `/mcp/tools` routes are **not** behind any app-wide auth middleware (this binary has
none); the per-handler constant-time token check **is** the boundary. (STEP-1.5 from the
capability-exposure template — there is no auth-cookie redirect hazard here because the
binary has no cookie auth at all.)

---

## 3. Run / deploy the producer

```bash
# Build
go build -o reality-compute ./cmd/reality-compute

# Run (fail-closed: unset NEXUS_SERVICE_TOKEN ⇒ all calls 401; the binary still starts)
NEXUS_SERVICE_TOKEN=<shared-secret> PORT=8090 ./reality-compute
```

| Env var | Meaning | Default |
|---|---|---|
| `NEXUS_SERVICE_TOKEN` | shared machine-trust secret; MUST equal the token Nexus sends | unset ⇒ **fail-closed (401)** |
| `PORT` | listen port | `8090` |

The server sets bounded read/write/idle timeouts (slow-loris guard) and caps request
bodies at 5 MiB with a 1,048,576-element residual guard.

---

## 4. Nexus-side wiring (DEFERRED — spec mapping)

The Nexus consumer side is a Shape-2 Go shim modelled line-for-line on
`infrastructure/nexus/src/api/internal/ai/iris.go`. It is **not** built here (separate
repo, separate PR). The mapping (from `reviews/CAPABILITY_EXPOSURE_2026-06-01/B_specs/reality.md` §5):

```go
// internal/ai/capability.go
CapConformalInterval Capability = "conformal_interval"   // + append to AllCapabilities()

// internal/ai/request.go
//   ConformalIntervalRequest{ UserID (REQUIRED, provenance), PointEstimate,
//     CalibrationResiduals []float64, Alpha, HalfLifeSteps }
//   ConformalIntervalResponse{ Lo, Hi, Coverage, EffectiveN } (or json.RawMessage passthrough)

// internal/ai/reality_compute.go — RealityComputeProvider embedding
//   NewBaseProvider("reality", nil, []Capability{CapConformalInterval});
//   var _ CapableProvider = (*RealityComputeProvider)(nil);
//   Execute() type-switches the request; forwards to
//   POST {REALITY_COMPUTE_URL}/mcp/tools/reality.conformal_interval
//   with X-Nexus-Service-Token + X-User-Id; ProviderError taxonomy
//   (401→Auth, 429→RateLimit, 5xx→Server, 4xx→InvalidRequest).

// internal/ai/bootstrap.go (env-gated, copied from the iris bootstrap)
//   if url := os.Getenv("REALITY_COMPUTE_URL"); url != "" {
//     registry.RegisterCapable("reality",
//       NewRealityComputeProviderWithURL(os.Getenv("REALITY_SERVICE_TOKEN"), url),
//       prioFromEnvOr("REALITY_COMPUTE_PRIORITY", 40)) }
```

Routing is the generic capability gateway (no gateway change), identical to iris→`ocr`:
`registry.ProvidersForCapability(CapConformalInterval)` → `GetCapable("reality")` →
`provider.Execute`.

**Alternatively (Shape 1, zero Nexus code):** add this producer to Nexus's
`FLAGSHIP_TOOL_PROVIDERS` so an LLM agent can pull an interval mid-chat:

```
FLAGSHIP_TOOL_PROVIDERS += reality|https://reality-compute.promptboy.dev|<shared-secret>
```

The shared secret in `NEXUS_SERVICE_TOKEN` (this producer) == the token field in
Nexus's `FLAGSHIP_TOOL_PROVIDERS` == `REALITY_SERVICE_TOKEN` (Shape-2). One value.

---

## 5. Smoke

```bash
BASE=http://localhost:8090
TOK=<shared-secret>

# 1. Reachability + manifest (200, lists reality.conformal_interval)
curl -i -H "X-Nexus-Service-Token: $TOK" $BASE/mcp/tools/

# 2. Fail-closed: no/wrong token ⇒ 401; with NEXUS_SERVICE_TOKEN unset even a header ⇒ 401
curl -i $BASE/mcp/tools/

# 3. Provenance: token but no X-User-Id ⇒ 400
curl -i -X POST -H "X-Nexus-Service-Token: $TOK" \
  -d '{"point_estimate":100,"calibration_residuals":[0.5,1,2,1.5,3],"alpha":0.1,"half_life_steps":5}' \
  $BASE/mcp/tools/reality.conformal_interval

# 4. Real invoke (200; content = {lo,hi,coverage,effective_n})
curl -s -X POST -H "X-Nexus-Service-Token: $TOK" -H "X-User-Id: user-42" \
  -d '{"point_estimate":100,"calibration_residuals":[0.5,1,2,1.5,3],"alpha":0.1,"half_life_steps":5}' \
  $BASE/mcp/tools/reality.conformal_interval
```

---

## 6. Why only `conformal_interval` (not wasserstein / garch)

Per the spec's honest caveat (§6): `conformal_interval` is the regulator-grade leg with
the clearest audit value (FDA AI/ML calibration; S55 L01 trio). `wasserstein_distance`
and `garch_volatility` are real, golden-pinned siblings backed by `optim/transport` and
`timeseries/garch`, but are left as **additive future tools on this same producer** until
a paying cross-language consumer asks for them. Adding one is a single handler + manifest
entry over functions that already exist.
