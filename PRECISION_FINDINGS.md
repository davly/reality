# Precision Property-Test Findings

A stdlib `testing/quick` property layer that PINS a focused, high-value set of
the `Precision:` docstring claims in `github.com/davly/reality` as TESTED
INVARIANTS — proving each bound holds, or surfacing an over-claim as an honest,
visible finding.

- **Additive only.** Test files (`*_precision_test.go`) + this doc. NO math,
  source, or behavior code was modified. The zero-external-dependency law is
  preserved (tests use only `testing`, `testing/quick`, `math`, `math/big`,
  `math/cmplx`, `sort` — all Go stdlib; `go.mod` still has zero requires).
- **Two distinct test outcomes, used deliberately — do NOT conflate them:**
  - **ENFORCED invariants (fail RED).** For a bound that DEMONSTRABLY HOLDS
    today, the failure path is `t.Errorf` / `t.Fatalf`, so a *future* regression
    of that bound turns the suite RED and is caught by CI. These are genuine
    regression guards: a `t.Skip` would NOT guard a bound (a SKIP is success to
    `go test` — exit code 0 — so a regressed bound would stay invisible-green).
    The 33 holding-bound pins (mel/sRGB/HSV/Lab round-trips, quaternion
    isometry/identity/normalize, NormalCDF monotone, NormalQuantile lower-half,
    StudentT, Wiener, Quantile range/monotone, bisection, LinearInterpolate, and
    especially the chi-sq regression guard) are all ENFORCED. *Proven:* forcing
    any of these guards to fire yields `go test` exit code 1 (RED), not a SKIP.
  - **DOCUMENTED over-claims (SKIP, never silent).** For the 4 bounds that
    genuinely do NOT hold over the full claimed domain, the test `t.Skip(...)`s
    with a precise reason (visible in `go test -v`) — surfacing the honest
    finding without inventing a failure or silently passing. `t.Skip` is
    reserved EXCLUSIVELY for these 4; it is never used to swallow a holding
    bound.
- The suite is GREEN today because every enforced bound currently holds and the
  4 documented over-claims are the only SKIPs — but "green" here means
  "no regression of an enforced bound", NOT "no failing test was allowed to
  fail". An enforced bound that regresses WILL turn it red.
- 37 new test functions across 9 packages: **33 PASS (enforced, fail-red), 4
  SKIP (documented over-claims).**

Run: `go test -v ./audio/ ./geometry/ ./optim/ ./prob/ ./prob/copula/ ./combinatorics/ ./color/ ./audio/separation/`

---

## Claims PINNED (bound holds — ENFORCED, fail-RED on regression)

These bounds hold today and are pinned as PASS; their failure path is
`t.Errorf`/`t.Fatalf`, so a future regression turns the suite RED (real CI
protection — a `t.Skip` would not guard them).

| Function (file:line) | Claimed bound | How tested | Worst observed |
|---|---|---|---|
| `MelToHz` round-trip (audio/melscale.go:38) | `HzToMel(MelToHz(m)) <= 1e-9` over [0,8000] | quick (200k) + dense 80k grid | **9.09e-13** |
| `HzToMel` / `MelToHz` (melscale.go:15/37) | monotonically increasing | quick monotonicity | holds |
| `QuatToAxisAngle∘QuatFromAxisAngle` (geometry/quaternion.go:132/158) | `<= 1e-12` (well-conditioned angles) | rotation-action on basis vectors, band [0.05, π-0.05] | **4.70e-15** |
| `QuatRotateVec` (quaternion.go:190) | "exact" → isometry `|R(v)|==|v|` | quick (200k), unit quats | rel **1.55e-15** |
| `QuatRotateVec(identity)` (quaternion.go:190) | "exact" → bit-exact no-op | quick (100k) bit-equality | bit-exact |
| `QuatNormalize` (quaternion.go:47) | "exact" → unit length | quick (100k) | `|mag-1|` **3.33e-16** |
| `BisectionMethod` (optim/rootfind.go:20) | `|root - x*| <= tol` | quick over known roots + cos | **4.99e-10** (tol 1e-9) |
| `LinearInterpolateRoot` (rootfind.go:110) | "exact (1 div + 1 mul)" — operation-count | quick, well-conditioned residual | rel **1.12e-12** (see caveat) |
| `LinearInterpolateRoot` NaN contract | `NaN` if `y0==y1` | direct | holds |
| `NormalQuantile` value (prob/distributions.go:64) | `max rel err < 1.15e-9` (LOWER half, p in [1e-12,0.5)) | vs 200-step bisection on machine-precision `NormalCDF` | **1.12e-9** (validates the Acklam figure) |
| `NormalCDF` (distributions.go) | monotone + symmetry `CDF(-x)+CDF(x)=1` | quick (100k) | holds (sym 1e-12) |
| `ChiSquaredTest` p-value (hypothesis.go:165) | correct CDF / monotone p | regression pin: χ²=14400 → p=0; monotone | **p=0** (gamma two-branch fix holds) |
| `StudentTQuantile∘StudentTCDF` (prob/copula/studentt.go:53) | `~1e-10` on x | CDF round-trip, p∈[1e-6,1-1e-6], df∈[1,200] | CDF err **2.4e-10** |
| `StudentTCDF` (studentt.go:23) | monotone + `CDF(0)=0.5` + [0,1] | quick (50k) | holds |
| `Factorial` (combinatorics/counting.go:21) | "exact for n<=20" | bit-exact vs `big.Int`, n=0..20 | bit-exact |
| `BinomialCoeff` (counting.go:48) | `rel err < 1e-12` for typical inputs | vs `big.Int`, n<=200 | **3.09e-13** |
| `BinomialCoeff` symmetry | `C(n,k)==C(n,n-k)` bit-exact | quick (20k) | bit-exact |
| `FibonacciNumber` (counting.go:108) | "exact (integer arithmetic)" | `F_n==F_{n-1}+F_{n-2}` bit-exact, n=3..93; F_93 golden | bit-exact |
| `SRGBToLinear`/`LinearToSRGB` (color/spaces.go:25/43) | "exact to float64" → round-trip | quick (200k) | **3.33e-16** |
| `RGBToHSV`/`HSVToRGB` (spaces.go:157/194) | "exact to float64" → round-trip | quick (200k) | **1.33e-15** |
| `XYZToLab`/`LabToXYZ` (spaces.go) | inverse pair round-trip | quick (200k), D65 | **1.33e-15** |
| `WienerFilter` (audio/separation/wiener.go:37) | gain∈[0,1] ⇒ `|out|<=|in|`; boundary cases | quick (200k) + boundary asserts | holds (bit-exact pass-through / full attenuation) |
| `Quantile`/`Percentile` (prob/percentile.go) | output ∈ [min,max]; monotone in q; clamp/edge | quick (100k) + edge asserts | holds |

---

## OVER-CLAIMS FOUND (honest findings — DOCUMENTED via `t.Skip`, never silent)

These 4 bounds do NOT hold over their full claimed domain. Each test
`t.Skip(...)`s with a precise reason (visible in `go test -v`) — surfacing the
over-claim honestly without manufacturing a failure or silently passing. `t.Skip`
is used ONLY here, never to swallow a holding bound.

### 1. `Factorial` — `< 1e-15` is over-claimed for `n > 20` (counting.go:21)
- **Claim:** "relative error < 1e-15 for n <= 170".
- **Observed:** worst **1.30e-13 at n=166** (~130× the claim).
- **Cause (understood):** for `n > 20` the impl uses `exp(lgamma(n+1))`. `lgamma`
  carries ~1e-15 relative error, which is AMPLIFIED by `ln(n!)` (~745 at n=166)
  when exponentiated: `exp(x(1±ε)) = result·(1 ± x·ε)`.
- **Honest bound:** `< 1e-13` for n<=170 (this is PINNED as a PASS). The
  `< 1e-15` figure is only true for the exact `n <= 20` path (bit-exact, also
  pinned). Suggested doc fix: state `< 1e-13` for `21 <= n <= 170`, keep the
  bit-exact claim scoped to `n <= 20`.
- Test: `TestFactorialRelErr170OverClaim` (SKIP) + `TestFactorialExactSmall` (PASS).

### 2. `NormalQuantile` — `< 1.15e-9` over-claimed in the UPPER tail (distributions.go:64)
- **Claim:** "maximum relative error < 1.15e-9 across ALL p in (0, 1)".
- **Observed:** worst **1.10e-6 at p = 1 − 1e-12** (~960× the claim). At
  p = 1 − 1e-10 it is ~1.3e-8 (~11×).
- **Cause (understood):** the upper branch computes `q = sqrt(-2·ln(1-p))`;
  `1-p` suffers catastrophic cancellation as `p → 1` (p = 1−1e-12 retains only
  ~4 significant digits of `1-p`). The LOWER half is unaffected (p is exact)
  and meets the claim (worst 1.12e-9, which validates the published Acklam
  figure) — this is PINNED as a PASS.
- **Honest framing:** the claim holds on the lower half but is over-claimed as
  `p → 1`. Suggested doc fix: scope the bound to p bounded away from 1, or note
  the `1-p` cancellation in the upper tail.
- Test: `TestNormalQuantileValueUpperTailOverClaim` (SKIP) +
  `TestNormalQuantileValueLowerAndBulk` (PASS).

### 3. `QuatToAxisAngle` round-trip — `1e-12` over-claimed near degenerate angles (quaternion.go:158)
- **Claim:** "Precision: 1e-12 (transcendental functions)" — stated
  unconditionally.
- **Observed:** worst rotation-action round-trip error **4.43e-11** for angles
  down to 1e-6 rad from 0 / π (~44× the claim).
- **Cause (understood, NOT an impl defect):** axis-angle is intrinsically
  ill-conditioned as `angle → 0` / `angle → π` — the axis becomes undefined and
  `axis = (x,y,z)/sin(angle/2)` divides by a vanishing `sin`. For
  well-conditioned angles [0.05, π−0.05] the round-trip is **4.70e-15** (PINNED
  PASS); [0.01, π−0.01] is ~4.2e-14 (still < 1e-12).
- **Honest framing:** the bound holds for typical angles; an honest docstring
  would scope it to angles bounded away from 0 and π.
- Test: `TestQuatAxisAngleRoundTripNearDegenerate` (SKIP) +
  `TestQuatAxisAngleRoundTripWellConditioned` (PASS).

### 4. `BinomialCoeff` — CAVEAT: `< 1e-12` exceeded for large n (counting.go:48)
- **Claim:** "relative error < 1e-12 for typical inputs".
- **Observed:** worst **2.45e-12 at C(990,86)** (large n); also ~1.1e-12 at
  C(420,12). For n <= 200 the bound holds comfortably (3.09e-13, PINNED PASS).
- **Cause:** accumulated `lgamma` error in `exp(lg(n) - lg(k) - lg(n-k))`.
- **Honest framing:** softer than the above — "for typical inputs" arguably
  scopes out n in the high hundreds. Recorded as a CAVEAT for large-n callers
  (expect ~few×1e-12), not a hard contract violation.
- Test: `TestBinomialRelErrLargeN` (SKIP) + `TestBinomialRelErrTypical` (PASS).

---

## Notes on test-bound calibration (NOT findings)

- `LinearInterpolateRoot`'s "exact" is an **operation-count** claim (exactly one
  correctly-rounded division + one multiply), not a small-residual guarantee.
  An initial 1e-12 test bound was too tight: near-coincident abscissae are
  inherently ill-conditioned (catastrophic cancellation in `(x1-x0)/(y1-y0)`),
  giving residuals up to ~2.4e-5. The test pins the **well-conditioned** regime
  (well-separated points: ~1.12e-12) and explicitly does NOT pin the
  ill-conditioned case. This is a test-tolerance calibration, not an impl
  over-claim.
