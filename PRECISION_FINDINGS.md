# Precision Property-Test Findings

A stdlib `testing/quick` property layer that PINS a focused, high-value set of
the `Precision:` docstring claims in `github.com/davly/reality` as TESTED
INVARIANTS — proving each bound holds, or surfacing an over-claim as an honest,
visible finding.

- **The property layer is test-only.** Test files (`*_precision_test.go`) +
  this doc. The fixes recorded below did change code; each entry says what
  changed. The zero-external-dependency law is preserved (tests use only
  `testing`, `testing/quick`, `math`, `math/big`, `math/cmplx`, `sort` — all
  Go stdlib; `go.mod` still has zero requires).
- **Two distinct test outcomes, used deliberately — do NOT conflate them:**
  - **ENFORCED invariants (fail RED).** For a bound that DEMONSTRABLY HOLDS
    today, the failure path is `t.Errorf` / `t.Fatalf`, so a *future* regression
    of that bound turns the suite RED and is caught by CI. These are genuine
    regression guards: a `t.Skip` would NOT guard a bound (a SKIP is success to
    `go test` — exit code 0 — so a regressed bound would stay invisible-green).
    Every pin (mel/sRGB/HSV/Lab round-trips, quaternion isometry/identity/
    normalize/axis-angle, NormalCDF monotone, NormalQuantile over the whole
    range, StudentT, Wiener, Quantile range/monotone, bisection,
    LinearInterpolate, Factorial, BinomialCoeff, and especially the chi-sq
    regression guard) is ENFORCED. *Proven:* forcing
    any of these guards to fire yields `go test` exit code 1 (RED), not a SKIP.
  - **DOCUMENTED over-claims (SKIP, never silent).** A bound that genuinely
    does NOT hold over its full claimed domain is `t.Skip(...)`ped with a
    precise reason (visible in `go test -v`): an honest finding, neither a
    manufactured failure nor a silent pass. `t.Skip` is reserved for such
    findings and never swallows a holding bound. The four over-claims this file
    recorded have all since been fixed (see below), so none skips today.
- The suite is GREEN because every enforced bound holds — "green" here means
  "no regression of an enforced bound", NOT "no failing test was allowed to
  fail". An enforced bound that regresses WILL turn it red.
- 40 test functions across 8 packages: **40 PASS (enforced, fail-red), 0
  SKIP.**

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
| `QuatToAxisAngle∘QuatFromAxisAngle` (geometry/quaternion.go:132/158) | `<= 1e-12` (all angles, down to 1e-8 from 0 and π) | rotation-action on basis vectors, bands [0.05, π-0.05] and [1e-8, π-1e-8] | **1.33e-15** / **1.55e-15** |
| `QuatRotateVec` (quaternion.go:190) | "exact" → isometry `|R(v)|==|v|` | quick (200k), unit quats | rel **1.55e-15** |
| `QuatRotateVec(identity)` (quaternion.go:190) | "exact" → bit-exact no-op | quick (100k) bit-equality | bit-exact |
| `QuatNormalize` (quaternion.go:47) | "exact" → unit length | quick (100k) | `|mag-1|` **3.33e-16** |
| `BisectionMethod` (optim/rootfind.go:20) | `|root - x*| <= tol` | quick over known roots + cos | **4.99e-10** (tol 1e-9) |
| `LinearInterpolateRoot` (rootfind.go:110) | "exact (1 div + 1 mul)" — operation-count | quick, well-conditioned residual | rel **1.12e-12** (see caveat) |
| `LinearInterpolateRoot` NaN contract | `NaN` if `y0==y1` | direct | holds |
| `NormalQuantile` value (prob/distributions.go) | within 2.5 ulps of the exact quantile, every p in (0,1) | vs mpmath at 50 digits (37,124-point grid incl. subnormal p and 1-10^-k, plus 200,000 random p); vs bisection on `NormalCDF` (upper half through the symmetry) | **1.95 ulps** (relative 3.1e-16) |
| `NormalCDF` (distributions.go) | monotone + symmetry `CDF(-x)+CDF(x)=1` | quick (100k) | holds (sym 1e-12) |
| `ChiSquaredTest` p-value (hypothesis.go:165) | correct CDF / monotone p | regression pin: χ²=14400 → p=0; monotone | **p=0** (gamma two-branch fix holds) |
| `StudentTQuantile∘StudentTCDF` (prob/copula/studentt.go:53) | `~1e-10` on x | CDF round-trip, p∈[1e-6,1-1e-6], df∈[1,200] | CDF err **2.4e-10** |
| `StudentTCDF` (studentt.go:23) | monotone + `CDF(0)=0.5` + [0,1] | quick (50k) | holds |
| `Factorial` (combinatorics/counting.go) | correctly rounded for n<=170; exact for n<=22 | every n=0..170 vs exact `big.Int` | 0 of 171 wrong; rel <= **1.05e-16** |
| `BinomialCoeff` / `Permutations` (counting.go) | correctly rounded | every 0<=k<=n<=300 vs exact `big.Int`, plus overflow rows and spot checks to n=2^63-1 | 0 wrong (was 39,562 / 31,035) |
| `BinomialCoeff` symmetry | `C(n,k)==C(n,n-k)` bit-exact | quick (20k) | bit-exact |
| `FibonacciNumber` (counting.go:108) | "exact (integer arithmetic)" | `F_n==F_{n-1}+F_{n-2}` bit-exact, n=3..93; F_93 golden | bit-exact |
| `SRGBToLinear`/`LinearToSRGB` (color/spaces.go:25/43) | "exact to float64" → round-trip | quick (200k) | **3.33e-16** |
| `RGBToHSV`/`HSVToRGB` (spaces.go:157/194) | "exact to float64" → round-trip | quick (200k) | **1.33e-15** |
| `XYZToLab`/`LabToXYZ` (spaces.go) | inverse pair round-trip | quick (200k), D65 | **1.33e-15** |
| `WienerFilter` (audio/separation/wiener.go:37) | gain∈[0,1] ⇒ `|out|<=|in|`; boundary cases | quick (200k) + boundary asserts | holds (bit-exact pass-through / full attenuation) |
| `Quantile`/`Percentile` (prob/percentile.go) | output ∈ [min,max]; monotone in q; clamp/edge | quick (100k) + edge asserts | holds |

---

## OVER-CLAIMS FOUND (all four since RESOLVED)

These 4 bounds did not hold over their full claimed domain when this file was
written; each was documented with a `t.Skip(...)`. All four have since been
fixed, and their tests are ENFORCED.

### 1. `Factorial` — `< 1e-15` was over-claimed for `n > 20` (counting.go) — RESOLVED
- **Status: resolved.** Values now come from exact integer arithmetic, rounded
  once: every n <= 170 returns the float64 nearest to n! (0 of 171 values
  wrong, relative error at most 1.05e-16; it was 150 of 171). The guard is
  `TestFactorialRelErrHalfUlp` (ENFORCED: relative error <= 2^-53).
- **As originally found:**
- **Claim:** "relative error < 1e-15 for n <= 170".
- **Observed:** worst **1.30e-13 at n=166** (~130× the claim).
- **Cause (understood):** for `n > 20` the impl uses `exp(lgamma(n+1))`. `lgamma`
  carries ~1e-15 relative error, which is AMPLIFIED by `ln(n!)` (~745 at n=166)
  when exponentiated: `exp(x(1±ε)) = result·(1 ± x·ε)`.
- Tests: `TestFactorialRelErrHalfUlp` (ENFORCED) + `TestFactorialExactSmall` (PASS).

### 2. `NormalQuantile` — RESOLVED: the upper-tail figure was an artifact of the test oracle; the real gap (~1e-9 everywhere) is closed (distributions.go)
- **Former claim:** "maximum relative error < 1.15e-9 for p bounded away from
  1", next to "full float64 precision across the entire range (0, 1)" in the
  same docstring; the two contradicted each other.
- **Former finding:** 1.10e-6 at p = 1 − 1e-12, attributed to cancellation in
  `1-p`. That number measured the oracle, not the function: `1-p` is exact in
  float64 for p ≥ 1/2, and bisecting `NormalCDF` near 1 resolves p only to
  1.1e-16, so x only to 1.1e-16/phi(x) (5.5e-7 relative at p = 1 − 1e-12).
  Against mpmath the upper tail was exactly as accurate as the lower tail
  (1.047e-9 at both p = 1e-12 and p = 1 − 1e-12).
- **The real gap:** Acklam's rational approximation is accurate only to about
  1e-9 relative over the whole range (worst 1.12e-9), far from full precision.
  For subnormal p it was much worse (−37.54 instead of −38.47 at p = 5e-324),
  because `math.Log` returns wrong values for subnormal arguments on amd64
  (`math.Log(5e-324)` is −709.09, not −744.44).
- **Fix:** one Halley step on Phi(x) − p, with the residual formed without
  cancellation (`math.Erfc` in the tails, `math.Erf` with the exact p − 1/2
  near the median, the exact 1 − p in the upper tail) and corrected for the
  rounding of x/√2; for subnormal p, Newton's method on log Phi(x) − log p
  with the asymptotic series of the normal tail, and log p taken on p·2^54.
- **Now:** within 2.5 ulps of the exact quantile for every p in (0, 1); worst
  measured 1.95 ulps (relative error 3.1e-16) over 237,000 values of p against
  mpmath at 50 digits, with and without fused multiply-adds. Most of the
  remaining error is that of `math.Erfc`, which is off by up to about 3.5
  ulps.
- Tests (all ENFORCED, fail RED): `TestNormalQuantileNearlyExact` (mpmath
  values), `TestNormalQuantileValueUpperTail` (bisection reference through
  Phi^{-1}(p) = −Phi^{-1}(1−p)), `TestNormalQuantileValueLowerAndBulk`,
  `TestNormalQuantileRoundTrip`.

### 3. `QuatToAxisAngle` round-trip — `1e-12` over-claimed near degenerate angles (quaternion.go:158) — RESOLVED
- **Status: resolved.** `QuatToAxisAngle` now takes the angle as
  `2*atan2(|v|, w)` instead of `2*acos(w)`. Measured against 50-digit values it
  has a relative angle error below 2.4e-16 from 1e-8 rad to 2π−1e-8, and the
  round-trip error over [1e-8, π−1e-8] is **1.55e-15**.
  `TestQuatAxisAngleRoundTripNearDegenerate` is now an ENFORCED guard (it fails
  RED), no longer a SKIP.
- **Claim (as originally found):** "Precision: 1e-12 (transcendental
  functions)" — stated unconditionally.
- **Observed then:** worst rotation-action round-trip error **4.43e-11** for
  angles down to 1e-6 rad from 0 / π (~44× the claim), **9.96e-9** down to
  1e-8 rad; below ~2e-8 rad the function returned angle 0 and the wrong axis.
- **Cause (corrected):** this WAS an implementation defect, not only a property
  of the representation. `w = cos(angle/2) = 1 − angle²/8` near `angle → 0`
  (and `−1 + …` near 2π), so `acos(w)` loses all precision there: `w` rounds to
  1 below ~2e-8 rad, and before that the absolute error is up to ~2e-16/angle
  (6e-5 relative at 1e-6 rad). `|v| = sin(angle/2)`
  keeps its relative precision. The original text also named `angle → π` as
  degenerate; it is not (`w → 0` is harmless, the error there was already
  ~1e-16). Only the axis of a rotation by nearly 0 or 2π is intrinsically
  ill-conditioned (the vector part is tiny).
- Test: `TestQuatAxisAngleRoundTripNearDegenerate` (PASS, enforced) +
  `TestQuatAxisAngleRoundTripWellConditioned` (PASS) +
  `TestQuatToAxisAngle_MatchesTheExactAxisAndAngle` (oracle rows, 50-digit).

### 4. `BinomialCoeff` — CAVEAT: `< 1e-12` was exceeded for large n (counting.go) — RESOLVED
- **Status: resolved.** `BinomialCoeff` and `Permutations` now return the
  float64 nearest to the exact integer, so they are exact wherever it is
  representable: 0 wrong over every 0 <= k <= n <= 300 (it was 39,562 and
  31,035; 242 binomials were wrong already for n <= 60, from C(49,20)).
  `TestBinomialRelErrLargeN` is ENFORCED (relative error <= 2^-53).
- **As originally found:**
- **Claim:** "relative error < 1e-12 for typical inputs".
- **Observed:** worst **2.45e-12 at C(990,86)** (large n); also ~1.1e-12 at
  C(420,12). For n <= 200 the bound holds comfortably (3.09e-13, PINNED PASS).
- **Cause:** accumulated `lgamma` error in `exp(lg(n) - lg(k) - lg(n-k))`.
- **Honest framing:** softer than the above — "for typical inputs" arguably
  scopes out n in the high hundreds. Recorded as a CAVEAT for large-n callers
  (expect ~few×1e-12), not a hard contract violation.
- Tests: `TestBinomialRelErrLargeN` (ENFORCED) + `TestBinomialRelErrTypical` (PASS).

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
