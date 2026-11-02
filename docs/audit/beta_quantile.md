# BetaQuantile: method and measured error

`prob` now has the inverse of `RegularizedBetaInc` in x. It adds three exported functions in `prob/betaquantile.go`:

| Function | Solves | Returns |
|---|---|---|
| `BetaQuantile(a, b, p)` | I_x(a, b) = p | x |
| `BetaQuantileComplement(a, b, q)` | 1 - I_x(a, b) = q | x, without forming 1 - q |
| `BetaQuantileXY(a, b, p)` | I_x(a, b) = p | x and y = 1 - x, each to full relative accuracy |

The name `BetaQuantileXY` follows the package's own notation. `RegularizedBetaInc` and `incBeta` call the two variables
x and y = 1 - x, and `incBeta` returns the pair (w, w1). Near x = 1, x rounds to 1 and only y still carries the
answer. For example, `BetaQuantileXY(5, 0.01, 0.999)` returns x = 1 and y = 1.25e-301.

## Domain

| Input | Result |
|---|---|
| p (or q) <= 0, >= 1, or NaN | NaN |
| a or b <= 0, NaN, or infinite | NaN |
| `RegularizedBetaInc` returns NaN (near the mean of shapes above 2^52) | NaN |

This matches the neighbours. `NormalQuantile` and `ExponentialQuantile` return NaN for p <= 0 and p >= 1, even
`ExponentialQuantile`, which also has a finite answer at p = 0. `RegularizedBetaInc` and `BetaCDF` return NaN for
shapes that are not finite and positive. `BetaQuantileXY` returns NaN for both values together. A quantile below the
smallest double comes back as 0 (or as x = 1 with y = 0). The table has no such cases.

## Method

1. **Exact target.** Of p and q = 1 - p, the smaller one is exact as a double: either it is the argument, or
   Sterbenz's lemma makes 1 - r exact for r >= 1/2. Only that one, t, enters the residual. That is why
   `BetaQuantileComplement` keeps a q of 1e-300.
2. **Side.** One evaluation at x = 1/2 shows whether the root lies in x <= 1/2 or in y <= 1/2. The iteration runs in that
   small variable s. The other one is formed at the end as 1 - s, which is at least 1/2 and so exact to half an ulp.
   `incBeta` takes x and y separately, so the small one is never rounded through the large one.
3. **Residual.** It is g = ln F(s) - ln t when t is F's own tail, and g = ln t - ln(1 - F(s)) otherwise. Both
   increase in s. `incBeta` returns each tail directly, so neither is formed by subtraction.
4. **Newton in ln s.** The derivative is s f(s)/F (or s f(s)/(1 - F)). The prefix s^a (1-s)^b / B(a, b) comes from
   `betaPrefixParts`, which has no cancellation at large shapes. In the tails, F ~ s^a is almost linear in ln s, so
   the step converges from far away. The update s·exp(-step) costs one rounding per step. Evaluating exp(u) instead
   would cost |u| ulps.
5. **Bracket and bisection (rtsafe).** Every residual sign narrows a bracket [lo, hi], starting from [0, 1/2]. A
   Newton step is replaced by bisection in ln s in two cases: when it leaves the bracket, or when it is not at most
   half the step before last. Until a point with g < 0 has been seen, bisection steps down from hi by 2^-32. The
   iteration stops at three points: when the Newton correction is below 2^-52, when the bracket is down to 4 ulps,
   or after 200 iterations.
6. **Start.** For a, b >= 1 the start is Abramowitz & Stegun 26.5.22, from the normal deviate (`standardNormalQuantile`).
   For smaller shapes it is the tail form F ~ s^a / (a B(a, b)). The start affects only speed, never the result.

## Measured error

The oracle is `testdata/stress/beta_quantile.json`: 101 cases solved with mpmath at 800 digits, SHA-256
`fccd2497…f8b221`, copied unchanged. "Core" means a, b in [0.5, 100] and p in [1e-6, 0.999] (42 cases). Errors are
relative, measured against the 40 stored digits. Default (GOAMD64=v1) and FMA (GOAMD64=v3) amd64 builds gave identical
numbers.

| Region | Check | Bound (VERIFY.md) | Measured max | Where |
|---|---|---|---|---|
| Core | `BetaQuantile` vs x | 1e-12 | 3.8e-15 | a = b = 0.5, p = 1e-6 |
| Non-core | `BetaQuantile` vs x | 1e-9 | 6.0e-14 | a = b = 0.1, p = 1e-30 |
| All | `BetaQuantileXY` y vs 1 - x | 1e-9 | 2.2e-13 | a = 5, b = 0.01, p = 0.999 |
| All | `BetaQuantileComplement(b, a, p)` vs 1 - x | 1e-9 | 2.2e-13 | same row, swapped |
| Core | `RegularizedBetaInc(BetaQuantile(a,b,p), a, b)` vs p | 1e-10 | 1.2e-12 | a = 100, b = 0.5, p = 0.999 |

By region, for `BetaQuantile` vs x:

| Region | Max relative error |
|---|---|
| Core | 3.8e-15 |
| Far tails, p <= 1e-10, shapes 0.5 to 20000 | 2.7e-14 (a = 0.5, b = 100, p = 1e-30) |
| Shapes 1000 to 20000, any p | 1.3e-16 |
| Small shapes (0.01, 0.1) | 6.0e-14 |
| y = 1 - x, core | 2.2e-15 |

Where the error comes from:

- **The forward function.** The quantile inherits the error of I_x. A relative error d in the tail moves x by about
  d/a. At a = 0.1 or 0.01 that multiplies `RegularizedBetaInc`'s error (up to about 4.5e-16·|ln I| in the far
  tails) by 10 or 100. At a = b = 1, p = 1e-300, the forward function evaluates x^a as exp(a ln x), and that exponential
  costs 690 · 1.1e-16 = 7.6e-14 at most (2.4e-14 measured). The solver adds about one ulp to this.
- **The double input.** The y = 1 - x figures at b = 0.01 come from p itself. The double nearest 0.999 differs from
  0.999 by 8.9e-16 relative in 1 - p, and the shape 0.01 amplifies that about 100 times. The function solves its double
  input to ~1e-15. The table's x is for the decimal p.
- **Rounding of x near 1.** The round-trip figure is limited by x itself. At a = 100, b = 0.5, p = 0.999,
  1 - x = 7.9e-9, so the half-ulp rounding of x moves I_x by about 1e-12 relative. Through (x, y), the round trip is
  ~1e-15.

The registered claims in `testdata/stress/precision_beta.json` round these maxima up, which leaves room for the float64
rounding of each reference value:

| Function | Core | Whole table |
|---|---|---|
| `BetaQuantile` | 4e-15 | 7e-14 |
| `BetaQuantileXY` y | 3e-15 | 2.5e-13 |
| `BetaQuantileComplement` | 2.5e-13 | 2.5e-13 |

The doc comments state the same numbers.

## Cost

| Case | Incomplete-beta evaluations |
|---|---|
| Every table case except one | 2 to 12 |
| The symmetric median at a = b = 0.1 | about 35 |
| 100,000 random cases (shapes 1e-3 to 1e7, p down to 1e-300) | half need at most 4 and 84% at most 14; most of the rest (14%) take 30 to 34, nearly all of them quantiles that underflow |
| A quantile that underflows to 0, or rounds to 1 with 1 - x below the double range | about 35 |
| Worst measured | 131 |
| Cap | 200 |

## Tests

- **`prob.TestBetaQuantileOracle`** reads the top-level `rows` key of the table. It calls `t.Fatal` unless it decodes
  exactly 101 rows, 42 of them core, and it enforces every VERIFY.md bound above. It never skips. It fails on the base
  commit because the functions do not exist there.
- **`prob.TestBetaQuantileInvalid`** covers the NaN cases.
- **`prob.TestBetaQuantileClosedForms`** checks shapes with an exact inverse: I_x(1,1) = x, I_x(a,1) = x^a, and
  1 - I_x(1,b) = (1-x)^b.
- **`prob.TestBetaQuantileRoundTripProperty`** is a seeded `testing/quick` check over 5,000 draws, with shapes 0.05
  to 500 and p from 1e-3 to 0.999. It checks monotonicity in p, that x + y = 1, and that the round trip through
  (x, y) is within 1e-10.
- **Precision registration.** `testdata/stress/precision_beta.json` holds 303 cases, three per table row. It is
  generated from the table by `tools/stressgolden/beta.py`, using the standard library only. `precision_beta_test.go`
  registers the evaluators `prob.BetaQuantile`, `prob.BetaQuantileComplement` and `prob.BetaQuantileXY.y` from `init`.
  `TestPrecisionClaims` runs them, and no violation is listed.
- The table's generator is kept as `tools/stressgolden/beta_quantile.py` (mpmath, 800 digits). Do not regenerate it
  at lower precision.

## Not run here

- **arm64.** `GOARCH=arm64 go vet ./...` only compiles. Floating-point results can differ there, because arm64 fuses
  multiply-adds. Only the pull request's arm64 job proves them. The amd64 FMA build (GOAMD64=v3) gave bit-identical
  errors, which is some evidence but not proof.

## Follow-up

`prob/trend.go` (`StudentTQuantile`, and the crossing search in `TrendCrossing`) and `prob/robust.go` (`CatoniMean`)
each bisect by hand. They were left alone, as this change adds one function and changes none. `StudentTQuantile` in
particular inverts `studentTCDF`, which is the regularized incomplete beta function, by plain bisection to ~1e-10
absolute. It could be expressed through `BetaQuantileXY`: t^2/(df + t^2) is a beta variate. That would give it the
accuracy measured here, including in the far tails. That is a separate change.
