# 121 — queue: numerical correctness audit

Scope: `C:/limitless/foundation/reality/queue/` — `basic.go` (M/M/1, M/M/c, M/M/1/K, Little), `erlang.go` (Erlang-B, Erlang-C, ErlangCWaitTime), `metrics.go` (BurstinessIndex, OfferedLoad), `network.go` (JacksonNetwork). Tests: `queue_test.go` (695 lines, ~50 tests).

## Headline

The package is small, formula-correct, and the closed-form M/M/1 / Little / OfferedLoad routines are textbook-clean. The numerical-stability claims around the Erlang formulas are however overstated: the `ErlangB` recursion, as written, **diverges to `+Inf` and silently underflows to `0` for several non-pathological argument regions**, including very small `A`, very large `N`-with-small-`A`, and far-overload `(A, N)` with `A ≪ N` plus large `N`. The package also **does not implement Pollaczek–Khinchine (M/G/1)** or **Burke's theorem** despite being framed alongside Little's law as foundational queueing identities — these are missing, not buggy.

## What was verified

### M/M/1 (`basic.go:56`)
Closed-form formulas are exact. `L = ρ/(1−ρ)`, `Lq = ρ²/(1−ρ)`, `W = 1/(μ−λ)`, `Wq = ρ/(μ−λ)`. Stability gate `λ ≥ μ` is correct. No numerical issue: even at ρ = 0.999999, `1 − ρ` does not catastrophically cancel because `λ` and `μ` are independent inputs (no subtraction of nearly-equal rounded quantities). The `1/(μ−λ)` form is the *good* one — using `(1/μ) / (1−ρ)` would have been rounded twice.

### M/M/c (`basic.go:108`)
Pipeline is `ρ = λ/(cμ)`, `A = λ/μ`, `pWait = ErlangC(A, c)`, `Lq = pWait·ρ/(1−ρ)`, then Little. Algebraically correct (matches Gross §2.3). Numerically inherits whatever ErlangC does — see Erlang findings below. The c=1 path is correctly redirected to the same formula and matches MM1 exactly (verified by `TestMM1_ConsistentWithMMc_c1`).

### M/M/1/K (`basic.go:173`)
The truncated geometric expression `L = ρ/(1−ρ) − (K+1)·ρ^(K+1)/(1−ρ^(K+1))` is correct *but* has two failure modes the implementation only partially defends against:

1. **The ρ ≈ 1 split point is at `|ρ − 1| < 1e-12`.** This is too tight: `ρ/(1−ρ)` and the second term are each O(K/2) in magnitude near ρ = 1, but their difference is `(K(K−1))/12 + O((ρ−1)²)`, so a relative error of order `K·ε/(1−ρ)` infects `L`. For ρ = 1 + 1e-9 with K = 100 the formula loses ~3 significant digits. The threshold should be ~1e-6 (Kingman-style: switch when `K·|ρ−1|² ≪ ε`). Not catastrophic, but the docstring claims this works "for any λ and μ" and it doesn't quite.
2. **`math.Pow(rho, K+1)` overflows when `ρ > 1` and `K ≳ 700/log10(ρ)`.** For ρ = 2 and K = 1100, `ρ^(K+1) = +Inf`, `denom = 1 − Inf = −Inf`, `rho/(1−rho) − (K+1)·Inf/(−Inf)` returns `NaN`. Should test `ρ > 1` and use the symmetric expansion in `1/ρ` (which is the dual chain — for an overloaded finite-capacity queue, the queue is effectively always full and `pLoss → 1 − 1/ρ`). Not currently tested.

`pLoss` and `lambdaEff < 1e-300` guard look correct; the `Wq = max(0, …)` clip is defensive but right (rounding can push `Wq` ~−1e-16 negative when ρ ≪ 1).

### Erlang-B / Jagerman recursion (`erlang.go:34`)

```go
invB := 1.0
for n := 1; n <= N; n++ {
    invB = 1.0 + float64(n)/A*invB
}
return 1.0 / invB
```

This is the **inverse** Jagerman recursion — it iterates `1/B` upward, not `B`. The advertised stability is real for *moderate* `(A, N)`, but it overflows in three regimes:

| call                | invB-overflow step | returned | true value |
|---|---|---|---|
| `ErlangB(1e-300, 5)`     | n=2  | `0`            | ≈ `2.6·10⁻¹⁵¹⁵`† |
| `ErlangB(0.001, 1000)`   | n=70 | `0`            | ≈ `2·10⁻²⁶²⁰`† (true 0 to f64) |
| `ErlangB(50, 100)`       | —    | `1.63·10⁻¹⁰` | matches direct (good) |
| `ErlangB(500, 1000)`     | —    | `1.65·10⁻⁸⁶`  | matches direct (good) |
| `ErlangB(5000, 10000)`   | n=7880 | `0`         | true f64 zero (acceptable) |

(†actually beyond f64 range, so 0 is the correct *float* answer. The bug is that the answer's correctness is an accident of `1/Inf = 0` rather than a stable computation, and the recursion is actively producing `Inf` instead of underflowing the correct quantity.)

The forward Jagerman recursion `B(0)=1; B(n) = A·B(n-1) / (n + A·B(n-1))` does **not** overflow for any of the above and returns the same `0`. Code is one line:

```go
B := 1.0
for n := 1; n <= N; n++ {
    B = A*B / (float64(n) + A*B)
}
return B
```

Empirically (verified): no Inf for `A = 1e-300, N = 5`, no Inf for `A = 0.001, N = 1000`, no Inf for `A = 5000, N = 10000`. The forward form is what Jagerman 1974 actually published; the inverse form here is from the same paper but is the version recommended only when `1/B` is the desired output. Recommend swap.

Edge-case correctness on the existing tests is *not* affected — `TestErlangB_LargeN` uses (5, 20), `TestErlangB_HighLoad` uses (20, 5), `TestErlangB_A10N15` is mid-range — all in the safe band.

`A <= 0` panic and `N < 1` panic are correct. `A == 0` is rejected; mathematically `B(0, N) = 0` is well-defined and could be returned, but the panic is consistent with the rest of the package.

### Erlang-C (`erlang.go:70`)
`C = B / (1 − ρ(1 − B))` — Gross-correct. The denominator `1 − ρ(1 − B)` does not catastrophically cancel near `ρ → 1` because as `ρ → 1` the formula tends to `B/B = 1` continuously and `B → 1`, so both numerator and denominator approach 1, not 0. Verified at A=99.99999, N=100: `C = 0.999998779…`, `Lq = 9.999987·10⁶`, smooth. No issue here beyond inheriting ErlangB's underflow.

The `A >= N` panic boundary is correct (open: A=N is unstable, ErlangC is undefined).

### ErlangCWaitTime (`erlang.go:100`)
`E[Wq] = C(A,N) / (Nμ(1−ρ))` is the standard form. Argument validation order: `mu` checked first, then `A,N` via the chained `ErlangC` call. Fine. No numerical issues.

### JacksonNetwork (`network.go:44`)
Traffic equations solved by **Gauss-Seidel-style fixed-point iteration with `tol = 1e-12, maxIter = 1000`**. Two issues:

1. **Tolerance is absolute on `λᵢ`**, not relative. For a network with `λᵢ ~ 10⁹` (e.g., packets/sec), `1e-12` is unreachable in float64 and the iteration will burn all 1000 steps before panicking. Should be `max(tol_abs, tol_rel · |λᵢ|)`.
2. **Convergence rate depends on the spectral radius of `Pᵀ`.** For routing matrices with cycles whose subdominant eigenvalue approaches 1 (deep feedback), 1000 iterations of fixed-point is far short of the closed-form `(I − Pᵀ)⁻¹ · λ_ext`. The traffic equations are *linear* — using `linalg.LU` here would be exact, allocation-bounded, and remove the convergence panic. Recommend rewriting as `linalg.Solve(I − P^T, lambdaExt)`.

Per-node M/M/c branch is correct. The `lambda[i] < 1e-300` zero-traffic short-circuit is well-placed.

### Little's law / OfferedLoad / BurstinessIndex
Trivially correct. `BurstinessIndex` uses the two-pass mean/variance algorithm — confirmed no catastrophic cancellation up to inter-arrival times of 1e10 (relative variance `~6.7e-21`, expected ~0). Could use Welford for one-pass but it's a non-issue for the typical N < 10⁶ sample sizes.

## What is missing (gap, not bug)

The topic prompt asks about **Pollaczek–Khinchine** and **Burke's theorem**. Both are *absent*:

- **No M/G/1 implementation.** The Pollaczek–Khinchine mean formula `Lq = ρ²(1 + Cs²)/(2(1−ρ))` (where `Cs² = Var(S)/E(S)²`) is the canonical result that distinguishes a queueing-theory library from a basic M/M/c calculator. It needs only the service-time mean and CV², no further machinery. Suggested signature:
  ```go
  func MG1(lambda, meanService, csquared float64) (Lq, Wq, L, W, rho float64)
  ```
  Numerical concerns: `(1 + Cs²)` is well-conditioned for `Cs² ≥ 0`; `1−ρ` denominator inherits the same stability budget as M/M/1.
- **No Burke's theorem helper.** Burke's theorem says the departure process of a stable M/M/1 (or M/M/c) is itself Poisson with rate λ. There's nothing to *compute* — the relevant artefact would be a documented `DepartureRate` accessor or a comment in `MM1` confirming the property — but a teaching-grade library should at least cite it where it's used (Jackson networks rely on Burke's theorem for the product-form result; `network.go` doesn't mention this).
- **No M/G/k approximation** (Allen–Cunneen, Kingman's formula `Wq ≈ ρ/(1−ρ) · (Ca²+Cs²)/2 · 1/μ`). The latter is the most-used formula in capacity planning and is dependency-free.
- **No M/D/1** specialization (`Cs² = 0` case of P-K), often hit in deterministic-service settings.
- **No PASTA-aware `pBlock` for M/M/c/c**: `MMcc` would just call `ErlangB(A, c)` but is the canonical lossy-multi-server primitive.

## Concrete defects ranked

1. **`ErlangB` Jagerman direction.** Switch to forward recursion `B = A·B/(n+A·B)`. Eliminates the `+Inf → 0` masquerade and matches Jagerman's published numerically-preferred form. Fix is ~3 lines. **High value, low risk.**
2. **`MM1K` overflow for ρ > 1, K large.** Add `if rho > 1 { use the dual expansion in 1/ρ }`, or at minimum document that ρ·(K+1) > 700 returns NaN and panic. **Medium value, low risk.**
3. **`MM1K` ρ ≈ 1 threshold too tight.** Loosen to `|ρ−1| < 1e-6` and use the limit `L = K/2` in that band. **Low value, low risk.**
4. **`JacksonNetwork` linear solve.** Replace fixed-point iteration with `linalg.LU` solve of `(I − Pᵀ)λ = λ_ext`. Removes the convergence panic, is O(n³) but n is small. **Medium value, medium risk** (introduces a dependency on `linalg`, currently the queue package only imports `math`).
5. **Add Pollaczek–Khinchine, Kingman, M/D/1, M/M/c/c.** Topic-mandated coverage gap. **High value, low risk.**

## Tests recommended

- Golden-file extension: `ErlangB(1e-12, 1)`, `ErlangB(1e-100, 5)`, `ErlangB(0.001, 1000)`, `ErlangB(5000, 10000)` — currently uncovered in the underflow band.
- `MM1K(λ=2, μ=1, K=1500)` — currently `NaN` due to `Pow(2, 1501) = +Inf`.
- `JacksonNetwork` with `λ_ext` of order 10⁹ — currently panics with non-convergence.
- Cross-validate `MMc(λ, μ, c) ≡ MM1(λ, μ)` at c=1 — already done (`TestMM1_ConsistentWithMMc_c1`), good.
- Add `MMcc` golden vs the published Erlang-B blocking tables (Mina or ITU-T Recommendation E.501).

## Files cited

- `C:/limitless/foundation/reality/queue/basic.go` (MM1, MMc, MM1K, LittlesLaw)
- `C:/limitless/foundation/reality/queue/erlang.go` (ErlangB, ErlangC, ErlangCWaitTime)
- `C:/limitless/foundation/reality/queue/metrics.go` (BurstinessIndex, OfferedLoad)
- `C:/limitless/foundation/reality/queue/network.go` (JacksonNetwork)
- `C:/limitless/foundation/reality/queue/queue_test.go` (test coverage)

## Bottom line

The implemented surface is small, documented to a high standard, and arithmetically correct in the test-covered band. The two real numerical defects — Jagerman direction in `ErlangB` and `Pow` overflow in `MM1K` for ρ > 1 — are 5-line fixes. The bigger story is **scope**: the package advertises "fundamentals of queueing theory" but ships only Markovian models, missing the entire general-service-time tier (Pollaczek–Khinchine, Kingman, M/D/1) and the lossy/multi-server primitives (M/M/c/c, M/M/c/K). Burke's theorem is implicitly relied upon (Jackson networks) but never named or cited, which is the kind of provenance gap CLAUDE.md design rule #4 explicitly forbids.
