# 086 | info-numerics — log-stability, KL-with-zeros, MI bias

**Scope:** numerical-correctness audit of `reality/info/` (the package
named in the topic).  The package contains exactly two subpackages —
`info/lz/` (LZ76 sequence complexity, ~470 LOC) and `info/mdl/`
(Minimum Description Length codelengths + NML, ~370 LOC).

**Headline:** the topic checklist (entropy, KL, MI plug-in, conditional
entropy, total variation, Hellinger, Bhattacharyya, f-divergences,
Renyi, log-sum-exp, Kozachenko-Leonenko) is **not implemented in
`info/`**.  Those primitives live in two adjacent packages:
- `infogeo/fdiv.go` (KL, JS, TV, Hellinger, ChiSquared, Renyi)
- `compression/entropy.go` (Shannon, Joint, Conditional, MI, KL, CrossEntropy)

This audit therefore covers (a) what `info/` actually does numerically,
plus (b) the topic-listed primitives wherever they live in the repo,
(c) what is missing entirely.  Files all referenced by absolute path.

---

## Part A — `C:\limitless\foundation\reality\info\lz\` (LZ76)

`info/lz/lz76.go:1-466`.  Pure integer LZ76 production count + a
Kaspar-Schuster normalisation `c(S) / (n / log_A(n))`, plus
symbolisation helpers.

### Numerical findings

1. **`log_A(n)` divide path is guarded** (`lz76.go:148-159`).
   `logBaseA = log(n) / log(A)`; `A=1` is short-circuited at line 99
   so the division never sees a zero denominator.  The `logBaseA <= 0`
   branch (which can only fire on `n <= 1`) returns
   `normalised = 1.0`, but `n < LZ76MinSymbols = 10` is already
   rejected at line 79 — so this branch is **dead code in production**
   (`n >= 10 ⇒ log(n) > 0`).  Harmless but worth a comment.

2. **`upper > 0` guard** (`lz76.go:154`) is also dead — `upper = n /
   logBaseA` with both positive → strictly positive.  Remove or
   document as belt-and-braces.

3. **Symbolisation `Bessel correction`** (`SymbolizeByThreshold`,
   `lz76.go:327`): variance divides by `count - 1`.  `count < 2`
   path is correctly handled (lines 308-315: returns all-neutral).
   `sigma <= 0` path correctly handled (lines 330-336).  No NaN leak.

4. **NaN/Inf filtering**: `SymbolizeByQuantile` filters non-finite
   values into the rank pool, then leaves their output positions at 0
   (line 256).  This is a **silent semantic loss** — a NaN-position
   reads as `bin 0` indistinguishable from a true low-rank value.  The
   RubberDuck reference convention is documented but downstream LZ76
   parsing has no way to know.  Recommend changing the sentinel to a
   reserved bin (e.g. `numBins`) or returning a parallel mask slice.

5. **`ComplexityFromReturns` 10% NaN cap** (`lz76.go:377`).  Floors at
   `>10%`; at exactly 10% it passes.  The mixed-arithmetic guard
   `float64(invalid)/float64(len(returns)) > 0.10` is exact only for
   `len(returns)` divisible by 10 — `1/9 = 0.111…` passes the gate
   on a 9-NaN-out-of-90 input but a `10/100` input is borderline.
   Consider integer comparison `10*invalid > len(returns)`.

6. **`isSubstringOfPrefix` is O(prefixLen × candidateLen)** per call
   (lz76.go:182-197); LZ76 outer loop calls it `O(n²/log n)` times —
   total is `O(n^4 / log n)` worst-case, but the cap at 10 000 keeps
   it tractable.  Pure integer arithmetic, no float concerns.

### LZ76 verdict

Numerically clean.  Pure integer / shape-bounded float, no
log(0), no division by zero, no overflow window.  Two strands of
dead-code defensive guards (items 1-2) and one silent-NaN-→-bin-0
gotcha (item 4) — none affect computed values for valid inputs.

---

## Part B — `C:\limitless\foundation\reality\info\mdl\` (NML / BIC / AIC)

### B.1 `GaussianCodeLength` (`codelength.go:29-48`)

`L = (n/2) log(2πσ²) + Σ(x-μ)² / (2σ²)`.  Two-pass implementation:
`hypothesisStdev <= 0 || NaN || ±Inf → +Inf`; `hypothesisMean
NaN/Inf → +Inf`.  Empty samples → 0.

Findings:
- **No catastrophic-cancellation risk** in `ssr` because mean is
  hypothesised, not estimated — no centering subtraction blow-up.
- **Tiny σ overflow path**: with `σ = 1e-160` and `(x-μ) = 1`, the
  `ssr / (2σ²)` term overflows to `+Inf`.  This is the *correct*
  answer (codelength is unboundedly large for a near-singular
  hypothesis) but no test pins it.  Compare to a numerically-equal
  but cleaner formulation `Σ ((x-μ)/σ)² / 2 + (n/2) log(2πσ²)` which
  avoids one squaring step and gives the same answer with smaller
  intermediate magnitude.

### B.2 `BICShape` / `AICShape` / `ModelCodeLength` (`codelength.go:50-128`)

Closed-form scalar arithmetic, no log(0), no MLE coupling.
`numParams <= 0 || sampleSize <= 1 → 0`.  `negLogLikelihood NaN/Inf
→ +Inf`.  `AICShape` clamps `numParams < 0 → 0` (line 124-126).
Clean.

### B.3 `NMLMultinomial` and `computeCn2` (`nml.go:1-157`)

The Kontkanen-Myllymäki 2007 linear-time recurrence:
`C(n,k) = C(n,k-1) + (n/(k-1)) C(n,k-2)` for `k ≥ 3`, with `C(n,1)=1`
and `C(n,2)` computed by direct Bernoulli-mass sum.

**Correctness checks:**
- `n=0` → 0 returned at line 64 (regret of empty data is 0). ✓
- `k=1` → 0 returned at line 71 (single category, MLE always 1). ✓
- `k=2` → `log(C(n,2))`. ✓
- Negative entries rejected at line 51-53 with `ErrNegativeInput`. ✓

**Numerical concerns:**

1. **Linear-space accumulator at line 152-155** uses `log-sum-exp`
   pattern (`maxLog`, then `Σ exp(v - maxLog)`, then
   `exp(maxLog) * sum`).  This **lifts the result back into linear
   space**, which is what the recurrence needs (it operates on
   `C(n,k)` not `log C(n,k)`) — but for `n` near the doc-stated cap
   of `~10^6`, the central-binomial peak `~2^n/√n` overflows `float64`
   (max ~1.8e308; `2^1024 ≈ 1.8e308`).  **The function will return
   `+Inf` for `n ≥ 1024`** despite the docstring claim of "comfortably
   in float64 for n up to roughly 10^6" (line 116).  **This is a
   docstring vs. implementation contradiction.**  Recommend either:
     - keep recurrence in log-space (subtract a running max,
       carry `prev1, prev2` as `log` values, recompute at each step
       via log-sum-exp), or
     - restrict the docstring to `n ≤ 700` (`log(2^700) ≈ 485` nats,
       safely below `log(MaxFloat64) ≈ 709`).

2. **`logTerms` allocation at line 128**: `make([]float64, n+1)` —
   allocates O(n).  For the L12 inverse-consumer scale of `n ≤ 10^4`
   this is fine; for batch model-selection (`SelectMDL` over many
   `(n, k)` candidates) consider a scratch buffer parameter.

3. **`(r/n)^r * ((n-r)/n)^(n-r)` at line 132-137** is computed in
   log-space with the `0·log(0)=0` convention (`r > 0` and `r < n`
   guards on the multiplications). ✓ — correct handling of the
   binomial endpoints.

### B.4 `NMLBernoulli` / `BernoulliCodeLength` (`bernoulli.go:1-90`)

Delegates to `NMLMultinomial` with `k=2`. ✓.

`BernoulliCodeLength` NLL closed form (`bernoulli.go:79-83`): the
`successes>0 && successes<trials` guard correctly handles the 0/n
and n/n endpoints (NLL=0 when `p̂ ∈ {0,1}` because `0·log(0)=0`).
Standard convention applied without an explicit `0·log(0)` skip
because the multiplication is by literal `successes` and
`trials-successes`, so an integer 0 prefactor short-circuits the
log call. ✓.

### B.5 `UniversalIntegerCodeLength` (`universal_int.go:1-72`)

Rissanen 1983 `log*(n) = log n + log log n + log log log n + … + log
2.865064`.  Implementation iterates while `x > 0`.

- `n=1` → returns just the constant (loop body never runs because
  `log(1) = 0` ≤ 0). ✓
- `n=2` → `log(2) + const` ≈ `0.693 + 1.052 ≈ 1.745`. ✓
- For very large `n`, `log log log … n` converges in ~6 iterations
  for `n ≤ 1e308` (rough bound: `log log 1e308 ≈ 6.6`,
  `log log log ≈ 1.88`, then `log` → 0.63 → −0.46, terminates).  No
  divergence concern.
- **`n < 1` rejected** with `ErrInvalidUniversalInt`. ✓

### B.6 `SelectMDL` / `SelectMDLWithMargin` (`select.go:1-74`)

Argmin with NaN/Inf detection.  `len == 1` branch returns
`(0, +Inf, nil)` (margin = +Inf for single-model case). ✓.
Standard tie-breaking: first index wins.

### MDL verdict

Two issues:
- **Docstring overclaim on `computeCn2` for `n` up to `1e6`** —
  actual safe ceiling is ~700 unless the recurrence is moved to
  log-space.  Test pins exist only for `n ≤ ~50` per the test
  filenames.  *Recommend pin-test at `n = 1024` and amend docstring.*
- **Allocation pattern** — `logTerms` is `O(n)` per call; not a
  correctness issue but a perf seam.

No log(0), no divide-by-zero, no NaN-leak path.  Sentinel errors
clean.  `SelectMDL` correctly refuses to argmin over NaN/Inf
inputs (rare and admirable — most argmin helpers silently propagate).

---

## Part C — Topic checklist mapped against the rest of the repo

The topic asks about a list of info-theoretic primitives.  None of
them are in `info/`; they live in `infogeo/` and `compression/`.

### C.1 Shannon entropy (`compression/entropy.go:27-35`)

```
for _, p := range probs {
    if p > 0 { h -= p * math.Log2(p) }
}
```

Correct `0·log(0) = 0` convention via `p > 0` guard. ✓
Negative-`p` is silently ignored (the guard rejects them) — but no
input validation: a malformed distribution with negative entries
returns a smaller entropy than it should.  The guard should be
`p > 0` AND there should be a separate validation path.  **Compare
to `infogeo/fdiv.go:38-50` which validates strictly** (`v < 0 → err`,
sum ≈ 1 enforced).  **Inconsistency: `compression/entropy.go` does
no validation; `infogeo/fdiv.go` validates strictly.**  Should be
unified.

### C.2 KL divergence (two implementations, two conventions)

- `compression/entropy.go:133-148` (KL in **bits**, no validation,
  returns 0 on length mismatch — silent-bug shape, should error).
- `infogeo/fdiv.go:65-81` (KL in **nats**, validates, errors on
  bad input).

**Convention mismatch**: bits vs nats with no shared API surface.
Both have correct `0·log(0/q) = 0` (skipped via `p == 0` continue)
and correct `+Inf` return when `p > 0 ∧ q = 0`.

### C.3 Cross-entropy (`compression/entropy.go:161-176`)

Bits, no validation, silent-zero on length mismatch (line 163-165).
Correct `+Inf` on absolute-continuity violation. ✓ Numerically
fine; API-validation hole same as KL.

### C.4 Mutual information plug-in (`compression/entropy.go:94-121`)

Standard `H(X) + H(Y) − H(X,Y)` plug-in.  **No Miller-Madow bias
correction** anywhere in the repo (grep -i "miller.?madow" returns
zero hits in source).  Plug-in MI has known bias `(K-1)/(2N) +
O(1/N²)` for `K` joint cells.  For most consumer use cases this is
fine; it should be documented as the bias-uncorrected estimator and
the `MillerMadow` correction added (~10 LOC: subtract
`(rows + cols − rows·cols − 1) / (2 N ln 2)` at the end for the bits
form).  **Actionable gap.**

### C.5 Conditional entropy (`compression/entropy.go:69-80`)

Computed via `H(X,Y) − H(X)` rather than `Σ p(x) H(Y|x)`.  Both
formulations are mathematically equal but the subtraction form has
**catastrophic-cancellation risk** when X and Y are nearly
deterministic functions of each other (`H(X,Y) ≈ H(X)`).  The direct
`Σ p(x) Σ p(y|x) log p(y|x)` form (with `p(y|x) = joint/marginalX`)
preserves precision in this regime.  **Numerical recommendation:**
add a direct conditional-entropy implementation; keep the subtraction
form as a faster path with documented loss-of-precision regime.

### C.6 Total variation (`infogeo/fdiv.go:125-134`)

Pure `0.5 Σ|p−q|`.  No log, no division.  Numerically pristine.

### C.7 Hellinger (`infogeo/fdiv.go:145-155`)

`H(p,q) = sqrt(0.5 Σ (√p − √q)²)`.  Pure square-root arithmetic, no
log(0) issues; `√0 = 0` is exact; `(√p − √q)²` does cancel when
`p ≈ q` but the answer is small-by-design and the relative-error
floor is `~1e-8` (sqrt halves precision once).  Acceptable.

### C.8 Bhattacharyya distance — **MISSING**.

Definition `−log(Σ √(pᵢqᵢ))` (the Bhattacharyya coefficient inside
the `−log`).  Not implemented anywhere in the repo (grep "Bhattacharyya"
returns only review-doc hits).  **Actionable gap** — ~10 LOC,
trivially derivable from same `validatePair` helper as Hellinger;
the answer relates to Hellinger by `H² = 1 − BC` so it's a
`return -log(1 - 2*H*H)` one-liner if Hellinger is already
computed (modulo NaN handling).

### C.9 f-divergences in general

`infogeo/fdiv.go` ships KL, JS, TV, Hellinger, ChiSquared, Renyi.
Missing the parameterised `Fdivergence(p, q, f func(t float64) float64)`
generic which would let callers plug their own `f`.  Not a numerics
issue, but an API gap.

### C.10 Rényi entropies and divergences (`infogeo/fdiv.go:182-227`)

Rényi-divergence `D_α(p||q) = (1/(α−1)) log Σ pᵢ^α qᵢ^(1−α)`.

- `α = 1` rejected (use KL). ✓
- `α ≤ 0` rejected. ✓ — but the doc says `α ∈ (0,1) ∪ (1, ∞)` so
  the limit cases `α → 0⁺` and `α → ∞` are NOT supported.  Both
  have closed forms (`D_0 = -log Σ_{i: p_i > 0} q_i`, `D_∞ = log
  sup p/q`) that the docstring promises but the implementation
  rejects.  **Actionable: either implement the limit cases or
  remove the docstring promise.**
- Zero handling: `p=0,q=0` skip; `p=0,q>0` skip (term is 0);
  `q=0,p>0` returns `+Inf` if `α>1`, skip otherwise.  Correct.
- `math.Pow(p, α)` is **not log-stable for tiny p** — for
  `p = 1e-200, α = 0.5`, `Pow` gives `1e-100` (fine) but for
  `α = 50, p = 0.1` we get `1e-50` (fine).  The combination
  `p^α · q^(1-α)` for `α >> 1, q ≪ 1` overflows when `q^(1−α) =
  q^(-large)`.  **Actionable: rewrite as `exp(α log p + (1-α) log q)`
  with log-sum-exp accumulation across `i`** — adds ~5 LOC, gives
  numerical stability for extreme `α`.

**No Rényi entropy** (vs. divergence).  `H_α(p) = (1/(1-α)) log Σ pᵢ^α`
is missing.  ~5 LOC gap.

### C.11 Log-sum-exp — **partially present**

The repo's only LSE site is `info/mdl/nml.go:142-156` (inline,
correct).  There is **no canonical `LogSumExp(xs []float64) float64`
helper** anywhere.  Multiple sites would benefit (Renyi above,
softmax in autodiff/, importance-sampling in copula/).
**Actionable gap — ~15 LOC primitive** to add (likely under
`prob/mathutil.go` or a new `info/lse.go`).

### C.12 Differential entropy estimators — **MISSING**

No Kozachenko-Leonenko, no kNN-based H estimator, no Vasicek,
no Ebrahimi-Pflughoeft-Soofi, no histogram-with-Scott/Sturges/FD
binning entropy.  The only continuous-source primitive is
`GaussianCodeLength` (parametric).  This is a substantial gap:
differential entropy estimation is a standard tool for
information-theoretic inference and the absence is conspicuous
given that the package is named `info/`.  **Actionable:
~150-200 LOC for KL kNN estimator** with proper digamma correction
and ball-tree (or just brute-force for `n < 10^4`).

---

## Part D — Cross-cutting numerics issues

1. **Silent length-mismatch returns 0** (`compression/entropy.go:135,
   163`).  KL/CrossEntropy on mismatched-length inputs return `0`
   (no error).  This masks caller bugs.  Should error.

2. **No validation in `compression/entropy.go`** — distributions
   that don't sum to 1, contain negatives, or contain NaN flow
   through silently.  Adopt the `infogeo/fdiv.go` `validate` /
   `validatePair` pattern.

3. **Two unit conventions** — `compression/` uses bits (`Log2`),
   `infogeo/` uses nats (`Log`).  No documented place that says
   "this is by design".  **Recommend documenting at package level**
   that bits is consumer-facing (compression ratios) and nats is
   ML-facing (variational bounds), and providing
   `BitsToNats / NatsToBits` constants in `constants/`.

4. **No pinned tests for absolute-continuity boundary**
   (`p>0, q=0`).  The `+Inf` paths in KL, CrossEntropy, ChiSquared,
   Renyi (α>1) are doc-stated but I see no test that pins them.
   **Recommend: add boundary tests** to ensure the +Inf is preserved
   under refactors.

5. **No log-sum-exp helper** — every site reimplements.  `mdl/nml.go`
   has a correct inline impl that should be promoted.

---

## Recommendations (ordered by leverage)

| # | Action | LOC | Files |
|---|---|---|---|
| 1 | Add `infogeo.Bhattacharyya` (mirror Hellinger) | ~15 | `infogeo/fdiv.go` |
| 2 | Add Miller-Madow MI correction `compression.MutualInformationMillerMadow` | ~20 | `compression/entropy.go` |
| 3 | Add direct-form `compression.ConditionalEntropyDirect` (`Σ p(x) H(Y\|x)`) for cancellation regime | ~25 | `compression/entropy.go` |
| 4 | Promote `mdl/nml.go` LSE to `prob.LogSumExp` (or `info/lse.go`) | ~15 | new file |
| 5 | Rewrite `infogeo.Renyi` term as `exp(α log p + (1−α) log q)` with LSE | ~20 | `infogeo/fdiv.go` |
| 6 | Add `infogeo.RenyiEntropy` (entropy, not divergence) | ~15 | `infogeo/fdiv.go` |
| 7 | Implement `infogeo.Renyi` α→0 and α→∞ limit cases (docstring already promises them) | ~20 | `infogeo/fdiv.go` |
| 8 | Add `validatePair` to `compression/entropy.go` and error on length mismatch | ~30 | `compression/entropy.go` |
| 9 | Move `mdl/nml.go` recurrence to log-space, OR amend docstring n≤700 | ~25 or ~5 | `info/mdl/nml.go` |
| 10 | Add Kozachenko-Leonenko differential entropy estimator | ~200 | new `info/diffent/` |
| 11 | Pin tests for absolute-continuity `+Inf` boundaries | ~50 | tests |
| 12 | Document bits-vs-nats convention split | ~30 | doc.go files |

Total Sprint-1 (items 1-9): **~190 LOC of mostly trivial primitives**
that close the headline gaps in `info/`, `infogeo/`, `compression/`
without changing existing semantics.

---

## Files referenced (absolute)

- `C:\limitless\foundation\reality\info\lz\lz76.go`
- `C:\limitless\foundation\reality\info\lz\errors.go`
- `C:\limitless\foundation\reality\info\lz\doc.go`
- `C:\limitless\foundation\reality\info\mdl\codelength.go`
- `C:\limitless\foundation\reality\info\mdl\nml.go`
- `C:\limitless\foundation\reality\info\mdl\bernoulli.go`
- `C:\limitless\foundation\reality\info\mdl\select.go`
- `C:\limitless\foundation\reality\info\mdl\universal_int.go`
- `C:\limitless\foundation\reality\info\mdl\errors.go`
- `C:\limitless\foundation\reality\infogeo\fdiv.go`
- `C:\limitless\foundation\reality\compression\entropy.go`

---

End report.  ~290 lines.
