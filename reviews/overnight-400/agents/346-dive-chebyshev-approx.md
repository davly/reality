# 346 — dive-chebyshev-approx (Chebyshev / Remez / Clenshaw / chebfun audit)

## Headline
Reality has zero Chebyshev approximation infrastructure despite owning a working FFT — a ~250 LOC day-1 PR (DCT-I coefficients + Clenshaw + Lobatto interpolation) unlocks special-function evaluation (slots 298–300), Parks–McClellan FIR design (slot 316), and Clenshaw–Curtis quadrature.

## Findings

### Audit (negative space)
- `Grep [Cc]hebyshev|[Rr]emez|[Cc]lenshaw|chebfun` across all `*.go` in reality returns ONE match: `topology/persistent/bottleneck.go:131` ("L^inf (Chebyshev) distance" — unrelated to polynomial approximation).
- `calculus/calculus.go` and `calculus/calculus_test.go` are the only files in `calculus/`. No Chebyshev, no Remez, no Lobatto/Gauss-Chebyshev quadrature, no Clenshaw recurrence, no barycentric interpolation at Chebyshev nodes.
- `signal/fft.go:49` `FFT(real,imag)` and `signal/fft.go:101` `IFFT` exist and are exercised by `signal/testdata/signal/fft.json` — substrate for fast DCT is already paid for.
- `optim/interpolate.go` exists but provides Lagrange / spline / Newton-form interpolation, not Chebyshev. (Confirmed via filename only — no Chebyshev string match in directory.)
- No Parks-McClellan / Remez in `signal/filter.go` (slot 316 confirms gap).
- No Clenshaw–Curtis quadrature in `calculus/` (would be the natural consumer of Chebyshev coefficients).

### Theory landmarks (the why)
- **Truncated Chebyshev series**: c_k = (2/π)∫_{-1}^{1} f(x) T_k(x) / √(1-x²) dx. Spectral convergence (geometric in k) for analytic f on [-1,1]; algebraic (k^{-s}) for C^s. Trefethen, *Approximation Theory and Approximation Practice* (2013, 2nd ed. 2019), Ch. 3, 7, 8.
- **Discrete Chebyshev coefficients via DCT-I**: sampling f at Chebyshev–Lobatto extrema x_j = cos(jπ/n), j=0..n, gives c_k via Type-I DCT in O(n log n). This is the workhorse — chebfun's `chebfun` constructor is essentially this loop with adaptive n-doubling until coefficient tail is below tolerance (Trefethen 2013, §6, §8).
- **Chebyshev interpolation ≡ near-best polynomial**: interpolant at Lobatto points has L^∞ error within factor (2/π)log(n)+1 of the minimax optimum (Lebesgue constant for Chebyshev nodes; Powell 1981 *Approximation Theory and Methods*, Ch. 7).
- **Clenshaw recurrence (1955)**: numerically stable evaluation of Σ c_k T_k(x). Backward recurrence b_k = 2x·b_{k+1} - b_{k+2} + c_k; result = (b_0 - b_2)/2 + c_0/2. Higham 2002 *Accuracy and Stability of Numerical Algorithms* §5.4 — proves backward stability when |x| ≤ 1.
- **Remez exchange (Evgeny Remez 1934)**: builds true minimax polynomial of degree n via the Chebyshev Alternation Theorem — a polynomial p* is minimax iff f - p* attains its max absolute error at ≥ n+2 points with alternating signs. Loop: solve linear system for current alternation set → find new extrema of error → swap → converge quadratically. Powell 1981 Ch. 8; Pachón & Trefethen 2009 "Barycentric–Remez algorithms for best polynomial approximation in the chebfun system" (BIT 49:721–741).
- **Carathéodory–Fejér (CF)**: rational approximation via SVD of Hankel matrix of Chebyshev coefficients; near-minimax for analytic f, doesn't require Remez iteration. Trefethen & Gutknecht 1983 SIAM J. Numer. Anal. 20:420; chebfun `cf` command.
- **chebfun (Battles & Trefethen 2004; Trefethen 2013)**: domain-decomposition + adaptive Chebyshev expansion = numerical functions as first-class objects. Frontier — out of scope for v1.
- **Boyd 2014 *Chebyshev and Fourier Spectral Methods*** (2nd ed., Dover): definitive practitioner reference; Ch. 2 algorithms, Ch. 8 Lobatto grids.
- **Mason & Handscomb 2002 *Chebyshev Polynomials*** (Chapman & Hall): identity catalog (T_n recurrence, derivative/integral closed forms), needed when implementing operations on coefficient vectors.

### Connection to other slots
- **Slot 316 (FIR Parks-McClellan)**: PM = Remez exchange in frequency domain. Sharing a Remez core between approximation and FIR design is the architectural win — McClellan, Parks & Rabiner 1973 IEEE T-AU 21:506 derived PM directly from Remez. Slot 346 should land first; slot 316 wraps it.
- **Slot 298/299/300 (special functions)**: Bessel J_n, hypergeometric, erfc are evaluated in production libraries (Boost, Cephes) via Chebyshev expansions on intervals — minimax error per coefficient byte. Once `Chebcoef` + `Clenshaw` exist, special-function tables become trivial.
- **Slot 245 (spectral methods PDE)**: Chebyshev–Galerkin / Chebyshev collocation requires the same DCT routines. Direct dependency.
- **Calculus / Clenshaw–Curtis quadrature**: ∫_{-1}^{1} f(x) dx ≈ Σ w_k f(x_k) with w_k from Chebyshev coefficients of 1, x, x², ... DCT-based; near-Gaussian accuracy at half the implementation cost. Direct application of T0 below.

## Concrete recommendations

### Tiered primitive list

| Tier | Primitive | LOC | Notes |
|------|-----------|-----|-------|
| T0 | `chebcoef.Coefficients(f, n) []float64` via DCT-I on Lobatto extrema | ~120 | Reuses `signal.FFT` via real-symmetric extension (length 2n FFT); fall back to O(n²) direct cosine sum if signal pkg not desired as dep — but it already is reality-internal. |
| T1 | `chebcoef.Clenshaw(c, x) float64` | ~50 | Numerically stable; the keystone evaluator. Backward recurrence per Higham 2002 §5.4. |
| T2 | `chebcoef.InterpolateLobatto(fvals, x)` barycentric form-2 | ~80 | Direct evaluation without coefficient transform; weights w_j = (-1)^j with halved endpoints (Berrut & Trefethen 2004 SIAM Rev 46:501). O(n) per point. |
| T3 | `chebcoef.Remez(f, n, tol) []float64` | ~300 | True minimax polynomial; alternation set via Newton on derivative of error; second exchange via single-swap rule (Pachón–Trefethen 2009). KEYSTONE — unblocks Parks-McClellan. |
| T4 | `chebcoef.CF(f, m, n)` Carathéodory–Fejér rational | ~200 | Frontier; SVD of Hankel of Chebyshev coefficients. Skip until slot 298–300 demand it. |
| T5 | chebfun-style `Chebfun` adaptive type with domain split | frontier | Out of v1 scope. Track as research item. |

### Day-1 PR (cheapest)
**T0 + T1 + T2, ~250 LOC** in new package `chebyshev/` (or extend `calculus/`):
1. `Coefficients(f func(float64) float64, n int) []float64` — sample at Lobatto extrema, run DCT-I via existing `signal.FFT` of length 2(n-1) symmetric extension; halve endpoint coefficients.
2. `Clenshaw(c []float64, x float64) float64` — backward recurrence; documented for x ∈ [-1,1], extrapolation warning outside.
3. `InterpolateLobatto(fvals []float64, x float64) float64` — barycentric form-2 for users who already have samples and don't want a coefficient transform.
4. Helper `LobattoNodes(n int, out []float64)` — x_j = cos(jπ/(n-1)), allocation-free per CLAUDE.md hot-path rule.
5. `MapToInterval(a, b float64, x float64) float64` and inverse — change of variable for f on arbitrary [a,b].

This is a clean, finite PR. No new external dependency. Composes existing FFT (`signal/fft.go:49`).

### Day-2/3 PR (the keystone)
**T3 Remez** in `chebyshev/remez.go`, ~300 LOC. Implementation outline:
- Initialize alternation set to Chebyshev nodes (degree n+2 nodes).
- Solve linear system [T_0(x_i), T_1(x_i), ..., T_n(x_i), (-1)^i] · [a_0, ..., a_n, E]^T = f(x_i) — n+2 equations, n+2 unknowns. Use existing `linalg.Solve` (LU is fine; well-conditioned at Chebyshev nodes).
- Find new error extrema by sampling on a fine grid (3n points) + local Newton on E'(x) = 0.
- Single-point exchange (swap one node at a time, Pachón–Trefethen rule) for stability.
- Converge when (E_max - E_min)/|E| < tol.
- Test against truncated Chebyshev series — Remez should beat it by factor < 2 on Lebesgue constant.

Once T3 lands, **slot 316 Parks-McClellan is a 100-line wrapper**: same Remez core, weighting function for desired magnitude response.

### R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities

For analytic f (e.g., f(x) = e^x, sin(x), 1/(1+25x²) — the Runge function — at sufficient n to drive truncation below tolerance):

1. **Path A (DCT coefficients)** vs **Path B (Lobatto barycentric interpolant)** vs **Path C (Clenshaw on coefficients)**: all three evaluators must agree to ~1e-12 on a fine grid. This pins T0/T1/T2 mutually.

2. **Remez n+2 alternation theorem**: for any test f and degree n, locate the extrema of f(x) - p*(x); verify count = n+2 AND signs strictly alternate. This is a structural invariant from approximation theory — it falsifies Remez bugs that pass a numerical residual check. Pin Remez against (a) Chebyshev interpolant minimax bound (Lebesgue constant ≤ 1.45 for n=10 ≤ Remez error ratio), (b) closed-form minimax for x^{n+1} on [-1,1] which is exactly 2^{-n} T_{n+1}(x) (folklore — Mason & Handscomb 2002 §3.4).

3. **Clenshaw ≡ Horner regression**: for c built from monomial coefficients via T_k → x^k expansion, `Clenshaw(c, x)` must equal direct polynomial evaluation to machine precision. Pin against Go stdlib `math.Pow`-based polynomial evaluation. (Beware Horner instability for large n; comparison should use Kahan summation for ground truth.)

### Cross-link consumer slots
- **Slot 298 (hypergeometric)** / **slot 299 (asymptotic analysis)** / **slot 300 (Bessel spherical)**: Chebyshev coefficient tables are the standard implementation strategy. Slot 346 lands first.
- **Slot 316 (FIR Parks-McClellan)**: direct dependency on T3 Remez.
- **Slot 245 (spectral PDE)** / **slot 248 (multigrid)**: Chebyshev collocation/Galerkin. Direct dependency on T0/T1.
- **calculus/** (no current slot): Clenshaw–Curtis quadrature ≈ 30 LOC after T0 lands. Recommend opening a follow-up.
- **slot 338 Hilbert transform** & **slot 270 graph signal proc**: lower coupling, but both benefit from spectral coefficient tooling.

### Naming / packaging
- Create `chebyshev/` as top-level package (peer to `signal/`, `calculus/`). Justification: 22 packages already; Chebyshev approximation has cross-cutting consumers (signal filter design, calculus quadrature, special functions). Burying it in `calculus/` would force `signal` to import `calculus`, which it doesn't today and shouldn't.
- Alternative: expose under `calculus/cheb` subpackage. Marginal preference for top-level.
- API style mirrors `signal/`: stateless functions, output buffers for hot paths (CLAUDE.md rule 3), per-function tolerance documented (rule 5), provenance citations (rule 4: cite Trefethen 2013 §6 / Higham 2002 §5.4 / Powell 1981 §8 / Pachón–Trefethen 2009).
- Golden files: 30 vectors per function. Mandatory IEEE 754 cases: x = ±1 endpoints (Lobatto boundary), x = NaN (must propagate), x = ±Inf (must yield NaN, document), x = -0.0 (T_k(-0) = T_k(0)), |x|=1+ε extrapolation (warning behavior).

### Risks / non-goals
- **Don't ship monomial-coefficient Remez**. Conditioning is hopeless for n ≥ 12. Always work in Chebyshev basis.
- **Don't try to match chebfun's adaptive `chebfun` constructor in v1.** It requires careful coefficient-tail logic, plateau detection, and rounding-noise floor estimation (Aurentz & Trefethen 2017 ACM TOMS 43:33). Defer to T5.
- **Lobatto vs root-grid choice**: ship both helpers, default to Lobatto (extrema include endpoints — necessary for consistent-with-chebfun semantics and for Clenshaw–Curtis quadrature with explicit endpoint values).
- **Floating-point cos cost**: `cos(jπ/n)` should be precomputed once per n; expose `LobattoNodes(n, out)` for caller reuse.

## Sources

### Reality repo
- `C:\limitless\foundation\reality\calculus\calculus.go` — only file in calculus/, no Chebyshev content.
- `C:\limitless\foundation\reality\signal\fft.go:49` — `FFT(real,imag)` substrate for fast DCT.
- `C:\limitless\foundation\reality\signal\fft.go:101` — `IFFT` for inverse transform.
- `C:\limitless\foundation\reality\optim\interpolate.go` — Lagrange/spline only; no Chebyshev nodes.
- `C:\limitless\foundation\reality\topology\persistent\bottleneck.go:131` — incidental "Chebyshev distance" L^∞ comment, unrelated to polynomial approximation.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\316-dive-fir-design.md` — Parks-McClellan slot, downstream consumer.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\298-new-hypergeometric.md` / `299-new-special-functions.md` / `300-new-bessel-spherical.md` — special-function consumers.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\245-new-spectral-methods.md` / `248-new-multigrid.md` — spectral-PDE consumers.
- `C:\limitless\foundation\reality\CLAUDE.md` — design rules (zero deps, per-function tolerance, golden files).

### Web / textbook
- Trefethen, L.N. *Approximation Theory and Approximation Practice* (Extended Edition), SIAM 2019. Definitive treatment; chebfun-aligned. Ch. 3, 6, 7, 8 cover all of T0–T2.
- Boyd, J.P. *Chebyshev and Fourier Spectral Methods*, 2nd ed., Dover 2001 (free PDF on author site). Ch. 2 algorithms, Ch. 8 Lobatto.
- Mason, J.C. & Handscomb, D.C. *Chebyshev Polynomials*, Chapman & Hall 2002. Identity catalog.
- Powell, M.J.D. *Approximation Theory and Methods*, Cambridge UP 1981. Ch. 7 (Chebyshev interpolation), Ch. 8 (Remez).
- Higham, N.J. *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002. §5.4 Clenshaw stability.
- Remez, E.Y. "Sur la détermination des polynômes d'approximation de degré donnée," Comm. Soc. Math. Kharkov 10 (1934). Original alternation algorithm.
- Pachón, R. & Trefethen, L.N. "Barycentric–Remez algorithms for best polynomial approximation in the chebfun system," BIT Numer. Math. 49 (2009) 721–741. Modern numerically-stable Remez.
- Trefethen, L.N. & Gutknecht, M.H. "The Carathéodory–Fejér method for real rational approximation," SIAM J. Numer. Anal. 20 (1983) 420–436.
- McClellan, J.H., Parks, T.W. & Rabiner, L.R. "A computer program for designing optimum FIR linear phase digital filters," IEEE Trans. Audio Electroacoust. 21 (1973) 506. Remez → PM bridge.
- Berrut, J.-P. & Trefethen, L.N. "Barycentric Lagrange interpolation," SIAM Rev. 46 (2004) 501–517. Form-2 weights.
- Aurentz, J.L. & Trefethen, L.N. "Chebfun and numerical quadrature," ACM TOMS 43 (2017) Art. 33. Adaptive coefficient-tail logic for T5.
- Battles, Z. & Trefethen, L.N. "An extension of MATLAB to continuous functions and operators," SIAM J. Sci. Comput. 25 (2004) 1743. Original chebfun paper.
