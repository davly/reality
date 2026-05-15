# 298 — new-hypergeometric (₂F₁ / ₁F₁ / Generalized ₚFq / Meijer G / Bessel / Orthogonal Polys / Mellin-Barnes)

## Headline
Zero hypergeometric / Bessel / orthogonal-polynomial surface in reality despite `prob/mathutil.go` already shipping the Γ / log Γ / regularized-incomplete-Γ / regularized-incomplete-B substrate; **~400 LOC day-1 PR (generic ₚFq series + ₁F₁ Kummer + ₀F₁ + closed-form ₂F₁ for orthogonal polynomials) unblocks Bessel J/Y/I/K, Legendre/Hermite/Laguerre/Chebyshev/Gegenbauer/Jacobi, NCBeta, Marcum-Q, Skellam (T1.1 of 117), Rice/Wishart, hydrogen orbitals, spherical harmonics, and the Mellin-Barnes evaluator (slot 297 T6). Saturates ≥4 R-MUTUAL 3/3 pins.

## Findings

- **Total surface absent.** `grep -E "func.*(Bessel|Legendre|Hermite|Laguerre|Chebyshev|Gegenbauer|Jacobi|Hypergeometric|Kummer|Hyper2F1|MeijerG)"` across `*.go` returns exactly two hits: `calculus/calculus.go:149 GaussLegendre` (the **quadrature rule**, not the polynomial) and the test using it. Nothing else. No special-functions package, no `prob/special.go`, no `signal/bessel.go`. `prob/copula/doc.go` mentions Bessel only in a docstring.
- **prob/mathutil.go already ships the substrate.** `prob/mathutil.go:23 LogGamma` (wraps `math.Lgamma`), `:36 Erfc`, `:59 RegularizedBetaInc` (continued-fraction Lentz), `:200 regularizedGammaLowerSeries`. Pochhammer symbol `(a)_n = Γ(a+n)/Γ(a)` is one ratio of these. **No `Pochhammer`, no `RisingFactorial` primitive** anywhere. Trivial 8-LOC add unblocks all hypergeometric series.
- **prob/distributions.go is short on hypergeometric-dependent distributions.** Only Normal, Exponential, Uniform, Beta, Poisson, Gamma, Binomial. Slot 117 enumerated the missing roster: **Hypergeometric (discrete)** — closed form, no special func; **Skellam** — needs `I_|k|(2√(μ₁μ₂))`; **Beta-binomial / NCBeta / NC-χ² / NC-F** — need ₁F₁ or ₂F₁; **Rice** — needs `I_0`; **Wishart** — needs Bessel-K (matrix-Γ Mellin-Barnes); **Marcum-Q** — needs ₁F₁. **Five distributions in T1.1/T1.2 of slot 117 directly block on this slot.**
- **constants/ has no hypergeometric values.** `constants/math.go` ships π, e, √2, Catalan, Apéry but no `Bessel0Zero1=2.404825…`, no `LegendreP_n_at_1=1`, no `Chebyshev1_at_minus1=(-1)^n`. Reference values for golden tests will need to be regenerated from `mpmath`-validated truth tables (BSD-3, OK to consult as oracle since we re-derive).
- **Slot 297 T6 explicitly defers to this slot.** Slot 297 line 43: "**Day-2** — required for Meijer-G and ₂F₁ outside Taylor radius." Mellin-Barnes is the **only** way to evaluate ₚFq with ill-conditioned series (e.g. ₂F₁ near z=1 with large c, or ₁F₁ with large `|z|·a/b`). This slot is the consumer-of-record for slot 297 T6.
- **Slot 295 cross-link.** Lerch transcendent Φ(z, s, a) = Σ z^k/(k+a)^s is a generalized hypergeometric (specifically a ₚFq with infinitely many parameter shifts via Mellin-Barnes); polylogarithm Li_s(z) = z · ₚ₊₁Fₚ(1,1,…,1; 2,…,2; z) is closed form for integer s. Hurwitz ζ(s, a) = Φ(1, s, a). **Slot 295 T-tier consumers for Lerch / polylog can compose this slot's T0 generic ₚFq evaluator** instead of writing ζ-specific code.
- **Bessel functions are confluent hypergeometric, period.** `J_ν(z) = (z/2)^ν / Γ(ν+1) · ₀F₁(; ν+1; -z²/4)`; `I_ν(z) = (z/2)^ν / Γ(ν+1) · ₀F₁(; ν+1; +z²/4)`; `Y_ν` and `K_ν` are connection coefficients across J/I to the second-kind solutions. **Single ₀F₁ implementation gives all four families for moderate |z|** (asymptotic for large |z| is separate — slot 297 T1 saddle-point covers it; this slot does the small-|z| series leg).
- **Orthogonal polynomials are all closed-form ₂F₁ or ₁F₁.** Recipe (DLMF 18.5):
  - Chebyshev 1st kind `T_n(x) = ₂F₁(-n, n; 1/2; (1-x)/2)`
  - Chebyshev 2nd kind `U_n(x) = (n+1) · ₂F₁(-n, n+2; 3/2; (1-x)/2)`
  - Legendre `P_n(x) = ₂F₁(-n, n+1; 1; (1-x)/2)`
  - Gegenbauer `C_n^α(x) = (Γ(n+2α)/(n!·Γ(2α))) · ₂F₁(-n, n+2α; α+1/2; (1-x)/2)`
  - Hermite (physicist) `H_n(x) = (2x)^n · ₂F₀(-n/2, -(n-1)/2; ; -1/x²)` (or via ₁F₁)
  - Laguerre `L_n^α(x) = ((n+α)!/(n!·α!)) · ₁F₁(-n; α+1; x)`
  - Jacobi `P_n^{α,β}(x) = ((α+1)_n / n!) · ₂F₁(-n, n+α+β+1; α+1; (1-x)/2)`
  
  **Once T1 (₂F₁) and T2 (₁F₁) ship, all seven orthogonal-polynomial families are 5–10 LOC each.** Compare to a 3-term-recurrence implementation: identical convergence numerics but recurrence accumulates roundoff for n>50 in some regions; ₂F₁ form is a finite series for `-n` parameter (terminates after n+1 terms by Pochhammer (-n)_k=0 at k=n+1) so it is **exact for any n in O(n)**.
- **Pfaff and Euler transformations** map ₂F₁(a,b;c;z) ↔ ₂F₁(a,c-b;c;z/(z-1)) ↔ ₂F₁(c-a,c-b;c;z) · (1-z)^{c-a-b}. Used to push convergence radius (z near 1 → transform to z' near 0). Kummer's 24 solutions = full connection group across z=0, z=1, z=∞ singularities. ~80 LOC for the practical 6-solution subset; full 24 only needed for analytic continuation across both critical points.
- **Pearson-Olver-Porter (2017) "Numerical methods for the computation of the confluent and Gauss hypergeometric functions"**, Numer. Algorithms 74:821-866, is the modern algorithmic reference. They evaluate 30+ algorithms across (a, b, c, z) parameter space and tabulate which is best where. **The single-region answer "always use the Taylor series" is wrong** — needs piecewise dispatch. Their Algorithm 1 (Maclaurin) handles `|z|<0.9` with terms ratio test; Algorithm 6 (Buhring analytic continuation) handles `0.9<|z|<2`; Algorithm 4 (single-fraction Stoer-Bulirsch) handles `|z|>2` via z=1/z transform.
- **Forrey (1997) "Computing the hypergeometric function"**, J. Comput. Phys. 137:79-100, is the classic single-precision reference; algorithm essentially Maclaurin with ratio termination, plus Pfaff fallback for slow convergence. ~120 LOC port.
- **Michel-Stoitsov (2008) "Fast computation of the Gauss hypergeometric function with all its parameters complex"**, Comput. Phys. Commun. 178:535-551, is the multi-precision modern reference; same dispatch as Pearson-Olver-Porter but in complex(a, b, c, z) space. **Beyond day-1 scope.**
- **q-hypergeometric / Heine ₂φ₁(a,b;c;q,z) = Σ ((a;q)_n (b;q)_n / ((c;q)_n (q;q)_n)) z^n.** q-shifted factorial `(a;q)_n = Π_{k=0}^{n-1}(1 - a·q^k)`. Heine 1846 transformations are q-deformations of Gauss/Pfaff/Euler. Practical use: q-orthogonal polynomials (Askey-Wilson scheme), affine combinatorics, partition theory (Slater 1966). **Frontier; defer past day-3** unless a consumer surfaces.
- **Appell F1, F2, F3, F4** are 2-variable hypergeometric — generalize ₂F₁ to two arguments with one or two contraction relations. F1 appears in incomplete elliptic integrals and Schwinger-parameter Feynman amplitudes. **Frontier.**
- **Lauricella F_A, F_B, F_C, F_D** — n-variable hypergeometric. Used in conformal-blocks, multi-loop amplitudes, multivariate special functions. **Frontier; mostly research-grade; defer.**
- **Meijer G-function and Fox H-function.** G subsumes virtually all classical special functions including ₚFq for any (p,q). G has Mellin-Barnes integral representation `G^{m,n}_{p,q}(z | a;b) = (1/2πi) ∫_L (Π Γ(b_j-s) Π Γ(1-a_j+s)) / (Π Γ(1-b_j+s) Π Γ(a_j-s)) · z^s ds`. Closing the contour gives a residue sum that for generic parameters reduces to ≤p hypergeometric ₚFq's — so **MeijerG = sum of ₚFq's** computed via T1+T2+T7. ~200 LOC for the generic case; subtle pole-merger logic. Fox H is yet more general (allows Γ-arguments with rational coefficients other than ±1) and is research-grade.
- **Pure-Go MIT zero-dep hypergeometric library = ABSENT.** Surveyed: `mpmath` BSD-3 (Python, slow, multiprecision), Boost.Math BSL-1.0 (C++, header-only, complete ₂F₁/₁F₁/Bessel/Legendre/Hermite/Laguerre), GSL GPL-3 (complete), ARB / FLINT LGPL (multi-precision), Mathematica/Maple/MATLAB proprietary, SageMath GPL-3, gonum BSD-3 has **no hypergeometric functions** (only Bessel J0/J1/Y0/Y1/I0/I1/K0/K1 from Cephes, plus erf-family). **Reality could be the only zero-dep MIT pure-Go ₂F₁/₁F₁ in the ecosystem.** Moat: high.
- **Single cheapest day-1 PR.** `Pochhammer` (8 LOC) + T0 generic ₚFq series (~80 LOC) + T2 ₁F₁ Kummer with Tricomi backup (~80 LOC) + T3 ₀F₁ (~30 LOC) + T4 Bessel J/I from T3 (~40 LOC) + T6 seven orthogonal polynomials (~80 LOC closed-form via T1 stub or via 3-term recurrence pre-T1). T1 full ₂F₁ (~120 LOC including Pfaff/Euler). **Total ~440 LOC, ships every classical special function listed in undergraduate physics curriculum.**

## Concrete recommendations

1. **T0 — `special/pochhammer.go` + `special/pfq.go` generic ₚFq series.** `Pochhammer(a float64, n int) float64` returns `Γ(a+n)/Γ(a)` with sign-stable path through `LogGamma`. `PFQ(numerator, denominator []float64, z float64) float64` evaluates `Σ_{k=0}^∞ (Π (a_i)_k / Π (b_j)_k) · z^k / k!` by accumulating term-ratio `t_{k+1}/t_k = z · Π(a_i+k)/Π(b_j+k)/(k+1)` and terminating when `|t_k| < ε · |sum|`. Auto-handles negative-integer numerator (terminating series) and detects c, b_j poles. ~90 LOC. **Day-1.** Pin: `PFQ([], [], z) ≡ math.Exp(z)` to 1e-13 for |z|<10. R-MUTUAL 3/3 vs `math.Exp` ≡ Taylor truncation.

2. **T1 — `special/hyper2f1.go` Gauss hypergeometric ₂F₁.** `Hyper2F1(a, b, c, z float64) float64`. Dispatch: `|z|<0.5` → direct series (T0); `0.5≤|z|<0.9` → Pfaff `(1-z)^{-a} ₂F₁(a, c-b; c; z/(z-1))`; `0.9≤|z|<1` → series after Euler transform `(1-z)^{c-a-b} ₂F₁(c-a, c-b; c; z)`; `1<|z|<2` → reflection `z → 1/(1-z)`; `|z|≥2` → `z → 1/z` Watson connection. ~150 LOC. **Day-1.** Pin: `₂F₁(½,½;1;k²) = (2/π)·K(k)` complete elliptic integral of 1st kind, ≡ AGM-Gauss recursion ≡ this implementation, **3-way for k=0.3, 0.7, 0.95**. Pin 2: `₂F₁(1,1;2;-z) = log(1+z)/z` ≡ `math.Log(1+z)/z`. Pin 3: `₂F₁(½,1;3/2;z²) = atanh(z)/z` ≡ `math.Atanh(z)/z`.

3. **T2 — `special/hyper1f1.go` Kummer confluent ₁F₁ + Tricomi U(a;b;z).** `Hyper1F1(a, b, z float64) float64` dispatch: `|z|<50` → T0 series; `|z|≥50, z>0` → Kummer transformation `e^z · ₁F₁(b-a; b; -z)` if that converges faster; `|z|≥50, z<0` → asymptotic `M(a;b;z) ~ Γ(b)/Γ(b-a) · (-z)^{-a} · ₂F₀(a, a-b+1; ; -1/z)` (Slater 13.5.1). `TricomiU(a, b, z float64) float64` = second linearly-independent solution; needed for non-integer `b` connection. ~120 LOC. **Day-1.** Pin: `₁F₁(1; 2; z) = (e^z-1)/z` ≡ `math.Expm1(z)/z`. Pin 2: `₁F₁(a; a; z) = e^z` ≡ `math.Exp(z)`. Pin 3: NCBeta-PDF identity vs scipy reference table at 5 grid points.

4. **T3 — `special/hyper0f1.go`.** `Hyper0F1(b, z float64) float64` = `Σ z^k / ((b)_k k!)`. ~25 LOC via T0. **Day-1.** Pin: `₀F₁(; ½; -z²/4) = cos(z)` ≡ `math.Cos(z)`; `₀F₁(; 3/2; -z²/4) = sin(z)/z` ≡ `math.Sin(z)/z`. Both R-MUTUAL 3/3.

5. **T4 — `special/bessel.go` Bessel J / Y / I / K of integer + half-integer + real-ν order.** Compose T3:
   - `BesselJ(nu, z) = (z/2)^ν / Γ(ν+1) · ₀F₁(; ν+1; -z²/4)` for `|z| ≤ 8`
   - `BesselI(nu, z) = (z/2)^ν / Γ(ν+1) · ₀F₁(; ν+1; +z²/4)` for `|z| ≤ 8`
   - `BesselY` via `(J_ν cos(νπ) - J_{-ν})/sin(νπ)` for non-integer; integer-ν limit needs L'Hopital + slot 297 T4 Stirling; defer integer-ν Y until 297 ships
   - `BesselK` via `(π/2)·(I_{-ν} - I_ν)/sin(νπ)` analogous
   
   Asymptotic forms `J_ν(z) ~ √(2/πz) cos(z - νπ/2 - π/4)` for `|z|>8` deferred to slot 297 T1 saddle-point composition. ~80 LOC. **Day-1 (J, I; Y, K limited to non-integer).** Pin: `J_0(2.404825…)=0` (first positive zero) ≡ this implementation ≡ Cephes table value. Pin 2: `I_½(z) = √(2/πz) sinh(z)` closed form. Pin 3: `J_½(z) = √(2/πz) sin(z)`. **R-MUTUAL 3/3 saturated** for half-integer ν.

6. **T5 — `special/legendre.go` Legendre P_n + associated P_n^m + Legendre Q_n.** `LegendreP(n int, x float64) float64` — for integer n, use 3-term Bonnet recurrence `(n+1) P_{n+1} = (2n+1) x P_n − n P_{n-1}` (the standard cheap path); cross-check via T1 `₂F₁(-n, n+1; 1; (1-x)/2)`. `LegendrePAssoc(n, m int, x float64)` standard ladder. `LegendreQ(n int, x float64)` for `|x|<1`. ~70 LOC. **Day-1.** Pin: `P_n(1)=1`, `P_n(-1)=(-1)^n`, `P_2(x) = (3x²-1)/2`, `P_3(x) = (5x³-3x)/2`. **R-MUTUAL 3/3:** Bonnet recurrence ≡ `₂F₁(-n, n+1; 1; (1-x)/2)` ≡ Rodrigues `(1/(2^n n!)) · d^n/dx^n[(x²-1)^n]` (numerical via slot 016 calculus FD, but FD-of-high-order is unstable; better to compare just the first two).

7. **T6 — `special/orthopoly.go` Hermite, Laguerre, Chebyshev T/U, Gegenbauer, Jacobi.** All via 3-term recurrence (cheap, exact, stable for the physicist orthogonality region) AND closed-form ₂F₁/₁F₁ (validation):
   - `HermiteH(n int, x float64)` physicist; `HermiteHe(n int, x float64)` probabilist
   - `LaguerreL(n int, alpha, x float64)` generalized
   - `ChebyshevT(n int, x float64)`, `ChebyshevU(n int, x float64)`
   - `GegenbauerC(n int, alpha, x float64)`
   - `JacobiP(n int, alpha, beta, x float64)`
   
   ~120 LOC total. **Day-1.** Pin per family: e.g. `H_2(x) = 4x²-2`, `L_1^0(x) = 1-x`, `T_n(cos θ) = cos(nθ)` (deeply exact: T_n(cos(0.7))=cos(0.7n) to 1e-15 for n=20). **Day-1 PR can ship recurrence-only and add ₂F₁ validation in day-2.**

8. **T7 — `special/pfq_general.go` generalized ₚFq for arbitrary (p, q).** Already implemented as T0; this entry adds the **convergence regularization** for (a) `p = q+1` (radius 1, needs Pfaff-style transformation past z=1 — non-trivial for p>2); (b) `p > q+1` (formally divergent — Borel summation via slot 297 T5; or treat as asymptotic). ~80 LOC. **Day-2.** Composes slot 297 T5 BorelSum.

9. **T8 — `special/meijer_g.go` Meijer G via Mellin-Barnes residue sum.** `MeijerG(m, n, p, q int, a, b []float64, z float64) float64` evaluates the canonical Mellin-Barnes contour integral by closing in left half-plane and summing residues at the `Γ(b_j - s)` poles `s = b_j + k, k=0,1,2,…`. Generic-parameter case reduces to `Σ_{j=1}^m C_j · z^{b_j} · ₚFq(...; z·(-1)^{p-m-n})` (Slater 1966 4.1.2). ~250 LOC including pole-merger handling for confluent parameters. **Day-3.** Composes slot 297 T6 (Mellin-Barnes contour) + this slot's T7 (ₚFq). Pin: `G^{1,0}_{0,1}(z|;0) = e^{-z}`; `G^{1,2}_{2,2}(z|1,1;1,0) = log(1+z)`. **R-MUTUAL 3/3 vs T0/T2.**

10. **T9 — `special/qhyper.go` q-hypergeometric ₂φ₁ Heine.** `QPochhammer(a, q float64, n int) float64` = `Π_{k=0}^{n-1}(1 - a q^k)`. `QHyper2Phi1(a, b, c, q, z float64) float64`. Heine transformations as q-Pfaff/q-Euler. ~80 LOC. **Frontier; defer past day-3** unless slot 280 (stochastic block models) or partition-theoretic consumer surfaces.

11. **T10 — `special/fox_h.go` Fox H-function.** Generalization of Meijer G allowing rational Γ-argument coefficients other than ±1. ~400 LOC. **Frontier; research-grade; defer.**

12. **T11 — `special/appell.go` Appell F1, F2, F3, F4 (2-variable).** Two-variable hypergeometric series with one/two contraction relations. Used in elliptic-integral generalizations and Feynman amplitudes. ~200 LOC. **Frontier; defer.**

13. **T12 — `special/lauricella.go` Lauricella F_A, F_B, F_C, F_D (n-variable).** Multivariable. ~300 LOC. **Frontier; research-grade; defer.**

14. **T13 — `special/contiguous.go` ₂F₁ contiguous-relation manipulator.** Gauss's 15 contiguous relations let you transform `₂F₁(a±1, b; c; z)` etc. into `₂F₁(a, b; c; z)` via 3-term recurrence in (a, b, c). Used to reach unstable parameter regions by recurring from stable seeds. ~100 LOC. **Day-2** (helps stabilize T1 in corners). DLMF 15.5.

15. **T14 — `special/pochhammer_rising.go` rising/falling factorials as standalone.** Already shipped in T0 as helper, but worth exposing publicly as `RisingFactorial(a, n)`, `FallingFactorial(a, n)`. Used by `combinatorics/` (binomial coefficient generalization), `prob/` (Stirling number 2nd kind via `S(n,k) = (1/k!) Σ (-1)^j C(k,j) (k-j)^n` involves no Pochhammer but the negative-Pochhammer identity ties them). 8 LOC. **Day-1.**

## Cross-cutting

- **Slot 117 (prob-missing) Hypergeometric/Skellam/Beta-binomial/NCBeta/Marcum-Q/Rice/Wishart** ← directly consumes T2 (₁F₁), T4 (Bessel I/K), T5 (associated Legendre for Wishart Mellin transform). Five missing distributions in slot 117 T1.1/T1.2 unblock here.
- **Slot 295 (L-functions, Lerch transcendent, polylogarithm)** ← Li_s(z) = z · ₚ₊₁Fₚ(1,…,1; 2,…,2; z) is direct T0 dispatch for integer s; Hurwitz ζ(s, a) needs Lerch which needs T0 + slot 297 T6 (Mellin-Barnes for non-integer s).
- **Slot 297 T6 (Mellin-Barnes integration)** ← reciprocal: this slot T8 (Meijer G) is the major consumer of slot 297 T6's contour-integration machinery; together they form the closed-form-special-function evaluation pipeline. Mutual day-2 unlock.
- **Slot 244 (PDE solvers — Schroeder rate decomposition mention)** ← if the reference Schroeder decomposition involves ₂F₁ (some semi-Markov rate-equation solutions do), T1 unblocks.
- **Slot 281 (temporal graphs — OU-process transition density)** ← OU-process kernel involves Bessel-K (Wishart-like covariance for matrix-valued OU). T4 unblocks.
- **Slot 217 / 220 (random-matrix theory / free probability)** ← Wishart Mellin transform involves matrix-Γ which is product of Γ's — but moments of Wishart eigenvalues involve ₂F₁; spectral density of Marchenko-Pastur is closed-form algebraic but tail expansions involve ₁F₁.
- **Slot 296 (generating functions, Flajolet-Odlyzko)** ← Plancherel-Rotach-style asymptotics for orthogonal polynomials need T6 + slot 297 T10. Cross-validates against this slot's recurrence-form T6.
- **aicore physics consumers** — hydrogen-atom radial wavefunctions are `R_{n,ℓ}(r) ∝ e^{-r/(na)} (2r/(na))^ℓ · L_{n-ℓ-1}^{2ℓ+1}(2r/(na))` (associated Laguerre). Spherical harmonics `Y_ℓ^m(θ, φ)` are associated Legendre × `e^{imφ}`. **Both are 2-line invocations of T6+T5 once shipped.**
- **calculus/calculus.go GaussLegendre quadrature** (line 149) currently uses hard-coded nodes/weights for n ≤ 5; could be rewritten to call T5 LegendreP zeros + Gauss formula for arbitrary n. 30 LOC win.

## Sources

- `C:\limitless\foundation\reality\prob\mathutil.go` — existing Γ / log Γ / regularized-Γ / regularized-B substrate, lines 13-220.
- `C:\limitless\foundation\reality\prob\distributions.go` — current 7-distribution roster.
- `C:\limitless\foundation\reality\calculus\calculus.go:149` — GaussLegendre quadrature (the only `Legendre` hit, unrelated to polynomials).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\297-new-asymptotic-analysis.md` — slot 297 T6 Mellin-Barnes deferral, mutual unlock.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\117-prob-missing.md` — Skellam/Hypergeometric/Beta-binomial/Rice consumers.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\295-new-l-functions.md` — Lerch / polylog cross-link.
- DLMF chapters 13 (Kummer ₁F₁), 15 (Gauss ₂F₁), 16 (generalized ₚFq), 18 (orthogonal polynomials), https://dlmf.nist.gov/.
- Slater (1966) "Generalized Hypergeometric Functions", Cambridge — canonical comprehensive reference.
- Bailey (1935) "Generalized Hypergeometric Series", Cambridge tracts — foundational identities (Saalschütz, Whipple, Dixon, Watson).
- Andrews-Askey-Roy (1999) "Special Functions", Cambridge Encyclopedia of Math vol 71 — modern textbook.
- Pearson, Olver, Porter (2017) "Numerical methods for the computation of the confluent and Gauss hypergeometric functions", Numer. Algorithms 74:821-866 — algorithm dispatch by parameter region.
- Forrey (1997) "Computing the hypergeometric function", J. Comput. Phys. 137:79-100 — single-precision reference port.
- Michel-Stoitsov (2008) "Fast computation of the Gauss hypergeometric function with all its parameters complex", Comput. Phys. Commun. 178:535-551 — multi-precision modern.
- Olver (1974) "Asymptotics and Special Functions" — Bessel asymptotics anchor.
- mpmath (BSD-3, Python) — `hyp2f1`, `hyper`, `meijerg`, `besselj` reference implementation; oracle for golden vectors.
- Boost.Math (BSL-1.0, C++ header-only) — `hypergeometric_2F1`, `cyl_bessel_j`, `legendre_p`, `hermite`, `laguerre` reference.
- gonum (BSD-3, Go) — has J0/J1/Y0/Y1/I0/I1/K0/K1 only (Cephes ports); **no ₂F₁/₁F₁/general-ν Bessel/orthogonal-polys** — confirms the reality moat.
