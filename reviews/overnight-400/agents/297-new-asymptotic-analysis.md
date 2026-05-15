# 297 — new-asymptotic-analysis (Watson / Laplace / Saddle-Point / Borel / Mellin-Barnes / Padé / Hayman)

## Headline
Zero asymptotic-analysis surface — calculus/ has only quadrature/root-finding (no expansion machinery), prob/ has no Edgeworth/Cornish-Fisher despite already shipping Lanczos-based LogGamma; ~310 LOC day-1 PR (Stirling expansion + Padé approximant + Edgeworth + Laplace-method skeleton) saturates 4 R-MUTUAL 3/3 pins and is the substrate consumed by slots 220, 244, 264, 269, 281, 295, 296.

## Findings

- **Total surface absent.** Grep across reality for `Asymptotic|WatsonLemma|Laplace[A-Z]|SteepestDescent|SaddlePoint|BorelSum|Resum|MellinBarnes|Hyperasympt|Stokes[A-Z]|PadeApprox|OptimalTrunc|Edgeworth|CornishFisher` returns only review files. No data type, no operation. No `Pade`, no `Romberg`, no `Aitken`/`Shanks`/`Wynn`/`Levin` series acceleration anywhere in the tree (`Levinson` matches are unrelated — Yule-Walker recursion). `Cornish-Fisher` is referenced in one comment in `prob/copula/studentt.go:52` (Hill 1970 high-df expansion for an initial bracket) but **not implemented as a callable primitive**.
- **calculus/ is 274 LOC and asymptotic-free.** `func [A-Z]` enumeration in `calculus/calculus.go`: `NumericalDerivative`, `NumericalGradient`, `TrapezoidalRule`, `SimpsonsRule`, `GaussLegendre`, `MonteCarloIntegrate`. All approximate finite-domain integrals at finite λ — no large-parameter / endpoint / saddle structure. No `Romberg` either, despite Romberg being standard in any Bender-Orszag-style chapter.
- **prob/ has Lanczos LogGamma already** (`prob/mathutil.go:23` wraps `math.Lgamma`) — the substrate Stirling's expansion validates against. No `Skewness`/`Kurtosis`/cumulant primitives; no MGF/CGF; no characteristic-function inversion. `prob/copula/studentt.go:52` mentions Cornish-Fisher in a comment only.
- **combinatorics/ ships 10 closed-forms** (counting.go: Factorial:25, BinomialCoeff:51, Catalan:94, Fibonacci:110, StirlingFirst:159, StirlingSecond:194, BellNumber:234, IntegerPartitions:268, DerangementCount:296). Each has a textbook **asymptotic expansion** that is NOT computed: Catalan ~ 4ⁿ/(n^{3/2} √π), Fibonacci ~ φⁿ/√5, Bell via Hayman saddle-point on `exp(eˣ-1)`, integer partitions via Hardy-Ramanujan p(n) ~ exp(π√(2n/3))/(4n√3). Slot 296 (T10 Flajolet-Odlyzko) covers these from the **GF side**; slot 297 covers them from the **integral side** and the two should agree (cross-validation pin).
- **Slot 296 boundary.** Slot 296 owns combinatorial-GF asymptotics via Flajolet-Odlyzko transfer theorem (T10): given `f(z) ~ (1-z/r)^{-α}`, automatic `[zⁿ]f ~ A r^{-n} n^{α-1}/Γ(α)`. **This slot owns the integral-asymptotics machinery** that the transfer theorem composes with: Watson's lemma, Laplace's method, saddle-point method, Stirling's expansion, Padé resummation. Hayman's method (combinatorial saddle-point) and Darboux's method (singularity-derivative) sit on the boundary — best implemented here as the analytic-machinery side, with slot 296 calling them on its formal-series side.
- **Slot 295 consumer.** Riemann-Siegel formula for ζ(½+it) is the textbook example of saddle-point applied to the Riemann-Siegel integral representation; slot 295 needs T1 (saddle-point) to derive RS coefficients. Approximate functional equation truncation bound also asymptotic (slot 295 T7).
- **prob/ consumers.** Edgeworth expansion ≡ inverse-FT of cumulant-expanded characteristic function — direct application of saddle-point (Daniels 1954, "Saddlepoint approximations in statistics", Ann. Math. Stat. 25:631-650). Cornish-Fisher is the formal inverse of Edgeworth: solve normal-quantile equation with skewness/kurtosis corrections. Both are ~30 LOC each given LogGamma already shipped.
- **Bender-Orszag (1978) "Advanced Mathematical Methods for Scientists and Engineers"** is the canonical syllabus: ch. 6 (asymptotic expansions of integrals) covers Watson, Laplace, stationary-phase, steepest-descent in 80 pages. **Bleistein-Handelsman (1986)** is the rigorous reference. **Olver (1974) "Asymptotics and Special Functions"** anchors the special-function-asymptotics side. **Wong (1989) "Asymptotic Approximations of Integrals"** is the encyclopedia. **DLMF chapter 2 (https://dlmf.nist.gov/2)** is the free authoritative reference; sections 2.3 (Laplace), 2.4 (saddle-point), 2.7 (resurgence/Stokes) are directly translatable.
- **Hyperasymptotics (Berry-Howls 1991, "Hyperasymptotics for integrals with saddles", Proc. R. Soc. A 434:657-675).** Optimal truncation of asymptotic series at smallest term, then exponentially-improved correction via secondary saddle-point integral. Frontier; defer.
- **Borel summation (Costin 2009 "Asymptotics and Borel Summability", Chapman & Hall).** Resum divergent factorial series Σ aₙ zⁿ via Borel transform Σ aₙ/n! followed by Laplace transform back. The integrand-side counterpart of slot 296 T6 BorelTransform — **slot 296 ships the formal Borel transform** (a_n → a_n/n!), **this slot ships the Laplace-back integral that completes summation**.
- **Padé approximant.** Rational [m,n] approximation built from a Taylor series via Toeplitz solve. `e_x` Padé[4,4] vs Taylor agree to 8 digits on |x|<2. Used everywhere: Bode plots (control/), Carleman linearization (chaos/), control-system reduction. **Standalone primitive useful far beyond asymptotics.** ~50 LOC.
- **Mellin-Barnes integrals.** Hypergeometric functions ₂F₁, ₃F₂, Meijer-G all have Mellin-Barnes contour-integral representations; closing the contour yields a residue sum. Computer-algebra heavy; defer to T6.
- **Stokes phenomenon.** As λ rotates through arg z = ±π/2, asymptotic-expansion coefficients jump by an exponentially-small "Stokes multiplier". Detector + multiplier extraction is research-grade; T13 frontier.
- **Plancherel-Rotach for orthogonal polynomials.** Hermite Hₙ(x), Laguerre Lₙ^α(x), Jacobi Pₙ^{α,β} all have asymptotic forms in different x-regions (oscillatory inside the bulk, exponential decay outside, transition Airy function near the edge). Used in random-matrix tail probabilities (slot 217 free probability consumer). T10.
- **Hayman's method (1956 "A generalisation of Stirling's formula", J. Reine Angew. Math. 196:67-95).** Saddle-point applied to `aₙ = (1/2πi) ∮ f(z)/z^{n+1} dz` for admissible `f`. Yields `aₙ ~ f(rₙ) / (rₙⁿ √(2π b(rₙ)))` where rₙ solves `a(rₙ) = n` and a, b are explicit `f`-derivatives. Direct path from GF (slot 296) to coefficient asymptotics. **Hayman is the cleaner alternative to Darboux when GF is entire** (Bell numbers, eˣ-class).
- **Darboux's method.** Asymptotics of `[zⁿ]f` from analytic structure of `f` near its dominant singularity. Composes slot 296 GF singularity-locator. T11.
- **Daniels saddle-point in statistics (1954).** Standard density approximation: `f(x) ≈ exp(K(t̂) - t̂x) / √(2π K''(t̂))` where K is the cumulant-generating function and t̂ solves K'(t̂) = x. Used for rare-event tails, importance sampling, financial pricing of out-of-the-money options. Direct prob/ primitive.
- **Automatic asymptotic from D-finite ODEs.** Given linear ODE `Σ pₖ(z) f^{(k)}(z) = 0`, classify singular points (regular vs irregular), Frobenius-method local solutions, asymptotic structure determined by indicial polynomial. Maple's `gfun:-asymptotic` GPL-2; `mpmath` BSD-3 has partial. Frontier.
- **Moat estimate.** Mathematica `Series[…,Asymptotic→True]` proprietary; SageMath builds on Maxima/SymPy GPL-3; Maple `asympt` GPL-2; Pari `asympt` GPL-2; mpmath BSD-3 has Padé but not Watson/saddle-point. **Pure-Go MIT zero-dep asymptotic-analysis library = ABSENT.** Reality could be the first. **Moderate-to-high moat, very high research utility** — most "fancy" math computations USE asymptotic methods at the bottom.
- **Single cheapest day-1 PR.** T4 Stirling expansion (~50 LOC) + T7 Padé approximant (~50 LOC) + T8 Edgeworth (~80 LOC) + T0 Laplace skeleton (~120 LOC) = ~300 LOC. T7 alone is the highest leverage standalone primitive (used as series-acceleration anywhere).

## Concrete recommendations

1. **T0 — `asymptotic/laplace.go` Laplace's method.** `LaplaceMethod(g, f func(float64) float64, fpp func(float64) float64, x0, lambda float64, terms int) float64` returning ∫ g(x) exp(-λ f(x)) dx ≈ g(x₀) e^{-λ f(x₀)} √(2π/(λ f''(x₀))) · Σ aₖ / λᵏ. Higher-order coefficients aₖ from Erdélyi-style Taylor expansion of g/√f'' at x₀. ~120 LOC. Pin: ∫₀^∞ e^{-λx²}dx = ½√(π/λ) — direct closed form ≡ Laplace(g=1, f=x², x₀=0) ≡ stdlib `math.Sqrt(math.Pi/lambda)/2` — R-MUTUAL 3/3.

2. **T1 — `asymptotic/saddlepoint.go` saddle-point method (real saddle).** `SaddlePoint(g, f func(complex128) complex128, fpp func(complex128) complex128, z0 complex128, lambda float64) complex128` for ∫ g(z) exp(λ f(z)) dz over a steepest-descent contour through saddle z₀ where f'(z₀)=0. ~80 LOC. Composed with T0 by phase rotation. Pin: ∫₋∞^∞ e^{iλx²/2}dx = √(2π/λ)·e^{iπ/4} ≡ Fresnel integral ≡ SaddlePoint(f=ix²/2). R-MUTUAL 3/3.

3. **T2 — `asymptotic/steepest_descent.go` rigorous steepest-descent.** Composes T1 with explicit contour deformation from the integration path to the steepest-descent path through the saddle; argument-of-f''(z₀) determines local angle. Required for non-trivial complex contours (Riemann-Siegel, Airy function asymptotic). ~100 LOC. Pin: Airy `Ai(x) ~ exp(-(2/3)x^{3/2})/(2√π x^{1/4})` for x → +∞ ≡ steepest-descent on Ai's integral representation ≡ existing `math.Pow(x, -0.25) * math.Exp(-2.0/3.0*math.Pow(x, 1.5)) / (2*math.Sqrt(math.Pi))`. Locked once Ai is shipped.

4. **T3 — `asymptotic/watson.go` Watson's lemma.** Special case of Laplace at endpoint x=0: `Watson(coefs []float64, lambda float64) float64` computes ∫₀^∞ x^α (a₀ + a₁ x + …) e^{-λx} dx = Σ aₖ Γ(α+k+1) / λ^{α+k+1}. ~30 LOC. Calls existing `prob.LogGamma`. Pin: `Watson(coefs={1}, λ=1, α=0) = Γ(1) = 1` ≡ closed form. R-MUTUAL 3/3 vs both stdlib `Gamma` and Stirling expansion T4.

5. **T4 — `asymptotic/stirling.go` Stirling's expansion via Bernoulli numbers.** `LogGammaStirling(x float64, terms int) float64` returns log Γ(x) ≈ (x-½)log x − x + ½log(2π) + Σ B_{2k}/(2k(2k-1) x^{2k-1}). Bernoulli numbers from constants/ if shipped, else hardcode B₂..B₂₀. ~50 LOC. **R-MUTUAL 3/3 PIN:** stdlib `math.Lgamma` (Lanczos) ≡ `LogGammaStirling(x, terms=10)` for x≥10 to ~1e-13 ≡ exact `combinatorics.Factorial(n-1)` log for integer n=20. Three-way agreement.

6. **T5 — `asymptotic/borel.go` Borel summation.** `BorelSum(coefs []float64, z float64) float64` resums `Σ aₙ zⁿ` via `∫₀^∞ e^{-t} (Σ aₙ tⁿ/n!) dt|_{t→tz}`. Composes slot 296's `BorelTransform` (formal a_n→a_n/n!) with Laplace-back integral computed via T0. ~70 LOC. Validates on Euler's divergent series `Σ (-1)ⁿ n! zⁿ` ≡ Ei-class integral.

7. **T6 — `asymptotic/mellin_barnes.go` Mellin-Barnes contour integration.** `MellinBarnes(numeratorPoles, denominatorPoles []complex128, residueGen func(complex128) complex128) float64` evaluates ∫_{c-i∞}^{c+i∞} (Γ-product) z^{-s} ds / (2πi) by closing contour and summing residues. ~150 LOC. **Day-2** — required for Meijer-G and ₂F₁ outside Taylor radius.

8. **T7 — `asymptotic/pade.go` Padé approximant.** `Pade(taylor []float64, m, n int) (num, den []float64)` builds rational [m,n] approximant from m+n+1 Taylor coefficients. Toeplitz solve via existing linalg. ~50 LOC. **Day-1.** Standalone utility for series acceleration anywhere. **R-MUTUAL 3/3 PIN:** Padé[4,4] of e^x at x=0 ≡ Taylor[8] of e^x ≡ stdlib `math.Exp` agree to 8 digits on |x|<2.

9. **T8 — `prob/edgeworth.go` Edgeworth expansion + Cornish-Fisher inverse.** `EdgeworthCDF(z, skew, exKurt float64) float64` returns Φ(z) − φ(z)·[γ₁ He₂(z)/6 + γ₂ He₃(z)/24 + γ₁² He₅(z)/72]. `CornishFisher(p, skew, exKurt float64) float64` inverts via series. ~80 LOC. Both call existing `prob.NormalCDF`/`NormalQuantile`. **R-MUTUAL 3/3 PIN:** Standard normal (skew=0, exKurt=0) ≡ existing `NormalCDF(z)` exactly; Gamma(k=10) Edgeworth approximation ≡ existing `GammaCDF(x, k=10, theta=1)` to 1e-3 in bulk. Locked.

10. **T9 — `asymptotic/hayman.go` Hayman's method for combinatorial GF coefficients.** `HaymanCoeff(f, fp, fpp func(float64) float64, n int) float64` solves `r·f'(r)/f(r) = n` for r, then returns `f(r) / (rⁿ √(2π b(r)))` with b explicit. ~60 LOC. Composes slot 296's GF representation. **Pin:** Bell numbers via Hayman on `exp(eˣ-1)` ≡ `combinatorics.BellNumber(n)` exact for n=20. R-MUTUAL 3/3 (slot 296 T10 Flajolet-Odlyzko third leg).

11. **T10 — `asymptotic/plancherel_rotach.go` orthogonal-polynomial asymptotics.** Hermite Hₙ(x) in oscillatory bulk |x|<√(2n+1), exponential outside, Airy transition. Laguerre/Jacobi analogous. ~100 LOC each. **Day-2** consumer of slot 217 (free probability) and slot 220.

12. **T11 — `asymptotic/darboux.go` Darboux's method.** Given GF `f(z) = g(z) (1−z/r)^{α} log^k(1−z/r) + analytic`, return `[zⁿ]f ~ g(r) r^{-n} n^{-α-1}/Γ(-α) (log n)^k`. Composes slot 296's singularity locator. ~70 LOC. **Pin:** Catalan via Darboux on `(1−√(1−4z))/(2z)` near z=¼ ≡ `combinatorics.CatalanNumber(n)` ≡ Stirling-validated 4ⁿ/(n^{3/2}√π).

13. **T12 — `asymptotic/hyperasymptotic.go` Berry-Howls hyperasymptotics.** Optimal truncation index N* ≈ |λ| (smallest term), exponentially-improved remainder via secondary saddle. ~200 LOC. **Frontier; defer past day-3.**

14. **T13 — `asymptotic/stokes.go` Stokes phenomenon detector + multipliers.** Detect Stokes lines (arg z where Re(λf(z₁)−λf(z₂)) = 0 between two saddles), compute Stokes multiplier (Berry's smoothing of the discontinuity). ~250 LOC. **Frontier; research-grade; defer.**

## Cross-cutting

- **slot 295 (l-functions)** ← T1 saddle-point + T2 steepest-descent derive Riemann-Siegel formula coefficients for ζ(½+it). Slot 295 T7 (generic AFE truncation bound) is a Watson-lemma application.
- **slot 296 (generating-functions)** ← T9 Hayman's method + T11 Darboux's method are the analytic-side primitives slot 296's symbolic-method T10 transfer theorem composes with. Three-way pin: Catalan / Bell / Fib via slot 296 closed-form ≡ this slot's saddle-point ≡ stdlib stirling.
- **prob/ (Edgeworth, Cornish-Fisher, Daniels saddlepoint density approximation)** ← T1 + T8. Density approximations, rare-event tails, importance-sampling control variates.
- **slot 220 (saddle-point for SDE rare events)** ← T1 saddle-point on the Onsager-Machlup action functional / Freidlin-Wentzell rate function.
- **slot 217 (free probability)** ← T10 Plancherel-Rotach for orthogonal-polynomial-edge eigenvalue distributions in random matrix theory.
- **slot 244 (PDE solvers) + slot 264 (MLMC)** ← T0 Laplace + T1 saddle-point for asymptotic boundary-layer analysis (Bender-Orszag ch. 9-11) and rare-event Monte Carlo importance sampling.
- **slot 269 (GP state-space)** ← T8 Edgeworth on non-Gaussian residuals; T1 saddle-point for marginal-likelihood approximation.
- **slot 281 (temporal graphs, large-time correlation asymptotics)** ← T0 Laplace + T11 Darboux for `n→∞` count expansions.
- **calculus/** ← T7 Padé as series-acceleration plug-in for Romberg-style table extrapolation; complements existing GaussLegendre.
- **constants/** ← Bernoulli numbers B₂..B₂₀ shipped here as a side-effect of T4 Stirling; consumer of slot 295's separate Bernoulli ask.
- **aicore advanced applied math** ← saddle-point density approximation underlies variational approximate-inference normalizers; T1 + T8 are direct.

## Day-1 PR (~310 LOC, 4 R-MUTUAL 3/3 pins)

```
asymptotic/
  stirling.go      ~50 LOC   T4   pin: stdlib Lgamma ≡ Stirling[10] ≡ Factorial(20)
  pade.go          ~50 LOC   T7   pin: Padé[4,4] e^x ≡ Taylor[8] ≡ math.Exp
  laplace.go      ~120 LOC   T0   pin: ∫e^{-λx²}dx = ½√(π/λ) ≡ Laplace ≡ stdlib
  watson.go        ~30 LOC   T3   uses prob.LogGamma; pin Γ(α+k+1) values

prob/
  edgeworth.go     ~80 LOC   T8   pin: skew=0 ≡ NormalCDF; Gamma(k=10) ≡ GammaCDF
```

Day-2: T1 SaddlePoint, T9 Hayman (cross-validates slot 296), T11 Darboux. Day-3: T2 SteepestDescent, T5 Borel, T10 Plancherel-Rotach. Frontier (defer): T6 Mellin-Barnes, T12 hyperasymptotic, T13 Stokes.

## Sources

**Repo files (zero asymptotic content):**
- `C:/limitless/foundation/reality/calculus/calculus.go:31-244` (NumericalDerivative, Trapezoidal, Simpson, GaussLegendre, MonteCarlo — finite-λ only).
- `C:/limitless/foundation/reality/prob/mathutil.go:13-26` (LogGamma wraps math.Lgamma — substrate for Stirling pin).
- `C:/limitless/foundation/reality/prob/distributions.go` (NormalCDF, GammaCDF, BetaPDF — Edgeworth/Cornish-Fisher consumers; no Skewness/Kurtosis primitives).
- `C:/limitless/foundation/reality/prob/copula/studentt.go:52` (only mention of "Cornish-Fisher" anywhere — comment, not implementation).
- `C:/limitless/foundation/reality/combinatorics/counting.go` (10 closed-forms whose asymptotic expansions are the natural cross-validation targets).
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/296-new-generating-functions.md` (slot 296 owns formal-series Borel transform; this slot owns Laplace-back integral; Hayman/Darboux are the analytic-side bridge).
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/295-new-l-functions.md` (slot 295 Riemann-Siegel needs T1 saddle-point).

**Web (textbook canon):**
- Bender & Orszag (1978/1999) "Advanced Mathematical Methods for Scientists and Engineers", Springer. Ch. 6 = Watson, Laplace, stationary-phase, steepest-descent — 80-page direct syllabus.
- Bleistein & Handelsman (1986) "Asymptotic Expansions of Integrals", Dover (rigorous reference).
- Olver (1974/1997) "Asymptotics and Special Functions", AKP Classics — special-function asymptotics.
- Wong (1989/2001) "Asymptotic Approximations of Integrals", SIAM Classics — encyclopedia.
- DLMF chapter 2 (https://dlmf.nist.gov/2) — free authoritative reference; §2.3 Laplace, §2.4 saddle-point, §2.7 resurgence/Stokes.
- Berry & Howls (1991) "Hyperasymptotics for integrals with saddles", Proc. R. Soc. A 434:657-675.
- Costin (2009) "Asymptotics and Borel Summability", Chapman & Hall/CRC Monographs.
- Daniels (1954) "Saddlepoint approximations in statistics", Ann. Math. Stat. 25:631-650 — Edgeworth/Cornish-Fisher/Daniels density.
- Hayman (1956) "A generalisation of Stirling's formula", J. Reine Angew. Math. 196:67-95.
- Flajolet & Sedgewick (2009) "Analytic Combinatorics", CUP — Darboux ch. VI, Hayman ch. VIII (free PDF on Flajolet's INRIA page).
- Boyd (2001/2014) "Chebyshev and Fourier Spectral Methods", 2nd ed., Dover — Padé and series-acceleration appendix.
- Plancherel & Rotach (1929) "Sur les valeurs asymptotiques des polynomes d'Hermite Hₙ(x)…", Comm. Math. Helv. 1:227-254.

**Moat (existing implementations, all non-MIT-Go-compatible):**
- Mathematica `Series[…,Asymptotic→True]` / `AsymptoticIntegrate` — proprietary.
- Maple `asympt`, `gfun:-asymptotic` — GPL-2.
- SageMath `asymptotic_expansions` — GPL-3.
- Pari/GP `asympt` — GPL-2.
- mpmath (Python BSD-3) `pade`, `taylor`, `chop` — partial; no Watson/saddle-point primitives.

**Pure-Go MIT zero-dep asymptotic-analysis library = absent. Reality would be first.**
