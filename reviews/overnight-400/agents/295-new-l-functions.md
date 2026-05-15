# 295 — new-l-functions (Riemann ζ / Dirichlet L / Hurwitz / Lerch / Polylog / Modular L)

## Headline
Reality v0.10.0 ships ZERO L-function / zeta surface (no `Zeta`, `RiemannZeta`, `HurwitzZeta`, `LerchZeta`, `Polylog`, `DirichletL`, `EulerMaclaurin`, `RiemannSiegel`, `GramPoint`, `Stieltjes`, `DedekindZeta`, `ArtinL`, `AutomorphicL` callable hits in any `*.go` source — only false-positive on the Greek letter "zeta" in `sequence/token_ratio_test.go`); `EulerGamma` is the only relevant constant present (`constants/math.go:70`); the Bernoulli substrate required for ζ(2k) closed forms is **owned by slot 294's T_substrate** and not yet shipped, so this slot's day-1 PR depends on slot 294 landing first OR ships its own Bernoulli table; the singular cheapest day-1 PR is **T0 Bernoulli (or slot-294 reuse) + T1 Riemann ζ via Euler-Maclaurin + T2 special values closed forms (~330 LOC)** which immediately enables four R-MUTUAL-CROSS-VALIDATION 3/3 pins (ζ(2)=π²/6, ζ(4)=π⁴/90, ζ(-1)=-1/12, Apéry ζ(3)) and unblocks slot 294 T12 modular L-functions and slot 292 BSD-conjecture verification; the singular highest-strategic-value primitive is **T3 Riemann-Siegel formula for ζ(½+it) (~250 LOC)**, the only known practical algorithm for moderate-t numerical zero verification.

## Findings

### State at HEAD (v0.10.0, 2026-05-09)

Repo-wide grep on `Zeta|RiemannZeta|HurwitzZeta|LerchZeta|DirichletL|LFunction|EulerMaclaurin|RiemannSiegel|GramPoint|TuringMethod|Stieltjes|Polylog|AutomorphicL|ArtinL|DedekindZeta` (case-insensitive) across `**/*.go`: **single hit, false-positive** in `sequence/token_ratio_test.go:N` ("delta epsilon zeta" as Greek-letter test fixture). **Zero callable surface.**

| Surface needed by this slot | Path | Status |
|---|---|---|
| Bernoulli numbers B_k as `*big.Rat` | absent | **REQUIRED** for ζ(2k) closed form `ζ(2k) = (-1)^(k+1) (2π)^(2k) B_(2k) / (2·(2k)!)`; **owned by slot 294 T_substrate** per `reviews/agents/294-new-modular-forms.md:84-97`. |
| Euler-Mascheroni γ | `constants/math.go:70 EulerGamma = 0.5772156649015329` | Present, float64. |
| Catalan G ≈ 0.91596... | absent | Optional; appears in some Dirichlet β values. |
| Apéry ζ(3) ≈ 1.20205690315959... | absent | Useful golden value. |
| Gamma function Γ(s), complex | absent in `*.go` (only stdlib `math.Gamma` real-arg) | **REQUIRED** for functional equation ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s); slot 294 T1 transitively needs complex Γ for analytic j(τ). |
| `complex128` arithmetic | Go stdlib `math/cmplx` | Sufficient for moderate-precision (~14 digit) ζ. |
| Multiple-precision float | `math/big.Float` | Sufficient for high-precision ζ; slow without FFT-based mult. |
| Dirichlet character χ mod q | absent | **REQUIRED** for L(s, χ); composes slot 290 number-field characters and slot 291 modular arithmetic. |
| Number-field arithmetic (ζ_K) | absent (slot 290 owns) | **REQUIRED** for Dedekind zeta ζ_K(s); slot 290 ships substrate. |
| q-series / newforms | absent (slot 294 owns) | **REQUIRED** for modular L-functions L(f, s) of newform f. |

### Cross-slot orientation

| Adjacent slot | Overlap | Resolution |
|---|---|---|
| **294 modular-forms** | T12 of slot 294 IS the modular-L-function `L(f, s) = Σ a_n(f)/n^s` (`reviews/agents/294-new-modular-forms.md:256-263`). Same mathematical object as this slot's T13. | **Slot 294 ships modular L-function as direct evaluator from a newform's q-expansion. Slot 295 ships the GENERIC L-function machinery (Euler-Maclaurin, Riemann-Siegel, approximate functional equation, Gram-point zero finder) that newform L-functions are an instance of.** Boundary: `LSeriesEval(coeffs []complex128, s complex128) complex128` lives here (T7 generic Dirichlet-series eval); `ModularLFunction(f *QSeries, s complex128) complex128` lives in slot 294 and *delegates to* slot 295's T7. |
| **290 Galois theory** | Artin L-functions = Galois-representation L-functions; Dedekind zeta ζ_K(s) is the ζ-fn of a number field, factorable as Π L(s, ρ) over Galois reps. | Slot 290 owns number-field arithmetic + Galois groups + character tables; slot 295 T11 `DedekindZeta(K *NumberField, s complex128) complex128` consumes them. T12 Artin L same — slot 290 produces the Galois rep, slot 295 evaluates the L-function. |
| **292 elliptic curves** | Wiles modularity: every E/Q has a modular newform f_E with L(E, s) = L(f_E, s). BSD conjecture: ord_{s=1} L(E, s) = rank(E(Q)). | Slot 292 + slot 294 + slot 295 all contribute. Slot 292 ships E; slot 294 ships f_E; slot 295 evaluates L(f_E, s) numerically and checks BSD by computing L(E, 1) and L'(E, 1) at the central point. |
| **293 NTT / 291 modular arithmetic** | NTT used in fast L-function arithmetic (Schönhage-style fast partial sums); modular arithmetic underlies Dirichlet characters mod q. | Optional speedup. |
| **prob package** | Number-theoretic distribution of zeros of ζ on the critical line follows GUE statistics (Montgomery-Odlyzko law) — random-matrix-theory cross-validation. | Cross-validation pin; not a hard dep. |
| **autodiff package** | L'(E, 1) — the central-point *derivative* of an L-function — is needed for BSD rank-1 case (Gross-Zagier). Complex differentiation. | Composes if shipped; standalone numerical differentiation also works. |
| **info package** | Apéry constant ζ(3), Catalan G, π²/6 appear in entropy / KL bounds. | Recreational link. |
| **constants package** | Apéry ζ(3) and Catalan G belong here as float64 literals. | Add 2 lines in T2 alongside ζ(2k) closed-form helpers. |
| **calculus package** | Euler-Maclaurin formula = trapezoidal correction with Bernoulli-number remainder. Adjacent to existing `Simpson` / `Trapezoidal` integrators. | Could ship `EulerMaclaurinSum(f, a, b int, M int) float64` as a *general* numerical-summation primitive in `calculus/`, then `RiemannZeta` becomes a one-line specialization. **Cleaner factorization than burying it inside `lfunc/zeta.go`.** |

### Distinct-from-slot-294 disambiguation

Slot 294 owns:
- Modular newform f as q-series object.
- Hecke eigenvalues a_p(f) as q-coefficients.
- L(f, s) = Σ a_n(f) n^(-s) **as a derived q-coefficient sum, formal level**.
- Approximate functional equation specialized to weight-k newforms with N-level functional-equation root number ε.

Slot 295 owns:
- Riemann ζ(s) = Σ n^(-s) for all s ∈ ℂ \ {1}, the *prototype* L-function with trivial conductor, weight 1, root number 1.
- Hurwitz ζ(s, a) = Σ (n+a)^(-s) — generalizes Riemann.
- Lerch Φ(z, s, a) = Σ z^n / (n+a)^s — generalizes Hurwitz and polylog.
- Polylogarithm Li_s(z) = Σ z^n / n^s — special case Φ(z, s, 1) with z prefactor shift.
- Dirichlet L(s, χ) for **arbitrary** χ mod q (not just modular-form-attached).
- **Generic L-function evaluator** `L(coeffs, conductor, weight, sign, s)` via approximate functional equation (Rubinstein 2005).
- **Zeros of ζ on critical line**: Riemann-Siegel-Z function, Gram points, zero counting N(t), Turing's method.

Boundary: `RiemannZeta(s complex128) complex128` is HERE. `LFunctionFromNewform(f *QSeries, s complex128) complex128` is in slot 294 and **calls back into slot 295's generic approximate-functional-equation evaluator**. The newform-specific arithmetic stays in slot 294; the analytic-continuation machinery stays in slot 295.

### Does math/big or stdlib cover this?

`math` ships `math.Gamma(real)` only. `math/cmplx` has no Γ, no ζ, no Bessel — nothing in the special-functions tier above `cmplx.Sqrt` / `cmplx.Exp`. `math/big` ships `*big.Float` — sufficient substrate, but no ζ. **No stdlib zeta, no Hurwitz, no polylog, no Dirichlet L.** `golang.org/x/...` has nothing.

| Library | License | Coverage | Reality moat |
|---|---|---|---|
| mpmath (Python) | BSD-3 | Riemann ζ, Hurwitz ζ, Lerch Φ, polylog, Dirichlet η, all to arbitrary precision; Borwein / Riemann-Siegel / Euler-Maclaurin auto-selected | Python-only; not Go. **MIT-compatible license.** Reality could *port the algorithms* (algorithm description is not copyrightable; clean-room reimplementation from the cited papers, not the code). |
| PARI/GP `lfun` | GPL-2 | Full L-function suite including Artin / Dedekind / modular | GPL — not MIT-compatible. |
| Arb (FLINT) | LGPL | Rigorous-error-bound ζ, Hurwitz, Lerch, polylog at arbitrary precision; uses Hiary's exponent-1/3 method for ζ(½+it) | LGPL — copyleft, contaminating in static-link scenarios. |
| LMFDB | MIT data, GPL engines (PARI/Sage) | Precomputed tables of zeros, special values for tens of thousands of L-functions | Data layer MIT but generated by GPL software — derivative-work questions. |
| `golang.org/x/...` | BSD-3 | NONE | — |
| Mathematica `Zeta[s]`, `HurwitzZeta`, `LerchPhi`, `PolyLog`, `DirichletL` | proprietary | Full | Closed source. |
| SageMath | GPL-3 | Full via PARI | GPL. |

**Reality's positioning**: would be the FIRST MIT pure-Go zero-dep L-function library. Audience: academic analytic-number-theorists, BSD-conjecture researchers, mathematical physicists doing zeta-function regularization (string theory, Casimir effect via ζ(-3) = 1/120), prime-distribution / explicit-formula consumers. **Modest moat — researchers default to Python (mpmath) or PARI; Reality's value is "available *in Go* without GPL contamination" rather than "fastest" or "highest-precision".**

### Practically infinite frontier — commit to subset with clear consumer

L-function theory is unbounded (Iwaniec-Kowalski 2004 *Analytic Number Theory* AMS Colloq. 53 is 615 pages). Recommended subset:

- **MUST ship** (slot 294 modular L-functions hard-depend, BSD-conjecture verification, slot-internal cross-validation): T0 Bernoulli (slot-294-shared), T1 Riemann ζ Euler-Maclaurin, T2 special values + Apéry/Catalan constants, T7 generic Dirichlet-series approximate-functional-equation evaluator.
- **SHOULD ship** (Reality "universal truth" positioning + standard textbook surface): T3 Riemann-Siegel ζ(½+it), T4 Hurwitz ζ, T5 Lerch Φ, T6 polylog Li_s(z), T8 Dirichlet L(s, χ).
- **NICE-to-ship** (research utility): T9 Stieltjes constants γ_n, T10 Gram points + Riemann-Siegel-Z function + zero counting N(t).
- **DEFER** (frontier, no Reality consumer for v1): T11 Dedekind ζ_K, T12 Artin L, T13 modular L (slot 294 ships its own), T14 Turing's method for rigorous N(t), T15 automorphic L-functions, T16 Hiary exponent-1/3 algorithm (extreme-precision; not needed at float64).

## Concrete recommendations

Tier numbering: T0 = Bernoulli substrate (shared with slot 294); T1 = Riemann ζ via Euler-Maclaurin (load-bearing); T2 = special values closed forms; T3 = Riemann-Siegel ζ(½+it); T4 = Hurwitz; T5 = Lerch; T6 = polylog; T7 = generic L-function evaluator; T8 = Dirichlet L; T9 = Stieltjes; T10 = Gram points / Z(t) / zero counting; T11-T16 = frontier.

Proposed package name: `lfunc/` (avoiding `zeta/` because the package covers L-functions broadly; avoiding `ntheory/` because slot 290 may claim that).

### T0 — Bernoulli numbers — shared with slot 294 T_substrate

If slot 294 lands first: `import "github.com/davly/reality/constants"` and use `constants.BernoulliRat(k)`. Zero LOC here.

If slot 295 lands first: ship `constants/bernoulli.go` ~80 LOC with the same signature `func BernoulliRat(k int) *big.Rat` and a sibling `func BernoulliFloat(k int) float64` for fast float64 use. Frozen literal table B_0…B_30. Slot 294 then imports the same. **Coordinate ship order.**

### T1 — `lfunc/zeta.go` ~150 LOC — DAY-1 KEYSTONE

Riemann ζ(s) for arbitrary s ∈ ℂ \ {1} via Euler-Maclaurin summation:

```go
// RiemannZeta returns ζ(s) for s ∈ ℂ \ {1}.
// Algorithm: Euler-Maclaurin summation. Choose N s.t. error ≤ 10^-15.
//   ζ(s) = Σ_{n=1}^{N-1} n^(-s) + N^(1-s)/(s-1) + N^(-s)/2
//        + Σ_{k=1}^{M} B_(2k) / (2k)! · (s)(s+1)...(s+2k-2) · N^(-s-2k+1)
//        + R_M (remainder, |R_M| ≤ next-term magnitude bound).
// For Re(s) ≤ 0, use functional equation ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s)
// to reduce to Re(s') ≥ 1 case.
// Source: Edwards 1974 §6.4; Cohen 2007 *Number Theory II* §10.3.
func RiemannZeta(s complex128) complex128

// RiemannZetaReal is a real-arg specialization for σ > 0, σ ≠ 1.
func RiemannZetaReal(sigma float64) float64
```

Default tuning: N = 10, M = 8 yields ~14 decimal digits for |Im(s)| ≤ 50. For larger |Im(s)|, the Euler-Maclaurin tail explodes — divert to T3 Riemann-Siegel.

**R-MUTUAL-CROSS-VALIDATION 3/3 pins**:
1. **Closed-form pin**: `RiemannZeta(2)` ≡ π²/6 to 10⁻¹⁴; `RiemannZeta(4)` ≡ π⁴/90; `RiemannZeta(6)` ≡ π⁶/945; `RiemannZeta(-1)` ≡ -1/12 (Casimir / string theory normalization); `RiemannZeta(-3)` ≡ 1/120; `RiemannZeta(0)` = -1/2.
2. **Functional equation pin**: compute ζ(s) directly for Re(s) > 1; compute ζ(1-s) via Euler-Maclaurin and apply functional equation to recover ζ(s); both paths must agree to 10⁻¹².
3. **Apéry pin**: `RiemannZeta(3)` ≡ 1.2020569031595942853997381... to 10⁻¹⁴ (OEIS A002117).

Three independent computational paths agreeing on the keystone value ⇒ saturates R-MUTUAL-CROSS-VALIDATION 3/3.

Refs: Edwards 1974 *Riemann's Zeta Function* ch. 6; Cohen 2007 *Number Theory Vol. II: Analytic and Modern Tools* GTM 240 §10.3; Borwein-Bradley-Crandall 2000 *Computational strategies for the Riemann zeta function* J. Comp. Appl. Math. 121:247-296.

### T2 — `lfunc/special_values.go` ~80 LOC + `constants/special.go` ~30 LOC — DAY-1 KEYSTONE

```go
// ζ(2k) = (-1)^(k+1) (2π)^(2k) B_(2k) / (2·(2k)!) for k ≥ 1.
// ζ(-(2k-1)) = -B_(2k) / (2k) for k ≥ 1.
// ζ(-(2k)) = 0 for k ≥ 1 (trivial zeros).
// ζ(0) = -1/2.
func RiemannZetaEven(k int) float64    // exact via Bernoulli
func RiemannZetaNeg(n int) float64     // ζ(-n) via Bernoulli (incl. trivial zeros)
```

Constants:
```go
// Apery is Apéry's constant ζ(3) = 1.2020569031595942...; Source: OEIS A002117.
const Apery = 1.2020569031595943
// Catalan is Catalan's constant G = Σ(-1)^n/(2n+1)² = 0.9159655941772190...; Source: OEIS A006752.
const Catalan = 0.915965594177219
```

These compose with T1: `RiemannZeta(2)` evaluator and `RiemannZetaEven(1)` closed form must agree byte-for-byte (within representation error ≤ 1 ulp). This is the closed-form arm of T1's three-way pin.

### T3 — `lfunc/riemann_siegel.go` ~250 LOC — STRATEGIC KEYSTONE

Riemann-Siegel formula for ζ(½+it), the standard practical algorithm for moderate t (10⁻¹ ≤ t ≤ 10⁶). Cost O(√(t/(2π))) — **dramatically faster** than Euler-Maclaurin for large t.

```go
// RiemannSiegelZ returns Z(t) = e^(iθ(t)) ζ(½+it), the Riemann-Siegel-Z function.
// Real-valued for real t; sign changes ↔ zeros of ζ on the critical line.
//   Z(t) = 2 Σ_{n=1}^{N} cos(θ(t) - t ln n) / √n + remainder R(t)
// where N = ⌊√(t/(2π))⌋ and R(t) given by Riemann-Siegel asymptotic expansion (4 terms suffice for t ≤ 10⁶, ~10⁻¹² accuracy).
// Source: Edwards 1974 §7; Berry 1995 *The Riemann-Siegel formula for the zeta function: high orders and remainders*.
func RiemannSiegelZ(t float64) float64

// RiemannSiegelTheta returns θ(t) = arg Γ(¼ + it/2) - (t/2) ln π, the Riemann-Siegel theta function.
// Source: Edwards 1974 §6.5; uses Stirling's approximation for Im Γ.
func RiemannSiegelTheta(t float64) float64

// RiemannZetaCriticalLine returns ζ(½+it).
//   ζ(½+it) = Z(t) e^(-iθ(t))
func RiemannZetaCriticalLine(t float64) complex128
```

**Three-way cross-validation**: for t ∈ [0, 100], evaluate ζ(½+it) via (a) Euler-Maclaurin T1, (b) Riemann-Siegel T3, (c) direct partial sum N=10⁶ for sanity. All three agree to 10⁻¹⁰ ⇒ R-MUTUAL-CROSS-VALIDATION 3/3.

**First-zero pin**: first nontrivial zero of ζ at t₁ ≈ 14.134725141734693. `RiemannSiegelZ(14.134725141734693)` ≈ 0; sign change between t=14 and t=15. Pin against Odlyzko's tabulated first 1000 zeros (https://www.dtc.umn.edu/~odlyzko/zeta_tables/) — byte-identical regression at 10⁻⁹ precision for first 50 zeros.

Refs: Edwards 1974 §7; Berry-Keating 1992 *A new asymptotic representation for ζ(½+it) and quantum spectral determinants*; Odlyzko 1992 *The 10²⁰-th zero of the Riemann zeta function and 175 million of its neighbors* (unpublished manuscript, dtc.umn.edu); Hiary 2011 *Fast methods to compute the Riemann zeta function* Annals of Math 174(2):891-946 (frontier — exponent-1/3 method, not needed at float64).

### T4 — `lfunc/hurwitz.go` ~100 LOC

```go
// HurwitzZeta returns ζ(s, a) = Σ_{n≥0} (n+a)^(-s) for Re(s) > 1, a > 0.
// Analytic continuation via Euler-Maclaurin (same algorithm as T1, shifted).
// ζ(s, 1) = ζ(s) (Riemann); ζ(s, ½) = (2^s - 1) ζ(s).
// Source: Apostol 1976 *Introduction to Analytic Number Theory* GTM §12.
func HurwitzZeta(s complex128, a float64) complex128
```

Three-way pin: `HurwitzZeta(s, 1)` ≡ `RiemannZeta(s)`; `HurwitzZeta(s, 0.5)` ≡ `(math.Pow(2, s) - 1) * RiemannZeta(s)` (multiplication-form); `HurwitzZeta(s, 0.5)` direct computation. All three agree ⇒ R-MUTUAL-CROSS-VALIDATION 3/3.

### T5 — `lfunc/lerch.go` ~120 LOC

```go
// LerchPhi returns Φ(z, s, a) = Σ_{n≥0} z^n / (n+a)^s, |z| ≤ 1 (and analytic continuation).
// Generalizes Hurwitz (z=1) and polylog (a=1, with prefactor z).
// Algorithm: direct series for |z| ≤ 0.95 with Euler acceleration; for z near 1, contour integral.
// Source: Erdélyi 1953 *Higher Transcendental Functions* vol. I §1.11.
func LerchPhi(z complex128, s complex128, a float64) complex128
```

### T6 — `lfunc/polylog.go` ~100 LOC

```go
// Polylog returns Li_s(z) = Σ_{n≥1} z^n / n^s, |z| ≤ 1.
// Composes T5: Li_s(z) = z · Φ(z, s, 1).
// Special cases: Li_1(z) = -ln(1-z); Li_2(z) = Spence's function (dilogarithm); Li_3(z) = trilogarithm.
// Source: Lewin 1981 *Polylogarithms and Associated Functions*.
func Polylog(s complex128, z complex128) complex128
// Dilog is the dilogarithm Li_2(z), commonly needed standalone.
func Dilog(z complex128) complex128
```

Pin: Li_2(1) = π²/6 = ζ(2). Li_2(-1) = -π²/12 = -η(2) where η is Dirichlet eta. Li_2(½) = π²/12 - (ln 2)²/2 (Euler 1768). Three closed-form pins ⇒ R-MUTUAL-CROSS-VALIDATION 3/3.

### T7 — `lfunc/generic.go` ~180 LOC — load-bearing for slot 294 T12

**Generic Dirichlet-series approximate functional equation** (Rubinstein 2005 / Booker 2006). All L-functions L(s) satisfying L(s) = ε X(s) L̄(k-s) (where X(s) is the Γ-factor product, ε the root number, k the weight, and N the conductor) admit:

L(s) ≈ Σ_{n ≤ X₁} a_n n^(-s) G_+(s, n/√N) + ε Σ_{n ≤ X₂} a_n n^(s-k) G_-(k-s, n/√N)

where G_± are incomplete-Γ-style cutoff functions chosen so the sums converge in O(√N) terms.

```go
// LFunctionData fully describes an L-function for numerical evaluation.
type LFunctionData struct {
    Coeffs    func(n int) complex128 // Dirichlet coefficients a_n
    Conductor float64                // N (analytic conductor)
    Weight    int                    // motivic weight k (1 for ζ, 2 for elliptic, k for weight-k newform)
    Sign      complex128             // root number ε ∈ {±1, complex on unit circle}
    GammaFact func(s complex128) complex128 // Γ-factor product
}

// LValue evaluates L(s) via the approximate functional equation.
func LValue(L LFunctionData, s complex128) complex128
```

**This is the abstraction that slot 294 T12 `ModularLFunction(f *QSeries, s complex128) complex128` calls into**, supplying `Coeffs = func(n) { return f.Coeffs[n] }`, `Conductor = N` (level), `Weight = k`, `Sign = AtkinLehnerSign(f)`, `GammaFact = (2π)^(-s) Γ(s)`.

Pin: instantiate with `Coeffs = func(n) { return 1 }`, `Conductor = 1`, `Weight = 1`, `Sign = 1`, `GammaFact = π^(-s/2) Γ(s/2)` and recover Riemann ζ. Cross-check against T1 to 10⁻¹². Saturates **R-MUTUAL-CROSS-VALIDATION 3/3** (T1 vs T7, T3 vs T7 on critical line, closed form vs T7 at integers).

Refs: Rubinstein 2005 *Computational methods and experiments in analytic number theory* in Surveys in Number Theory, MSRI vol. 44; Booker 2006 *Artin's conjecture, Turing's method, and the Riemann hypothesis* Experimental Math. 15(4):385-407; Dokchitser 2004 *Computing special values of motivic L-functions* Experimental Math. 13(2):137-149.

### T8 — `lfunc/dirichlet.go` ~150 LOC

```go
// DirichletCharacter mod q, primitive or not.
type DirichletChar struct{ Modulus int; Values []complex128 }
// PrimitiveCharacters lists primitive characters mod q (Conrey labels).
func PrimitiveCharacters(q int) []DirichletChar
// DirichletL evaluates L(s, χ) = Σ χ(n) n^(-s) via T1's Euler-Maclaurin with character twist.
func DirichletL(s complex128, chi DirichletChar) complex128
```

Specializations:
- L(s, χ_trivial mod 1) = ζ(s). Pin against T1.
- L(s, χ_-4) (the unique non-trivial char mod 4) at s = 1: L(1, χ_-4) = π/4 (Leibniz 1673).
- L(2, χ_-4) = G (Catalan's constant). Three-way pin against T2 Catalan.

### T9 — `lfunc/stieltjes.go` ~120 LOC

```go
// Stieltjes constants γ_n: ζ(s) = 1/(s-1) + Σ_{n≥0} (-1)^n γ_n (s-1)^n / n!
// γ_0 = EulerGamma. γ_1 ≈ -0.0728158... .
// Algorithm: Borwein-Bradley-Crandall 2000 series acceleration.
func Stieltjes(n int) float64
// Generalized Stieltjes γ_n(a) for Hurwitz ζ.
func GeneralizedStieltjes(n int, a float64) float64
```

Pin: Stieltjes(0) ≡ EulerGamma ≡ 0.5772156649015329. Stieltjes(1) ≈ -0.0728158454836767... (OEIS A082633).

### T10 — `lfunc/zeros.go` ~200 LOC

```go
// GramPoint returns the n-th Gram point g_n: solution of θ(g_n) = nπ.
// Source: Edwards 1974 §6.5; Gram 1903.
func GramPoint(n int) float64

// ZetaZerosOnCriticalLine returns the imaginary parts of zeros of ζ(½+it) for t ∈ [tMin, tMax].
// Algorithm: bisect Z(t) sign changes between consecutive Gram points; verify via Riemann-Siegel.
func ZetaZerosOnCriticalLine(tMin, tMax float64) []float64

// ZeroCountN returns N(T) = #{ρ : 0 < Im(ρ) ≤ T, ζ(ρ) = 0} via N(T) = θ(T)/π + 1 + S(T)
// where S(T) = (1/π) arg ζ(½+iT). Heuristic for moderate T; rigorous via Turing (T14, deferred).
func ZeroCountN(T float64) int
```

Pin: first 10 Gram points and first 10 zeros of ζ should interleave (Gram's law) — though Gram's law fails for ~few % of zeros at large t. First 100 zeros against Odlyzko table. **Cross-validation pin**: ZeroCountN(t₁₀₀) = 100 where t₁₀₀ ≈ 236.524 (the 100th zero).

Recreational frontier: ship a `testdata/odlyzko_first_1000_zeros.golden` file (public-domain numerical data published by Odlyzko 1989) as a regression fixture.

### T11 — `lfunc/dedekind.go` (FRONTIER — DEFER unless slot 290 commits)

```go
// DedekindZeta returns ζ_K(s) for number field K, s ∈ ℂ.
// ζ_K(s) = Σ_{ideals a ⊆ O_K} N(a)^(-s); composes slot 290 number-field arithmetic.
// Functional equation: ζ_K(s) = |d_K|^(½-s) (Γ-factor) ζ_K(1-s).
// Class number formula at s=0: lim_{s→0} ζ_K(s) / s^(r_1+r_2-1) = -h_K R_K / w_K (Dirichlet).
func DedekindZeta(K *NumberField, s complex128) complex128
```

Refs: Cohen 2007 *Number Theory II* GTM 240 §10.5.

### T12 — `lfunc/artin.go` (FRONTIER — DEFER)

Artin L-functions L(s, ρ) for Galois representations ρ : Gal(K̄/K) → GL_n(ℂ). Direct consumer of slot 290 Galois groups + character tables. Dedekind ζ_K factors as Π over irreducible Galois reps. Frontier — Artin's holomorphy conjecture is open in general (proved for solvable by Brauer 1947, monomial by Artin).

### T13 — Modular L-functions (slot 294 T12, **NOT this slot**)

Slot 294 ships modular L-functions directly via its newform substrate, calling back into slot 295's T7 generic AFE evaluator. **This slot does NOT duplicate.**

### T14 — Turing's method (FRONTIER — DEFER)

Turing 1953 *Some calculations of the Riemann zeta function*: rigorous integer N(T) without finding all zeros, using bound `|S(T)| ≤ C log T / log log T` (de la Vallée Poussin) + careful integration of `arg ζ(½+it)` over a Gram-point interval. Modern variants by Booker 2006. Use case: rigorous Riemann-hypothesis verification "first 10^k zeros lie on critical line" — research curio, no Reality consumer.

### T15 — Automorphic L-functions, T16 — Hiary exponent-1/3 (FRONTIER — DEFER)

T15 = Langlands-program-level L-functions (GL_n automorphic representations); requires representation theory of adele groups; vastly out of scope. T16 = Hiary 2011 t^(4/13) algorithm; only matters at t ≥ 10¹⁰ where float64 fails anyway — needs Arb-style multiprecision interval arithmetic to be useful.

## Day-1 PR shape

**Singular cheapest, highest-immediate-value PR**: T0 Bernoulli (or slot-294 reuse, ~80 LOC) + T1 Riemann ζ Euler-Maclaurin (~150 LOC) + T2 special values + Apéry/Catalan (~100 LOC). **Total ~330 LOC** (or ~250 LOC if slot 294 lands first). Pure Go, MIT, zero-dep beyond `math`/`math/cmplx`. Saturates R-MUTUAL-CROSS-VALIDATION 3/3 at four pins (ζ(2)=π²/6, ζ(-1)=-1/12, ζ(3)=Apéry, functional equation). Provides immediate cross-validation substrate for `prob/info` constants and `info/mdl` regret bounds (which sometimes invoke ζ(3) and Catalan G).

**Singular highest-strategic-value follow-up**: T3 Riemann-Siegel formula + T10 Gram-point zero finder (~450 LOC). Single block that ships practical zero verification of ζ on the critical line — the canonical analytic-number-theory primitive. Composes T1 for sanity-overlap; pinned against Odlyzko's published 1000-zero golden table.

**Strategic third PR**: T7 generic L-function evaluator (~180 LOC). Single primitive that promotes Reality from "ζ library" to "L-function library", and is **the thing slot 294 T12 calls back into**. Without T7, slot 294 either ships its own (duplicates work) or skips T12 entirely.

## Cross-cutting

- **Slot 294 T12 modular L-functions** ← T7 generic AFE evaluator + T1 Riemann ζ as instance check. **Hard dependency in the direction 294 → 295 for T7; soft in the direction 295 → 294 for T0 Bernoulli (either slot can land first).**
- **Slot 292 BSD-conjecture verification** ← composes slot 292 (E) + slot 294 (f_E q-expansion) + slot 295 T7 (evaluate L(f_E, 1) and L'(f_E, 1)). Three-slot composition for end-to-end BSD numerical regression.
- **Slot 290 Dedekind ζ_K, Artin L (T11/T12)** ← composes slot 290 number-field arithmetic + slot 295 generic AFE. Frontier; defer.
- **prob package** ← Riemann ζ-zero spacings follow GUE statistics (Montgomery-Odlyzko law, 1973-1989). Cross-validation against `prob` random-matrix-theory primitives (if shipped); otherwise standalone histogram regression.
- **info / mdl package** ← Apéry ζ(3), Catalan G, π²/6 appear in entropy-bound asymptotics (e.g., NML regret tail constants). T2 ships these.
- **calculus package** ← Euler-Maclaurin formula is *also* a general numerical-summation primitive (`Σ f(n)` with Bernoulli correction). Could ship as `calculus.EulerMaclaurinSum` and have `lfunc.RiemannZeta` specialize. **Better factorization than burying inside lfunc.**
- **constants package** ← T2 adds `Apery`, `Catalan` (and possibly `Stieltjes1` …) as float64 literals. Increases constants/ scope from "physical/math constants" to include "named transcendental special values".
- **autodiff** ← T7 + autodiff yields L'(s) for free at any complex s; useful for BSD rank-1 case central derivative. Optional.
- **chaos package** ← Riemann ζ-zeros ↔ random Hermitian matrices (Berry-Keating Hilbert-Pólya conjecture). Recreational link only.
- **physics package** ← ζ(-3) = 1/120 in Casimir-effect / Stefan-Boltzmann constant derivations; ζ(2) = π²/6 in Stefan-Boltzmann black-body integral; ζ(-1) = -1/12 in bosonic-string critical-dimension proof. Marginal consumer; T2 ships the values.
- **Pistachio (downstream)** ← NO consumer. 60 FPS rendering does not need ζ. **Research-positioning, not product-load-bearing.**
- **aicore (downstream)** ← NO direct consumer.

### Recommendation: **SHIP T0+T1+T2 (~330 LOC) as day-1; T3 follow-up; defer T11-T16**

T0+T1+T2 has the best LOC/value ratio in this slot: 330 lines unblock four R-MUTUAL-CROSS-VALIDATION 3/3 pins, ship a usable zeta evaluator immediately, and provide constants (Apéry, Catalan) that other Reality packages benefit from. T3 (Riemann-Siegel) is the singular research-positioning primitive — ship as second PR. T7 generic evaluator should ship before slot 294 attempts T12. Everything else is research curio for v1.

### Moat estimate

Pure-Go MIT zero-dep zeta library = **ABSENT**. mpmath BSD-3 Python-only. PARI GPL-2. Arb LGPL. SageMath GPL-3. Mathematica proprietary. Reality could be the FIRST clean MIT pure-Go zeta — but the audience defaults to mpmath in Python; the moat is "available in Go without GPL contamination" rather than "best precision". **Modest moat; ship it for completeness and cross-validation utility, not for market disruption.**

## Sources

### Repo files (citations)
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md:312` — slot 295 line definition.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\294-new-modular-forms.md:84-97` — slot 294 T_substrate Bernoulli ownership (shared with this slot).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\294-new-modular-forms.md:256-263` — slot 294 T12 modular L-functions; the direct downstream consumer of THIS slot's T7 generic evaluator.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\292-new-elliptic-curves.md` — BSD-conjecture L-function consumer (composes 292 + 294 + 295).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\290-new-galois-theory.md` — number-field substrate for T11 Dedekind ζ_K and T12 Artin L (deferred).
- `C:\limitless\foundation\reality\constants\math.go:65-70` — existing `EulerGamma` constant (= Stieltjes γ_0; cross-references this slot's T9).
- `C:\limitless\foundation\reality\constants\math.go` — target home for `Apery`, `Catalan` constants.
- `C:\limitless\foundation\reality\sequence\token_ratio_test.go` — false-positive grep hit (Greek-letter "zeta" in test fixture, NOT a zeta function).

### Mathematical references (cited above per tier)
- Edwards 1974, *Riemann's Zeta Function*, Academic Press / Dover reprint 2001 — canonical textbook for ζ, Euler-Maclaurin, Riemann-Siegel, Gram points.
- Cohen 2007, *Number Theory Vol. II: Analytic and Modern Tools*, GTM 240 — algorithms ch. 10.
- Apostol 1976, *Introduction to Analytic Number Theory*, GTM (reprinted 1998) — Hurwitz ζ.
- Erdélyi 1953, *Higher Transcendental Functions* vol. I — Lerch transcendent.
- Lewin 1981, *Polylogarithms and Associated Functions*, North-Holland.
- Iwaniec-Kowalski 2004, *Analytic Number Theory*, AMS Colloquium Pubs. 53 — comprehensive L-function reference (615pp).
- Borwein-Bradley-Crandall 2000, *Computational strategies for the Riemann zeta function*, J. Comp. Appl. Math. 121:247-296 — Stieltjes constants algorithm; series acceleration.
- Berry-Keating 1992, *A new asymptotic representation for ζ(½+it) and quantum spectral determinants*, Proc. Roy. Soc. A — refined Riemann-Siegel remainder.
- Berry 1995, *The Riemann-Siegel formula for the zeta function: high orders and remainders*, Proc. Roy. Soc. A 450 — explicit remainder bounds.
- Hiary 2011, *Fast methods to compute the Riemann zeta function*, Annals of Math. 174(2):891-946 — exponent-1/3 algorithm; T16 deferred.
- Odlyzko 1989, *Distribution of zeros of the Riemann zeta function: conjectures and computations*, manuscript — first 10⁵ zeros tabulation; cross-validation source.
- Odlyzko 1992-2001, "10²⁰-th zero" / "10²²-nd zero" computational records — dtc.umn.edu/~odlyzko/zeta_tables/ — public-domain golden-data source for T10 zero regression.
- Rubinstein 2005, *Computational methods and experiments in analytic number theory*, MSRI vol. 44 — approximate functional equation.
- Booker 2006, *Artin's conjecture, Turing's method, and the Riemann hypothesis*, Experimental Math. 15(4):385-407 — Turing T14 modern incarnation.
- Dokchitser 2004, *Computing special values of motivic L-functions*, Experimental Math. 13(2):137-149.
- Turing 1953, *Some calculations of the Riemann zeta function*, Proc. London Math. Soc. (3) 3:99-117 — original Turing's method.
- Gram 1903, *Sur les zéros de la fonction ζ(s)*, Acta Math. 27:289-304 — Gram points.
- Montgomery 1973, *The pair correlation of zeros of the zeta function*, Proc. Sympos. Pure Math. 24 — GUE statistics conjecture.
- Brauer 1947, *On Artin's L-series with general group characters*, Annals of Math. 48:502-514 — Artin L for solvable Galois.
- OEIS A002117 (Apéry constant), A006752 (Catalan), A082633 (γ_1 Stieltjes), A027641/A027642 (Bernoulli num/denom).

### Library / license sources
- mpmath: https://mpmath.org/doc/current/functions/zeta.html (BSD-3) — algorithm reference (Borwein / Riemann-Siegel / Euler-Maclaurin auto-selection).
- PARI/GP `lfun`: https://pari.math.u-bordeaux.fr/, GPL-2 — full L-function suite, **license-incompatible** with Reality MIT.
- Arb: https://arblib.org/, LGPL — rigorous-error-bound arbitrary-precision; **license-contaminating in static-link**.
- LMFDB: https://www.lmfdb.org/, MIT data + GPL-derived engines — golden cross-validation source for T10 (zero regression).
- SageMath: https://www.sagemath.org/, GPL-3 — full via PARI; license-incompatible.

### Frontier deferred
- T11 Dedekind ζ_K (composes slot 290; defer).
- T12 Artin L (composes slot 290 + reps; defer).
- T13 Modular L (slot 294 owns).
- T14 Turing's method for rigorous N(T).
- T15 Automorphic L-functions (Langlands).
- T16 Hiary exponent-1/3 algorithm (multiprecision-only; not needed at float64).
- p-adic L-functions (Kubota-Leopoldt 1964; Kato-Mazur-Wiles).
