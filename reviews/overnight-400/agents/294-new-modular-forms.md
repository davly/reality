# 294 — new-modular-forms (Eisenstein / Theta / q-Expansions / Hecke / Modular Polynomial Φ_ℓ)

## Headline
Reality v0.10.0 ships ZERO computational modular-forms surface (no `Eisenstein`, `Theta`, `qExpansion`, `Eta`, `Dedekind`, `JInvariant`, `Hecke`, `AtkinLehner`, `NewForm`, `Eigenform`, `Sturm`, `ManinSymbol`, `EichlerSelberg` callable hits in any source file); the slot is the most theoretically sophisticated in Block C with the narrowest direct consumer set, and the singular load-bearing primitive is **T11 modular polynomial Φ_ℓ(X, Y) generator (~380 LOC) — the precomputed-table source slot 292 SEA Elkies-branch hard-depends on for ℓ ≤ 100**; the singular cheapest day-1 PR is **T0+T1+T3 (formal q-series + Eisenstein E_4/E_6/E_8 via σ_{k-1}(n) + j-invariant series via E_4³/Δ, ~360 LOC)** which unblocks T11 and provides slot-292 a deterministic golden-file substrate for the modular polynomial table.

## Findings

### State at HEAD (v0.10.0, 2026-05-09)

Repo-wide grep on `ModularForm|Eisenstein|ThetaFunction|qExpansion|EtaFunction|DedekindSum|ModularDiscriminant|j.*Invariant|JInvariant|HeckeOperator|AtkinLehner|NewForm|Eigenform|CongruenceSubgroup|ManinSymbol|SturmBound|EichlerSelberg`: **zero callable hits in any `*.go` source file**. Every match is in `reviews/overnight-400/` markdown (slots 290, 292, 213, 211, etc.).

| Surface needed by this slot | Path | Status |
|---|---|---|
| Bernoulli numbers B_k (rational) | absent | **REQUIRED** for E_k constant term `1 - 2k/B_k · Σ σ_{k-1}(n)q^n` |
| Divisor sum σ_k(n) = Σ_{d|n} d^k | absent | **REQUIRED** for E_k Fourier coefficients |
| Euler totient φ(n), Möbius μ(n) | absent | Required for level-N dimensions, oldform/newform decomposition |
| Partition number p(n) | `combinatorics/counting.go:268 IntegerPartitions` | float64; OK for small n; need bigint for q-series mult |
| Stirling, Bell | `combinatorics/counting.go:159,194,234` | float64; orthogonal use |
| `BinomialCoeff` | `combinatorics/counting.go:51` | log-gamma float64; need exact bigint for q-series arith |
| `*big.Int` / `*big.Rat` polynomial type | absent | **REQUIRED** — q-expansion = formal power series in q with rational coefficients |
| j-invariant of an algebraic curve (point form) | absent (slot 292 T1) | Different object: this slot's j(τ) is the q-series `1/q + 744 + 196884q + …`; slot 292's j(E) = 1728·4a³/(4a³+27b²) is the algebraic-geometric *value*. **Same constant 1728 appears; the bridge is `j(E) = j(τ_E)` for τ_E in the upper half-plane.** |
| Eta function η(τ) | absent | η(τ) = q^(1/24) Π(1−qⁿ); 24-th-power gives Δ |
| Modular discriminant Δ(τ) = (2π)^12 η^24 | absent | Cusp form of weight 12, level 1; q-coefficients are the Ramanujan τ-function |

### Cross-slot orientation

| Adjacent slot | Overlap | Resolution |
|---|---|---|
| **292-new-elliptic-curves T5 modular polynomial Φ_ℓ database** | `reviews/agents/292-new-elliptic-curves.md:117-121` *requires* a precomputed Φ_ℓ table for ℓ ∈ {2,3,5,7,11,…,97} as `data/modular_phi_l.go` golden file; **does not say where the generator lives**. | **THIS SLOT (294) owns the generator**. T11 below. Slot 292 owns the *consumer* (SEA Elkies-prime branch + Atkin-Morain CM). |
| **292 T8 Hilbert class polynomial H_D(X)** | Computed via complex-analytic floating-point (Watkins 2004) or CRT (Sutherland 2011); roots are CM j-invariants. | Composes T1 (Eisenstein) + T2 (Δ) + T3 (j-invariant series) of THIS slot, evaluated at CM points τ ∈ Q(√−|D|) ∩ H. Slot 292 may import slot 294's `JInvariant(tau complex128) complex128` evaluator. |
| **291-new-modular-arithmetic** | Tonelli-Shanks, Montgomery, Garner-CRT. | Slot 294's T11 modular-polynomial generator uses CRT-of-many-primes (Sutherland's algorithm) to lift Φ_ℓ from F_p back to Z. Direct consumer of slot 291's Garner-CRT. |
| **290-new-galois-theory** | Number-field arithmetic, polynomial factoring over F_p. | Slot 290's `PolyFp` and `PolyZ` (T0/T0' of slot 290) are the substrate for Φ_ℓ as a bivariate polynomial in Z[X, Y]. Slot 290 also provides Berlekamp factoring used to factor Φ_ℓ(X, j) over F_p in the SEA Elkies test. |
| **213-new-isogeny** | Velu's formulae, modular curves X_0(N). | Disjoint algebra at the level here (slot 213 owns the *EC group-action* side; slot 294 owns the *modular-form* side). The bridge is Φ_ℓ — both slots reference it; slot 294 ships, slot 213 consumes. |
| **constants** package | Mathematical constants. | Slot 294 *should add* Bernoulli numbers B_0, B_2, B_4, …, B_30 as exact `*big.Rat` constants to `constants/math.go` or new `constants/bernoulli.go`. ~32 entries; CODATA-style frozen literal data. |
| **combinatorics** package | Already has `IntegerPartitions`, `BinomialCoeff`, Stirling, Bell. | Slot 294 *should add* `SigmaDivisor(k, n int) *big.Int`, `EulerTotient(n int) int`, `Mobius(n int) int` to `combinatorics/` (or new `numbertheory/` package — recommend folding into combinatorics to keep package count at 22). |
| **prob** package | Sums of squares ≡ θ-function evaluations (`r_k(n) = #{(x_1,…,x_k) ∈ Z^k : Σx_i² = n}`). | Slot 294 T4 `Theta3(q)^k` Fourier coefficients ≡ `r_k(n)`; orthogonal cross-validation pin against slot prob's combinatorial enumeration of representation counts. |
| **autodiff / infogeo** | Complex differentiation, Wirtinger calculus. | Slot 294 may consume autodiff's complex differentiation to compute q-derivatives `qd/dq` (Ramanujan-Serre derivative). Optional. |
| **chaos** package | Complex dynamics. | The j-invariant J: H/Γ → C is the canonical example of a Hauptmodul; Mandelbrot/Julia sets *are* Riemann-surface dynamics. Recreational link only. |

### Distinct-from-slot-292 disambiguation

Slot 292 owns:
- `EllipticCurve` data type, `Point` arithmetic, j-invariant of *a specific curve* `j(E) = 1728·4a³/(4a³+27b²) mod p` (a point in F_p).
- Schoof / SEA / CM curve construction; consumes the modular polynomial Φ_ℓ as **fixed precomputed data**.

Slot 294 owns:
- Formal q-series `qSeries{Coeffs []*big.Rat; q-precision N}`. Object lives in Q[[q]], not in F_p.
- Modular forms M_k(Γ_0(N)) = vector spaces of holomorphic functions on H satisfying f((aτ+b)/(cτ+d)) = (cτ+d)^k f(τ).
- Eisenstein series E_k, modular discriminant Δ, j-invariant *as a q-series* j(τ) = 1/q + 744 + 196884q + …, eta η(τ) and Dedekind sum.
- Hecke operators T_p as linear operators on q-series, Atkin-Lehner involutions w_d, newform decomposition.
- **Modular polynomial Φ_ℓ(X, Y) GENERATOR**: produces the bivariate polynomial in Z[X, Y] from Eisenstein/Delta/j-series via interpolation across X_0(ℓ) → X(1)×X(1) double cover; output feeds slot 292.

Boundary: a *q-coefficient of Δ* is slot-294 (it's the Ramanujan τ-function arithmetic invariant of the modular form). A *coefficient of Φ_ℓ used to compute #E(F_p) via Elkies* is slot-292 *consumption* of the slot-294 *generation*.

### Does math/big or stdlib cover this?

`math/big` ships `*big.Rat` and `*big.Int` — sufficient substrate. **No stdlib q-series, no Eisenstein, no Hecke**. `golang.org/x/...` has nothing modular.

| Library | License | Coverage | Reality moat |
|---|---|---|---|
| PARI/GP `mfinit` / `mfeisenstein` / `mfheckedata` | GPL-2 | Full M_k(Γ_0(N)), Hecke, newforms, L-functions | GPL — not MIT-compatible. |
| SageMath `ModularForms(Γ_0(N), k)` | GPL-3 | Full; via PARI under the hood | GPL — not MIT-compatible. |
| Magma `ModularForms` | proprietary | full | Closed source. |
| LMFDB | MIT (data layer) | precomputed L-functions / newforms / Hecke eigenvalues | MIT BUT compute engine = PARI/Sage; the data file is MIT but *generated by* GPL software → derivative-work questions. Reality could regenerate from scratch. |
| `golang.org/x/...` | various BSD-3 | NONE | — |
| Mathematica `EisensteinE[k, τ]` | proprietary | Built-in; closed | — |
| `gmpy2` / Arb | LGPL / Apache-2 | High-precision; not modular-form-specific | — |

**Reality's positioning**: the FIRST MIT pure-Go zero-dep deterministic-golden-file-validated computational-modular-forms framework. **Audience is narrow** — academic computational number theorists, BSD-conjecture researchers, monstrous-moonshine enthusiasts, mathematical physicists doing string-theory partition functions, and **slot 292 SEA Elkies-branch (the only direct in-Reality consumer)**. Pistachio/aicore do not consume modular forms.

### Practically infinite frontier — commit to subset with clear consumer

Computational modular forms is unbounded (Cohen-Stromberg 2017 is 700 pages and only covers the basics). Recommended subset:

- **MUST ship** (slot 292 SEA Elkies + CM hard-depend): T0 q-series, T1 E_k, T2 Δ + η, T3 j-invariant series, T11 Φ_ℓ generator + golden table for ℓ ≤ 100.
- **SHOULD ship** (Reality "universal truth" positioning + textbook-anchored): T4 θ_2/θ_3/θ_4 (Jacobi theta), T5 Hecke T_p, T7 dim formulas M_k / S_k for Γ_0(N), T8 Sturm bound.
- **NICE-to-ship** (research utility, no current consumer): T6 Manin symbols (1972), T9 Hecke eigenform basis, T10 Atkin-Lehner involutions, T12 L-function via approximate functional equation, T13 newform/oldform decomposition.
- **DEFER** (frontier, no Reality consumer): T14 Eichler-Selberg trace formula, harmonic Maass forms (Bringmann-Folsom-Ono-Rolen 2017), quantum modular forms (Zagier 2010), monstrous moonshine character tables (Conway-Norton 1979).

## Concrete recommendations

Tier numbering: T0 = q-series substrate; T1-T3 = level-1 forms (E_k, Δ, j); T4 = theta; T5 = Hecke; T6 = Manin symbols; T7-T8 = dimensions + Sturm; T9-T10 = eigenforms + Atkin-Lehner; T11 = Φ_ℓ KEYSTONE; T12-T13 = L-functions + newforms; T14 = Eichler-Selberg frontier.

### T_substrate — `constants/bernoulli.go` ~80 LOC + `combinatorics/numbertheory.go` ~120 LOC (PREREQUISITE)

Add to `constants/`:
```go
// BernoulliRat returns B_k as *big.Rat for even k ∈ [0, 30]; B_k = 0 for odd k > 1.
// Sources: Knuth TAOCP vol. 1 §1.2.11.2; OEIS A027641/A027642.
func BernoulliRat(k int) *big.Rat
```
Frozen literal table (B_0 = 1, B_2 = 1/6, B_4 = -1/30, …, B_30 = 8615841276005/14322); 16 even entries. Add to `combinatorics/numbertheory.go`:
```go
func SigmaDivisor(k, n int) *big.Int   // σ_k(n) = Σ_{d|n} d^k
func EulerTotient(n int) int           // φ(n)
func Mobius(n int) int                 // μ(n) ∈ {-1, 0, 1}
```

### T0 — `modular/qseries.go` ~180 LOC — DAY-1 KEYSTONE
```go
type QSeries struct{ Coeffs []*big.Rat; Prec int }   // Σ a_n q^n, n ∈ [0, Prec)
func NewQSeries(prec int) *QSeries
func (s *QSeries) Add(t *QSeries) *QSeries
func (s *QSeries) Mul(t *QSeries) *QSeries           // O(Prec²); FFT later if hot
func (s *QSeries) Pow(k int) *QSeries                // binary exponentiation
func (s *QSeries) Inv() *QSeries                     // requires a_0 ≠ 0
func (s *QSeries) Eval(q complex128) complex128      // truncated complex evaluation
```
Foundation for every higher tier. Pin against trivial identity `(1-q)·(1-q)^{-1} = 1` to Prec=100; expand `1/(1-q)^k` and check coefficients are `BinomialCoeff(n+k-1, k-1)` for k=1..10, n=0..50.

### T1 — `modular/eisenstein.go` ~120 LOC — DAY-1 KEYSTONE
```go
// E_k(q) = 1 - (2k/B_k) Σ_{n≥1} σ_{k-1}(n) q^n,  k ∈ {4, 6, 8, 10, 12, 14}.
func Eisenstein(k, prec int) *QSeries
```
- E_4(q) = 1 + 240 Σ σ_3(n) q^n
- E_6(q) = 1 − 504 Σ σ_5(n) q^n
- E_8 = E_4², E_10 = E_4·E_6, E_14 = E_4²·E_6 (all *holomorphic-modular-form identities*, not just numerical equalities).

Pin: compute E_8 directly from B_8=−1/30 → coefficient `1 + 480 Σ σ_7(n) q^n`, AND independently from `Eisenstein(4).Mul(Eisenstein(4))`; assert byte-identical for first 100 coefficients. Same for E_10 = E_4·E_6 vs direct, E_14 = E_4²·E_6 vs direct. **Three independent computational paths agreeing on q^n coefficients for n ≤ 100 saturates R-MUTUAL-CROSS-VALIDATION 3/3.**

Refs: Serre 1973 *A Course in Arithmetic* §VII.4; Diamond-Shurman 2005 *A First Course in Modular Forms* GTM 228 §1.1.

### T2 — `modular/delta_eta.go` ~140 LOC
```go
// Modular discriminant Δ(τ) = q Π_{n≥1} (1 - q^n)^24, weight 12 cusp form on Γ(1).
func Delta(prec int) *QSeries
// Dedekind eta η(τ) = q^(1/24) Π_{n≥1} (1 - q^n); ship as eta24(q) = η^24(τ)/q = Π(1-q^n)^24.
func Eta24(prec int) *QSeries
// DedekindSum s(h, k) = Σ_{i=1}^{k-1} (i/k)((hi/k)) where ((x)) = x - ⌊x⌋ - 1/2 if x ∉ Z else 0.
func DedekindSum(h, k int) *big.Rat
```
Compute Δ via THREE independent formulas:
1. **Eta product**: Δ = q · Π(1−q^n)^24 (direct).
2. **Jacobi triple product** giving Δ = q Π(1−q^n)^24 = (Σ (-1)^n (2n+1) q^{n(n+1)/2})^8 by Jacobi's identity for η^3.
3. **Eisenstein identity**: 1728·Δ = E_4³ − E_6².

All three must yield the same `q^n` coefficient (= Ramanujan τ(n)) for n ≤ 100. Saturates **R-MUTUAL-CROSS-VALIDATION 3/3** at the keystone-cusp-form level.

Refs: Hardy-Wright 1979 *An Introduction to the Theory of Numbers* §19.9; Apostol 1990 *Modular Functions and Dirichlet Series in Number Theory* GTM 41 ch. 1; Knuth-Buckholtz 1967 (Dedekind sums).

### T3 — `modular/jinvariant.go` ~80 LOC — DAY-1 KEYSTONE
```go
// j(τ) = E_4(τ)^3 / Δ(τ) = 1/q + 744 + 196884 q + 21493760 q^2 + 864299970 q^3 + ...
// (Note: returned as Laurent series with leading 1/q term — use NegPrec field.)
func JInvariant(prec int) *QLaurent  // Laurent series, principal part 1/q
// JInvariantC(tau complex128) complex128 — analytic evaluation at τ ∈ H
func JInvariantC(tau complex128, prec int) complex128
```
Three pins:
1. j(i) = 1728 (CM by Z[i], discriminant −4) — analytic numerical regression to 10⁻¹².
2. j(ω) = 0 (CM by Z[ω], discriminant −3) — analytic regression.
3. q-coefficients of j match OEIS A000521 = `[1, 744, 196884, 21493760, ...]` byte-identical for first 50 terms (these are also the **monstrous moonshine McKay-Thompson series 1A**, the dimensions of irreducible Monster representations after Conway-Norton normalization).

Saturates **R-MUTUAL-CROSS-VALIDATION 3/3** (analytic-at-i, analytic-at-ω, q-series-vs-OEIS).

Refs: Lang 1987 *Elliptic Functions* GTM 112 §V; Conway-Norton 1979 *Monstrous Moonshine* Bull. LMS 11; OEIS A000521.

### T4 — `modular/theta.go` ~160 LOC
```go
// Jacobi θ functions as q-series; q = e^(πiτ).
func Theta2(prec int) *QSeries  // 2 q^(1/4) Σ q^(n(n+1)) — half-integer indexed
func Theta3(prec int) *QSeries  // 1 + 2 Σ q^(n²)
func Theta4(prec int) *QSeries  // 1 + 2 Σ (-1)^n q^(n²)
// Identity: θ_2^4 + θ_4^4 = θ_3^4 (Jacobi 1829)
```
Cross-validation pin: θ_3(q)^k coefficient at q^n equals `r_k(n) = #{(x_1,…,x_k) ∈ Z^k : Σ x_i² = n}` — the **representation-as-sum-of-squares** counting function. For k=2 (Jacobi 1829: r_2(n) = 4(d_1(n) − d_3(n)) where d_j counts divisors ≡ j mod 4); k=4 (Jacobi: r_4(n) = 8 σ(n) if n odd, 24σ(odd part) if n even); k=8 (Jacobi: r_8(n) = 16 σ_3*(n) where * = signed-sum). Three independent closed forms (k=2, k=4, k=8) cross-checked against direct combinatorial enumeration in `prob/sum_of_squares` ⇒ **R-MUTUAL-CROSS-VALIDATION 3/3**.

Refs: Whittaker-Watson 1927 *A Course of Modern Analysis* §21; Mumford 1983-1984 *Tata Lectures on Theta* I-III.

### T5 — `modular/hecke.go` ~140 LOC
```go
// Hecke operator T_p acting on M_k(Γ_0(1)) at level 1, weight k, p prime.
// (T_p f)(q) has q^n coefficient: a_{pn} + p^(k-1) a_{n/p}  (where a_{n/p} = 0 if p ∤ n)
func HeckeTpLevel1(f *QSeries, p, k int) *QSeries
// Generic T_n via multiplicative recurrence:
//   T_{mn} = T_m T_n if gcd(m,n) = 1
//   T_{p^{r+1}} = T_p T_{p^r} - p^(k-1) T_{p^{r-1}}
func HeckeTnLevel1(f *QSeries, n, k int) *QSeries
```
Verify: Δ is a Hecke eigenform with `T_p Δ = τ(p) Δ` for every prime p (Mordell 1917); Ramanujan's conjecture |τ(p)| ≤ 2 p^(11/2) (Deligne 1974) gives a numerical sanity check. Pin: τ(2)=−24, τ(3)=252, τ(5)=4830, τ(7)=−16744, τ(11)=534612 (OEIS A000594) — three primes × independent T_p computation × eigenvalue extraction = **R-MUTUAL-CROSS-VALIDATION 3/3**.

Refs: Hecke 1937 *Über Modulfunktionen und die Dirichletschen Reihen mit Eulerscher Produktentwicklung* I-II; Atkin-Lehner 1970 *Hecke operators on Γ_0(m)* Math. Ann. 185; Mordell 1917 *On Mr. Ramanujan's empirical expansions of modular functions* Proc. Camb. Phil. Soc. 19.

### T6 — `modular/manin_symbol.go` ~220 LOC
Manin symbols (Manin 1972) — combinatorial model for H_1(X_0(N), cusps; Z) = M_2(Γ_0(N))-dual basis. Each modular symbol {α, β} for α, β ∈ Q ∪ {∞} represented as `(c : d) ∈ P^1(Z/N)`. Defines the Hecke action combinatorially without any complex analysis. Stein 2007 (*Modular Forms: A Computational Approach* §8) is the canonical algorithmic reference.

```go
type ManinSymbol struct{ C, D int; N int }  // (c:d) in P^1(Z/N)
func ManinSymbolBasis(N int) []ManinSymbol  // generators of M_2(Γ_0(N))
func ManinHeckeMatrix(N, p int) [][]int     // matrix of T_p on Manin basis, integer entries
```
Composes T7 dim formulas (matrix size = dim M_2(Γ_0(N))). Used by T9 Hecke eigenform diagonalization without any q-series convergence concerns.

Refs: Manin 1972 *Parabolic points and zeta functions of modular curves* Izv. Akad. Nauk SSSR 36; Cremona 1997 *Algorithms for Modular Elliptic Curves* 2nd ed., CUP — chapter 2 ships the practical algorithm.

### T7 — `modular/dimensions.go` ~120 LOC
```go
// dim M_k(Γ_0(N)) and dim S_k(Γ_0(N)) closed-form (Cohen-Oesterlé 1977).
func DimMk(k, N int) int
func DimSk(k, N int) int
func DimEisensteinSpace(k, N int) int  // = dim M_k - dim S_k = number of cusps - 1 (for k > 2 even)
```
Pin against table in Diamond-Shurman §3.5 / Stein 2007 ch. 6: dim M_12(Γ_0(1)) = 2, dim S_12(Γ_0(1)) = 1 (Δ); dim M_2(Γ_0(11)) = 2, dim S_2(Γ_0(11)) = 1 (the modular form attached to the elliptic curve `y² + y = x³ - x²`, conductor 11, the *first non-CM elliptic curve*).

Refs: Cohen-Oesterlé 1977 *Dimensions des espaces de formes modulaires* in *Modular Functions of One Variable VI* LNM 627.

### T8 — `modular/sturm.go` ~40 LOC
```go
// Sturm 1987 bound: two modular forms in M_k(Γ_0(N)) with q-coefficients agreeing
// up to index ⌊k·[SL_2(Z):Γ_0(N)] / 12⌋ are equal.
// [SL_2(Z):Γ_0(N)] = N · Π_{p|N} (1 + 1/p).
func SturmBound(k, N int) int
```
40 LOC including the index computation. Tiny but load-bearing — every numerical-equality check in computational modular forms uses it ("compute first SturmBound(k, N) coefficients; if they agree, the forms are equal").

Refs: Sturm 1987 *On the congruence of modular forms* in *Number Theory (NY 1984-85)* LNM 1240.

### T9 — `modular/eigenforms.go` ~180 LOC
Diagonalize T_p simultaneously on S_k(Γ_0(N)) (commuting operators ⇒ joint eigenbasis). Output: list of newforms f_i with q-expansions f_i = q + a_2(f_i) q² + a_3(f_i) q³ + …, where a_p(f_i) is the T_p eigenvalue. Composes T6 (Manin symbol matrix of T_p) + T7 (dimension) + T8 (Sturm bound to certify equality of computed q-series with true eigenform).

Pin: weight-12 level-1 yields exactly Δ (one 1-dim eigenspace). Weight-2 level-11 yields the conductor-11 elliptic-curve newform with `a_p = p + 1 - #E(F_p)` for E: y² + y = x³ - x²; verify by counting points on E directly via slot 292 T0 + T6 ⇒ three-way pin (eigenform a_p, slot-292 Schoof #E, brute-force enumeration over F_p for p ≤ 100). **R-MUTUAL-CROSS-VALIDATION 3/3 + cross-slot consumer of slot 292.**

Refs: Atkin-Lehner-Li 1970-1975 *newform theory*; Stein 2007 *Modular Forms: A Computational Approach* AMS GSM 79 — algorithmic reference.

### T10 — `modular/atkin_lehner.go` ~100 LOC
Atkin-Lehner involutions w_d for d ‖ N (d exactly divides N). Each w_d acts on S_k(Γ_0(N)) with eigenvalues ±1. Newforms decompose into ±-eigenspaces. Used by L-function functional equation sign.

Refs: Atkin-Lehner 1970 (same paper as T5 ref).

### T11 — `modular/modular_polynomial.go` ~380 LOC + ~200-800 KB precomputed `data/modular_phi_l.go` — **HIGHEST-LEVERAGE PRIMITIVE FOR THIS SLOT**

Modular polynomial Φ_ℓ(X, Y) ∈ Z[X, Y] characterized by Φ_ℓ(j(τ), j(ℓτ)) = 0 for all τ ∈ H. Symmetric (Φ_ℓ(X,Y) = Φ_ℓ(Y,X)) and of bidegree (ℓ+1, ℓ+1) for prime ℓ.

Two-stage algorithm (Sutherland 2011):
1. **Per-prime stage**: for many small primes p (p ∤ ℓ, p ≡ 1 mod 4ℓ for the "good" reductions), compute Φ_ℓ mod p via supersingular j-invariant computation + isogeny graph traversal — yields `Φ_ℓ mod p ∈ F_p[X, Y]`.
2. **CRT lift**: aggregate over enough primes (size bound: log ‖Φ_ℓ‖_∞ ≈ 6ℓ log ℓ + O(ℓ) bits, Cohen 1984) and Garner-CRT (slot 291) back to Z.

```go
type ModularPoly struct{ L int; Coeffs map[[2]int]*big.Int }  // sparse bivariate
func ModularPhi(L int) *ModularPoly                            // computed lazily, cached
func (m *ModularPoly) EvalAtY(j *big.Int, p *big.Int) *PolyFp  // returns Φ_L(X, j) mod p
```

Ship a precomputed `data/modular_phi_l.go` golden table for ℓ ∈ {2, 3, 5, 7, 11, 13, ..., 97} (25 primes ≤ 100) generated once at build time (or shipped as a `.golden` file under `testdata/`) — total size ~5-10 MB compressed. Slot 292 SEA imports this table directly; slot 213 isogeny may also.

Pin: Φ_2(X, Y) = X³ + Y³ - X²Y² + 1488(X²Y + XY²) - 162000(X² + Y²) + 40773375 XY + 8748000000(X + Y) - 157464000000000 (Mahler 1976 closed form, byte-identical regression). Φ_3 known closed form (Smith 1879). For ℓ ∈ {2, 3} the table is pinned analytically; for ℓ ∈ {5, …, 97} the per-prime CRT yields three independent witnesses (CRT lift, evaluation at random j_0 = j(τ_0) for τ_0 = (i+1)/2 with cross-check Φ_ℓ(j(τ), j(ℓτ)) ≈ 0 to 10⁻¹⁰, and pairwise table cross-validation across slot 213 isogeny graph) ⇒ **R-MUTUAL-CROSS-VALIDATION 3/3** at the keystone-table level.

**What it unblocks**:
- **Slot 292 T6 SEA Elkies-prime branch** — without Φ_ℓ table, SEA cannot detect whether ℓ is Elkies (Φ_ℓ(X, j(E)) splits) or Atkin (does not split); falling back to Schoof's O(ℓ²) is a 10-100× slowdown on cryptographic-size primes.
- **Slot 292 T9 Atkin-Morain CM curve construction** — Hilbert class polynomial computation can be done either by complex-analytic floating-point evaluation of j(τ_D) at CM points τ_D (uses T3 of slot 294) or by Sutherland's CRT method which itself uses Φ_ℓ for isogeny-volcano traversal.
- **Mathematical research consumers**: BSD conjecture L-function computation, Heegner point construction, Birch-Stevens L'(E, 1) via Gross-Zagier formula.

Refs: Sutherland 2011 *Computing Hilbert class polynomials with the Chinese Remainder Theorem* Math. Comp. 80(273):501-538; Bröker-Lauter-Sutherland 2012 *Modular polynomials via isogeny volcanoes* Math. Comp. 81(278):1201-1231; Atkin 1988 PARI/GP `polmodular` documentation; Mahler 1976 *On the modular function and its Mahler-Zassenhaus polynomials*.

### T12 — `modular/lfunction.go` ~200 LOC
L-function L(f, s) = Σ a_n(f) / n^s of a newform f, analytic continuation via the **approximate functional equation** (Booker 2006 + Rubinstein 2005):

L(f, s) ≈ Σ_{n ≤ X} a_n n^{-s} G_+(n, s) + ε · N^{-s+k/2} Σ_{n ≤ Y} a_n n^{s-k} G_-(n, s)

where G_± are incomplete-Gamma cutoff functions, ε = ±1 the Atkin-Lehner sign (T10), N the level. Computes L-values numerically for BSD verification.

Refs: Rubinstein 2005 *Computational methods and experiments in analytic number theory* in MSRI vol. 44; Booker 2006 *Artin's conjecture, Turing's method, and the Riemann hypothesis*.

### T13 — `modular/newform_decomposition.go` ~140 LOC
Atkin-Lehner-Li 1970-1975 newform/oldform decomposition: S_k(Γ_0(N)) = ⊕_{M | N} ⊕_{d | (N/M)} S_k^new(Γ_0(M))[d], where the inner sum is over "shifts" by `f(τ) ↦ f(dτ)`. Newform basis is the Hecke eigenform basis stripped of these oldform images.

Refs: Atkin-Lehner 1970 (T5 ref); Li 1975 *Newforms and functional equations* Math. Ann. 212.

### T14 — `modular/eichler_selberg.go` (FRONTIER — DEFER)
Eichler-Selberg trace formula: closed-form expression for trace(T_p | S_k(Γ_0(N))) as a sum over conjugacy classes of elliptic / hyperbolic / parabolic elements. Bypasses Manin-symbol diagonalization for *traces* (but not eigenvalues). Frontier; defer until BSD-conjecture researcher shows up.

Refs: Eichler 1956 *Modular correspondences and their representations* J. Indian Math. Soc. 20; Selberg 1956 *Harmonic analysis and discontinuous groups in weakly symmetric Riemannian spaces with applications to Dirichlet series* J. Indian Math. Soc. 20.

## Day-1 PR shape

**Singular cheapest, highest-immediate-value PR**: T_substrate (Bernoulli + σ_k + φ + μ, ~200 LOC) + T0 (q-series, ~180 LOC) + T1 (Eisenstein E_4/E_6/E_8, ~120 LOC) + T2 (Δ + η, ~140 LOC) + T3 (j-invariant series, ~80 LOC). **Total ~720 LOC**. Pure-Go, MIT, zero-dep beyond `math/big`. Ships the level-1 modular forms machinery up to j-invariant. Saturates four R-MUTUAL-CROSS-VALIDATION 3/3 pins (E_8 = E_4², Δ via three formulas, j(i)=1728 / j(ω)=0 / OEIS, monstrous moonshine constants).

**Singular highest-strategic-value follow-up**: T11 modular polynomial Φ_ℓ generator + golden table for ℓ ≤ 100 (~380 LOC + ~5 MB data). Single primitive that unblocks slot 292 SEA Elkies-branch and slot 213 isogeny graph. Composes T0-T3 + slot 290 (PolyFp / Berlekamp factoring) + slot 291 (Garner CRT). **Without T11, slot 292 SEA falls back to O(ℓ²) Schoof — substantial cryptographic-curve-generation slowdown.**

## Cross-cutting

- **Slot 292 SEA Elkies-prime branch (T5/T6)** ← T11 Φ_ℓ table — **HARD DEPENDENCY**. Slot 292's `reviews/.../292.md:117-121` explicitly enumerates "ship a precomputed table for ℓ ∈ {2, 3, 5, 7, ..., 97} as `data/modular_phi_l.go`" with no source. **THIS SLOT IS THE SOURCE.**
- **Slot 292 Atkin-Morain CM curve construction (T9)** ← T3 j-invariant evaluator at CM points τ_D, used to compute Hilbert class polynomial roots; alternative is Sutherland CRT method which itself recursively uses Φ_ℓ.
- **Slot 213 isogeny graph algorithms** ← T11 Φ_ℓ table for isogeny-volcano traversal.
- **Slot 290 Galois theory** ← bidirectional. Slot 290's `PolyFp` and Berlekamp factoring is consumed by T11 (factor Φ_ℓ(X, j(E)) over F_p to detect Elkies); slot 290's number-field arithmetic is consumed by T13 newform Hecke-field computation.
- **Slot 291 modular arithmetic** ← T11 Sutherland CRT-lift uses slot 291's Garner-CRT.
- **Slot 213 isogenies** ← shares T11 modular polynomial with slot 292; slot 213 may also consume T4 theta as a *modular-form generator* for Magma-style isogeny computation.
- **prob/sum-of-squares** ← T4 theta function powers ≡ representation-as-sum-of-squares counting (Jacobi 1829 / Hurwitz 1898) ⇒ orthogonal cross-validation pin against any direct combinatorial enumeration.
- **constants package** ← This slot adds Bernoulli numbers as `*big.Rat` literal data, increasing constants/ scope from "physical/math constants" to include "named integer/rational sequences" (B_k, possibly later τ(n), c(n) Mathieu-moonshine, etc.).
- **combinatorics package** ← This slot adds σ_k(n), φ(n), μ(n) (basic multiplicative number-theoretic functions); a natural augmentation to the existing Stirling/Bell/Partition trio.
- **Pistachio (downstream)** ← NO consumer. Pistachio renders 60 FPS; modular forms are research math. **This slot is research-positioning, NOT product-load-bearing.**
- **aicore (downstream)** ← NO direct consumer.
- **String-theory / mathematical physics audience** ← Theta functions are partition functions of bosonic strings on tori; j-invariant appears in monstrous moonshine (Monster sporadic group representation theory). Audience: theoretical-physics researchers and pure mathematicians. Reality positioning, not product.

### Recommendation: **DEFER unless slot 292 commits to SEA Elkies-branch**

The honest cost-benefit: ~3,000 LOC across T0-T13 (everything except T14 frontier) serves *one* in-Reality consumer (slot 292 T5+T6). If slot 292 chooses to ship Schoof-only (O(log⁸ p), no Elkies/Atkin distinction) for v1, slot 294 has zero load-bearing consumer and can defer wholesale. If slot 292 commits to full SEA, slot 294 T0+T1+T3+T11 (~760 LOC + ~5 MB data) is mandatory. **Decision gate: does slot 292 v1 ship SEA or Schoof-only?**

Per `reviews/agents/292-new-elliptic-curves.md:124`: slot 292 explicitly proposes T6 SEA at ~1500 LOC. **Therefore slot 294 T11 is mandated; T0+T1+T3 are mandated as substrate. Other tiers (T4 theta, T5 Hecke, T6-T13 Manin/eigenform/L-function) are research-positioning — defer until a research consumer materializes.**

### Moat estimate

Pure-Go MIT zero-dep computational modular forms = **ABSENT in any language**. PARI is GPL-2, SageMath GPL-3, Magma proprietary, LMFDB MIT-data but GPL-engine-derived. Reality could be the FIRST clean MIT pure-Go modular-forms framework — but the audience is genuinely narrow (academic computational number theorists ≪ general developers). Recommend **scope to T0-T3 + T11 only** unless a downstream BSD-conjecture / mathematical-physics consumer commits.

## Sources

### Repo files (citations)
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md:311` — slot 294 line definition.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\292-new-elliptic-curves.md:117-121` — slot 292's ask for precomputed Φ_ℓ table (data sink for THIS slot's T11).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\292-new-elliptic-curves.md:124-130` — slot 292's SEA Elkies-prime branch hard-depends on Φ_ℓ.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\290-new-galois-theory.md:62-72` — slot 290's `PolyFp`/`PolyZ` substrate used by THIS slot's T11.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\291-new-modular-arithmetic.md` — Garner-CRT consumed by THIS slot's T11.
- `C:\limitless\foundation\reality\combinatorics\counting.go:43-98` — existing BinomialCoeff/Catalan/StirlingFirst/StirlingSecond/BellNumber/IntegerPartitions float64 surface (need bigint companions).
- `C:\limitless\foundation\reality\info\mdl\bernoulli.go:5-51` — namespace-collision warning: existing `NMLBernoulli` is the *probability distribution* MDL regret, NOT Bernoulli numbers; new `constants.BernoulliRat` does not conflict.
- `C:\limitless\foundation\reality\constants\math.go` — target home for new Bernoulli-numbers literal table.
- `C:\limitless\foundation\reality\crypto\modular.go` — uint64 ModPow/ModInverse used in per-prime stage of T11.

### Mathematical references (cited above per tier)
- Diamond-Shurman 2005, *A First Course in Modular Forms*, GTM 228 (canonical textbook).
- Stein 2007, *Modular Forms: A Computational Approach*, AMS GSM 79 (the algorithmic reference; chapters 6-9 cover dim formulas, Hecke, eigenforms, Manin).
- Cohen-Stromberg 2017, *Modular Forms: A Classical Approach*, AMS GSM 179 (700-page comprehensive reference).
- Apostol 1990, *Modular Functions and Dirichlet Series in Number Theory*, GTM 41 (classical analytic).
- Serre 1973, *A Course in Arithmetic*, GTM 7 §VII (introductory).
- Lang 1987, *Elliptic Functions*, GTM 112 §V (j-invariant).
- Cremona 1997, *Algorithms for Modular Elliptic Curves*, 2nd ed., CUP (ch. 2 Manin symbols).
- Manin 1972, *Parabolic points and zeta functions of modular curves*, Izv. Akad. Nauk SSSR Ser. Mat. 36(1):19-66.
- Atkin-Lehner 1970, *Hecke operators on Γ_0(m)*, Math. Ann. 185:134-160.
- Li 1975, *Newforms and functional equations*, Math. Ann. 212:285-315.
- Sturm 1987, *On the congruence of modular forms*, in *Number Theory (NY 1984-85)* LNM 1240:275-280.
- Cohen-Oesterlé 1977, *Dimensions des espaces de formes modulaires*, in *Modular Functions of One Variable VI* LNM 627:69-78.
- Sutherland 2011, *Computing Hilbert class polynomials with the Chinese Remainder Theorem*, Math. Comp. 80(273):501-538.
- Bröker-Lauter-Sutherland 2012, *Modular polynomials via isogeny volcanoes*, Math. Comp. 81(278):1201-1231.
- Mahler 1976, *On the modular function and its Mahler-Zassenhaus polynomials*.
- Mordell 1917, *On Mr. Ramanujan's empirical expansions of modular functions*, Proc. Camb. Phil. Soc. 19:117-124.
- Deligne 1974, *La conjecture de Weil. I*, Pub. IHES 43:273-307 (Ramanujan-Petersson bound).
- Eichler 1956 / Selberg 1956 — Eichler-Selberg trace formula.
- Conway-Norton 1979, *Monstrous Moonshine*, Bull. London Math. Soc. 11(3):308-339.
- Borcherds 1992, *Monstrous moonshine and monstrous Lie superalgebras*, Inventiones 109 (proof of Conway-Norton via vertex operator algebras).
- Bringmann-Folsom-Ono-Rolen 2017, *Harmonic Maass Forms and Mock Modular Forms: Theory and Applications*, AMS Colloquium Pubs. 64 (frontier).
- Zagier 2010, *Quantum modular forms*, in *Quanta of Maths*, Clay Math. Proc. 11 (frontier).
- Hardy-Wright 1979, *An Introduction to the Theory of Numbers*, 5th ed. (classical).
- Whittaker-Watson 1927, *A Course of Modern Analysis*, 4th ed. §21 (theta).
- Mumford 1983-1984, *Tata Lectures on Theta*, vols I-III.
- Knuth-Buckholtz 1967, computation of Dedekind sums.
- Rubinstein 2005, *Computational methods and experiments in analytic number theory*, MSRI vol. 44 (L-function numerics).
- Booker 2006, *Artin's conjecture, Turing's method, and the Riemann hypothesis*.
- OEIS A000521 (j-invariant q-coefficients); OEIS A000594 (Ramanujan τ); OEIS A027641/A027642 (Bernoulli num/denom).
- LMFDB (lmfdb.org) — MIT-data-licensed precomputed L-functions, newforms, Hecke eigenvalues; GPL-engine-derived (PARI/Sage). Cross-validation pin source for slot 294 numerical regressions.

### License moat sources
- PARI/GP: https://pari.math.u-bordeaux.fr/, GPL-2.
- SageMath: https://www.sagemath.org/, GPL-3.
- Magma: http://magma.maths.usyd.edu.au/, proprietary.
- LMFDB: https://www.lmfdb.org/, MIT (data) + CC-BY (documentation).
- gmpy2 / Arb: LGPL-3 / Apache-2 (high-precision but not modular-form-specific).

### Frontier deferred
- Bringmann-Folsom-Ono-Rolen harmonic Maass forms.
- Zagier quantum modular forms.
- Conway-Norton / Borcherds monstrous moonshine character tables.
- p-adic L-functions (Kato-Mazur-Wiles).
- Eichler-Selberg trace formula T14.
