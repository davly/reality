# 296 — new-generating-functions (OGF/EGF/DGF, Lagrange Inversion, Faà di Bruno, Symbolic Method, Singularity Analysis)

## Headline
Zero generating-function surface in reality despite combinatorics/ shipping ~10 sequences (each defined "natively" by a generating function); ~300 LOC day-1 PR (FormalPowerSeries + OGF/EGF/DGF arithmetic + Lagrange inversion) cross-validates every existing combinatorics primitive and is the natural symbolic backbone for slots 277-282 / 294-295.

## Findings

- **Zero surface.** Grep across reality for `GeneratingFunction|FormalPowerSeries|FormalSeries|Hadamard|LagrangeInversion|FaaDiBruno|BorelTransform|Mellin|SingularityAnalysis|SymbolicMethod|AnalyticCombinatorics|OGF|EGF|DGF` returns only review files (PROGRESS.md, prior agents 257/262/210/etc.). No data type, no operation. Nothing under combinatorics/, calculus/, prob/, or signal/ uses series semantics.
- **combinatorics/counting.go ships 10 closed-forms / recurrences** (Factorial:25, BinomialCoeff:51, Permutations:75, CatalanNumber:94, FibonacciNumber:110, StirlingFirst:159, StirlingSecond:194, BellNumber:234, IntegerPartitions:268, DerangementCount:296). Each has a textbook generating function: Catalan `C(z)=(1-√(1-4z))/(2z)`, Fibonacci `F(z)=z/(1-z-z²)`, Bell `B(z)=exp(eˣ-1)` (EGF), Stirling-2 `Σₙ S(n,k) zⁿ/n! = (eˣ-1)^k / k!`, partitions `P(z)=Π 1/(1-zᵏ)`, derangements `D(z) = e^{-z}/(1-z)` (EGF). None of these GFs are encoded — every primitive is a free-standing recurrence with no symbolic backbone tying them together.
- **Prerequisites already shipped:** signal/fft.go gives O(N log N) polynomial multiplication, sufficient for FormalPowerSeries product up to truncation N. constants/ provides π, e, γ. calculus/ provides Lagrange inversion's numerical fallback (Newton iteration). The infrastructure exists; the symbolic layer is missing.
- **Prior slot context:**
  - Slot 295 (l-functions) flagged Bernoulli numbers and Apéry constant absent; both are EGF coefficient extractions: B_n from `x/(eˣ-1)`, ζ(3) numerically from a series — natural consumers of T0+T1.
  - Slot 294 (modular forms) builds Eisenstein series `E_k = 1 + (2k/B_k) Σ σ_{k-1}(n) qⁿ` — pure EGF/q-series arithmetic.
  - Slots 277-282 NP-hard combinatorial counting (ILP, SBM, hypergraphs) want asymptotic-counting via Flajolet-Odlyzko transfer theorem from rational/algebraic GFs.
- **Symbolic combinatorics (Flajolet-Sedgewick "Analytic Combinatorics" 2009, Cambridge, free PDF on Flajolet's INRIA page).** The dictionary class→GF: SEQ(F)↔1/(1-F(z)), SET(F)↔exp(F(z)), CYC(F)↔ln(1/(1-F(z))), MSET(F)↔exp(Σ F(zᵏ)/k). With this dictionary plus singularity analysis, an enormous fraction of textbook combinatorics becomes mechanical. Reality's combinatorics/ should be derivable from it.
- **Convolution semantics differ by GF type.** OGF: (a*b)_n = Σ a_k b_{n-k} (Cauchy). EGF: (a*b)_n = Σ C(n,k) a_k b_{n-k} (binomial / labeled). DGF: (a*b)_n = Σ_{d|n} a_d b_{n/d} (Dirichlet). Three multiplication operators on the same array of coefficients — the entire point of having three wrappers.
- **Transforms are coefficient-wise scalar multiplies.** OGF↔EGF: a_n ↔ a_n·n!. Borel transform a_n → a_n/n! (asymptotic resummation). Mellin pairs OGF/DGF via integral that, for finite truncation, reduces to scalar identity. All cheap.
- **Lagrange inversion theorem.** If w = z·φ(w) with φ(0)≠0, then [zⁿ]w(z) = (1/n)·[w^{n-1}] φ(w)ⁿ. One identity, computes inverse-function coefficients, replaces an entire numerical Newton iteration. Catalan satisfies w = z(1+w²) modulo a substitution — Lagrange inversion gives Catalan in 5 lines.
- **Faà di Bruno / composition.** f(g(z)) coefficient-by-coefficient when g(0)=0, via Bell polynomials of partial derivatives of f at g(0). For truncated series this is straightforward O(N²) by direct convolution loop; avoid full Bell-polynomial machinery initially.
- **Hadamard product** = elementwise. Trivial loop. Useful for diagonal extraction in bivariate GFs and σ_k in Dirichlet world.
- **Partial-fraction decomposition for rational OGF.** Given p(z)/q(z), factor q via slot 290 (when 290 lands), then a_n = Σⱼ cⱼ αⱼⁿ in closed form. Composes with slot 290's polynomial GCD/factoring. Right now numerical: roots → vandermonde → coefficients, all in linalg/.
- **Singularity analysis (Flajolet-Odlyzko 1990, "Singularity analysis of generating functions", SIAM J. Discrete Math).** Transfer theorem: if f(z) ~ (1-z/r)^{-α} near dominant singularity r, then [zⁿ] f(z) ~ r^{-n} · n^{α-1} / Γ(α). One-shot asymptotic from the GF — automatic, no Stirling juggling.
- **Multivariate / D-finite frontier.** Pemantle-Wilson "Analytic Combinatorics in Several Variables" (CUP 2013) extends Flajolet-Odlyzko to multivariate. D-finite series (Stanley EC2 ch. 6) are series satisfying a linear ODE with polynomial coefficients — closure under +,×,Hadamard, integral. Mathematica's gfun.m, Maple's `gfun` package (GPL-2). Defer past day-1 but T11 is the obvious "Mathematica-Series[]-equivalent" target.
- **Moat estimate.** SageMath has `LazyPowerSeriesRing`/`PowerSeriesRing` (GPL-3). Maple gfun GPL-2. Mathematica proprietary. Python sympy `Poly`/`series` MIT-ish but not a true GF framework. **Pure-Go MIT zero-dep generating functions = ABSENT.** Reality could be the first. Moderate moat, very high research utility (every combinatorialist's daily tool).

## Concrete recommendations

1. **T0 — `series/series.go` `FormalPowerSeries` data type.** Truncated-to-order-N coefficient vector over ℝ (float64) and ℚ (big.Rat) variants. ~80 LOC.
   ```go
   type FPS struct { Coef []float64; Order int } // [zⁿ] f = Coef[n] for n ≤ Order
   func New(coef []float64) FPS
   func (f FPS) At(n int) float64
   func (f FPS) Truncate(N int) FPS
   ```
   Unblocks: everything below.

2. **T1 — `series/ogf.go` / `egf.go` / `dgf.go` wrappers + multiplication.** ~80 LOC total. Three types `OGF`, `EGF`, `DGF` wrap FPS; type-distinct `Mul` operators (Cauchy, binomial, Dirichlet). Sum / scalar-mul are shared.
   - Cauchy `(a*b)_n = Σ a_k b_{n-k}`: O(N²) direct, or signal/fft.go FFT path for N>256.
   - Binomial: O(N²) with one BinomialCoeff lookup table.
   - Dirichlet: O(N log N) via divisor enumeration, indexes start at n=1.

3. **T2 — `series/calculus.go` derivative + integral.** D[OGF]: a_n → (n+1)·a_{n+1}. ∫[OGF]: a_n → a_{n-1}/n with constant of integration. ~30 LOC.

4. **T3 — `series/lagrange.go` Lagrange inversion theorem.** `LagrangeInverse(phi FPS, N int) FPS` returning w(z) such that w = z·φ(w). Closed-form via [zⁿ]w = (1/n)[w^{n-1}] φ^n. ~40 LOC. Pin: Catalan C(z) emerges from φ(w)=1+w². Locked R-MUTUAL 3/3 with closed-form Catalan and recurrence (see §R-MUTUAL below).

5. **T4 — `series/compose.go` composition f(g(z)) with g(0)=0.** Direct O(N²) Horner-style. ~25 LOC. Optional Faà di Bruno coefficient extraction later.

6. **T5 — `series/hadamard.go` componentwise Hadamard product.** 5 LOC. Useful for diagonal extraction and bivariate GF reduction.

7. **T6 — `series/transforms.go` OGF↔EGF, Borel, Laplace.** All scalar multiplies of a_n by n!, 1/n!, etc. Mellin-Dirichlet bridge: pair OGF coefficients with the Dirichlet test functions n^{-s}. ~40 LOC.

8. **T7 — `series/rational.go` partial-fraction decomposition for rational OGF.** Numerical now (linalg roots + Vandermonde solve), upgrade to exact via slot 290's polynomial factoring when it lands. Yields closed-form a_n = Σ c_j α_j^n. ~60 LOC. Cross-validates Fibonacci closed form (Binet).

9. **T8 — `series/polya.go` Pólya cycle index polynomial.** Z_G(s_1,…,s_n) for permutation groups (cyclic, dihedral, symmetric). ~80 LOC. Combined with T1+T4 gives Pólya enumeration (necklaces, etc.).

10. **T9 — `series/symbolic.go` symbolic method classes.** Combinatorial-class type with constructors `Atom`, `Empty`, `Plus`, `Times`, `SEQ`, `SET`, `CYC`, `MSET`. `(c Class).GF(N int) FPS` translates to truncated series. ~100 LOC. This is the "Flajolet-Sedgewick textbook in 100 LOC."

11. **T10 — `series/asymptotic.go` Flajolet-Odlyzko singularity analysis.** Given GF as rational or algebraic, locate dominant singularity (roots of denominator nearest 0; algebraic case branch points), extract (1-z/r)^{-α} type, return asymptotic [zⁿ] f ~ A · r^{-n} · n^{α-1} / Γ(α). ~120 LOC. Day-2.

12. **T11 — `series/dfinite.go` D-finite series.** Series defined by linear ODE Σ p_k(z) f^{(k)}(z) = 0. Closure under +, ×, Hadamard, ∫ via Gröbner-style ODE-multiplication of operators. Mathematica's `Series[]` / `gfun.m` equivalent. ~250 LOC. **Frontier; defer to dedicated effort.**

13. **T12 — `series/multivariate.go` Pemantle-Wilson ACSV.** Bivariate at minimum. ~200 LOC. Frontier.

14. **T14 — `series/guess.go` "guess GF from N terms" via Padé approximation.** Continued-fraction / Berlekamp-Massey on coefficients to produce minimal rational GF. OEIS-style. ~80 LOC. High wow-factor; cheap once T0 exists.

**Day-1 PR:** T0+T1+T2+T3 = ~230 LOC. Independently testable. Every result of combinatorics/ becomes cross-checkable.

## R-MUTUAL-CROSS-VALIDATION 3/3 pins (saturating)

- **Catalan (3 paths).**
  1. closed form `C_n = C(2n,n)/(n+1)` (existing combinatorics.CatalanNumber:94)
  2. Lagrange inversion of φ(w)=1+w² → C(z) = [zⁿ] w
  3. recurrence `C_{n+1} = Σ_{i=0}^{n} C_i C_{n-i}` via OGF self-product (T1 Cauchy mul of C with C plus shift)
  Pin: identical first 100 coefficients to ULP via Rat coefficients.

- **Fibonacci (3 paths).**
  1. matrix exponentiation (existing FibonacciNumber:110)
  2. partial-fraction decomposition of `z/(1-z-z²)` via T7 → Binet
  3. T1 OGF multiplication of `1/(1-z-z²)` (multiply-and-truncate)

- **Bernoulli numbers (saturating one R-MUTUAL once added — keystone for slot 295).**
  EGF `x/(eˣ-1) = Σ Bₙ xⁿ/n!` ≡ recurrence `Σ_{k=0}^{n} C(n+1,k) B_k = δ_{n,0}` ≡ closed form via ζ(-n) = -B_{n+1}/(n+1) (consumer of slot 295). Locks Bernoulli into 3-path identity.

- **Stirling-2 (3 paths).**
  1. recurrence (existing StirlingSecond:194)
  2. EGF `Σₙ S(n,k) xⁿ/n! = (eˣ-1)^k / k!` via T1 EGF power + T6 OGF↔EGF transform
  3. closed form S(n,k) = (1/k!) Σⱼ (-1)ʲ C(k,j) (k-j)ⁿ

- **Bell.** EGF `B(z) = exp(eˣ-1)` ≡ Bell triangle (existing) ≡ Σ S(n,k) over k.

## Cross-cutting

- **combinatorics/ (slots 037-039)** ← every existing primitive is the [zⁿ] coefficient of a textbook GF; T0+T1 enables symbolic verification of the entire package via 3-way pins.
- **slot 277/278 (ILP, COPO)** ← Stanley EC1 ch. 4: Ehrhart polynomials and quasipolynomials of integer polytopes are themselves rational GFs; the `(1-z)^{-d}`-type singularity → polynomial-in-n lattice-point count. T7+T10 give automatic Ehrhart asymptotics.
- **slot 280-282 (SBM, temporal-graphs, hypergraphs)** ← motif counts have GF-based asymptotics via Flajolet-Sedgewick "marked classes."
- **slot 294 (modular forms)** ← Eisenstein-series q-expansions, theta series, η-products are all q-series — direct OGF arithmetic. The "1 + (2k/B_k) Σ σ_{k-1}(n) qⁿ" Eisenstein E_k construction is one OGF + one Dirichlet sum.
- **slot 295 (L-functions)** ← Bernoulli numbers via x/(eˣ-1) EGF; Hurwitz ζ via Σ (n+a)^{-s} DGF.
- **prob/** ← probability generating function `G_X(z) = E[z^X] = Σ P(X=k) zᵏ`; moment generating function `M_X(t) = E[e^{tX}]` is just G_X with z = eᵗ (T6 transform). Consumer for discrete-distribution tooling.
- **info/** ← entropy rate of finite-state Markov source = log of dominant eigenvalue of GF-transfer-matrix; T10 singularity analysis gives it.
- **signal/fft.go** ← consumed by T1 Cauchy convolution for N>256.
- **calculus/** ← consumes T2 derivative/integral for sanity (taylor-coefficient verification of math.Sin/Cos/Exp etc.).
- **slot 290 (Galois / polynomial factoring)** ← consumed by T7 to upgrade rational-OGF partial-fraction from numerical to exact.
- **slot 042 (Bernoulli? if exists)** — none; this is currently a gap.

## Cheapest day-1 PR

**~230 LOC, single PR `series/{series,ogf,egf,dgf,calculus,lagrange}.go` + tests:**
- T0 FormalPowerSeries (~80)
- T1 OGF/EGF/DGF wrappers + three Mul semantics (~80)
- T2 derivative + integral (~30)
- T3 Lagrange inversion (~40)

Tests: 3-way Catalan pin, 3-way Fibonacci pin, 3-way Stirling-2 pin, EGF↔OGF round-trip on `exp`, Bernoulli-via-EGF generates first 16 Bernoulli numbers (which slot 295 needs).

## Why this matters more than the slot number suggests

This is a **CORE primitive**. It should ideally have shipped before half of combinatorics/. Every combinatorial sequence has its "true" definition via generating function — the GF *is* the canonical encoding, the closed form / recurrence are derivatives. Adding the GF infrastructure unlocks symbolic verification of every existing combinatorial primitive AND becomes the natural input format for slots 277-282 (combinatorial NP-hard asymptotics), slot 294 (q-series modular forms), slot 295 (Bernoulli, Hurwitz ζ). The R-MUTUAL pins above demonstrate that the existing combinatorics/ tests are *under-pinned* — currently each primitive validates against only one or two paths; with GFs each gets a third independent path "for free."

## Sources

- `C:/limitless/foundation/reality/combinatorics/counting.go:25-305` (existing combinatorial primitives, all GF-derivable)
- `C:/limitless/foundation/reality/combinatorics/generate.go` (generation functions, complement to counting)
- `C:/limitless/foundation/reality/signal/fft.go` (consumed by T1 large-N Cauchy product)
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/295-new-l-functions.md` (Bernoulli, Hurwitz ζ flagged absent — direct consumers)
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/294-new-modular-forms.md` (q-series Eisenstein construction needs OGF arithmetic)
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/290-new-galois-theory.md` (T7 partial-fraction upgrade path)
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/{037,038,039}-combinatorics-*.md` (existing combinatorics review context)
- Flajolet & Sedgewick, *Analytic Combinatorics*, Cambridge 2009 (free PDF on INRIA Algorithms Project page) — symbolic method, transfer theorems, singularity analysis
- Wilf, *generatingfunctionology*, 2nd ed. 1994 (free PDF on author's UPenn page) — pedagogical canonical reference for OGF/EGF
- Stanley, *Enumerative Combinatorics* vol. 1 (CUP 2nd ed. 2012) and vol. 2 (1999) — D-finite series ch. 6.4, exponential structures
- Pemantle & Wilson, *Analytic Combinatorics in Several Variables*, CUP 2013 (T12)
- Flajolet & Odlyzko, "Singularity analysis of generating functions," SIAM J. Discrete Math 3 (1990) 216-240 (T10 transfer theorem)
- Salvy & Zimmermann, "GFUN: a Maple package for the manipulation of generating and holonomic functions in one variable," ACM TOMS 20 (1994) 163-177 (D-finite reference implementation, GPL-2)
- OEIS https://oeis.org — every sequence has its GF; corpus for testing T14 "guess GF" tooling
- SageMath `LazyPowerSeriesRing` (GPL-3), Mathematica `Series[]` (proprietary), Maple `gfun` (GPL-2): the existing landscape — none MIT, none pure-Go
