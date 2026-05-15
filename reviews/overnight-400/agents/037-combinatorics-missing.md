# 037 — combinatorics-missing

**Topic:** canonical combinatorics primitives not yet in `combinatorics/`.

**Scope reviewed:** `combinatorics/counting.go` (306 LOC, 10 fns), `combinatorics/generate.go`
(191 LOC, 4 fns), 036's catalog of present-but-numerically-suspect functions.
036 covered numerical contracts of what exists; this report enumerates what is missing.

## Headline

Today `combinatorics/` exports **14 callables** (10 counting + 4 generation), versus a
canonical-2026 textbook surface — measured against Stanley's *Enumerative Combinatorics*
(vol. 1 ch. 1-2, vol. 2 ch. 5-7), Sage's `sage.combinat` (≈600 named functions), FLINT
`arith` + `fmpz_combinat` (≈90 functions), Mathematica's `Combinatorica` legacy +
`DiscreteMath` modern surface, and Python `sympy.combinatorics` — of roughly **180-220
named primitives** at the level reality should ship. So **~7%** of the canonical surface
is present, and the missing 93% splits cleanly into counting (Tier 1, fast wins, small
LOC each), structure-generators (Tier 2, RSK / Young / Dyck families), and poset/symbolic
machinery (Tier 3, requires architectural decisions on representation).

The single most consequential **missing primitive** is `Multinomial(k_1, ..., k_m)` — a
4-line wrapper over `Lgamma` that every probability/Bayesian consumer needs (multinomial
PMF, Dirichlet normalization, χ²-test exact-tail) and is not in the package. The single
most consequential **missing structure** is `RSK` (Robinson-Schensted-Knuth) — the
bijection at the centre of algebraic combinatorics, tableau enumeration, and longest-
increasing-subsequence; every serious combinatorics library has it.

This report enumerates the missing surface in three tiers. Tier 1 = textbook-canonical
counting fns + foundational structure generators (≤80 LOC each, all golden-file-testable
against OEIS, all zero-dep). Tier 2 = canonical structures requiring a representation
choice (Young tableaux, partitions-as-objects, lattice paths, Catalan structures) and
the polynomial families. Tier 3 = poset/lattice/symbolic machinery (Möbius on a poset,
chromatic/Tutte for graphs, generating-function algebra, Hall-Littlewood / Schur).

## Tier 1 — must ship (textbook canonical, ≤80 LOC each, OEIS-pinnable)

Every entry below is a direct OEIS lookup with a closed form or trivial recurrence,
testable to 30+ vectors, zero new dependencies. All return `float64` per the package
convention; per 036, each should also ship a `*Exact` companion returning `(uint64, bool)`
where the value fits and a `*Big` companion via `math/big`.

### Counting — sequence values

| # | Item | OEIS | Recurrence / closed form | Why Tier 1 |
|---|---|---|---|---|
| T1.C1 | **Multinomial(k_1,...,k_m)** | n/a | `n! / ∏ k_i!` via `Lgamma` | Most-cited missing primitive — multinomial PMF, Dirichlet normalization, exact χ²-tail. Topic-prompt-named. ~8 LOC. |
| T1.C2 | **FallingFactorial(x, n)** | n/a | `x·(x-1)···(x-n+1)` | Topic-prompt-named. Foundation of finite-difference calculus, used by Newton-forward-diff in `calculus/`. ~6 LOC. |
| T1.C3 | **RisingFactorial(x, n)** = Pochhammer `(x)_n` | n/a | `x·(x+1)···(x+n-1)` = `Γ(x+n)/Γ(x)` | Topic-prompt-named. Foundation of hypergeometric series; needed by `prob/` for negative-binomial PMF closed forms. ~6 LOC. |
| T1.C4 | **LucasNumber(n)** | A000032 | `L_n = L_{n-1}+L_{n-2}; L_0=2, L_1=1` | Sister to Fibonacci; same matrix-exp trick. ~25 LOC. |
| T1.C5 | **TribonacciNumber(n)** | A000073 | `T_n = T_{n-1}+T_{n-2}+T_{n-3}` | Standard 3-term recurrence; matrix-exp 3×3. ~30 LOC. |
| T1.C6 | **PellNumber(n)** | A000129 | `P_n = 2P_{n-1}+P_{n-2}` | Number-theoretic standard; √2 continued-fraction convergents. ~20 LOC. |
| T1.C7 | **EulerianFirst(n,k)** = `⟨n k⟩` | A008292 | `⟨n k⟩ = (k+1)⟨n-1,k⟩ + (n-k)⟨n-1,k-1⟩` | Permutations by ascent count; sums to n!. Topic-prompt-named. ~25 LOC DP. |
| T1.C8 | **EulerianSecond(n,k)** = `⟨⟨n k⟩⟩` | A008517 | `⟨⟨n k⟩⟩ = (k+1)⟨⟨n-1,k⟩⟩ + (2n-k-1)⟨⟨n-1,k-1⟩⟩` | Worpitzky; Stirling-poset count. Topic-prompt-named. ~25 LOC DP. |
| T1.C9 | **BernoulliNumber(n)** = `B_n` | A027641/A027642 | Akiyama-Tanigawa table OR `B_n = -1/(n+1) Σ C(n+1,k) B_k` | Topic-prompt-named. Needed by Faulhaber's formula, Riemann-ζ at integer args, Euler-Maclaurin (used by `calculus/` Romberg). Akiyama-Tanigawa is the numerically-stable choice. ~40 LOC. |
| T1.C10 | **EulerNumber(n)** = `E_n` | A000364 | `Σ E_{n-2k} C(n,2k) = 0` for `n≥2`; secant-tangent series | Topic-prompt-named. Sister to Bernoulli; alternating-permutation count. ~35 LOC. |
| T1.C11 | **MotzkinNumber(n)** | A001006 | `M_{n+1} = M_n + Σ M_k M_{n-1-k}` OR `(n+2)M_n = (2n+1)M_{n-1} + 3(n-1)M_{n-2}` | Catalan-cousin: paths/RNA-secondary-structure. Linear-recurrence form is faster. ~20 LOC. |
| T1.C12 | **NarayanaNumber(n,k)** = `N(n,k)` | A001263 | `(1/n)·C(n,k)·C(n,k-1)` | Refinement of Catalan: Dyck paths by peak count. ~6 LOC. |
| T1.C13 | **LahNumber(n,k)** | A105278 (signed) / A271703 (unsigned) | `L(n,k) = C(n-1,k-1) · n!/k!` | Topic-prompt-named. Bridge between rising and falling factorials. ~10 LOC. |
| T1.C14 | **PartitionsIntoKParts(n,k)** = `p_k(n)` | A008284 (triangle) | `p_k(n) = p_{k-1}(n-1) + p_k(n-k)` | Refines `IntegerPartitions(n)`. Topic-prompt mentioned via "all compositions/partitions". ~15 LOC DP. |
| T1.C15 | **PartitionsIntoDistinctParts(n)** = `Q(n)` | A000009 | Euler pentagonal-number recurrence | Distinct-parts; equals odd-parts (Euler). ~25 LOC. |
| T1.C16 | **CompositionCount(n)** = `2^(n-1)` | A011782 | trivial | Topic-prompt-named (compositions). 2 LOC. |
| T1.C17 | **CompositionsIntoKParts(n,k)** | n/a | `C(n-1, k-1)` | Stars-and-bars. 2 LOC. |
| T1.C18 | **WeakCompositions(n,k)** | n/a | `C(n+k-1, k-1)` | Stars-and-bars allowing zeros. 2 LOC. |
| T1.C19 | **PentagonalNumber(n)** | A001318 | `n(3n-1)/2` (generalized pentagonal: `±1, ±2, ...`) | Used by Euler's recurrence for `p(n)`; needed if T1.C15 ships. 5 LOC. |
| T1.C20 | **HarmonicNumber(n)** = `H_n` | A001008/A002805 | `Σ 1/k` directly OR `ψ(n+1)+γ` | Combinatorial constant: expected cycle length, coupon collector. ~5 LOC. |
| T1.C21 | **BellTriangle(n)** | A011971 | the triangle from which `B_n` falls out | The package computes Bell via the triangle but throws the table away. Returning the full triangle is "free" (already computed). ~15 LOC. |

### Counting — exact / generating-function coefficients

| # | Item | Reference | Why Tier 1 |
|---|---|---|---|
| T1.C22 | **PochhammerLog(x, n)** | n/a | `Lgamma(x+n) - Lgamma(x)` — log-domain Pochhammer for `x` near zero or large `n`; needed when raw `RisingFactorial` overflows. Symmetric to existing `Lgamma`-based `BinomialCoeff`. ~5 LOC. |
| T1.C23 | **LogBinomial(n, k)** | Numerical Recipes ch.6 | `Lgamma(n+1)-Lgamma(k+1)-Lgamma(n-k+1)` exposed directly — needed by every Bayesian / hypergeometric calculation that wants the result in log-space without round-tripping through `BinomialCoeff` and back through `Log`. ~3 LOC. |
| T1.C24 | **LogFactorial(n)** | n/a | `Lgamma(n+1)` exposed — same rationale. ~2 LOC. |
| T1.C25 | **LogMultinomial(k_1,...,k_m)** | n/a | `Lgamma(Σk_i+1) - Σ Lgamma(k_i+1)`. ~5 LOC. |

### Generators — single-step / iterator-style

| # | Item | Reference | Why Tier 1 |
|---|---|---|---|
| T1.G1 | **NextCombination(c, n)** | Knuth TAOCP 4A §7.2.1.3 Algorithm L | Single-step counterpart to `NextPermutation`; lets the caller stream C(n,k) combos without the O(C(n,k)·k) memory of `GenerateCombinations`. ~15 LOC. |
| T1.G2 | **NextSubset(s, n)** = Gosper's hack | Knuth TAOCP 4A §7.1.3 (Hakmem 175) | Topic-prompt-named ("Gosper's hack"). Pure-bitwise next-k-subset for k ≤ 64. The fastest known same-popcount-next algorithm. ~8 LOC. |
| T1.G3 | **NextPartition(p)** | Stanley EC1 §1.8 | Lex-next of an integer partition (decreasing-sequence form). Streams `IntegerPartitions(n)` partitions in O(n) memory. ~25 LOC. |
| T1.G4 | **NextSetPartition(p)** | Knuth TAOCP 4A §7.2.1.5 | Lex-next of a set partition in restricted-growth-string form. ~30 LOC. |
| T1.G5 | **NextComposition(c)** | n/a | Lex-next of a composition. ~15 LOC. |
| T1.G6 | **PermutationRank(p) / PermutationUnrank(r, n)** | Knuth TAOCP 4A §7.2.1.2 | Bijection `S_n ↔ {0,...,n!-1}` via factorial-base / Lehmer code. Lets reservoir-sampling and SRS-PRNG generate random permutations without full Fisher-Yates. ~20 LOC each. |
| T1.G7 | **CombinationRank(c, n) / CombinationUnrank(r, n, k)** | Knuth TAOCP 4A §7.2.1.3 | Combinatorial number system. Bijection to `{0,...,C(n,k)-1}`. ~15 LOC each. |
| T1.G8 | **GeneratePermutationsSJT([n]int)** | Steinhaus-Johnson-Trotter | Topic-prompt-named ("SJT"). Gray-code permutation generator: each successive permutation differs by one adjacent transposition (Heap's differs by an arbitrary swap). Needed for sign-tracking / sign-reversing applications. ~50 LOC. |
| T1.G9 | **GenerateDerangements(n)** | Sedgewick "Permutation generation methods" | Enumerate all `!n` derangements; companion to existing `DerangementCount`. Topic-prompt-named. ~40 LOC. |
| T1.G10 | **GenerateIntegerPartitions(n)** | Stanley EC1 | Materialise all `p(n)` partitions; companion to existing `IntegerPartitions`. Topic-prompt-named ("all compositions / partitions"). ~30 LOC. |
| T1.G11 | **GenerateSetPartitions(n)** | Knuth TAOCP 4A | Materialise all `B_n` set-partitions; companion to existing `BellNumber`. Topic-prompt-named ("Set partitions / Stirling 2nd kind enumeration"). ~40 LOC. |
| T1.G12 | **GenerateCompositions(n)** | n/a | Materialise all `2^(n-1)` compositions. ~20 LOC. |
| T1.G13 | **GenerateSubsets(n)** | n/a | Materialise all `2^n` subsets in either lex or Gray-code order. ~15 LOC. |

### Inclusion-exclusion helper

| # | Item | Reference | Why Tier 1 |
|---|---|---|---|
| T1.I1 | **InclusionExclusion(n int, count func(subset []int) float64)** | Stanley EC1 §2.1 | Topic-prompt-named. Fixed pattern: alternating sum over `2^n` subsets. Avoids each consumer reimplementing the bit-loop and the `(-1)^|S|` sign. ~15 LOC. |

**Tier 1 totals:** ~46 named items, ~510 LOC, every one OEIS-pinnable to ≥30 vectors.
Per-call cost: zero. None changes existing function signatures.

## Tier 2 — should ship (canonical structures, representation choices required)

These items each require a small representation decision (e.g., "how do we store a Young
tableau?"). Once that decision is made, each algorithm is textbook.

### Young tableaux & partitions as objects

A `Partition` type — `[]int` decreasing, with helpers — is the prerequisite for all of these.

| # | Item | Reference | Type / Why Tier 2 |
|---|---|---|---|
| T2.Y1 | **`Partition` type + `Conjugate(λ)` + `IsPartition` + `PartitionDominates`** | Stanley EC2 §7.2 | foundational `[]int` wrapper; conjugate (Ferrers transpose) is 5 LOC; dominance order check is 8 LOC. Prerequisite for everything below. |
| T2.Y2 | **FerrersDiagram(λ) → [][]bool** + ASCII renderer | Stanley EC2 §7.2 | Topic-prompt-named ("Ferrers diagrams"). Visualisation primitive. ~15 LOC. |
| T2.Y3 | **YoungDiagram(λ)** + cell-iterator | n/a | French/English convention selectable; same data as Ferrers but row-indexed. ~10 LOC. |
| T2.Y4 | **HookLength(λ, i, j)** | Frame-Robinson-Thrall (1954) | Topic-prompt-named ("Hook-length formula"). `λ_i - j + λ'_j - i + 1`. ~5 LOC. |
| T2.Y5 | **HookLengthFormula(λ)** = `f^λ` | Frame-Robinson-Thrall | Topic-prompt-named. `n! / ∏ h(c)` — counts standard Young tableaux of shape λ. ~10 LOC. |
| T2.Y6 | **GenerateStandardYoungTableaux(λ)** | Stanley EC2 §7.10 | Topic-prompt-named ("All Young tableaux of shape λ"). Greene-Nijenhuis-Wilf or direct cell-by-cell enumeration. ~60 LOC. |
| T2.Y7 | **GenerateSemistandardYoungTableaux(λ, μ)** | Stanley EC2 §7.10 | Topic-prompt-named ("Standard / semi-standard Young tableaux"). Filling λ with content μ; weak-increasing-rows + strict-increasing-cols. ~70 LOC. |
| T2.Y8 | **RSK(σ) → (P, Q)** | Knuth (1970); Stanley EC2 §7.11 | Topic-prompt-named ("Robinson-Schensted-Knuth correspondence"). The single most-cited missing item. Bijection between matrices/permutations and pairs of SSYT. ~80 LOC. |
| T2.Y9 | **InverseRSK(P, Q) → σ** | Stanley EC2 §7.11 | Companion to T2.Y8; row-bumping in reverse. ~60 LOC. |
| T2.Y10 | **Promotion(T) / Evacuation(T)** | Schützenberger | Tableau-level permutation actions. ~50 LOC. |
| T2.Y11 | **JeuDeTaquin(T, c)** | Schützenberger | Sliding moves on skew tableaux; foundation for LR-coefficient computation. ~70 LOC. |

### Catalan-family structures (topic-prompt-named explicitly)

| # | Item | Reference | Why Tier 2 |
|---|---|---|---|
| T2.K1 | **GenerateDyckPaths(n)** | Stanley EC2 §6 | Topic-prompt-named ("Dyck paths"). All `C_n` lattice paths; bit-string (U/D) form. ~30 LOC. |
| T2.K2 | **GenerateBallotSequences(n)** | Stanley EC2 §6 | Topic-prompt-named ("ballot sequences"). Reflection-principle relative. ~30 LOC. |
| T2.K3 | **GenerateBinaryTrees(n)** | Knuth TAOCP 4A §7.2.1.6 | Topic-prompt-named ("binary trees"). All `C_n` shapes; recursive product. ~40 LOC. |
| T2.K4 | **GenerateParenthesizations(n)** | Knuth TAOCP 4A | Topic-prompt-named. String form; bijection-companion to T2.K3. ~25 LOC. |
| T2.K5 | **GenerateNonCrossingPartitions(n)** | Stanley EC2 ex.6.19 | Catalan-counted; needed for free-probability consumers. ~40 LOC. |
| T2.K6 | **GeneratePermutationsAvoiding(n, pattern)** | Bóna "Combinatorics of Permutations" | Length-3 patterns all give Catalan; longer patterns are open research. ~50 LOC. |

### Lattice paths

| # | Item | Reference | Why Tier 2 |
|---|---|---|---|
| T2.L1 | **GenerateLatticePaths(m, n)** | Stanley EC1 §1 | Topic-prompt-named ("Lattice paths"). All `C(m+n,m)` E/N paths from (0,0) to (m,n). ~25 LOC. |
| T2.L2 | **DelannoyNumber(m, n)** | A008288 | E/N/NE paths; classical. ~10 LOC DP. |
| T2.L3 | **SchroderNumber(n)** | A006318 | Large Schröder; E/N/NE paths above diagonal. ~15 LOC. |
| T2.L4 | **MotzkinPathCount via T1.C11** | already Tier 1 | (cross-ref) |

### Polynomials

| # | Item | Reference | Why Tier 2 |
|---|---|---|---|
| T2.P1 | **BellPolynomialPartial(n, k, x_1, x_2, ...)** = `B_{n,k}` | Comtet "Advanced Combinatorics" §3.3 | Topic-prompt-named ("Bell polynomials"). Faà di Bruno's formula for chain-rule of derivatives — needed by `autodiff/` for higher-order. ~40 LOC. |
| T2.P2 | **BellPolynomialComplete(n, x_1, x_2, ...)** = `B_n` | Comtet | Sum over k of T2.P1; cumulants ↔ moments. ~10 LOC. |
| T2.P3 | **TouchardPolynomial(n, x)** = `T_n(x)` | Touchard (1939) | Topic-prompt-named. `Σ S(n,k) x^k`; Bell-poly evaluated at x=1 gives B_n. ~15 LOC. |
| T2.P4 | **CyclePolynomial(σ)** / **CycleIndex(G)** | Stanley EC1 §1.3 | Pólya enumeration foundation. Group-action variant in Tier 3. ~30 LOC for permutation form. |
| T2.P5 | **PolynomialFromCoeffs / Eval / Mul / Compose** (small ring helper) | n/a | Required infrastructure for T2.P1-P4 and T3 (chromatic, Tutte). ~80 LOC. Could live in a new `combinatorics/poly` sub-file or share with `linalg/poly`. |
| T2.P6 | **OrdinaryGeneratingFunction series ops** | Wilf "generatingfunctionology" | Topic-prompt-named ("ordinary, exponential GF compose/multiply/series"). Truncated-series multiply, divide, compose. ~70 LOC. |
| T2.P7 | **ExponentialGeneratingFunction series ops** | Wilf | EGF flavour: multiply by `n!` after compose. ~30 LOC over T2.P6. |
| T2.P8 | **CoefficientExtraction `[x^n] f(x)`** | Wilf | Topic-prompt-named. Trivial after T2.P5 (return `coeffs[n]`). ~3 LOC. |
| T2.P9 | **LagrangeInversion(f, n)** | Wilf §5.1 | Topic-prompt-named. `[x^n] g(x) = (1/n)·[w^(n-1)] (w/f(w))^n` — the canonical species/tree-counting identity. ~30 LOC. |

### Combinatorial number-theory adjacent

| # | Item | Reference | Why Tier 2 |
|---|---|---|---|
| T2.N1 | **q-Binomial `[n k]_q`** | Stanley EC1 §1.7 | Topic-prompt-named ("q-binomial"). Gaussian binomial; subspace count over F_q. ~20 LOC. |
| T2.N2 | **q-Pochhammer `(a;q)_n`** | Andrews-Askey-Roy | Topic-prompt-named. Foundation of q-series. ~15 LOC. |
| T2.N3 | **q-Factorial `[n]_q!`** | Stanley EC1 | Companion to T2.N1. ~10 LOC. |
| T2.N4 | **q-Catalan `C_n(q)`** | Stanley EC1 ex.1.140 | MacMahon's q-analog. ~15 LOC. |
| T2.N5 | **MajorIndex(σ)** / **InversionCount(σ)** | MacMahon (1916) | Permutation statistics; equidistributed (MacMahon's theorem). Foundation of q-counting. ~15 LOC each. |

### Sequences not in Tier 1 (less universally cited but still canonical)

| # | Item | Reference | Why Tier 2 |
|---|---|---|---|
| T2.S1 | **SchroderHipparchusNumber(n)** = little Schröder | A001003 | Plane-tree count. ~15 LOC. |
| T2.S2 | **SuperCatalanNumber(m,n)** | A068555 | Two-variable Catalan generalization. ~10 LOC. |
| T2.S3 | **JacobiStirlingFirst/Second(n,k,z)** | Andrews-Egge-Gawronski-Littlejohn (2013) | Topic-prompt-named ("Jacobi-Stirling"). Bessel / Legendre poly connection. ~30 LOC each. |
| T2.S4 | **WhitneyNumberFirst/Second(n,k)** of Boolean lattice | Stanley EC1 §3.13 | Topic-prompt-named ("Whitney numbers"). Refinement of binomial / Stirling for graded posets; full poset version in Tier 3. Boolean-lattice version is closed-form. ~10 LOC. |
| T2.S5 | **RamseyR(s,t) bounds** | Radziszowski survey | Known small values + Erdős-Szekeres bounds; data table not algorithm. ~30 LOC. |

**Tier 2 totals:** ~38 items, ~1,200 LOC. The four sub-clusters can ship independently;
Y-cluster is the biggest single win because RSK is so widely cited.

## Tier 3 — research-grade or large coordination scope

These items each require either a new representation (`Poset`, `Graph` outside the
existing `graph/` package's algorithmic focus, or symmetric-functions ring) or a
non-trivial algorithm beyond what one agent can scope.

### Posets / lattices (topic-prompt-named: "Posets / lattices" cluster)

| # | Item | Reference | Why Tier 3 |
|---|---|---|---|
| T3.P1 | **`Poset` type + Hasse diagram + cover relation + interval enumeration** | Stanley EC1 ch.3 | Foundational data structure; ~150 LOC just for the type + 5-6 helpers. Forces a representation decision (matrix vs adjacency-list vs cover-graph). Once shipped, T3.P2-P7 all follow. |
| T3.P2 | **MöbiusFunction(P, x, y)** on a finite poset | Rota (1964) | Topic-prompt-named ("Möbius function on a poset"). Recursive `μ(x,y) = -Σ_{x≤z<y} μ(x,z)`. ~25 LOC after T3.P1. |
| T3.P3 | **MöbiusInversion(P, f)** | Topic-prompt-named ("Möbius inversion") | ~15 LOC after T3.P2. |
| T3.P4 | **PartitionLattice(n)** | Topic-prompt-named ("Partition lattice") | The poset of set-partitions of [n] under refinement; `B_n` elements. Build as concrete `Poset`. ~40 LOC after T3.P1. |
| T3.P5 | **YoungLattice(N)** | Topic-prompt-named ("Young's lattice") | Partitions ordered by containment; truncated to partitions of size ≤ N. ~30 LOC after T3.P1. |
| T3.P6 | **BooleanLattice(n)** | Topic-prompt-named ("Boolean lattice") | `2^[n]` under inclusion. ~15 LOC after T3.P1. |
| T3.P7 | **OrderPolynomial(P, t)** = `Ω(P, t)` | Stanley EC1 §3.12 | Topic-prompt-named ("Stanley reciprocity / order polynomial"). Reciprocity: `Ω(P,-t) = (-1)^|P| Ω*(P,t)`. ~40 LOC after T3.P5. |
| T3.P8 | **ZetaPolynomial(P, n)** | Stanley EC1 §3.12 | Number of multichains; reciprocal of Möbius. ~20 LOC. |

### Graph polynomials (topic-prompt-named explicitly)

| # | Item | Reference | Why Tier 3 |
|---|---|---|---|
| T3.G1 | **ChromaticPolynomial(G, k)** | Birkhoff (1912); Topic-prompt-named | Deletion-contraction recurrence over `graph.Graph`; exponential in worst case. ~80 LOC + needs poly type from T2.P5. Cross-package: extend `graph/`. |
| T3.G2 | **TuttePolynomial(G, x, y)** | Tutte (1947); Topic-prompt-named | Two-variable refinement of chromatic. Same del-con recursion. ~100 LOC. |
| T3.G3 | **ReliabilityPolynomial(G, p)** | derived from T3.G2 | `T(G; 1, 1/p)·p^(n-1)·(1-p)^(m-n+1)` flavour. ~10 LOC after T3.G2. |
| T3.G4 | **MatchingPolynomial(G, x)** / **IndependencePolynomial(G, x)** | Heilmann-Lieb | Bipartite-perfect-matching link; complementary identities. ~60 LOC each. |
| T3.G5 | **PermanentBruteOrRyser(M)** | Ryser (1963) | Permanent via inclusion-exclusion in `O(2^n · n)` — vastly better than expansion's `O(n!)`. Foundation of bipartite matching count. ~30 LOC. |

### Symmetric functions (Tier 3 because the ring representation is large)

| # | Item | Reference | Why Tier 3 |
|---|---|---|---|
| T3.S1 | **`SymmetricFunction` type + e/h/p/m/s basis change** | Macdonald "Symmetric Functions and Hall Polynomials" | Topic-prompt-named ("Schur, Hall-Littlewood polynomials (advanced)"). The whole infrastructure (`SymmetricFunction` value, basis-change matrices, multiplication, Hall inner product) is ≥800 LOC done right. Major architectural commit. |
| T3.S2 | **SchurPolynomial(λ, x_1,...,x_n)** | Macdonald | Topic-prompt-named. Bialternant formula or Jacobi-Trudi. ~50 LOC after T3.S1 + T2.Y8 (RSK). |
| T3.S3 | **HallLittlewoodPolynomial(λ, x; t)** | Macdonald III.2 | Topic-prompt-named ("Hall-Littlewood polynomials"). One-parameter Schur deformation. ~80 LOC after T3.S1. |
| T3.S4 | **MacdonaldPolynomial(λ; q, t)** | Macdonald (1988) | Two-parameter generalization; current frontier of algebraic combinatorics. ~150 LOC after T3.S3. |
| T3.S5 | **LittlewoodRichardsonCoeffs(λ, μ, ν)** | Stanley EC2 §A1.3 | Via skew-tableau enumeration (T2.Y11). ~120 LOC. |
| T3.S6 | **KostkaNumbers(λ, μ)** | Macdonald | SSYT count of shape λ content μ; specializes T2.Y7. ~30 LOC. |

### Generating-function infrastructure beyond Tier 2

| # | Item | Reference | Why Tier 3 |
|---|---|---|---|
| T3.G1 | **SpeciesAlgebra(F + G, F · G, F ∘ G)** | Joyal (1981); Bergeron-Labelle-Leroux | Combinatorial-species categorical framework. Encompasses EGF/OGF unifies Tier-2 series ops. ~250 LOC. |
| T3.G2 | **FormalPowerSeries (lazy/infinite)** | Concrete Mathematics §7 | Lazy infinite-series rep with on-demand coefficient extraction. ~150 LOC; complementary to T2.P6's truncated-series approach. |
| T3.G3 | **D-finite series identification (DEtools / Gfun)** | Salvy-Zimmermann | Identify holonomic recurrences; cross-package with `calculus/` ODEs. ~300 LOC + non-trivial. |

### Other Tier-3-scoped items

| # | Item | Reference | Why Tier 3 |
|---|---|---|---|
| T3.O1 | **ExponentialFormula** & **CycleLemma** & **TransferMatrix** | Stanley EC1 §4 | Three heavy-machinery enumeration tools; require T2.P + T3.G2. |
| T3.O2 | **Necklaces / Bracelets / Lyndon words** under cyclic rotation / dihedral action | Pólya; Sage `combinat.necklace` | ~80 LOC each + Pólya enumeration framework (T2.P4 cycle index). |
| T3.O3 | **AbacusModel for partitions** | James-Kerber | Bijection cores ↔ quotients ↔ abacus; needed by modular-rep-theory consumers. ~100 LOC. |
| T3.O4 | **CrystalGraph(λ)** for `gl_n` | Kashiwara | Crystal-basis structure on SSYT; ~150 LOC. Modern algebraic combinatorics. |

## Cross-package coordination

| Tier item | Requires from | What |
|---|---|---|
| T1.C9 (Bernoulli) | none | but `calculus/` Romberg/Euler-Maclaurin should call it |
| T1.C22-C25 (log-versions) | none | but `prob/` should switch its log-domain PMFs to use these |
| T2.P5 (poly ring) | possibly `linalg/` | unify with `linalg/poly` if such exists; else ship in `combinatorics/poly.go` |
| T2.Y8 (RSK) | none | but `prob/` longest-increasing-subsequence will want it |
| T3.G1-G5 (graph polys) | `graph/` | needs `graph.Graph` + edge-deletion/contraction primitives |
| T3.P1 (Poset) | none | new type; `linalg/` zeta-matrix would compose with T3.P2 |
| T3.S1 (SymFun) | T2.P5 | major architectural commit |

## What 036 already covered (not duplicated here)

- Numerical-contract gaps in the **existing** functions (Factorial 2^53 cliff, Fibonacci
  uint64 wrap, Catalan/Stirling/Bell float-DP losses, `*Exact` and `*Big` companions for
  the existing 10 fns). Those are **fix-the-existing-API** topics; this report is
  **add-missing-API** topics.
- Memory-bound generation (`GeneratePermutations` for n=15 OOM, `RandomSubset` allocates
  n-element pool). Tier 1 here adds the `NextPermutation`-style streaming counterparts
  (T1.G1-G5) and rank/unrank (T1.G6-G7) that fix that bound; the diagnostic was 036's,
  the missing-primitive that closes it is here.

## Web-research notes (Sage, FLINT, post-2020 canonical surface)

- **Sage `sage.combinat`** (≈600 named callables): Tier-1 items above are ~100% present
  in Sage; Tier-2 items are ~95% present (only `q-Catalan` at multiple non-equivalent
  definitions is sometimes user-confusing); Tier-3 is the SymmetricFunctions(QQ)
  infrastructure plus PermutationGroup-action and Posets module.
- **FLINT 3.x `arith` + `fmpz_combinat`** (≈90 callables): all Tier-1 counting (with
  `fmpz`/`fmpq` exact arithmetic), partition functions, Bernoulli/Euler, no structure
  generators (FLINT scope is exact arithmetic of values, not enumeration).
- **Post-2020 additions worth flagging:**
  - **Sage 9.x (2020-2023)** added the `cluster_algebra` module (Fomin-Zelevinsky cluster
    algebras), `path_tableau` (general tableau-path framework subsuming RSK), and
    `Q_t-Catalan numbers` from the diagonal-coinvariants story.
  - **OEIS itself** (Sloane et al.) is now considered the canonical reference for any
    integer-sequence-valued combinatorial function; reality should annotate every counting
    function with its A-number (current package: zero A-number annotations).
  - **`combinat-sf`** (Haskell) and **`Symmetrica`** (C, abandoned 2021) are the other
    open-source SymmetricFunctions implementations; reality could be the third.
  - **No new Tier-1 sequences** have entered the canonical-must-ship set since 2020;
    the foundational cohort (binomial, factorial, Catalan, Bell, Stirling, Bernoulli,
    Euler, Eulerian, Motzkin, Lucas, Fibonacci, Pell, Tribonacci, partitions) has been
    stable for >50 years.
  - The **single post-2020 algorithmic improvement** worth flagging is **fast Bernoulli
    via Akiyama-Tanigawa** vs the older recurrence-from-zeta approach — the 1999
    Akiyama-Tanigawa "saalschütz table" form is now the standard implementation in Sage,
    Mathematica, and Maple, and it is the form T1.C9 should ship.

## Naming / placement recommendations

- New file `combinatorics/sequences.go` for T1.C4-C21 (the named-sequence functions).
- New file `combinatorics/special.go` for T1.C1-C3, T1.C22-C25, T1.I1.
- New file `combinatorics/streaming.go` for T1.G1-G7.
- New file `combinatorics/enumerate.go` for T1.G8-G13.
- New sub-package `combinatorics/young/` for T2.Y* (separation of concerns; tableau code
  is structurally different from sequence code).
- New sub-package `combinatorics/path/` for T2.K*, T2.L* (lattice-path family).
- New sub-package `combinatorics/poly/` for T2.P5-P9 (polynomial ring + GF series ops).
- New file `combinatorics/qseries.go` for T2.N* (q-analogs).
- New sub-package `combinatorics/poset/` for all T3.P*.
- New sub-package `combinatorics/symfun/` for T3.S* (large enough to deserve its own).

## Summary table

| Tier | Item count | LOC estimate | Golden-vector estimate | Architectural impact |
|---|---|---|---|---|
| T1 (counting + simple generation) | ~46 | ~510 | ~1,400 (30/fn) | none — additive only |
| T2 (Young, Catalan, lattice paths, polys, q-series) | ~38 | ~1,200 | ~600 (varies; structures fewer vectors) | introduces `Partition`, `Tableau`, `Polynomial`, `Path` types |
| T3 (Poset, graph polys, SymFun, species) | ~26 | ~3,000+ | ~400 | introduces `Poset`, `SymmetricFunction`; cross-package coupling |
| **Total** | **~110 new items** | **~4,700 LOC** | **~2,400 vectors** | versus current 14 items / 497 LOC / 10 vectors |

That brings the package from ~7% canonical-coverage to ~75% (Tier 1+2+3). Tier 1 alone
takes coverage to ~30% and is **strictly additive** — no existing function changes
signature, every new function is golden-file-pinnable to OEIS, every one is ≤80 LOC, and
the entire cohort is one PR.

## File paths (existing surface enumerated)

- `C:\limitless\foundation\reality\combinatorics\counting.go` (10 functions: `Factorial`, `BinomialCoeff`, `Permutations`, `CatalanNumber`, `FibonacciNumber`, `StirlingFirst`, `StirlingSecond`, `BellNumber`, `IntegerPartitions`, `DerangementCount`)
- `C:\limitless\foundation\reality\combinatorics\generate.go` (4 functions: `GeneratePermutations`, `GenerateCombinations`, `NextPermutation`, `RandomSubset`)
- `C:\limitless\foundation\reality\combinatorics\combinatorics_test.go` (676 LOC)
- `C:\limitless\foundation\reality\combinatorics\testdata\combinatorics\binomial_coeff.json` (10 vectors, only golden file)
