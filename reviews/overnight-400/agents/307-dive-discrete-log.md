# 307 — dive-discrete-log (BSGS / Pollard-rho / Kangaroo / Pohlig-Hellman / Index Calculus audit)

## Headline
Reality v0.10.0 ships **zero DLP surface** — no BSGS, Pollard-rho-DL, kangaroo, Pohlig-Hellman, index calculus, MOV reduction, or any callable `DiscreteLog` function — and slot 307 LARGELY OVERLAPS slot 291 §T9-T12; recommend ABSORPTION into slot 291's `crypto/dlp/` sub-package as a single cohesive ~440-LOC Day-1 PR (BSGS + Pollard-rho-DL + Pohlig-Hellman, composing slot 291 T0 Montgomery + T3 Garner CRT + slot 306 IsPrime fix), with kangaroo and ECDLP-rho deferred to Tier 1, MOV reduction deferred behind slot 292 pairings, and index calculus / Joux 2014 deferred indefinitely as research frontiers without a concrete reality consumer.

## Findings (existing audit)

### State at HEAD (v0.10.0, 2026-05-09)
Repo-wide grep on `DiscreteLog|BSGS|Shanks|PollardRho|Kangaroo|PohligHellman|IndexCalculus|MOV|baby.step|giant.step|Pollard.*rho.*EC` against `*.go`: **zero callable hits in source files**. The 12 grep matches are all in `reviews/overnight-400/agents/*.md` (slot 057, 211, 213, 214, 291, 292, 293, 294, 297, 300, 200) and `MASTER_PLAN.md`. `crypto/` package (1,869 LOC across 6 files) has no DLP surface whatsoever:
| File | LOC | DLP-relevant content |
|---|---|---|
| `crypto/modular.go` | 135 | `ModPow` (used DOWNSTREAM by BSGS/rho/PH); no DLP |
| `crypto/prime.go` | 325 | `IsPrime` (slot 306 found correctness bug — fix is prerequisite for safe-prime detection used by PH) |
| `crypto/hash.go` | 223 | unrelated (consistent hash, Merkle) |
| `crypto/rng.go` | 197 | LCG/xorshift PRNG (used by Pollard-rho random walk init) |

### Cross-slot orientation

| Adjacent slot | Overlap with 307 | Resolution |
|---|---|---|
| **291-new-modular-arithmetic §T9-T12** | T9 BSGS ~80 LOC, T10 Pollard-rho-DL ~120 LOC, T11 Pohlig-Hellman ~120 LOC, T12 Index calculus ~280 LOC (deferred). Slot 291 already names ALL of slot 307's primary surface and proposed `crypto/discretelog_*.go` files. | **ABSORB**: slot 307 deepens slot 291's §T9-T12 with kangaroo (van Oorschot-Wiener 1996 parallelization), ECDLP-rho (slot 292 hand-off), MOV (slot 292 pairing hand-off), and Joux 2014 small-characteristic frontier. Single `crypto/dlp/` sub-package (or `crypto/discretelog_*.go` per slot 291's naming). |
| **292-new-elliptic-curves §T16-T18** | T16 Pollard-rho-EC + T17 MOV reduction + T18 Semaev summation-polys (frontier index calculus on EC). | Slot 307 owns the *generic* DLP algorithms; slot 292 owns the *EC instantiation*. ECDLP-rho is slot 292 T16 calling slot 307's rho cycle-detection generics. MOV is slot 292 T17 (needs pairing). |
| **306-dive-prime-tests** | Found correctness bug in `IsPrime` (separate audit). | Pohlig-Hellman REQUIRES correct primality + factorization of `p-1` to find safe-prime sub-orders. Slot 307 inherits slot 306 fix as a hard prerequisite. |
| **057-crypto-missing T1-FIELD** | Names DH key exchange; doesn't break DH. | DLP solver is offensive cryptanalysis tooling, complementary to defensive DH/ECDSA shape. |
| **214-new-pairings** | Pairings = the *bridge* MOV uses to reduce ECDLP → F_{p^k}× DLP. | Slot 307 T5 MOV reduction depends on slot 214 Weil/Tate pairing. |
| **293-new-ntt** | NTT is for FFT-over-F_p (polynomial multiply), not DLP. No overlap. | Independent. |
| **200-synergy-zkmark-info** | zkSNARK soundness ASSUMES DLP/ECDLP hardness; rarely needs runtime DLP solver. | Negative dependency: zkmark wants DLP to remain hard; runtime solving is irrelevant. |

### Algorithmic landscape (target surface)

Generic prime-field DLP `g^x ≡ h (mod p)`:
| Algorithm | Year | Time | Memory | Best for |
|---|---|---|---|---|
| Trial g^k loop | — | O(p) | O(1) | p ≤ 10^4 (regression sanity) |
| **Shanks BSGS** | 1971 | O(√p) | O(√p) | small p ≤ 2^32 (~10^5 hashtable entries) |
| **Pollard rho** | 1978 | O(√p) | O(1) | medium p ≤ 2^48 (memory-constrained) |
| **Pollard kangaroo** | 1978 / vOW 1996 | O(√(b−a)) | O(√(b−a)) | bounded-range x ∈ [a, b] (Bitcoin private-key recovery from leaked range) |
| **Pohlig-Hellman** | 1978 | O(Σ e_i √p_i) | as composed | smooth subgroup order n = ∏ p_i^e_i |
| **Index calculus** | Adleman 1979 / Coppersmith-Odlyzko-Schroeppel 1986 | L_p[1/2, c] sub-exp | large | prime-field DLP for large p (Logjam-class attacks) |
| **NFS-DLP** | Gordon 1993 / Joux-Lercier 2003 | L_p[1/3, c] sub-exp | large | very large prime fields ≥ 2^768 |
| **Joux quasi-polynomial** | 2014 | quasi-poly in log p | large | small-characteristic F_{p^n} DLP only — does NOT break prime-field DLP |

EC-DLP `[x]G = H` on E(F_p):
| Algorithm | Year | Applies to | Complexity |
|---|---|---|---|
| **Pollard rho on E** | Pollard 1978 + EC adaptation | any curve | O(√r), r = subgroup order (ONLY known generic ECDLP algorithm — best in class for non-degenerate curves) |
| **MOV reduction** | Menezes-Okamoto-Vanstone 1991/1993 | supersingular OR low embedding degree k ≤ 6 | reduces to F_{p^k}× DLP via Weil pairing → index calculus (sub-exp) |
| **Frey-Rück** | 1994 | low embedding degree | similar to MOV via Tate pairing |
| **Anomalous-curve / Smart-Satoh-Araki** | 1997 | curves with #E(F_p) = p (trace = 1) | polynomial-time — breaks the curve |
| **Semaev summation polynomials** | 2004 | EC index-calculus, frontier | sub-exp heuristic, not faster than rho in practice |

### Reality-specific consumers
1. **Cryptanalysis bench / forge** — verify weak DH groups in test vectors are actually breakable (Pohlig-Hellman demonstration that smooth p-1 is broken). Currently no concrete consumer.
2. **Diffie-Hellman parameter validation** — `IsSafePrime(p) = IsPrime(p) ∧ IsPrime((p-1)/2)`. Doesn't need DLP solver, just primality (slot 306).
3. **Bitcoin / blockchain research** — kangaroo for private-key recovery from leaked range (Pollard kangaroo recovers x ∈ [a, b] given pubkey, in O(√(b-a))). Specialized but real research utility.
4. **zkSNARK soundness analysis** — assumes ECDLP hard; no runtime DLP solver needed.
5. **Educational / pedagogical** — Reality's golden-file cross-language testing infrastructure (Go ↔ Python ↔ C++ ↔ C#) makes it a strong fit for "demonstrate the textbook algorithms with bit-exact reference values".

### MIT zero-dep moat

| Library | License | DLP coverage | Reality moat |
|---|---|---|---|
| Go `math/big` | BSD-3 | none (no DLP) | Reality fills the gap |
| SageMath | GPL | full (BSGS, rho, Pohlig-Hellman, index calculus, NFS-DL, Joux) | License moat (MIT) |
| PARI/GP | GPL | full | License moat |
| `mathnet/numerics` | MIT but no number-theory | none | — |
| Cloudflare CIRCL | BSD-3 | none (defensive crypto, not cryptanalysis) | Reality is the only MIT pure-Go DLP-solver candidate |

Reality's positioning: the MIT pure-Go zero-dep cross-language-deterministic-golden-file BSGS / Pollard-rho-DL / Pohlig-Hellman toolkit. Narrow research utility; not a runtime crypto path.

## Concrete recommendations

Tier numbering follows slot 291 §T9-T12 to enable absorption: T0 = BSGS (cheapest, most useful for small p), T1 = Pollard-rho-DL, T2 = kangaroo, T3 = Pohlig-Hellman, T4 = ECDLP-rho (composes slot 292), T5 = MOV reduction (composes slot 214), T6 = index calculus (frontier, defer), T7 = Joux 2014 small-characteristic (frontier, defer indefinitely).

### T0 — `crypto/dlp/bsgs.go` ~80 LOC — DAY-1 KEYSTONE
```go
// BSGS solves g^x ≡ h (mod p) for x ∈ [0, n) where n is the order of g (default p-1).
// Returns (x, true) if found, else (0, false). Time O(√n), memory O(√n).
func BSGS(g, h, p, n uint64) (x uint64, ok bool)
```
Algorithm: m = ⌈√n⌉; baby table {(j, g^j mod p) : j ∈ [0, m)} as `map[uint64]uint64`; giant step factor f = g^(-m) mod p; iterate γ = h, γ·f, γ·f², ... and look up each γ in baby table; if hit at giant index i with baby value j, then x = i·m + j. Uses slot 291 T0 Montgomery for the inner `g^j` and `f^i` chains (10x speedup over current Russian-peasant `mulmod`). Falls back to Floyd cycle-walk-and-store if memory-constrained.

**Composition**: slot 291 T0 Montgomery × T1 (this) × T3 Pohlig-Hellman as inner-loop sub-DLP solver.

Refs: Shanks 1971 *Class number, a theory of factorization, and genera*, Proc. Symp. Pure Math. 20:415-440.

### T1 — `crypto/dlp/pollard_rho.go` ~120 LOC — DAY-1 KEYSTONE
```go
func PollardRhoDL(g, h, p, n uint64) (x uint64, ok bool)
```
Algorithm: deterministic 3-partition `f(y) = g·y / h·y / y²` indexed by `y mod 3`; track exponent pairs (a_i, b_i) with y_i = g^{a_i} h^{b_i}; **Brent 1980** cycle detection (different schedule than Floyd, ~24% faster); when collision y_i = y_j found, solve a_i + b_i x ≡ a_j + b_j x (mod n) for x via `ModInverse(b_i - b_j, n)`. Time O(√n), memory O(1). Combine **van Oorschot-Wiener 1996** parallelization (distinguished-point method) for the multi-core case (~80 extra LOC; defer).

**Composition**: slot 291 T0 Montgomery for inner mulmod; reality's existing `crypto/modular.go:54 ModInverse` for the final exponent-recovery linear equation.

Refs: Pollard 1978 *Monte Carlo methods for index computation (mod p)*, Math. Comp. 32:918-924; Brent 1980 *An improved Monte Carlo factorization algorithm*, BIT 20:176-184; van Oorschot-Wiener 1996 *Improving implementable meet-in-the-middle attacks by orders of magnitude*, CRYPTO '96 LNCS 1109; Teske 1998 *Speeding up Pollard's rho method for computing discrete logarithms*, ANTS-III LNCS 1423 (the r-partition optimization with r ≥ 16).

### T2 — `crypto/dlp/kangaroo.go` ~120 LOC — Tier 1
```go
// Kangaroo solves g^x ≡ h (mod p) for x ∈ [a, b] (KNOWN range).
// Time O(√(b-a)), memory ~64 distinguished points.
func Kangaroo(g, h, p, a, b uint64) (x uint64, ok bool)
```
Algorithm: **Pollard 1978** lambda method — tame kangaroo from g^a hopping by pseudo-random jumps {1, 2, 4, ..., 2^k}; wild kangaroo from h hopping the same way; both deposit "distinguished points" (e.g. low 16 bits of point = 0) into a hashmap; first collision recovers x = a + (tame_dist - wild_dist). Optimal jump-set size k ≈ log₂√(b-a). **van Oorschot-Wiener 1996** parallelization scales linearly with cores.

**Practical use**: Bitcoin private-key recovery from a leaked range (Puzzle-130 challenge etc.); short-key DLP cryptanalysis.

Refs: Pollard 1978 §5 (lambda method); Pollard 2000 *Kangaroos, monopoly and discrete logarithms*, J. Cryptology 13:437-447 (the modern survey); van Oorschot-Wiener 1996.

### T3 — `crypto/dlp/pohlig_hellman.go` ~120 LOC — DAY-1 KEYSTONE
```go
// PohligHellman solves g^x ≡ h (mod p) given factorization of order n = ∏ p_i^{e_i}.
// Reduces to e_i sub-DLPs each in an order-p_i group, recombines via CRT.
func PohligHellman(g, h, p, n uint64, factors map[uint64]uint64) (x uint64, ok bool)
```
Algorithm: for each prime power p_i^{e_i} dividing n, project g and h into the order-p_i^{e_i} subgroup via exponentiation by n/p_i^{e_i}; solve x mod p_i^{e_i} by **lifting** (Pohlig-Hellman 1978 §III) — recover digits x = x_0 + x_1 p_i + ... + x_{e_i-1} p_i^{e_i-1} one at a time, each digit via inner BSGS or rho on order-p_i subgroup. Recombine all (x mod p_i^{e_i}) via reality's existing `ChineseRemainder` (or slot 291 T3 Garner CRT).

**Composition**: T0 BSGS or T1 rho as inner-loop solver (digit recovery in order-p_i group); slot 291 T3 Garner CRT for recombination; slot 291 T6 Pollard-rho factoring to obtain the `factors` map automatically (`func PohligHellmanAuto(g, h, p uint64) (uint64, bool)` wrapper).

**Pedagogical value**: demonstrates that DLP in Z/(p-1)× is broken when p-1 is smooth → motivates "safe primes" p = 2q+1 with q prime → drives Diffie-Hellman parameter validation (RFC 7919 / RFC 3526 named groups all use safe primes for this reason).

Refs: Pohlig-Hellman 1978 *An improved algorithm for computing logarithms over GF(p) and its cryptographic significance*, IEEE TIT 24(1):106-110.

### T4 — `crypto/dlp/ec_rho.go` ~150 LOC — Tier 1 (composes slot 292)
Pollard-rho on the EC group E(F_p). Same cycle-detection generic as T1 but with EC point addition replacing modular multiplication. Requires slot 292 T0 (curve type + point addition). The ONLY known generic ECDLP algorithm — its O(√r) complexity is the security guarantee for 256-bit cryptographic curves (r ≈ 2^256 → ~2^128 work, infeasible).

**Cryptanalysis benches**: weak curves with small r (e.g. test curves over F_p with p ~ 2^60), supersingular curves before MOV, anomalous curves where Smart-Satoh-Araki is faster.

Refs: Pollard 1978; Bernstein-Lange 2010 *Computing small discrete logarithms faster*, INDOCRYPT '12 LNCS 7668 (the modern ECDLP-rho with negation map and r-partition).

### T5 — `crypto/dlp/mov.go` ~80 LOC — Tier 2 (composes slot 214 pairings)
Menezes-Okamoto-Vanstone reduction: for supersingular E or E with small embedding degree k ≤ 6, transform ECDLP `[x]G = H` into multiplicative-group DLP `g^x = h` in F_{p^k}× via the Weil or Tate pairing, then attack F_{p^k}× DLP with index calculus (sub-exp). Demonstrates *why* cryptographic curves require large embedding degree (≥ 12 typical, ≥ 2^160 for security).

Requires slot 214 Miller-loop pairing primitive. Composes T6 index calculus (or imports from slot 291 §T12).

Refs: Menezes-Okamoto-Vanstone 1993 *Reducing elliptic curve logarithms to logarithms in a finite field*, IEEE TIT 39(5):1639-1646; Frey-Rück 1994.

### T6 — `crypto/dlp/index_calculus.go` ~500+ LOC — DEFER (frontier)
Index calculus for prime-field DLP: factor base of small primes B = {2, 3, 5, ..., p_k}; sieve random `g^r mod p` for B-smooth values; collect ≥ |B| relations of form `g^r_i ≡ ∏ p_j^{a_{ij}}`; solve linear system (a_{ij}) over Z/(p-1)Z to recover individual logs `log_g p_j`; final logarithm `log_g h` via "special-q" descent. Sub-exponential L_p[1/2]. Requires linear-algebra-mod-(p-1) (Lanczos or Wiedemann over Z/(p-1)Z; slot 291 NTT path orthogonal). 500+ LOC; document algorithm; ship `ErrNotImplementedV2` until a concrete consumer appears (forge cryptanalysis bench is the most likely first consumer).

Refs: Adleman 1979 *A subexponential algorithm for the discrete logarithm problem*, FOCS '79; Coppersmith-Odlyzko-Schroeppel 1986 *Discrete logarithms in GF(p)*, J. Cryptology 1:1-15; Joux-Lercier 2003 *Improvements to the general number field sieve for discrete logarithms in prime fields*, Math. Comp. 72:953-967; Adrian-Bhargavan-Durumeric et al. 2015 *Imperfect forward secrecy: how Diffie-Hellman fails in practice* (Logjam, CCS '15).

### T7 — Joux 2014 quasi-polynomial small-char DLP — DEFER INDEFINITELY
Joux 2014 *A new index calculus algorithm with complexity L(1/4 + o(1)) in small characteristic*, EUROCRYPT '14 LNCS 8441; Granger-Kleinjung-Zumbrägel 2014 *Breaking '128-bit secure' supersingular binary curves*, CRYPTO '14. Quasi-polynomial DLP in **F_{p^n} with small characteristic only** (does NOT touch prime-field DLP, does NOT break ECDLP on standard curves). Reality has no F_{2^n} or F_{3^n} consumer; defer indefinitely. Document as known frontier.

## Day-1 PR shape
**Singular cheapest, highest-immediate-value PR**: T0 BSGS + T1 Pollard-rho-DL + T3 Pohlig-Hellman. ~320 LOC, sub-package `crypto/dlp/`. Composes slot 291 T0 Montgomery + slot 291 T3 Garner CRT + slot 306-fixed `IsPrime` + reality's existing `ModPow` / `ModInverse` / `ChineseRemainder`. Ships the textbook DLP triad with bit-exact 3-way cross-validation. **Ship as part of slot 291 absorption**: a single coordinated PR titled "crypto: ship dlp/ sub-package — BSGS, Pollard-rho-DL, Pohlig-Hellman" that adds slot 291 §T9-T11 and slot 307 T0/T1/T3 simultaneously.

**Tier 1 follow-up**: T2 Kangaroo + T4 ECDLP-rho. ~270 LOC; depends on slot 292 EC point arithmetic landing first.

**Tier 2 follow-up**: T5 MOV reduction. ~80 LOC; depends on slot 214 pairings landing first.

**Defer indefinitely**: T6 index calculus, T7 Joux 2014.

## Cross-validation pin opportunities (R-MUTUAL-CROSS-VALIDATION 3/3)

1. **Brute-force ≡ BSGS for tiny p ≤ 1000** — regression test: for p ∈ {7, 11, 13, 17, 23, 31, 53, 97, 257, 521, 997}, generator g (smallest primitive root), random h ∈ [1, p): trial `g^k mod p` for k = 0..p-1 returns x_brute; BSGS returns x_bsgs; **assert x_brute == x_bsgs**. Cheap, exhaustive, catches off-by-one errors in BSGS bounds.

2. **BSGS ≡ Pollard-rho-DL ≡ Pohlig-Hellman for small smooth p** — for 100 random `(g, h, p)` tuples with p ∈ [10^6, 10^9] and p-1 = ∏ small primes (so PH applies), all three algorithms return the same x. Saturates R-MUTUAL-CROSS-VALIDATION 3/3 (three independent algorithmic paths agree bit-exactly). **Slot 291 §6 already names this pin**.

3. **`g^BSGS(g,h,p) ≡ h (mod p)` round-trip** — for 1,000 random valid inputs across 5 different primes (small, mid, 2^31-1 Mersenne, NIST DH group p_1024, NIST DH group p_2048-truncated to fit uint64), exponentiate the recovered x and verify equals h. Catches "off-by-multiple-of-order" bugs.

4. **Pohlig-Hellman ≡ Pollard-rho-DL on smooth-order subgroup** — for p with p-1 = q · m where q is prime and m is smooth: PH-on-full-group ≡ rho-on-full-group; PH should be ~Σ √p_i faster than rho's √(p-1).

5. **Kangaroo ≡ Pollard-rho-DL when range = full** — for p with x ∈ [0, p-1) i.e. unbounded: kangaroo's behavior should match rho asymptotically; for x ∈ [a, b] with b-a < √(p-1), kangaroo should be strictly faster (timing assertion, not bit-exact).

6. **MOV reduction round-trip** (post slot 214 pairing): for a supersingular curve E/F_p with embedding degree k=2: ECDLP-rho returns x_ec; MOV → F_{p^2}× DLP → index calculus returns x_mov; **x_ec == x_mov mod r**.

7. **Edge cases** mandatory: g = identity (BSGS returns 0 trivially), h = identity (BSGS returns 0), order(g) = 1, h not in ⟨g⟩ (return false), p = 2 (degenerate), p = 3 (smallest non-trivial), composite p (algorithms misbehave silently if not checked — assert IsPrime(p) at API boundary).

## Cross-cutting (consumers of slot 307)

- **Slot 291-new-modular-arithmetic §T9-T12** ← slot 307 IS the deepening of slot 291's DLP scope. Single shared `crypto/dlp/` sub-package.
- **Slot 292-new-elliptic-curves §T16** (ECDLP-rho) ← T4 (this) provides the generic rho cycle-detection used by EC instantiation.
- **Slot 292-new-elliptic-curves §T17** (MOV reduction) ← T5 (this) is the slot's MOV implementation.
- **Slot 214-new-pairings** ← T5 MOV reduction depends on Weil/Tate pairing.
- **Slot 306-dive-prime-tests** ← T3 Pohlig-Hellman REQUIRES correct primality (slot 306 fix is hard prerequisite for safe-prime detection).
- **Slot 057-crypto-missing** (DH key exchange) ← negative dependency: T3 demonstrates *why* DH parameter validation must use safe primes; doesn't break correct DH.
- **Forge cryptanalysis bench** (downstream consumer outside reality) ← T0+T1+T3 are useful research tools for verifying weak DH groups in test corpora.

## Priority assessment

DLP solver primitives are **research utility, not runtime crypto**. Reality's runtime consumers (slots 057 RSA/DH/ECDSA, 211 Kyber/Dilithium, 200 zkSNARK, 175 KZG) all *assume* DLP/ECDLP hard and never call a DLP solver at runtime. The legitimate consumers are:
1. Cryptanalysis benches (forge testing weak DH groups) — possible but no concrete consumer named.
2. Educational / pedagogical (golden-file demonstration of textbook DLP algorithms across 4 languages — fits reality's mission).
3. Bitcoin private-key recovery research (kangaroo on bounded range) — niche but real.

**Recommendation**: bundle slot 307 T0+T1+T3 with slot 291 §T9-T11 absorption as a single ~320-LOC coordinated PR, and **defer T2/T4/T5/T6/T7 until a concrete consumer appears** (avoid speculative API surface). Slot 291's existing recommendation already captures this; slot 307 adds the kangaroo + ECDLP-rho + MOV detail and the consumer-priority assessment.

## Sources

### Repo files
- `C:\limitless\foundation\reality\crypto\modular.go:1-135` (existing `ModPow`, `ModInverse`, `ChineseRemainder` — direct dependencies of all DLP algorithms below).
- `C:\limitless\foundation\reality\crypto\prime.go:26-176, 284-306` (Miller-Rabin used by safe-prime detection; `mulmod` used by inner exponentiation; **slot 306 found correctness bug — fix is prerequisite**).
- `C:\limitless\foundation\reality\crypto\rng.go:1-197` (PRNG used to seed Pollard-rho random walk and kangaroo distinguished-point flags).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\291-new-modular-arithmetic.md:195-222, 248` (slot 291 §T9-T12 + cross-validation pin #6 — slot 307 absorbs into this).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\292-new-elliptic-curves.md` (slot 292 §T16-T18 ECDLP-rho + MOV + Semaev — slot 307 T4/T5 hand-off).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\306-dive-prime-tests.md` (`IsPrime` correctness bug — prerequisite for safe-prime detection used by Pohlig-Hellman).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\214-new-pairings.md` (Weil/Tate pairings — prerequisite for T5 MOV reduction).
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md:327` (slot 307 line definition).

### Foundational papers
- Shanks 1971, *Class number, a theory of factorization, and genera*, Proc. Symp. Pure Math. 20:415-440 (BSGS).
- Pollard 1978, *Monte Carlo methods for index computation (mod p)*, Math. Comp. 32:918-924 (rho-DL **and** lambda/kangaroo, originally one paper).
- Pohlig-Hellman 1978, *An improved algorithm for computing logarithms over GF(p) and its cryptographic significance*, IEEE TIT 24(1):106-110.
- Brent 1980, *An improved Monte Carlo factorization algorithm*, BIT 20:176-184 (cycle-detection schedule, applies to rho-DL identically to rho-factor).
- van Oorschot-Wiener 1996, *Improving implementable meet-in-the-middle attacks by orders of magnitude*, CRYPTO '96 LNCS 1109 (parallelization of rho/kangaroo via distinguished points).
- Pollard 2000, *Kangaroos, monopoly and discrete logarithms*, J. Cryptology 13:437-447 (kangaroo modern survey).
- Teske 1998, *Speeding up Pollard's rho method for computing discrete logarithms*, ANTS-III LNCS 1423 (r-partition with r ≥ 16, ~20% speedup over 3-partition).

### Index calculus & sub-exponential DLP
- Adleman 1979, *A subexponential algorithm for the discrete logarithm problem*, FOCS '79 (original index calculus).
- Coppersmith-Odlyzko-Schroeppel 1986, *Discrete logarithms in GF(p)*, J. Cryptology 1:1-15.
- Pomerance 1987, *Fast, rigorous factorization and discrete logarithm algorithms*, in *Discrete Algorithms and Complexity*, Academic Press.
- Gordon 1993, *Discrete logarithms in GF(p) using the number field sieve*, SIAM J. Discrete Math. 6:124-138 (NFS-DL).
- Joux-Lercier 2003, *Improvements to the general number field sieve for discrete logarithms in prime fields*, Math. Comp. 72:953-967.
- Joux 2014, *A new index calculus algorithm with complexity L(1/4 + o(1)) in small characteristic*, EUROCRYPT '14 LNCS 8441 (quasi-polynomial F_{p^n} small-char only).
- Granger-Kleinjung-Zumbrägel 2014, *Breaking '128-bit secure' supersingular binary curves*, CRYPTO '14 LNCS 8617.

### ECDLP & MOV
- Menezes-Okamoto-Vanstone 1993, *Reducing elliptic curve logarithms to logarithms in a finite field*, IEEE TIT 39(5):1639-1646.
- Frey-Rück 1994, *A remark concerning m-divisibility and the discrete logarithm in the divisor class group of curves*, Math. Comp. 62:865-874.
- Smart 1999, *The discrete logarithm problem on elliptic curves of trace one*, J. Cryptology 12:193-196 (anomalous-curve attack).
- Satoh-Araki 1998, *Fermat quotients and the polynomial time discrete log algorithm for anomalous elliptic curves*, Comment. Math. Univ. Sancti Pauli 47:81-92.
- Semaev 2004, *Summation polynomials and the discrete logarithm problem on elliptic curves*, eprint.iacr.org/2004/031.
- Bernstein-Lange 2010, *Computing small discrete logarithms faster*, INDOCRYPT '12 LNCS 7668.

### Real-world DLP attacks
- Adrian-Bhargavan-Durumeric et al. 2015, *Imperfect forward secrecy: how Diffie-Hellman fails in practice* (Logjam), CCS '15 (1024-bit DH broken via NFS-DL precomputation).
- Pollard kangaroo for Bitcoin private-key recovery — Puzzle 130 challenge (J. Pollard 2000 framework, applied to secp256k1 with x ∈ [2^129, 2^130)).

### Reference textbooks
- Knuth TAOCP Vol. 2 *Seminumerical Algorithms* §4.5.4 (Pollard rho schema applies to both factoring and DLP).
- Hankerson-Menezes-Vanstone 2004 *Guide to Elliptic Curve Cryptography*, Springer §3.6 (ECDLP), §A.5 (MOV).
- Galbraith 2012 *Mathematics of Public Key Cryptography*, Cambridge UP §13-14 (DLP), §17 (ECDLP), §21 (index calculus).
- von zur Gathen-Gerhard 2013 *Modern Computer Algebra* 3rd ed., Cambridge UP.
