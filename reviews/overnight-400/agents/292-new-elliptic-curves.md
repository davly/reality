# 292 — new-elliptic-curves (SEA / CM / Isogenies / Velu / Miller / Pairings / BLS12-381)

## Headline
Reality v0.10.0 ships ZERO elliptic-curve surface (no `EllipticCurve`, `Point`, `JInvariant`, `Schoof`, `Velu`, `Isogeny`, `MillerLoop`, `TatePairing`, `WeilPairing`, `BLS12_381` anywhere in `*.go`); slot 057 documented the *basic* EC layer (Weierstrass / Edwards / Montgomery point arithmetic) but did not ship it; this slot owns the *number-theoretic* depth on top — and the singular highest-leverage primitive in the slot is **T13+T14 BLS12-381 + optimal-Ate pairing (~2,400 LOC bundled)** because zkmark/forge ZK-SNARK readiness collapses to one question: "does Reality ship a deterministic golden-file-validated optimal-Ate pairing on BLS12-381?" — and the singular cheapest day-1 PR is **T0 generic `EllipticCurve(F_p)` Weierstrass type + T1 j-invariant + T2 division polynomials ψ_n (~480 LOC)** which composes slot 057 (EC point arithmetic) and slot 291 (Tonelli-Shanks for compression decode) and unblocks every higher-tier algorithm in this slot.

## Findings

### State at HEAD (v0.10.0, 2026-05-09)

Repo-wide grep on `EllipticCurve|JInvariant|j-invariant|isogeny|Schoof|SEA|Velu|Atkin|Elkies|MOV|Frey-Rück|Semaev|MillerLoop|MillerAlgorithm|TatePairing|WeilPairing|OptimalAte|ModularPolynomial|EndomorphismRing|ComplexMultiplication|HilbertClassPolynomial|BLS12|BN254|secp256k1|P256|P384|Curve25519|Edwards|Montgomery.*curve|division.*polynomial|ECDLP|ECDSA|ECIES`: **zero callable hits in any source file**. All matches are in review documents (`reviews/overnight-400/agents/057-crypto-missing.md`, `211-new-lattice-crypto.md`, `290-new-galois-theory.md`, `291-new-modular-arithmetic.md`).

| Surface | Path | Status | Slot-292 relevance |
|---|---|---|---|
| `crypto/modular.go` ModPow / ModInverse / CRT (uint64) | `crypto/modular.go:20-135` | textbook uint64 | EC over F_p needs bigint; this surface caps at p < 2^32 securely |
| `crypto/prime.go` Miller-Rabin / mulmod / addmod | `crypto/prime.go:26-306` | uint64-only | Used to verify prime moduli of curves; otherwise unused for EC math |
| `geometry/quaternion.go` (presumed) | `geometry/` | quaternions for SO(3) | NOT applicable — different algebraic structure (skew field H ≠ EC group) |
| Big-int finite field Fp / Fp2 / Fp4 / Fp6 / Fp12 | absent | — | T1-FIELDTOWER prerequisite (slot 057). **Without Fp12, no pairing ships.** |
| Generic EC point group (Weierstrass / Edwards / Montgomery) | absent | — | T1-EC prerequisite (slot 057). **Without EC points, no curve algorithm ships.** |
| Tonelli-Shanks sqrt-mod-p | absent (slot 291 T1) | — | Required for compression decode + division-polynomial roots |
| Montgomery / Barrett reduction | absent (slot 291 T0) | — | EC scalar mul inner loop; pairing Miller-loop inner loop |
| NTT | absent (slot 291 T4) | — | Velu's formulae for large-degree isogenies use polynomial multiplication mod ℓ |
| Hilbert class polynomial database | absent | — | T8 prerequisite for Atkin-Morain CM |
| Modular polynomial database Φ_ℓ | absent | — | T5 prerequisite for SEA Elkies-prime branch |

### Cross-slot orientation

| Adjacent slot | Overlap | Resolution |
|---|---|---|
| **057-crypto-missing T1-EC + T1-FIELDTOWER + T1-BIGINT** | Names Weierstrass / Edwards / Montgomery point group + Fp12 tower as PREREQUISITE for any pairing scheme. Doesn't itself ship them; defers all curves to follow-up. | Slot 057 is the *substrate* layer (point arithmetic, scalar mul via Montgomery ladder, compression). Slot 292 is the *number-theoretic depth*: point counting, isogenies, pairings, CM curve construction. **Slot 292 consumes slot 057's `EllipticCurve` and `Fp12`** — must be co-shipped or 057 ships first. |
| **058-crypto-sota** | Constant-time discipline. | Pairings need Miller-loop CT only when input scalars are secret (BLS signing key). Public-input pairings (verifier side) can use variable-time Miller-loop. |
| **211-new-lattice-crypto** | Disjoint algebra (lattice → polynomial rings R_q = Z_q[x]/(x^n+1), not EC group law). | NTT sharing is the only intersection. Slot 292 *imports* slot 211's NTT iff Velu degree-ℓ isogeny inner kernel polynomial multiplication exceeds ~ℓ=64. |
| **290-new-galois-theory** | Galois theory is the abstract substrate of *both* finite fields and EC endomorphism rings (Frobenius is a Galois generator of F_q over F_p; CM endomorphism ring is an order in an imaginary quadratic field — Galois objects). | No code overlap. Slot 290 is *abstract* (groups, splitting fields, polynomial factoring); slot 292 is *concrete* (specific curve, specific prime, specific pairing). Slot 292 may consume slot 290's Berlekamp factoring to factor division polynomials over F_p. |
| **291-new-modular-arithmetic** | KEYSTONE consumer. Tonelli-Shanks (T1), Montgomery (T0), Barrett (T0), Pollard rho EC version (T6 + T16-here), ECM factoring (T7 + T15-here). | Slot 291 ships *generic* uint64 / bigint modular substrate; slot 292 ships *EC-specific* curve construction + counting + pairings on top. **ECM factoring (slot 291 T7) re-uses slot 292 T0 EC point arithmetic** — bidirectional dependency, recommend co-shipping ECM as a slot-291/292 boundary. |
| **zkmark / forge** | Halo2 / Plonk / FRI all need a pairing-friendly curve (BLS12-381 or BN254 are the only realistic options for MIT pure-Go). Halo2 specifically uses Pasta (Pallas / Vesta) — non-pairing, recursion-friendly. | **Decision needed**: zkmark v1 → BLS12-381 + optimal Ate pairing (Plonk/Marlin verifier); zkmark v2 → Pasta (Halo2 recursion). Slot 292 SHOULD ship BLS12-381 first (universal SNARK verifier surface) and Pasta as a follow-up T13'. |
| **Pistachio** (downstream consumer per CLAUDE.md) | 60 FPS rendering pipeline; signature verification on stream metadata. | Verification uses pairing on small inputs; latency budget is ~10ms — optimal Ate on BLS12-381 fits in ~1.5ms, well under budget. |

### Distinct-from-slot-057 disambiguation

Slot 057 owns:
- `EllipticCurve` type (Weierstrass / Edwards / Montgomery models, point group Add / Double / Negate / Scalar-mul).
- Fp / Fp2 / Fp6 / Fp12 tower fields.
- Compression / decompression (consumes slot 291 Tonelli-Shanks).
- ECDSA / EdDSA / X25519 *signature* algorithms (these are sit-on-top *protocols*, not pure math primitives — borderline per slot 057's scope filter).

Slot 292 owns:
- Point counting algorithms (Schoof 1985, SEA = Schoof+Elkies+Atkin 1991-1998).
- Isogenies (Velu 1971 formulae, Costello-Hisil 2017 even-degree variant).
- Pairings (Miller 1986 algorithm; Tate / Weil / optimal Ate).
- Complex multiplication theory (Atkin-Morain 1990 CM method, j-invariant + Hilbert class polynomial).
- Endomorphism ring computation, modular polynomial Φ_ℓ database.
- ECDLP attacks (MOV reduction, Pollard rho on EC, Frey-Rück, Semaev).
- Curve-specific instantiations: BLS12-381, BN254, secp256k1 (curves with hardcoded params + optimized Frobenius).

Boundary: **j-invariant and division polynomial ψ_n are slot-292** — they are number-theoretic invariants of a curve, not point operations. Compression decode is **slot-057** — it's a point operation parameterized by a curve. The CM-method curve *generator* is slot-292 (number theory); the resulting curve's point arithmetic is slot-057 (group law).

### Does math/big or stdlib cover this?

`crypto/elliptic` (stdlib) ships P-224 / P-256 / P-384 / P-521 and a generic Weierstrass — but: hardcoded curves only, no Schoof / SEA, no pairings, no isogenies, no CM. `golang.org/x/crypto/bn256` ships BN254 pairing — BSD-3, but Go-only and no cross-language golden parity. **There is no MIT pure-Go zero-dep deterministic-golden-file-validated SEA / Velu / Miller-on-BLS12-381 implementation in any language ecosystem.** The closest:

| Library | License | Coverage | Reality moat |
|---|---|---|---|
| `crypto/elliptic` (Go stdlib) | BSD-3 | NIST P-curves point arithmetic only | No SEA, no pairings, no isogenies. Reality fills. |
| `golang.org/x/crypto/bn256` | BSD-3 | BN254 pairing | Go-only, no cross-language golden. Reality adds BLS12-381 (zkSNARK standard) + parity. |
| Cloudflare CIRCL | BSD-3 | BLS12-381, BN254, X25519, SIDH, CSIDH (deprecated 2022 due to attack), kyber | Go-only, no Python/C++/C# golden contract. Reality is the cross-language layer. |
| `gnark-crypto` (Consensys) | Apache-2 | BLS12-381 + BN254 pairings, gnark zkSNARK frontend | Apache-2 not MIT; coupled to gnark frontend; Go-only. |
| `blst` (Supranational) | Apache-2 | BLS12-381 only, hand-tuned ASM | Apache-2; ASM-coupled; not portable. |
| Pari/GP | GPL-2 | Full SEA, CM, modular polynomials, all algorithms | GPL — not MIT-compatible for MIT downstream. |
| Magma | proprietary | full | Closed source. |
| SageMath | GPL-3 | Full SEA via Pari, CM via Magma-compatible | GPL — not MIT-compatible. |
| NTL | LGPL-2.1 | Subset (Schoof, factoring) | LGPL not MIT. |
| MIRACL Core | Apache-2 | BLS12-381 + BN254 | Apache-2; multi-language but each language is hand-written, no shared golden. |

**Reality's positioning**: the MIT pure-Go zero-dep cross-language deterministic-golden-file-validated EC number-theory toolkit (SEA, CM, Velu, BLS12-381 optimal Ate). Narrow but real moat — no other library has all of {MIT, pure-Go, zero-dep, cross-language golden}.

## Concrete recommendations

Tier numbering: T0 = generic curve type; T1-T2 = curve invariants + division polynomials; T3 = Schoof (textbook); T4-T5 = isogeny machinery (Velu + modular polys); T6-T7 = SEA + endomorphism rings; T8-T9 = CM curve construction; T10-T14 = pairings (Miller, Tate, Weil, optimal Ate, BLS12-381); T15-T18 = ECM factoring + ECDLP attacks (MOV, rho, Frey-Rück, Semaev frontier).

### T0 — `ec/curve.go` ~220 LOC — DAY-1 KEYSTONE — composes slot 057
```go
type Weierstrass struct{ A, B *big.Int; P *big.Int; N *big.Int; H uint64 }  // y² = x³ + ax + b mod p, order N, cofactor h
type Point struct{ X, Y, Z *big.Int }                                       // Jacobian projective
func NewCurve(a, b, p, n *big.Int, h uint64) (*Weierstrass, error)
func (c *Weierstrass) IsOnCurve(P *Point) bool
func (c *Weierstrass) Add(P, Q *Point) *Point
func (c *Weierstrass) Double(P *Point) *Point
func (c *Weierstrass) ScalarMul(k *big.Int, P *Point) *Point   // window-w fixed-base
func (c *Weierstrass) Discriminant() *big.Int                  // Δ = -16(4a³ + 27b²)
```
Sits underneath every higher tier. **If slot 057 ships its own EC type first, slot 292 imports it; otherwise slot 292 ships the canonical type and slot 057 layers signature schemes on top.**

### T1 — `ec/jinvariant.go` ~80 LOC — DAY-1 KEYSTONE
```go
func (c *Weierstrass) JInvariant() *big.Int        // j = 1728 · 4a³ / (4a³ + 27b²) mod p
func (c *Weierstrass) IsIsomorphicTo(c2 *Weierstrass) bool   // iff j(c) == j(c2)
```
The j-invariant classifies elliptic curves over F_p̄ up to isomorphism — same j ⇒ isomorphic over algebraic closure. Pin: random isomorphism `(x,y) → (u²x, u³y)` with u random in F_p×, recompute j on transformed curve, assert byte-identical for n=10⁴ trials × 5 primes.

### T2 — `ec/division_polynomial.go` ~180 LOC
**Division polynomial ψ_n(x, y)** — degree n²-1 polynomial in F_p[x,y] / curve, zero exactly at the n-torsion points (excluding identity). Recurrence: ψ_1 = 1, ψ_2 = 2y, ψ_3 = 3x⁴ + 6ax² + 12bx − a², ψ_4 = 4y(x⁶ + 5ax⁴ + 20bx³ − 5a²x² − 4abx − 8b² − a³); odd ψ_{2m+1} = ψ_{m+2}ψ_m³ − ψ_{m-1}ψ_{m+1}³, even 2y ψ_{2m} = ψ_m(ψ_{m+2}ψ_{m-1}² − ψ_{m-2}ψ_{m+1}²). Composes T0 + bigint polynomial arithmetic. Used by T3 Schoof, T4 Velu, T6 SEA, T16 EC-DL.

### T3 — `ec/schoof.go` ~600 LOC
**Schoof 1985** — point count #E(F_p) deterministic polynomial-time algorithm. By Hasse: t = p + 1 − #E(F_p), |t| ≤ 2√p. For each prime ℓ ≤ L (with ∏ℓ > 4√p), compute t mod ℓ by working in the ℓ-torsion subring F_p[x,y]/ψ_ℓ(x): the Frobenius (x,y) → (x^p, y^p) acts on E[ℓ] satisfying φ² − tφ + p = 0; identify t mod ℓ via comparison of φ²(P) + p_ℓ P with t_ℓ φ(P) for each candidate t_ℓ ∈ {0..ℓ−1}. CRT recombine the t mod ℓ list. **O(log⁸ p)** baseline. ~600 LOC includes division-polynomial arithmetic mod ψ_ℓ, Frobenius computation, CRT recombination.

**What it unblocks**: deterministic curve-suitability check (does this random curve have prime order? Is the cofactor 1?). Pre-cryptographic curves used trial-and-error Pohlig-Hellman to detect bad orders; Schoof certifies deterministically.

Refs: Schoof 1985 *Elliptic curves over finite fields and the computation of square roots mod p* Math. Comp. 44(170):483-494.

### T4 — `ec/velu.go` ~280 LOC
**Velu 1971 formulae** — given a kernel subgroup K ⊂ E (typically cyclic of order ℓ, generated by a torsion point), construct the quotient curve E/K = E' and the isogeny φ: E → E' explicitly as a rational map (x, y) → (X(x,y), Y(x,y)). For odd ℓ: X(P) = x(P) + Σ_{Q ∈ K\{0}} (x(P+Q) − x(Q)). Costello-Hisil 2017 √élu variant runs in O(√ℓ) instead of O(ℓ).

**What it unblocks**: SIDH / SIKE (broken 2022 by Castryck-Decru, but the Velu primitive lives on); **CSIDH** post-quantum key exchange (Castryck-Lange-Martindale-Panny-Renes 2018, still secure); SQIsign post-quantum signatures (deBoer-De Feo-Leroux-Wesolowski 2020); zkSNARK isogeny-based commitments.

Refs: Velu 1971 *Isogénies entre courbes elliptiques* C.R. Acad. Sci. Paris Ser. A-B 273; Costello-Hisil 2017 *A simple and compact algorithm for SIDH with arbitrary degree isogenies*; Bernstein-De Feo-Leroux-Smith 2020 *Faster computation of isogenies of large prime degree* ANTS-XIV (the √élu paper).

### T5 — `ec/modular_poly.go` ~140 LOC + ~10-50 KB precomputed data
Modular polynomial Φ_ℓ(X, Y) ∈ Z[X, Y] satisfies Φ_ℓ(j(E), j(E')) = 0 iff E and E' are ℓ-isogenous. Symmetric: Φ_ℓ(X, Y) = Φ_ℓ(Y, X). For ℓ ≤ 100 the polynomials are precomputed (Sutherland 2011 has tables for ℓ ≤ 10000). Ship a precomputed table for ℓ ∈ {2, 3, 5, 7, 11, 13, ..., 97} as a `data/modular_phi_l.go` golden file.

**What it unblocks**: T6 SEA Elkies-prime branch — Elkies 1990 observed that Φ_ℓ(X, j(E)) factors over F_p iff the Frobenius eigenvalue λ ∈ F_p (not just F_{p²}); when ℓ is Elkies, the polynomial-arithmetic work in T3 collapses from O(ℓ²) to O(ℓ).

Refs: Sutherland 2011 *Computing Hilbert class polynomials with the Chinese Remainder Theorem* Math. Comp. 80; Atkin database via PARI/GP `polmodular`.

### T6 — `ec/sea.go` ~1500 LOC
**Schoof-Elkies-Atkin 1991-1998** — practical point counting on cryptographically sized curves (256-bit primes routinely; 512-bit feasible). Branches per ℓ:
- **Elkies prime** (Φ_ℓ(X, j) splits in F_p[X] into linear factors): work in F_p[X]/f_ℓ where deg f_ℓ = (ℓ-1)/2 instead of (ℓ²-1)/2. Compute t mod ℓ in O(ℓ) ops.
- **Atkin prime** (Φ_ℓ does not split linearly): cannot pin t mod ℓ exactly, only narrow to a small set of candidates. Match-and-sort / BSGS final step combines Atkin candidate sets.

**What it unblocks**: cryptographic curve generation. Take random curve, run SEA, check #E is prime (or has cofactor 1, 2, 4, 8) — if not, discard and resample. **This is the algorithm by which secp256k1, NIST P-256, BLS12-381 were constructed and certified.**

Refs: Schoof 1995 *Counting points on elliptic curves over finite fields* J. Théor. Nombres Bordeaux 7; Atkin 1991 *The number of points on an elliptic curve modulo a prime* (manuscript, never published); Elkies 1998 *Elliptic and modular curves over finite fields and related computational issues*; Lercier-Morain 1995 *Counting the number of points on elliptic curves over finite fields: strategies and performances* EUROCRYPT '95.

### T7 — `ec/endomorphism.go` ~220 LOC
Compute End(E) over F_p̄. For a curve E over F_p, the Frobenius π_p satisfies π² − tπ + p = 0. End(E) is either:
- An order in an imaginary quadratic field Q(√(t² − 4p)) (ordinary curve, generic case).
- An order in a quaternion algebra (supersingular curve, t ≡ 0 mod p).

Distinguish via the trace t (from T3/T6); supersingular iff t ≡ 0 (mod p). For ordinary E, compute the conductor of the order End(E) inside the maximal order of Q(√D).

**What it unblocks**: T9 CM construction, T17 MOV reduction (which exploits supersingular curves), pairing-friendly curve detection (low embedding degree is a marker).

Refs: Kohel 1996 *Endomorphism rings of elliptic curves over finite fields* PhD thesis, UC Berkeley.

### T8 — `ec/hilbert_class.go` ~280 LOC + ~5-20 KB precomputed data
**Hilbert class polynomial H_D(X) ∈ Z[X]** — minimal polynomial whose roots are j-invariants of elliptic curves with CM by the order of discriminant D in Q(√D). Computed via either complex-analytic floating-point (Watkins 2004) or CRT mod many primes (Sutherland 2011, polynomial-time). Ship precomputed H_D for the small-class-number discriminants used by Atkin-Morain (D ∈ {-3, -4, -7, -8, -11, -19, -43, -67, -163} have class number 1; ~40 discriminants give class number ≤ 4).

### T9 — `ec/cm_atkin_morain.go` ~340 LOC
**Atkin-Morain 1990/1993** — CM method for curve construction with prescribed order. Algorithm: pick a CM discriminant D; find prime p such that 4p = a² + |D|b² has a solution (use Cornacchia 1908 algorithm); compute the Hilbert class polynomial H_D mod p (T8); take a root j_0 of H_D mod p; construct curve E: y² = x³ − 3kx + 2k where k = j_0/(j_0 − 1728) mod p. Then #E(F_p) = p + 1 ± a (sign determined by twist).

**What it unblocks**: pairing-friendly curve construction with prescribed embedding degree. BLS12-381 itself was discovered by a CM-like search (Barreto-Lynn-Scott 2002, Bowe 2017 final parameter pin).

Refs: Atkin-Morain 1993 *Elliptic curves and primality proving* Math. Comp. 61(203):29-68; Cornacchia 1908 *Su di un metodo per la risoluzione in numeri interi dell'equazione Σ_{h=0}^n c_h x^{n-h} y^h = P*; Bröker-Stevenhagen 2007 *Constructing elliptic curves of prime order*.

### T10 — `ec/pairing/miller.go` ~260 LOC
**Miller 1986 algorithm** — workhorse for all pairings. Compute f_{r, P}(Q) — a function whose divisor is r[P] − [rP] − (r − 1)[O] — by double-and-add over the binary expansion of r, accumulating line and tangent functions. ~160 LOC for the main loop + ~100 LOC for line-function evaluation in projective coordinates.

```go
type Pairing interface{ Compute(P, Q *Point) *Fp12Element; FinalExponent(f *Fp12Element) *Fp12Element }
func MillerLoop(P, Q *Point, r *big.Int, c *Curve) *Fp12Element
```

Refs: Miller 1986/2004 *The Weil pairing, and its efficient calculation* J. Cryptology 17(4):235-261 (the 2004 published version of the 1986 IBM technical memo).

### T11 — `ec/pairing/tate.go` ~120 LOC
**Tate pairing** t_r(P, Q) = f_{r,P}(Q)^((p^k − 1)/r) where k is embedding degree. Composes T10. Final exponentiation kills the cofactor. For pairing-friendly curves with embedding degree k ≥ 6, factor (p^k − 1)/r = (p^(k/2) − 1)(p^(k/2) + 1)/r and use Frobenius-twist tricks.

### T12 — `ec/pairing/weil.go` ~80 LOC
**Weil pairing** w_r(P, Q) = f_{r,P}(Q) / f_{r,Q}(P). Composes T10 (twice). Symmetric and bilinear; |w_r(P, Q)| = 1. Slower than Tate for cryptographic use but mathematically canonical.

### T13 — `ec/pairing/optimal_ate.go` ~340 LOC
**Optimal Ate pairing** Vercauteren 2010 — optimal-length Miller loop for pairing-friendly curves with non-trivial Galois automorphisms. For BLS12-381: loop length is the parameter `u = -0xd201000000010000` (~63 bits) instead of the full r (~255 bits) — 4x speedup over naive Tate. Hess-Smart-Vercauteren 2006 Ate pairing is the predecessor; optimal Ate is its modern descendant.

Refs: Vercauteren 2010 *Optimal pairings* IEEE TIT 56(1):455-461; Hess-Smart-Vercauteren 2006 *The Eta pairing revisited* IEEE TIT 52(10):4595-4602.

### T14 — `ec/curves/bls12_381.go` ~320 LOC (composes T0 + T13) — **HIGHEST-LEVERAGE NEW PRIMITIVE**
Hardcoded BLS12-381 curve parameters + optimized Frobenius + final exponentiation. Constants:
- `p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab` (381-bit base prime).
- `r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001` (255-bit subgroup prime, also the BLS12-381 scalar field).
- Embedding degree k = 12, so target group is in F_{p^12}.
- u = -0xd201000000010000 (63-bit BLS parameter generating the curve).
- G1 generator (x, y) ∈ E(F_p), G2 generator (x, y) ∈ E'(F_{p²}) (sextic twist).

```go
var BLS12_381 *Weierstrass    // y² = x³ + 4 over F_p
var BLS12_381_G2 *Weierstrass // sextic-twist y² = x³ + 4(1+i) over F_{p²}
func BLS12_381_Pairing(P *Point, Q *PointFp2) *Fp12Element  // optimal Ate
```

**Composes T13 + T0 + Fp12 (slot 057 T1-FIELDTOWER) + Tonelli-Shanks (slot 291 T1).**

**What it unblocks (ENORMOUS)**:
- **zkmark Plonk / Marlin / Groth16 / Halo verifier** — every pairing-based SNARK uses BLS12-381 e: G1 × G2 → G_T.
- **BLS aggregate signatures** (Boneh-Lynn-Shacham 2001, Boneh-Drijvers-Neven 2018 BLS aggregation) — Ethereum 2.0 consensus, Filecoin, Chia.
- **KZG polynomial commitments** (Kate-Zaverucha-Goldberg 2010) — used in Plonk, Marlin, Halo, EIP-4844 (Ethereum proto-danksharding).
- **VRF** (Galindo-Liu-Ortiz-Yang 2021 "BLS-VRF") — pseudo-random functions with public verifiability.

Refs: Bowe 2017 *BLS12-381: New zk-SNARK Elliptic Curve Construction* (the canonical parameter pin); Barreto-Lynn-Scott 2002 *Constructing elliptic curves with prescribed embedding degrees* SCN 2002; Bos-Costello-Hisil-Lauter 2013 *High-performance scalar multiplication using 8-dimensional GLV/GLS decomposition*; ZCash protocol specification §5.4.9.5; IETF draft-irtf-cfrg-pairing-friendly-curves-11.

### T14' — `ec/curves/bn254.go` ~280 LOC (alternative pairing curve, composes T13)
BN254 (Barreto-Naehrig 254) — Ethereum precompile alt_bn128 (EIP-196/197/1108). Embedding degree k = 12; 254-bit base prime. **Less secure than BLS12-381** (Kim-Barbulescu 2016 special-TNFS attack reduces BN254 security from 128 to ~110 bits) but ubiquitous due to Ethereum mainnet precompile. Ship for Ethereum interop; recommend BLS12-381 for new constructions.

### T14'' — `ec/curves/pasta.go` (~280 LOC, defer to follow-up)
Pallas + Vesta cycle of curves (Hopwood 2020 for Halo2). NOT pairing-friendly; 2-cycle for recursion. Defer until Halo2 consumer materializes (likely zkmark v2).

### T15 — `ec/factor/ecm.go` ~280 LOC (composes slot 291 + T0)
**Lenstra 1987 ECM** — composes slot 291 T7 (Pollard p-1 stage logic) with slot 292 T0 EC point arithmetic. Already enumerated in slot 291's review; the *EC point arithmetic over Z/nZ* is the slot-292 contribution. **Co-ship with slot 291 ECM driver.**

### T16 — `ec/dlog/pollard_rho.go` ~140 LOC
Pollard rho on E(F_p) discrete log. Same cycle-detection idea as multiplicative-group rho but over the EC group. Combine with **Pollard kangaroo** (Pollard 1978) for the bounded-interval case. Time O(√r) where r is the subgroup order; useless against well-chosen 256-bit curves, useful for cryptanalysis benches against weak curves.

Refs: Pollard 1978 *Monte Carlo methods for index computation (mod p)* Math. Comp. 32; van Oorschot-Wiener 1999 *Parallel collision search with cryptanalytic applications* J. Cryptology 12.

### T17 — `ec/dlog/mov.go` ~180 LOC
**Menezes-Okamoto-Vanstone 1991/1993 reduction** — for supersingular curves OR curves with low embedding degree k, ECDLP on E(F_p) reduces to DLP in F_{p^k}× via the Weil pairing. If k is small (k ≤ 6 for cryptographic insecurity), F_{p^k}× DLP is sub-exponential via NFS (T12 in slot 291) and the curve is broken.

**What it unblocks**: cryptanalytic certification — given a candidate curve, compute embedding degree via T7 endomorphism analysis + Frey-Rück 1994 generalization; reject curves with k ≤ 6 (or k ≤ 12 for paranoid security); accept curves with k ≥ 50.

Refs: Menezes-Okamoto-Vanstone 1993 *Reducing elliptic curve logarithms to logarithms in a finite field* IEEE TIT 39(5):1639-1646; Frey-Rück 1994 *A remark concerning m-divisibility and the discrete logarithm in the divisor class group of curves* Math. Comp. 62(206):865-874.

### T18 — `ec/dlog/semaev.go` (FRONTIER — DEFER)
**Semaev 2004** summation polynomials approach to ECDLP. Index-calculus on EC; sub-exponential heuristic complexity but practically slower than Pollard rho. Frontier research; no production use case. Document as a known frontier; revisit if EC index calculus ever becomes practical (currently it does not — Pollard rho remains the best known generic ECDLP algorithm, hence the security of cryptographic EC).

Refs: Semaev 2004 *Summation polynomials and the discrete logarithm problem on elliptic curves*; Diem 2011 *On the discrete logarithm problem in elliptic curves* Comp. Math. 147.

## Day-1 PR shape

**Singular cheapest, highest-immediate-value PR**: T0 (`EllipticCurve(F_p)` Weierstrass type) + T1 (j-invariant) + T2 (division polynomials ψ_n). ~480 LOC. Composes slot 057 (if shipped) or stands alone (provides the shared substrate slot 057 needs anyway). Unblocks T3 Schoof. No new dependencies beyond `math/big`.

**Singular highest-strategic-value follow-up**: T13 + T14 (optimal Ate pairing + BLS12-381 hardcoded curve). ~660 LOC bundled. Single primitive that unblocks zkmark Plonk/Marlin/Groth16, BLS aggregate signatures, KZG commitments, EIP-4844-compatible verifiers. Depends on slot 057 T1-FIELDTOWER (Fp12) and slot 291 T0 (Montgomery) + T1 (Tonelli-Shanks). **Without T13+T14, zkmark cannot ship a real Plonk verifier.**

## Cross-validation pin opportunities (R-MUTUAL-CROSS-VALIDATION 3/3)

1. **Schoof T3 ≡ SEA T6 ≡ naive O(p) baby-step on small p** — for 1000 random primes p ∈ [2^16, 2^24] and random curves over F_p: all three independent point-counting methods return identical #E(F_p). T3 and T6 use fundamentally different polynomial-arithmetic substrates (T3 works mod ψ_ℓ for all ℓ, T6 splits into Elkies/Atkin branches); a baseline brute-force baby-step over the (small-p) finite group gives the third independent witness. Saturates 3/3.
2. **j-invariant T1 isomorphism-invariance** — random transformation `(x, y) → (u²x + r, u³y + su²x + t)` for random (u, r, s, t) ∈ F_p×: T1 j-invariant returns byte-identical bigint for n=10⁴ random transforms × 5 primes (2³¹-1, 2⁶¹-1, BN254 base, BLS12-381 base, secp256k1 base). Saturates 3/3 (3 independent prime moduli × isomorphism family is a 3-way orthogonal pin).
3. **BLS12-381 pairing T14 bilinearity** — `e(aP, bQ) == e(P, Q)^(ab)` for 1000 random (a, b, P, Q); `e(P + R, Q) == e(P, Q) · e(R, Q)`; `e(P, Q + S) == e(P, Q) · e(P, S)`. Three independent bilinearity identities, each tested 1000×. Saturates 3/3.
4. **BLS12-381 pairing T14 non-degeneracy** — `e(P, Q) ≠ 1_{F_{p^12}}` for any non-zero P, Q. Round-trip identity `e(P, -Q) · e(P, Q) = 1` for 1000 random pairs. Independent third witness: `e(P, Q)^r = 1` where r is the subgroup order. Saturates 3/3.
5. **Velu T4 isogeny preserves pairing degree** — for random ℓ-isogeny φ: E → E' constructed via T4, verify `e_E(P, Q) = e_{E'}(φ(P), φ̂(Q))^ℓ` where φ̂ is the dual isogeny. Independent: compute degree(φ) via {kernel size, polynomial degree of X-rational map, Vélu summation count} — all three must agree. Saturates 3/3.
6. **Endomorphism ring T7 vs Schoof trace** — for ordinary E, the Frobenius trace t (from T3/T6) determines the discriminant of End(E) via t² − 4p = D · f² (D fundamental, f conductor). Independent witnesses: (a) factor t² − 4p directly (slot 291 ECM); (b) compute End(E) via Kohel's algorithm (T7); (c) for CM curves, recover D via Hilbert class polynomial root-test (T8). Three independent algorithms, must agree on D. Saturates 3/3.

These pins establish per-tier R-MUTUAL-CROSS-VALIDATION 3/3 saturation; each primitive has at least three independent algorithmic witnesses and the cross-checks are bit-exact (mod final-exponentiation in F_{p^12} — which is canonical, not lossy).

## Cross-cutting

- **zkmark / forge ZK-SNARK readiness** ← T13 + T14 BLS12-381 optimal Ate pairing is the single load-bearing primitive. Without it, `zkmark.go:147 HonestProver` stays as the placeholder mirror-mark wrapper; with it, a real Groth16/Marlin/Plonk verifier ships. **HIGHEST PRIORITY FOR THIS SLOT.**
- **Slot 057 crypto-missing** ← T0 EC type composes slot 057 T1-EC (or slot 057 ships first and slot 292 imports). T13 BLS12-381 pairing enables slot 057 BLS aggregate signatures (one of nine topic-listed primitives in slot 057).
- **Slot 058 crypto-sota** ← Constant-time discipline applies to BLS signing (secret key); pairing verification is public-input and may be variable-time.
- **Slot 211 lattice-crypto** ← Disjoint algebra; no overlap. NTT can be shared if Velu-ℓ exceeds 64.
- **Slot 290 galois-theory** ← Frobenius is a Galois generator; CM endomorphism ring is an order in an imaginary quadratic field (Galois object). Slot 292 may consume slot 290's Berlekamp polynomial factoring to factor division polynomials over F_p.
- **Slot 291 modular-arithmetic** ← Bidirectional: slot 291 ships Tonelli-Shanks + Montgomery + Barrett (slot 292 consumes); slot 291's ECM factoring (T7) consumes slot 292's T0 EC point arithmetic. **Recommend co-shipping ECM at the slot-291/292 boundary.**
- **Pistachio (downstream)** ← Pairing verification on stream metadata; ~1.5ms latency budget fits pairing computation.
- **Post-quantum SIDH/SIKE/CSIDH/SQIsign** ← T4 Velu's formulae are the workhorse. SIKE was broken (Castryck-Decru 2022), but Velu lives on for CSIDH (still secure) and SQIsign.
- **Future Ethereum interop** ← T14' BN254 (alt_bn128 precompile) for backward compatibility; T14 BLS12-381 for forward-looking proto-danksharding (EIP-4844).

## Sources

- `C:\limitless\foundation\reality\crypto\modular.go:1-135` (existing uint64-only modular surface).
- `C:\limitless\foundation\reality\crypto\prime.go:26-306` (Miller-Rabin, mulmod, addmod).
- `C:\limitless\foundation\reality\zkmark\zkmark.go:84,147` (placeholder MarkChainStatement / HonestProver — awaiting real pairing-based SNARK).
- `C:\limitless\foundation\reality\zkmark\README.md:7` (Tranche 2 = real Halo2 implementation, conditional).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\057-crypto-missing.md:28-50` (T1-BIGINT, T1-FIELD, T1-FIELDTOWER, T1-EC prerequisites).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\290-new-galois-theory.md` (Berlekamp polynomial factoring; Galois substrate).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\291-new-modular-arithmetic.md:96-107,178-190` (Tonelli-Shanks T1; ECM T7 — bidirectional dependency with slot 292).
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md:309` (slot 292 line definition).
- Schoof 1985, *Elliptic curves over finite fields and the computation of square roots mod p*, Math. Comp. 44(170):483-494.
- Schoof 1995, *Counting points on elliptic curves over finite fields*, J. Théor. Nombres Bordeaux 7(1):219-254.
- Atkin 1991, *The number of points on an elliptic curve modulo a prime* (manuscript circulated).
- Elkies 1998, *Elliptic and modular curves over finite fields and related computational issues*, in Buell-Teitelbaum eds. *Computational Perspectives on Number Theory*, AMS-IP Studies in Adv. Math. 7:21-76.
- Lercier-Morain 1995, *Counting the number of points on elliptic curves over finite fields: strategies and performances*, EUROCRYPT '95, LNCS 921:79-94.
- Vélu 1971, *Isogénies entre courbes elliptiques*, C.R. Acad. Sci. Paris Ser. A-B 273:A238-A241.
- Bernstein-De Feo-Leroux-Smith 2020, *Faster computation of isogenies of large prime degree*, ANTS-XIV (the √élu paper).
- Atkin-Morain 1993, *Elliptic curves and primality proving*, Math. Comp. 61(203):29-68.
- Cornacchia 1908, *Su di un metodo per la risoluzione in numeri interi dell'equazione Σ_{h=0}^n c_h x^{n-h} y^h = P*, Giornale di Matematiche di Battaglini 46:33-90.
- Bröker-Stevenhagen 2007, *Constructing elliptic curves of prime order*, in *Computational Arithmetic Geometry* AMS Contemp. Math. 463:17-28.
- Sutherland 2011, *Computing Hilbert class polynomials with the Chinese Remainder Theorem*, Math. Comp. 80(273):501-538.
- Miller 1986/2004, *The Weil pairing, and its efficient calculation*, J. Cryptology 17(4):235-261.
- Vercauteren 2010, *Optimal pairings*, IEEE TIT 56(1):455-461.
- Hess-Smart-Vercauteren 2006, *The Eta pairing revisited*, IEEE TIT 52(10):4595-4602.
- Boneh-Lynn-Shacham 2001, *Short signatures from the Weil pairing*, ASIACRYPT 2001, LNCS 2248:514-532.
- Boneh-Drijvers-Neven 2018, *Compact multi-signatures for smaller blockchains*, ASIACRYPT 2018, LNCS 11273:435-464.
- Kate-Zaverucha-Goldberg 2010, *Constant-size commitments to polynomials and their applications*, ASIACRYPT 2010, LNCS 6477:177-194.
- Barreto-Lynn-Scott 2002, *Constructing elliptic curves with prescribed embedding degrees*, SCN 2002, LNCS 2576:257-267.
- Bowe 2017, *BLS12-381: New zk-SNARK Elliptic Curve Construction*, ZCash blog, March 2017.
- Bos-Costello-Hisil-Lauter 2013, *High-performance scalar multiplication using 8-dimensional GLV/GLS decomposition*, CHES 2013, LNCS 8086:331-348.
- Menezes-Okamoto-Vanstone 1993, *Reducing elliptic curve logarithms to logarithms in a finite field*, IEEE TIT 39(5):1639-1646.
- Frey-Rück 1994, *A remark concerning m-divisibility and the discrete logarithm in the divisor class group of curves*, Math. Comp. 62(206):865-874.
- Pollard 1978, *Monte Carlo methods for index computation (mod p)*, Math. Comp. 32:918-924.
- van Oorschot-Wiener 1999, *Parallel collision search with cryptanalytic applications*, J. Cryptology 12(1):1-28.
- Lenstra 1987, *Factoring integers with elliptic curves*, Annals of Math. 126(3):649-673.
- Semaev 2004, *Summation polynomials and the discrete logarithm problem on elliptic curves*, IACR ePrint 2004/031.
- Castryck-Decru 2022, *An efficient key recovery attack on SIDH*, EUROCRYPT 2023 (the SIKE break).
- Castryck-Lange-Martindale-Panny-Renes 2018, *CSIDH: An efficient post-quantum commutative group action*, ASIACRYPT 2018, LNCS 11274:395-427.
- Cohen-Frey 2005, *Handbook of Elliptic and Hyperelliptic Curve Cryptography*, CRC Press — comprehensive reference.
- Silverman 2009, *The Arithmetic of Elliptic Curves*, 2nd ed., GTM 106 — the canonical mathematical reference.
- IETF draft-irtf-cfrg-pairing-friendly-curves-11 (BLS12-381, BN254, BLS12-377, BLS24-315 specifications).
- ZCash protocol specification §5.4.9.5 (BLS12-381 parameter pin).
- EIP-2537 (BLS12-381 precompiles for Ethereum), EIP-4844 (proto-danksharding using KZG over BLS12-381).
