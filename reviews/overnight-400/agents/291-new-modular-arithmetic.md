# 291 — new-modular-arithmetic (Montgomery / Barrett / CRT / NTT / Tonelli-Shanks / Pollard / ECM)

## Headline
Reality v0.10.0 ships ZERO advanced modular-arithmetic surface — no Montgomery, Barrett, Tonelli-Shanks, NTT, Pollard rho, ECM, BSGS, Pohlig-Hellman, Brent-Kung, Karatsuba-on-bigint, Adleman-Manders-Miller, or Garner CRT — only a 135-LOC uint64 textbook kit (ModPow / ModInverse / ChineseRemainder via inverse-per-modulus / Russian-peasant mulmod) in `crypto/modular.go`; the singular Day-1 PR is **T0 Barrett + Montgomery contexts (~280 LOC, replaces Russian-peasant `mulmod` and ~3-5x speeds every existing primality and modexp call) plus T1 Tonelli-Shanks square-root mod p (~120 LOC, unblocks every elliptic-curve point-decompression in slot 057-T1-EC, every QR-residue test, and Cipolla fallback)**, with the singular highest-leverage NEW primitive being **T4 NTT (forward + inverse negacyclic, ~280 LOC)** — without it Kyber/Dilithium/Falcon (slot 211) and zkSNARK FRI low-degree-extension (slot 200) all fall back to O(n²) polynomial multiplication and become unshippable.

## Findings

### State at HEAD (v0.10.0, 2026-05-09)

Repo-wide grep on `Montgomery|Barrett|TonelliShanks|Tonelli|AdlemanMandersMiller|BrentKung|Karatsuba|ToomCook|Schönhage|NumberTheoreticTransform|\bNTT\b|BabyStepGiantStep|PohligHellman|IndexCalculus|EllipticCurveMethod|\bECM\b|PollardRho|PollardP1|DiscreteLog|SqrtMod|SquareRootMod|RootMod`: **zero callable hits in any source file**. All 20 matches are in prior agent reviews — slot 057, 058, 059, 060 (crypto), 211 (lattice-crypto), 290 (galois-theory) all flag these as missing.

| Surface | Path:line | Status | Modular-arithmetic relevance |
|---|---|---|---|
| `ModPow(base, exp, mod uint64) uint64` | `crypto/modular.go:20-40` | uint64-only, binary exp | OK shape but `mulmod` is Russian-peasant O(log b) — should be Montgomery O(1) |
| `ModInverse(a, mod uint64) (uint64, bool)` | `crypto/modular.go:54-75` | ext-Euclidean | Adequate; could add Fermat fallback `a^(p-2) mod p` for prime p |
| `ChineseRemainder(residues, moduli []uint64)` | `crypto/modular.go:96-135` | uint64, pairwise-coprime check | NOT Garner mixed-radix — uses M=∏m and ModInverse per term. Quadratic memory in M; overflows at ~5 small primes. **Garner avoids the overflow**. |
| `mulmod(a, b, m uint64)` | `crypto/prime.go:284-296` | Russian-peasant | **O(log b)** — every ModPow / Miller-Rabin call pays this. Montgomery would be O(1) per multiply. |
| `addmod(a, b, m uint64)` | `crypto/prime.go:299-306` | branch on `a >= m-b` | Adequate; not constant-time but variable-time is OK for non-secret inputs. |
| `IsPrime` Miller-Rabin (det. witnesses) | `crypto/prime.go:26-59` | uint64 | Burns mulmod-O(log b) per witness; Montgomery would 5-10x speed up uint64 case. |
| `PrimeFactors` trial division | `crypto/prime.go:149-176` | O(√n) | Adequate for n ≤ 2^32; **for n ≥ 2^48 Pollard rho or ECM are mandatory** (trial division is hours, rho is milliseconds). |
| Cooley-Tukey radix-2 FFT (float64) | `signal/fft.go:49,101` | float64 complex | The *math twin* of NTT — same butterfly + bit-reversal shape but over `complex128` not `Z_q`. NOT directly reusable but the algorithmic blueprint is shared. |
| LLL primitive | absent | — | Slot 290's T6 keystone; consumed by slot 290 (van Hoeij) + slot 211 (lattice-crypto) + slot 291 (integer-relation finding for discrete-log index-calculus) — recommend single shared `lattice/lll.go` per slot 290's note. |

### Cross-slot orientation

| Adjacent slot | Overlap | Resolution |
|---|---|---|
| **057-crypto-missing** §T1-BIGINT/T1-FIELD | Mentions Barrett-OR-Montgomery + Tonelli-Shanks for sqrt-mod-p as PRE-REQUISITE for any signature scheme. Doesn't itself ship them; defers. | Slot 291 owns the implementation. 057's Schnorr/EdDSA/Ed25519 consume slot 291's Tonelli-Shanks for point decompression. |
| **211-new-lattice-crypto** L2 | Names Barrett + Montgomery for Kyber/Dilithium/Falcon NTT inner loop with **specific precomputed constants** (Kyber q=3329 mu=5039 qInv=62209, Dilithium q=8380417, Falcon q=12289). | **Coordinate**: ship the *generic* Barrett/Montgomery context in `crypto/` (slot 291), let `lattice/ring/barrett.go` (slot 211) instantiate with the specific PQ moduli. Avoids duplication. |
| **211-new-lattice-crypto** L3+L4 | NTT forward + inverse, ~280 LOC, KEYSTONE for PQ schemes. | **Promote** out of `lattice/` into shared `crypto/ntt/` (or `linalg/ntt/`) — NTT is also the workhorse of zkSNARK polynomial commitments (slots 147/148/175/200) and integer multiplication via Schönhage-Strassen (frontier). Single implementation, 3+ consumers. |
| **290-new-galois-theory** T2 (Berlekamp 1967 polynomial-factoring) + T11 (Brent-Kung 1978 modular composition) | Slot 290 names Brent-Kung as the workhorse for fast `f(g(x)) mod h(x)` composition used in Galois-group computation. | **Slot 291 ships Brent-Kung** as a low-level primitive (composes well with NTT-multiply when NTT-friendly modulus is chosen); slot 290 consumes via `PolyComposeMod`. |
| **150-zkmark-perf / 175-synergy-zkmark-crypto** | FRI / KZG / Plonk all need O(n log n) polynomial multiplication = NTT. | NTT (T4 below) directly unblocks zkmark perf path. |
| **058-crypto-sota** | Likely flags constant-time discipline gaps. | Montgomery multiply is naturally constant-time (no data-dependent branches if implemented carefully); Barrett requires 1 conditional subtract — both are CT-friendly. Pollard rho / ECM are publicly variable-time (factoring is not a secret-handling op). |

### Distinct-from-slot-290 disambiguation of "Berlekamp"

MASTER_PLAN line 291 lists "Berlekamp" as the fourth keyword. Slot 290 already owns **Berlekamp 1967 polynomial factoring over F_p** (slot 290's T2, ~180 LOC). This slot's "Berlekamp" reference is best filled by **Berlekamp-Rabin 1970 root-finding mod p** — a randomized algorithm to find a square-root-mod-p (and roots of any polynomial mod p) running in expected O((log p)² · deg(f)) time. Berlekamp-Rabin generalizes Tonelli-Shanks to arbitrary polynomial roots and is the textbook companion to Tonelli-Shanks for **p ≡ 1 (mod 4) with high 2-adicity** where Tonelli-Shanks degrades — slot 291 ships both as alternatives.

### Does math/big cover this?

`math/big.Int` ships Karatsuba (and Toom-Cook 3-way for large operands) for multiplication, ext-Euclidean for `ModInverse`, Barrett-like reduction internally (`*big.Int` does NOT expose Montgomery to userland; the optimization is hidden inside `*big.Int.Exp` for prime-shaped moduli only). **Reality's `crypto/` package is uint64-only and bypasses `math/big` entirely**, so:

- For uint64 path: math/big is irrelevant — slot 291 owns Montgomery/Barrett at the uint64 limb level.
- For big.Int path (post slot-057-T1-BIGINT): math/big provides Karatsuba + Toom-Cook *implicitly*; slot 291 should NOT re-implement these. Slot 291 SHOULD provide Montgomery context wrapper around `*big.Int` (since math/big hides it), Tonelli-Shanks over `*big.Int`, Pollard rho over `*big.Int`, ECM over `*big.Int`.

### MIT zero-dep moat

| Library | License | Coverage | Reality moat |
|---|---|---|---|
| Go `math/big` | BSD-3 | Karatsuba, Toom-Cook 3, Burnikel-Ziegler division, hidden Montgomery | Slot 291 EXPOSES Montgomery (math/big does not), adds Tonelli-Shanks / Pollard / ECM / NTT / Brent-Kung |
| `mathnet/numerics` | MIT but no number-theory | — | Reality fills the gap |
| FLINT, NTL, GMP, Pari | LGPL/GPL — not MIT-compatible | full | License moat for MIT-licensed downstream |
| Cloudflare CIRCL | BSD-3-style | Montgomery + NTT for Kyber/Dilithium specifically | Reality is *cross-language deterministic* — Python/C++/C# golden-file parity, CIRCL is Go-only |
| `crypto/elliptic` (stdlib) | BSD-3 | P-256 specific Montgomery | Hard-wired to one curve |

Reality's positioning: the MIT pure-Go zero-dep cross-language-deterministic-golden-file Tonelli-Shanks + Montgomery context + NTT toolkit. Narrow but real.

## Concrete recommendations

Tier numbering: T0 = fast modular reduction substrate (Barrett, Montgomery); T1 = sqrt-mod-p; T2 = higher roots; T3 = multi-modulus CRT; T4 = NTT (KEYSTONE for PQ + zkSNARK); T5 = modular polynomial composition; T6-T8 = factoring; T9-T12 = discrete log; T13-T14 = bigint multiplication frontiers; T15 = polynomial GCD over F_p (delegates to slot 290).

### T0 — `crypto/montgomery.go` + `crypto/barrett.go` ~280 LOC — **DAY-1 KEYSTONE**

Fixed-modulus context that precomputes magic constants once, then accepts inputs at amortized O(1) per multiply.

```go
// crypto/montgomery.go
type MontgomeryCtx struct{ N, R, NPrime, R2 uint64 } // R = 2^64, N*NPrime ≡ -1 (mod R)
func NewMontgomeryCtx(n uint64) MontgomeryCtx
func (c MontgomeryCtx) ToMont(a uint64) uint64        // a · R mod n
func (c MontgomeryCtx) FromMont(aBar uint64) uint64   // REDC(aBar)
func (c MontgomeryCtx) MulMont(aBar, bBar uint64) uint64  // REDC(aBar · bBar)
func (c MontgomeryCtx) ExpMont(aBar, e uint64) uint64

// crypto/barrett.go
type BarrettCtx struct{ N uint64; Mu uint64; K uint } // mu = floor(2^k / N)
func NewBarrettCtx(n uint64) BarrettCtx
func (c BarrettCtx) Reduce(a uint64) uint64           // a mod n in O(1)
func (c BarrettCtx) ReduceWide(hi, lo uint64) uint64  // 128-bit input
```

**Replaces** the Russian-peasant `crypto/prime.go:284 mulmod` — a typed-context call site survives and `mulmod` becomes the slow-path fallback for moduli > 2^63 (Barrett/Montgomery both want a precomputed inverse-of-modulus).

**What it unblocks:**
- 5-10x faster `IsPrime`, `MillerRabin`, `ModPow`, `NextPrime`.
- Slot 057 RSA / DH / ECC point arithmetic (Montgomery is the *standard* RSA reduction).
- Slot 211 Kyber/Dilithium NTT inner loop (with PQ-specific moduli precomputed).

Refs: Montgomery 1985 *Modular multiplication without trial division* Math. Comp. 44; Barrett 1986 *Implementing the Rivest-Shamir-Adleman public key encryption algorithm on a standard digital signal processor* CRYPTO '86; Hankerson-Menezes-Vanstone 2004 *Guide to Elliptic Curve Cryptography* §2.2.4-§2.2.5.

### T1 — `crypto/sqrtmod.go` ~140 LOC — **DAY-1 KEYSTONE**

`SqrtMod(a, p uint64) (uint64, bool)` returns x such that x² ≡ a (mod p) for prime p, and `true` iff a is a QR mod p. Algorithm dispatch:
- p ≡ 3 (mod 4): direct formula x = a^((p+1)/4) mod p, ~10 LOC.
- p ≡ 5 (mod 8): Atkin's formula, ~20 LOC.
- p ≡ 1 (mod 4) general: **Tonelli-Shanks 1891+1972**. Find quadratic non-residue z (Euler's criterion); decompose p-1 = Q · 2^S; loop with R = a^((Q+1)/2), c = z^Q, t = a^Q, M = S; while t ≠ 1 find least i with t^(2^i) = 1, b = c^(2^(M-i-1)), update (M, c, t, R) ← (i, b², b²·t, R·b). O((log p)² · g(p)) where g = 2-adic valuation of p-1.
- Optional **Cipolla 1903** as Tonelli-Shanks alternative for p with very high 2-adicity (Tonelli-Shanks degrades to O(log²p · S) where S = ν₂(p-1); Cipolla is O((log p)³) regardless). ~40 LOC; uses arithmetic in F_{p²}.

**Companion**: `IsQuadraticResidue(a, p uint64) bool` via Euler's criterion `a^((p-1)/2) mod p`. ~10 LOC.

**What it unblocks:**
- Slot 057 EC point decompression (`y² = x³ + ax + b mod p` → solve for y given x).
- Slot 058 Tonelli-Shanks is THE standard sqrt-mod-p; required by every curve compression decode.
- Slot 200 zkSNARK FRI verifier (low-degree extension testing requires QR checks).
- Reed-Solomon decoding root-finding (slot 210) when working over F_p.

Refs: Tonelli 1891 *Bemerkung über die Auflösung quadratischer Congruenzen* Göttinger Nachrichten; Shanks 1972 *Five number-theoretic algorithms* Proc. 2nd Manitoba Conference; Cipolla 1903 *Un metodo per la risoluzione della congruenza di secondo grado*; Atkin (1992) *Probabilistic primality testing* (the p ≡ 5 mod 8 closed-form).

### T2 — `crypto/rootmod.go` ~180 LOC

`NthRootMod(a, n, p uint64) (uint64, bool)` for arbitrary n-th root mod prime p. Algorithm: **Adleman-Manders-Miller 1977** — generalizes Tonelli-Shanks to n-th roots. Decompose p-1 = n^s · t with gcd(n, t) = 1; if gcd(n, p-1) divides ord(a) then a is an n-th-power residue. Plus **Berlekamp-Rabin 1970** randomized root-finding fallback for the *polynomial* `x^n - a mod p` (O((log p)² · n) expected).

**What it unblocks:** post-quantum crypto cube-roots (some lattice trapdoors); polynomial root-finding mod p generally; AKS / ECPP primality auxiliaries.

Refs: Adleman-Manders-Miller 1977 *On taking roots in finite fields* FOCS '77; Berlekamp 1970 *Factoring polynomials over large finite fields* Math. Comp. 24.

### T3 — `crypto/crt_garner.go` ~140 LOC — replaces existing CRT for n ≥ 3

`GarnerCRT(residues, moduli []uint64) (mixedRadix []uint64, err error)` — Garner 1965 mixed-radix algorithm. Given r_i mod m_i for pairwise-coprime m_i, computes coefficients v_i such that x = v_0 + v_1·m_0 + v_2·m_0·m_1 + ... + v_{n-1}·m_0·...·m_{n-2}. Each v_i lives in [0, m_i) so the mixed-radix form **never overflows even when ∏m_i exceeds uint64** — fixes the silent overflow bug in the existing `crypto/modular.go:113-115 M *= m` loop (M overflows the moment ∏m_i > 2^64).

Companion `GarnerToInt(mixedRadix, moduli) *big.Int` reconstructs the standard residue when needed. Companion `GarnerCRTBig(residues, moduli []*big.Int) *big.Int` for the bigint case.

**What it unblocks:**
- RSA-CRT decryption (4x speedup over straight RSA): given (p, q, dp, dq, qInv), decrypt via two half-size modexps + CRT recombination. Slot 057 RSA-private-key shape.
- Slot 211 BFV/BGV multi-prime modulus chains (RNS — residue number system).
- Slot 290 Hensel-lift CRT recombination.

Refs: Garner 1965 *The residue number system* IRE Trans. Electronic Computers EC-8; Knuth TAOCP Vol. 2 §4.3.2.

### T4 — `crypto/ntt/forward.go` + `crypto/ntt/inverse.go` ~320 LOC — **HIGHEST-LEVERAGE NEW PRIMITIVE**

Number-theoretic transform = FFT over F_p. Replaces O(n²) polynomial multiplication with O(n log n) for any **NTT-friendly prime** p satisfying p ≡ 1 (mod 2n) (so a primitive 2n-th root of unity exists in F_p).

```go
type NTTCtx struct{ N int; Q uint32; Zetas, ZetasInv []uint32; NInv uint32; Mont MontgomeryCtx }
func NewNTTCtx(n int, q uint32) (*NTTCtx, error)   // requires q ≡ 1 (mod 2n)
func (c *NTTCtx) Forward(a []uint32)                // in-place, output in bit-reversed order
func (c *NTTCtx) Inverse(a []uint32)                // input bit-reversed, output natural
func (c *NTTCtx) MulPoly(a, b []uint32) []uint32    // pointwise multiply via NTT
func (c *NTTCtx) NegacyclicForward(a []uint32)      // for Z_q[x]/(x^n+1) — Kyber/Dilithium/Falcon
```

Negacyclic variant needed for cyclotomic ring R_q = Z_q[x]/(x^n + 1) — twists by a primitive 2n-th root, embedding the x^n + 1 reduction into the transform.

**Composes T0 Montgomery**: every butterfly multiply uses `MulMont`. **Composes T2 sqrt-mod-p**: 2n-th root extraction via Tonelli-Shanks on a 2n-th root of unity.

**What it unblocks (ENORMOUS):**
- Slot 211 Kyber, Dilithium, Falcon (without NTT they ship at 1/100x speed = unshippable).
- Slot 200 zkSNARK Plonk / Halo2 / FRI all need O(n log n) polynomial multiplication.
- Slot 175 synergy-zkmark-crypto KZG commitments require NTT for fast polynomial evaluation at roots-of-unity.
- Schönhage-Strassen integer multiplication (T14 frontier).

Refs: Pollard 1971 *The fast Fourier transform in a finite field* Math. Comp. 25; Cooley-Tukey 1965 (the underlying butterfly); Lyubashevsky-Pöppelmann-Buchmann 2014 *Speeding up the lattice-based digital signature scheme BLISS using sub-linear-size GS basis*; Longa-Naehrig 2016 *Speeding up the number-theoretic transform for faster ideal-lattice-based cryptography*; Seiler-Lyubashevsky-Schwabe 2018 *Faster AVX2 NTT for Ring-LWE*.

### T5 — `crypto/polycompose.go` ~180 LOC

**Brent-Kung 1978** modular polynomial composition: compute `f(g(x)) mod h(x)` in O(n^((ω+1)/2)) where ω is the matrix-multiplication exponent. Naive composition is O(n²) — Brent-Kung achieves sub-quadratic by reducing to matrix-matrix multiplication (precompute the powers `g, g², ..., g^√n mod h` then multiply with the coefficient matrix of f).

**What it unblocks:**
- Slot 290 Galois-group computation (Stauduhar resolvent + transitive group lookup heavily uses modular composition).
- Slot 290 Cantor-Zassenhaus polynomial factoring (the equal-degree split uses x^((p^d−1)/2) mod f — a modular composition of x with a giant exponent).
- Subproduct trees for fast multipoint polynomial evaluation (zkSNARK KZG batch opening).

Refs: Brent-Kung 1978 *Fast algorithms for manipulating formal power series* JACM 25; Kedlaya-Umans 2008 *Fast polynomial factorization and modular composition* (the modern improvement, optional T5'); von zur Gathen-Gerhard 2013 *Modern Computer Algebra* §12.

### T6 — `crypto/factor_pollard.go` ~140 LOC

`PollardRhoBrent(n uint64) (factor uint64, ok bool)` — **Pollard 1975** rho with **Brent 1980** improvement (different cycle-detection schedule, ~24% faster than Floyd's tortoise-and-hare). Expected O(n^(1/4)) per factor found; works for n up to ~2^60 in milliseconds. Used inside a recursive factoring driver: trial divide by small primes, then Pollard rho until composite remainders disappear, then Miller-Rabin primality on each remaining piece.

`PollardP1(n, B uint64) (factor uint64, ok bool)` — Pollard 1974 p-1 algorithm. Smoothness-based: works iff p-1 is B-smooth for some prime factor p. Cheap auxiliary for `crypto/factor_driver.go`.

**What it unblocks:**
- 30+ year speedup over the existing trial-division `crypto/prime.go:149 PrimeFactors`; replaces it for n ≥ 2^32.
- Slot 290 number-field discriminant factoring (Galois group inputs).

Refs: Pollard 1975 *A Monte Carlo method for factorization* BIT Numerical Math. 15; Brent 1980 *An improved Monte Carlo factorization algorithm* BIT 20; Pollard 1974 *Theorems on factorization and primality testing* Proc. Cambridge Phil. Soc.

### T7 — `crypto/factor_ecm.go` ~280 LOC

**Lenstra 1987 ECM** — elliptic curve method. Pick a random elliptic curve E over Z/nZ; compute [k]P for a smooth k = lcm(1..B); if at any point an addition fails (denominator non-invertible mod n), gcd reveals a factor. **Best general-purpose factoring algorithm for n with a factor < 2^60-2^80**, complexity sub-exponential L_p[1/2, √2]. Two-stage: stage 1 uses smoothness bound B1, stage 2 uses prime-pair table up to B2 ≫ B1.

Requires elliptic-curve point addition mod n (where n is composite — the whole point is that the addition law fails on a hidden factor). ~150 LOC for the stage-1 + stage-2 logic + ~130 LOC for Edwards-form curve point arithmetic mod n.

**What it unblocks:**
- Factoring 60-80 bit semiprimes that Pollard rho misses.
- Cryptanalysis benches against weak RSA / Diffie-Hellman moduli.
- The mathematical *core* of Atkin-Morain ECPP primality proving (deferred T6' if ever needed).

Refs: Lenstra 1987 *Factoring integers with elliptic curves* Annals of Math. 126; Brent 1986 *Some integer factorization algorithms using elliptic curves*; Zimmermann-Dodson 2006 *20 years of ECM* ANTS-VII.

### T8 — `crypto/factor_driver.go` ~80 LOC

`Factor(n uint64) []uint64` — composite driver: trial-divide < 10^4, Pollard rho until O(2^32) remainder, then ECM if still composite, then declare prime via Miller-Rabin. Replaces the current `PrimeFactors(n)` API, keeping byte-identical output for n < 2^32.

### T9 — `crypto/discretelog_bsgs.go` ~80 LOC

`BSGS(g, h, p uint64) (x uint64, ok bool)` — **Shanks 1971** baby-step giant-step. Find x such that g^x ≡ h (mod p). Time O(√p), space O(√p). For composite-order subgroups use T11 Pohlig-Hellman to reduce to prime-power factors, each solved by BSGS over the smaller order.

Refs: Shanks 1971 *Class number, a theory of factorization, and genera* Proc. Symp. Pure Math. 20.

### T10 — `crypto/discretelog_pollard.go` ~120 LOC

**Pollard rho discrete log 1978** — same cycle-detection idea as factoring rho but in the multiplicative group. Time O(√p), space O(1). Combined with **Brent 1980** improvement.

Refs: Pollard 1978 *Monte Carlo methods for index computation (mod p)* Math. Comp. 32.

### T11 — `crypto/discretelog_pohlig.go` ~120 LOC

**Pohlig-Hellman 1978** — given a factorization of group order n = ∏ p_i^e_i (input from T6/T7/T8), reduce DL in the order-n group to e_i DL problems each in an order-p_i group, recombine via CRT (T3). Drops complexity from O(√n) to O(√(largest prime factor of n)). Demonstrates *why* DL-based crypto requires the group order to have a large prime factor.

**What it unblocks:**
- Cryptanalysis benches: educational demonstration that Z/((p-1)) with smooth p-1 is broken.
- Validation that Diffie-Hellman parameters use safe primes (p = 2q+1, q prime).

Refs: Pohlig-Hellman 1978 *An improved algorithm for computing logarithms over GF(p) and its cryptographic significance* IEEE TIT 24.

### T12 — `crypto/discretelog_indexcalculus.go` ~280 LOC (SKETCH — defer to v2)

Index calculus for prime fields: factor base of small primes, sieve for relations of form g^k ≡ ∏ p_i^a_i, solve linear system mod (p-1), final logarithm by special-q. Complexity sub-exponential L_p[1/2]. Frontier-grade; defer unless a concrete consumer appears. Document the algorithm; ship a stub with `ErrNotImplementedV2` until then.

Refs: Adleman 1979 *A subexponential algorithm for the discrete logarithm problem* FOCS '79; Coppersmith-Odlyzko-Schroeppel 1986 *Discrete logarithms in GF(p)*.

### T13 — `crypto/karatsuba.go` ~120 LOC (likely SKIP)

Karatsuba 1962 multiplication is **already in math/big** — slot 291's bigint path inherits it. Re-implementing in Reality competes with stdlib for zero added value. **Recommendation: SKIP for bigints.** The case for shipping a *standalone* uint64-pair Karatsuba is for the polynomial-multiplication path in `galois/poly_z.go` (slot 290 T0') where math/big multiplies coefficients but the polynomial layer needs to multiply *polynomials*; a Karatsuba poly-multiply is ~80 LOC and 3x faster than schoolbook for degree ≥ 32. Track as a slot-290 helper, not slot-291.

### T14 — Schönhage-Strassen (FRONTIER — DEFER)

O(n log n log log n) integer multiplication via complex-FFT or NTT-based fast convolution. Worth it only for n ≥ 2^15 bits; math/big switches over internally at large sizes. **Defer indefinitely** — no current Reality consumer hits the regime where Schönhage-Strassen beats Toom-Cook 4. Document as a known frontier; revisit if zkSNARKs ever need post-million-bit modular multiplications. (Even Furer 2007 / Harvey-van der Hoeven 2019 O(n log n) algorithms are practically slower than Schönhage-Strassen below 2^25 bits.)

### T15 — Polynomial GCD over F_p[x] — delegate to slot 290 T1

Slot 290 T1 already names `PolyGCDFp` ~80 LOC. No duplication.

## Day-1 PR shape

**Singular cheapest, highest-immediate-value PR**: T0 (Barrett + Montgomery contexts) + T1 (Tonelli-Shanks) + T3 (Garner CRT). ~540 LOC, replaces `crypto/modular.go:96 ChineseRemainder` and `crypto/prime.go:284 mulmod`, makes `IsPrime` 5-10x faster, fixes the silent ∏m_i overflow bug in current CRT, ships sqrt-mod-p that 057's EC code needs. No new dependencies.

**Singular highest-strategic-value follow-up**: T4 NTT (forward + inverse). ~320 LOC. Single primitive that unblocks slot 211 (Kyber/Dilithium/Falcon), slot 200 (zkSNARK polynomial commitments), and slot 175 (KZG). Without NTT, all three are unshippable.

## Cross-validation pin opportunities (R-MUTUAL-CROSS-VALIDATION 3/3)

1. **`MulMont(a,b) ≡ Barrett((a*b)) ≡ (a*b) mod m`** — 10⁴ random pairs across 5 moduli (small prime, 2³² prime, 2⁵² random odd, Mersenne 2⁶¹−1, Kyber q=3329). Bit-exact in all three reductions.
2. **`SqrtMod(a, p)` round-trip** — 10³ random a ∈ QR(p) for 5 different primes (p ≡ 3 mod 4, p ≡ 5 mod 8, p ≡ 1 mod 4 small 2-adic, p ≡ 1 mod 4 high 2-adic, p = NIST P-256 base prime). Verify `x² mod p == a` for every result; verify `IsQuadraticResidue` returns false for non-QR cases.
3. **`Factor(n)` 3-way agreement** — for 100 random semiprimes n = pq (p, q ∈ [10⁶, 10⁹]): Pollard rho, Pollard p-1 (with B = 10⁵), and ECM (B1 = 10⁴, B2 = 10⁶) all return the same prime factor, and `Factor(n)` driver returns `[p, q]` sorted.
4. **`INTT(NTT(x)) ≡ x`** — 1,024 random length-256 polynomials over Kyber's q=3329; 256 length-1024 polynomials over Falcon's q=12289; bit-exact roundtrip.
5. **CRT consistency** — `GarnerCRT(residues, moduli)` ≡ existing `ChineseRemainder` for n ≤ 4 small moduli (where existing version doesn't overflow); fuzz n=8-16 moduli where existing version overflows and assert `ChineseRemainder` returns an error while `GarnerCRT` succeeds.
6. **BSGS ≡ Pollard-rho-DL ≡ Pohlig-Hellman** — for 100 random (g, h, p) with p < 2³⁰: all three discrete-log algorithms return the same x (or all three return "no solution").

These pins saturate the R-MUTUAL-CROSS-VALIDATION 3/3 pattern: each primitive has at least one independent algorithmic alternative (Montgomery vs Barrett vs naive; Tonelli-Shanks vs Cipolla; rho vs p-1 vs ECM; BSGS vs rho-DL vs Pohlig-Hellman) and the cross-check is bit-exact.

## Cross-cutting

- **Slot 057-crypto-missing T1-FIELD** ← T0 Montgomery + T1 Tonelli-Shanks land the *prerequisite* substrate for every signature scheme (Schnorr, EdDSA, BLS, ECDSA all need fast modular reduction + sqrt-mod-p).
- **Slot 058-crypto-sota / 060-crypto-perf** ← T0 closes the constant-time gap (Montgomery is naturally CT; Russian-peasant `mulmod` is data-dependent in iteration count) and the perf gap (5-10x faster `IsPrime`).
- **Slot 211-new-lattice-crypto L1-L7** ← T0 (Barrett/Montgomery) + T4 (NTT) land directly as the slot 211 ring/ntt prerequisites; slot 211 instantiates with Kyber/Dilithium/Falcon-specific moduli.
- **Slot 200-synergy-zkmark-info / 175-synergy-zkmark-crypto / 147-zkmark-missing** ← T4 NTT is the workhorse for Plonk/Halo2/FRI polynomial commitments; without it zkSNARKs are O(n²).
- **Slot 290-new-galois-theory T2 + T11** ← T5 Brent-Kung modular composition is consumed by Galois-group computation and Cantor-Zassenhaus equal-degree splitting.
- **Slot 210-new-coding-theory** ← T1 Tonelli-Shanks supports Reed-Solomon root-finding over F_p; T15 (= slot 290 T1 PolyGCDFp) serves Reed-Solomon decoder.
- **Slot 057 RSA-CRT** ← T3 Garner CRT enables the 4x speedup on RSA private-key decryption (the canonical RSA-CRT optimization).

## Sources

- `C:\limitless\foundation\reality\crypto\modular.go:1-135` (existing surface — ModPow, ModInverse, ChineseRemainder).
- `C:\limitless\foundation\reality\crypto\prime.go:26-176, 284-306` (Miller-Rabin, PrimeFactors, mulmod, addmod).
- `C:\limitless\foundation\reality\signal\fft.go:49,101` (Cooley-Tukey float64 — algorithmic twin of NTT).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\057-crypto-missing.md:28-65` (T1-BIGINT, T1-FIELD prerequisites).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\211-new-lattice-crypto.md:3,41-46` (NTT + Barrett/Montgomery for Kyber/Dilithium/Falcon, with specific precomputed constants).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\290-new-galois-theory.md:27-28,242` (Brent-Kung consumer, Berlekamp disambiguation).
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md:308` (slot 291 line definition).
- Montgomery 1985, *Modular multiplication without trial division*, Math. Comp. 44(170):519-521.
- Barrett 1986, *Implementing the Rivest-Shamir-Adleman public key encryption algorithm on a standard digital signal processor*, CRYPTO '86.
- Tonelli 1891, *Bemerkung über die Auflösung quadratischer Congruenzen*, Göttinger Nachrichten 1891:344-346.
- Shanks 1972, *Five number-theoretic algorithms*, Proc. 2nd Manitoba Conf. on Numerical Math.
- Cipolla 1903, *Un metodo per la risoluzione della congruenza di secondo grado*, Rend. Accad. Sci. Fis. Mat. Napoli.
- Adleman-Manders-Miller 1977, *On taking roots in finite fields*, FOCS '77.
- Berlekamp 1970, *Factoring polynomials over large finite fields*, Math. Comp. 24:713-735.
- Garner 1965, *The residue number system*, IRE Trans. Electronic Computers EC-8(2):140-147.
- Pollard 1971, *The fast Fourier transform in a finite field*, Math. Comp. 25:365-374.
- Brent-Kung 1978, *Fast algorithms for manipulating formal power series*, JACM 25(4):581-595.
- Pollard 1974, *Theorems on factorization and primality testing*, Proc. Cambridge Phil. Soc. 76:521-528.
- Pollard 1975, *A Monte Carlo method for factorization*, BIT Num. Math. 15:331-334.
- Brent 1980, *An improved Monte Carlo factorization algorithm*, BIT 20:176-184.
- Lenstra 1987, *Factoring integers with elliptic curves*, Annals of Math. 126:649-673.
- Shanks 1971, *Class number, a theory of factorization, and genera*, Proc. Symp. Pure Math. 20.
- Pollard 1978, *Monte Carlo methods for index computation (mod p)*, Math. Comp. 32:918-924.
- Pohlig-Hellman 1978, *An improved algorithm for computing logarithms over GF(p) and its cryptographic significance*, IEEE TIT 24(1):106-110.
- Karatsuba-Ofman 1962, *Multiplication of multi-digit numbers on automata*, Soviet Phys. Dokl. 7:595-596.
- Schönhage-Strassen 1971, *Schnelle Multiplikation großer Zahlen*, Computing 7:281-292.
- Knuth TAOCP Vol. 2 *Seminumerical Algorithms* §4.3.2 (CRT), §4.5.4 (Pollard rho), §4.6.3 (modexp).
- von zur Gathen-Gerhard 2013, *Modern Computer Algebra* 3rd ed., Cambridge UP — comprehensive reference for T0-T15.
- Hankerson-Menezes-Vanstone 2004, *Guide to Elliptic Curve Cryptography*, Springer §2.2.
- FIPS-203 (Kyber/ML-KEM, 2024), FIPS-204 (Dilithium/ML-DSA, 2024) — NTT specifications.
