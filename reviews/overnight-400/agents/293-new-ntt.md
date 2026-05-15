# 293 — new-ntt (NTT Variants: Cooley-Tukey / Harvey / Negacyclic / Bluestein / Multi-Modular)

## Headline
Reality v0.10.0 ships ZERO number-theoretic transform surface (only `signal/fft.go:49` Cooley-Tukey over `complex128` — the *math twin* but not directly reusable since element type is float64 not Z_q); slot 291's T4 sketched a SINGLE generic NTT primitive at ~320 LOC — slot 293 deepens that into the FULL variant catalog (12 tiers, ~2,150 LOC across new sub-package `ntt/`) optimized for distinct consumers — and the singular cheapest day-1 PR is **N0 NTTCtx (precomputed roots-of-unity table over F_p with q ≡ 1 mod 2n) + N1 Cooley-Tukey radix-2 forward+inverse + N3 negacyclic NTT for Z_q[x]/(x^n+1) (~360 LOC, three primitives)** which directly unblocks slot 211 (Kyber/Dilithium/Falcon — they CANNOT ship without negacyclic NTT) and slot 200 (zkSNARK polynomial commitments — Plonk/Halo2/FRI all reduce to NTT on the prover's hot path).

## Findings

### State at HEAD (v0.10.0, 2026-05-09)

Repo-wide grep on `NTT|NumberTheoreticTransform|negacyclic|Negacyclic|Harvey|Bluestein|Rader|Stockham|Pease|number.theoretic|primitive.*root.*unity|PolyMul|ConvolutionMod`: **zero callable hits in any source file**. All matches are in prior agent reviews (slots 211, 290, 291, 292). The CLAUDE.md package table has no `ntt/` entry.

| Surface | Path:line | Status | NTT relevance |
|---|---|---|---|
| Cooley-Tukey radix-2 FFT (float64 complex) | `signal/fft.go:49,61-91` | float64 in-place | **Math twin of NTT** — same butterfly + bit-reversal shape, but `complex128 cos/sin` twiddle, not `uint32 mod q` zeta. Not directly reusable; algorithmic blueprint is shared. |
| `bitReverse(real, imag, n)` | `signal/fft.go:21-35` | shared util | Bit-reversal permutation is identical between FFT and NTT — **share via `signal.BitReverseUint32(a []uint32)`** (or duplicate, ~14 LOC). |
| IFFT (conjugate-FFT-conjugate-scale) | `signal/fft.go:101-127` | float64 | Method does NOT translate to F_q (no complex conjugation in F_q); inverse-NTT uses ζ^{-1} instead. |
| `ModPow(base, exp, mod uint64)` | `crypto/modular.go:20-40` | uint64 binary exp | Used to precompute NTT zetas: `zeta[i] = ψ^i mod q` for primitive 2n-th root ψ. |
| `ModInverse(a, mod uint64)` | `crypto/modular.go:54-75` | ext-Euclidean | Used to precompute n^{-1} mod q (final inverse-NTT scaling) and ζ^{-1} (Gentleman-Sande inverse butterflies). |
| Polynomial type over Z_q[x]/(x^n+1) | absent | — | Slot 211 L1 names `lattice/ring/poly.go:Polynomial{Coeffs []int32; N int; Q int32}` — slot 293 NTT consumes this OR a generic `[]uint32`. |
| Barrett/Montgomery reduction | absent | (slot 291 T0) | NTT inner-loop multiply uses Montgomery reduction (5-10x faster than naive `% q`). |
| Tonelli-Shanks sqrt-mod-p | absent | (slot 291 T1) | Used to construct primitive 2n-th roots of unity in F_q (ψ² = ζ where ζ is primitive n-th root). |

### Slot-coordination map

| Slot | Overlap | Resolution |
|---|---|---|
| **291-new-modular-arithmetic** T4 | Slot 291 sketches a SINGLE NTT primitive (`crypto/ntt/forward.go + inverse.go ~320 LOC`) as a follow-up to T0-T3 modular-arithmetic substrate. Slot 291 names it the "highest-leverage NEW primitive". | **Slot 293 absorbs and EXPANDS slot 291's T4**: not 1 NTT, but 12 variants tiered for distinct consumers (PQ vs. ZK vs. HPC vs. arbitrary-precision). Slot 291 owns the *substrate* (Montgomery / Barrett / Tonelli-Shanks / Garner-CRT); slot 293 owns the *transform layer* on top. Bidirectional dependency: N0 NTTCtx imports `crypto.MontgomeryCtx`; T6 multi-modular NTT imports `crypto.GarnerCRT`. |
| **211-new-lattice-crypto** L3+L4 | Names negacyclic Cooley-Tukey forward-NTT + Gentleman-Sande inverse-NTT with PQ-specific moduli (Kyber q=3329, Dilithium q=8380417, Falcon q=12289). | **Slot 211 instantiates** slot 293's generic `NTTCtx` with PQ moduli. Avoids per-scheme NTT duplication. Generic `ntt/` ships first; `lattice/ntt/{kyber,dilithium,falcon}.go` are 30-LOC wrappers each. |
| **200-synergy-zkmark-info / 175-synergy-zkmark-crypto / 147-zkmark-missing** | zkSNARKs (Plonk / Halo2 / FRI / KZG) need O(n log n) polynomial multiplication mod scalar-field-prime; for BLS12-381, scalar field r ≈ 2^255 — too large for uint32 NTT. **Multi-modular NTT (T6) required**. | Slot 293 T6 multi-modular CRT-recombined NTT is THE keystone for ZK prover side. Without it, prover hot-path is O(n²). |
| **292-new-elliptic-curves** T17 (Velu's formulae, large-degree isogenies) | Velu degree-ℓ isogenies need polynomial multiplication mod ℓ; for ℓ ≥ 64 this benefits from NTT. | Slot 292 OPTIONAL consumer — only matters if SEA Elkies-prime branch goes to ℓ ≥ 64. Default schoolbook for small ℓ. |
| **132-signal-missing / 134-signal-api** | Bluestein chirp-Z transform handles arbitrary-length FFT (not just powers of 2). | Bluestein algorithm structure is identical between complex-FFT and NTT. **Slot 293 T4 Bluestein-NTT** could share the chirp-Z structure with a slot-132 Bluestein-FFT (parametric over the multiply ring). Recommend slot 293 ships NTT-Bluestein first; slot 132 follow-up ships float64-Bluestein with shared structure documented. |

### NTT-friendly primes (the "modulus zoo")

NTT requires q ≡ 1 (mod 2n) so that F_q* has a primitive 2n-th root of unity. The canonical PQ + ZK moduli:

| q | bits | n_max | Used by | Primitive 2n-th root ψ |
|---|---:|---:|---|---|
| 3329 | 12 | 256 | Kyber (FIPS-203) | ψ=17 (n=256) |
| 7681 | 13 | 512 | early NewHope | ψ=12 (n=512) |
| 12289 | 14 | 1024 | Falcon (FIPS-206 draft), NewHope | ψ=49 (n=1024) |
| 40961 | 16 | 2048 | research / pedagogy | ψ chosen at runtime |
| 8380417 | 23 | 256 | Dilithium (FIPS-204) | ψ=1753 (n=256) |
| 998244353 = 119·2^23+1 | 30 | 2^23 | competitive programming, ZK proofs | ψ=15311432 |
| 754974721 = 45·2^24+1 | 30 | 2^24 | ZK proofs | g=362 |
| 167772161 = 5·2^25+1 | 28 | 2^25 | ZK proofs | g=243 |
| 0xffffffff_00000001 (Goldilocks) | 64 | 2^32 | Plonky2 / RISC-Zero / StarkWare-friendly | g=7 |
| BLS12-381 scalar field r ≈ 2^255 | 255 | 2^32 | KZG / Plonk on BLS | requires multi-modular |
| Mersenne 2^31-1 | 31 | (n must divide 2^31-2 = 2·3²·7·11·31·151·331) | Mersenne31 (Plonky3 2024) | g=7, special |

Schönhage-Strassen-style three-primes-NTT typically picks q_1, q_2, q_3 as 3 NTT-friendly primes near 2^62 with pairwise-coprime products, then uses Garner-CRT (slot 291 T3) to recombine — slot 293 T6 keystone.

### Does math/big or stdlib cover this?

**No.** Go's stdlib has zero NTT support. `math/big` provides Karatsuba + Toom-Cook for Z multiplication but no F_q polynomial multiplication. The third-party landscape:

| Library | License | NTT coverage | Reality moat |
|---|---|---|---|
| Cloudflare CIRCL | BSD-3-style | Kyber/Dilithium-specific NTT (hard-wired moduli) | Reality = generic `NTTCtx` over arbitrary NTT-friendly primes |
| Microsoft SEAL (C++) | MIT | Harvey-2014 fast NTT for HE moduli | C++ only — no Go binding without CGO |
| PALISADE (C++) | BSD-2 | Multi-modular NTT for FHE | C++ only |
| `ldsec/lattigo` (Go) | Apache-2 | Negacyclic NTT for BFV/BGV/CKKS | Apache-2 — incompatible with Reality MIT moat? Actually compatible (Apache-2 is permissive) but Reality's *cross-language deterministic golden file* angle differs from lattigo's Go-only positioning. |
| `consensys/gnark-crypto` (Go) | Apache-2 | NTT over BLS12-381 / BN254 scalar fields for ZK | Apache-2; tightly coupled to gnark; not zero-dep |
| FLINT, NTL | LGPL/GPL | full | License-incompatible with MIT downstream |
| `liboqs` (C) | MIT | Kyber/Dilithium NTT | C only |

Reality's positioning: the MIT pure-Go zero-dep cross-language-deterministic-golden-file NTT toolkit covering 12 variants spanning PQ + ZK + HE + arbitrary-precision arithmetic. **No competitor ships this matrix in pure Go.**

## Concrete recommendations

Tier numbering: N0 = context (NTTCtx with precomputed zetas); N1 = textbook Cooley-Tukey radix-2; N2 = Harvey 2014 fast butterfly; N3 = negacyclic; N4 = Bluestein arbitrary-length; N5 = Rader prime-length; N6 = multi-modular CRT-recombined; N7 = Stockham in-place cache-friendly; N8 = six-step (cache-blocked); N9 = batched (parallel-friendly); N10 = NTT-friendly prime selection; N11 = bit-reversal-elimination via composed transforms.

### N0 — `ntt/context.go` ~120 LOC — **DAY-1 KEYSTONE**

```go
// ntt/context.go
type NTTCtx struct {
    N        int          // transform length (power of 2 in standard variants)
    Q        uint32       // NTT-friendly prime, Q ≡ 1 (mod 2N)
    Psi      uint32       // primitive 2N-th root of unity in F_Q (negacyclic)
    Zeta     uint32       // primitive N-th root of unity = Psi² (cyclic)
    Zetas    []uint32     // precomputed zeta^{br(i)} for i in [0,N) — bit-reversed order
    ZetasInv []uint32     // precomputed zeta^{-br(i)}
    PsisBR   []uint32     // psi^{br(i)} for negacyclic forward
    PsisInvBR[]uint32     // psi^{-br(i)} · n^{-1} merged (saves one pass)
    NInv     uint32       // n^{-1} mod q
    Mont     crypto.MontgomeryCtx  // imports slot 291 T0
}
func NewNTTCtx(n int, q uint32) (*NTTCtx, error) // verifies q ≡ 1 (mod 2n), finds ψ via Tonelli-Shanks (slot 291 T1)
```

**Composes slot 291**: `MontgomeryCtx` for fast butterfly multiply; `Tonelli-Shanks` to construct ψ from a primitive n-th root (ψ² = ζ); `IsQuadraticResidue` to verify validity.

**What it unblocks:** every NTT variant (N1-N11). Build the context once per (n,q) pair; amortize across thousands of transforms.

Refs: Pollard 1971 *The fast Fourier transform in a finite field*; Lyubashevsky-Pöppelmann-Buchmann 2014 §3.1.

### N1 — `ntt/cooley_tukey.go` ~150 LOC — **DAY-1 KEYSTONE (cyclic baseline)**

Forward + inverse Cooley-Tukey decimation-in-time radix-2 over F_q. Identical butterfly structure to `signal/fft.go:63-90` but with:
- twiddle = `Zetas[i]` (uint32 mod q) instead of `cos+i·sin`,
- multiply = `Mont.MulMont(a, ζ)` instead of complex multiply,
- subtract = `(a - b + q) mod q` instead of float64 subtract,
- final inverse-scale = `MulMont(x, NInv)` instead of `*= 1/n`.

```go
func (c *NTTCtx) Forward(a []uint32)         // in-place, output in bit-reversed order
func (c *NTTCtx) Inverse(a []uint32)         // input in BR, output in natural; multiplies by n^{-1}
func (c *NTTCtx) MulPolyCyclic(a, b []uint32) []uint32  // polynomial mul in Z_q[x]/(x^n - 1)
```

**Bit-exact regression**: `Inverse(Forward(a)) == a` for 1024 random length-256 inputs over q=3329. Cross-language golden file: Go produces `testdata/ntt/q3329_n256_forward.json` at 256-bit math/big precision; Python/C++/C# validate.

Refs: Cooley-Tukey 1965 *An algorithm for the machine calculation of complex Fourier series* Math. Comp. 19; Pollard 1971; Knuth TAOCP Vol. 2 §4.6.4.

### N2 — `ntt/harvey.go` ~80 LOC delta over N1 — **HE/PQ HOT-PATH**

David Harvey 2014, *Faster arithmetic for number-theoretic transforms* (J. Symbolic Computation). Replaces the standard Montgomery-multiply-then-conditional-subtract butterfly with a "lazy" reduction that lets intermediate values grow up to 4q before reducing. Net effect: **eliminates ~50% of conditional subtracts in the inner loop, ~30% NTT speedup in practice**, used by Microsoft SEAL since 2018. Same algebraic result as N1; differs only in the modular-reduction schedule. Two-word arithmetic on uint64 to hold the 4q-range intermediate.

```go
func (c *NTTCtx) ForwardHarvey(a []uint32)   // lazy-reduction Cooley-Tukey
func (c *NTTCtx) InverseHarvey(a []uint32)
```

**Bit-exact agreement** with N1 after final reduction — the saturation pin.

Refs: Harvey 2014 *Faster arithmetic for number-theoretic transforms* JSC 60:113-119; Microsoft SEAL `seal/util/ntt.cpp`.

### N3 — `ntt/negacyclic.go` ~100 LOC delta over N1 — **DAY-1 KEYSTONE for PQ**

Operates in Z_q[x]/(x^n + 1) directly — the canonical Kyber/Dilithium/Falcon ring. Pre-twist input by `psi^i` (folded into `Zetas` table to cost zero extra passes), apply standard Cooley-Tukey, post-twist inverse by `psi^{-i}` (folded into `ZetasInv·NInv`). **Saves a factor of 2 vs. zero-padding to 2n + cyclic-NTT**.

```go
func (c *NTTCtx) ForwardNegacyclic(a []uint32)
func (c *NTTCtx) InverseNegacyclic(a []uint32)
func (c *NTTCtx) MulPolyNegacyclic(a, b []uint32) []uint32  // mul in Z_q[x]/(x^n + 1)
```

**Without N3, slot 211 (Kyber/Dilithium/Falcon) is unshippable** — the entire FIPS-203/204 spec is written in terms of negacyclic NTT.

Refs: Lyubashevsky-Peikert-Regev 2010 *On ideal lattices and learning with errors over rings* (the founding paper); Longa-Naehrig 2016 *Speeding up the number theoretic transform for faster ideal-lattice-based cryptography*; FIPS-203 §4.3; Roy-Vercauteren-Mentens-Chen-Verbauwhede 2014 *Compact ring-LWE cryptoprocessor*; Seiler-Lyubashevsky-Schwabe 2018 *Faster AVX2 optimized NTT for Ring-LWE*.

### N4 — `ntt/bluestein.go` ~180 LOC

Bluestein 1970 chirp-Z algorithm: handles arbitrary length n (not just powers of 2) by reducing length-n NTT to a length-2^⌈log2(2n-1)⌉ cyclic NTT via the chirp identity `nk = (n² + k² - (n-k)²)/2`. Useful when downstream needs n=192 (Falcon-512 has n=512 but some lattice schemes use n ∈ {192, 384, 768} — non-power-of-2). Composes N1 internally.

```go
func BluesteinNTT(a []uint32, q uint32) []uint32  // arbitrary length
```

Refs: Bluestein 1970 *A linear filtering approach to the computation of the discrete Fourier transform*.

### N5 — `ntt/rader.go` ~140 LOC

Rader 1968 prime-length NTT: reduces a length-p (prime) NTT to a length-(p-1) cyclic NTT plus a primitive root permutation. Useful for prime-length transforms in Reed-Solomon over F_p (slot 210). Less critical than N4; ship if a consumer requests prime-length explicitly.

Refs: Rader 1968 *Discrete Fourier transforms when the number of data samples is prime*.

### N6 — `ntt/multimodular.go` ~280 LOC — **HIGHEST-LEVERAGE for ZK**

Multi-modular CRT-recombined NTT: when coefficients exceed uint32 range (e.g. ZK proofs over BLS12-381 scalar field r ≈ 2^255), pick 3 NTT-friendly primes q_1, q_2, q_3 each ≈ 2^62, run forward-NTT mod q_i for i=1,2,3, pointwise-multiply, run inverse-NTT mod q_i, recombine via Garner-CRT (slot 291 T3) into a `*big.Int` coefficient. Three primes give a ~186-bit result coefficient; six primes give ~372-bit; etc.

```go
type MultiModularNTTCtx struct {
    Sub []*NTTCtx        // one NTTCtx per residue prime
    Garner crypto.GarnerCtx  // slot 291 T3
}
func NewMultiModularNTTCtx(n int, coefficientBits int) *MultiModularNTTCtx  // auto-picks enough primes
func (c *MultiModularNTTCtx) MulPolyBig(a, b []*big.Int) []*big.Int
```

**What it unblocks:**
- zkSNARK prover over BLS12-381 scalar field (slot 200, 175): polynomial multiplication of 2^20-degree polynomials with 255-bit coefficients in O(n log n).
- Schönhage-Strassen integer multiplication (frontier for slot 057 T1-BIGINT).
- BFV/BGV/CKKS multi-prime modulus chains (slot 211 L24, full FHE — moduli are products of NTT-friendly primes).

Refs: Schönhage-Strassen 1971 *Schnelle Multiplikation großer Zahlen* Computing 7:281-292; Bernstein 2008 *Fast multiplication and its applications* §9.

### N7 — `ntt/stockham.go` ~180 LOC

Stockham 1966 auto-sort variant: bit-reversal-free (uses ping-pong buffer instead of in-place permutation) — better cache behavior than Cooley-Tukey at large n because index arithmetic is sequential, not bit-reversed. Costs 2x memory (two buffers instead of one). Used in HPC FFT libraries (FFTW's `out-of-place` mode).

```go
func (c *NTTCtx) ForwardStockham(in, out []uint32)  // out-of-place, no bit-reversal
```

Refs: Stockham 1966 *High-speed convolution and correlation* AFIPS Conf. Proc. 28; van Loan 1992 *Computational Frameworks for the Fast Fourier Transform* §1.7.

### N8 — `ntt/sixstep.go` ~220 LOC

Bailey 1989 / Frigo-Johnson six-step "cache-blocked" NTT: factor n = n_1 · n_2 with n_1 ≈ n_2 ≈ √n, reshape input as n_1×n_2 matrix, apply n_2 length-n_1 NTTs row-wise, multiply by twiddle matrix elementwise, transpose, apply n_1 length-n_2 NTTs row-wise, transpose. **Cache-friendly above n ≈ 2^20** (above L2 cache). Unblocks ZK prover on million-degree polynomials.

Refs: Bailey 1989 *FFTs in external or hierarchical memory* J. Supercomputing 4; Frigo-Johnson 2005 *The design and implementation of FFTW3* Proc. IEEE 93.

### N9 — `ntt/batched.go` ~120 LOC

Batched NTT: process k polynomials in parallel by interleaving coefficient layouts. Each goroutine handles a contiguous batch; cache-friendly because butterfly twiddles are reused across k polynomials. **Used by Kyber-batched-signing** (some HSMs sign 1000 messages/sec — the per-message NTT cost dominates).

Refs: Sinha-Kim-Reagen 2022 *Optimization of the AVX2 NTT for batched Kyber*.

### N10 — `ntt/primes.go` ~80 LOC

`FindNTTPrime(n int, minBits int) uint32` — search for smallest prime q ≥ 2^minBits with q ≡ 1 (mod 2n). Algorithm: start at q = ⌈2^minBits / (2n)⌉ · 2n + 1, increment by 2n, test primality via Miller-Rabin (`crypto/prime.go:26`), return first prime found.

Companion `FindPrimitiveRoot(q uint32) uint32` — find smallest primitive root g of F_q* by testing g^((q-1)/p) ≠ 1 for each prime divisor p of q-1.

**What it unblocks:** dynamic NTT context creation for arbitrary (n, coefficient-bits) requests — used by N6 multi-modular and by user-defined ZK circuits with custom field sizes.

Refs: Knuth TAOCP Vol. 2 §4.5.4 (primitive roots); FIPS-203 Appendix B.

### N11 — `ntt/composed.go` ~140 LOC

Bit-reversal-elimination via composed transforms: for `(NTT^{-1} ∘ pointwise-mul ∘ NTT)` polynomial multiplication, the bit-reversal permutations at output of forward and input of inverse cancel. Skip both. **~10% speedup, zero algorithmic loss.** Standard optimization in Kyber/Dilithium reference impls.

```go
func (c *NTTCtx) MulPolyFast(a, b []uint32) []uint32  // skips middle bit-reversal
```

Refs: Pöppelmann-Oder-Güneysu 2015 *High-performance ideal lattice-based cryptography on 8-bit ATxmega microcontrollers*.

### Day-1 PR shape

**Singular cheapest, highest-immediate-value**: N0 (NTTCtx) + N1 (Cooley-Tukey) + N3 (negacyclic) + N10 (prime selection) + N11 (composed bit-reversal-skip). ~470 LOC total. Single PR. Directly unblocks slot 211 Kyber/Dilithium/Falcon (negacyclic is mandatory) and provides the substrate for slot 200 zkSNARK prover (cyclic + polynomial multiplication).

**Singular highest-strategic-value follow-up**: N6 (multi-modular CRT-recombined NTT). ~280 LOC. Single primitive that unblocks ALL of zkSNARK prover work over BLS12-381 / BN254 / Goldilocks scalar fields. Without it, ZK prover hot-path remains O(n²).

**Phase-2 (HPC + perf)**: N2 Harvey + N7 Stockham + N8 six-step + N9 batched. ~600 LOC. ~2-3x speedup on million-degree polynomials.

## Cross-validation pin opportunities (R-MUTUAL-CROSS-VALIDATION 3/3)

1. **Cooley-Tukey ≡ Harvey ≡ negacyclic-then-cyclic-on-2n** — three independent algorithms compute polynomial mul in Z_q[x]/(x^n+1); bit-exact agreement on 1,000 random length-256 inputs over Kyber q=3329. **3-way pin.**
2. **`Inverse(Forward(x)) == x` round-trip** — 10⁴ random length-1024 vectors over Falcon q=12289; bit-exact regression. Saturating r-pattern instance.
3. **NTT-mul ≡ schoolbook-O(n²)-mul** — for n=64 (small enough to brute-force), `MulPolyNegacyclic(a, b)` agrees with the `O(n²)` schoolbook reference on 10³ random pairs. Cross-validates the entire NTT pipeline (forward + pointwise + inverse) against an algorithmically independent baseline. **3rd independent algorithm in the pin.**
4. **Multi-modular NTT ≡ math/big direct multiplication** — for 1,000 random pairs of length-1024 polynomials with 200-bit coefficients, N6 multi-modular result equals `*big.Int`-based schoolbook polynomial multiplication. Validates the CRT recombination.
5. **Bluestein-NTT ≡ zero-padded-cyclic-NTT** — for n=192 (non-power-of-2), Bluestein result agrees with zero-padded length-512 cyclic NTT trimmed to 192. **3-way pin** if also cross-validated against schoolbook O(n²).
6. **Cross-language golden file** — Go generates `testdata/ntt/{q3329_n256, q8380417_n256, q12289_n1024}_negacyclic.json` (forward output + inverse-of-forward for 30 random inputs each); Python/C++/C# validate at byte-exactness (NTT is over F_q so ALL precision is exact, tolerance = 0).

These pins saturate R-MUTUAL-CROSS-VALIDATION 3/3: each NTT variant has at least 2 algorithmically independent alternatives (Harvey vs. textbook vs. schoolbook; negacyclic vs. zero-padded-cyclic; multi-modular vs. math/big). Tolerance is exactly 0 (modular arithmetic in F_q is bit-exact).

## Cross-cutting

- **Slot 211-new-lattice-crypto L3+L4** ← N0+N1+N3 land directly as the slot 211 NTT prerequisite; slot 211's `lattice/ntt/{kyber,dilithium,falcon}.go` become 30-LOC wrappers that call `ntt.NewNTTCtx(256, 3329)` etc. **Without slot 293, slot 211 cannot ship.**
- **Slot 200-synergy-zkmark-info / 175-synergy-zkmark-crypto / 147-zkmark-missing / 148-zkmark-sota / 150-zkmark-perf** ← N6 multi-modular NTT is the prover hot-path workhorse; without it, prover is O(n²) and unshippable. Slot 200 also consumes N1 cyclic-NTT for FRI low-degree-extension testing.
- **Slot 292-new-elliptic-curves T17 (Velu's formulae)** ← N1 cyclic-NTT consumed when isogeny degree ℓ ≥ 64. Optional.
- **Slot 290-new-galois-theory T2 (Berlekamp polynomial factoring over F_p)** ← N1 cyclic-NTT speeds up the polynomial-mul step inside Berlekamp from O(n²) to O(n log n) for high-degree polynomials.
- **Slot 291-new-modular-arithmetic** ← bidirectional. Slot 293 N0 imports slot 291's `MontgomeryCtx`, `Tonelli-Shanks`, `IsQuadraticResidue`. Slot 293 N6 imports slot 291's `GarnerCRT`. Slot 291 T4 NTT line item is **subsumed by** slot 293 N0+N1.
- **Slot 132-signal-missing / 134-signal-api** ← N4 Bluestein structure (chirp-Z) parallels float64 Bluestein-FFT; share pseudocode + algorithm doc. Independent implementations (different element type) but shared design notes.
- **Schönhage-Strassen integer multiplication (slot 057 T14 frontier)** ← N6 multi-modular NTT is the algorithmic engine for SS-style integer multiplication when n exceeds the math/big Toom-Cook crossover. Defer unless a consumer hits that regime.
- **Reed-Solomon decoding over F_p (slot 210)** ← N1 cyclic-NTT for syndrome computation; N5 Rader for prime-length codes.
- **Microsoft SEAL / TFHE / CKKS / BFV / BGV (slot 211 L23+L24+L25)** ← N6 multi-modular + N2 Harvey + N9 batched are the production-grade combination Microsoft SEAL uses. Reality MIT pure-Go shipping equivalent feature-set at golden-file-cross-language-determinism is the unique angle.

## Sources

- `C:\limitless\foundation\reality\signal\fft.go:21-91` (Cooley-Tukey float64 — algorithmic blueprint, bit-reversal util reusable).
- `C:\limitless\foundation\reality\signal\fft.go:101-127` (IFFT conjugate method — does NOT translate to F_q; needs ζ^{-1} approach instead).
- `C:\limitless\foundation\reality\crypto\modular.go:20-75` (ModPow, ModInverse — used for zeta-table precomputation).
- `C:\limitless\foundation\reality\crypto\prime.go:26-59` (Miller-Rabin — used in N10 NTT-friendly prime search).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\211-new-lattice-crypto.md:43-45` (L3 forward-NTT + L4 inverse-NTT spec — slot 293 owns the generic implementation; slot 211 instantiates).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\291-new-modular-arithmetic.md:130-153` (T4 NTT sketch — slot 293 expands to 12 variants).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\292-new-elliptic-curves.md:21,31` (Velu's formulae optional NTT consumer).
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md:293` (slot 293 line definition: cooley-tukey, harvey, butterfly negacyclic).
- Cooley-Tukey 1965, *An algorithm for the machine calculation of complex Fourier series*, Math. Comp. 19(90):297-301.
- Pollard 1971, *The fast Fourier transform in a finite field*, Math. Comp. 25(114):365-374.
- Harvey 2014, *Faster arithmetic for number-theoretic transforms*, J. Symbolic Computation 60:113-119.
- Lyubashevsky-Peikert-Regev 2010, *On ideal lattices and learning with errors over rings*, Eurocrypt 2010.
- Longa-Naehrig 2016, *Speeding up the number theoretic transform for faster ideal-lattice-based cryptography*, CANS 2016.
- Roy-Vercauteren-Mentens-Chen-Verbauwhede 2014, *Compact ring-LWE cryptoprocessor*, CHES 2014.
- Seiler-Lyubashevsky-Schwabe 2018, *Faster AVX2 optimized NTT for Ring-LWE*, IACR ePrint 2018/039.
- Bluestein 1970, *A linear filtering approach to the computation of the discrete Fourier transform*, IEEE Trans. AU 18:451-455.
- Rader 1968, *Discrete Fourier transforms when the number of data samples is prime*, Proc. IEEE 56:1107-1108.
- Stockham 1966, *High-speed convolution and correlation*, AFIPS Conf. Proc. 28:229-233.
- Bailey 1989, *FFTs in external or hierarchical memory*, J. Supercomputing 4:23-35.
- Frigo-Johnson 2005, *The design and implementation of FFTW3*, Proc. IEEE 93(2):216-231.
- Pöppelmann-Oder-Güneysu 2015, *High-performance ideal lattice-based cryptography on 8-bit ATxmega microcontrollers*, LATINCRYPT 2015.
- Schönhage-Strassen 1971, *Schnelle Multiplikation großer Zahlen*, Computing 7(3-4):281-292.
- Bernstein 2008, *Fast multiplication and its applications*, AMS Proc. 44:325-384.
- FIPS-203 (Kyber/ML-KEM, NIST 2024) §4.3 (negacyclic NTT specification with q=3329, n=256).
- FIPS-204 (Dilithium/ML-DSA, NIST 2024) §4 (negacyclic NTT with q=8380417, n=256).
- Microsoft SEAL `seal/util/ntt.cpp` (Harvey-2014 reference C++ implementation).
- gnark-crypto `field/fft/` (BN254 / BLS12-381 NTT for ZK).
- van Loan 1992, *Computational Frameworks for the Fast Fourier Transform*, SIAM §1.7.
- Knuth TAOCP Vol. 2 *Seminumerical Algorithms* §4.6.4 (FFT/NTT).
