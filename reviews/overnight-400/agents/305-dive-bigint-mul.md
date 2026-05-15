# 305 — dive-bigint-mul (Karatsuba / Toom-Cook / SS / Fürer audit of reality's big-int usage)

## Headline
Reality has **zero `math/big` usage** in source code (uint64-only crypto); big-int multiply algorithms (Karatsuba, Toom-Cook, SS, Fürer) are a non-issue today, but the moment slot 057-T1-BIGINT or slot 211 NTT lands, the right call is to lean on Go's stdlib `math/big` (schoolbook + Karatsuba only — *not* Toom-Cook as the prompt assumed) and ship Montgomery wrapping (slot 291) on top. Schönhage-Strassen and Fürer/Harvey-van der Hoeven are pure frontier; defer indefinitely.

## Findings (existing usage)

- **No `"math/big"` import anywhere in `reality/*.go` source.** `Grep` for `"math/big"` and `big.Int|big.NewInt|big.Mul|big.Exp` over `*.go` returns 0 hits in non-review files. Only the `reviews/overnight-400/` folder mentions math/big (in prior agent prose).
- **`crypto/prime.go` and `crypto/modular.go` are uint64-only.** `IsPrime`, `MillerRabin`, `ModPow`, `ModInverse`, `ChineseRemainder` all operate on `uint64`. `crypto/prime.go:284` defines `mulmod(a,b,m uint64) uint64` as Russian peasant (the comment claims "Karatsuba-like 128-bit decomposition" but the implementation is shift-and-add — see slot 056 issue **CRY-NUM-6** and slot 056 fix **CRY-MULMOD-OPT** which already proposed `bits.Mul64`+`bits.Div64`).
- **`IsPrime` is deterministic up to 3.317×10²⁴** with a 7-witness set, well inside `uint64`. There is **no path in current reality that needs >64-bit primes** — no RSA, no DSA, no DH, no ECC scalar multiply, no ECDSA, no BLS pairing, no zk proof system shipped. Slots 057/058/059/060/175/200/211/213/214/290/291/292/293 all flag this gap; none have landed.
- **The "Karatsuba-like" comment in `crypto/prime.go:283` is a misnomer.** Russian peasant doubling is *not* Karatsuba; it is the doubling counterpart of binary exponentiation. Pure cosmetic, but it's the only mention of Karatsuba in actual source.
- **No re-implementation of any big-int arithmetic.** Reality does not violate the "zero deps / use stdlib / no reinvention" rule on this axis — there is simply nothing to violate yet.
- **Sibling-slot agreement.** Slot 291 (`new-modular-arithmetic`) explicitly recommends: *do not* re-implement Karatsuba/Toom-Cook over big.Int — inherit from `math/big`. Slot 060 (`crypto-perf`) charts the crossover regimes. Slot 057 (`crypto-missing`) catalogs the BigInt API that should be added. Slot 148 (`zkmark-sota`) notes gnark-crypto's algorithmic patterns translate 1:1.

## Algorithmic facts that constrain the recommendation

Verified directly against `C:\Program Files\Go\src\math\big\natmul.go`:

```
natmul.go:12:  var karatsubaThreshold = 40       // word count crossover for Mul
natmul.go:68:  var basicSqrThreshold = 12        // word count crossover into karatsuba sqr
natmul.go:69:  var karatsubaSqrThreshold = 80    // word count crossover into karatsuba sqr proper
```

Word size on amd64/arm64 = 64 bits, so on those platforms:

| Algorithm in Go stdlib | Crossover (operand bit length) |
|---|---|
| Schoolbook (basicMul) | n < 2,560 bits (40 words) for general Mul |
| Karatsuba (karatsubaMul) | 2,560 bits ≤ n |
| **Toom-Cook 3-way** | **NOT IN MATH/BIG** |
| **Schönhage-Strassen / FFT** | **NOT IN MATH/BIG** |
| Burnikel-Ziegler | division only, not multiplication |

**Correction to the prompt's premise.** The audit-task description claimed `math/big` "implements Karatsuba and Toom-Cook 3-way, switching at threshold ~80 words for Karatsuba and ~250 words for Toom-Cook 3-way". This is **incorrect for upstream Go** as of the current toolchain (`C:\Program Files\Go\src\math\big\natmul.go`, lines 9–12). Go's `math/big` only does schoolbook → Karatsuba; the next algorithm up the ladder is *missing*. Go has had open proposals (golang/go #20461 Toom-Cook, golang/go #21963 SS) but neither has merged. GMP/MPIR ship Toom-2.5/3/4/6.5/8.5 and SS; Go does not.

This actually *strengthens* the slot's "math/big is enough" argument for current reality (Miller-Rabin on 4096-bit RSA-class moduli = 64 words = squarely in **schoolbook** regime — Karatsuba doesn't even kick in), but it *weakens* it the moment reality's planned ZK / pairing work goes wide:

- **BLS12-381 base field**: 381 bits = 6 words. Schoolbook. Karatsuba never fires. Need fixed-size 6-word multiply (slot 291 Montgomery context).
- **BN254 base field**: 254 bits = 4 words. Schoolbook. Same story.
- **RSA 2048**: 2048 bits = 32 words. Schoolbook (still below 40-word threshold). One Miller-Rabin round ≈ 32 mulmods of 32-word ops ≈ 1024 word·word multiplies plus the modular reduction; ~10 µs on modern hardware. Adequate.
- **RSA 4096**: 64 words. **Karatsuba kicks in.** Still adequate (~25–50 µs per mulmod).
- **RSA 8192 / 16384**: 128 / 256 words. Karatsuba dominates. No Toom-Cook in stdlib costs ~30 % vs GMP, not catastrophic.
- **zkSNARK NTT-of-million-points**: 2²⁰ point NTT modulo BLS scalar field — *not* a big-int multiply at all; that is fixed-size 4-word modular arithmetic in a tight loop, which `math/big` is *terrible* at (it allocates) and where slot 291's Montgomery context + slot 293's NTT must own the path.

## Where reality could ever need Toom-Cook / SS / Fürer

Honest catalog:

1. **Toom-Cook 3-way (Cook 1966 / Toom 1963)**: O(n^1.465). Wins from ~3,000–15,000 bit operands. Reality's **only** plausible consumer would be RSA-4096+ key generation in a primality loop, or an experimental cryptosystem with multi-thousand-bit moduli. None planned. **DEFER.**
2. **Toom-Cook 4-way (Bodrato-Zanoni 2007)**: O(n^1.404). Wins from ~15,000–50,000 bit. **No plausible reality consumer.**
3. **Schönhage-Strassen 1971**: O(n log n log log n) via Fermat-ring NTT and modular FFT. Wins from ~33,000 bit (GMP `MUL_FFT_THRESHOLD`) or larger; in practice ≥ 50,000 bits. **No plausible reality consumer.** Possibly relevant if reality ever computes proof-of-Pell-equation, Lehmer's totient problem, RSA-2048 challenge-style factorizations, or class-number computations — but these are research, not Limitless product.
4. **Fürer 2007**: O(n · log n · 2^O(log* n)). Galactic algorithm; *worse* than SS below ~2^25 bits in practice; no production implementation exists. **Strict frontier — never write.**
5. **Harvey-van der Hoeven 2019**: O(n log n) — the optimal-up-to-constant-factor multiplication. Same story: galactic; no production implementation. The crossover point with SS is conjectured around 2^(2^36) bits. **Strict frontier — never write.**

## Concrete recommendations

1. **No bigint algorithm work in slot 305 itself.** Reality has no big-int arithmetic at all today. The right ordering is: slot 057-T1-BIGINT introduces a `*big.Int` API for `IsPrimeBig`, `ModPowBig`, `ChineseRemainderBig` ⇒ slot 291 wraps with Montgomery/Barrett ⇒ slot 293 ships NTT. Slot 305's contribution is documentation + benchmark scaffold, not new code.

2. **Document the Go `math/big` reality on CLAUDE.md (≤ 8 lines).** Add a short "Big-integer algorithms" subsection stating: (a) reality uses Go stdlib `math/big` for any future big-int path; (b) Go's `math/big` ships schoolbook + Karatsuba *only* (no Toom-Cook, no FFT — corrected against natmul.go); (c) crossover is 40 words (≈ 2,560 bits) on 64-bit hosts; (d) below that, schoolbook; (e) reality's planned consumers (RSA up to 4096, BLS12-381, BN254, all primality testing) sit *at or below* this threshold, so Karatsuba is essentially never invoked and stdlib is sufficient; (f) Toom-Cook / SS / Fürer / HvdH are out-of-scope.

3. **Cheapest day-1 PR (~80 LOC).** Add `crypto/bigmul_bench_test.go` with `Benchmark(Mul/Karatsuba/Schoolbook)` over operand sizes {64, 256, 1024, 4096, 8192, 16384, 32768} bits. Two benefits: (a) we *prove* the Karatsuba-vs-schoolbook crossover rather than assume it; (b) gives a regression baseline if Go ever lands Toom-Cook (proposal #20461). Pseudocode:
   ```go
   func BenchmarkBigMul(b *testing.B) {
       for _, bits := range []int{64, 256, 1024, 4096, 8192, 16384, 32768} {
           x := randBig(bits); y := randBig(bits)
           b.Run(fmt.Sprintf("bits=%d", bits), func(b *testing.B) {
               z := new(big.Int)
               for i := 0; i < b.N; i++ { z.Mul(x, y) }
           })
       }
   }
   ```
   Place under `crypto/` once `*big.Int` lands via slot 057-T1-BIGINT; until then keep this benchmark in a `// build ignore` block of the slot 305 review.

4. **R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunity (defer until slot 211/293 NTT and slot 290 polynomial multiply land).** When NTT-based or Karatsuba-based custom multipliers exist, write a golden test asserting `schoolbook(x,y) ≡ math/big.Mul(x,y) ≡ NTT(x,y)` over ≥ 30 random (x,y) pairs at sizes {64, 256, 1024, 4096, 16384} bits with seeded `crypto/rng`. Three independent algorithms ⇒ saturates R-MUTUAL-CROSS-VALIDATION. Place in `crypto/bigmul_cross_test.go`.

5. **Do NOT write Karatsuba-over-`*big.Int`.** It exists in `math/big` already; reimplementing in reality is dead-weight. The *only* place a reality-native Karatsuba is justified is over **polynomials** (slot 290 `galois/poly_z.go` poly_mul) where math/big multiplies coefficients but the polynomial layer needs poly·poly. That is slot 290's responsibility, not slot 305's.

6. **Do NOT write Toom-Cook, Schönhage-Strassen, or Fürer.** Zero current consumer. The work product would be a research paper, not a Limitless feature.

7. **Fix the misleading comment in `crypto/prime.go:283`.** Replace `// via two 64-bit halves (Karatsuba-like decomposition)` with `// via Russian-peasant doubling (binary expansion of b)`. 1-line edit; aligns comment with implementation. (Slot 056 already has this on its punchlist as **CRY-NUM-6**; slot 305 confirms.)

8. **For Pistachio's 60-FPS hot path** (cited in CLAUDE.md design rule 3): big-int multiplication is *never* on a hot path. If a 60 FPS path ever pulls a `*big.Int` it is a design error. Document this explicitly in slot 305 review and any future bigint API doc-comment.

## Threshold-table cheat-sheet (for inclusion in reality docs)

| Operand bit-length | Words (64-bit host) | Algorithm Go uses | Reality consumer |
|---|---|---|---|
| ≤ 64 | 1 | native uint64 | current `crypto/*` (Miller-Rabin uint64) |
| 64–2,560 | 1–40 | math/big schoolbook | RSA-2048, BLS12-381, BN254, Curve25519, secp256k1 — *all* planned reality crypto |
| 2,560–∞ | ≥ 40 | math/big Karatsuba | RSA-4096+, multi-thousand-bit moduli (rare) |
| ≥ ~3,000 (in GMP) | ≥ ~50 | (GMP) Toom-Cook 3 — **Go does not have this** | n/a in reality |
| ≥ ~15,000 (in GMP) | ≥ ~250 | (GMP) Toom-Cook 4/6.5/8.5 — **Go does not have this** | n/a in reality |
| ≥ ~33,000 (in GMP) | ≥ ~520 | (GMP) Schönhage-Strassen — **Go does not have this** | n/a in reality |
| ≥ ~10⁹ | ≥ ~10⁷ | (theoretical) Fürer / Harvey-vdH — no production impl | n/a anywhere |

## Sources

- `C:\limitless\foundation\reality\crypto\prime.go` (lines 17–132 IsPrime/MillerRabin; 282–306 mulmod/addmod) — uint64-only, no math/big.
- `C:\limitless\foundation\reality\crypto\modular.go` (lines 20–40 ModPow; 96–135 ChineseRemainder) — uint64-only.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\056-crypto-numerics.md` lines 115–117, 252 — flagged the "Karatsuba-like" misnomer; CRY-MULMOD-OPT punchlist item.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\057-crypto-missing.md` — catalogs missing big-int crypto API surface.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\060-crypto-perf.md` lines 86–110, 256–269, 385–387 — schoolbook/Karatsuba/Toom-3/NTT crossover regime analysis.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\148-zkmark-sota.md` line 91 — gnark-crypto reference for 4-limb Karatsuba.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\291-new-modular-arithmetic.md` lines 39–50, 225–229 — "do NOT re-implement Karatsuba over big.Int; defer SS indefinitely."
- `C:\Program Files\Go\src\math\big\natmul.go` lines 9–12, 36–38, 65–69, 100, 115, 171, 263 — proves Go ships only schoolbook + Karatsuba.
- `C:\Program Files\Go\src\math\big\calibrate.md` lines 119, 149, 166 — calibration justifying `karatsubaThreshold = 40`, `basicSqrThreshold = 12`, `karatsubaSqrThreshold = 80`.
- Karatsuba & Ofman (1962) "Multiplication of multidigit numbers on automata", Doklady Akad. Nauk SSSR 145.
- Toom (1963) "The complexity of a scheme of functional elements realizing the multiplication of integers"; Cook (1966) PhD thesis "On the Minimum Computation Time of Functions".
- Schönhage & Strassen (1971) "Schnelle Multiplikation großer Zahlen", Computing 7(3–4):281–292.
- Fürer (2007) "Faster Integer Multiplication", STOC 2007 (and 2009 SICOMP refinement).
- Harvey & van der Hoeven (2019) "Integer multiplication in time O(n log n)", HAL preprint hal-02070778; Annals of Mathematics 2021.
- Bodrato & Zanoni (2007) "Integer Multiplication: Toom-Cook Algorithms", ISSAC 2007 — Toom-4 / Toom-6.5 / Toom-8.5 calibration.
- Burnikel & Ziegler (1998) MPI-I-98-1-022 — Go uses this for division only, not multiplication.
- GMP documentation §15.1 "Multiplication Algorithms" — `MUL_TOOM22_THRESHOLD` ≈ 30, `MUL_TOOM33_THRESHOLD` ≈ 90, `MUL_FFT_THRESHOLD` ≈ 3000–5000 limbs depending on chip.
- Go issues golang/go#20461 (Toom-Cook proposal, open) and golang/go#21963 (SS proposal, closed-as-deferred) — confirms upstream Go does not ship them.
