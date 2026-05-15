# 321 — dive-finite-field (GF(2^m) Mul / Inv / Sqrt / Irreducibility audit)

## Headline
reality v0.10.0 ships **zero binary-extension-field arithmetic** — `crypto/modular.go` is GF(p) prime-field uint64 only (ModPow / ModInverse / ExtendedGCD / CRT), and the `coding/galois/` keystone scoped by slot 210 / 290 / 320 still does not exist; this dive narrows that scope to the 6 GF(2^m) primitives needed to unblock AES MixColumns, RS-(255, 223) CCSDS deep-space, BCH POCSAG, ECC-over-binary-curves (NIST B/K), and binary-tower-field STARKs (Diamond-Posen 2024), and ranks them T0 (Exp/Log table GF(2^8)) through T5 (bit-sliced SIMD).

## Findings

### F1 — repo-wide GF(2^m) surface = zero callable hits
- Grep `GF\(2|Galois|Frobenius|Itoh|Tsujii|Rabin\b.*irreducible|Ben-Or|PCLMUL|CLMUL|carryless|Conway` over `**/*.go` returns **zero** callable matches (only Conway's *Game of Life* in `chaos/systems.go:139` and Carmichael number test data in `crypto/crypto_test.go:81`).
- `crypto/modular.go:20-40` ships `ModPow` (GF(p) binary exp), `:54-75` `ModInverse` (extended Euclidean over Z), `:96-135` `ChineseRemainder`. All uint64 prime-field; **no binary extension field tower, no irreducible polynomial table, no Frobenius square-root, no carryless multiply.**
- `crypto/prime.go` Miller-Rabin 1976 / 1980 (note: this is a *primality* test in Z, NOT Rabin's 1980 *irreducibility* test in F_q[x]; namespace collision worth flagging).
- No `coding/`, no `field/`, no `gf2m.go`, no `binary/` directory anywhere. `linalg/` is float64-only — no `BitMatrix`.
- The MASTER_PLAN slot 210 (full coding/ scope) and slot 290 (galois-theory) both name `coding/galois/gf2m.go` as KEYSTONE C3 (slot 210) / fast-path for binary extensions (slot 290 T13 `gfpn.go` covers odd-prime extensions). Slot 320 (dive-error-correction) confirms: without GF(2^m), every code from RS onward stalls.

### F2 — slot 320 already named the day-1 GF(2^8) tabular implementation; this dive zooms in
Slot 320 F4 names the keystone explicitly: `Galois{Modulus uint16, ExpTable, LogTable [256]uint8}` GF(2^8) with primitive polynomial `0x11d` (AES + RS-CCSDS), `Mul = ExpTable[(LogTable[a]+LogTable[b]) % 255]`, `Inv = ExpTable[255 - LogTable[a]]`. ~280 LOC. This dive **does not re-scope the table** — it audits the *six different multiplication algorithms* and *three different inversion algorithms* a serious GF(2^m) library should expose, and ranks them by deployment ubiquity vs LOC cost.

### F3 — six GF(2^m) multiplication algorithms, ranked
| # | Algorithm | Complexity | Year | Use case | LOC |
|---|---|---|---|---|---:|
| M1 | Schoolbook bitwise (shift + XOR + reduce) | O(m^2) bit ops | textbook | reference / m > 16 generic | ~30 |
| M2 | Exp/Log table lookup (`a*b = exp[(log a + log b) mod (q-1)]`) | O(1), 2·q bytes | Berlekamp 1968 | GF(2^4) / GF(2^8) (AES, RS-CCSDS, QR) | ~80 |
| M3 | Russian-peasant carryless (PCLMULQDQ x86 / VMULL ARMv8) | O(m^2 / w) word ops, ~3 ns on Skylake | Intel 2008 / Gueron-Kounavis 2010 | GF(2^64) / GF(2^128) (AES-GCM, BLS12-381 inner field) | ~60 + asm |
| M4 | Karatsuba split-and-recurse | O(m^{log_2 3}) ≈ O(m^{1.585}) | Karatsuba 1962 | GF(2^163) / GF(2^233) NIST B/K curves | ~120 |
| M5 | Field-tower GF(2^8) = GF((2^4)^2) | O(1) sub-field ops | Itoh-Tsujii 1988 / Satoh-Morioka 2003 | hardware AES, bit-sliced AES | ~100 |
| M6 | Bit-sliced word-parallel (process W elements simultaneously) | O(m^2 / W) per element | Käsper-Schwabe 2009 | bulk RS / LDPC encode at SIMD width | ~200 |

reality should ship M1 (reference, all m), M2 (table, fast m=8), M4 (Karatsuba, m up to 256 portable Go); M3/M5/M6 are frontier (need either `cpu.X86.HasPCLMULQDQ` runtime detection or sub-field tower discipline). Per CLAUDE.md design rule 6 ("Reimplement from first principles. Do not wrap existing libraries.") and rule 3 ("No allocations in hot paths") — no `crypto/aes.NewCipher`-style wrapping; M3 PCLMULQDQ would need plan9 asm in `gf2m_amd64.s`. Frontier; defer.

### F4 — three GF(2^m) inversion algorithms, ranked
| # | Algorithm | Complexity | Year | Constant-time? | LOC |
|---|---|---|---|---|---:|
| I1 | Exp/Log table (`inv(a) = exp[q-1 - log a]`) | O(1), reuse mul tables | Berlekamp 1968 | NO (table lookup is data-dependent) | ~10 |
| I2 | Extended-Euclidean over F_2[x] (binary extended gcd) | O(m^2) | Knuth TAOCP §4.6.2 | NO (control-flow data-dep) | ~80 |
| I3 | Itoh-Tsujii (Fermat: `a^{-1} = a^{2^m - 2}`, addition-chain) | O(log m) field-mult + (m-1) Frobenius | Itoh-Tsujii 1988 | YES (fixed addition chain) | ~80 |

**Itoh-Tsujii** 1988 is the canonical fast inversion in GF(2^m): factor `a^{q-2} = a^{2(2^{m-1} − 1)}` using the addition chain `1 → 2 → 3 → 6 → 7 → ...` for `m-1` and `m-1` Frobenius squarings (each is one multiplication via Exp/Log or one squaring by table). For m=8, ~3 multiplications + 7 squarings vs ~m^2 = 64 bit-ops for ext-Euclidean. Slot 292 (ECC) and AES SubByte both use I3.

For **constant-time crypto** (which slot 292 will care about once ECDSA / Schnorr lands), I3 + a Niederreiter-style branchless reduction is the only path. Table lookups (I1) leak via cache timing — the modern (2020-2026) AES bit-sliced implementations (Käsper-Schwabe 2009 follow-ups) avoid I1 entirely. Flag for slot 060 (crypto-perf) when ECC arrives.

### F5 — GF(2^m) square root is *free* in characteristic 2
In char 2, the squaring map `x → x^2` is a *field automorphism* (the Frobenius). Therefore every element has exactly one square root, and `sqrt(x) = x^{2^{m-1}}` (Fermat: `x · x^{q-2} = 1` ⇒ `(x^{2^{m-1}})^2 = x^{2^m} = x · x^{2^m - 1} = x · 1 = x`). One-line implementation: `m-1` repeated squarings, or a precomputed 256-byte `SqrtTable` for GF(2^8). **Round-trip property `sqrt(x)^2 == x ∀ x ∈ GF(2^m)`** is the cleanest R-MUTUAL-CROSS-VALIDATION 3/3 pin in the entire field-arithmetic catalog (3 witnesses: Frobenius m-1 squarings, table lookup, brute-force `for y in field { if y*y == x: return y }`).

This contrasts sharply with GF(p) for odd p: there sqrt requires Tonelli-Shanks 1891 (~80 LOC, randomised) or Cipolla 1903. Char-2 sqrt is the *single biggest reason* binary fields are easier to implement than prime fields.

### F6 — irreducibility test: Rabin 1980 + Ben-Or 1981 + factor-distinct-degree
For random irreducible polynomial generation (needed when m is non-standard, e.g. binary-tower STARK fields where m grows), the canonical algorithms:

| # | Algorithm | Complexity | Year | When to use |
|---|---|---|---|---|
| R1 | **Rabin 1980 irreducibility test** — `f` irreducible ⇔ `gcd(f, x^{q^{m/p_i}} - x) = 1` for every prime `p_i \| m` AND `x^{q^m} ≡ x mod f` | O(m^3) field ops | Rabin 1980 | random-and-test irreducible search |
| R2 | **Ben-Or 1981 irreducibility test** — randomised, 1-pass: pick random `g ∈ F_q[x]`, compute `g^{q^m} mod f`; if `≠ g` then `f` is reducible (whp) | O(m^3) field ops | Ben-Or 1981 | faster heuristic; complements R1 |
| R3 | **Distinct-degree factorisation (DDF)** — repeatedly compute `gcd(f, x^{q^k} - x)` for k=1..m/2; `f` irreducible ⇔ no factor for k < m | O(m^3) field ops | von zur Gathen-Shoup 1992 | full factor enumeration; gold standard |
| R4 | **Pre-built table for m ≤ 32** | O(1) | tabulated | 95% of practical use (m ∈ {4, 8, 16, 32, 64, 128, 163, 233, 283, 409, 571}) |

The standard primitive polynomials for AES/RS/NIST-curves are well-known and should ship as a static table:
- m=4: `0x13` (`x^4 + x + 1`)
- m=8: `0x11d` (`x^8 + x^4 + x^3 + x^2 + 1`, AES + RS-CCSDS)
- m=16: `0x1100b` (`x^16 + x^12 + x^3 + x + 1`, CRC-16-CCITT same poly)
- m=32: `0x104c11db7` (CRC-32-IEEE)
- m=64: ECMA-182
- m=128: AES-GCM (`x^128 + x^7 + x^2 + x + 1`)
- m=163: NIST K-163 / B-163 (`x^163 + x^7 + x^6 + x^3 + 1`)
- m=233: NIST K-233 / B-233
- m=283: NIST K-283 / B-283 trinomial
- m=409: NIST K-409 / B-409
- m=571: NIST K-571 / B-571

R1/R2 only fire when a consumer wants a non-standard m (binary-tower-field STARK construction Diamond-Posen 2024 grows `m_{i+1} = 2·m_i` — needs irreducibility tests when extending the tower).

### F7 — field tower GF(2^8) = GF((2^4)^2) is the AES hardware fast path
Satoh-Morioka 2003 / Canright 2005 represent each GF(2^8) element as a degree-1 polynomial over GF(2^4). Inversion in GF((2^4)^2) reduces to: 1 inversion in GF(2^4) (cheap, 16-byte table) + 3 multiplications in GF(2^4). Hardware AES SubByte uses this — gate count drops from ~520 (direct GF(2^8) table) to ~110 (Canright). **Software relevance for reality is limited** (Go can just use the 256-byte table) — but ZK proof systems that need to express AES inside an arithmetic circuit (Schwartz-Zippel-ised constraints) prefer the tower because each GF(2^4) multiplication maps to fewer R1CS constraints. Defer to T3 unless slot 200 (synergy-zkmark-info) pulls.

### F8 — binary-tower STARK fields are the 2026 frontier (Diamond-Posen / Binius)
Diamond-Posen 2024 *Succinct Arguments over Towers of Binary Fields* (Binius / "binary-tower STARKs") replaces the prime-field FFT-friendly Goldilocks/Mersenne31 with a tower `F_2 ⊂ F_{2^2} ⊂ F_{2^4} ⊂ ... ⊂ F_{2^{128}}` where each level is a quadratic extension of the previous. **Properties:**
- Multiplications at the bottom (F_2) are AND gates — constraint cost ≈ 0.
- Reed-Solomon over GF(2^m) replaces RS-over-prime-field as the FRI low-degree-test substrate.
- Prover speedups of 10-50× vs prime-field STARKs reported.
- `zkmark/zkmark.go:280` Halo2-honest-pending placeholder is the natural consumer when slot 200 (synergy-zkmark-info) lands.

A serious reality binary-field package should be **tower-aware** from day 1: the type signature `Field[m]` should compose, so that GF(2^128) = GF((((((F_2)^2)^2)^2)^2)^2)^2)^2 can be constructed by repeated quadratic extension rather than as a flat 128-bit field. Slot 200 should pin this requirement.

### F9 — consumer demand audit
| Consumer | Slot | Field needed | Status without GF(2^m) |
|---|---|---|---|
| AES SubByte / MixColumns | (slot 057 crypto-missing) | GF(2^8) Inv + Mul | blocked |
| RS-(255, 223) CCSDS | 320 | GF(2^8) Mul + Inv + Pow | blocked |
| BCH POCSAG (63, 45, 3) | 320 | GF(2^6) Mul + Inv | blocked |
| Reed-Muller (1, 5) Mariner-9 | 210 | GF(2) BitVector | blocked |
| ECC NIST B-163 / K-163 | 292 | GF(2^163) Mul + Inv (Itoh-Tsujii) | blocked |
| ECC NIST B-571 / K-571 | 292 | GF(2^571) Mul + Inv | blocked |
| AES-GCM authentication | 057 | GF(2^128) Mul (PCLMULQDQ ideal) | blocked |
| Binius / binary-tower STARKs | 200 + 147 | GF(2^m) tower for m=2..128 | blocked |
| Pairing-friendly BLS12-381 | 214 | NOT GF(2^m) (prime field — slot 291) | orthogonal |

7/9 consumers blocked by absence of GF(2^m). Of those 7, **slot 320's RS-(255, 223)** is the cheapest unblock (only needs M2 + I1); slot 292's ECC over binary curves is the most demanding (needs M4 Karatsuba + I3 Itoh-Tsujii at m=163..571, plus constant-time discipline). Slot 200 binary-tower STARKs is the highest-leverage future consumer (whole zkmark backend rests on it).

### F10 — naming / namespace collision flags
- `crypto/prime.go` Miller-Rabin (1980 randomised primality test in Z) shares the surname "Rabin" with Rabin 1980 *irreducibility test in F_q[x]*. When slot 321 lands, namespace `IsIrreducibleRabin` (not `RabinTest`) to disambiguate.
- `coding/galois/` (slot 210, binary fields) vs `crypto/field/` shared sub-package proposed by slot 057 (prime fields): keep them sibling. Do NOT call the binary one `gf.go` — `coding/galois/gf2m.go` is unambiguous.
- Slot 290 (galois-theory) is *Galois group* of polynomials over Q (theoretical Galois theory). Slot 321 is *Galois field* GF(q) arithmetic (computational). Both legitimate uses of "Galois"; consider naming the abstract Galois group package `galoistheory/` to avoid confusion with `coding/galois/`.

### F11 — IEEE 754 edge cases do not apply (per slot 320 F13)
GF(2^m) elements are integers (uint8 / uint16 / uint32 / uint64 / [m]uint64 word-array). Float edge cases vacuous. Per-package edge-case catalog:
- **Zero element**: `Mul(0, *) = 0`, `Inv(0)` is undefined — must return `(0, false)` or panic with `crypto.ErrZeroInverse`. Slot 321 should follow `ModInverse` convention `(uint8, bool)`.
- **One element**: `Mul(1, x) = x`, `Inv(1) = 1`, `Pow(x, 0) = 1`.
- **Primitive element α**: `Pow(α, q-1) = 1`, `Pow(α, k)` for k ∈ [0, q-1] enumerates all non-zero elements.
- **Frobenius idempotent**: `Pow(x, q) = x` for every `x ∈ GF(q)` (this is the defining property of the field — pin as a 3/3 cross-validation across all algorithms).
- **Polynomial overflow**: when constructing GF(2^m) from a non-irreducible modulus, every operation eventually produces zero divisors. The constructor `NewGF(m, modulus)` MUST validate `IsIrreducible(modulus)` once at construction (cost amortised); document the panic vs error mode explicitly.

### F12 — R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities
Five clean 3/3 pins for slot 321 day-1 PR:

- **(P1) GF(2^8) Mul 3-way**: schoolbook (M1) ≡ table-lookup (M2) ≡ Karatsuba (M4) for all 65,536 ordered pairs. Three independent implementations; byte-equality required. ~80 LOC test.
- **(P2) GF(2^8) Sqrt 3-way (F5)**: Frobenius `x^{2^7}` ≡ table-lookup ≡ brute-force `for y { if y*y == x }` for all 256 elements. Cleanest pin in the catalog; ~50 LOC test.
- **(P3) GF(2^8) Inv 3-way**: Exp/Log (I1) ≡ extended-Euclidean (I2) ≡ Itoh-Tsujii (I3) for all 255 non-zero elements + zero-handling agreement. ~80 LOC test.
- **(P4) Frobenius idempotence**: `Pow(x, 256) == x` for all 256 elements of GF(2^8) — single witness per element, but the pin is across all four field-construction paths (raw uint8, polynomial Polynomial[]uint8, tower GF((2^4)^2), generic GF(2^m) at m=8). ~30 LOC test.
- **(P5) Rabin (R1) ≡ Ben-Or (R2) ≡ DDF (R3)**: irreducibility verdict on every degree-≤8 binary polynomial (256 polys total) — all three algorithms must agree. Cross-validates against the **published primitive-polynomial tables** in Lidl-Niederreiter 1997 §3 / Hankerson-Menezes-Vanstone 2004 §A.1 (canonical fixture). ~150 LOC test.

P2 alone saturates a clean 3/3; P1+P3+P5 together saturate three more independent 3/3s.

### F13 — six-primitive day-1 catalog with LOC budget
| Tier | Primitive | Operations | LOC | Cumulative |
|---|---|---|---:|---:|
| T0 | `coding/galois/gf2m.go` Exp/Log table for GF(2^m), m ≤ 16 | Add (XOR), Mul, Inv, Pow, Sqrt (Frobenius) | 280 | 280 |
| T0 | `coding/galois/poly.go` polynomials over GF(2^m) | PolyEval / Add / Mul / Div / Mod / GCD / Derivative / Eval | 120 | 400 |
| T1 | `coding/galois/itoh_tsujii.go` Itoh-Tsujii inversion for GF(2^m) | Inv via fixed addition chain | 80 | 480 |
| T2 | `coding/galois/irreducible.go` Rabin 1980 + Ben-Or 1981 + DDF | IsIrreducibleRabin, IsIrreducibleBenOr, FactorDistinctDegree, RandomIrreducible(m) | 120 | 600 |
| T3 | `coding/galois/tower.go` GF((2^k)^2) quadratic extension constructor | NewQuadraticExtension(F, irreducible) | 100 | 700 |
| T4 | `coding/galois/clmul_amd64.s` PCLMULQDQ carryless mul (frontier) | Mul64 carryless × 2 → uint128 reduce | 60 + asm | 760 |
| T5 | `coding/galois/bitsliced.go` Käsper-Schwabe word-parallel | bulk Encode/Decode for RS / LDPC | 200 | 960 |

T0+T1+T2 = ~600 LOC ships day-1 and unblocks slot 320 (RS, BCH, Reed-Muller), AES SubByte (slot 057), and slot 290 (computational Galois). T3 unblocks AES-circuit ZK consumers. T4/T5 are frontier (asm + SIMD discipline) — defer to slot 060 (crypto-perf) when measured benchmarks justify.

### F14 — cross-language golden-file strategy
Per CLAUDE.md "Golden files are the proof": GF(2^m) golden vectors should pin:
1. **Cayley table** for GF(2^4) (full 16×16 mul table — 256 entries, single JSON file). Trivially cross-validated in Python (`galois` library), C++ (`NTL`), C# (custom).
2. **AES MixColumns column** through ten rounds for a fixed plaintext (the AES test vector NIST FIPS-197 Annex C). 16 bytes × 10 rounds × 4 GF(2^8) muls per col = 640 GF(2^8) operations as one golden vector.
3. **RS-(255, 223) syndrome computation** on a known-bad codeword; 32 syndrome bytes as one golden vector.
4. **Itoh-Tsujii inversion** of every non-zero element of GF(2^8): 255-entry table, one JSON.
5. **Irreducibility verdicts** on degree-1..8 binary polynomials (511 polynomials total, one bit each = 64 bytes of golden output).

Per-function tolerance: **0** (binary-field arithmetic is exact integer arithmetic; no float tolerance needed). This is one of the few packages where golden-file equality is byte-exact across all 4 languages with zero numerical concerns.

## Concrete recommendations

1. **Day-1 PR (~600 LOC) — T0+T1+T2.** Ship `coding/galois/{gf2m.go, poly.go, itoh_tsujii.go, irreducible.go}` as a single PR. T0 alone (Exp/Log table + Polynomial) unblocks slot 320 RS/BCH; T1 (Itoh-Tsujii) provides the constant-time inversion path slot 292 ECC will need; T2 (Rabin/Ben-Or) provides the random-poly-generation needed when m goes non-standard (binary-tower STARKs slot 200). Estimated 2-3 days.

2. **Pin five R-MUTUAL-CROSS-VALIDATION 3/3s in the day-1 PR per F12.** P2 (sqrt round-trip 3-way) is the cleanest in the catalog and saturates trivially — that should be the *first* test written. P5 (irreducibility 3-way over all degree-≤8 polys) cross-validates against Lidl-Niederreiter 1997 Table 3.1 published primitives.

3. **Pre-build static primitive-polynomial table** `IrreducibleTable(m uint8) uint64` for m ∈ {2..16, 32, 64, 128, 163, 233, 283, 409, 571}. Hard-code from Hankerson-Menezes-Vanstone 2004 *Guide to Elliptic Curve Cryptography* §A.1. The function `RandomIrreducible(m, rng)` only fires for non-standard m; for standard m the table is the canonical answer.

4. **Namespace `IsIrreducibleRabin` not `RabinTest`** (collision with crypto/prime Miller-Rabin per F10). Add a doc comment cross-link: "For Rabin's 1980 *primality* test in Z, see crypto/prime.go MillerRabin. This is Rabin's 1980 *irreducibility* test in F_q[x]."

5. **Type signature should be tower-composable from day 1.** Recommended interface:
   ```go
   type Field interface {
       Q() uint64                          // field size
       Add(a, b uint64) uint64
       Mul(a, b uint64) uint64
       Inv(a uint64) (uint64, bool)
       Pow(a uint64, n uint64) uint64
       Sqrt(a uint64) (uint64, bool)       // char 2: always (a^{2^{m-1}}, true)
       Frobenius(a uint64) uint64          // a^p (= a^2 in char 2)
   }
   ```
   So `GF((2^4)^2)` constructed from `GF(2^4)` plus an irreducible polynomial returns the same `Field` type. This unblocks slot 200 binary-tower STARKs without re-architecting later. (May need `[]uint64` for m > 64 — wrap in a polymorphic `Element []uint64`.)

6. **Defer T4 PCLMULQDQ asm path.** Per CLAUDE.md design rule 6 (reimplement-from-first-principles) the M1+M2+M4 portable Go path is the canonical implementation. PCLMULQDQ is a 5-10× perf win for GF(2^128) AES-GCM, but slot 057 should land AES-GCM first to *measure* the bottleneck before adding asm. Cross-link to slot 060 (crypto-perf).

7. **Defer T5 bit-sliced SIMD.** Käsper-Schwabe 2009 is a 2-4× win for bulk RS/LDPC encode and a constant-time AES win, but it's 2026-frontier and benchmarks-driven. Defer until slot 320 RS-(255, 223) ships and a measured perf gap emerges.

8. **Cross-link `crypto/modular.go` (GF(p)) and `coding/galois/gf2m.go` (GF(2^m)) via a shared `crypto/field/` sub-package** as slot 210 §5 recommends. The `Field` interface above lives in `crypto/field/iface.go`; concrete implementations are `crypto/field/gfp.go` (slot 291), `coding/galois/gf2m.go` (slot 321), `coding/galois/gfpn.go` (slot 290 T13 — odd-prime extensions).

9. **Add CLAUDE.md row** when day-1 lands:
   ```
   coding/galois | Finite field arithmetic GF(2^m): table, Itoh-Tsujii, Rabin irreducibility (T0-T2 only at v0.X.0).
   ```
   And cross-reference from `crypto`: "See coding/galois/ for binary extension fields. See galoistheory/ for abstract Galois groups (slot 290)."

10. **Pin the binary-tower-STARK consumer (slot 200).** When T0+T1 land, slot 200 (synergy-zkmark-info) and slot 147 (zkmark-missing) gain Binius (Diamond-Posen 2024) as a viable Halo2-honest-pending replacement. The Tranche-2 path becomes: GF(2^m) → RS-over-GF(2^m) → FRI → STARK → Halo2-real. Add a `zkmark/binius.go` follow-up scoping slot.

11. **Document char-2 sqrt is *not* Tonelli-Shanks.** A naive implementer reading the GF(p) sqrt literature will reach for Tonelli-Shanks 1891. In char 2, sqrt is one Frobenius application — much simpler. Add a doc comment: `// Sqrt in characteristic 2 is the inverse Frobenius: sqrt(x) = x^{2^{m-1}}. Always exists, always unique. Cf. Tonelli-Shanks 1891 (odd p), Cipolla 1903 (alternative odd-p path).`

12. **Defer Goppa codes / McEliece.** Goppa polynomials (Goppa 1981) over GF(2^m) gate post-quantum McEliece-style signatures (slot 057 / 212 territory). Same field machinery, different higher-level construction. NOT in this slot's scope.

## Sources

### Repo files (all paths absolute)
- `C:\limitless\foundation\reality\crypto\modular.go:20-135` — GF(p) ModPow / ModInverse / ChineseRemainder; uint64 prime field only; no GF(2^m) surface.
- `C:\limitless\foundation\reality\crypto\prime.go:18` — Miller-Rabin 1976/1980 primality (note name collision with Rabin 1980 irreducibility per F10).
- `C:\limitless\foundation\reality\crypto\hash.go` — FNV1a + Murmur3 (no algebraic structure; orthogonal).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\210-new-coding-theory.md` (lines 38-46, 113-127) — full coding/galois/ scope, GF(2^m) C3 KEYSTONE.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\290-new-galois-theory.md:189-191` — slot 290 GFpn for odd-prime extensions; cross-link recommendation to slot 210/321 binary-extension fast path; AES `0x11d` reduction polynomial pinned.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\292-new-elliptic-curves.md` — ECC consumer of GF(2^m); NIST B/K binary curves at m ∈ {163, 233, 283, 409, 571}.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\320-dive-error-correction.md:39-44, 76-81` — slot 320 KEYSTONE statement: every code from RS onward needs GF(2^m); day-1 PR includes `coding/galois/gf2m.go` ~280 LOC.
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md:341` — slot 321 line `dive-finite-field | Finite field GF(2^n) operations: poly mul, inverse, sqrt, irreducible test`.
- `C:\limitless\foundation\reality\CLAUDE.md` — design rules 1 (golden files are the proof), 3 (no allocations in hot paths), 6 (reimplement from first principles); per-package precision policy.

### Primary literature
- Berlekamp, E. R. (1968). *Algebraic Coding Theory*. McGraw-Hill. — Exp/Log table GF(2^m), classical reference.
- Itoh, T.; Tsujii, S. (1988). "A fast algorithm for computing multiplicative inverses in GF(2^m) using normal bases." *Information and Computation* 78(3):171-177.
- Rabin, M. O. (1980). "Probabilistic algorithms in finite fields." *SIAM J. Computing* 9(2):273-280. — Both Miller-Rabin primality AND irreducibility test in F_q[x]; same paper.
- Ben-Or, M. (1981). "Probabilistic algorithms in finite fields." *FOCS 1981* 22:394-398. — alternative randomised irreducibility test.
- Karatsuba, A.; Ofman, Y. (1962). "Multiplication of multidigit numbers on automata." *Soviet Physics Doklady* 7:595-596. — Karatsuba multiplication, applies to F_2[x].
- von zur Gathen, J.; Shoup, V. (1992). "Computing Frobenius maps and factoring polynomials." *Computational Complexity* 2(3):187-224. — distinct-degree factorisation.
- Satoh, A.; Morioka, S. (2003). "Hardware-Focused Performance Comparison for the Standard Block Ciphers AES, Camellia, and Triple-DES." *Information Security Conference 2003*. — GF((2^4)^2) tower for AES.
- Canright, D. (2005). "A very compact S-box for AES." *CHES 2005* LNCS 3659:441-455. — minimal-gate AES SubByte via field tower.
- Käsper, E.; Schwabe, P. (2009). "Faster and Timing-Attack Resistant AES-GCM." *CHES 2009* LNCS 5747:1-17. — bit-sliced AES + GF(2^128) PCLMUL.
- Gueron, S.; Kounavis, M. E. (2010). "Intel Carry-Less Multiplication Instruction and its Usage for Computing the GCM Mode." Intel white paper, rev. 2.02. — PCLMULQDQ programming.
- Diamond, B. E.; Posen, J. (2024). "Succinct Arguments over Towers of Binary Fields." Cryptology ePrint Archive Report 2023/1784. — Binius binary-tower STARK construction.
- Hankerson, D.; Menezes, A.; Vanstone, S. (2004). *Guide to Elliptic Curve Cryptography*. Springer. — §2.3-§2.4 GF(2^m) arithmetic, §A.1 NIST B/K curve primitive polynomial table.
- Mullen, G. L.; Panario, D. (2013). *Handbook of Finite Fields*. CRC Press. — comprehensive reference.
- Lidl, R.; Niederreiter, H. (1997). *Finite Fields* (2nd ed.). Cambridge University Press. — Encyclopedia of Mathematics Vol. 20; Table 3.1 primitive polynomials.
- Cohen, H.; Frey, G. (eds.) (2005). *Handbook of Elliptic and Hyperelliptic Curve Cryptography*. CRC Press. — §11 binary-field arithmetic engineering.
- NIST FIPS-197 (2001). *Advanced Encryption Standard*. — AES uses GF(2^8) with `0x11d` reduction polynomial; Annex C test vectors.
- NIST SP 800-186 (2023). *Recommendations for Discrete Logarithm-Based Cryptography*. — NIST B/K curves over GF(2^m) deprecated for new use; legacy support only.
- CCSDS 131.0-B-3 (2017). *TM Synchronization and Channel Coding*. — RS-(255, 223) over GF(2^8) primitive polynomial `0x11d` deep-space standard.
