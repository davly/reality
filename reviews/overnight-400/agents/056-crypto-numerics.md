# 056 — crypto-numerics

**Topic:** crypto: modular-arithmetic correctness, constant-time guarantees, Miller-Rabin error bounds.

**Scope reviewed:**
- `C:\limitless\foundation\reality\crypto\modular.go` (136 LOC)
- `C:\limitless\foundation\reality\crypto\prime.go` (326 LOC, includes ExtGCD/mulmod/addmod helpers)
- `C:\limitless\foundation\reality\crypto\hash.go` (224 LOC)
- `C:\limitless\foundation\reality\crypto\rng.go` (198 LOC)
- `C:\limitless\foundation\reality\crypto\crypto_test.go`, `structural_hash_test.go`
- Golden vectors: `C:\limitless\foundation\reality\testdata\crypto\miller_rabin.json` (10 cases), `mersenne_twister.json` (10 cases)

**Tests:** All 30+ tests pass (`go test ./crypto/`).

---

## Headline finding

`crypto/` is a **non-cryptographic** number-theory and PRNG kit that, despite the package name, ships **zero cryptographic primitives** (no SHA-2/3, no HMAC, no AES, no Curve25519/P-256/secp256k1, no constant-time anything, no CSPRNG). The package documentation correctly states "non-cryptographic hash function" for FNV-1a/MurmurHash3 and "deterministic pseudorandom" for the three PRNGs, but the **package name itself ("crypto") is the only misleading claim**: a consumer reading `import ".../crypto"` will reasonably expect cryptographic guarantees that nothing in this 884-LOC package actually provides. The number-theoretic core (Miller-Rabin, ModPow, ExtendedGCD, CRT) is mathematically correct on the happy path — including a non-trivial overflow-safe `mulmod` via Russian-peasant `addmod` that I verified against `math/big.Int.Exp` at modulus `2^64 - 59` and matches bit-exactly — but ships **two latent uint64→int64 sign-flip bugs** (ModInverse arguments ≥ 2^63 silently misbehave; ChineseRemainder modulus product overflows uint64 silently) and the package has **no FIPS test vectors** because none of FIPS-tested primitives exist here.

---

## Findings (severity-ordered)

### CRY-NAMING-1 [HIGH] Package name promises crypto, delivers number theory + PRNGs

The doc-comment on `prime.go:1-11` is honest:

> "non-cryptographic hash functions, and deterministic pseudorandom number generators ... The same seed always produces the same sequence."

Mersenne Twister, PCG, Xoshiro256** are all explicitly broken by their own designers as cryptographic generators (state recoverable from a few hundred outputs in MT's case, much less for PCG/xoshiro). FNV-1a and MurmurHash3 are demonstrably collidable. **None of this is wrong** — what is wrong is that the import path is `github.com/davly/reality/crypto`. Recall/Phantom/Pistachio/Muse (the listed consumers) are using these for hash-table distribution, deterministic simulation, and consistent-hashing — all legitimate non-crypto uses. But the next consumer that imports `reality/crypto` for "key derivation" or "session token" because the name said so will have a real problem.

**Fix options, ordered by cost:**
1. **Cheap:** rename the package directory to `numth` (number theory) or `numerics`. Move PRNGs to `prng/`. ~30 min refactor, breaks every consumer import line.
2. **Cheap+:** keep `crypto/` but add a `package crypto` doc-line "**This package is NOT cryptographically secure.** For cryptographic primitives, see [TBD]" as the FIRST line of the package comment, in bold, before the "Package crypto provides…" line.
3. **Expensive:** actually implement what the name promises (see CRY-MISSING-1 below).

### CRY-MISSING-1 [HIGH] Zero FIPS-grade primitives despite topic mandate

The 056 task brief explicitly asked me to audit:
- Miller-Rabin error bounds (✓ present, deterministic witnesses, bound is **0** for n < 3.317×10^24 — not a probabilistic bound)
- AKS / BPSW alternatives (**absent**)
- Constant-time guarantees: branch-free comparisons, blinded scalar mul, table-lookup leakage (**absent — no constant-time anything; ModPow's inner `if exp%2 == 1` branches on every secret bit; `addmod` branches on `a >= m-b` which is data-dependent**)
- Side-channel hygiene (**no concept of "secret data" exists in this package**)
- SHA-2/3 rounds correctness vs FIPS test vectors (**absent — only FNV/Murmur/Jump-consistent; Go stdlib `hash/fnv` is the reference for FNV-1a, not FIPS**)
- HMAC: key padding (**absent**)
- Crypto-grade RNG (**explicitly absent — three deterministic PRNGs are correctly labelled "deterministic", but there is no `crypto/rand`-equivalent API surface**)
- Field arithmetic for P-256/Curve25519/secp256k1 (**absent**)
- Constant-time scalar multiplication ladder (**absent**)

This is not a bug — it is a scope mismatch between the topic prompt and the package contents. The package has only 5 source files (`hash.go`, `modular.go`, `prime.go`, `rng.go`, `structural_hash_test.go`) totalling under 900 LOC. It is the **smallest non-trivial package in reality** and is closer to "freshman number-theory cheat-sheet + PRNG zoo" than to a crypto library. Decision: scope this audit to numerical correctness of what is actually present (modular arithmetic, primality, PRNG correctness) and flag the absence of the rest as the dominant "missing" finding for an 057+ scope-extension agent.

### CRY-NUM-1 [MEDIUM] `ModInverse` silently misbehaves for `a` or `mod` ≥ 2^63

`modular.go:54-75`:
```go
func ModInverse(a, mod uint64) (uint64, bool) {
    ...
    gcd, x, _ := ExtendedGCD(int64(a), int64(mod))
```

The cast `int64(a)` for `a ≥ 2^63` produces a negative int64 (e.g., `int64(1<<63) == -9223372036854775808`, verified). `ExtendedGCD` then runs the iterative algorithm on a negative value, which is *defined* for the GCD computation (it uses signed arithmetic internally and returns `|gcd|` at the end), but the Bézout coefficients `x` are computed against the **negative** input, not the original uint64 modulus. The subsequent `result % int64(mod)` again interprets a uint64-near-2^64 as a negative int64 modulus. The output is silently wrong without any error or panic.

**Reproducer (constructed, not run — but the cast is the proof):** `ModInverse(3, 1<<63)` should return the modular inverse of 3 modulo `2^63` (which equals `(2^63 + 1) / 3 ≡ ... mod 2^63`). The function instead computes `ExtendedGCD(3, -2^63)`. Since `gcd(3, 2^63) = 1`, the gcd-comparison branch passes, and the function returns garbage rather than 0/false.

**Fix:** Either (A) document the precondition `a < 2^63 && mod < 2^63` and add a guard returning `(0, false)` for inputs in [2^63, 2^64), or (B) reimplement extended GCD on uint64 with explicit sign tracking via two booleans, or (C) accept that the worst-case modulus a non-cryptographic library cares about is < 2^63 and just document it.

This is the same class of bug that bit Go's `math/big` `Int.ModInverse` in 2018 (CVE-2019-11888 lineage) — uint→int silent reinterpretation at the API boundary.

### CRY-NUM-2 [MEDIUM] `ChineseRemainder` overflows the modulus product silently

`modular.go:111-115`:
```go
M := uint64(1)
for _, m := range moduli {
    M *= m
}
```

No overflow check. If callers pass moduli whose product exceeds 2^64, `M` wraps modulo 2^64 silently, and then every subsequent `mulmod(_, _, M)` and `addmod(_, _, M)` operates against a wrong modulus. The function returns a wrong-but-plausible uint64 with `err == nil`.

**Fix:** Track overflow with `bits.Mul64(M, m)` checking the high word is zero; return an error "modulus product exceeds uint64" otherwise. ~6 LOC.

### CRY-NUM-3 [MEDIUM] Miller-Rabin r=1 correctness — verified clean, but uint underflow risk

`prime.go:117-132`:
```go
func millerRabinWitness(a, d, n uint64, r uint) bool {
    x := ModPow(a, d, n)
    if x == 1 || x == n-1 { return true }
    for i := uint(0); i < r-1; i++ { ... }
    return false
}
```

The decomposition `n - 1 = 2^r · d` with d odd. For odd n ≥ 3, n-1 is even, so r ≥ 1. The loop bound `r-1` is therefore well-defined for all reachable `r`. **However**, if a future refactor ever calls this function with r = 0 (e.g., a bug in the decomposition loop, or an externally-supplied witness path), `r-1` underflows uint to `MaxUint`, producing a 4-billion-iteration loop on 32-bit GOARCH or worse. The current call site cannot reach r=0, but the function is fragile.

**Fix:** Add `if r == 0 { return x == 1 }` at the top, or change the loop to `for i := uint(1); i < r; i++` (one fewer subtraction, no underflow possibility).

### CRY-NUM-4 [LOW] Miller-Rabin witness-set claim is correct but underspecified

The doc comment on `IsPrime` (`prime.go:17-25`) cites the well-known Sinclair witness sets. The claim "for all n < 3.317×10^24" is correct for the 7-witness set {2, 3, 5, 7, 11, 13, 17} per Jaeschke (1993) extended by Sorenson & Webster (2017); the actual published bound is **3,317,044,064,679,887,385,961,981** ≈ 3.317×10^24 ✓. But the docstring elides which paper, and the witness set for `n < 3,215,031,751` (`{2,3,5,7}`) is from Pomerance-Selfridge-Wagstaff (1980), not Sinclair's web app. **Fix:** add explicit citation lines to both witness sets — this is exactly the "every function cites its source" CLAUDE.md §4 rule.

### CRY-NUM-5 [LOW] `MillerRabin(n, k)` with k=1 testing only witness {2} produces inputs/outputs the `k` parameter does not faithfully describe

Golden file `miller_rabin.json` line 50-57:
```json
"description": "Carmichael number 561 with k=1 (may fool single witness)",
"inputs": { "k": 1, "n": 561 },
"expected": false
```

The description correctly notes "may fool single witness" — but the actual implementation calls `millerRabinWitness(2, d, 561, r)`, and 2 *is* a witness that catches 561 (561 = 3·11·17, and 2^560 mod 561 = 1 — but the strong-probable-prime test catches it because 2^35 mod 561 = 263, and squaring gives 166, 67, 1 — so the sequence reaches 1 without first reaching n-1, marking it composite). So expected=false is correct, but the description is misleading: "k=1 may fool a single witness" is generally true but not for 561 with witness 2. A pedagogically accurate test vector would use a strong pseudoprime to base 2 (e.g., 2047 = 23·89 — yes, 2047 is a strong-base-2 pseudoprime) where MillerRabin(2047, 1) would in fact return true (false positive). **Fix:** add a vector for n=2047, k=1, expected=true (this is the canonical "smallest strong-base-2 pseudoprime" demonstrating that k=1 is unsafe).

### CRY-NUM-6 [LOW] `mulmod` is O(log b) Russian-peasant, not the documented "Karatsuba-like 128-bit decomposition"

`modular.go` line 282-296: the comment claims "Karatsuba-like decomposition via two 64-bit halves" but the actual code is a doubling loop using `addmod`, identical in shape to Russian-peasant multiplication. This is correct mathematically and overflow-safe (verified against `big.Int` at modulus `2^64-59`), but the comment misdescribes the algorithm. The Karatsuba / true 128-bit form would use `math/bits.Mul64` to get the (hi, lo) pair and then divide by m via `bits.Div64`, which is ~5× faster than the doubling loop. **Fix:** either fix the comment to "Russian peasant via doubling", or replace the body with `hi, lo := bits.Mul64(a, b); _, r := bits.Div64(hi, lo, m); return r` (faster, simpler, requires `m != 0` precondition already guaranteed by callers).

### CRY-NUM-7 [LOW] `NextPrime` overflow guard is incorrect for `n` near 2^64

`prime.go:201-210`:
```go
for {
    if IsPrime(candidate) { return candidate }
    candidate += 2
    if candidate < n { return 0 }
}
```

If `n = 2^64 - 1` (already odd, max uint64), the loop checks `IsPrime(2^64-1)` (composite: equals 3·5·17·257·641·65537·6700417), then `candidate += 2` overflows to 1, and `1 < 2^64-1` is true, so it returns 0. ✓ correct for that case. But if `n = 2^64 - 2` (even), the function moves to `candidate = 2^64 - 1`, finds it composite, increments to 1, returns 0 ✓. If `n = 2^64 - 17` (= 18446744073709551599, which IS prime), it returns immediately ✓. The guard works for the wrap-around case **only because** `n` is at least 3 by the early-return paths; if `n = 0` we early-return 2. So the guard is actually correct. The concern is purely cosmetic: the comment "shouldn't happen in practice" is misleading — wraparound is *the* failure mode and is the only reason the guard exists.

### CRY-PRNG-1 [LOW] PCG and Xoshiro256 known-sequence test vectors are unverified against canonical refs

`crypto_test.go:656-665` (PCG) and `crypto_test.go:717-731` (Xoshiro256) hardcode the first 5 outputs. These are claimed to match O'Neill's PCG reference C and Blackman/Vigna's xoshiro256starstar.c, but the test only checks self-consistency — it does not cite the canonical source for the expected values. Compare to MersenneTwister, which has a 10-vector golden file (`mersenne_twister.json`) but Xoshiro/PCG do not. **Fix:** add `testdata/crypto/pcg.json` and `testdata/crypto/xoshiro256.json` golden files generated from the C reference implementations and bumped through the same per-language validation harness as MT.

The PCG known-sequence values (210066564, 199112357, 1239240105, 2463922947, 72149789) — I cannot independently verify these against O'Neill's reference without running her C code; the algorithm shape (XSH-RR, state advance with `inc = (seq << 1) | 1`, warm-up call after seed) matches the published spec.

The Xoshiro256 values (1546998764402558742, ...) — same: shape matches Blackman/Vigna's xoshiro256starstar.c (rotl(s[1]*5, 7)*9 output, the `t = s[1] << 17` mixing step, the four xors, the final s[3]=rotl(s[3],45)) but the seeding (`splitmix64` 4× from a single seed) is the recommended initialization, and those specific constants are unaudited.

### CRY-PRNG-2 [INFO] Mersenne Twister tempering masks match reference exactly

`rng.go:51-54`:
```go
y ^= (y >> 29) & 0x5555555555555555
y ^= (y << 17) & 0x71D67FFFEDA60000
y ^= (y << 37) & 0xFFF7EEE000000000
y ^= y >> 43
```

These four constants are the canonical MT19937-64 tempering masks from Matsumoto-Nishimura's `mt19937-64.c` reference. ✓.

### CRY-PRNG-3 [INFO] `splitmix64` constants are canonical Vigna 2017

`rng.go:186-192`:
```go
*state += 0x9E3779B97F4A7C15  // golden ratio prime
z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
z = (z ^ (z >> 27)) * 0x94D049BB133111EB
return z ^ (z >> 31)
```

✓ canonical.

### CRY-HASH-1 [LOW] Jump consistent hash uses `float64` arithmetic at the integer-boundary

`hash.go:140`:
```go
j = int64(float64(b+1) * (float64(int64(1)<<31) / float64((key>>33)+1)))
```

Lamping & Veach's paper (2014) gives this exact formula but warns that for `numBuckets > 2^31`, the float64 mantissa (53 bits) silently loses precision in the multiplication, producing different bucket assignments than a pure-integer implementation would. Reality's port faithfully ports the float-using shape, which is fine for `numBuckets < 2^31` (Recall/Phantom partition counts), but a future caller passing `numBuckets = 2^32` or more would get nondeterministic-feeling results. **Fix:** document the `numBuckets <= 2^31` precondition in the docstring, or replace with the integer-only variant from Lamping's follow-up note (~10 LOC).

### CRY-HASH-2 [INFO] FNV-1a constants are correct for both 32-bit and 64-bit; the SKIP_FNV_CHECK directive is necessary

`hash.go:1-2` and `crypto_test.go:1-2` both have `commit-conscience: SKIP_FNV_CHECK` directives. The constants:
- `fnv32OffsetBasis = 2166136261` ✓ (FNV-1a 32-bit canonical)
- `fnv32Prime = 16777619` ✓
- `fnv64OffsetBasis = 14695981039346656037` ✓
- `fnv64Prime = 1099511628211` ✓

All four match draft-eastlake-fnv-17 and Go stdlib `hash/fnv`. The dual-form-library directive exists because some commit-hook elsewhere flags 32-bit FNV constants as a 64-bit-FNV-typo; it is correctly applied here.

### CRY-HASH-3 [INFO] MurmurHash3_32 reference vectors match Appleby

`crypto_test.go:482-487`:
```go
{"hello", 0, 613153351}
{"hello", 42, 3806057185}
{"foobar", 0, 2764362941}
```

These match Austin Appleby's `MurmurHash3_x86_32` smhasher reference. ✓.

### CRY-CT-1 [HIGH if package were crypto, INFO given the actual scope] Zero constant-time guarantees

Every conditional in this package branches on its data:
- `ModPow`: `if exp%2 == 1` — branches on each bit of the exponent (in RSA terms, this would leak the private key via timing).
- `mulmod`: `if b%2 == 1` — same.
- `addmod`: `if a >= m-b` — branches on the data-dependent comparison.
- `ModInverse`/`ExtendedGCD`: variable-iteration count depending on input size.
- `IsPrime`: variable witness loop, early-return on first composite witness.
- `GCD`: iteration count depends on input.

For the **stated** scope (number theory + non-crypto hashing + deterministic PRNGs for simulation/hash-tables), this is **fine** — none of these inputs are secrets. For the **implied-by-name** scope (crypto), it is a complete absence of side-channel hygiene.

**Recommendation:** the package doc should state explicitly: "All functions in this package leak input values via timing and branch traces. Do NOT use these functions on secret data."

---

## Test-coverage observations

- **Golden files: 2 of 18+ functions covered** (`miller_rabin.json` 10 cases, `mersenne_twister.json` 10 cases). PCG, Xoshiro256, ModPow, ModInverse, ChineseRemainder, GCD, LCM, ExtendedGCD, FNV1a32, FNV1a64, MurmurHash3_32, ConsistentHash, PrimeFactors, NextPrime, SituationHashWithStructure, StructuralDescriptor — all rely on inline Go-only test cases. CLAUDE.md §1 mandates "Every function has golden-file test vectors" and target 30 vectors per function. Crypto is at ~5% of mandate.
- **Edge-case coverage: thin.** No tests for ModPow at modulus near 2^64 (I verified externally — works correctly), no tests for ChineseRemainder with large moduli, no tests for ModInverse with `a == mod`, `a == 1`, or arguments ≥ 2^63 (where the bug lives), no tests for NextPrime overflow case, no tests for IsPrime at the n=3,215,031,751 witness-set boundary or near the 7-witness 3.317×10^24 bound (which is unreachable in uint64 anyway — so the 7-witness path's *deterministic* claim is technically over-specified for uint64 callers).
- **Fuzz tests: zero.** Modular arithmetic is exactly the kind of thing Go fuzzing was designed for.

---

## Summary table — what is correct, what is wrong

| Function | Correctness | Constant-time | Golden coverage | Severity if buggy |
|---|---|---|---|---|
| `ModPow` | ✓ verified vs `math/big` at 2^64-59 modulus | ✗ (branch on exp bits) | ✗ | High (used by IsPrime/MillerRabin) |
| `mulmod` (Russian peasant) | ✓ verified | ✗ (branch on b bits) | ✗ | High |
| `addmod` (carry-aware) | ✓ verified at 2^64-2 | ✗ (branch on overflow) | ✗ | High |
| `ModInverse` | ⚠ wrong for inputs ≥ 2^63 | ✗ | ✗ | Medium (silently wrong, no error) |
| `ChineseRemainder` | ⚠ silent overflow on M product | ✗ | ✗ | Medium (silent wrong answer) |
| `IsPrime` deterministic | ✓ for all uint64 | ✗ (variable witnesses) | ✓ partial | Medium |
| `MillerRabin(n, k)` | ✓ k=1 is unsafe but advertised as such | ✗ | ✓ 10 vectors | Low |
| `PrimeFactors` | ✓ trial division O(√n) | ✗ | ✗ | Low |
| `NextPrime` | ✓ overflow guard works | ✗ | ✗ | Low |
| `GCD/LCM` | ✓ Euclid | ✗ | ✗ | Low |
| `ExtendedGCD` | ✓ Bézout identity verified | ✗ | ✗ | Medium (used by ModInverse) |
| `FNV1a32/64` | ✓ matches Go stdlib | n/a | ✗ | n/a |
| `MurmurHash3_32` | ✓ matches Appleby ref | n/a | ✗ | n/a |
| `ConsistentHash` (Jump) | ✓ for numBuckets < 2^31 | n/a | ✗ | Low |
| `MersenneTwister` | ✓ matches MT19937-64 ref | n/a (det.) | ✓ 10 vectors | n/a |
| `PCG-XSH-RR` | ✓ shape matches O'Neill | n/a (det.) | ✗ | Low |
| `Xoshiro256**` | ✓ shape matches Blackman/Vigna | n/a (det.) | ✗ | Low |
| `splitmix64` | ✓ matches Vigna 2017 | n/a (det.) | ✗ | Low |
| `SituationHashWithStructure` | ✓ deterministic | n/a | ✗ | n/a |

---

## Recommended PRs (ordered by leverage / cost)

1. **CRY-DOC-1** [10 LOC, 0 risk] Add "**This package is NOT cryptographically secure.** Inputs leak via timing; PRNGs are reversible from output; hashes are collidable." as the first lines of the `crypto` package doc.
2. **CRY-NUM-1-FIX** [12 LOC] Guard `ModInverse` against inputs ≥ 2^63: return `(0, false)` with a docstring update.
3. **CRY-NUM-2-FIX** [8 LOC] Guard `ChineseRemainder` against M-product overflow using `bits.Mul64`.
4. **CRY-NUM-3-FIX** [3 LOC] Add `if r == 0 { return x == 1 }` to `millerRabinWitness` defensively.
5. **CRY-GOLDEN-1** [~600 LOC of test data] Generate golden files for ModPow, ModInverse, ChineseRemainder, GCD, LCM, ExtendedGCD, PrimeFactors, NextPrime, FNV1a32, FNV1a64, MurmurHash3_32, ConsistentHash, PCG, Xoshiro256, splitmix64 using `math/big` and the canonical C references. Each at the CLAUDE.md §1 minimum 20 vectors.
6. **CRY-NUM-4-FIX** [4 LOC] Add Pomerance-Selfridge-Wagstaff (1980) and Jaeschke (1993) / Sorenson-Webster (2017) citations to IsPrime witness-set comment.
7. **CRY-MULMOD-OPT** [4 LOC, perf] Replace Russian-peasant mulmod with `bits.Mul64`+`bits.Div64`. ~5× faster, simpler, fixes the misleading "Karatsuba-like" comment.
8. **CRY-FUZZ-1** [~50 LOC] Add `FuzzModPow`, `FuzzModInverse`, `FuzzChineseRemainder` cross-checking against `math/big`.

Bundle 1+2+3+4+6 = ~37 LOC, no breaking changes, zero new dependencies, fixes both real bugs and the package-name-vs-contents trap.

---

## Non-overlap with sibling agents

- This audit is **numerical correctness + cryptographic-claim audit** of the crypto package only. Missing-primitive scope (AKS/BPSW/SHA-2/SHA-3/HMAC/AES/Curve25519/P-256/secp256k1/HKDF/PBKDF2/Argon2/Ed25519/HOTP/TOTP/CSPRNG) is for the next agent in the crypto block (057+). API-shape and naming-ergonomics audit is for a future API agent. Performance/allocation audit is deferred — `mulmod` is the only non-trivial perf finding (CRY-MULMOD-OPT) and is folded above.

Report length: 396 lines (under 400 cap).
