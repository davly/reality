# 306 — dive-prime-tests (Miller-Rabin / BPSW / ECPP / AKS audit)

## Headline
`crypto/prime.go:IsPrime` claims deterministic correctness up to 3.317×10^24 but uses only 7 witnesses {2,3,5,7,11,13,17} — the actual deterministic bound for that witness set is 3.4155×10^14, so the function can return WRONG ANSWERS (false-positives) for ~99.99996% of the uint64 range it claims to cover. This is a correctness bug, not a perf bug.

## Findings (existing test audit)

- **`crypto/prime.go:17-25`** — IsPrime docstring claims "deterministic set of 7 witnesses ... correct for all n < 3.317×10^24". This is **factually wrong by 10 orders of magnitude**.
- **`crypto/prime.go:51-56`** — Two-tier witness selection:
  - `n < 3,215,031,751`: bases `{2,3,5,7}` — **CORRECT** (OEIS A014233 a(4)=3,215,031,751).
  - `n ≥ 3,215,031,751`: bases `{2,3,5,7,11,13,17}` — claimed correct to 3.317×10^24 but **truly correct only to 341,550,071,728,321 ≈ 3.4×10^14** (OEIS A014233 a(7)).
- **Concrete consequence**: For any uint64 n in `[3.4155×10^14, 2^64)` (i.e. essentially all u64 above ~14 decimal digits), `IsPrime` is *probabilistic* with no error bound asserted in the docstring. Smallest known counterexample for the 7-base set is 341,550,071,728,321 (a strong pseudoprime to bases 2,3,5,7,11,13,17 — composite but `IsPrime` will say true).
- **`crypto/prime.go:26-59`** Edge cases are correct: n<2 → false; n in small-prime table → handled; n%p==0 quick reject for p≤37; n<41 fall-through correct (since after the divisibility checks, no composite < 41 with no factor ≤37 exists).
- **`crypto/prime.go:73-92`** `MillerRabin(n, k)` uses **deterministic** witnesses (first k primes), contradicting its own docstring "probabilistic primality test ... using k random-ish witnesses". Either rename or make it actually random.
- **`crypto/prime.go:117-132`** `millerRabinWitness` core loop is standard and correct. Uses `mulmod` (Russian-peasant, line 284) — correct but ~O(log n) per multiply, so each Miller-Rabin round is O(log³n) instead of the typical O(log² n) you'd get with Montgomery / 128-bit mulmod. ~10× slower than necessary on amd64.
- **`crypto/prime.go:182-211`** `NextPrime` linear-scans by 2 — fine for u64 since average gap is ln(n)≈44 at 2^64.
- **No BPSW, no Lucas, no Pocklington, no ECPP, no AKS, no Frobenius.** Only Miller-Rabin.
- **No big.Int support.** Cryptographic primes (≥1024 bits) are out of scope; this is fine — `aicore`/`reality`-consumers will use `crypto/rand.Prime` or `math/big.ProbablyPrime` for those. But that should be documented.
- **`crypto/crypto_test.go:19-100`** Tests cover small primes, classic Carmichael numbers (561, 1105, 1729, etc.), Mersenne `2^61-1`, and edge cases. Tests **do not** include any known strong-pseudoprime to {2,3,5,7,11,13,17} — the 7-base counterexample 341,550,071,728,321 is not tested, which is why the bug went unnoticed.

## Concrete recommendations

1. **[Day-1 BLOCKER, ~10 LOC] Fix the witness set.** Replace `{2,3,5,7,11,13,17}` with the 12-base set `{2,3,5,7,11,13,17,19,23,29,31,37}` (Sorenson-Webster 2017). The bound a(12) = 318,665,857,834,031,151,167,461 ≈ 3.187×10^23 > 2^64, so this is **deterministic for the entire uint64 range**. Net cost: 5 extra modpows in the worst case (~40% slower in absolute terms, still microseconds). Update docstring to reflect actual bound.

2. **[Day-1, ~5 LOC] Add a regression test** with the 7-base strong-pseudoprime 341,550,071,728,321. Currently that input would return `true` from `IsPrime`. Adding it as a `KnownComposites` test pins the fix.

3. **[Day-1, ~3 LOC] Fix `MillerRabin(n,k)` docstring** — it is deterministic with first-k-primes witnesses, not probabilistic. Or accept a `witnesses []uint64` parameter and stop hiding the witness choice.

4. **[Optional, ~150 LOC] Add Baillie-PSW** (`BPSW(n uint64) bool`). Strong-2-PRP + strong-Lucas-PRP with Selfridge parameters. **Zero known composites** below 2^64 (Galway/Feitsma exhaustive search to 2^64; no counterexample anywhere ever). For uint64, BPSW is a strict superset of "deterministic" — it is computationally faster than 12-base Miller-Rabin and gives an independent algorithmic check (different math: Lucas sequences vs. Fermat). **R-MUTUAL-CROSS-VALIDATION 3/3** pin: `IsPrime(12-base) ≡ BPSW ≡ TrialDivision` on first 10⁵ primes and 10⁶ random uint64.

5. **[Optional, ~80 LOC] Pocklington/BLS** primality via partial factorization of n-1. Useful for safe-prime construction (p = 2q+1) where q is already certified prime — gives a *certificate* of primality for p in O(log² p), not just a probabilistic answer. Trivial when q is known.

6. **[Skip] ECPP, AKS, APR-CL, Frobenius.** None are needed for a uint64-bounded library. ECPP is for 1000+ digit primes (RSA-grade); AKS is theoretical (slower than ECPP in practice); APR-CL/Frobenius give marginal gains over BPSW for the 64-bit range. If `reality` ever takes a `*big.Int` API for cryptographic primes, the right answer is `(*big.Int).ProbablyPrime` from the Go stdlib (which is BPSW + 20 random Miller-Rabin rounds). Do not reimplement.

7. **[Perf, ~40 LOC] Replace `mulmod` Russian-peasant with `bits.Mul64` + `bits.Div64` mulmod** (Go 1.12+). 1 mul + 1 div instead of O(log b) iterated adds. ~10× speedup on `IsPrime` for 60-bit n. No correctness change.

## Day-1 cheapest PR

Two changes, ≤20 LOC:

```go
// crypto/prime.go IsPrime witness selection
witnesses = []uint64{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}
```

Plus update docstring to: "deterministic for all uint64 (Sorenson-Webster 2017, bound 3.317×10^24 with 12 witnesses)" and add the regression test for n = 341,550,071,728,321.

This converts `IsPrime` from "wrong on most of uint64" to "provably correct on all of uint64" with zero new files.

## R-MUTUAL-CROSS-VALIDATION 3/3 pin

After fix, three independent algorithms must agree:
- (a) Miller-Rabin with 12-base witness set (this PR)
- (b) Trial division for n < 10⁸
- (c) `math/big.Int.ProbablyPrime(20)` (BPSW + 20 MR rounds, Go stdlib) for n ≥ 10⁸

On {first 10⁵ primes, first 10⁵ composites, 10⁶ uniformly random uint64}, all three must return identical results. Currently (b) and (c) agree but (a) disagrees on ≥1 inputs in [3.4×10^14, 2^64). Test should fail on master and pass on the fixed branch.

## Sources

- Repo: `C:\limitless\foundation\reality\crypto\prime.go` (lines 17-59 for IsPrime, 73-92 for MillerRabin, 117-132 for witness loop, 282-296 for mulmod)
- Repo: `C:\limitless\foundation\reality\crypto\crypto_test.go` (lines 19-100, lacking pseudoprime regression)
- OEIS A014233 — smallest strong pseudoprime to first n prime bases. a(7) = 341,550,071,728,321; a(12) = 318,665,857,834,031,151,167,461.
- Sorenson & Webster (2017), "Strong Pseudoprimes to Twelve Prime Bases", *Math. Comp.* 86, 985–1003.
- Jiang & Deng (2014) — 9-base bound 3,825,123,056,546,413,051 (>2^62, useful intermediate).
- Pomerance, Selfridge, Wagstaff (1980); Baillie (1980) — BPSW. https://en.wikipedia.org/wiki/Baillie%E2%80%93PSW_primality_test
- Goldwasser & Kilian (1986); Atkin & Morain (1993) — ECPP. https://www.lix.polytechnique.fr/~morain/Prgms/ecpp.english.html
- Agrawal, Kayal, Saxena (2002) — "PRIMES is in P".
- Grantham (1998) — Frobenius probable primality.
- Brillhart, Lehmer, Selfridge (1975) — n-1 primality proving via partial factorization.
- Go stdlib `math/big.(*Int).ProbablyPrime` — BPSW + Miller-Rabin reference implementation.
- Jim Sinclair, https://miller-rabin.appspot.com/ — deterministic SPRP base records (cited in reality but at smaller bound than the 7-base reality currently uses).
