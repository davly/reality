# 059 — crypto-api

**Topic:** crypto: hazmat surface, misuse resistance, key types, constant-time operators, random source injection, domain separation, byte-order, naming, stdlib idioms.

**Premise (carried from 056/057/058).** crypto/ is 880 LOC of source over 4 files: `prime.go` (Miller-Rabin / GCD / ExtGCD), `modular.go` (ModPow / ModInverse / CRT), `hash.go` (FNV-1a / Murmur / Jump-consistent / SituationHashWithStructure), `rng.go` (MT19937-64 / PCG-XSH-RR / Xoshiro256**). **Zero cryptographic primitives ship today** (056). 057 owns the missing-primitive map. 058 owns architectural transferability from libsodium / RustCrypto / dalek / arkworks / Crypto++. **059 owns the API ergonomics question that has to be answered before any of 057's primitives lands** — the package's *current* API shape, *which* misuse patterns it already invites, and what the seam between "math primitives that exist today" and "cryptographic primitives that 057 wants to add tomorrow" must look like.

The review topic uses the word "hazmat" deliberately. In `cryptography` (Python), hazmat is `cryptography.hazmat.primitives.*` — explicitly walled off behind a name that says *touch only if you know what you are doing*. **Reality has no such wall**. Today that's harmless because the package only has number-theory primitives (touching `ModPow` cannot leak a secret key — there is no key). The moment 057's T1-EC / T2-EDDSA / T2-SCHNORR lands, the *same* package will hold both tools. This review's job is to call the seam *before* it gets papered over.

---

## Headline

**Reality's `crypto` package is a uint64-in / uint64-out arithmetic kit with no key types, no `io.Reader` injection, no `subtle.ConstantTimeCompare` analog, no domain-separation parameter on any hash, and no byte-order documentation on the two functions (`MurmurHash3_32`, `Uint64`-via-`Float64`) where endianness would matter.** The naming convention is *idiomatic-Go-flat* (`ModPow`, `IsPrime`, `FNV1a64`) — argument order is consistently *(data-then-modifier)*: `ModPow(base, exp, mod)`, `MillerRabin(n, k)`, `MurmurHash3_32(data, seed)`. This is the right convention for *math* and the wrong convention for *crypto* — crypto stdlib idiom is `Sign(priv, msg)` (key first, message second, because `priv.Sign(msg)` reads naturally as a method). When 057's signature schemes land, Reality has to choose: keep `Func(data, params...)` consistency and ship `EdDSASign(msg, priv)` (math-style, breaks `crypto/ed25519` muscle memory), or pivot to `Func(key, data...)` for crypto APIs only (introduces an inconsistency boundary the consumer must learn). **Recommendation: pivot at the seam, keep math-style for non-crypto and key-first for crypto, document the seam as a one-line invariant in the package doc.** The package doc currently does not even mention that *cryptographic* primitives are out of scope for the present version, so a consumer who imports `github.com/davly/reality/crypto` plausibly believes `MT19937-64` is a cryptographically secure PRNG (it absolutely is not). Fixing the doc is the single highest-leverage 30-LOC change in the entire crypto/ surface.

---

## 1. Hazmat surface — how easy is it to misuse what's already there?

**Today's surface is small but already mis-sells.** Five concrete footguns audible from the source:

**(1.1) `MersenneTwister`, `PCG`, `Xoshiro256` are *deterministic* PRNGs — none are CSPRNGs.** The package doc on `rng.go` says "deterministic pseudorandom number generators" (good), but the package-level doc in `prime.go` line 1–11 says "Package crypto provides number theory and cryptographic primitives" (overclaim). A consumer searching for `crypto.NewPCG` and reading the godoc will see "package crypto" + "PCG-XSH-RR variant — a 64-bit state / 32-bit output PRNG with excellent statistical quality" and reasonably conclude this is suitable for nonces / session IDs / API tokens. **It is not.** PCG is broken under chosen-state recovery (Bouillaguet et al. 2020 attack recovers state from 512 outputs). MT19937 is famously broken (624 outputs reveal full state). Xoshiro256** is broken under known-output attacks. The fix is two lines: (a) rename `prime.go` package doc to "Package crypto provides number theory primitives, non-cryptographic hash functions, and **deterministic, NON-cryptographic** pseudorandom number generators." (b) Add a `// SECURITY: NOT cryptographically secure. Do not use for keys, nonces, IVs, or any value that must be unpredictable to an attacker.` comment as the first line of every `New*` constructor in `rng.go`.

**(1.2) `FNV1a32`, `FNV1a64`, `MurmurHash3_32`, `SituationHashWithStructure` are ALL non-cryptographic.** Same overclaim risk. None are collision-resistant. None are pre-image-resistant. `MurmurHash3` is famously HashDoS-vulnerable (Aumasson-Bernstein 2011 paper showed seed-recovery from ≤24 short outputs). The hash.go file does say "non-cryptographic hash function" in each docstring (good — line 19, 35, 55, 116) but the package name `crypto` undermines every individual docstring. **The only function in `hash.go` whose name signals its danger is `ConsistentHash` (because no one expects sharding to be cryptographic). Every other function name is innocent of context.** A future consumer doing `crypto.FNV1a64(password)` and storing the output as a "hash of the password" is having a bad day; the API didn't help them.

**(1.3) `ModPow` is the *one* function in the package today that is literally a building block for asymmetric crypto** (RSA encrypt / decrypt / Diffie-Hellman exponentiation are all `ModPow` over a sufficiently large modulus). It is **not constant-time**. Line 31–37 of `modular.go`:
```
for exp > 0 {
    if exp%2 == 1 {
        result = mulmod(result, base, mod)
    }
    exp /= 2
    base = mulmod(base, base, mod)
}
```
The `if exp%2 == 1` branch leaks the bit pattern of the exponent through timing — exactly the textbook side-channel that *every* RSA / DH timing attack has exploited since Kocher 1996. Today, this is fine because `ModPow` is only used internally by `IsPrime` (where the exponent is `n-1`, a public value) — but **the function is exported**. A consumer who reads "modular exponentiation" and uses it to implement RSA-CRT decryption with their own bigint type will leak the private exponent. The fix is two-pronged: (a) rename the exported function to `ModPow` *clearly documented* "uses non-constant-time square-and-multiply; do NOT use with secret exponents," (b) reserve the name `ModPowConstTime` for 057's future bigint-based version with Montgomery ladder. The name `ModPow` *itself* should never imply constant-time-ness.

**(1.4) `ModInverse` calls `ExtendedGCD` which is also non-constant-time** (line 261–266 of `prime.go` — the loop body has data-dependent quotients). Same risk profile as `ModPow`: fine for prime testing, dangerous for ECDSA's `s = k^-1 (z + r·d)` where k is secret. The cure is identical — document the limitation, reserve `ModInverseConstTime` for the future.

**(1.5) `IsPrime` uses `[]uint64` literals for the witness set, allocated *fresh on every call* (line 53, 55).** Not a security bug, but a performance pitfall in hot paths and an API smell — a `package var witnessSet32 = [...]uint64{2,3,5,7}` would cost 0 LOC of API change and 0 allocations per call. (Strictly an API hygiene comment, not misuse.)

**Summary.** Today's hazmat surface is *small* (one mathematically-cryptographic primitive, `ModPow`, that's documented as non-constant-time only by omission), but the *package framing* over-promises. The hazmat wall in cryptography (Python) and the "hazmat" prefix in `crypto/internal/...` (Go stdlib) exist because every library that didn't build the wall ended up with users importing low-level primitives directly. **Reality should adopt one of the two conventions before 057 lands**: either (a) move all "use-with-care" primitives under `crypto/hazmat/...` (Python style — no enforcement, just naming as discoverability), or (b) keep flat namespace but rigorously document each function's misuse properties and use the Go-stdlib convention of placing constant-time helpers in a sibling `crypto/subtle`-style package. **Option (b) is closer to Reality's existing flat-namespace style; recommend it.**

---

## 2. Key types — secret-key vs public-key — distinct types or just bytes?

**Not applicable today** (no keys exist), but this is *the* decision that 058's headline calls out and that 059 must echo as binding for 057's primitive landings.

The four candidate shapes (in increasing strictness):

| Shape | Example | Misuse-prevention strength | Cost |
|-------|---------|---------------------------|------|
| **A. Plain `[]byte`** | `func Ed25519Sign(priv []byte, msg []byte) []byte` | Zero. Caller can swap pub/priv at the call site, library only catches via `len()` checks. | 0 LOC |
| **B. Named slice types** | `type Ed25519PrivateKey []byte; type Ed25519PublicKey []byte` | Compile-time catches pub-vs-priv swap. Doesn't prevent `Ed25519PrivateKey([]byte("hello"))` cast. | ~5 LOC per key type |
| **C. Named fixed arrays** | `type Ed25519PrivateKey [64]byte; type Ed25519PublicKey [32]byte` | Compile-time catches pub-vs-priv swap AND length errors (a `[]byte` of len 32 won't auto-cast to `[64]byte`). Dalek-style. | ~10 LOC per key type |
| **D. Opaque struct with private fields** | `type Ed25519PrivateKey struct { bytes [64]byte }` + `Bytes()` / `FromBytes()` | All of (C) + opaque to external mutation, can carry validation invariants ("clamped scalar / canonical encoding"), can implement `Zero()` for hygiene. RustCrypto / dalek style. | ~30 LOC per key type |

`crypto/ed25519` stdlib chose **(B)** in 2018: `type PrivateKey []byte; type PublicKey []byte`. This is the *minimum viable* shape — distinct types but no length enforcement, no opacity, no hygiene. It works because Go's interface dispatch (`PrivateKey.Sign(...)`) gives the *method-call site* type protection even when the *value passing* doesn't.

`crypto/ecdsa` stdlib chose **(D)**: `type PrivateKey struct { PublicKey; D *big.Int }`. Opaque-fielded, full struct semantics, no auto-cast.

**Recommendation for Reality:** start at **(C)** for the symmetric/hash side (no scalar arithmetic to enforce, just a sized buffer) and **(D)** for elliptic curves (where the key value carries algebraic invariants — a Curve25519 private scalar must be clamped, an Ed25519 public point must be in the prime-order subgroup). Cost: ~150 LOC of type machinery for the full T1+T2 surface (058 already estimated this). **The single load-bearing decision is "do not use plain `[]byte` for keys."** Every CVE in the 2010-2020 decade in OpenSSL / PyCrypto / Bouncy Castle that involved key-type confusion came from APIs that took `byte[]` / `unsigned char *` and trusted the caller's discipline.

---

## 3. Constant-time operator surface — is there `crypto.ConstEqual([]byte, []byte) bool`?

**No.** `grep`-able evidence: zero matches for `ConstantTime`, `ConstEq`, `subtle`, `ct`, or any timing-related primitive in the entire crypto/ source. The package re-exports nothing from `crypto/subtle`.

**This is the #1 actionable gap for an API review.** Go's stdlib ships `crypto/subtle` with `ConstantTimeCompare(x, y []byte) int`, `ConstantTimeSelect(v, x, y int) int`, `ConstantTimeByteEq(x, y uint8) int`, `ConstantTimeEq(x, y int32) int`, `ConstantTimeCopy(v int, x, y []byte)`, `ConstantTimeLessOrEq(x, y int) int`. These are the *minimum viable* primitives for any constant-time crypto code, and they cost approximately *zero* to expose because Go already provides them — Reality just needs to decide *where they live*.

Three options:

**(3a) Re-export from `crypto/subtle` directly.** `import "crypto/subtle"` in any consumer. **Rejected by CLAUDE.md §2: "zero dependencies. Only the language's standard math library."** `crypto/subtle` is *not* `math` — but it's stdlib. The rule needs an explicit clause: "stdlib `crypto/subtle` is allowed for constant-time primitives, because reimplementing them defeats the purpose (the compiler may optimize away naive constant-time code; the stdlib version has hand-vetted assembly on amd64/arm64)." Recommend adding this clause. ~1 sentence.

**(3b) Reimplement in a `crypto/ct` sub-package.** ~80 LOC. Reimplementing `ConstantTimeCompare` looks like:
```go
func Equal(x, y []byte) bool {
    if len(x) != len(y) { return false }
    var v byte
    for i := range x { v |= x[i] ^ y[i] }
    return v == 0
}
```
The risk: Go 1.22+ inlines and may merge across function boundaries. Without compiler intrinsics, the *only* way to guarantee non-constant-time compilation doesn't happen is to test the assembly on every release. This is what the stdlib does (see `src/crypto/subtle/xor_amd64.s`) and what Reality almost certainly should not duplicate.

**(3c) Hybrid — re-export in a thin wrapper.** Best of both:
```go
// Package ct wraps crypto/subtle for constant-time primitives.
package ct
import "crypto/subtle"
func Equal(x, y []byte) bool { return subtle.ConstantTimeCompare(x, y) == 1 }
func ByteEq(x, y byte) bool  { return subtle.ConstantTimeByteEq(x, y) == 1 }
func Select(c bool, x, y int) int { ... }
```
Naming idiomatic-Go (`bool` returns instead of `int`-flag returns), zero new assembly to maintain. Cost: ~40 LOC + a CLAUDE.md clause permitting `crypto/subtle`.

**Recommendation: (3c). Add to CLAUDE.md §2 the explicit allowance for `crypto/subtle`. Add a `crypto/ct` sub-package or top-level `crypto.Equal` / `crypto.Select` functions. This is a Tier-0 prerequisite for every 057 primitive that uses keys.**

---

## 4. Random source injection — `io.Reader` or hardcoded `crypto/rand`?

**No randomness functions in the package today.** `MT19937-64`/`PCG`/`Xoshiro256**` all take an explicit `seed uint64` constructor argument and produce a deterministic stream — *they are themselves io.Reader-shaped state machines*, not consumers of randomness. So today the question is moot.

**For 057's primitives** (where signing schemes need a nonce, key generation needs entropy), the API decision is: do `Sign` / `Generate` take an `io.Reader` parameter, or hardcode `crypto/rand.Reader`?

**Stdlib precedent is split:**
- `crypto/ed25519.GenerateKey(rand io.Reader) (PublicKey, PrivateKey, error)` — explicit `io.Reader` parameter, default suggests `nil` → `rand.Reader`.
- `crypto/ed25519.Sign(privateKey PrivateKey, message []byte) []byte` — *no* `io.Reader` because Ed25519 is deterministic (k = H(prefix||msg)).
- `crypto/ecdsa.SignASN1(rand io.Reader, priv *PrivateKey, hash []byte) ([]byte, error)` — explicit `io.Reader`, mandatory.
- `crypto/rsa.SignPKCS1v15(rand io.Reader, priv *PrivateKey, hash crypto.Hash, hashed []byte) ([]byte, error)` — explicit `io.Reader`.

**The pattern:** if the algorithm is deterministic-by-design (Ed25519, RFC 6979 deterministic-ECDSA), no randomness param. If the algorithm needs entropy (key gen, randomized-ECDSA, Schnorr), `io.Reader` is *always* the first parameter, *always* called `rand`.

**Recommendation for Reality's future API:** follow the stdlib convention exactly. `io.Reader` first, named `rand`, optional default to `crypto/rand.Reader` only if the param is nil. The argument *for* mandatory injection (no nil-default): testability — `rand io.Reader` lets tests pass a deterministic stream (one of Reality's existing PRNGs!) for golden-file validation. This is *the* synergy between Reality's deterministic-PRNG side and its future cryptographic side: **a `MersenneTwister.AsReader()` adapter (~20 LOC) lets every signature scheme be golden-file-tested with bit-exact reproducibility**, which is otherwise impossible for randomized signature schemes. **This is Tier-1 and unique to Reality** (libsodium / RustCrypto / dalek don't offer this because they don't expose deterministic PRNGs as a first-class type — they just have `crypto/rand`).

The API shape:
```go
type Reader interface { Read(p []byte) (n int, err error) }

// AsReader adapts MersenneTwister to io.Reader for testing.
func (mt *MersenneTwister) Read(p []byte) (n int, err error) {
    for n < len(p) {
        v := mt.Uint64()
        for j := 0; j < 8 && n < len(p); j++ {
            p[n] = byte(v); v >>= 8; n++
        }
    }
    return n, nil
}
```
Same for `PCG` and `Xoshiro256`. ~15 LOC per type, 45 LOC total. Pre-empts the entire "how do we golden-test a randomized signature scheme" problem.

---

## 5. Domain separation — do hash/sig functions expose domain-separation strings?

**No.** None of the four hash functions (`FNV1a32`, `FNV1a64`, `MurmurHash3_32`, `SituationHashWithStructure`) take a domain-separation parameter. None document one.

**Current state is defensible** for non-cryptographic hashes — domain separation is a cryptographic concern (preventing protocol-confusion attacks where a hash output computed in context A is replayed in context B). Non-crypto hashes don't *need* domain separation because they're not security-relevant. So: no critique of the present.

**For 057's primitives:** every modern signature/commitment scheme requires domain separation as a *mandatory* parameter, not optional:

- **BLS12-381 signatures (RFC 9380, hash-to-curve):** mandatory DST (Domain Separation Tag), e.g., `"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_"`. Without it, an attacker who has BLS signatures over messages signed in context A can forge "signatures" in context B.
- **Schnorr (BIP-340 secp256k1 Schnorr):** the challenge hash is `tagged_hash("BIP0340/challenge", ...)` where the tag is the domain separator.
- **Pedersen commitments:** the H generator must be domain-separated from G via `H = hash_to_curve("commitment_h", ...)`.
- **EdDSA (Ed25519):** RFC 8032 §5.1.6 specifies the prefix `dom2(F, C) || ...` for Ed25519ph and Ed25519ctx variants — the "ctx" *is* the domain separator. Bare Ed25519 omits it (and is the source of cross-context replay attacks against Ed25519ph users).

**Recommendation:** every hash-to-X / signature primitive 057 ships *must* take a `domain string` or `dst []byte` parameter as a mandatory argument, *not* an optional one and *not* with a default value. The stdlib idiom is `Sign(priv, msg)` because Ed25519 has no DST; for any new primitive, the idiom is `Sign(priv, dst, msg)` or `HashToCurve(input, dst)`. The argument *against* mandatory: it adds friction for the simple case. The argument *for* mandatory: every protocol-confusion CVE in the BLS / Schnorr / hash-to-curve literature traces to an optional-or-defaulted DST. **Mandatory wins.**

For the *existing* hash functions, **add an optional `Salted*` variant** if Reality wants to expose domain-separation discipline as a pattern early:
```go
// FNV1a64Tagged hashes (tag || data) for caller-provided domain separation.
// Useful when a single hash table contains keys from multiple sources.
func FNV1a64Tagged(tag, data []byte) uint64 { ... }
```
~15 LOC per hash function, signals to consumers that "yes, domain separation is a real concern here too" without overcommitting to crypto-grade DST yet.

---

## 6. Encoding — clear about big-endian vs little-endian?

**Mixed.** Three call sites where it matters:

**(6.1) `MurmurHash3_32` line 73–76:**
```go
k := uint32(data[i*4]) |
    uint32(data[i*4+1])<<8 |
    uint32(data[i*4+2])<<16 |
    uint32(data[i*4+3])<<24
```
This is **little-endian** by the byte indexing (`data[0]` is the low byte). This matches the reference C implementation by Austin Appleby. **Endianness is not documented in the docstring.** A C# port reading 4 bytes via `BitConverter.ToUInt32(data, 0)` on a big-endian platform would produce different output and silently disagree with the Go canonical value. Reality's golden-file system would catch this in test, but the *docstring* should call it out: `// Bytes are read in little-endian order.`

**(6.2) `Float64` methods on `MersenneTwister` / `PCG` / `Xoshiro256`:** convert uint64 → float64 by bit shifting. Endianness-neutral (bit operations are well-defined). But the *cross-language contract* — does Python's `(mt.next_u64() >> 11) / (1 << 53)` produce *exactly* the same float as Go's? Yes, IEEE 754 bit-exact. The doc could say so explicitly.

**(6.3) `StructuralDescriptor` line 187–204:** produces a byte slice where the layout is documented prose-style ("count, then key lengths") but not byte-precisely. A C++ port could emit count as a 16-bit big-endian value and silently disagree. The current Go impl uses single bytes (capped at 255), so the format is unambiguous *as long as you read the source*.

**Recommendation:** add an "Encoding" section to every function whose output is byte-level (currently `MurmurHash3_32`, `StructuralDescriptor`, and any future hash output). Format: one line, e.g., `// Encoding: little-endian byte read; little-endian uint32 output to caller.` This is a 4-LOC change across the package. **For 057's future primitives, the convention should be: every signature / hash / commitment that emits bytes documents big-endian-vs-little-endian for every multi-byte field.** The standard convention in cryptography is *big-endian network byte order* (RFC standards) for compatibility — Reality should default to big-endian for serialization to match RFC 8032 (Ed25519: little-endian, exception), RFC 6979 (deterministic ECDSA: big-endian), RFC 9380 (hash-to-curve: big-endian). The exception is Ed25519's little-endian scalar encoding, which is mandated by the spec — but inside the spec body the s-value, R-point, etc. are all little-endian *because that's what RFC 8032 says*, not because Reality chose. **Document the spec citation alongside the byte-order choice.** This is the "every function cites its source" rule (CLAUDE.md §4) extended to byte-layout.

---

## 7. Naming — `Sign(privKey, msg)` vs `SignWithKey(msg, privKey)`?

**Today's package is uniformly *(data, params)* order** — `ModPow(base, exp, mod)`, `ModInverse(a, mod)`, `MillerRabin(n, k)`, `MurmurHash3_32(data, seed)`, `IsPrime(n)`, `ChineseRemainder(residues, moduli)`. This is correct math-style: the *thing being operated on* comes first, the *modifier* comes second. Math-papers, Knuth-volumes, and number-theory textbooks all read this way (`gcd(a, b)`, `a^b mod m`).

**Crypto-stdlib convention is *(key, data)* order** — `ed25519.Sign(privateKey, message)`, `hmac.New(h, key).Write(data)` (where `key` is bound at constructor time, *before* data). The convention exists because keys are *the long-lived value*; data is *the transient value flowing through*. Method-style reads naturally: `priv.Sign(msg)` is `priv` doing the signing, `msg` being the operand. Free-function style mirrors this: `Sign(priv, msg)`.

**The seam.** When 057's signature schemes land, Reality has two consistent choices:

- **Option A: keep math-style everywhere.** `EdDSASign(msg, priv)`. Consistent within Reality, *inconsistent* with `crypto/ed25519` muscle memory. Any consumer migrating from stdlib will be tripped up. Risk: silent wrong-arg-order use, where `msg` and `priv` are both `[]byte` (in stdlib API) and the compiler doesn't catch the swap. Caught only by length mismatch at runtime — and a 32-byte short message would pass the length check.
- **Option B: math-style for non-crypto, crypto-style for crypto.** `ModPow(base, exp, mod)` (math) and `Ed25519Sign(priv, msg)` (crypto). Requires consumer to learn that the seam is "is this function in the crypto-primitives section of the package or the number-theory section?" Doc-able, navigable, but a real cognitive cost.
- **Option C: pivot the whole package to crypto-style.** `ModPow(mod, base, exp)` reads weird. **Rejected** — math-style is standard for math.

**Recommendation: Option B.** The cost is one line of doc per crypto function and a clear visual distinction in godoc (group all crypto primitives under a `// Cryptographic signatures` comment block; group all number-theory under `// Number theory`). This is what `crypto/ecdsa` and `crypto/ed25519` *implicitly* do by being separate packages — Reality's flat-namespace can do it via comment-block-grouping at zero structural cost.

**Crucial invariant for option B:** *secret values come before public values* in the argument list. `Sign(priv, msg)` (priv first), `Verify(pub, msg, sig)` (pub first), `Encrypt(pub, msg)`, `Decrypt(priv, ct)`. Consumer learns one rule: "the key is the first argument, public-or-private depending on direction."

A second invariant worth adopting: **never accept a key as `[]byte`** (see §2 above). `Ed25519Sign(priv Ed25519PrivateKey, msg []byte) Ed25519Signature` makes the wrong-arg-order swap a compile error. *This is the protection layer that compensates for crypto-style ordering.*

---

## 8. Comparison with `crypto/ed25519`, `crypto/cipher` stdlib idioms

**`crypto/ed25519` (Go stdlib, since Go 1.13).** Pattern:
```go
func GenerateKey(rand io.Reader) (PublicKey, PrivateKey, error)
func Sign(privateKey PrivateKey, message []byte) []byte
func Verify(publicKey PublicKey, message, sig []byte) bool

type PrivateKey []byte
type PublicKey  []byte

func (priv PrivateKey) Public() crypto.PublicKey
func (priv PrivateKey) Sign(rand io.Reader, message []byte, opts crypto.SignerOpts) ([]byte, error)
```
**Rules Reality should adopt verbatim:**
- Distinct named types for `PublicKey` / `PrivateKey` (slice, not opaque struct, in stdlib — but see §2 recommendation to upgrade to fixed-array or opaque struct).
- `GenerateKey` takes `io.Reader` first, *named* `rand`. Returns `(pub, priv, error)` in *that* order — public first, **a deliberate stdlib choice to match the order they appear in derived APIs**.
- `Sign(priv, msg)` and `Verify(pub, msg, sig)` are top-level functions, *not* methods on the key type (because Go interfaces over the slice type would require the slice to satisfy `crypto.Signer`, and the interface method `Sign(rand, msg, opts)` doesn't fit Ed25519's deterministic shape).
- **The interface `crypto.Signer` exists for polymorphic signing across schemes.** Reality's API can ignore this until a third signature scheme lands, then introduce `reality/crypto.Signer` as a sibling interface. Don't invent it preemptively (058's recommendation: "do NOT borrow the trait/generic substrate itself").

**Rules Reality should NOT adopt:**
- `crypto/ed25519` puts `Sign` / `Verify` as both top-level functions and methods. Slight redundancy. Reality should pick one and stick — recommend top-level functions only, until polymorphism is needed.
- `crypto/ed25519`'s `Sign` returning `[]byte` instead of a named type `Signature [64]byte`. The stdlib chose this for backward compatibility; Reality has no such constraint and should ship `Ed25519Signature [64]byte` from day one.

**`crypto/cipher` (Go stdlib).** Pattern:
```go
type Block interface {
    BlockSize() int
    Encrypt(dst, src []byte)
    Decrypt(dst, src []byte)
}

type AEAD interface {
    NonceSize() int
    Overhead() int
    Seal(dst, nonce, plaintext, additionalData []byte) []byte
    Open(dst, nonce, ciphertext, additionalData []byte) ([]byte, error)
}

func NewCBCEncrypter(b Block, iv []byte) BlockMode
```
**Out of scope for Reality** per 057's filter (AEAD, block ciphers excluded). But the pattern is instructive:
- **`Seal` and `Open`** as the AEAD verb-pair — opinionated naming, no `Encrypt`/`Decrypt`/`Authenticate` confusion. If Reality ever ships AEAD, mirror this verbatim.
- **`dst` as the first parameter** — output buffer reuse, zero-allocation friendly (matches CLAUDE.md §3 "no allocations in hot paths").
- **Nonce / IV as a separate argument**, never bundled with the key. `Seal(dst, nonce, pt, ad)` with the nonce as a free argument lets the caller manage uniqueness. (This is the libsodium misuse-resistance trick at the API level.)

**Reality's existing `MurmurHash3_32(data, seed)` already matches this convention in spirit** — `seed` is a free parameter, not bundled into a "context" struct. Good. Continue this for all future primitives.

---

## 9. Specific 30-LOC fixes (highest leverage, can land before 057)

In priority order:

**(F1) Fix package doc.** Replace the first 11 lines of `prime.go` (the `package crypto` doc-comment) with explicit non-cryptographic disclaimer:
```go
// Package crypto provides number theory primitives, NON-cryptographic hash
// functions, and deterministic NON-cryptographic pseudorandom number
// generators. All functions are pure, deterministic, and use only the Go
// standard library.
//
// SECURITY WARNING: Despite the package name, NO function in this package
// is suitable for cryptographic use. The hash functions are not collision-
// resistant; the PRNGs (MersenneTwister, PCG, Xoshiro256) are not
// cryptographically secure; ModPow and ModInverse are not constant-time.
// For cryptographic primitives, see [planned future API]. For randomness
// suitable for keys/nonces, use crypto/rand from the standard library.
```
**4 lines, prevents the most likely consumer mistake.**

**(F2) Add per-function security comments to `rng.go`.** One line each at the top of `NewMersenneTwister`, `NewPCG`, `NewXoshiro256`: `// SECURITY: NOT a CSPRNG. Outputs are predictable from seed (and recoverable from a few outputs). Do not use for keys, nonces, or any value requiring unpredictability.` **3 lines.**

**(F3) Add `// Encoding:` line to `MurmurHash3_32` and `StructuralDescriptor`.** Document little-endian byte read on Murmur, byte-precise descriptor format on StructuralDescriptor. **2 lines.**

**(F4) Add `// Constant-time: NO.` to `ModPow`, `ModInverse`, `ExtendedGCD`.** Explicit, short, unmissable. **3 lines.**

**(F5) Add `Read(p []byte) (n int, err error)` adapter methods to all three PRNGs.** Enables `io.Reader` injection for golden-file testing of future randomized schemes. **45 lines (~15 per type).** This is the single highest-leverage *capability* addition of the whole review — because it pre-empts the entire question "how do we golden-file-test a randomized signature scheme."

**(F6) Allocate witness slices once at package level in `prime.go`.** Move `witnesses32 := []uint64{2,3,5,7}` and `witnesses64 := []uint64{2,3,5,7,11,13,17}` to package vars. Hot-path allocation removed. **4 lines.**

**(F7) (CLAUDE.md change, not crypto/.go change) Add explicit allowance for `crypto/subtle` import.** Without this, the `ct.Equal` / `ct.Select` re-export package can't be added cleanly. **1 sentence change.**

**(F8) Add a `crypto/ct` sub-package** wrapping `crypto/subtle` with idiomatic-Go `bool`-returning function names. ~40 LOC. *Single most important pre-057 change for the misuse-resistance posture* of the package.

Total: ~100 LOC (mostly comments + adapter methods + ct sub-package). Net reduction in misuse risk: substantial. None of these conflict with 057's primitive-landings or 058's architectural recommendations.

---

## 10. Cross-references and what 059 deliberately doesn't cover

- **Numerical correctness of the existing primitives:** owned by **056**, not repeated here.
- **Which primitives to *add*:** owned by **057**, not enumerated here.
- **Architectural transferability from libsodium / RustCrypto / dalek / arkworks:** owned by **058**, not duplicated here.
- **Consumer-side migration of Recall / Phantom / Pistachio / Muse to use new key types:** out of scope; consumer-side is separate review tracks.
- **Memory hygiene (zeroization on drop, secret-buffer locking):** mentioned briefly in §2 (point D) but full treatment is a separate Tier-1 audit topic; deferred.
- **Cross-language API parity (Python / C++ / C# bindings):** out of scope for an API-ergonomics review of Go-canonical surface; deferred.

---

## Files referenced (absolute paths)

- C:\limitless\foundation\reality\crypto\prime.go (package doc, IsPrime, MillerRabin, GCD, ExtendedGCD)
- C:\limitless\foundation\reality\crypto\modular.go (ModPow, ModInverse, ChineseRemainder)
- C:\limitless\foundation\reality\crypto\hash.go (FNV1a32/64, MurmurHash3_32, ConsistentHash, SituationHashWithStructure, StructuralDescriptor)
- C:\limitless\foundation\reality\crypto\rng.go (MersenneTwister, PCG, Xoshiro256)
- C:\limitless\foundation\reality\crypto\crypto_test.go (880 LOC of test coverage; current API surface evidenced)
- C:\limitless\foundation\reality\reviews\overnight-400\agents\056-crypto-numerics.md (sibling: numerics)
- C:\limitless\foundation\reality\reviews\overnight-400\agents\057-crypto-missing.md (sibling: missing primitives)
- C:\limitless\foundation\reality\reviews\overnight-400\agents\058-crypto-sota.md (sibling: SOTA peers)

---

**Bottom line.** Reality's crypto/ today has a small, defensible API for what it actually is — a number-theory + non-crypto-hash + deterministic-PRNG kit. Its single most-load-bearing risk is the *package name* over-promising, which the F1+F2 doc fixes resolve in 7 lines. Its second-most-load-bearing risk is having *no constant-time primitive surface* and *no precedent for io.Reader injection* — both of which must be settled *before* 057's first cryptographic primitive lands, otherwise every subsequent primitive will inherit (or work around) the wrong shape. The F5 adapter (PRNG-as-io.Reader) is the hidden-gem capability that makes Reality uniquely well-positioned for golden-file-tested randomized-signature schemes — a property no SOTA peer has and that 058 implicitly counted but didn't name. Land F1–F8 before the first byte of 057 ships.
