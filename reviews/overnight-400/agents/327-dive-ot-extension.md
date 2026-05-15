# 327 — dive-ot-extension (OT / IKNP / KOS / Silver / VOLE — scope decision)

## Headline
OT extension is **out of scope for `reality`** — it is MPC protocol engineering (adversarial security model, malicious-vs-semi-honest threat reasoning, SHA-256 random oracles, network-message ordering) sitting one layer above the math primitive boundary; recommend an explicit OUT-OF-SCOPE clause in `CLAUDE.md` and defer to a downstream `realityMPC` library, since shipping IKNP/KOS without the surrounding crypto-protocol discipline (constant-time discipline 056, AES/SHA-256 322, ECC 292) creates a footgun the math-library posture cannot honor.

## Findings

### F1 — Zero MPC / OT surface confirmed
Repo-wide grep for `ObliviousTransfer|IKNP|OTExtension|GarbledCircuit|GMW|VOLE|SilentOT` across all `*.go` returns **zero matches** (`crypto/*.go`, all 22 packages). No `mpc/`, no `ot/`, no `gc/` package exists. CLAUDE.md package-table (22 entries) has no slot for any MPC-flavored package. The only crypto-adjacent code is `crypto/{hash.go, modular.go, prime.go, rng.go}` — 884 LOC of FNV/Murmur/MillerRabin/ModPow/MT19937/PCG/Xoshiro per slot 057. None of this is cryptographically loadbearing in the OT sense.

### F2 — OT extension is a protocol stack, not a math primitive
The 1-out-of-2 OT functionality `(m_0, m_1; b) -> m_b` does not factor into a clean "compute f(x)" math signature. A full OT protocol consists of:
1. **Setup phase** — public params, random-oracle instantiation (SHA-256 / SHAKE), CSPRNG seeding from OS entropy.
2. **Base-OT** — typically Naor-Pinkas 2001 (DDH) or Chou-Orlandi 2015 ("simplest OT" over Curve25519); both need a hardness assumption + EC point ops.
3. **Extension** — IKNP-style matrix transpose of correlated bits, KOS-style consistency check, Silver-style LPN decoding.
4. **Network message protocol** — k rounds of ciphertext exchange, ordered, with abort-on-mismatch semantics.
5. **Adversary model assertion** — semi-honest vs covert vs malicious; UC-secure vs stand-alone; static vs adaptive corruption.

Items (1), (4), (5) are explicitly **OUT** under 057 §scope-filter: "AEAD modes / TLS / network protocol state machines / OS entropy / wire formats". Only the matrix-transpose math in (3) and the EC scalar-mul in (2) are math primitives. Shipping just the math fragment without the protocol is dangerous: users will compose it incorrectly. Shipping the protocol violates `reality`'s "pure math, zero deps, MIT" charter.

### F3 — Five candidate primitives, tiered (if reality reverses scope)
| Tier | Primitive | LOC | Math content | Gates on |
|------|-----------|-----|--------------|----------|
| T0 | `ot.Sender / ot.Receiver` interfaces (Send(m0,m1), Receive(b)→mb), abstract message channel | ~40 | trivial — pure plumbing | nothing |
| T1 | Naor-Pinkas 2001 base 1-of-2 OT over a prime-order group (DDH-based: r→A=g^r, sender computes B0=A^r0, B1=A^r1, mask m_b with H(g^{a·r_b})) | ~150 | EC scalar-mul, SHA-256 KDF | slot 292 EC + slot 322 SHA-256 |
| T2 | IKNP 2003 OT extension — n base OTs become m≫n cheap OTs via bit-matrix transpose. Sender holds Δ ∈ {0,1}^κ, receiver holds choice bits; correlation `q_i = t_i ⊕ x_i·Δ`. Uses RO H, output `m_{i,0}=H(q_i), m_{i,1}=H(q_i⊕Δ)`. | ~250 | bit-matrix transpose mod 2 (cache-aware Eklundh transpose), SHA-256 RO calls, PRG expansion | T1 + slot 322 SHA-256 + AES-CTR PRG (slot 322 again) |
| T3 | KOS 2015 — IKNP + consistency-check (random linear combo of receiver's t-rows tested against Δ-mask via χ-squared style universal hash); maliciously secure under standard model with slack factor s≈40-bit. | ~300 | adds GF(2^κ) inner-products, statistical security slack proof | T2 + GF(2^κ) (slot 321) |
| T4 | Silver 2021 / Roy SoftSpokenOT 2022 — silent OT via LPN (Learning Parity with Noise) and an LDPC/expander code. Cuts communication from O(m·κ) to O(κ + small overhead) ≈ "communication-free OT" after setup. | ~400 | LPN syndrome decoding, expander-graph code (Silver code), short-LPN assumption | T3 + slot 320 ECC + slot 290 finite-field |
| T5 | VOLE / SoftSpokenOT vector-OLE F_p / Pseudorandom Correlation Generators (PCG, Boyle-Couteau-Gilboa-Ishai-Kohl-Scholl 2019/2020) | ~600 | LPN over F_p, syndrome decoding, expander codes, Reverse-MultiplicationFriendlyEmbeddings | T4 + slot 293 NTT + slot 320 RS |

### F4 — Why this slot's "should we ship?" answer is **NO** for v0.x
Three independent reasons converge:

**R1 — Scope charter violation.** CLAUDE.md §6 "Reimplement from first principles" + §1 "Zero dependencies" + §2 "math primitives" are violated by anything in T2+. KOS's malicious security relies on a tight-knot statistical argument that is *not* a self-contained math fact — it is a protocol-level claim ("if A_i passes consistency check then receiver is honest with prob ≥ 1−2^{-s}"). This cannot be cross-validated by Python/C++/C# golden vectors in the same way as `Sin(0.5) = 0.479425...`. The IKNP transpose is testable (deterministic given seeds), but the *security claim* is not — and shipping the transpose without the security claim is shipping a footgun.

**R2 — Random-oracle dependency unavailable.** Slot 322 found reality ships **zero** SHA-256 / SHAKE-256. IKNP needs ~m·κ SHA-256 invocations as a random oracle (~80M for m=2^20, κ=128). T1 base OT needs SHA-256 as a KDF. T2+ all gate on slot 322's SHA-256 landing first, and slot 322 itself recommends "defer to stdlib (`crypto/sha256`)" rather than a from-scratch impl — which means a from-scratch reality OT would inevitably import `crypto/sha256` (a stdlib import, technically allowed) but golden-file determinism becomes language-dependent on stdlib SHA-256 availability (Python `hashlib`, C++ has no stdlib SHA, C# `System.Security.Cryptography.SHA256`).

**R3 — Adversarial-review gap.** Reality's review process (this overnight-400) is mathematician-flavored: numerical stability, golden-file precision, IEEE-754 edge cases. It is NOT cryptographer-flavored: no constant-time-CT discipline (056), no side-channel review, no UC-framework proof verification. Shipping a "half-secure" OT extension is strictly worse than shipping nothing — downstream users (Pistachio, aicore) would assume `reality.OT` is malicious-secure when it is at best semi-honest.

### F5 — But IF reality decides to ship: it composes neatly
Should the scope decision later flip (e.g., a `realityMPC` sister-package is launched with separate cryptographic-review process), the math primitives compose cleanly atop already-scoped slots:
- **Bit-matrix transpose mod 2** (the IKNP keystone) is pure combinatorics, fits in `combinatorics/` or a new `linalg/gf2/` sub-package. Cache-aware Eklundh transpose is well-defined and golden-file-able byte-exactly.
- **GF(2^κ) inner products** for KOS consistency check sit in slot 321 GF(2^m).
- **LPN syndrome decoding** for Silver/SoftSpokenOT sits in slot 320 error-correction (Silver code is LDPC-style; same machinery as Reed-Muller / LDPC consumers).
- **Naor-Pinkas base OT** is just three EC scalar-muls + one SHA-256, sits atop slot 292 EC.
- **VOLE PCG (T5)** is the math frontier — pseudorandom-correlation expansion via LPN is genuinely new mathematics (2019-2024) and is the only OT-extension primitive with first-class "this is interesting math" character (vs. T2-T3 which are clever protocol gadgets).

### F6 — Day-1 PR recommendation: **NONE — instead ship a CLAUDE.md scope clarification**
The cheapest day-1 PR is a 5-line addition to CLAUDE.md §Key Design Rules:

```md
7. **Out of scope: cryptographic protocols.** Reality ships math primitives
   (finite-field, EC point arithmetic, hash compression, bit-matrix transpose,
   LPN syndrome decode), NOT MPC protocols built atop them (oblivious transfer,
   garbled circuits, secret sharing, ZK proof systems above the polynomial-
   commitment layer). Protocols belong in a sister package (e.g., realityMPC,
   realityZK) with separate cryptographic-review discipline.
```

This single rule resolves not just OT-extension but garbled circuits (Yao 1986, Kolesnikov-Schneider 2008 free-XOR, Zahur-Rosulek-Evans 2015 half-gates), GMW 1987, BGW 1988, SPDZ (Damgård-Pastro-Smart-Zakarias 2012), MASCOT (Keller-Orsini-Scholl 2016), Overdrive (Keller-Pastro-Rotaru 2018), private-set-intersection protocols (KKRT 2016, RR17 BaRK-OPRF, CM20, OOS17), federated-learning aggregation protocols (Bonawitz 2017 secure-agg). All of these get answered by the same scope rule.

### F7 — Cross-link map (downstream impact if reality stays scope-pure)
- **Pistachio** — does not need OT. No impact.
- **aicore** — uses ML inference, not MPC. No impact.
- **conduit / forge** — if either ever needs private-state replication, would consume `realityMPC` (downstream), not `reality` directly. No impact.
- **External MPC consumers** — if any future limitless-stack project needs OT (e.g., a private-DB-join or federated-learning module), it imports `realityMPC` which itself imports `reality` for SHA-256, EC, GF(2^m), bit-transpose, LPN-decode. Clean DAG.

### F8 — Counterfactual: if reality DID ship OT (sequencing)
For the record, if the scope decision flips, the recommended sequence is:
1. **Pre-req**: slots 322 SHA-256, 292 ECC (Curve25519 or BLS12-381 G1), 321 GF(2^m), 320 ECC-codes.
2. **Stage 1** (T0+T1, ~190 LOC): generic `ot.Protocol` interface + Naor-Pinkas DDH base OT. Semi-honest only. Ships golden-vector `(seed, m0, m1, b) -> mb` test.
3. **Stage 2** (T2, ~250 LOC): IKNP extension. Semi-honest. Ships golden-vector `(seed, n_base=128, m_extended=2^20, choices, messages) -> bit-exact transcript hash`.
4. **Stage 3** (T3, ~300 LOC): KOS consistency-check upgrade. Maliciously secure. Requires statistical-soundness proof annotation in code comments + UC-citation.
5. **Stage 4** (T4-T5, ~1000 LOC): Silver / Soft-SpokenOT / VOLE-PCG. Frontier; ship after T3 has been audited externally.

Total ~1740 LOC. **Not recommended** until charter is amended.

### F9 — Implementation pitfalls (if shipped — for future reference)
1. **Random-oracle modeling.** IKNP proves security in ROM. Using SHA-256 directly is fine; using a non-collision-resistant compression (e.g., FNV which `crypto/hash.go` ships) is a complete break. Must use SHA-256 / SHAKE-256 / BLAKE2b — none of which reality ships today.
2. **Base-OT count κ.** Computational security parameter; 128 for AES-128 level. KOS 2015 strengthens this to κ + s where s ≈ 40-80 statistical security; running with κ=128, s=0 (vanilla IKNP) is malicious-insecure (Asharov-Lindell-Schneider-Zohner 2017 attack on IKNP w/o consistency check).
3. **Bit-matrix transpose memory layout.** Eklundh O(n·m·log) cache-blocked vs naive O(n·m); naive thrashes L1 for κ=128, m=2^20 (16MB matrix). Cache-aware version is 30-100x faster, must be the default.
4. **PRG choice for column expansion.** AES-CTR is standard (using AES-NI on x86); reality has no AES. Could fall back to ChaCha20 (also absent in reality per slot 322) or to SHAKE-128 (also absent). All paths require landing slot 322 first.
5. **Constant-time conditional select** in base OT — receiver computes `B_b = A^r_b` where b is the secret choice bit. Branch on `b` leaks via timing. Must use `(b·B1) + ((1-b)·B0)` with constant-time scalar select. Reality has zero CT discipline today (056).
6. **Consistency check in KOS.** Computes `χ = sum_i x_i · t_i ∈ GF(2^κ)`; sender verifies `χ_sender = χ_receiver ⊕ Δ·sum(x_i)`. Get the GF(2^κ) reduction polynomial wrong and the protocol breaks. Reality has no GF(2^κ) (slot 321).
7. **Silent OT setup non-uniformity.** Silver-code parameters are tuned per (n, t)-LPN-noise pair; using off-the-shelf Boyle 2019 Silver5 params for arbitrary `m` reduces concrete security below claimed level. Must ship parameter table from libOTe.

### F10 — Comparison to existing OT/MPC libraries (zero-deps angle)
| Library | Lang | LOC | Deps | Coverage | License |
|---------|------|-----|------|----------|---------|
| EMP-toolkit (Wang-Malozemoff-Katz 2017) | C++17 | ~50k | OpenSSL, GMP, RELIC | Yao GC + IKNP/KOS/Ferret OT + ZK | MIT |
| MOTION (Braun-Demmler-Schneider-Zohner 2022) | C++20 | ~80k | Boost, OpenSSL, fmt, flatbuffers | BMR/GMW/BGMW/SP-DZ-flavored | MIT |
| MP-SPDZ (Keller 2020) | C++14 | ~100k | OpenSSL, MPIR/GMP, libsodium | SPDZ family + offline OT extension | BSD-3 |
| libOTe (Rindal 2017+) | C++17 | ~30k | OpenSSL, libsodium, RELIC, coproto | IKNP/KOS/Ferret/Silver/SoftSpokenOT | Unlicense |
| swanky/ocelot (Galois 2019) | Rust | ~10k | curve25519-dalek, sha2, blake2 | IKNP/KOS/ALSZ | MIT |

All depend on cryptographic libraries (OpenSSL, libsodium, dalek). **None** are zero-dep — reality's posture is genuinely incompatible. Even libOTe, the most "from-scratch" of the OT-only libs, vendors RELIC for EC and OpenSSL for AES-NI/SHA. There is no zero-dep, golden-file-cross-language OT library in any language; building one is a 6-12-month effort in its own right and should not happen as a side-effect of `reality`'s math charter.

## Concrete recommendations
1. **Accept this slot as OUT-OF-SCOPE.** Ship a CLAUDE.md amendment per F6 to declare cryptographic protocols (OT, GC, SS, ZK above polynomial-commitment math, secure-aggregation) as out-of-scope. This is the cheapest day-1 PR (5 LOC of docs).
2. **Note the in-scope math fragments.** When slots 320 (ECC), 321 (GF(2^m)), 322 (SHA-256), 292 (EC) land, the *underlying math primitives* (bit-matrix transpose mod 2, GF(2^κ) inner product, LPN syndrome decode, EC scalar mul, SHA-256 RO) all exist. A future `realityMPC` package can compose them.
3. **Cross-reference 057-crypto-missing.** That slot already lists "oblivious transfer, garbled circuits" as topic-listed primitives but defers them. This slot (327) provides the deeper analysis explaining *why* deferral is correct — the math is reachable but the protocol layer isn't.
4. **If/when scope flips, sequence as:** slot 322 SHA-256 → slot 292 EC → T0 interface (~40 LOC) → T1 Naor-Pinkas base OT (~150 LOC) → T2 IKNP semi-honest extension (~250 LOC) → T3 KOS malicious extension (~300 LOC) → defer T4 Silver / T5 VOLE to a v2 frontier track.
5. **Do NOT ship the bit-matrix transpose alone "as a math primitive".** It is mathematically clean but its sole purpose is OT extension; landing it in `linalg/gf2/` invites users to implement broken OT atop it. If the math primitive is wanted standalone (e.g., for binary-LDPC codes in slot 320), name it `linalg/binarymatrix.Transpose` with explicit non-OT documentation.
6. **Track frontier via slot 327 followups.** SoftSpokenOT 2022, Ferret 2020, Silver 2021, PCG 2019-2024 are an active research area — even if reality stays scope-pure, the math primitives those papers introduce (Silver code, LPN-PCG, expander codes) are math worth tracking for slot 320 / 321 / 290 expansions.

## Sources

### Repo files (audited)
- `C:\limitless\foundation\reality\CLAUDE.md` — package list (22), zero MPC slot
- `C:\limitless\foundation\reality\crypto\hash.go` — FNV/Murmur (non-cryptographic) only
- `C:\limitless\foundation\reality\crypto\modular.go` — uint64 ModPow / ModInverse / ExtGCD / CRT
- `C:\limitless\foundation\reality\crypto\prime.go` — Miller-Rabin
- `C:\limitless\foundation\reality\crypto\rng.go` — MT19937 / PCG / Xoshiro (non-CSPRNG)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\057-crypto-missing.md` — lists OT/GC as topic-listed-but-deferred
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\322-dive-aes-vs-poly1305.md` — confirms zero SHA-256/AES surface (gate for OT)
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\292-???-elliptic-curves` (planned, slot 292 in MASTER_PLAN line 309) — gate for base-OT
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\321-dive-finite-field.md` — GF(2^m) gate for KOS consistency check
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\320-dive-error-correction.md` — Reed-Solomon / LDPC gate for Silver

### External / academic
- Ishai, Kilian, Nissim, Petrank 2003. *Extending Oblivious Transfers Efficiently*. CRYPTO. [foundational IKNP]
- Asharov, Lindell, Schneider, Zohner 2013. *More Efficient Oblivious Transfer and Extensions for Faster Secure Computation*. CCS.
- Keller, Orsini, Scholl 2015. *Actively Secure OT Extension with Optimal Overhead*. CRYPTO. [KOS]
- Naor, Pinkas 2001. *Efficient oblivious transfer protocols*. SODA.
- Chou, Orlandi 2015. *The Simplest Protocol for Oblivious Transfer*. LATINCRYPT.
- Boyle, Couteau, Gilboa, Ishai, Kohl, Scholl 2019. *Efficient Pseudorandom Correlation Generators: Silent OT Extension and More*. CRYPTO.
- Couteau, Rindal, Raghuraman 2021. *Silver: Silent VOLE and Oblivious Transfer from Hardness of Decoding Structured LDPC Codes*. CRYPTO.
- Roy 2022. *SoftSpokenOT: Quieter OT Extension From Small-Field Silent VOLE in the Minicrypt Model*. CRYPTO.
- Yang, Sarkar, Weng, Wang 2020. *Ferret: Fast Extension for Correlated OT with Small Communication*. CCS.
- Yao 1986. *How to Generate and Exchange Secrets*. FOCS. [garbled circuits]
- Goldreich, Micali, Wigderson 1987. *How to Play any Mental Game*. STOC. [GMW]
- Damgård, Pastro, Smart, Zakarias 2012. *Multiparty Computation from Somewhat Homomorphic Encryption*. CRYPTO. [SPDZ]

### External libraries (referenced for scoping comparison only — none vendored)
- libOTe (Rindal 2017+, Unlicense): https://github.com/osu-crypto/libOTe — ~30k LOC C++17, with OpenSSL/RELIC deps
- EMP-toolkit (Wang-Malozemoff-Katz 2017, MIT): https://github.com/emp-toolkit
- MP-SPDZ (Keller 2020, BSD-3): https://github.com/data61/MP-SPDZ
- MOTION (Braun-Demmler-Schneider-Zohner 2022, MIT): https://github.com/encryptogroup/MOTION
- swanky / ocelot (Galois 2019, MIT): https://github.com/GaloisInc/swanky
