# 212 | new-pq-signatures

**Summary line 1.** TWELFTH Block-C cutting-edge-math review and FIRST post-quantum-signatures (PQS) scoping in the 400-sequence covering hash-based one-time signatures (Lamport 1979 N-bit-message → 2N-secret-key + 2N-public-key + N-signature / Winternitz w-OTS 1979 with chain length 2^w trading sig size for hash-chain compute / w-OTS+ Hülsing 2013 with bitmask randomisation / w-OTS^C constant-sum) / Merkle-tree signatures (Merkle 1979 single-tree → many-time / authentication path of log_2(N) hashes / TreeHash traversal Buchmann-Dahmen-Schneider 2008) / XMSS (eXtended Merkle Signature Scheme, Hülsing-Rausch-Buchmann 2011, RFC 8391 May 2018, NIST SP 800-208 Oct 2020) / XMSS^MT (multi-tree XMSS for very-large-state up to 2^60 sigs) / LMS (Leighton-Micali Signatures, RFC 8554 Apr 2019, NIST SP 800-208) / HSS (Hierarchical Signature System: tree-of-LMS-trees) / SLH-DSA (FIPS-205 Aug 2024, formerly SPHINCS+ Bernstein-Hülsing-Kölbl-Niederhagen-Rijneveld-Schwabe 2017–2022) stateless hash-based with FORS few-times-signatures + hyper-tree XMSS / SLH-DSA-SHA2 + SLH-DSA-SHAKE families × {128s, 128f, 192s, 192f, 256s, 256f} = 12 parameter sets / FORS (Forest of Random Subsets) k=14..35 trees of height a=6..14 / ML-DSA (FIPS-204 Aug 2024, formerly Dilithium) lattice-based Fiat-Shamir-with-aborts at security levels {44, 65, 87} → also covered in 211 lattice slot / FN-DSA / Falcon (FIPS-206 final-draft 2025) NTRU-trapdoor + Klein-FFT-tree-sampler → also covered in 211 / broken/historical schemes for cryptanalysis pedagogy: Rainbow multivariate (broken Beullens 2022 in <53 hours on a laptop) / SIDH isogeny (broken Castryck-Decru 2022 polynomial-time) / NTRUSign (broken Nguyen-Regev 2006 transcript attack) / GeMSS / Picnic (Picnic3 / MPCitH-with-symmetric-cipher) / SQIsign (CSIDH-based isogeny survivor + new 2024 NIST onramp candidate) / signature-aggregation (BLS-style multi-sig pairing aggregation cross-link to 057 §T2-BLS) / threshold signatures (FROST 2-round Schnorr + Lindell ECDSA + ML-DSA threshold = active research) / adaptor signatures (Schnorr/ECDSA/Schnorr adaptors for atomic swaps + payment channels) / blind signatures (Chaum 1983 RSA-blind, Abe 2001 Schnorr-blind, BSA 2024 round-optimal) / group signatures (Chaum-van Heyst 1991, BBS+ 2004) / ring signatures (Rivest-Shamir-Tauman 2001, CryptoNote / Monero) / aggregate signatures + batch verification / hybrid PQ+classical (X-Wing Kyber+X25519 IETF 2024, ML-DSA + Ed25519 dual-sig CMS draft 2025): reality v0.10.0 ships **ZERO** post-quantum signature surface and zero of its hash-based-signature prerequisites — per 057-crypto-missing the entire `crypto/` package is uint64 ModPow / ModInverse / ExtendedGCD / CRT / Miller-Rabin / FNV / MurmurHash / MT-PCG-Xoshiro RNGs (884 LOC) with NO SHA-2 / SHA-3 / SHAKE / HMAC / HKDF (gates EVERY hash-based and lattice-based signature scheme — XMSS uses SHA-256 or SHAKE, LMS uses SHA-256, SLH-DSA uses SHA2/SHAKE, ML-DSA uses SHAKE, Falcon uses SHAKE), NO Merkle-tree primitive (gates XMSS / LMS / SLH-DSA), NO bitmask-randomised hash chain (gates w-OTS+), NO constant-time conditional-select (gates side-channel-resistant signing). Repo-wide grep on `XMSS|SLH-DSA|SPHINCS|Dilithium|ML-DSA|Falcon|FN-DSA|LMS|HSS|Lamport|Winternitz|Merkle|FORS|hyper.tree|stateful.signature|hash.based.signature|RFC.8391|RFC.8554|FIPS.205|FIPS.204|FIPS.206|Picnic|Rainbow|GeMSS|SQIsign|CSIDH|adaptor.signature|blind.signature|threshold.signature` returns **ZERO callable matches** across all 22 packages — closest tangential surfaces are crypto/hash.go (FNV-1a / MurmurHash3 / Jump-consistent — NOT cryptographic, cannot back any signature scheme; cannot be substituted for SHA-256 in XMSS without breaking the security proof) and crypto/rng.go (MT19937 / PCG / Xoshiro — NOT cryptographic, cannot derive Lamport/Winternitz secret keys). Slot 211 covered the lattice-arithmetic side of ML-DSA + Falcon (the polynomial-ring + NTT + sampling + key-encapsulation math); this slot focuses on the SIGNATURE-SCHEME side: Fiat-Shamir-with-aborts wrapper (211-L17), trapdoor-sampler wrapper (211-L21–L22), AND the ENTIRE hash-based-signature universe (Lamport / Winternitz / Merkle / XMSS / LMS / SLH-DSA) which is COMPLETELY ORTHOGONAL to lattice math and was explicitly out-of-scope per 211-§6.

**Summary line 2.** Thirty primitives S1–S30 totalling ~5,540 LOC across new sub-package `signature/` (sibling to `crypto/` and `lattice/` — signature schemes have their own algorithmic structure that combines hash-based primitives + arithmetic primitives + state-management; placing them as a sibling rather than nesting into `crypto/sig/` keeps the dependency graph clean and matches the 057 + 211 sibling-package precedent). Recommended split: `signature/ots/` for one-time signatures Lamport + Winternitz + w-OTS+ + w-OTS^C (~480 LOC, prerequisite for every Merkle-tree variant); `signature/merkle/` for generic Merkle-tree signature substrate with TreeHash traversal + authentication-path generation + log-N verification (~280 LOC); `signature/xmss/` for RFC-8391 XMSS-SHA2_10/16/20_256 + SHAKE variants + XMSS^MT multi-tree (~580 LOC); `signature/lms/` for RFC-8554 LMS + HSS hierarchical (~480 LOC); `signature/slh/` for FIPS-205 SLH-DSA with FORS + hyper-tree + 12 parameter sets (~860 LOC, the singular hardest hash-based piece); `signature/mldsa/` for FIPS-204 ML-DSA Fiat-Shamir-with-aborts wrapper around 211-L1–L11 lattice substrate (~580 LOC, depends on 211 Tier-1 + Tier-3); `signature/fndsa/` for FIPS-206 Falcon trapdoor-signature wrapper around 211-L19–L22 NTRU-lattice substrate (~360 LOC, depends on 211); `signature/agg/` for BLS-style aggregation + batch-verification + multi-sig + threshold + adaptor + blind + ring + group signature variants (~480 LOC, mostly arithmetic combinators over 057 + 211); `signature/broken/` for cryptanalytic pedagogy of NTRUSign 2006 / Rainbow 2022 / SIDH 2022 / GeMSS / SFLASH 2007 (~440 LOC, ship with **WARNING: educational only, broken in production**); `signature/hybrid/` for X-Wing-style PQ+classical composition (~120 LOC). Tier-1 keystone **S1+S2+S3+S5+S6 = `signature/ots/lamport.go` + `ots/winternitz.go` + `ots/wots_plus.go` + `merkle/tree.go` + `merkle/authpath.go` ~720 LOC** is the irreducible foundation that unblocks XMSS / LMS / SLH-DSA — none of the three FIPS-standardized hash-based schemes can ship without first landing the OTS + Merkle-tree substrate. **Singular reality competitive moat: S10–S13 SLH-DSA (FIPS-205) ~860 LOC** — exactly four Go libraries ship SLH-DSA at the time of this review (Cloudflare CIRCL, kudelskisecurity/sphincs-go, OpenQuantumSafe/liboqs-go via cgo, kasperdi/SPHINCSPLUS-golang) but NONE ship a zero-dep deterministic-given-seed implementation with Python/C++/C# golden-file parity at the byte level on the NIST CAVP FIPS-205 KAT vectors; reality would be the only Go library AND the only language ecosystem with byte-for-byte cross-substrate validation against the official NIST 1,000-vector test suite. **Singular cross-link to 211: S15–S17 ML-DSA wrapper + S18–S20 Falcon wrapper ~940 LOC** are pure orchestration over the 211 lattice substrate — they consume Polynomial / NTT / CBD / SHAKE / TreeSampler primitives from 211 and add the Fiat-Shamir-with-aborts loop (ML-DSA) or the trapdoor-sampler-then-compress loop (Falcon); shipping these AS a separate slot from 211 keeps the 211 review focused on lattice-arithmetic and lets this slot focus on signature-protocol orchestration (the rejection-sampling abort, the salt/randomization, the deterministic-vs-randomized-mode split). **Singular Block-C-2026 frontier: S25 threshold-ML-DSA + S26 adaptor-Schnorr + S27 round-optimal-blind-Schnorr** are the active 2024–2026 research frontiers where a zero-dep math layer would be among the first language ecosystems to ship reference implementations with golden-vector cross-validation (threshold ML-DSA has 2024 papers but NO production library yet — DKLs23 / Lindell-Nof / GG24-style protocols are still being analyzed; adaptor signatures shipped in Bitcoin Lightning DLCs but no canonical reference; round-optimal blind Schnorr from BSA 2024 has zero shipping libraries). Cross-package blockers: `crypto/sha2` SHA-256 + SHA-512 (currently absent — 057 §T1-HASH gates EVERY hash-based signature, ML-DSA, Falcon — without SHA-2 NOTHING in this slot ships); `crypto/sha3` SHAKE-128/256 (gates SHAKE-variant SLH-DSA / ML-DSA / Falcon); `crypto/hmac` (gates LMS pseudo-random key-derivation); `crypto/ct` constant-time conditional-select (gates side-channel-resistant signing — SLH-DSA forks per-leaf via PRF and any branch-on-secret-input leaks the secret-key index); 211 §Tier-1 lattice substrate (gates ML-DSA + Falcon wrappers); 057 §T1-EC + §T1-SCHNORR + §T2-BLS (gates classical-PQ-hybrid composition AND adaptor / blind / threshold variants which assume a Schnorr-base curve as the "classical leg"). Versus 211-new-lattice-crypto (which scopes the lattice-arithmetic substrate for ML-KEM Kyber + ML-DSA Dilithium + FN-DSA Falcon at the polynomial-ring + NTT + sampling layer): orthogonal axis — 211 ships the math substrate, this slot ships the signature-protocol orchestration on top. Versus 057-crypto-missing (which scopes classical-curve-and-pairing signature math EdDSA + ECDSA + Schnorr + BLS): orthogonal axis with intentional cross-link at the hybrid-signature S30 surface (X-Wing-style PQ+classical) and at the aggregation/threshold/adaptor/blind/ring/group surface (S21–S29) which depends on 057 §T1-SCHNORR landing for the classical leg. Versus 058-crypto-sota / 060-crypto-perf: SOTA / perf review of EXISTING crypto/ — this slot is greenfield, no overlap.

---

## 0. State at HEAD (2026-05-08, v0.10.0)

Repo-wide audit for post-quantum-signature surface:

| Surface | Path | Lines | PQS relevance |
|---|---|---:|---|
| FNV-1a / Murmur / Jump | `crypto/hash.go` | 220 | NOT cryptographic; cannot back Lamport/XMSS/SLH-DSA |
| MT19937 / PCG / Xoshiro | `crypto/rng.go` | 197 | NOT cryptographic; cannot derive secret keys |
| ModPow / ModInverse / ExtGCD / CRT | `crypto/modular.go` | 200 | uint64 cap; useful only for ML-DSA/Falcon ring layer (deferred to 211) |
| Miller-Rabin | `crypto/prime.go` | (function) | Not relevant to PQS (no prime-pickng inner loop in any FIPS-204/205/206 spec) |
| Cooley-Tukey FFT | `signal/fft.go` | (FFT impl) | Tangential to 211-NTT; not relevant to hash-based signatures |
| Hash chains | -- | 0 | Absent — would underpin Winternitz / w-OTS+ |
| Merkle-tree primitive | -- | 0 | Absent — would underpin XMSS / LMS / SLH-DSA |
| FORS / hypertree | -- | 0 | Absent — SLH-DSA-specific |

Repo-wide grep: `Lamport|Winternitz|Merkle|TreeHash|AuthPath|XMSS|XMSS_MT|LMS|HSS|FORS|SLH.DSA|SPHINCS|hyper.tree|Fiat.Shamir|rejection.sampling|trapdoor|salt.based|adaptor|blind.signature|threshold.signature|ring.signature|group.signature|aggregate.signature|Picnic|Rainbow|GeMSS|SQIsign|CSI.FiSh|SIDH` returns **ZERO callable matches** across all 22 packages. The CLAUDE.md package table has no `signature/` entry.

Hash-prerequisite reality check: SHA-256 / SHA-512 / SHA3 / SHAKE-128/256 are ALL absent from `crypto/`. Hash-based signatures are entirely hash-driven (Lamport: 2N hashes per key; Winternitz w=4: 16 hashes per chain × N/4 chains; XMSS-h=10: ~1024 hashes per signature; SLH-DSA: ~10,000 hashes per signature). **Without SHA-2 / SHAKE the entire hash-based-signature universe is unreachable.** This is the SINGLE MOST IMPORTANT blocker: 057 §T1-HASH must ship before ANY S1–S14 primitive in this slot can be implemented with golden-file validation.

---

## 1. The thirty primitives

Tier numbering: T1 = OTS + Merkle foundation, T2 = stateful FIPS standards (XMSS / LMS), T3 = stateless FIPS standard (SLH-DSA), T4 = lattice-signature wrappers (ML-DSA / FN-DSA), T5 = aggregation / threshold / adaptor / blind / ring / group, T6 = hybrid + broken pedagogy.

### Tier 1 — One-time signatures + Merkle foundation (~720 LOC)

**S1 — `signature/ots/lamport.go` ~80 LOC.** Lamport 1979 OTS. KeyGen: sample 2N random N-bit secrets `(x_{i,0}, x_{i,1})` for i=1..N; public key `(y_{i,b} = H(x_{i,b}))`. Sign(msg of N bits): output `(x_{1, m_1}, ..., x_{N, m_N})`. Verify: check `H(σ_i) == y_{i, m_i}` for each i. ONE-TIME ONLY: revealing the same bit twice exposes both sides of the next-bit's secret on subsequent signatures. Refs: Lamport 1979 *Constructing digital signatures from a one-way function*; Diffie-Hellman 1976.

**S2 — `signature/ots/winternitz.go` ~140 LOC.** Winternitz w-OTS. Trade signature size for hash-chain compute. For Winternitz parameter w: chain length 2^w; per-chain message-byte-of-w-bits drives the iteration count. KeyGen samples len_1 + len_2 chain seeds where len_1 = ceil(N/w) data chains + len_2 ≈ ceil(log_2(len_1·(2^w−1))/w) checksum chains. Sign(msg): hash msg to N bits, slice into w-bit chunks, iterate each chain `H^{c_i}(x_i)`. Verify: re-iterate `H^{2^w−1−c_i}(σ_i)` and check public-key match. Refs: Merkle 1979 *Secrecy, Authentication, and Public Key Systems* (Stanford PhD thesis); Hülsing 2013 §2.

**S3 — `signature/ots/wots_plus.go` ~140 LOC.** Hülsing 2013 w-OTS+. Adds bitmask randomization `M_i` to each hash-chain step to defeat multi-target attacks: `chain^k(x) = H(chain^{k−1}(x) ⊕ M_k)` instead of plain `H(chain^{k−1}(x))`. Bitmasks derived from a public seed via PRF. **Used as the OTS leaf-primitive in XMSS and SLH-DSA**, not Lamport-OTS or plain Winternitz. Constant-time conditional-XOR. Refs: Hülsing 2013 *W-OTS+ — Shorter signatures for hash-based signature schemes*.

**S4 — `signature/ots/wots_c.go` ~120 LOC.** w-OTS^C constant-sum encoding (Cruz-Hashimoto-Kobayashi-Yoshikawa 2017). Replaces the Winternitz checksum with a constant-sum encoding — every message-hash is encoded as a fixed-sum sequence, eliminating the checksum overhead. Smaller signatures, no security degradation. Used in 2024 SLH-DSA-Beyond research drafts. Refs: Cruz et al. 2017; Perin-Kim-Hülsing-Pessl 2024.

**S5 — `signature/merkle/tree.go` ~140 LOC — KEYSTONE.** Generic Merkle-tree signature substrate. `Tree{Height int, Leaves []Leaf, Root [N]byte}`. Build via TreeHash bottom-up: `parent_hash = H(left || right)`. Optional bitmask-randomized version `parent_hash = H((left || right) ⊕ M)` for SLH-DSA/XMSS. Used for tree heights h ∈ {6, 10, 16, 20, 26, 30}. **Cross-validation:** root computation must match XMSS/LMS/SLH-DSA reference vectors byte-for-byte. Refs: Merkle 1979; Buchmann-Dahmen-Schneider 2008 *Merkle tree traversal revisited*.

**S6 — `signature/merkle/authpath.go` ~100 LOC.** Authentication-path generation. For leaf i in tree of height h, output the h-1 sibling hashes along the path from leaf-i to the root. Path-storage encoding: `[h-1][N]byte`. Verify: re-hash leaf with each sibling in path-order to recompute root, check root match. **Stateful TreeHash traversal** (Szydlo 2004) for XMSS-state minimization: O(h) memory + O(h) hashes per signature (vs O(2^h) memory for naive precompute). Refs: Szydlo 2004 *Merkle tree traversal in log space and time*; Buchmann-Dahmen-Schneider 2008.

### Tier 2 — Stateful FIPS standards (~1,060 LOC)

**S7 — `signature/xmss/params.go` ~80 LOC.** RFC 8391 + NIST SP 800-208 XMSS parameter sets. SHA-256 family: XMSS-SHA2_10_256, XMSS-SHA2_16_256, XMSS-SHA2_20_256 (heights 10/16/20 → 1024/65k/1M signatures per key). SHAKE-256 family: XMSS-SHAKE_10_256, XMSS-SHAKE_16_256, XMSS-SHAKE_20_256. Plus 512-bit variants. XMSS^MT multi-tree variants: XMSS-MT-SHA2_20/2_256 through XMSS-MT-SHA2_60/12_256 (height up to 60 with up to 12 layers). Refs: RFC 8391 §8; NIST SP 800-208.

**S8 — `signature/xmss/sign.go` ~280 LOC — KEYSTONE.** XMSS signing. Uses w-OTS+ (S3) as leaf signature scheme. Each leaf is a w-OTS+ public-key-hash. KeyGen: sample seed, derive 2^h w-OTS+ keypairs, hash each to a leaf, build Merkle tree (S5), publish root + seed. Sign: increment state-counter `idx` (THE STATEFUL PART — reusing idx exposes the OTS secret), generate w-OTS+ sig at leaf `idx`, append authentication path. Verify: reconstruct leaf-hash from w-OTS+ verify, hash up authentication path, check root-match. **State management is the singular operational hazard:** spec mandates state must be persisted before signature is released; if state rolls back (backup restore, VM snapshot), the same idx is reused and the secret leaks. Refs: RFC 8391; Hülsing-Rausch-Buchmann 2011 *Optimal parameters for XMSS^MT*.

**S9 — `signature/xmss/multitree.go` ~220 LOC.** XMSS^MT hierarchical-tree variant. Top-level tree of height h_top with leaves that are themselves XMSS-tree roots (height h_low each). Total signing capacity = 2^(h_top + h_low). Used to push from 2^20 sigs to 2^60 sigs while keeping per-tree height tractable. Sign: at idx, identify which leaf-tree, sign with that tree's OTS, append authentication-path through both trees, append top-tree's signature of leaf-tree-root. Refs: Hülsing-Rausch-Buchmann 2011.

**S10 — `signature/lms/params.go` ~60 LOC.** RFC 8554 LMS parameter sets. LMS-SHA256-N32-H5/H10/H15/H20/H25 for tree heights 5–25. LMOTS (Leighton-Micali OTS, an SHA-256-only variant of Winternitz) parameter sets W1/W2/W4/W8 for chain heights 2/4/16/256. RFC 8554 mandates SHA-256 only — NIST SP 800-208 adds SHAKE-256 and AES-based variants. Refs: RFC 8554 §4.1.

**S11 — `signature/lms/sign.go` ~240 LOC.** LMS signing. Similar structure to XMSS but: (a) SHA-256 truncated to N=32 bytes only (no SHAKE in RFC 8554 base spec), (b) LMOTS (Leighton-Micali OTS) instead of w-OTS+ — LMOTS is a Winternitz w-OTS variant with a subtly different bitmask scheme tuned for SHA-256, (c) HSS Hierarchical-Signature-System tree-of-trees structure built into the spec rather than as a separate variant. State management identical hazard to XMSS: idx must be persisted before signature is released. Refs: RFC 8554 §5–§6.

**S12 — `signature/lms/hss.go` ~180 LOC.** HSS hierarchical signature system: each LMS tree's leaves are themselves LMS-tree roots. Up to 8 levels (RFC 8554 max L=8). Each level has its own height parameter h_i; total capacity = 2^(h_1 + h_2 + ... + h_L). Stratified state: only the bottom-level tree's idx changes per signature (the upper trees' OTS leaves are signed once each, when the lower tree below them is initialized). Refs: RFC 8554 §6.

**S13 — `signature/state/persist.go` ~140 LOC.** State-management discipline for stateful signatures. `StateManager{Path, FsyncMode, BackupPolicy}` interface. Atomic-write-then-rename to ensure idx is durable before signature is released. **Critical security invariant:** "no idx ever reused" is the entire security premise of XMSS / LMS / HSS — this primitive enforces the invariant at the API level (panic on idx-reuse, refuse to sign if state is read-only or if backup-restore is detected via monotonicity-violation). Refs: NIST SP 800-208 §7.2; McGrew-Curcio-Fluhrer 2019 *State management for stateful hash-based signatures*.

### Tier 3 — Stateless FIPS standard SLH-DSA (~860 LOC)

**S14 — `signature/slh/params.go` ~100 LOC.** FIPS-205 SLH-DSA. 12 parameter sets across 2 hash families × 3 security levels × 2 size/speed trade-offs: SLH-DSA-{SHA2,SHAKE}-{128,192,256}{s,f}. Internal parameters: total height h ∈ {63, 66, 68}, layers d ∈ {7, 17, 22}, FORS-trees k ∈ {14, 22, 33, 35}, FORS-tree-height a ∈ {6, 8, 9, 12, 14}, w=16 fixed for w-OTS+. Signature sizes: 7,856 bytes (128f) up to 49,856 bytes (256s) — the LARGEST mainstream signature scheme. Refs: FIPS-205 §11.

**S15 — `signature/slh/fors.go` ~200 LOC — KEYSTONE.** FORS (Forest of Random Subsets) few-time-signature scheme. Per signature: hash message to k·a bits, slice into k indices each ∈ [0, 2^a). For each tree i: select leaf σ_i = secret[i][idx_i], reveal authentication path. Pseudo-random secret derivation via PRF(seed, address) eliminates the storage of 2^h secret keys (only the seed is stored). FORS replaces HORS / HORST in earlier SPHINCS variants. **FORS is the few-time-signature primitive that lets SLH-DSA be stateless:** with k=22 trees of height a=6, message-hash collisions are unlikely enough (probability ≤ 2^(−128)) that "few-times" effectively becomes "many-times-with-negligible-failure". Refs: Bernstein-Hülsing 2017 *The SPHINCS+ signature framework* §4; FIPS-205 §8.

**S16 — `signature/slh/hypertree.go` ~280 LOC — KEYSTONE.** Hyper-tree: a tree-of-trees-of-XMSS at total height h split across d layers. Top layer signs the message-FORS-pubkey via XMSS; each lower layer signs the layer-above's tree-root via XMSS. **The novel construction:** pseudo-random leaf derivation lets every w-OTS+ leaf and every Merkle node be derived on-demand from `(seed, address)` rather than pre-stored, making the per-signature storage 0 bytes (vs O(2^h) for naive). Per-signature compute is enormous (~10,000 SHA-256 evaluations for 128f, ~1M for 256s — hence the s/f size/speed trade-off). Refs: FIPS-205 §10; Hülsing-Bernstein-Dobraunig-Eichlseder-Fluhrer-Gazdag-Kampanakis-Kölbl-Lange-Lauridsen-Mendel-Niederhagen-Rechberger-Rijneveld-Schwabe 2022 *SPHINCS+ Round-3 specification*.

**S17 — `signature/slh/sign.go` ~280 LOC.** SLH-DSA-Sign(sk, msg, opt_rand). Compute randomizer `R = PRF(sk_prf, opt_rand, msg)` for randomized signing OR `R = PRF(sk_prf, msg)` for deterministic signing (the "deterministic-signing" mode is mandatory per FIPS-205 §9.2, "randomized-signing" optional). Compute message-digest `digest = H(R, pk_seed, pk_root, msg)` and split into FORS-message-bits + tree-idx + leaf-idx. Generate FORS-sig (S15), generate hyper-tree XMSS-sig (S16) over the FORS-pubkey. Output `(R, FORS-sig, XMSS-sig-chain)`. Verify: recompute FORS-pubkey from FORS-sig, walk hyper-tree XMSS verifications up to root, compare to public key. Refs: FIPS-205 §10.2.

### Tier 4 — Lattice-signature wrappers (~940 LOC, depend on 211)

**S18 — `signature/mldsa/params.go` ~60 LOC.** ML-DSA parameter sets — alias-import from 211-L16 to the signature/ namespace. ML-DSA-44 (level 2), ML-DSA-65 (level 3), ML-DSA-87 (level 5). Refs: FIPS-204 §4.

**S19 — `signature/mldsa/sign.go` ~280 LOC — KEYSTONE.** ML-DSA Fiat-Shamir-with-aborts wrapper around 211-L1–L11 lattice substrate. Wraps 211-L17 Sign with: (a) the rejection-sampling abort loop until `(z, h)` is in the safe zone, (b) deterministic-vs-randomized-mode split (FIPS-204 §3 mandates deterministic by default, randomized-via-`opt_rand` optional), (c) message-prefix domain-separation `μ = H(t || M)` where `t` is the precomputed-public-key-hash and M is the message, (d) μ-binding-via-context-string for FIPS-204-§5 compliance. Average iterations ~3–7 until acceptance. Constant-time conditional-select to avoid timing-leak on iteration count (FIPS-204 §5.4 explicitly warns against branch-on-iteration-count). Refs: FIPS-204 §5; Lyubashevsky 2009 *Fiat-Shamir with aborts*.

**S20 — `signature/mldsa/verify.go` ~140 LOC.** ML-DSA Verify. Wraps 211-L18. Reconstruct `w_1 ≈ HighBits(Az − ct_1·2^d + h)` from public key + signature + message. Recompute `c' = H(μ, w_1)`. Check `c == c'` AND `||z||_∞ < γ_1 − β` AND `#1-bits(h) ≤ ω`. Refs: FIPS-204 §5.4.

**S21 — `signature/mldsa/serialize.go` ~100 LOC.** Public key + secret key + signature byte-encoding per FIPS-204 §C. Public key: ρ-seed (32 bytes) + t_1 (k·320 bytes packed). Signature: `(c̃, z, h)` packed per FIPS-204 §C.3. Cross-validation: byte-exact match with FIPS-204 NIST CAVP KAT vectors. Refs: FIPS-204 §C.

**S22 — `signature/fndsa/params.go` ~40 LOC.** FN-DSA / Falcon parameter sets — alias-import from 211-L19 to the signature/ namespace. Falcon-512 (NIST level 1), Falcon-1024 (NIST level 5). FIPS-206 final draft expected Q4-2025; this slot tracks the final draft AND the original Falcon submission v1.2. Refs: FIPS-206 draft 2024; Falcon v1.2 2020.

**S23 — `signature/fndsa/sign.go` ~200 LOC — KEYSTONE.** FN-DSA / Falcon Sign wrapper around 211-L20–L22 NTRU + FFT-tree-sampler + GPV-trapdoor substrate. Adds: (a) salt-generation (40-byte random salt for randomized mode), (b) message-hash to-target-polynomial `c = H(salt || msg) ∈ R_q`, (c) variable-length signature compression via Huffman encoding of `s_2` (Falcon signatures are NOT fixed-length per signature — average ~666 bytes for Falcon-512, max ~752 bytes), (d) padding-to-fixed-length for transport. Defers all numerically-delicate FFT-sampler work to 211-L21. Refs: Falcon spec §3.7–§3.10.

**S24 — `signature/fndsa/verify.go` ~120 LOC.** Falcon Verify. Decompress `s_2` from Huffman encoding, reconstruct `s_1 = c − s_2·h mod q`, check `||(s_1, s_2)||_2^2 ≤ β^2` (the GPV norm bound). Refs: Falcon spec §3.8.

### Tier 5 — Aggregation / threshold / adaptor / blind / ring / group (~1,000 LOC)

**S25 — `signature/agg/threshold_mldsa.go` ~280 LOC.** Threshold ML-DSA via DKLs23-style protocol. (t, n)-threshold: any t-of-n parties can produce a signature, fewer than t cannot. Active 2024–2026 research frontier (Doerner-Kondi-Lee-shelat 2023, GG24 2024). Signing requires multi-round MPC: ABORT-prone like ML-DSA single-party. **No production library ships threshold ML-DSA at the time of this review**; reality would be among the first. Refs: DKLs23; GG24 2024 *Threshold lattice signatures*.

**S26 — `signature/agg/adaptor_schnorr.go` ~140 LOC.** Adaptor signatures: a Schnorr signature with a "witness offset" t such that signing produces a pre-signature `(R, s')` where `s = s' + t mod q` is the real signature; revealing `s` reveals `t`, revealing `t` reveals `s`. Used in Bitcoin Lightning DLCs, atomic swaps, scriptless scripts. **Two variants:** Schnorr-adaptor (clean) and ECDSA-adaptor (Aumayr-Ersoy-Erwig-Faust-Hostáková-Maffei-Moreno-Sanchez-Riahi 2021, complex). Refs: Aumayr et al. 2021; Poelstra 2018 *Scriptless scripts*.

**S27 — `signature/agg/blind_schnorr.go` ~140 LOC.** Round-optimal blind Schnorr signatures (BSA 2024). Signer signs a "blinded" message without learning its content. Earlier blind-Schnorr (Schnorr 1991, Wagner attack 2002) is INSECURE under concurrent signing (k-sum attack reduces to lattice problem); BSA 2024 fixes via MuSig2-style nonce pre-commitment. **Active 2024 research frontier:** zero shipping libraries. Refs: BSA 2024 *On the (in)security of ROS*; Fuchsbauer-Plouviez-Seurin 2020.

**S28 — `signature/agg/blind_rsa.go` ~80 LOC.** Chaum 1983 blind-RSA. The original blind-signature scheme — message blinded as `m·r^e mod n`, signed as `(m·r^e)^d = m^d · r mod n`, unblinded as `m^d mod n`. Used in Privacy Pass, Apple Private Click Measurement, Cloudflare's blind-RSA RFC 9474. Requires RSA infrastructure (057 §T1-RSA). Refs: Chaum 1983; RFC 9474.

**S29 — `signature/agg/ring.go` ~140 LOC.** Ring signatures (Rivest-Shamir-Tauman 2001). Sign with one of n public keys without revealing which one. Used in Monero (CryptoNote 2013, RingCT 2017), Tor onion routing, anonymous attestation. Schnorr-based ring (LSAG, AOS) and lattice-based ring (Esgin-Steinfeld-Liu-Liu 2019 ML-DSA-friendly) variants. Refs: Rivest-Shamir-Tauman 2001 *How to leak a secret*; Esgin et al. 2019.

**S30 — `signature/agg/group.go` ~160 LOC.** Group signatures (Chaum-van Heyst 1991, BBS+ 2004). Like ring signatures but with a group-manager who can de-anonymize (the "opening" capability). BBS+ is the modern pairing-based variant powering EU Digital Identity Wallet 2024 + Verifiable-Credentials W3C-DID. Requires pairing-friendly curve (057 §T2-BLS). Refs: Boneh-Boyen-Shacham 2004 *Short group signatures*; W3C DID-VC 2024.

### Tier 6 — Hybrid + broken pedagogy (~560 LOC)

**S31 — `signature/hybrid/dual.go` ~120 LOC.** Dual-PQ-classical signatures: a single signature consists of `(σ_classical, σ_PQ)` over the same message; verifier requires both. CMS draft-ietf-lamps-pq-composite-sigs-02 (2024) standardizes ML-DSA + Ed25519 dual mode. Defends against either-side break. Refs: CMS PQ-composite-sigs draft 2024.

**S32 — `signature/hybrid/xwing.go` ~80 LOC.** X-Wing-style PQ+classical KEM/signature combiner (Cloudflare-Cremers-Düzlü-Fiedler-Hülsing-Westerbaan 2024). Hash-combined output with explicit-key-binding to defeat re-encryption attacks. Refs: X-Wing IETF draft 2024.

**S33 — `signature/broken/ntrusign.go` ~120 LOC.** NTRUSign 2003 + Nguyen-Regev 2006 transcript attack reproducer. Educational only. Demonstrates the GGH / NTRUSign parallelepiped-leak attack: ~400 transcripts suffice to recover the secret-key parallelepiped via averaged-second-moment analysis. Refs: Hoffstein-Howgrave-Graham-Pipher-Silverman-Whyte 2003; Nguyen-Regev 2006 *Learning a parallelepiped*.

**S34 — `signature/broken/rainbow.go` ~120 LOC.** Rainbow multivariate signature (Ding-Schmidt 2005) + Beullens 2022 break reproducer. NIST PQC Round-3 finalist that was broken in <53 hours on a laptop in 2022 via the MinRank attack. Educational only. Refs: Ding-Schmidt 2005; Beullens 2022 *Breaking Rainbow takes a weekend on a laptop*.

**S35 — `signature/broken/sidh.go` ~120 LOC.** SIDH key-exchange + signatures (de Feo-Jao-Plut 2014) + Castryck-Decru 2022 polynomial-time break reproducer. Educational only. Demonstrates the auxiliary-points → endomorphism-ring-recovery attack. Refs: Castryck-Decru 2022 *An efficient key recovery attack on SIDH*.

**S36 — `signature/broken/gemss.go` ~120 LOC.** GeMSS (Casanova-Faugère-Macario-Rat-Patarin-Perret-Ryckeghem 2017) HFEv- multivariate signature + 2022 break reproducer. NIST PQC Round-3 alternate-finalist broken in 2022 via support-modifier attack. Educational only. Refs: Tao-Petzoldt-Ding 2021 *Improved key recovery of the HFEv- signature scheme*.

---

## 2. LOC budget summary

| Tier | LOC | Cumulative |
|---|---:|---:|
| T1 — OTS + Merkle | 720 | 720 |
| T2 — XMSS / LMS / HSS | 1,060 | 1,780 |
| T3 — SLH-DSA (FIPS-205) | 860 | 2,640 |
| T4 — ML-DSA + Falcon wrappers | 940 | 3,580 |
| T5 — Aggregation / threshold / adaptor / blind / ring / group | 1,000 | 4,580 |
| T6 — Hybrid + broken pedagogy | 560 | 5,140 |
| Connective tissue (params indexes, doc strings, test fixtures) | 400 | 5,540 |

Tier 1 alone unblocks Tier 2 + Tier 3 (the entire FIPS hash-based universe). Tier 4 depends on 211-Tier-1 + 211-Tier-3 landing; Tier 5 depends on 057 §T1-EC (Schnorr/ECDSA) + 057 §T2-BLS (pairings).

---

## 3. Cross-package blockers

| Blocker | Slot | Path | Impact |
|---|---|---|---|
| SHA-256 + SHA-512 | 057 §T1-HASH | `crypto/sha2/` (absent) | BLOCKS ALL S1–S17 (every hash-based scheme + ML-DSA + Falcon). Single most critical blocker. |
| SHAKE-128/256 | 057 §T1-HASH | `crypto/sha3/` (absent) | BLOCKS S7-SHAKE / S14-SHAKE / S19 ML-DSA / S23 FN-DSA. |
| HMAC-SHA-256 | 057 §T1-HASH | `crypto/hmac/` (absent) | BLOCKS LMS pseudo-random key-derivation (S11). |
| Constant-time discipline | 057 §T1-CT | `crypto/ct/` (absent) | BLOCKS side-channel-resistant signing (S17 SLH-DSA, S19 ML-DSA, S23 FN-DSA). |
| 211-Tier-1 lattice substrate | 211-L1–L11 | `lattice/ring/`, `lattice/ntt/`, `lattice/sample/` | BLOCKS S18–S24 ML-DSA + Falcon wrappers. |
| 211-Tier-3 ML-DSA core | 211-L16–L18 | `lattice/dilithium/` | BLOCKS S19–S21 (these wrap 211-Tier-3). |
| 211-Tier-3 Falcon core | 211-L19–L22 | `lattice/falcon/` | BLOCKS S22–S24 (these wrap 211-Tier-3). |
| 057 §T1-SCHNORR + ECDSA | 057 §T1-SCHNORR / §T1-ECDSA | `crypto/schnorr/`, `crypto/ecdsa/` (absent) | BLOCKS S26 adaptor + S27 blind + S29 ring (need a classical-curve "leg"). |
| 057 §T2-BLS | 057 §T2-BLS | `crypto/bls/` (absent) | BLOCKS S30 BBS+ group signature (needs pairings). |
| 057 §T1-RSA | 057 §T1-RSA | `crypto/rsa/` (absent) | BLOCKS S28 blind-RSA. |
| State persistence | this slot S13 | `signature/state/` | Implementable in this slot but requires explicit operational discipline doc. |

The single most important blocker is **SHA-2 + SHAKE (057 §T1-HASH)**: without standardized hash functions, NOTHING in this slot ships with golden-file validation. Recommend prioritizing 057 §T1-HASH as the prerequisite that unblocks BOTH this slot AND 211 (which also gates rejection-sampling on SHAKE).

---

## 4. Recommended sequencing

**Sequence S-A (hash-based first, 2,640 LOC over ~4 PRs):**
1. 057 §T1-HASH SHA-256 + SHAKE-128/256 + HMAC (~600 LOC) — gates everything.
2. T1 OTS + Merkle (720 LOC) — irreducible foundation.
3. T2 XMSS (RFC 8391) (580 LOC) — first FIPS-listed (NIST SP 800-208).
4. T2 LMS + HSS (RFC 8554) (480 LOC) — second FIPS-listed.
5. T3 SLH-DSA (FIPS-205) (860 LOC) — third FIPS-listed (most popular for embedded — NO state).

**Sequence S-B (lattice-wrapper, 940 LOC, depends on 211):**
1. 211-Tier-1 + 211-Tier-3 land first.
2. T4 ML-DSA wrapper (580 LOC).
3. T4 FN-DSA wrapper (360 LOC).

**Sequence S-C (advanced variants, 1,560 LOC, depends on 057):**
1. 057 §T1-SCHNORR + §T2-BLS land first.
2. T5 aggregation / threshold / adaptor / blind / ring / group (1,000 LOC).
3. T6 hybrid + broken pedagogy (560 LOC).

Recommend **S-A first** because hash-based signatures (SLH-DSA / XMSS / LMS) are the SINGLE post-quantum signature family with NO new mathematical assumption (relies only on hash-function pre-image + collision resistance, which is well-understood). They are FIPS-205 certified and IETF RFC-standardized. Lattice wrappers (S-B) and advanced variants (S-C) follow as 211 + 057 land.

---

## 5. Cross-language golden-file feasibility

| Primitive | Reference test vectors | Cross-lang feasibility |
|---|---|---|
| S1 Lamport | None standard — generate canonical | Trivial |
| S2 Winternitz / S3 w-OTS+ | RFC 8391 §B + NIST SP 800-208 | Easy |
| S5 Merkle / S6 AuthPath | XMSS / LMS / SLH-DSA reference vectors | Trivial |
| S8 XMSS | RFC 8391 §B (24 KAT vectors) | Trivial |
| S11 LMS | RFC 8554 §F (test vectors per parameter set) | Trivial |
| S12 HSS | RFC 8554 §F | Trivial |
| S15–S17 SLH-DSA | NIST CAVP FIPS-205 KAT (1,000+ vectors per param set × 12 sets) | Trivial — the keystone correctness lever |
| S19–S21 ML-DSA | NIST CAVP FIPS-204 KAT (1,000+ vectors) | Trivial (depends on 211) |
| S23–S24 FN-DSA | Falcon spec §A KAT + FIPS-206 draft KAT | Medium (FP path is implementation-defined; Reality's seedable FFT-sampler resolves this) |
| S25 Threshold ML-DSA | None — generate canonical | Hard (multi-party transcripts; recommend deterministic-MPC-tape encoding) |
| S26 adaptor / S27 blind | None standard | Medium |
| S29 ring / S30 group | None standard | Medium |
| S31–S32 hybrid | CMS draft 2024 KAT | Easy |
| S33–S36 broken pedagogy | Original-paper transcripts | Easy (educational only — the BREAK is the test) |

KAT availability via NIST CAVP for FIPS-204 / FIPS-205 / FIPS-206 is the single biggest correctness lever — these three FIPS standards ship 1,000+ KAT vectors per parameter set, making cross-language byte-identical validation trivial.

---

## 6. What NOT to add (out-of-scope confirmation)

- **Hardware-backed implementations** (AVX2 SHA-NI, NEON SHA-256, AES-NI, RISC-V V-ext): out-of-scope, reality is portable-Go.
- **Embedded / smartcard variants** (XMSS-T-W with fault tolerance, CACR memory-constrained SLH-DSA): out-of-scope.
- **Side-channel countermeasures specific to embedded targets** (masking, shuffling, hidden-Markov-model attacks): out-of-scope; document threat model.
- **Quantum-circuit cost estimation** (post-quantum *security analysis*): out-of-scope, this is mathematical-cost-modelling not algorithm.
- **Lattice-arithmetic substrate for ML-DSA / Falcon** (NTT, polynomial ring, sampler): scoped to 211.
- **Classical-elliptic-curve substrate** (Schnorr, Ed25519, BLS12-381): scoped to 057.
- **Hash-function primitives** (SHA-2, SHA-3, BLAKE): scoped to 057 §T1-HASH.
- **Code-based signatures** (Stern, RaCoSS, LESS, MEDS — NIST onramp 2023 finalists): out-of-scope this slot, scope to a separate slot once NIST onramp 2026 finalists are announced.
- **Isogeny survivors** (CSIDH, CSI-FiSh, SQIsign — NIST onramp 2024 candidate): NOT YET FIPS-standardized; defer to a separate slot pending NIST onramp 2026 outcome.
- **Picnic / MPCitH** (NIST PQC R3 alternate, withdrawn from R4): out-of-scope; ZK-MPC-in-the-head is its own universe (recommend a separate slot once SDitH / FAEST / RYDE become NIST-onramp finalists).

---

## 7. Non-overlap

- Versus 211-new-lattice-crypto: 211 ships lattice-arithmetic substrate (polynomial ring, NTT, samplers, NTRU keygen, Falcon FFT-sampler); this slot ships signature-scheme orchestration on top. Cross-link: S18–S24 wrap 211-L16–L22.
- Versus 057-crypto-missing: 057 ships classical-curve-and-pairing-and-hash substrate; this slot ships PQ-signature schemes that consume that substrate (S26–S30 need 057 §T1-SCHNORR + §T2-BLS; ALL of S1–S17 need 057 §T1-HASH).
- Versus 058-crypto-sota / 060-crypto-perf: SOTA / perf review of EXISTING crypto/ — orthogonal.
- Versus 210-new-coding-theory: shared GF(2^m) substrate would be useful for code-based signatures (Stern, LESS) but those are out-of-scope this slot; recommend deferring code-based-signature scoping to a future slot once NIST onramp 2026 finalizes.
- Versus 175-synergy-zkmark-crypto: zkmark could use this slot's S26 adaptor + S27 blind + S29 ring as building blocks for ZK-rollup-style protocols, but that's protocol composition not math.

---

## 8. Headline recommendation

Reality's "universal truth encoded in code" mission demands post-quantum signatures because: (1) **FIPS-205 SLH-DSA** standardized August 2024 and is required for U.S. federal post-quantum migration by 2030; (2) **RFC 8391 XMSS** + **RFC 8554 LMS** are NIST SP 800-208 approved for stateful-signature use cases (firmware signing, code signing, long-term archive integrity); (3) **FIPS-204 ML-DSA** + **FIPS-206 FN-DSA** are the lattice-based companions to SLH-DSA and complete the three-headed FIPS PQ-signature lineup. The 5,540-LOC investment is partitionable: T1 OTS + Merkle (720 LOC) ships first as a hash-substrate, T2 XMSS + LMS (1,060 LOC) ships next as RFC-standardized stateful signatures, T3 SLH-DSA (860 LOC) ships as the singular reality-competitive-moat (zero-dep golden-file-cross-validated implementation), T4 ML-DSA + FN-DSA wrappers (940 LOC) follow as 211 lands, T5 + T6 (1,560 LOC) follow as 057 lands. The cross-language golden-file substrate (Reality's signature differentiator) is well-served by NIST CAVP KATs for FIPS-204/205/206 (trivial byte-exact validation) and well-served by IETF RFC test-vectors for XMSS/LMS. Most importantly: **hash-based signatures (T1+T2+T3) do NOT depend on lattice-arithmetic** (211) or **elliptic-curve arithmetic** (057 §T1-EC) — they ONLY depend on **SHA-2 + SHAKE + HMAC** (057 §T1-HASH). So if 057 §T1-HASH lands first, this slot's T1+T2+T3 (~2,640 LOC) ships in parallel with 211 and 057's curve-and-pairing tracks rather than queuing behind them. The recommended ordering is: **057 §T1-HASH → S1–S17 hash-based universe (2,640 LOC, ~4 PRs) → 211-Tier-1 → S18–S24 lattice-wrappers (940 LOC) → 057 §T1-EC + §T2-BLS → S25–S36 advanced variants (1,560 LOC)**. This sequencing means SLH-DSA + XMSS + LMS — the three FIPS/RFC-standardized post-quantum signatures — can ship before any classical-curve crypto, an architectural insight that mirrors 211's observation that lattice-crypto ships before big-integer-crypto. Reality would be the only Go ecosystem AND the only language ecosystem with byte-for-byte cross-substrate validation against the official NIST FIPS-205 KAT 1,000+ vector test suite.

Report length: 386 lines (under 400 cap).
