# 326 — dive-merkle-trees (Binary / Sparse / MMR / Patricia / Verkle / KZG / IPA / FRI audit)

## Headline
Reality has zero Merkle/vector-commitment surface; cheapest day-1 PR is T0 generic interface + T1 binary Merkle + T2 proof verify (~310 LOC) atop slots 322 (SHA-256) / 325 (Poseidon), unblocking content addressing, certificate transparency, blockchain rollups, and the entire ZK commitment stack (KZG/IPA/FRI all gate on slot 292 EC + slot 293 NTT + slot 324 MSM).

## Findings

### F1 — Zero Merkle / commitment surface confirmed
Grep `Merkle|MerkleTree|SparseMerkle|MerkleProof|VerkleTree|MerkleMountainRange|MMR|KZG|PolynomialCommitment|VectorCommitment` across all `*.go` in `C:/limitless/foundation/reality/` returns **zero matches** (matches exist only in `reviews/overnight-400/` agent files). No package owns commitments. CLAUDE.md package list (22) confirms: no `merkle`, no `commit`, no `vc`. This is a clean greenfield.

### F2 — Adjacent foundations are present (or pinned by sibling slots)
- `crypto/` — has hash primitives surface (slot 322 SHA-256/Poly1305 audit suggests SHA-2 family lives here). Merkle binary tree consumes a `Hash([]byte) []byte` interface and is hash-agnostic.
- Slot 325 (Poseidon, ZK-friendly hash) — gives Merkle trees the ZK-circuit-cheap variant Ethereum/zkSync use; SHA-256 is fine for classic content-addressing.
- Slot 292 (elliptic curves BLS12-381 / BN254), slot 293 (NTT), slot 324 (MSM Pippenger), slot 214 (pairings), slot 290 (Galois/finite-field) — the entire prerequisite stack for **KZG, IPA, Verkle, FRI**. KZG needs G1/G2 + pairing for `e(C-y·g, g) = e(π, x·g - τ·g)`. IPA needs only G1 + scalar mul + MSM (no pairing, no trusted setup). FRI needs only NTT + Reed-Solomon (slot 320 error-correction) + Merkle commitments — **uses Merkle internally**, so T1 binary Merkle gates FRI.
- Slot 320 (Reed-Solomon) — FRI low-degree-test prerequisite.

### F3 — Eight primitives, tier-able
| Tier | Primitive | LOC | Gates on |
|------|-----------|-----|----------|
| T0 | `merkle.Hasher` interface (`HashLeaf`, `HashNode`, domain-separation prefixes) | ~80 | nothing |
| T1 | Binary Merkle tree (RFC 6962-style: `0x00 \|\| leaf`, `0x01 \|\| left \|\| right`) | ~150 | T0 |
| T2 | Inclusion proof + verify (audit path, `O(log n)` siblings + bit-direction) | ~80 | T1 |
| T3 | Sparse Merkle Tree (SMT) with default-node optimization (height = hash bits, 256 for SHA-256) | ~200 | T0, T2 |
| T4 | Merkle Mountain Range (MMR) — append-only, peak-bagging root | ~150 | T0 |
| T5 | Patricia-Merkle Trie (Ethereum hex-prefix encoding, 16-ary radix) | ~300 | T0, RLP encoding (likely punted) |
| T6 | KZG10 polynomial commitment (`C = [p(τ)]₁`, opening `π = [(p(x)-y)/(x-z)]₁`) | ~250 | slots 292, 214 |
| T7 | IPA / Bulletproofs-style inner-product argument (Halo, no trusted setup) | ~300 | slots 292, 324 |
| T8 | FRI (Fast Reed-Solomon IOP, low-degree test) | ~400 | slots 293, 320, T1 |
| T9 | Verkle tree (256-ary radix + KZG sub-commitments at each node) | ~300 | T6 |

### F4 — Day-1 PR (cheapest, highest-leverage)
**T0 + T1 + T2 = ~310 LOC.** Composes any `Hasher` (slot 322 SHA-256 today, slot 325 Poseidon when landed). Unblocks:
- Content-addressing (Pistachio asset checksum trees, IPFS-compat)
- Certificate transparency log validators (RFC 6962)
- Bitcoin SPV / payment channel proofs
- ZK Merkle inclusion (when Poseidon arrives, the same Merkle code with a Poseidon `Hasher` becomes a ZK-friendly Sparse Merkle for state commitments — Tornado Cash, zkSync, Mina pattern)
- FRI (T8) — FRI's Merkle commits to the codeword

Approximate signature:
```go
type Hasher interface {
    Size() int
    HashLeaf(leaf []byte) []byte                  // RFC6962: H(0x00 || leaf)
    HashNode(left, right []byte) []byte           // RFC6962: H(0x01 || L || R)
}
type Tree struct { ... }
func New(h Hasher, leaves [][]byte) *Tree
func (t *Tree) Root() []byte
func (t *Tree) Prove(index int) Proof
func (t *Tree) ProveRange(lo, hi int) RangeProof   // RFC 6962 §2.1.2

type Proof struct{ Index, Size int; Siblings [][]byte }
func Verify(h Hasher, root, leaf []byte, p Proof) bool
```

### F5 — Implementation pitfalls (variants vary on these — must pin in golden tests)
1. **Leaf domain separation.** Bitcoin: no prefix (vulnerable to second-preimage when leaf coincidentally equals an internal hash). RFC 6962 (CT log): `0x00` leaf, `0x01` node. Ethereum: tag-based or RLP. **Pick RFC 6962 as canonical** — it's the only one immune to interior-collision attacks. Document explicitly.
2. **Odd-leaf handling.** Bitcoin duplicates the last leaf (CVE-2012-2459 reorg attack). RFC 6962 promotes the unpaired node up the level (no duplication). **Pick RFC 6962.**
3. **Sorted siblings (sorted-pair Merkle, OpenZeppelin style).** Pair (a,b) hashed as H(min(a,b) ‖ max(a,b)). Eliminates direction bits in proof — proof is just sibling list. Strictly weaker (loses index-binding) but used by Solidity verifiers. Offer as a separate `SortedHasher` mode behind a feature flag, **default off**.
4. **Sparse Merkle default subtree.** The SMT optimization: precompute `D_0 = H(0)`, `D_{i+1} = H(D_i ‖ D_i)`. An empty subtree at level `i` always hashes to `D_i`, so a proof through empty space is `O(log n)` `D_i` siblings. Without this, an SMT is unusable (2^256 nodes).
5. **Endianness in tree position bits.** MSB-first (Ethereum) vs LSB-first (Bitcoin) for index→path direction. Pick one (MSB-first is standard) and pin in golden tests.
6. **MMR peak-bagging order.** Right-to-left fold (Todd's spec): `root = H(size, bag(peaks))`. Get this wrong and the appendable-log property breaks.

### F6 — R-MUTUAL-CROSS-VALIDATION 3/3 pin opportunities
1. **Binary Merkle root ≡ recursive hash chain.** For `n=1024` random 32-byte leaves, root computed by `Tree.Root()` ≡ root computed by naive `O(n log n)` recursive function `merkle(leaves)` defined inline ≡ root computed by an iterative bottom-up pairing loop. Three independent computational paths → same digest.
2. **Inclusion ⇔ proof verifies.** For 100 random trees of size 1..256, every leaf's proof verifies (`Verify(root, leaf, p) == true`); for 100 attempted forgeries (leaf swapped, sibling flipped, index off-by-one), `Verify == false`. Bidirectional saturates the property.
3. **Sparse Merkle empty-tree root = known constant.** For SHA-256 SMT, height 256, `D_256` is a fixed 32-byte value. Compute three ways: (a) iterative `D_{i+1} = SHA256(D_i ‖ D_i)` 256 times; (b) recursive `emptyRoot(256)`; (c) build SMT with zero insertions and read `Root()`. All three ≡. Pin against external (e.g., Plasma, Diem) reference implementation if compatible domain separation is chosen.
4. **(Bonus) RFC 6962 CT log compatibility.** Pin against the published Google CT test vectors — gives a fourth witness, third-party.

### F7 — Cross-link map (consumers)
- **Pistachio (asset pipeline)** — content-addressed asset chunks; T1 binary Merkle gives `O(log n)` integrity proofs for partial loads.
- **zkmark / ZK proving (slots 146–150, 175, 200)** — Plonk-class proofs use **KZG (T6)** for polynomial commitments; STARKs use **FRI (T8)**; Halo2/Nova use **IPA (T7)**. All three commitment schemes are gated by Merkle and curve infra. Slot 200 (zkmark-info synergy) likely flagged this.
- **Ethereum compat / rollups** — Patricia-Merkle (T5) for state, MMR (T4) for log accumulators (Mina), Verkle (T9) for the planned Verge upgrade. Reality doesn't *need* L1 compat but exposing Patricia is the path of least friction for any rollup tooling that consumes `reality`.
- **conduit / forge** — if either does any kind of distributed log or replicated state, MMR (T4) is the right primitive (append-only, O(log n) proofs, no rebalancing).
- **Certificate transparency / supply-chain** — RFC 6962 binary Merkle (T1+T2) is *the* building block.

### F8 — Slot ordering & dependency DAG
- **T1 (binary Merkle)** is the Stage-1 gate. Lands today on slot 322 SHA-256, costs ~310 LOC, no curve math, no field math.
- **T3 (SMT)** and **T4 (MMR)** are Stage-2, each ~200 LOC, parallel-implementable after T1.
- **T6 (KZG)** is Stage-3, gated on slots 292 (BLS12-381), 214 (pairings), 324 (MSM). Entire ZK stack benefits.
- **T7 (IPA)** is Stage-3, gated on 292 + 324 (no pairings, no trusted setup → easier to ship than T6 in some respects, but proofs are O(log n) instead of O(1)).
- **T8 (FRI)** is Stage-3, gated on 293 (NTT) + 320 (Reed-Solomon) + T1 (Merkle for codeword commits). Frontier.
- **T9 (Verkle)** is Stage-4, composes T6 KZG inside a 256-ary trie. ~300 LOC on top of T6.
- **T5 (Patricia)** is the awkward one — needs RLP/hex-prefix encoding which lives at the Ethereum protocol layer, not "universal truth". Recommend **defer indefinitely or punt to a separate `realityeth` repo** (it's compat code, not pure math).

### F9 — Precision / numerics notes (per CLAUDE.md key design rules)
Merkle trees are bit-exact — no floating point, no tolerance. Golden file tolerance = **0**. Test vectors are `(leaves, root)` and `(leaves, index, proof)` byte-for-byte equality. KZG/IPA opening verification is also exact (group equality on EC points) once curve arithmetic is exact. FRI similarly bit-exact over a finite field. This makes the entire commitment family ideal for golden-file cross-language validation (Go canonical, Python/C++/C# verify) — exactly the regime CLAUDE.md `testutil` was designed for.

### F10 — Allocation discipline
Hot-path requirement (CLAUDE.md rule 3, "Pistachio calls these at 60 FPS"): expose `RootInto(buf []byte)`, `ProveInto(buf *Proof, index int)`, and a level-wise scratch buffer `Tree.Scratch []byte` to amortize across rebuilds. Naive `make([]byte, 32)` per node will allocate `2n-1` times per `Build`; use one slab `[]byte` of size `(2n-1)*hashSize` and slice into it.

## Concrete recommendations

1. **Day-1 PR: `merkle/` package, T0 + T1 + T2.** ~310 LOC. Hasher interface, RFC 6962 binary tree, inclusion proof, range proof (RFC 6962 §2.1.2). Pin Google CT test vectors as golden. Land before Poseidon (slot 325) so Poseidon ships with a `merkle.Hasher` implementation slot ready.
2. **Default to RFC 6962 domain separation** (`0x00` leaf prefix, `0x01` node prefix, last-unpaired-promoted). Document Bitcoin's odd-leaf-duplication and second-preimage gap as anti-patterns. Offer Bitcoin-mode as `BitcoinHasher` for SPV interop, **clearly tagged** "compat, not recommended for new code".
3. **R-MUTUAL-CROSS-VALIDATION 3/3 saturate on T1 day one.** Three independent root computations + bidirectional inclusion property + Google CT external vectors = pin from the first commit.
4. **T3 SparseMerkle next** (~200 LOC). The default-subtree optimization is the only subtle bit; pin `D_0..D_256` table as a golden constant. Targets ZK state-commitment use (Tornado, zkSync, Mina pattern) once Poseidon lands.
5. **T4 MMR next-after** (~150 LOC, Todd 2017 spec). Append-only, peak-bagging. Targets logs / accumulators / fraud-proof bridges. Independent of T3 — implement in parallel.
6. **T6 KZG, T7 IPA, T8 FRI as separate package `commit/`** (or `merkle/poly/`). Each ~250-400 LOC. Gate T6/T7 on slot 292 BLS12-381 landing. T8 gates on slot 293 NTT + slot 320 RS + T1. These are the ZK trinity — Plonk uses KZG, STARKs use FRI, Halo uses IPA. Reality should ship all three so zkmark and downstream provers can swap commitment scheme without leaving the foundation.
7. **T9 Verkle** after T6 KZG. Composes 256-ary radix trie with KZG sub-commitments. Document explicitly that this is the Ethereum Verge primitive; pin against EIP-6800 test vectors when stable.
8. **Defer T5 Patricia-Merkle indefinitely.** It's Ethereum protocol compat (RLP, hex-prefix nibbles, extension/branch/leaf node distinction), not universal math. Belongs in a downstream `realityeth` package or an external consumer; would violate CLAUDE.md "zero dependencies / pure math" intent if implemented inside reality.
9. **Hot-path allocation discipline:** single-slab `[]byte` for the entire tree, `RootInto/ProveInto` variants, no per-node `make`. Pistachio requirement.
10. **Document precision = 0 (bit-exact).** Per-function tolerance in golden files = 0 across the entire commitment family. No transcendental math, no IEEE 754 edge cases (these primitives don't touch floats).

## Sources

- Repo state: `C:/limitless/foundation/reality/` — grep for Merkle/KZG/IPA/FRI/MMR/PolynomialCommitment/VectorCommitment in `*.go` returns zero matches (only `reviews/overnight-400/` references).
- `C:/limitless/foundation/reality/CLAUDE.md` — package list (22, no commitments), key design rules 1–6.
- `C:/limitless/foundation/reality/reviews/overnight-400/MASTER_PLAN.md:326` — slot definition.
- Sibling slot reviews already on disk: `325-dive-poseidon.md` (ZK-friendly hash, will be Merkle's preferred `Hasher` for ZK use), `324-dive-msm.md` (Pippenger MSM gates KZG/IPA), `293-new-ntt.md` (FRI prereq), `292-new-elliptic-curves.md` (KZG/IPA/Verkle prereq), `214-new-pairings.md` (KZG verify), `320-dive-error-correction.md` (FRI Reed-Solomon prereq), `200-synergy-zkmark-info.md` and `175-synergy-zkmark-crypto.md` (consumer demand).
- Merkle, R. (1979). *Secrecy, Authentication, and Public Key Systems*. PhD thesis, Stanford. Original binary Merkle tree.
- Laurie, B. & Kasper, E. (2014). *Revocation Transparency*. Google. Sparse Merkle tree with default-subtree precomputation. RFC 6962 (Certificate Transparency).
- Todd, P. (2017). *Merkle Mountain Ranges*. python-merkle-mountain-range / OpenTimestamps spec.
- Kuszmaul, J. (2018). *Verkle Trees*. MIT. Vector-commitment radix trie.
- Buterin, V. (2021). *Verkle trees in Ethereum*. notes.ethereum.org. EIP-6800 (Verkle state migration).
- Kate, A.; Zaverucha, G.; Goldberg, I. (2010). *Constant-Size Commitments to Polynomials and Their Applications*. ASIACRYPT 2010. KZG.
- Bowe, S.; Grigg, J.; Hopwood, D. (2020). *Halo: Recursive Proof Composition without a Trusted Setup*. ePrint 2019/1021. IPA / inner-product argument.
- Ben-Sasson, E.; Bentov, I.; Horesh, Y.; Riabzev, M. (2018). *Fast Reed-Solomon Interactive Oracle Proofs of Proximity*. ICALP / ePrint 2017/134. FRI.
- Ethereum Yellow Paper §4.1 — Modified Merkle Patricia Trie (hex-prefix encoding, 16-ary radix).
- Bitcoin Core CVE-2012-2459 — odd-leaf duplication attack on Bitcoin's Merkle root.
- RFC 6962 §2.1 — domain separation `0x00` leaf, `0x01` node; §2.1.2 consistency proofs and audit paths.
- Tendermint IAVL+ tree — balanced + versioned Merkle, alternative to Patricia for app state.
- OpenZeppelin `MerkleProof.sol` — sorted-pair convention common in Solidity verifiers.
