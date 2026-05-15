# 147 — zkmark-missing

**Topic:** zkmark: missing — KZG, FRI, IPA, PlonK, Halo2, STARK polynomials, Reed-Solomon, multilinear MLE, lookup arguments (logUp, cq, plookup).

**Premise from 146:** the `zkmark/` package is a 262-line Tranche-1 *interface stub*: a `Prover`/`Verifier`/`Proof` envelope wrapping a caller-injected mirror-mark `SignerFunc`. There is **no math/big import**, **no field element**, **no transcript**, **no commitment scheme**, **no polynomial type**, **no curve**, **no hash function**, and the only sentinel is `ErrNotYetWired` returned by `Halo2Prover.Prove`. The README says "Tranche 2 (Halo2) lands in S62-S63 conditional on Marc dry-run encouragement" — i.e. every primitive in this audit's brief lives in the *future* `Halo2Prover.Prove` body and the (unwritten) Rust sidecar it will FFI to. This audit enumerates what *would* need to land in `reality` (not the sidecar) for an in-Go ZK substrate, scoped against the same constraints as 057 (crypto-missing): pure math, zero deps beyond `math/big`, golden-file validated, no network/OS-entropy.

---

## Headline

`zkmark/` shares `crypto/`'s pre-foundation problem: every primitive listed in the brief — KZG, FRI, IPA, PlonK, Halo2, STARK, Reed-Solomon, MLE, sumcheck, logUp/cq/plookup — depends on a stack of **five missing layers** that 057 already enumerated (big-integer, prime-field, field-tower, elliptic-curve, hash) plus **four ZK-specific layers** that 057 doesn't cover (NTT/iNTT, MSM/Pippenger, polynomial-arithmetic, transcript). Of the 11 topic-listed primitives, **zero** are reachable today; **all 11** require the lower-five layers from 057, and **9 of 11** additionally require all four ZK-specific layers. The cheapest reachable entry-point is **Reed-Solomon encoding** over a small prime field (≈80 LOC after T1-FIELD lands from 057), the next is the **sumcheck protocol** over a multilinear polynomial (≈200 LOC, pure-univariate, no curve, no hash beyond a stub Fiat-Shamir), and a Tier-1 anchor that exercises the full forward stack is **FRI low-degree-test** over a Goldilocks-style 64-bit prime field (≈600 LOC, requires NTT, Merkle, transcript, but no curve and no pairing). Every curve-bound primitive (KZG, IPA, PlonK, Halo2, Groth16, BLS-aggregation lookups) is gated on 057's T1-EC + T1-FIELDTOWER landing first. Per `reality`'s "no allocations in hot paths" rule (CLAUDE.md §3) every primitive below must ship a `*Into` companion accepting a caller-supplied scratch buffer, and per the "golden files are the proof" rule (§1) every primitive must ship Go-canonical JSON test vectors at ≥20 cases including IEEE-754-equivalent field-arithmetic edges (zero, one, p-1, q-th-root-of-unity, inversion of generator). The `zkmark/` package itself, per the substrate-only README §"Substrate-only ship", should **not** absorb these primitives — they belong in a new sibling `zkmark/poly/`, `zkmark/field/`, `zkmark/commit/`, `zkmark/proof/` set, exactly mirroring `crypto/`'s subdirectory plan once that lands.

---

## Scope filter — what counts as "in-scope" for `reality`

Inheriting 057's filter and tightening for ZK:

- IN: pure math primitives — finite-field arithmetic (already 057), polynomial arithmetic (univariate + multilinear), NTT/iNTT, MSM/Pippenger, Reed-Solomon coding, Lagrange interpolation, low-degree-test (FRI, DEEP-FRI), polynomial commitment open/verify (KZG, IPA, FRI-based), sumcheck protocol, multilinear extension (MLE).
- IN: lookup-argument math — logarithmic derivatives (logUp), grand-product polynomials (plookup), polynomial-multiplication-based lookups (cq) — all are polynomial identities reducible to sumcheck or KZG/FRI.
- IN: algebraic hashes — Poseidon, Rescue-Prime, Anemoi, Reinforced-Concrete, Griffin, Tip5 — round functions are pure math (S-box + MDS matrix + round constants).
- IN: Fiat-Shamir transcript algebra — domain-separation tags, label ordering, challenge derivation are determined by hash + encoding, but the *protocol* of "what to absorb in what order" is math (a deterministic function from public inputs to a transcript byte sequence).
- IN: SNARK-friendly curve arithmetic — BLS12-381, BN254, Pallas/Vesta (Pasta), Bandersnatch, Jubjub — same status as 057's T1-EC.
- IN: small-field arithmetic — Goldilocks (p = 2^64 − 2^32 + 1), Mersenne-31 (p = 2^31 − 1), BabyBear (p = 2^31 − 2^27 + 1) — these are 64-bit / 32-bit primes with FFT-friendly two-adicity; pure arithmetic in `uint64`/`uint32`.
- BORDERLINE: R1CS / Plonkish / AIR constraint-system *types* — these are data structures, not algorithms; arguably belong in a `zkmark/cs/` subpackage but are themselves zero-math (pure indexing).
- OUT: anything requiring `crypto/rand` / OS entropy. ZK proofs require *prover-side* randomness for hiding/zero-knowledge; reality is deterministic, so any ZK substrate here is **non-zero-knowledge** by construction (the soundness still holds, but the *zero-knowledge* property is sacrificed). This is a real semantic mismatch with the package name and must be flagged in zkmark.go's docstring.
- OUT: trusted-setup ceremony / SRS generation — this is one-shot and external; the SRS *consumption* (load + use) is in scope, the *generation* is not.
- OUT: serialization wire formats (compressed/uncompressed point encoding, EIP-2537 byte layout) at protocol-compatibility level — `reality` ships canonical JSON for cross-language tests; the EVM precompile byte format is a separate engineering concern.
- OUT: recursion bridges to L1 contracts (Solidity verifiers, Cairo verifiers) — engineering, not math.

---

## Missing primitives — Tier 1 (foundation, must ship before *any* ZK primitive)

These inherit 057's T1-* and add four ZK-only foundation items that 057 doesn't cover.

### T1-BIGINT, T1-FIELD, T1-FIELDTOWER, T1-EC, T1-HASH, T1-CT — see 057
057 already enumerated big-integer, prime-field, field-tower, elliptic-curve, hash, and constant-time discipline. **ZK reuses every line of that stack.** Specific ZK additions on top:
- T1-FIELD must add **two-adicity metadata** (the largest power of 2 dividing p−1) and **2^k-th root-of-unity tables** — required for NTT.
- T1-FIELD must add **Goldilocks** (p = 2^64 − 2^32 + 1, 32-bit two-adicity, NTT-friendly), **Mersenne-31** (p = 2^31 − 1, with circle-NTT for two-adicity = 1), and **BabyBear** (p = 2^31 − 2^27 + 1, 27-bit two-adicity) as small-prime fields. These are **not** pairing-friendly so don't need T1-FIELDTOWER, which lets STARKs ship before pairings do.
- T1-EC must add the **Pasta cycle** (Pallas: q_pal = p_ves; Vesta: q_ves = p_pal — used by Halo2 for IVC) and **Bandersnatch** (Edwards curve over BLS12-381 scalar field — for in-circuit Pedersen).
- T1-HASH must add **SHA-256 + Keccak-256 + BLAKE2s** (transcript hashes used by every protocol below) — already in 057's T1-HASH list.

### T1-NTT — Number-Theoretic Transform / Inverse-NTT (≈300 LOC)
The radix-2 Cooley-Tukey FFT in a finite field. Required for: every polynomial commitment scheme, every fast multiplication, every Reed-Solomon encoder, every FRI low-degree test, every PlonK proof generation step. Three variants needed:
- **Forward NTT** (coefficients → evaluations on a 2^k-th root-of-unity coset): radix-2 decimation-in-time, Bailey's algorithm for cache-friendly large sizes.
- **Inverse NTT** (evaluations → coefficients): same shape, multiply by N^−1 at end.
- **Coset NTT** (forward NTT shifted by a generator) — the FRI low-degree test absorbs evaluations on a coset of the trace-domain subgroup.
- **Truncated NTT / Mixed-radix** — for non-power-of-2 sizes (less common in production but needed if the user wants n=192 etc.).
- Constant-time discipline: NTT itself does not branch on field elements, so naturally CT once T1-FIELD is.

### T1-MSM — Multi-Scalar Multiplication via Pippenger's algorithm (≈400 LOC)
Given vectors `(s_0..s_{n-1}, P_0..P_{n-1})`, compute Σ s_i · P_i. The single most expensive operation in any curve-based ZK proof. Pippenger gives O(n / log n) scalar multiplications vs the naive O(n).
- **Bucket method** (Pippenger 1976): split each scalar into c-bit windows, accumulate per-window buckets of the same window value, sum buckets at end with a doubling chain.
- **Window-size selection**: c ≈ log_2(n) − 2 for typical n ∈ [2^14, 2^22].
- **GLV / GLS endomorphism splitting** (BLS12-381 scalar split into (k1, k2) with k1, k2 ≈ √r) halves the bit-length of each scalar; depends on T1-EC having the curve endomorphism precomputed.
- **Two-buckets-per-window signed digit** halves bucket count again.
- **Constant-time MSM**: production ZK prover is *not* constant-time on scalars (the scalars are public-input-derived after Fiat-Shamir, so constant-time is unnecessary). Reality should ship both a public-scalar fast-path and a CT-scalar fall-back labeled by a `MSMVariant` enum; the docstring must name which is which.
- **Allocation discipline**: a 2^20 MSM allocates ~32 MB of bucket points if naive — ship a `MSMScratch` workspace type per CLAUDE.md §3.

### T1-POLY — Polynomial arithmetic, univariate + multilinear (≈500 LOC)
Required by: KZG (univariate), FRI (univariate), Halo2 (univariate), STARK (univariate), Spartan (multilinear), HyperPlonk (multilinear), every sumcheck-based system (multilinear).
- **Univariate Polynomial type**: `[]F` of coefficients, low-to-high, plus an `Evaluations` companion type (`[]F` of evaluations on a Lagrange domain).
- **Add/Sub/Scale/Negate** over coefficient form.
- **Multiply** via NTT (degree d × degree e in O((d+e) log(d+e)) once T1-NTT lands).
- **Divide-with-remainder** via long division (for opening proofs: f(X) − f(z) divisible by (X−z)).
- **Lagrange interpolation** from evaluations (via iNTT for power-of-2 evaluation sets, naive O(n²) Lagrange for arbitrary point sets).
- **Vanishing polynomial** Z_H(X) = X^n − 1 for the multiplicative subgroup H of size n.
- **Multilinear extension (MLE)** of f: {0,1}^n → F: the unique multilinear polynomial agreeing with f on the boolean cube. Two representations: (a) coefficient list (length 2^n), (b) **dynamic-evaluation form** (Thaler's algorithm) — evaluate at r = (r_1, …, r_n) in O(2^n) time, no allocation per round.
- **Batched MLE evaluation**: evaluate k MLEs at the same point r in O(k·2^n) by sharing the partial-product table.
- **Identity / permutation / lookup polynomials** as named factory functions (so PlonK/lookup callers don't redo bookkeeping).

### T1-MERKLE — Merkle commitment with arity-2 and arity-N variants (≈200 LOC)
Required by: FRI, STARK, Brakedown, Ligero, Orion, every transparent (no-trusted-setup) commitment scheme.
- **Binary Merkle tree** with caller-supplied hash (so the same code works with SHA-256, Keccak-256, BLAKE2s, or Poseidon).
- **Arity-N Merkle tree** (typical N = 4 or 8 for STARK to amortise hash cost — Plonky2 uses arity 8 with Poseidon).
- **Path proof generation** + **path proof verification**.
- **Batch opening** with deduplication of shared path nodes (saves ~10× on FRI proof size).
- **Frontier-only commitment** for streaming proofs (commit to the rolling frontier without storing the full tree).

### T1-TRANSCRIPT — Fiat-Shamir transcript with domain-separation discipline (≈150 LOC)
The single most-foot-gunned primitive in deployed ZK code (146's §C also flags this). Every protocol below needs it.
- **Transcript interface**: `Absorb(label string, bytes []byte)`, `Squeeze(label string, n int) []byte`, `SqueezeChallenge(label string) FieldElement`.
- **Two implementations**: (a) hash-based (BLAKE2b or Keccak with explicit length-prefixed labels — Merlin-style), (b) sponge-based (Poseidon-rate-2-arity-3, in-field absorption — for STARK and recursive systems).
- **Versioned protocol-label root** so different protocols never collide on the same transcript bytes (`b"reality-zkmark-v1-plonk"`, etc.).
- **Empty-transcript rejection**: any `SqueezeChallenge` before at least one `Absorb` must error — common attack vector.
- **Replay-protection**: each call appends a monotonically incrementing call counter into the absorbed bytes, so swapping two `Absorb`/`Squeeze` calls yields a different challenge.
- Critical: this is the package where 146's §G7 JSON-ordering finding lives in spirit — the *byte sequence* fed to the transcript hash *is* the canonical proof artifact and must be golden-file-pinned across Go/Python/C++/C#.

### T1-RS — Reed-Solomon encode + decode (≈100 LOC after NTT lands)
Almost free once NTT exists. Required by: FRI, STARK, Ligero, Brakedown.
- **Encode** (k coefficients → n evaluations, n ≥ 2k): zero-pad to length n, NTT.
- **Erasure decode** (recover k coefficients from any k evaluations): Welch-Berlekamp for general settings, or direct Lagrange when the evaluation positions are a subgroup.
- **Berlekamp-Massey** for syndrome decoding (used in some interactive PCS).
- **Rate parameter** as named constant — STARKs typically use rate ρ = 1/8 (n = 8k); FRI permits ρ ∈ {1/2, 1/4, 1/8, 1/16} with security trade-offs.

### T1-POSEIDON — Poseidon / Rescue / Anemoi algebraic hash (≈400 LOC)
Required by: every recursive ZK system (Halo2, Nova, Plonky2, RISC0, Pickles), every transcript-in-circuit, every Merkle-in-circuit, every lookup-table-in-circuit.
- **Poseidon hash** (Grassi-Khovratovich-Lüftenegger-Rechberger-Roy-Schofnegger 2020): width t = 3 (rate 2, capacity 1), field-dependent S-box (x^5 for BN254/BLS12, x^7 for Goldilocks), MDS matrix, full + partial round structure, round-constant table.
- **Rescue-Prime** (Aly-Ashur-Ben-Sasson-Dhooghe-Szepieniec 2020) — same shape, different round counts and S-box (x^α for the smallest α such that gcd(α, p−1) = 1).
- **Anemoi** (Bouvier-Briaud-Chaidos-Perrin-Salen-Velichkov-Willems 2022) — Flystel S-box, optimised for STARKs.
- **Tip5** / **Reinforced-Concrete** / **Griffin** — Tier 3, less-deployed.
- Per-field round-constant tables must be golden-pinned (specific (p, t, α) → specific RC[]) so the C# / Python / C++ ports can't drift.
- **In-circuit-friendly**: number-of-multiplications metric (not byte throughput) is the relevant cost.

---

## Missing primitives — Tier 2 (the topic-listed ZK primitives)

Topic enumerates: **KZG, FRI, IPA, PlonK, Halo2, STARK, Reed-Solomon, MLE, lookup arguments (logUp, cq, plookup)**. Each requires the Tier 1 foundation above.

### T2-KZG — Kate-Zaverucha-Goldberg polynomial commitment (≈300 LOC)
Univariate, pairing-based, requires **trusted setup** (powers-of-tau SRS).
- **Setup** (SRS-load only — generation is OUT): given `[1, τ, τ², …, τ^d]·G_1` and `[1, τ]·G_2`, deserialise into pre-validated curve points.
- **Commit(f)**: c = Σ f_i · [τ^i]·G_1 — one MSM call, n = degree(f).
- **Open(f, z)**: π = Σ q_i · [τ^i]·G_1 where q(X) = (f(X) − f(z)) / (X − z) — long division then MSM.
- **Verify(c, z, y, π)**: pairing check e(c − [y]·G_1, [1]·G_2) == e(π, [τ]·G_2 − [z]·G_2).
- **Batched opening** (KZG batch): single proof for k polynomials at the same point z (random-linear-combination via transcript challenge).
- **Multi-point opening** (Feist-Khovratovich): single proof for one polynomial at k points.
- **Citation**: KZG10 paper, EIP-4844 / Danksharding wire format.
- **Depends on**: T1-FIELDTOWER (G_2 is in Fp2), T1-EC (BLS12-381 G_1, G_2), T1-MSM, T1-POLY, T1-TRANSCRIPT (for batching challenge).
- **Reality status**: zero of the above exist. KZG is **un-reachable** until T1-EC + T1-FIELDTOWER land.

### T2-FRI — Fast Reed-Solomon Interactive Oracle Proof of Proximity (≈600 LOC)
Univariate, *transparent* (no trusted setup), the workhorse of every STARK.
- **Commit phase**: prover commits to Reed-Solomon-encoded f over a domain D_0 of size n via Merkle root r_0; verifier sends folding challenge α_0; prover splits f = f_even(X²) + X·f_odd(X²), commits to r_1 = Merkle(α_0·f_odd + f_even on D_1 = D_0² halved); repeats log_2(n) − log_2(rate) times.
- **Query phase**: verifier samples ≈ λ / log_2(1/ρ) random positions in D_0, receives Merkle-path-proven evaluations at those positions, checks consistency through the folding tree.
- **Soundness**: requires `λ` random queries for `λ` bits of security at rate ρ ≥ 1/8 (Block-Holmgren-Rotem 2018 → DEEP-FRI proximity gaps).
- **Optimisations**: (a) fold-by-arity-N instead of N=2 (Plonky2 uses arity 8 — fewer Merkle commitments, more challenges per round); (b) **batched FRI** — commit to k polynomials and prove low-degree of a random linear combination; (c) **DEEP-FRI** — quotient by an out-of-domain point to eliminate the "list-decoding gap" attack.
- **Depends on**: T1-RS, T1-NTT, T1-MERKLE, T1-TRANSCRIPT (for fold challenges + query positions).
- **No curve, no pairing, no SRS** — FRI is the Tier-1 anchor for transparent ZK.
- **Citation**: Ben-Sasson-Bentov-Horesh-Riabzev 2018; DEEP-FRI 2020; Block-Garreta-Hall-Katz-Liu-Tairi 2023 (proximity gaps).

### T2-IPA — Inner Product Argument (Bulletproofs-style polynomial commitment) (≈400 LOC)
Univariate, *transparent* (no trusted setup), used by **Halo2**.
- **Setup**: deterministic generators `G_1..G_n, H` from a hash-to-curve nothing-up-my-sleeve construction.
- **Commit(f)**: c = Σ f_i · G_i — one MSM call.
- **Open**: log_2(n) rounds of Bulletproofs-style folding: prover commits to L_i, R_i; verifier sends u_i; both sides fold (G, f) ↦ (G' = u·G_left + u^{−1}·G_right, f' analogously). Final round is one scalar / one curve point.
- **Verify**: O(n) (single MSM of n generators against the recursion-folded basis) — important: **IPA verifier is linear in n**, unlike KZG's O(1). This is why Halo2 introduces *recursive accumulation* (Halo / Pickles / Mina) to amortise verifier cost.
- **Depends on**: T1-EC (Pasta curves), T1-MSM, T1-POLY, T1-TRANSCRIPT.
- **No pairing required** — IPA is why Halo2 can use the Pasta cycle (no pairing-friendly cycle is known).
- **Citation**: Bulletproofs (Bünz-Bootle-Boneh-Poelstra-Wuille-Maxwell 2018); Halo / Halo2 (Bowe-Grigg-Hopwood 2019, ECC 2021).

### T2-PLONK — Permutation-Argument SNARK (≈800 LOC)
Universal-SRS, KZG-based-or-FRI-based, the workhorse SNARK.
- **Plonkish constraint system**: rows of (q_L, q_R, q_O, q_M, q_C) selectors plus (a, b, c) wire columns plus copy-constraint permutation σ.
- **Permutation argument** (grand-product polynomial Z_σ): proves Σ values in (a,b,c) match Σ values under σ via Σ-product identity.
- **Custom gates** (turbo-PlonK): higher-degree selectors for ad-hoc constraints (e.g. range checks).
- **Lookup gate** (PlonK + plookup or PlonK + logUp — see below).
- **Prove**: commit to wire polys, permutation-Z poly, quotient poly t = (constraint identity) / Z_H, evaluate at random challenge ζ, batch-KZG-open everything.
- **Verify**: KZG-batch-verify a single linearisation polynomial.
- **Depends on**: T2-KZG (or T2-FRI for Plonky2-style "Plonkish over FRI"), T1-POLY, T1-NTT, T1-TRANSCRIPT, T1-POSEIDON (for in-circuit transcript).
- **Citation**: Gabizon-Williamson-Ciobotaru 2019.

### T2-HALO2 — Halo2 (PlonK over IPA, recursive) (≈1000 LOC on top of T2-PLONK + T2-IPA)
The system zkmark.go names in the README. Halo2 = PlonK custom-gates + Plookup lookups + IPA polynomial commitment + Pasta cycle + recursive accumulation.
- **Accumulation scheme**: each proof accumulates the linear MSM check from the verifier into a single growing "accumulator" point that is itself proven correct by the next proof.
- **In-circuit verifier**: the verifier circuit has cost log_2(n) Poseidon hashes per recursion step (vs Halo's full O(n) IPA verifier cost).
- **Pasta cycle**: G1 over Pallas verifies a proof over Vesta and vice versa; needs both curves at T1-EC.
- **The pkg-level decision the README asserts (Halo2)** is the *most* expensive of the topic-listed primitives — ~3× the LOC of Plonky2 (FRI-based) for the same expressiveness, in exchange for cycle-of-curves recursion without pairings.

### T2-STARK — Scalable Transparent ARgument of Knowledge (≈1200 LOC)
AIR (Algebraic Intermediate Representation) constraint system + FRI commitment + DEEP composition.
- **AIR**: trace columns over a small prime field (Goldilocks, BabyBear, M31), transition constraints (polynomials over (current_row, next_row)), boundary constraints, periodic constraints.
- **DEEP composition** (Ben-Sasson-Goldberg-Kopparty-Saraf 2020): combine all constraints into a single low-degree polynomial via an out-of-domain challenge.
- **FRI low-degree-test** on the composition poly (T2-FRI).
- **No curve, no pairing** — STARKs run entirely in a small prime field.
- **Citation**: StarkWare 2018 white-paper; ethSTARK; Plonky2 (which is "PlonK over FRI", so technically a STARK by FRI-commitment but with a PlonK-style constraint system).
- **Depends on**: T1-FIELD (Goldilocks), T1-NTT, T1-RS, T2-FRI, T1-POSEIDON, T1-TRANSCRIPT, T1-MERKLE.

### T2-RS — Reed-Solomon (already T1, listed in topic for completeness)
Already covered at T1-RS above. The topic groups it with the SNARK list because every transparent SNARK uses RS encoding as its commitment substrate.

### T2-MLE — Multilinear Extension (already T1, listed in topic)
Already covered at T1-POLY above. The topic groups it with the SNARK list because Spartan, HyperPlonk, and Lasso all commit to MLEs (rather than univariate polynomials), and a multilinear KZG variant (Zeromorph, PST) is a separate primitive — see T2-MLE-COMMIT below.

### T2-MLE-COMMIT — Multilinear polynomial commitment (≈400 LOC)
Not univariate KZG. Variants:
- **Zeromorph** (Kohrita-Towa 2024): reduces multilinear KZG to univariate KZG via a quotient identity.
- **PST** (Papamanthou-Shi-Tamassia 2013): direct multilinear extension of KZG with O(log n) opening proof.
- **Sona** / **Hyrax** (Wahby-Tzialla-shelat-Thaler-Walfish 2018): multilinear IPA, transparent, no trusted setup — directly used by Spartan.
- **Brakedown** (Golovnev-Lee-Setty-Thaler-Wahby 2021): code-based commitment, transparent, hash-based, faster than FRI for prover but larger proofs.
- **Citation**: Spartan 2020, HyperPlonk 2022, Lasso/Jolt 2023-2024 (Setty-Thaler-Wahby).

### T2-LOGUP — Logarithmic-derivative lookup argument (≈200 LOC)
"For each (witness column w, table column t), prove every entry of w lies in t."
- **logUp identity**: Σ_i 1/(α − w_i) = Σ_j m_j/(α − t_j) where m_j is the multiplicity of t_j in w. Holds iff w ⊂ t (with multiplicities).
- **Reduction to sumcheck**: clear denominators via partial fractions, get a polynomial identity in α; use grand-product (PlonK-style) or sumcheck (Spartan-style).
- **Bag equality** (multi-set check): same construction with α a transcript challenge.
- **Citation**: Haböck 2022 ("Multivariate lookups based on logarithmic derivatives"); Eagen 2024 (extension to bivariate).
- **Depends on**: T1-POLY, T1-TRANSCRIPT, plus *one* of {T2-KZG, T2-FRI, T2-IPA, T2-MLE-COMMIT} for the underlying poly-commit.

### T2-PLOOKUP — Grand-product lookup argument (≈200 LOC)
Predecessor to logUp; still common in production.
- **plookup identity**: Z(X) = Π_i (β + w_i)·Π_j (β + t_j) / (Π products of pre-merged sorted column) — a grand-product polynomial whose closure proves multi-set equality of the merged sorted column with the witness ⊕ table.
- **Citation**: Gabizon-Williamson 2020.
- **Variants**: **plookup-2** (PlonK-friendly, additive challenges), **caulk / caulk+** (sub-linear-prover lookups for very large tables — Zapico-Bühlmann-Zhang-Pailoor-Maller-Goldberg-Boneh 2022).

### T2-CQ — cq lookup (Zapico-Buterin-Maller-Nitulescu-Pailoor-Goldberg-Lipmaa 2022) (≈300 LOC)
Sub-linear-prover lookup with KZG.
- **Pre-processed table commitments** to KZG-encoded `1/(X − t_j)` for each table entry.
- **Online prover cost** is O(n log n) in *witness* size, independent of *table* size — the headline win.
- **Depends on**: T2-KZG, T1-POLY, T1-TRANSCRIPT.

---

## Missing primitives — Tier 3 (modern frontier, optional)

### T3-FOLDING — Nova / Sangria / ProtoStar / HyperNova (≈600 LOC each)
Folding schemes that compress two relaxed-R1CS / customisable-gate instances into one without an in-circuit verifier.
- **Nova** (Kothapalli-Setty-Tzialla 2022): folds two relaxed-R1CS instances. Uses Pasta cycle + IPA.
- **Sangria** (Mohnblatt 2022): Nova for Plonkish.
- **ProtoStar** (Bünz-Chen 2023): folds *any* special-sound protocol; optimal communication.
- **HyperNova** (Kothapalli-Setty 2023): folds customisable-constraint-systems via sumcheck, no in-circuit MSM.
- **Citation chain**: Halo (2019) → Nova (2022) → Sangria/ProtoStar (2023) → HyperNova (2024).

### T3-LASSO-JOLT — Lookup-argument-based zkVM (≈800 LOC)
Setty-Thaler-Wahby 2023-2024: every CPU instruction → lookup into a precomputed instruction-table; folded via Spartan + Lasso. The frontier of "ZK for general computation".
- Depends on: T2-MLE-COMMIT, T2-LOGUP (Lasso uses a generalised logUp), Spartan-style sumcheck.

### T3-SPARTAN — Setty 2020 (≈500 LOC)
Multilinear-extension SNARK with sumcheck verifier, no FFT, transparent setup with Hyrax / Brakedown / Orion commitment.

### T3-HYPERPLONK — Chen-Bünz-Boneh-Zhang 2022 (≈700 LOC)
PlonK-on-the-boolean-hypercube; sumcheck instead of NTT-based polynomial-evaluation; supports MLE commit.

### T3-GROTH16 — Groth 2016 SNARK (≈400 LOC)
The original universal-circuit pairing-based SNARK, still used by Zcash Sapling, Tornado Cash, Filecoin.
- **Per-circuit trusted setup** (worst-case from the trusted-setup perspective) — superseded by PlonK universal setup.
- **Tiny proof**: 3 group elements (~192 bytes on BN254).
- **Constant-time verifier**: 3 pairings.
- **Depends on**: T2-KZG-style commitment but per-circuit (not universal), full T1-FIELDTOWER + T1-EC + T1-PAIRING.

### T3-LIGERO-BRAKEDOWN-ORION — Code-based PCS family (≈400 LOC each)
Transparent, hash-only, post-quantum-friendly polynomial commitments. Brakedown is faster prover than FRI; Orion is a 2022 improvement.

### T3-AIR-EXT — RAP / lookups-in-AIR (≈300 LOC)
Randomised-AIR-with-Preprocessing extends STARK's AIR with auxiliary trace columns sampled from verifier challenges; required for in-AIR lookup arguments (Plonky2/Plonky3, RISC0).

### T3-CIRCLE-STARK — Mersenne-31-based STARK (≈500 LOC)
StarkWare 2024: STARK over the **circle group** of Mersenne-31 (p = 2^31 − 1), enabling FFT despite zero two-adicity. Frontier 2024-2026 work.

### T3-POSEIDON2 / TIP5 / GRIFFIN — Updated algebraic hashes (≈300 LOC each)
Poseidon2 (2023, 2× faster), Tip5 (2024, optimised for Tip-top STARK), Griffin (2022, 4 round-types).

### T3-CYCLES-OF-CURVES — Beyond Pasta (≈600 LOC)
- **Pasta** (Pallas/Vesta) — Halo2.
- **secp/secq** — for Bitcoin-compatible recursion.
- **Bandersnatch** — Edwards over BLS12-381's scalar field; not a cycle but an "embedded curve" enabling EdDSA-in-circuit on BLS12-381.

### T3-ZK-IN-PRACTICE engineering layers (OUT)
- Circuit IR (Halo2 chip API, Plonky2 circuit-builder) — engineering, not math, OUT.
- Witness generators / R1CS-to-Plonkish compilers — engineering, OUT.
- ZK-rollup state-trees (Verkle, Sparse-Merkle-Tree of polynomials) — borderline; the Verkle math (vector commit + Pedersen tree) is in scope but the rollup-state-machine is not.

---

## Sprint ordering recommendation

Identical to 057's "what unblocks the most" reasoning, plus ZK-specific dependencies.

1. **Sprint 1 (foundation, depends on 057's sprint 1)**: T1-NTT, T1-POLY, T1-MERKLE, T1-TRANSCRIPT, T1-RS. Total ≈1,250 LOC. **Unblocks**: every transparent ZK primitive in Tier 2 modulo T1-POSEIDON.
2. **Sprint 2 (algebraic hash + small-field)**: T1-POSEIDON, T1-FIELD additions for Goldilocks/M31/BabyBear. Total ≈600 LOC. **Unblocks**: STARKs, Plonky2-style protocols, in-circuit recursion.
3. **Sprint 3 (Tier-1 anchor: STARK)**: T2-FRI + T2-STARK over Goldilocks. Total ≈1,800 LOC. **Single shippable artifact**: a transparent-setup, no-curve-needed proof system that exercises every Sprint 1+2 piece.
4. **Sprint 4 (curve work, depends on 057 T1-EC + T1-FIELDTOWER)**: T1-MSM with Pippenger+GLV. Total ≈400 LOC. **Unblocks**: every pairing-or-IPA-based primitive.
5. **Sprint 5 (Tier-1 anchor: PlonK)**: T2-KZG + T2-PLONK over BLS12-381. Total ≈1,100 LOC. **Single shippable artifact**: universal-SRS SNARK.
6. **Sprint 6 (lookups)**: T2-LOGUP + T2-PLOOKUP + T2-CQ. Total ≈700 LOC. Each composes with whichever poly-commit is already shipped.
7. **Sprint 7 (Halo2)**: T2-IPA + T2-HALO2 over Pasta. Total ≈1,400 LOC. Closes zkmark.go's stated Tranche-2 commitment.
8. **Sprint 8+ (frontier)**: any combination of T3-FOLDING, T3-LASSO-JOLT, T3-SPARTAN, T3-HYPERPLONK, T3-CIRCLE-STARK, T3-POSEIDON2 — driven by consumer pull.

Total to reach Halo2 parity with the README's S62-S63 promise: **≈5,250 LOC** of math (Sprints 1, 2, 4, 5, 7) plus 057's foundation (~2,500 LOC) = **≈7,750 LOC**, before any consumer code or test harness. With ≥20 golden-file test vectors per primitive at minimum-30 target (CLAUDE.md §"Golden-File Testing Infrastructure"), test surface is roughly equal to implementation surface, so total addition to `reality` is ~15,000 LOC across ~8 sprints.

---

## Constructive recommendations for `zkmark/` itself, separate from primitive enumeration

These are the **package-level** observations that fall out of doing this missing-primitive audit, separate from 146's findings:

1. **The package name `zkmark` is misleading.** "ZK" implies zero-knowledge, but per the §"Scope filter" note above, `reality`'s deterministic-by-construction posture means *no prover-side randomness*, which means the prover must either (a) accept that proofs are *not* zero-knowledge (only sound — "succinct argument of knowledge", not "zero-knowledge succinct argument of knowledge"), or (b) accept a `randomness io.Reader` parameter at the public API and break determinism. **This decision must be pinned in zkmark.go's package docstring before any Tranche 2 code is written**, because it affects every Tier-2 protocol's API shape.
2. **Subpackage layout should mirror `crypto/`'s eventual layout.** Recommend `zkmark/field/`, `zkmark/poly/`, `zkmark/commit/` (with `commit/kzg`, `commit/fri`, `commit/ipa` sibling dirs), `zkmark/proof/` (with `proof/plonk`, `proof/halo2`, `proof/stark`), `zkmark/lookup/`, `zkmark/hash/` (with `hash/poseidon`, `hash/rescue`). The current single-file zkmark.go is fine for Tranche 1 but **cannot scale** to Tier 2.
3. **The Tranche 2 commitment to "Halo2" specifically (zkmark.go line 65–69, README line 7) over-specifies.** The README should keep the Tranche-2 *interface contract* (a real zero-knowledge proof in `Proof.ProofBytes`) but should not commit to Halo2 as the only Tier-2 backend — a STARK backend (Sprint 3 above) is *cheaper to ship in Go* than Halo2 (no curve, no pairing, just NTT + FRI + Poseidon) and gives the same regulator-grade audit property at a different proof-size / verifier-cost trade-off. Recommend renaming `AlgorithmHalo2` to `AlgorithmTranche2` until the actual backend is selected, OR adding `AlgorithmStark` as a sibling constant now so the parser shape is forward-compatible.
4. **The C# port (R80b) byte-parity contract is impossible to satisfy with the topic-listed primitives** unless either (a) both Go and C# implement the field/curve/poly stack from scratch with shared golden vectors (`reality`'s actual model), or (b) both wrap the same Rust sidecar (the README's actual model). The current zkmark.go README §"Cross-substrate parity contract" claims only the *envelope* parity (which is trivial — JSON), not the *proof bytes* parity (which requires (a) or (b)). Recommend adding a sentence to the README acknowledging that proof-byte parity is delegated to the Tranche 2 backend choice.
5. **Test count is below `reality` floor.** 10 tests for a regulator-facing audit substrate is light vs the 1,965-test repo norm and the CLAUDE.md "≥20 golden vectors per function" floor. The honest counter is that there are no math functions yet to *have* golden vectors for — but the *envelope* JSON ordering, the `ErrNotYetWired` sentinel, and the algorithm-whitelist closed-set behavior should each have at least 5 cases. 146's §G11 already flagged this; this audit reinforces.
6. **Tier-1 anchor decision pending.** Of the three "first ZK proof to ship": (a) Reed-Solomon + Sumcheck (≈300 LOC, no transcript, no commitment, demo-only), (b) FRI low-degree-test on Goldilocks (≈600 LOC, real soundness, transparent), (c) full STARK (≈1,800 LOC, regulator-shippable). The cheapest *shipped* artifact that *honestly proves something* is (b) — FRI on a small prime field. Recommend that when Tranche 2 starts, FRI is the first PR, *not* Halo2, because it gates fewer 057-foundation pieces and exercises every Sprint 1+2 primitive.

---

## Bottom line

`zkmark/` is the same shape as `crypto/`: a name-staking placeholder above a missing 5-layer foundation. The 11 topic-listed primitives are **not** going to land as math in `reality` until 057's T1-* lands (big-int, prime-field, field-tower, EC, hash) **and** this audit's four ZK-only T1 layers land (NTT, MSM, Poly, Merkle, Transcript, Reed-Solomon, Poseidon — call it T1-ZK collectively). That's ~7,750 LOC of math primitives plus equal test surface, distributed across ~8 sprints. The single highest-leverage *shippable* artifact, once the foundation is in place, is FRI-over-Goldilocks (Sprint 3 = T2-FRI + T2-STARK ≈ 1,800 LOC) — it requires no curve and no pairing, gives a transparent transparent-setup ZK proof system (in the SNARK-not-zkSNARK sense per recommendation #1), and exercises every Sprint-1+2 piece including the algebraic hash, the NTT, the Merkle commitment, and the Fiat-Shamir transcript that 146 flagged as the most-foot-gunned forward-compat hazard. **Halo2 specifically (the backend zkmark.go currently names) is the most expensive of the topic-listed primitives**, requires the entire 057 + this-audit foundation including pairings and IPA and the Pasta cycle, and should not be the first Tranche-2 PR; recommend renaming the Tranche-2 algorithm constant to be backend-agnostic now (`AlgorithmTranche2` or `AlgorithmZk` rather than `AlgorithmHalo2`) so the substrate envelope doesn't lock in a 5,250-LOC backend choice 2-3 sprints before that backend ships.

## Two-line summary

zkmark/ is a 262-line interface stub: zero of the 11 topic-listed primitives (KZG, FRI, IPA, PlonK, Halo2, STARK, Reed-Solomon, MLE, logUp, cq, plookup) exist in the repo or its dependencies — they all gate on 057's foundation (big-int / field / curve / pairing / hash) plus four ZK-only foundation layers this audit enumerates (NTT, MSM, polynomial+MLE, Merkle, Fiat-Shamir transcript, Reed-Solomon, Poseidon-family algebraic hashes), totalling ~7,750 LOC of math across ~8 sprints. The single-highest-leverage Tranche-2 PR is FRI-over-Goldilocks (≈1,800 LOC, no curve, no pairing, transparent setup), not Halo2 as the package's README currently commits to — recommend renaming `AlgorithmHalo2` to a backend-agnostic constant before the substrate locks in a 5,250-LOC pairing-based backend choice that can be replaced by a 1,800-LOC FRI-based one with equivalent regulator-grade soundness.

## Progress

- 2026-05-08 agent-147 zkmark-missing: enumerated 11 topic-listed primitives across 3 tiers + 8 ZK-only T1 layers (NTT/MSM/Poly/Merkle/Transcript/RS/Poseidon) on top of 057's 5-layer foundation; sprint ordering identifies FRI-over-Goldilocks (~1,800 LOC, transparent, no curve) as cheapest Tranche-2 anchor vs Halo2's ~5,250 LOC pairing-bound backend; flagged AlgorithmHalo2 over-specification, deterministic-vs-zero-knowledge naming mismatch, and missing C#/Rust proof-byte parity contract.
