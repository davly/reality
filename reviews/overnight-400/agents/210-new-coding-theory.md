# 210 | new-coding-theory

**Summary line 1.** TENTH Block-C cutting-edge-math review and FIRST coding-theory scoping in the 400-sequence covering linear block codes (generator G / parity-check H / syndrome decoding) / Hamming codes (Hamming 1950 single-error-correct) / Hadamard / Reed-Muller (Muller 1954, Reed 1954) / cyclic codes / CRC-8/16/32/64 (Peterson-Brown 1961) / Galois-field arithmetic GF(2)/GF(2^m)/GF(p) / Reed-Solomon (Reed-Solomon 1960) over GF(2^m) / BCH (Bose-Ray-Chaudhuri-Hocquenghem 1959-60) / Berlekamp-Massey 1968 / Forney 1965 syndrome inversion / Sudan-Guruswami list decoding 1997-99 / convolutional codes (Elias 1955) / Viterbi decoding 1967 / BCJR forward-backward 1974 / LDPC (Gallager 1963 / MacKay-Neal 1996) / sum-product belief-propagation / min-sum decoder / polar codes (Arıkan 2009) / channel polarisation / successive-cancellation + SCL decoders / turbo codes (Berrou-Glavieux-Thitimajshima 1993) / iterative decoding via parallel/serial concatenation / repeat-accumulate codes / fountain codes (Luby Transform 2002) / Raptor codes (Shokrollahi 2006) / RaptorQ RFC-6330 / network coding (Ahlswede-Cai-Li-Yeung 2000) / random linear network coding / MDS codes / Singleton bound / Plotkin / Hamming / Gilbert-Varshamov bound / BSC / BEC / AWGN channel models / soft- vs hard-decision / bit-error-rate analysis / Goppa / algebraic-geometry codes / concatenated codes (Forney 1966) / spatially-coupled LDPC / 5G NR LDPC + polar specs: reality v0.10.0 ships **ZERO** error-correction-code surface. `crypto/modular.go` ships ModPow / ModInverse / ExtendedGCD / CRT / GCD on uint64 (the prime-field building blocks for BCH-syndrome inversion and Reed-Solomon-over-GF(p)) but no GF(2) bit-vector arithmetic, no GF(2^m) extension-field tower, no irreducible polynomial table, no Cayley primitive-element generator. `compression/coding.go` is RLE+delta only (lossless source coding, not channel coding). `crypto/hash.go` is FNV/MurmurHash3 (non-cryptographic hash, not error detection). `sequence/distance.go:107-115` HammingDistance is the *string-comparison* distance (the metric Hamming codes are designed to maximise) but no Hamming code itself. `signal/window.go` Hamming refers to the spectral window. Repo-wide grep on `Reed|BCH|LDPC|polar|turbo|Galois|GF\(|Viterbi|trellis|BCJR|Berlekamp|fountain|Luby|CRC|syndrome|generator matrix|parity check` returns ZERO callable hits — three false positives only (string-distance Hamming, signal Hamming window, polar-angle in geometry). The single most-cited textbook in the field (Lin-Costello, *Error Control Coding*, 2nd ed. 2004) is unrepresented; the modern reference (Richardson-Urbanke, *Modern Coding Theory*, 2008) is unrepresented. Reality has the three CLAIMED-by-CLAUDE.md prerequisite packages — `crypto/modular.go` (prime field), `linalg/` (matrix algebra), `compression/` (information theory) — but never assembled them into a coding layer. Closest tangential surface is `zkmark/` Halo2-honest-pending placeholder (no FFT-friendly field, no Reed-Solomon-IOP commitment), which would BENEFIT from an in-repo Reed-Solomon (FRI uses RS as the low-degree-test primitive).

**Summary line 2.** Twenty-eight primitives C1–C28 totalling ~5,420 LOC across new sub-package `coding/` (sibling to `crypto/`, `compression/`, `info/` — those three handle source coding and number-theoretic primitives; `coding/` handles channel coding and finite-field linear algebra). Recommended split: `coding/galois/` for GF(2)/GF(2^m)/GF(p) field arithmetic + Cayley tables + log-antilog tables (~520 LOC, prerequisite for everything else); `coding/block/` for linear-block / Hamming / Reed-Muller / Hadamard / generic G,H syndrome decode (~720 LOC); `coding/cyclic/` for cyclic-shift-invariant codes + CRC-8/16/32/64 + polynomial division (~480 LOC); `coding/rs/` for Reed-Solomon encode + Berlekamp-Massey + Forney + Chien search + erasure decoder (~720 LOC); `coding/bch/` for BCH binary-Reed-Solomon-over-subfield (~380 LOC); `coding/conv/` for convolutional encoder + Viterbi (hard+soft) + BCJR forward-backward (~520 LOC); `coding/ldpc/` for parity-check matrix construction (Gallager / MacKay / progressive-edge-growth) + sum-product + min-sum + offset-min-sum decoders (~640 LOC); `coding/polar/` for channel-polarisation construction + successive-cancellation + SCL (list-L) + CRC-aided SCL (~580 LOC); `coding/turbo/` for parallel-concatenated convolutional + iterative BCJR (~320 LOC); `coding/fountain/` for LT + Raptor + soliton degree distributions + ideal-soliton + robust-soliton (~280 LOC); `coding/channel/` for BSC / BEC / AWGN / Rayleigh channel-simulation primitives + LLR conversions + capacity formulas (~280 LOC). Tier-1 keystone **C1+C2+C3+C4+C5 = `Galois{Modulus uint16, ExpTable, LogTable [256]uint8}` GF(2^8) + `Add`/`Mul`/`Inv`/`Pow`/`Polynomial` + `BitVector` GF(2) packed-bit + `LinearBlockCode{G, H *BitMatrix}` + `SyndromeDecode` ~1,240 LOC** is the irreducible foundation that unblocks every subsequent primitive. **Singular reality competitive moat: C9 Reed-Solomon + C12 Berlekamp-Massey + C13 Forney + C14 Chien-search ~720 LOC** — no zero-dependency Go library ships byte-for-byte cross-substrate parity on RS encode/decode (closest: `klauspost/reedsolomon` is dep-heavy SIMD-tuned, `vivint/infectious` is GF(2^8)-only with no list-decoding path, `protobuf/protoc-gen-go-grpc` ships no RS, `go-rs` is unmaintained); Reality would be the only Go shop with golden-file C# / Python / C++ contract on RS-(255, 223) the deep-space CCSDS standard. **Singular Block-C-2026 frontier: C19 polar-codes SCL decoder ~280 LOC** — Arıkan's 2009 construction is the FIRST capacity-achieving code with sub-exponential complexity, adopted as the 5G NR control-channel standard (3GPP TS 38.212), and the single most-active research area in coding 2020-2026 (CRC-aided SCL, partitioned SC, fast-SSC, simplified-SC); zero Go libraries ship a polar-code SCL decoder. **Singular cross-link: C8 CRC-32 + C20 LT-fountain + C16 Viterbi ~280 LOC** would let `compression/` evolve from "lossless source coding" to "source + channel coding" in one PR, and would let `zkmark/`'s Halo2-pending honestly degrade to a Reed-Solomon FRI-compatible commitment substrate. Cross-package blockers: `linalg.GaussJordan` (currently absent) gates C5 (handwritten ~30 LOC fallback ships unblocked); `prob/random.go` PRNG-stream gates C26 channel-simulation (currently `crypto/rng.go` ships PCG/xorshift — adequate). Versus 057-crypto-missing (which scopes elliptic-curve / lattice / hashing primitives) — orthogonal axis, both ship; specifically crypto Reed-Solomon-over-prime-fields is the math twin of reality's Halo2 FRI commitment. Versus 200-synergy-zkmark-info (which scopes the zkmark/info bridge) — this slot's C9 RS-over-GF(2^m) is a direct dep for any honest-pending FRI replacement.

---

## 0. State at HEAD (2026-05-08, v0.10.0)

Repo-wide audit:

| Surface | Path | Lines | Coding-theory relevance |
|---|---|---:|---|
| ModPow/ModInverse/ExtendedGCD | `crypto/modular.go` | 240 | Prime-field GF(p) building blocks. Uint64-only, no extension-field tower, no GF(2^m). |
| ChineseRemainder | `crypto/modular.go:96` | 40 | CRT (used in BCH-decode for RS-over-prime-power-of-different-primes). Adequate. |
| FNV1a32/64, MurmurHash3 | `crypto/hash.go` | 220 | Non-crypto hash; NOT error-detection (no algebraic structure). |
| RunLength + Delta | `compression/coding.go` | 104 | Source coding only. No channel coding. |
| ShannonEntropy / KL / MutualInfo | `compression/entropy.go` | 177 | Information theory measures. Channel-capacity formulas missing. |
| HammingDistance | `sequence/distance.go:115` | 30 | String metric only. No Hamming code, no decoder. |
| Hamming window | `signal/window.go` | (window) | Spectral window; unrelated. |
| LZ76, MDL/NML, BIC | `info/lz`, `info/mdl` | 1,300 | Algorithmic / parametric complexity. NOT channel coding. |
| Halo2 honest-pending | `zkmark/zkmark.go` | 280 | Tranche-1 placeholder. Tranche-2 needs RS-IOP / FRI; no in-repo RS available. |
| `linalg/` matrix ops | `linalg/*.go` | 2,800 | Float64 matrix algebra. NO GF(2)/GF(2^m) packed-bit matrix; NO `BitMatrix` type; NO `GaussJordan` over a generic field. |

Repo-wide grep audit: `Reed|BCH|LDPC|polar|turbo|Galois|GF\(|Viterbi|trellis|BCJR|Berlekamp|fountain|Luby` returns ZERO callable matches. Three false positives (string-distance Hamming, signal Hamming window, polar-angle in `geometry/polygon.go:92`).

The CLAUDE.md package-table line for `compression` reads "Lossless/lossy compression primitives: entropy, RLE, delta encoding, Huffman, LZ77" — but `Huffman` and `LZ77` are not present in `compression/coding.go` (RLE + delta only). LZ76 is in `info/lz/` (different algorithm: complexity measure not codec). This is a CLAUDE.md drift bug (slot-022 territory) — Huffman should land alongside any channel-coding sub-package since it is a compression-side neighbour to the source-coding theorem that frames Shannon-limit channel coding.

---

## 1. The twenty-eight primitives

Tier numbering: T1 = irreducible foundation (Galois + BitMatrix + linear block), T2 = textbook block / cyclic / Reed-Solomon / BCH, T3 = Shannon-limit-approaching modern (LDPC / polar / turbo), T4 = networking / streaming (fountain / network coding), T5 = channel models + capacity. Each entry: name, LOC, reference, API sketch.

### Tier 1 — Galois + linear-block foundation (~1,240 LOC, ship-now-unblocked)

**C1 — `coding/galois/gf2.go` ~120 LOC.** `BitVector` packed `[]uint64` with `Set/Get/Toggle/PopCount/XOR/AND/OR`, byte-and-bit endianness pinned. `BitMatrix` row-major packed-bit with `RowOp/ColumnSwap/RowEchelon/Rank/Nullspace`, `MulVec` returns `BitVector`. Foundation for Hamming / Reed-Muller / LDPC parity-check matrices. Refs: Lin-Costello 2004 §3.1; MacWilliams-Sloane 1977 *The Theory of Error-Correcting Codes* §1.2. Tested against `H · G^T = 0` identity for every constructed code.

**C2 — `coding/galois/gfp.go` ~80 LOC.** `GFP{P uint32}` prime-field wrapper around `crypto.ModPow` / `crypto.ModInverse`. Adds `Add/Sub/Mul/Inv/Pow/Order` with one-step short-circuit when `P=2` to dispatch to GF(2) bit ops. Reuses crypto package (no math duplication). Useful for non-binary RS / CRC variants and zkmark FRI commitment.

**C3 — `coding/galois/gf2m.go` ~280 LOC — KEYSTONE.** `Galois{M uint8; Modulus uint16; ExpTable, LogTable []uint16}` extension field GF(2^m) for m ≤ 16 (covers GF(256) AES/RS-CCSDS, GF(2^16) extra-strength Reed-Solomon). Constructor `NewGF(m, modulus)` accepts irreducible polynomial; pre-built table `IrreducibleTable(m) uint16` returns the standard primitive polynomial (e.g., `0x11d` for GF(256) AES-Rijndael). API: `Add(a,b) = a^b`, `Mul(a,b) = ExpTable[(LogTable[a]+LogTable[b]) mod (2^m-1)]`, `Inv(a) = ExpTable[(2^m-1) - LogTable[a]]`, `Pow(a,n)`, `Polynomial[]uint16` with `PolyEval/PolyMul/PolyDiv/PolyMod`. Alphabet element 0 handled as the additive identity (LogTable[0] = sentinel). Cross-link to `crypto/aes`-style sub-byte tables (currently absent from crypto/ but fits naturally). Refs: Lin-Costello 2004 §2.7-2.9; MacWilliams-Sloane 1977 §3.

**C4 — `coding/galois/poly.go` ~80 LOC.** `Polynomial` over Galois with `Eval/Add/Mul/Div/Mod/GCD/IsIrreducible/Roots/Derivative`. The `Roots` finder uses Chien search (C14) for GF(2^m), brute force for small GF(p). `IsIrreducible(p)` runs Berlekamp's irreducibility test in GF(2^m). Foundation for RS generator polynomial `g(x) = Π_{i=0}^{2t-1}(x − α^i)`.

**C5 — `coding/block/linear.go` ~280 LOC.** `LinearBlockCode{N, K int; G, H *BitMatrix; SyndromeTable map[uint64]BitVector}` with constructors `FromGenerator(G)` (computes systematic-form H), `FromParityCheck(H)`, `Hamming(m)` (constructs (2^m−1, 2^m−1−m) Hamming code), `RepetitionCode(n)`, `SinglePartiyCheck(n)`. Methods `Encode(msg)` / `Decode(received) (codeword BitVector, syndrome BitVector, err error)` / `MinimumDistance()` / `WeightDistribution() []int`. Syndrome-decode lookup `syndromeTable[s]` precomputes coset-leader for each of the 2^(n−k) possible syndromes (memory cost 2^(n−k) — feasible for n−k ≤ 20). Refs: Hamming 1950; Lin-Costello §3-§4.

**C6 — `coding/block/hamming.go` ~80 LOC.** `Hamming74` (4 data + 3 parity, single-bit-correct), `Hamming1511` (extended Hamming with 1-error-correct + 2-error-detect), `HammingExtended(m)` (Hamming + overall parity bit). Direct G/H matrices baked into static tables (no runtime construction). Used as the textbook validation case for C5.

**C7 — `coding/block/reedmuller.go` ~120 LOC.** `ReedMuller(r, m)` constructs RM(r, m) as the linear span of monomials of degree ≤ r in m boolean variables. (1, m) is the first-order RM with majority-logic decoding; (m, m) is the full code; (1, 5) is the Mariner-9 deep-space code. API: `RMEncode(msg)`, `RMMajorityDecode(received)`, `RMRecursiveSCDecode` (Reed 1954 algorithm). Refs: Muller 1954; Reed 1954; Mariner-9 mission archive.

### Tier 2 — Cyclic + Reed-Solomon + BCH (~1,580 LOC)

**C8 — `coding/cyclic/crc.go` ~80 LOC.** Standard CRC tables with byte-aligned slice-by-8 lookup. `CRC8(data, poly, init)`, `CRC16(data)` Modbus / CCITT / XMODEM variants, `CRC32` IEEE-802.3 + Castagnoli + Koopman, `CRC64` ECMA-182 + ISO-3309. Reflected and forward modes both shipped. Polynomial table `CRCPolynomial{name, poly, init, refIn, refOut, xorOut, check}` matches the reveng catalogue (Cook 2018) byte-for-byte. **Cross-link:** `crypto/hash.go` should add `CRCPolynomial` registration so `crypto.HashByName("crc-32")` returns the standard implementation. Refs: Peterson-Brown 1961; IEEE 802.3.

**C9 — `coding/cyclic/cyclic.go` ~120 LOC.** `CyclicCode{N int; Generator *Polynomial}` with systematic encoder `Encode(msg) = msg·x^{n−k} mod g(x) ∥ msg` and Meggitt syndrome decoder. `MeggittDecode(received)` shifts received polynomial through n positions, querying syndrome → error pattern lookup at each shift. Reuses C4 polynomial division. Refs: Prange 1957; Lin-Costello §5.

**C10 — `coding/rs/encode.go` ~120 LOC.** `ReedSolomon{N, K int; Field *Galois; Generator *Polynomial; FCR int}` (FCR = first consecutive root, typically 0 or 1). `Encode(msg [K]byte) [N]byte` for RS-(255, 223) deep-space CCSDS standard, RS-(255, 251) DVB, RS-(15, 11) GF(16) tutorial case. Systematic encode via polynomial division with byte-buffer reuse (zero allocation per encode). Pre-built fixtures: `RSCCSDS()`, `RSDVB()`, `RSQR()` (QR-code RS), `RSPAR2()` (Parchive). Refs: Reed-Solomon 1960; CCSDS 131.0-B-3; ISO/IEC 18004 (QR).

**C11 — `coding/rs/syndrome.go` ~60 LOC.** `Syndromes(received [N]byte, t int) [2t]byte` evaluates received polynomial at α^FCR, α^FCR+1, ..., α^FCR+2t−1. Zero syndromes ⇒ no error. Hot path: vectorisable inner loop, golden-file cross-validated against vivint/infectious reference.

**C12 — `coding/rs/berlekamp_massey.go` ~140 LOC — KEYSTONE.** Berlekamp-Massey 1968 algorithm to find the minimum-degree LFSR producing a given syndrome sequence. Returns `(Λ Polynomial, L int)` where Λ is the error-locator polynomial of degree L = number of errors. Equivalent to extended Euclidean algorithm on `(x^{2t}, S(x))` (Sugiyama 1975) — both shipped, the Sugiyama path is cleaner Go code, BM is the textbook canonical. Refs: Berlekamp 1968; Massey 1969; Sugiyama-Kasahara-Hirasawa-Namekawa 1975.

**C13 — `coding/rs/forney.go` ~80 LOC.** Forney 1965 algorithm for error-magnitude evaluation. Given Λ(x) error-locator and Ω(x) = Λ(x)·S(x) mod x^{2t} error-evaluator, magnitude at error position i is `Y_i = X_i^{1−FCR} · Ω(X_i^{−1}) / Λ'(X_i^{−1})`. Reuses C4 polynomial derivative. Refs: Forney 1965.

**C14 — `coding/rs/chien.go` ~60 LOC.** Chien 1964 search: find roots of Λ(x) by exhaustive evaluation at α^0, α^1, ..., α^{n−1}. Each step is one multiply by α (since Λ(α^i+1) = Σ Λ_j · (α^i+1)^j = Σ (Λ_j α^j) · α^i+1 — incremental). Refs: Chien 1964.

**C15 — `coding/rs/decode.go` + erasure/list path ~120 LOC.** `Decode(received [N]byte) ([K]byte, decErrors int, ok bool)` chains Syndromes → BM → Chien → Forney. Erasure-aware variant `DecodeWithErasures(received, eraseMask)` accepts up to `2t` erasures + errors with `2·errors + erasures ≤ 2t` (the modified Berlekamp-Massey path). **List-decoding extension:** `SudanList(received, ℓ int)` runs Sudan 1997 / Guruswami-Sudan 1999 list decoding for RS codes, breaking the half-minimum-distance bound (decode `(n − √(nk))` errors instead of `(n−k)/2`). LOC budget: ~80 LOC for stub-pinned interface, ~+250 LOC for full Guruswami-Sudan (defer to v2). Refs: Sudan 1997; Guruswami-Sudan 1999.

**C16 — `coding/bch/bch.go` ~280 LOC.** BCH binary code over GF(2^m), the binary-Reed-Solomon-with-subfield path. `NewBCH(m, t)` constructs (n=2^m−1, k, t)-BCH with generator polynomial g(x) = lcm(M_1(x), M_3(x), ..., M_{2t−1}(x)) where M_i is the minimal polynomial of α^i over GF(2). `Encode/Syndromes/Decode` reuse C12-C14 from `coding/rs/`. `BCHDecodeBM` is byte-for-byte the same routine as RS; the difference is the symbol alphabet (GF(2) vs GF(2^m)). Pre-built fixtures: `BCH63_45_3()` (POCSAG paging), `BCH127_92_5()` (deep-space). Refs: Hocquenghem 1959; Bose-Ray-Chaudhuri 1960.

**C17 — `coding/cyclic/extra.go` ~100 LOC.** `Golay24` (24, 12, 8) extended binary Golay perfect-3-error-correcting code (deep-space Voyager mission), `Golay23` perfect (23, 12, 7), `Hadamard(n)` Sylvester construction, `Walsh(n)` Walsh-Hadamard transform-based decoder for Reed-Muller-1. Refs: Golay 1949; Voyager mission archive.

### Tier 3 — Convolutional / Viterbi / BCJR / Turbo (~840 LOC)

**C18 — `coding/conv/encode.go` ~100 LOC.** `ConvEncoder{K int; Generators [][]uint64; FeedbackPoly uint64}` for rate-1/n binary convolutional codes. `K` = constraint length (typical 3 to 9), `Generators` = polynomial taps (e.g., `[0o171, 0o133]` for industry-standard rate-1/2 K=7 NASA/Voyager). Methods `Encode(input []byte) []byte`, `EncodeStreaming(in <-chan, out chan<-)`. Recursive systematic convolutional (RSC) variant (`FeedbackPoly` non-zero) for turbo-code component encoders. Refs: Elias 1955; Forney 1973.

**C19 — `coding/conv/viterbi.go` ~180 LOC — KEYSTONE.** Hard-decision and soft-decision Viterbi 1967 trellis decoder. State = previous K−1 input bits (so 2^(K−1) trellis states; K=7 ⇒ 64 states). `ViterbiHard(received []byte, code *ConvEncoder) []byte` minimises Hamming distance; `ViterbiSoft(received []float64, code *ConvEncoder) []byte` minimises squared distance (AWGN log-likelihood). Uses Lazar-Forney path-metric difference normalisation to avoid integer overflow. Traceback length = 5K is the textbook rule. **Singular reality moat:** the `klauspost/reedsolomon` Go ecosystem ships zero Viterbi decoders. Refs: Viterbi 1967; Forney 1973.

**C20 — `coding/conv/bcjr.go` ~140 LOC.** BCJR forward-backward algorithm (Bahl-Cocke-Jelinek-Raviv 1974) for soft-output decoding of convolutional codes (MAP/SOVA). Returns log-likelihood ratios `LLR_i = log(P(b_i=0|y) / P(b_i=1|y))` per bit; required for turbo-code iterative decoding (C21) where extrinsic LLRs are passed between component decoders. Log-domain implementation (`max*` operator = max + log(1+exp(−|Δ|))) avoids underflow. Refs: Bahl-Cocke-Jelinek-Raviv 1974.

**C21 — `coding/turbo/turbo.go` ~120 LOC.** Parallel-concatenated turbo encoder (Berrou-Glavieux-Thitimajshima 1993) with two RSC component encoders (C18) separated by a pseudorandom interleaver. Decoder runs alternating BCJR (C20) on the two components, exchanging extrinsic LLRs, for a fixed number of iterations (typically 6-8). `TurboCode{Component *ConvEncoder; Interleaver []int}` with `Encode/Decode(received []float64, iterations int)`. Refs: Berrou-Glavieux-Thitimajshima 1993; Hagenauer-Offer-Papke 1996.

**C22 — `coding/conv/interleaver.go` ~80 LOC.** Pseudorandom interleavers used by turbo codes. `S-Random(N, S)` (Dolinar-Divsalar 1995), `QPP(N, f1, f2)` quadratic-permutation-polynomial used in 3GPP LTE / 5G NR (TS 38.212). Pre-built tables for standard sizes 40, 56, 80, ..., 6144 (LTE). Refs: 3GPP TS 36.212 / 38.212.

**C23 — `coding/conv/codes_zoo.go` ~80 LOC.** Pre-built standard codes catalogue: NASA-Voyager (K=7, [0o171, 0o133]), CDMA2000 K=9, IEEE-802.11a K=7 punctured to rate 2/3 + 3/4, GSM K=5, DVB-S2 inner code. Refs: CCSDS 131.0-B-3; IEEE 802.11; ETSI TS 138.

### Tier 4 — Modern capacity-approaching: LDPC + Polar (~1,220 LOC)

**C24 — `coding/ldpc/parity.go` ~140 LOC.** LDPC parity-check matrix construction. `Gallager(n, j, k)` Gallager-1963 random regular construction. `MacKay(n, λ, ρ)` MacKay-Neal 1996 irregular degree distributions (variable-degree distribution λ(x), check-degree distribution ρ(x)). `ProgressiveEdgeGrowth(n, k, degDist)` PEG (Hu-Eleftheriou-Arnold 2005). `QuasiCyclic(n, k, lift)` 5G NR QC-LDPC (3GPP TS 38.212 base graphs BG1/BG2). Returns sparse `*BitMatrix` (uses C1 BitMatrix). Refs: Gallager 1963; MacKay-Neal 1996; Hu-Eleftheriou-Arnold 2005.

**C25 — `coding/ldpc/decode.go` ~280 LOC — KEYSTONE.** Sum-product belief-propagation decoder (Pearl 1988 / MacKay 1999) and min-sum approximation. `SumProductDecode(llrs []float64, H *BitMatrix, maxIter int) ([]byte, ok bool)` runs message-passing on the Tanner graph; converged when `H · ĉ = 0` (parity-check satisfied). `MinSumDecode` replaces the `tanh / atanh` check-node update with `min` operator (~10× faster, ~0.5 dB worse). `OffsetMinSum(beta float64)` and `NormalisedMinSum(alpha float64)` are the practical 5G implementations. Quantisation modes: `LDPCQuantised(qbits int)` for fixed-point implementations matching hardware decoders. Refs: Gallager 1963; MacKay 1999; Chen-Dholakia-Eleftheriou-Fossorier-Hu 2005.

**C26 — `coding/ldpc/codes_zoo.go` ~80 LOC.** Pre-built LDPC codes: 5G NR base graphs BG1 (rate ≥ 1/3, large block) and BG2 (rate ≥ 1/5, small block) with lift factors Z=2 to Z=384 covering all block sizes from 40 to 8448 (3GPP TS 38.212 §5.3.2). DVB-S2 LDPC codes (n=64800, rates 1/4 to 9/10, ETSI EN 302 307). 802.11n / 802.11ac WLAN LDPC. **Cross-link:** these are the DEPLOYED codes in 5G phones; reality would be the only Go library shipping them.

**C27 — `coding/polar/polar.go` ~280 LOC — SINGULAR FRONTIER.** Polar codes (Arıkan 2009). Channel polarisation construction: given underlying B-DMC `W`, the construction recursively splits N = 2^n channels into "frozen" (bad) and "data" (good) bit positions via Bhattacharyya parameter or Mutual Information evolution (Tal-Vardy 2013 quantisation-based construction). API: `PolarCode{N int; FrozenSet []int; CRC *CRCPolynomial}`; `NewPolar(n, k, designSNR float64)` with `BhattacharyyaConstruction` / `GaussianApproximation` / `TalVardy(maxLevel int)` constructors. Encoder: `Encode(msg)` ⊕ recursive butterfly (FFT-shape, O(N log N)). **Successive-cancellation decoder `SCDecode(received []float64) []byte`** O(N log N), processes f-function `f(L1,L2) = sgn(L1)·sgn(L2)·min(|L1|,|L2|)` (LLR domain) and g-function `g(L1,L2,u) = L2 + (1−2u)·L1`. **SCL decoder `SCLDecode(L int)`** Tal-Vardy 2011 list-L extension keeps L candidate paths through the trellis, using path-metric pruning. **CRC-aided SCL `CASCL(L int, crc *CRCPolynomial)`** is the 5G NR control-channel decoder (3GPP TS 38.212 §5.3.1.2). Refs: Arıkan 2009; Tal-Vardy 2011, 2013; 3GPP TS 38.212.

### Tier 5 — Networking + Channel models (~580 LOC)

**C28 — `coding/fountain/lt.go` + `raptor.go` ~280 LOC.** Luby-Transform fountain code (Luby 2002): each output symbol is XOR of d randomly-selected input symbols where d is drawn from a Robust-Soliton distribution. Decoder: belief-propagation on Tanner graph, succeeds with prob ≥ 1−δ given (1+ε)k output symbols. `RobustSoliton(k int, c, delta float64) []float64`, `IdealSoliton(k int) []float64`. Raptor codes (Shokrollahi 2006) precode with high-rate LDPC outer + LT inner — linear-time encoding, sublinear-time decoding. RaptorQ (RFC 6330) is the IETF standard. **Cross-link:** would let `compression/` add a "lossy-network-tolerant compression" sub-mode. Refs: Luby 2002; Shokrollahi 2006; RFC 6330.

**C29 — `coding/network/rlnc.go` ~120 LOC.** Random Linear Network Coding (Ahlswede-Cai-Li-Yeung 2000; Ho-Médard-Koetter-Karger-Effros-Shi-Leong 2006). Each network node combines incoming packets as random linear combinations over GF(2^m) with random coefficients drawn uniform on GF(2^m). Decoder Gauss-Jordan eliminates the coefficient matrix to recover originals. Useful in distributed storage (Maxprop/MORE/COPE) and decentralised data dissemination. Refs: Ahlswede-Cai-Li-Yeung 2000; Ho et al. 2006.

**C30 — `coding/channel/models.go` ~180 LOC.** Channel-model simulator package. `BSC(p float64)` binary symmetric, `BEC(eps float64)` binary erasure, `AWGN(sigma float64)` real-valued additive white gaussian, `Rayleigh(sigma float64)` flat-fading, `RicianK(k, sigma float64)` LOS-fading. Each implements `Channel` interface with `Transmit(in []byte) (out []float64, llrs []float64)`. `LLRFromHard(received []byte, p float64)` and `LLRFromAWGN(received []float64, sigma float64)` LLR converters. **Capacity formulas:** `CapacityBSC(p)` = 1 − H₂(p), `CapacityBEC(eps)` = 1 − eps, `CapacityAWGN(snr_dB)` = 0.5·log2(1+SNR). **Cutoff rate** R₀ for sequential-decoding bounds (Wozencraft-Reiffen 1961). Refs: Shannon 1948; Cover-Thomas 2006 *Elements of Information Theory* §7-§9.

**C31 — `coding/channel/bounds.go` ~80 LOC.** Code-existence bounds. `SingletonBound(n, k) int` = n−k+1 max possible distance. `HammingBound(n, k, q) int` sphere-packing upper bound on distance. `PlotkinBound(n, q) int` for high-rate codes. `GilbertVarshamovBound(n, k, q) int` lower bound on distance via probabilistic existence. `MDS(n, k, q) bool` reports whether (n, k, d=n−k+1) MDS code can exist over GF(q). Refs: Singleton 1964; Hamming 1950; Plotkin 1960; Gilbert 1952; Varshamov 1957.

---

## 2. LOC budget summary

| Tier | Sub-package | LOC | Cumulative |
|---|---|---:|---:|
| T1 | `coding/galois/` (gf2, gfp, gf2m, poly) | 560 | 560 |
| T1 | `coding/block/` (linear, hamming, reedmuller) | 480 | 1,040 |
| T2 | `coding/cyclic/` (crc, cyclic, extra) | 300 | 1,340 |
| T2 | `coding/rs/` (encode, syndrome, BM, forney, chien, decode) | 580 | 1,920 |
| T2 | `coding/bch/` | 280 | 2,200 |
| T3 | `coding/conv/` (encode, viterbi, bcjr, interleaver, zoo) | 580 | 2,780 |
| T3 | `coding/turbo/` | 120 | 2,900 |
| T4 | `coding/ldpc/` (parity, decode, zoo) | 500 | 3,400 |
| T4 | `coding/polar/` | 280 | 3,680 |
| T5 | `coding/fountain/` (LT, raptor) | 280 | 3,960 |
| T5 | `coding/network/rlnc.go` | 120 | 4,080 |
| T5 | `coding/channel/` (models, bounds) | 260 | 4,340 |
| | TOTAL primitive code | 4,340 | |
| | + Tests (golden-file fixtures, cross-substrate parity, IEEE-754) | ~1,080 | 5,420 |

---

## 3. Connective tissue (LOC into adjacent packages)

| From | To | LOC | What |
|---|---|---:|---|
| `crypto/modular.go` | `coding/galois/gfp.go` | 30 | Re-export `crypto.ModInverse` as `galois.GFP.Inv`. No code duplication. |
| `linalg/` | `coding/galois/gf2.go` | 80 | New `BitMatrix` type. NOT a `Matrix` subtype — the float64 path is incompatible — but reuse the row-echelon-form algorithm shape. |
| `compression/entropy.go` | `coding/channel/bounds.go` | 20 | `H_2(p) = ShannonEntropy([p, 1−p])` re-export for capacity formulas. |
| `crypto/hash.go` | `coding/cyclic/crc.go` | 40 | Cross-register `CRCPolynomial` table so `crypto.HashByName("crc-32")` resolves. |
| `crypto/rng.go` | `coding/channel/models.go` | 30 | Channel models pull from existing PCG/xorshift PRNG; no new PRNG. |
| `compression/coding.go` | `coding/fountain/` | 0 | No coupling; orthogonal source-vs-channel coding split. |
| `info/lz/`, `info/mdl/` | `coding/` | 0 | Source coding (LZ76, MDL) and channel coding (RS/LDPC/polar) are textbook-disjoint. |
| `zkmark/zkmark.go` Tranche 2 | `coding/rs/` | ~150 | FRI commitment uses RS-over-prime-field as the low-degree-test primitive. Halo2-honest-pending → Halo2-RS-FRI swap unblocked once C9-C14 land. |
| **Total connective tissue** | | **~350** | (most is metadata + cross-imports, not new math) |

---

## 4. Highest-leverage one-day sprints

**Sprint S1 (1 day, ~580 LOC) = C1+C2+C3+C5 = `BitVector` + `BitMatrix` + `Galois` GF(2^8) + `LinearBlockCode`.** Unblocks every subsequent primitive. Validates against `H · G^T = 0` for Hamming(7,4), Hamming(15,11), and the (8, 4, 4) extended Hamming. Three cross-substrate witnesses (R-MUTUAL-CROSS-VALIDATION 3/3): Python `numpy.linalg`-mod-2 round-trip, C++ `boost::dynamic_bitset` round-trip, C# `BitArray` byte-equality.

**Sprint S2 (1 day, ~360 LOC) = C10+C11+C12+C13+C14 RS-(255, 223) deep-space CCSDS standard.** Saturates the "single most-deployed code in human history" axis (every Voyager / Cassini / Mars-rover packet decoded with RS-(255, 223)). Cross-validates against the CCSDS reference test vectors (1024 bytes of test corruption, decoder output byte-equal to ground truth). Singular reality moat: zero Go libraries ship golden-file CCSDS parity.

**Sprint S3 (1 day, ~280 LOC) = C19 Viterbi K=7 [0o171, 0o133] NASA Voyager rate-1/2.** Hard + soft. Pairs with C18 encoder + C23 codes_zoo. Cross-validates against the GNU-radio reference at every BER point on AWGN @ Eb/N0 = {1, 2, 3, 4, 5} dB. Singular reality moat: no Go library ships Viterbi at all.

**Sprint S4 (2 days, ~580 LOC) = C24+C25 LDPC sum-product on Gallager-(3,6) regular code at n=1008.** The MacKay-Neal 1996 paper validation point. Cross-validates against the `MacKay-Neal-1996-Fig5` BER curve. Hardest sprint of the list (the message-passing implementation has many subtle log-domain bugs; the Tanner-graph data-structure choice matters for cache behaviour).

**Sprint S5 (2 days, ~520 LOC) = C27 polar SCL decoder for the 5G NR PDCCH (control-channel) configuration.** Singular Block-C-2026 frontier. The 3GPP TS 38.212 §5.3.1.2 fixture pinpoints exact frozen-set for n=8 to n=10. Cross-validates against the 3GPP-published BLER curves at Eb/N0 ∈ {0, 1, 2, 3} dB.

---

## 5. Cross-link audits

**Versus 057-crypto-missing.** Slot-057 scopes lattice-based / elliptic-curve / signature primitives (out of scope here); the overlap is GF(p) arithmetic which slot 057 needs anyway. C2 `coding/galois/gfp.go` could merge with crypto-EC's prime-field arithmetic; recommend a shared `crypto/field/` sub-package. **Coordinate with 057.**

**Versus 200-synergy-zkmark-info.** The zkmark/info synergy slot did not name Reed-Solomon FRI as a deferred Tranche-2 dep, but the Halo2-honest-pending → Halo2-real path absolutely requires RS-over-prime-field as the IOP commitment primitive (FRI = Fast Reed-Solomon IOP Proximity test, Ben-Sasson-Bentov-Horesh-Riabzev 2018). C2 + C9-C13 unblock this. **Slot-200 should be amended with a Reed-Solomon dep mention.**

**Versus 044-compression-api.** The compression-api slot did not flag the source-coding-vs-channel-coding split. CLAUDE.md description "Lossless/lossy compression primitives" is unambiguously source-side, but a user looking for "Hamming code" would naturally check `compression/` first. **Recommendation:** CLAUDE.md package-table line for `compression` should cross-reference a new `coding` package once landed: `compression: source coding (entropy + RLE + delta + Huffman + LZ77). See coding/ for channel coding.`

**Versus 175-synergy-zkmark-crypto.** That slot scopes elliptic-curve pairing + Halo2 backend-real. The shared dep is GF(p) arithmetic and FFT-friendly fields. C9 RS-over-prime is the bridge.

**Versus 089-info-api / 088-info-sota / 087-info-missing.** The info package family is codelength + algorithmic-complexity (Solomonoff / Rissanen / NML) — orthogonal to channel coding. No re-home.

**Versus 060-crypto-perf.** Slot-060 likely flagged the lack of SIMD-vectorised crypto hot paths; a future Reed-Solomon SIMD path (Plank-Greenan-Miller 2013 GF(2^8) PSHUFB) would slot into the same perf-tier.

---

## 6. What's deferred to v2 (calling it now)

- Algebraic-geometry Goppa codes (McEliece cryptosystem variant, Goppa 1981) — niche; the McEliece public-key crypto consumer would pull this, but no current zkmark consumer demands it.
- Spatially-coupled LDPC (Felström-Zigangirov 1999; Kudekar-Richardson-Urbanke 2011) — capacity-achieving on BMS channels but academic; defer until a 5G/6G research consumer lands.
- Non-binary LDPC (Davey-MacKay 1998) over GF(2^m) — better waterfall but ~5× slower decode; v2.
- Lattice codes (Erez-Zamir 2004; Conway-Sloane 1999) for AWGN-channel capacity-approaching with structure — niche; defer.
- Burst-error codes (Fire 1959; Reiger bound) — RS already handles bursts via interleaving.
- Repeat-accumulate codes (Divsalar-Jin-McEliece 1998) — turbo-LDPC hybrid; defer.
- Quantum error-correction codes (Shor 1995, Steane 1996, surface codes) — out of scope (reality has no quantum sub-package).
- McEliece / Niederreiter post-quantum signatures — slot-057 territory.
- Concatenated codes generalised (Forney 1966) — implicit in turbo (C21) and 5G LDPC+CRC; explicit composition layer is v2.

---

## 7. Priority matrix

```
                    HIGH consumer demand                LOW consumer demand
                  ┌────────────────────────────────────┬────────────────────┐
HIGH                │ C1-C5 Galois + BitMatrix          │ C7 Reed-Muller     │
LOC                 │ C9 RS-(255,223)                   │ C29 RLNC           │
                    │ C19 Viterbi                       │ C28 Raptor         │
                    │ C25 LDPC sum-product               │                    │
                    │ C27 Polar SCL                     │                    │
                  ├────────────────────────────────────┼────────────────────┤
LOW                 │ C8 CRC                            │ C17 Golay          │
LOC                 │ C12 Berlekamp-Massey              │ C31 Bounds         │
                    │ C13 Forney                        │                    │
                    │ C30 Channel models (BSC/BEC/AWGN) │                    │
                    │ C16 BCH                            │                    │
                  └────────────────────────────────────┴────────────────────┘
```

**Recommended ship-order (saturated R-pattern coverage):**
1. **Week 1:** Sprint S1 (T1 keystone) — unblocks all.
2. **Week 2:** Sprint S2 (RS deep-space) — singular reality moat.
3. **Week 3:** Sprint S3 (Viterbi NASA) + C30 channel models.
4. **Week 4:** Sprint S4 (LDPC) — 2026-frontier infrastructure.
5. **Week 5:** Sprint S5 (polar SCL) — 5G NR-grade frontier.
6. **Week 6:** C16 BCH + C17 Golay + C8 CRC catalogue + C31 bounds — completes the textbook surface.
7. **Backlog:** C28 fountain + C29 RLNC + C21 turbo (when a streaming/distributed consumer pulls).

Each sprint stands alone. T1 must land first; T2-T5 can land in parallel after.

---

## 8. Closing note — why this slot matters

Coding theory is the **applied** side of Shannon 1948. reality already ships the **pure** side (information theory in `compression/entropy.go`). The Shannon limit is a number; codes are how you reach it. Reality has every prerequisite — finite-field arithmetic in `crypto/`, packed-bit data structures conceptually in `linalg/`, polynomial primitives in `crypto/modular.go`, capacity-formula entropy in `compression/` — and never assembled them. A 5,420-LOC PR (4,340 LOC math + 1,080 LOC tests) closes the **single largest applied-math gap in reality v0.10.0**: every device that talks to a satellite, every QR code, every Wi-Fi packet, every 5G phone, every Blu-ray disc, every NAND-flash chip, every CD, every DVB stream, every mesh network goes through a Reed-Solomon, BCH, LDPC, polar, turbo, or convolutional code. reality cannot be the universal-truth substrate without these.
