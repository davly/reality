# 320 — dive-error-correction (Reed-Solomon / Hamming / BCH / LDPC / CRC / Turbo / Polar audit)

## Headline
reality v0.10.0 ships **zero callable error-correcting-code primitives** — no Hamming code, no Reed-Solomon, no BCH, no CRC, no LDPC/turbo/polar/convolutional/Viterbi; only `sequence/distance.go:115` `HammingDistance` (the *string metric* Hamming codes are designed to maximise) — and slot 210 has already scoped the full 4,340-LOC `coding/` sub-package; this dive narrows that scope to a 6-primitive day-one ECC catalog ranked by deployment ubiquity vs LOC cost.

## Findings

### F1 — repo-wide ECC surface = zero callable hits
- Grep `Reed|BCH|LDPC|polar|turbo|Galois|GF\(|Viterbi|trellis|BCJR|Berlekamp|fountain|CRC` over `**/*.go` returns ZERO callable matches outside reviews/.
- The only `Hamming` symbol is `sequence/distance.go:107-128` `HammingDistance(a, b string) (int, error)` — string metric (positions where a[i] != b[i]), not a code. Reference cites Hamming 1950 *Error Detecting and Error Correcting Codes* but the function does not encode/decode.
- `signal/window.go` has `Hamming` window (spectral) — orthogonal.
- `compression/coding.go:1-103` is **source coding only**: `RunLengthEncode/Decode` + `DeltaEncode/Decode`. No channel coding. CLAUDE.md description "Lossless/lossy compression primitives: entropy, RLE, delta encoding, **Huffman, LZ77**" is drift — Huffman + LZ77 are NOT in `compression/coding.go` (slot-022/044 territory).
- `crypto/modular.go` ships `ModPow / ModInverse / ExtendedGCD / CRT` (uint64 prime field) — building blocks for BCH-syndrome inversion and RS-over-GF(p), never assembled into a coding layer.
- `crypto/hash.go` ships FNV1a32/64 + MurmurHash3 (non-crypto hash, NOT error detection — no algebraic structure, no minimum-distance guarantee).
- No `coding/` directory exists (Glob `**/coding/**/*.go` = no files).

### F2 — slot 210 has the full 28-primitive scope; this dive is the day-1 narrow
Slot 210 (`reviews/overnight-400/agents/210-new-coding-theory.md`) exhaustively scopes a `coding/` sub-package totalling 28 primitives / 5,420 LOC across 11 sub-packages: `galois`, `block`, `cyclic`, `rs`, `bch`, `conv`, `turbo`, `ldpc`, `polar`, `fountain`, `network`, `channel`. That review identifies the keystone (`coding/galois/gf2m.go` GF(2^8) extension field with Exp/Log tables) and the singular reality moats (RS-(255, 223) CCSDS deep-space; Viterbi K=7; polar SCL for 5G NR PDCCH).

This dive does NOT re-scope. This dive **identifies the cheapest day-1 PR** that saturates the textbook-canonical 50% of practical ECC deployments.

### F3 — standard ECC catalog vs reality state
| Family | Primitive | Year | Deployment | reality state |
|---|---|---|---|---|
| Detection | CRC-8/16/32/64 | Peterson-Brown 1961 | Ethernet, USB, SATA, ZIP, modems, IPv4 header | absent |
| Block | Hamming(7,4), Hamming(15,11) | Hamming 1950 | ECC RAM, telecom | absent |
| Block | Extended Golay(24,12) | Golay 1949 | Voyager 1/2 imagery | absent |
| Block | Reed-Muller RM(1, 5) | Muller/Reed 1954 | Mariner-9 | absent |
| Cyclic / Block-MDS | Reed-Solomon RS(255, 223) | Reed-Solomon 1960 | CD/DVD/Blu-ray, QR codes, RAID-6, satellite (CCSDS), DVB | absent |
| Cyclic | BCH | Bose-Ray-Chaudhuri 1959 / Hocquenghem 1959 | POCSAG paging, NAND flash | absent |
| Convolutional | NASA K=7 [171,133]_8 + Viterbi | Elias 1955 / Viterbi 1967 | Voyager, GSM, 802.11a, CDMA | absent |
| Modern | Turbo (parallel-concat RSC + BCJR iter) | Berrou-Glavieux-Thitimajshima 1993 | 3G UMTS, 4G LTE | absent |
| Modern | LDPC + sum-product | Gallager 1962 / MacKay-Neal 1995 | DVB-S2, Wi-Fi 6 (802.11ax), 5G NR data | absent |
| Modern | Polar + SCL | Arıkan 2009 | 5G NR control channels (3GPP TS 38.212) | absent |
| Streaming | LT / Raptor / RaptorQ | Luby 2002 / Shokrollahi 2006 / RFC 6330 | 3GPP MBMS, file-delivery | absent |

11/11 absent. Closest in-repo surface is `crypto/modular.go` ModPow (GF(p) prime field) and the `sequence/distance.go` Hamming string metric — neither is a code.

### F4 — keystone is Galois GF(2^8) (slot 210 C3); without it, RS / BCH / LDPC / polar all stall
Every code from RS onward needs GF(2^m) with Exp/Log tables. `crypto/modular.go` is GF(p) prime-field only — not adequate for byte-symbol RS-(255, 223). The Exp/Log table approach (one Mul = `Exp[(Log[a] + Log[b]) mod 255]`, one Inv = `Exp[255 - Log[a]]`) is ~280 LOC including IrreducibleTable and unblocks RS / BCH / Reed-Muller / Goppa simultaneously. This is THE single dependency. Slot 210 already named it C3 KEYSTONE.

### F5 — CRC is the cheapest day-1 win and ships independently
CRC requires no Galois extension field — only polynomial division over GF(2). 80 LOC for CRC-8 + CRC-16-CCITT + CRC-32 (IEEE 802.3) + CRC-64 (ECMA-182) with the standard reveng catalogue table (Cook 2018). Ships independently of T1 keystone. Validated against `hash/crc32` in Go stdlib (already a `crypto/hash.go` neighbour) and against the standard `123456789` → `0xCBF43926` (CRC-32 IEEE) check vector. **Day-1 PR candidate.**

### F6 — Hamming(7,4) is the textbook entry that pins R-MUTUAL-CROSS-VALIDATION 3/3 trivially
Hamming(7,4) has only 16 message words and 128 codeword candidates. **Direct enumeration table-lookup ground truth** is feasible: every 4-bit input → unique 7-bit codeword → flip each of 7 bits → decoder must recover original 4 bits in all 7×16 = 112 single-bit-error cases. Three witnesses: (a) handwritten G/H matrices, (b) syndrome-table decoder, (c) brute-force minimum-distance decoder — all three must produce byte-identical output for all 112 corrupted-codeword inputs. This is the **cheapest R-3/3 saturation in the entire ECC catalog** (~80 LOC including test).

### F7 — RS(15, 11) over GF(16) is the cheapest cross-validation harness for the BM/Forney/Chien chain
RS(15, 11) over GF(2^4) corrects t=2 errors with 2t=4 parity symbols. Full enumeration of all `C(15,2) = 105` two-error patterns × `15^2 = 225` error magnitudes = 23,625 corrupted codewords per encoded message — all must decode back to the original. Three witnesses: (a) Berlekamp-Massey error-locator + Forney + Chien, (b) extended Euclidean (Sugiyama 1975) error-locator, (c) brute-force "try every 2-error pattern, pick minimum-distance" oracle. All three must agree. Validates the entire RS pipeline at small-N before scaling to RS(255, 223). **R-3/3 in ~250 LOC of test.**

### F8 — RS(255, 223) is the deep-space CCSDS standard and reality's singular ECC moat
The CCSDS 131.0-B-3 telemetry channel-coding standard mandates RS(255, 223) over GF(2^8) with primitive polynomial `0x11D` (same as AES). Every Voyager / Mars-rover / Cassini packet decoded with RS(255, 223). Tolerates 16 byte-errors per 255-byte block (32 parity bytes). Cross-substrate parity vs the CCSDS published reference vectors — no Go library currently ships golden-file CCSDS parity (klauspost/reedsolomon is RAID-6 SIMD-tuned, vivint/infectious is Cauchy-based with no BM path). **Singular reality moat. Day-1 PR candidate alongside CRC.**

### F9 — BCH is RS-with-binary-symbols (same machinery, free re-use)
BCH(n=2^m−1, k, t) over GF(2) with codewords whose generator polynomial roots include {α, α^2, ..., α^{2t}} reuses the **identical BM + Forney + Chien chain** as RS — only the symbol alphabet changes (GF(2) vs GF(2^m)) and a minimal-polynomial table replaces the linear `(x − α^i)` factors. After RS lands, BCH(63, 45, 3) (POCSAG paging) and BCH(127, 92, 5) ship in ~280 LOC mostly fixture data. **Free piggyback on the RS sprint.**

### F10 — CRC-32 is *already in* Go stdlib `hash/crc32` — reality must reimplement-from-first-principles
Per CLAUDE.md design rule 6 ("Reimplement from first principles. Do not wrap existing libraries.") reality's CRC implementation must be table-driven slice-by-8 with explicit polynomial tables, NOT a wrapper over `hash/crc32`. The byte-for-byte cross-substrate parity (Go vs Python `binascii.crc32` vs C `zlib.crc32` vs C# `System.IO.Hashing.Crc32`) is the proof-of-correctness — wrapping stdlib makes that test vacuous. Slot 210 C8 already names this; flag it for the implementer.

### F11 — consumer demand exists in zkmark, storage, comms, fingerprinting
- `zkmark/zkmark.go:280` Halo2 honest-pending placeholder. Slot 200 (synergy-zkmark-info) and slot 175 (synergy-zkmark-crypto) both name FRI as the Tranche-2 dep. **FRI = Fast Reed-Solomon IOP Proximity test (Ben-Sasson-Bentov-Horesh-Riabzev 2018).** RS over a prime field is the IOP commitment primitive. Without `coding/rs/`, zkmark cannot honestly degrade from Halo2-honest-pending to Halo2-real.
- `compression/` source coding has no channel-coding neighbour. A natural next step is `compression/network` mode = source-then-channel composition (Huffman + RS).
- Pistachio renderer (downstream consumer of reality, per CLAUDE.md "Pistachio calls these at 60 FPS") would use CRC-32 for asset-cache integrity checks — currently uses non-cryptographic FNV1a (no error-detection guarantee).
- Sequence fingerprinting (`sequence/`) currently uses Hamming string distance for AcoustID-style audio fingerprints. A 32-bit-block Hamming code surface would let the same package ship error-tolerant fingerprint matching.

### F12 — quoted slot-210 LOC is realistic but day-1 should be ~480 LOC, not 5,420
Slot 210 scopes 28 primitives × 4,340 LOC math + 1,080 LOC tests = 5,420 LOC. That's a 4-5 week sprint. Day-1 narrow = 4 primitives × ~480 LOC math + ~280 LOC tests = ~760 LOC delivers 80% of practical ECC deployment ubiquity (CRC + Hamming + RS-(255, 223) + BCH(63,45,3)). The remaining 20% (LDPC + polar + turbo + Viterbi) is 2026-frontier capacity-approaching territory and is where the LOC budget genuinely lives.

### F13 — IEEE 754 edge cases do not apply (ECC operates over finite fields, not floats)
Per CLAUDE.md design rule "IEEE 754 edge cases mandatory: +Inf, -Inf, NaN, -0.0, subnormals" — but ECC primitives over GF(2)/GF(2^m) operate on integers, not floats. The CLAUDE.md test catalog should add a per-package note: "ECC primitives have **integer** edge cases instead: alphabet-overflow (input symbol >= q), zero-syndrome correctness (no error → identity decode), max-correctable-errors boundary (t exact → recover, t+1 → fail-loud), and erasure-mask consistency."

The float edge cases DO apply to soft-decision Viterbi, BCJR, sum-product (LLR domain) — `+Inf` LLR = certain bit value, `-Inf` = certain other, `NaN` must be rejected. Pin those for T3/T4 LDPC sprints.

## Concrete recommendations

1. **Day-1 PR (~760 LOC) = the narrow ECC catalog.** Ship in this order:
   - **(a) `coding/cyclic/crc.go` ~80 LOC** — CRC-8 / CRC-16-CCITT / CRC-32-IEEE / CRC-64-ECMA. Slice-by-8 table-driven. No GF(2^m) dep. Validated against `123456789` → `0xCBF43926` (CRC-32 IEEE) and the full reveng catalogue (Cook 2018). Reimplement-from-first-principles (do NOT wrap `hash/crc32`).
   - **(b) `coding/galois/gf2m.go` ~280 LOC** — slot-210 C3 KEYSTONE. GF(2^8) with primitive polynomial `0x11D` (AES/RS-CCSDS), Exp/Log tables, `Add/Mul/Inv/Pow`. `IrreducibleTable(m uint8) uint16` for m ∈ [2, 16].
   - **(c) `coding/block/hamming.go` ~80 LOC** — Hamming(7, 4) + Hamming(15, 11) + Hamming-Extended (overall parity). Static G/H matrices. Day-1 R-MUTUAL-CROSS-VALIDATION 3/3 saturates trivially per F6.
   - **(d) `coding/rs/rs.go` ~320 LOC** — RS(255, 223) CCSDS + RS(15, 11) tutorial + RS(15, 9) deep-space short-block. Berlekamp-Massey error-locator + Forney magnitude + Chien search. Erasure-aware variant. Validated against CCSDS reference vectors. Day-1 R-MUTUAL-CROSS-VALIDATION 3/3 per F7.
   
   Stops short of BCH / Viterbi / LDPC / polar / turbo. Day-1 ships the textbook-canonical 80% of deployed ECC.

2. **Week-2 PR (~440 LOC) = BCH + Viterbi.** Free piggyback on day-1 RS machinery (per F9). `coding/bch/bch.go` ~280 LOC with BCH(63, 45, 3) POCSAG and BCH(127, 92, 5) deep-space; `coding/conv/encode.go` + `coding/conv/viterbi.go` ~160 LOC for K=7 [171, 133]_8 NASA-Voyager hard-decision Viterbi. Soft-decision Viterbi adds ~80 LOC and is the LLR-domain entry point for T3/T4.

3. **Week-3-onwards = slot-210 deferred.** LDPC (slot 210 C24-C26), polar (slot 210 C27), turbo (slot 210 C21), fountain/Raptor (slot 210 C28), RLNC (slot 210 C29), channel models (slot 210 C30). Defer to follow-up scoping; do NOT commit calendar to these in day-1 PR.

4. **Cross-link `sequence/distance.go:115` HammingDistance to `coding/block/hamming.go`.** Add a doc-comment redirect: "For the Hamming **code** (not the metric), see `coding/block/hamming.go`. For the Hamming **window** (spectral), see `signal/window.go`." Three different Hamming surfaces in reality is confusing.

5. **Add R-MUTUAL-CROSS-VALIDATION 3/3 pins for day-1 PR.** Pin opportunities (highest leverage):
   - **(P1) Hamming(7, 4) all-16-input × all-7-bit-flip enumeration (112 cases) decoded by 3 independent decoders** (handwritten G/H, syndrome-table lookup, brute-force minimum-distance). All three must agree. ~80 LOC test.
   - **(P2) RS(15, 11) over GF(16) all-23,625-corrupted-codeword enumeration decoded by 3 chains** (BM+Forney+Chien, Sugiyama-extended-Euclidean, brute-force). All three must agree. ~250 LOC test.
   - **(P3) CRC-32 IEEE byte-for-byte against Go `hash/crc32.ChecksumIEEE` for `[]byte("123456789")` → `0xCBF43926`, plus zero-length input → `0x00000000`, plus reveng catalogue `check` field for CRC-{8, 16-CCITT, 16-Modbus, 16-XMODEM, 32, 32-Castagnoli, 64-ECMA-182}. ~80 LOC test.

6. **CLAUDE.md update.** When `coding/` lands, add row to package table: `coding | Channel coding: CRC, Hamming, Reed-Solomon, BCH, Viterbi, LDPC, polar (T1 only at v0.X.0)`. Cross-reference from `compression`: "See coding/ for channel coding (Hamming, Reed-Solomon)." Cross-reference from `sequence`: "HammingDistance is the string metric. For the Hamming code, see coding/block/."

7. **Block ECC IEEE-754 catalog drift fix.** Per F13, ECC primitives over GF(2^m) have integer edge cases not float edge cases. Add to CLAUDE.md design rule 5 ("Precision documented, not assumed") a sub-clause: "For finite-field primitives, document alphabet-size, primitive-polynomial, FCR (first consecutive root), max-correctable t, and zero-syndrome identity-decode behaviour."

8. **Coordinate with slot 290 (galois-theory) + 293 (NTT) + 057 (crypto-missing).** GF(p) prime-field arithmetic is a shared dep; recommend `crypto/field/` shared sub-package per slot-210 §5. Do NOT duplicate `crypto/modular.go` ModInverse logic into `coding/galois/gfp.go`.

9. **Pin the FRI / zkmark dep explicitly.** When day-1 RS lands, slot 200 (synergy-zkmark-info) and slot 147 (zkmark-missing) gain an unblocking primitive. The Halo2-honest-pending placeholder can be replaced with FRI-RS (Ben-Sasson-Bentov-Horesh-Riabzev 2018) using `coding/rs/`. Add a `zkmark/fri.go` follow-up scoping slot.

10. **Defer Goppa / McEliece / quantum codes.** McEliece (Goppa 1981) is a post-quantum signature consumer (slot 057 / 212) not a channel-coding consumer. Surface codes / Shor / Steane are quantum (no quantum sub-package in reality). Spatially-coupled LDPC (Felström-Zigangirov 1999) is academic. Do NOT scope these in day-1 PR.

## Sources

### Repo files (all paths absolute)
- `C:\limitless\foundation\reality\sequence\distance.go:107-128` — `HammingDistance` string metric (Hamming 1950 cited; not a code).
- `C:\limitless\foundation\reality\compression\coding.go:1-103` — RLE + delta source coding only; CLAUDE.md drift on Huffman/LZ77 claims.
- `C:\limitless\foundation\reality\crypto\modular.go` — ModPow / ModInverse / ExtendedGCD / CRT (uint64 GF(p) prime field).
- `C:\limitless\foundation\reality\crypto\hash.go` — FNV1a32/64 + MurmurHash3 (non-crypto hash, NOT error detection).
- `C:\limitless\foundation\reality\signal\window.go` — Hamming window (spectral; orthogonal).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\210-new-coding-theory.md` — Block-C scoping of full coding/ sub-package (28 primitives / 5,420 LOC); this dive is the day-1 narrow on top.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\200-synergy-zkmark-info.md`, `175-synergy-zkmark-crypto.md`, `147-zkmark-missing.md` — FRI / RS-IOP dep on coding/rs/.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\290-new-galois-theory.md`, `293-new-ntt.md`, `057-crypto-missing.md` — shared GF(p) field dep; recommend `crypto/field/` shared sub-package.
- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md:340` — slot 320 line: `dive-error-correction | Error-correcting codes: Reed-Solomon, Hamming, BCH coverage`.
- `C:\limitless\foundation\reality\CLAUDE.md` — design rules cited (esp. rule 6 "Reimplement from first principles") + package table drift on compression Huffman/LZ77.

### Primary literature
- Hamming, R. W. (1950). "Error Detecting and Error Correcting Codes." *Bell System Technical Journal* 29(2):147–160.
- Reed, I. S.; Solomon, G. (1960). "Polynomial Codes over Certain Finite Fields." *J. SIAM* 8(2):300–304.
- Bose, R. C.; Ray-Chaudhuri, D. K. (1960). "On a Class of Error Correcting Binary Group Codes." *Information and Control* 3(1):68–79.
- Hocquenghem, A. (1959). "Codes correcteurs d'erreurs." *Chiffres* 2:147–156.
- Berlekamp, E. R. (1968). *Algebraic Coding Theory*. McGraw-Hill.
- Massey, J. L. (1969). "Shift-Register Synthesis and BCH Decoding." *IEEE Trans. Inf. Theory* 15(1):122–127.
- Forney, G. D. (1965). "On Decoding BCH Codes." *IEEE Trans. Inf. Theory* 11(4):549–557.
- Forney, G. D. (1973). "The Viterbi Algorithm." *Proc. IEEE* 61(3):268–278.
- Viterbi, A. J. (1967). "Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm." *IEEE Trans. Inf. Theory* 13(2):260–269.
- Bahl, L.; Cocke, J.; Jelinek, F.; Raviv, J. (1974). "Optimal Decoding of Linear Codes for Minimizing Symbol Error Rate." *IEEE Trans. Inf. Theory* 20(2):284–287.
- Berrou, C.; Glavieux, A.; Thitimajshima, P. (1993). "Near Shannon Limit Error-Correcting Coding and Decoding: Turbo-codes." *Proc. ICC '93*.
- Gallager, R. G. (1962). "Low-Density Parity-Check Codes." *IRE Trans. Inf. Theory* 8(1):21–28.
- MacKay, D. J. C.; Neal, R. M. (1995). "Good Codes Based on Very Sparse Matrices." *Cryptography and Coding*.
- Arıkan, E. (2009). "Channel Polarization: A Method for Constructing Capacity-Achieving Codes for Symmetric Binary-Input Memoryless Channels." *IEEE Trans. Inf. Theory* 55(7):3051–3073.
- Tal, I.; Vardy, A. (2011). "List Decoding of Polar Codes." *Proc. ISIT 2011*.
- Luby, M. (2002). "LT Codes." *Proc. FOCS 2002*.
- Shokrollahi, A. (2006). "Raptor Codes." *IEEE Trans. Inf. Theory* 52(6):2551–2567. RFC 6330.
- Peterson, W. W.; Brown, D. T. (1961). "Cyclic Codes for Error Detection." *Proc. IRE* 49(1):228–235.
- Singleton, R. C. (1964). "Maximum Distance q-ary Codes." *IEEE Trans. Inf. Theory* 10(2):116–118.
- Ben-Sasson, E.; Bentov, I.; Horesh, Y.; Riabzev, M. (2018). "Fast Reed-Solomon Interactive Oracle Proofs of Proximity." *ICALP 2018*.
- Cook, G. (2018). reveng CRC catalogue. https://reveng.sourceforge.io/crc-catalogue/.
- Lin, S.; Costello, D. J. (2004). *Error Control Coding* (2nd ed.). Pearson. — single most-cited textbook in the field.
- Richardson, T.; Urbanke, R. (2008). *Modern Coding Theory*. Cambridge UP. — modern LDPC reference.
- MacWilliams, F. J.; Sloane, N. J. A. (1977). *The Theory of Error-Correcting Codes*. North-Holland.
- Cover, T. M.; Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley. — capacity formulas.
- CCSDS 131.0-B-3 (2017). *TM Synchronization and Channel Coding*. — RS(255, 223) deep-space standard.
- 3GPP TS 38.212 v17.x. *NR; Multiplexing and channel coding*. — 5G NR LDPC + polar specs.
- ETSI EN 302 307 (2014). *DVB-S2*. — DVB LDPC codes.
- ISO/IEC 18004 (2015). *QR Code 2005*. — RS over GF(256).
- IEEE 802.3-2018. — CRC-32 Ethernet polynomial.
