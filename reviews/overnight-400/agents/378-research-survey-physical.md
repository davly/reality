# 378 — research-survey-physical (metrology updates 2025-2026)

## Headline
The 2019 SI defining constants are stable and reality encodes them exactly; the live metrology fronts for 2026 are the second-redefinition roadmap (CCTF), leap-second phase-out by 2035, and Coordinated Lunar Time (LTC) — none of which require constant-value changes, but several imply new time-scale APIs reality currently lacks.

## Survey

### 1. SI 2019 redefinition — five-year retrospective
Since 20 May 2019 the kg, A, K, mol have been defined by exact values of {h, e, k, N_A}. The Metrologia review (open-access, BIPM) and NIST SP 330-2019 confirm that practical-impact uncertainties for NMI calibration customers were broadly unaffected; the change is structural (universality) not numerical. Reality's `constants/physics.go` already pins the four exact values: Planck `6.62607015e-34`, e `1.602176634e-19`, k `1.380649e-23`, N_A `6.02214076e23`. No update needed. The 2019 redefinition also fixed Δν_Cs (9 192 631 770 Hz) and c (299 792 458 m/s) which remain exact; reality has c but does NOT have Δν_Cs as a named constant — gap.

### 2. CGPM-28 (Versailles, 13–15 Oct 2026) agenda
The 28th CGPM convenes Oct 2026 at Palais des Congrès de Versailles. Per the convocation: future definition of the second, UTC continuity (post-leap-second mechanism), Coordinated Lunar Time and its UTC-traceability, digital metrology transformation. Per CIPM Strategy 2030+, the 28th CGPM will validate (not enact) a roadmap to redefine the second by 2030 if consensus emerges in 2026; otherwise slip to CGPM-29 (2030) with possible enactment 2034. No defining-constant changes are scheduled for 2026.

### 3. Second-redefinition roadmap (CCTF / Resolution 5 of CGPM-27)
CGPM-27 (2022) Res 5 endorsed the CCTF roadmap. Mandatory criteria: ≥10–100× immediate accuracy gain over Cs fountains, continuity with the Cs definition, continuity of TAI/UTC, NMI consensus. Two-step process: present preferred species/ensemble at CGPM-28 (2026, earliest), ratify at CGPM-29 (2030, earliest). Three live candidates: Sr-87 lattice (429.228 THz), Yb-171 lattice (518.296 THz / 642.121 THz E3), Al-27+ ion (1.121 PHz). An "ensemble" definition (weighted set of optical transitions) is also under discussion per Lodewyck 2019.

### 4. Optical clock state-of-the-art 2025
arXiv 2512.21428 (NIST/JILA, Jan–Mar 2025): nine simultaneous Al+/Yb/Sr ratio comparisons with total fractional uncertainty ≤ 3.2×10^-18 — meets the redefinition milestone. arXiv 2509.13991: Sr lattice systematic uncertainty <1×10^-18. NIST Jul 2025: Al+ trapped-ion clock at ~19-decimal-place systematic uncertainty (41% improvement on prior record). Implication for reality: when redefinition lands, the Cs hyperfine line ceases to be the SI-defining transition; an Δν_Cs constant should be introduced now (still exact post-redefinition for continuity) plus optical-clock frequencies as named constants once enacted.

### 5. Leap-second phase-out (CGPM-27 Res 4)
CGPM-27 Resolution 4 (18 Nov 2022) directs that |UT1 − UTC| be allowed to grow to a "much larger value" no later than 2035; ITU WRC-23 (Dubai, Dec 2023) formally recognized Res 4. After 2035, leap seconds disappear for ~100 yr; TAI−UTC becomes a fixed offset (currently 37 s since 2017). IERS Bulletin (Jan 2026): NO leap second 30 Jun 2026; Earth's rotation has been accelerating since ~2020, raising the prospect of a first-ever negative leap second before 2035 — a known liability for code that assumes ΔUT ≥ 0. Reality has `SecondsPerDay = 86400` (exact, mean solar day) and no leap-second table — appropriate for a math library, but ephemeris / astronomy callers downstream will need explicit guidance.

### 6. Coordinated Lunar Time (LTC) — new in 2026
White House OSTP directive (Apr 2024) tasks NASA + international partners to deliver LTC by end-2026. Required properties: traceability to UTC, navigation-grade accuracy, resilience to Earth-link loss, scalability beyond cislunar. arXiv 2507.21597 (BIPM Time Dept) lays out the relativistic lunar reference frame; a lunar surface clock runs ≈ +56 µs/day relative to TT due to the gravitational potential difference (Earth ≈ −60.2 km²/s² vs Moon ≈ −2.82 km²/s² incl. orbit). CGPM-28 will likely receive a draft resolution. Reality currently has neither TT, TCG, TCB, nor LT scales — a future `time/relativistic` package (or extension to `orbital`) is the natural home.

### 7. CCM kilogram dissemination (2024-2025)
CCM.M-K8.2024 (10 participants: Kibble balances at BIPM, LNE, METAS, NIST, NRC, UME; joule balance at NIM; XRCD ²⁸Si spheres at CMS/ITRI, NMIJ, PTB) produced the third "consensus value" of the kg post-2019. The consensus value is the international dissemination basis until each NMI's own realization is sufficiently mature. NIST delivered a tabletop Kibble balance to the U.S. Army (2024); KBTM-2025 met at BIPM 18–20 Nov 2025. No numerical changes — the kg is still defined exactly by h. Reality has no "kg consensus value" concept (correct: it's a measurement-services artifact, not a constant).

### 8. Quantum electrical triangle (volt/ohm/ampere, NIST Aug 2025)
NIST integrated a quantum anomalous Hall resistor (QAHR) and programmable Josephson voltage standard (PJVS) in a single cryostat — the first all-in-one realization of V, Ω, A consistent with SI-2019. Output 0.24–6.5 mV at 3 µV/V; co-located QAHR at zero field, ~1 µΩ/Ω. Significance: QAHE eliminates the multi-tesla magnet of conventional QHE, making quantum standards portable. Defining constants underlying these standards are exact: Josephson constant K_J = 2e/h = 483597.8484... GHz/V, von Klitzing R_K = h/e² = 25812.80745... Ω. Reality currently lacks `JosephsonConstant` and `VonKlitzingConstant` as named constants — gap (both are rationals in {h, e}, easy to add).

### 9. IERS Bulletin / ΔT / Earth orientation
IERS Bulletin A (rapid EOP) and B (final) continue weekly/monthly cadence in 2026. ΔT = TT − UT1 ≈ 69.4 s (early 2026, projected) and growing slowly; the rotation-rate acceleration since 2020 has flattened ΔT growth. No leap second for Jun 2026; Dec 2026 decision pending. Reality has no Earth-rotation/ΔT polynomial (Espenak-Meeus or Stephenson-Morrison). For an `orbital` or `time` extension, the standard reference is IERS Conventions 2010 (still current; no 2025 update yet).

### 10. CIPM Strategy 2030+ themes (digital SI, traceability)
CIPM-2025 Session-III documents emphasize digital-SI (machine-readable units/uncertainties, JSON-LD), the QUDT/UCUM convergence, and a global Digital Calibration Certificate format. None of this changes constant values, but it argues for reality's per-function metadata (provenance, valid range, precision) to align with the digital-SI vocabulary if/when it stabilizes — see slot 367 (CODATA) and slot 379 (uncertainty propagation).

## Reality positioning

- **No numerical changes required** to `constants/physics.go` for 2025-2026. The four 2019 defining constants are exact and stable; CCM consensus value of kg and CCM.M-K8.2024 are dissemination, not redefinition.
- **Add Δν_Cs (9 192 631 770 Hz)** as a named exact constant — currently missing despite being one of the seven SI defining constants. Even post-optical-redefinition it remains exact for continuity.
- **Add JosephsonConstant `2*e/h`** (= 483 597 848 416 983.6... Hz/V) and **VonKlitzingConstant `h/(e*e)`** (= 25 812.807 459 304 5... Ω). Both are exact rationals in {h, e} after 2019.
- **Add `c` already present** — confirm and document that the metre is defined via c (exact) and via the Cs hyperfine transition (i.e. c·Δν_Cs gives a length scale).
- **Time scales**: reality is a pure-math library and correctly avoids leap-second tables. But document in package doc that `SecondsPerDay = 86400` is mean solar (UT1-flavoured) and that TAI/UTC differ by 37 s in 2026, growing to a fixed (but-then-frozen) offset post-2035.
- **Future `time/relativistic`** package candidate (out of current scope but flagged): TT−TAI = 32.184 s exactly, TCG−TT and TCB−TCG rates, ΔT polynomials, lunar-surface frequency offset (+56 µs/day vs TT). Cross-link slot 368 (IAU frames) and slot 380 (ephemerides).
- **Cross-link slot 367** (CODATA 2022 audit): all 2019-defined exact constants unchanged in CODATA-2022; only measured constants (G, m_e, α^-1) shifted within their previous uncertainties. The audit and this slot are complementary — slot 367 covers values, this slot covers definitions.

## Sources
- BIPM, [Metrologia review of the 2019 SI revision](https://www.bipm.org/en/-/2019-review-metrologia-si)
- BIPM, [FAQ on redefinition of the second](https://www.bipm.org/en/faq-redefinition-second)
- BIPM, [28th meeting of the CGPM (2026) — Versailles](https://www.bipm.org/en/cgpm-2026)
- BIPM, [CIPM Strategy 2030+](https://www.bipm.org/documents/20126/258863197/CIPM-Strategy-2030.pdf/)
- CGPM-27, [Resolution 4 (leap second, 2022)](https://www.bipm.org/en/cgpm-2022/resolution-4) and [Resolution 5 (CCTF roadmap, 2022)](https://www.bipm.org/en/cgpm-2022/resolution-5)
- Dimarcq et al., [Roadmap towards the redefinition of the second (Metrologia 2024)](https://iopscience.iop.org/article/10.1088/1681-7575/ad17d2)
- BIPM CCTF, [Task Force Roadmap](https://www.bipm.org/documents/20126/52354657/CCTF_22_TF_A_20200930.pdf/fa223f43-2fea-9cf9-f594-2ad3b2718f5a)
- arXiv [2512.21428](https://arxiv.org/abs/2512.21428) — Al+/Yb/Sr ratio uncertainty ≤3.2×10⁻¹⁸
- arXiv 2509.13991 — Sr clock systematic <1×10⁻¹⁸
- NIST, [All-in-one V/Ω/A quantum standard (Aug 2025)](https://www.nist.gov/news-events/news/2025/08/all-one-nist-develops-single-device-realize-electrical-standards)
- BIPM, [CCM.M-K8.2024 final report](https://www.bipm.org/documents/20126/48150799/CCM.M-K8.2024.pdf/94c74f06-9cbc-0299-03a5-1ffefe8dc88f)
- NIST, [Kibble balance Mass revolution (Jun 2025)](https://www.nist.gov/news-events/news/2025/06/nists-kibble-balance-mass-revolution)
- arXiv [2507.21597](https://arxiv.org/html/2507.21597v1) — Lunar Reference Timescale (BIPM)
- ESA, [Telling time on the Moon](https://www.esa.int/Applications/Satellite_navigation/Telling_time_on_the_Moon)
- IERS, [Bulletins A and B](https://www.iers.org/IERS/EN/Publications/Bulletins/bulletins.html)
- USNO, [Leap second announcement portal](https://maia.usno.navy.mil/products/leap-second)
- Royal Institute of Navigation, [Leap seconds to be phased out by 2035](https://rin.org.uk/news/624222/Leap-Seconds-To-Be-Phased-Out-By-2035.htm)
