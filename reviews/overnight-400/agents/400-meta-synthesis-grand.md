# 400 — Grand Synthesis: Top 30 Highest-Leverage Findings

## Headline

Across 397 prior reviews, the single most consequential finding is that **`reality`'s "four-language golden-file validation" — the design decision the README treats as the moat — does not actually exist yet**: zero non-Go sources, golden values stored as decimal literals not raw bits, oracles that re-use the same libm they're meant to validate, and 30+ growth packages with no goldens at all. Fix that and the rest of the work (live correctness bugs in `crypto/prime.go`, `prob/mathutil.go`, `optim/gradient.go`, `orbital/orbital.go`, `queue/erlang.go`, alloc-clean hot paths, doc/CLAUDE.md drift, and a tightly-bounded list of ~10 highest-yield new packages — Krylov, special functions, FDTD/PDE, Kalman, IIR design, attention, forward-mode AD, Bayesian-opt, sketches, NTT) is mechanical. The library is mathematically honest where it ships; it is over-claimed in CLAUDE.md/README and structurally under-validated.

## Methodology

- Read MASTER_PLAN line 400 and confirmed the assignment.
- Read PROGRESS.md (542 lines, 1.3 MB) by streaming the 4th/5th headline field for slots 1-399. Found 5 explicit FAILED slots (035, 152, 168 retried, others retry-OK), all other ~395 slots completed.
- Skimmed ~40 highest-stakes files in full: 306 (prime bug), 116 (chi-squared bug), 101 (LBFGS bug), 106 (Kepler bug), 121 (ErlangB bug), 126 (sequence defects), 081 (graph numerics), 041 (compression doc lie), 386 (alloc audit), 382 (test coverage), 394 (cross-language), 360 (CR oracle), 389 (doc cohesion), 398 (streaming), 399 (domain coverage), 311 (Krylov), 300 (Bessel), 220-F/335/336 (optim modern), 244 (PDE), 308 (Kalman), 315/317/337 (signal extensions), 328 (forward AD), 350 (numfmt), 393 (frame conventions), 376 (PRNG modern), 388 (thread safety), 351-356 (research libs).
- Headline-grepped the remaining ~360 slots; clustered by repeated motifs (alloc, golden gap, doc drift, IEEE specials, missing primitives) and by package.
- Ranked by leverage = (# independent slot votes × user impact) / engineering cost. Tie-breaks favor: live-correctness bugs > foundational infrastructure (oracle / fuzz / PBT) > consumer-pull additions (Pistachio/aicore) > breadth additions.

---

## Top 30 (ranked by leverage)

### 1. Fix `crypto/prime.go:IsPrime` — 7-witness set is wrong by 10 orders of magnitude  [HIGH/HIGH leverage, S effort, 2 votes]
- The 7-base Miller-Rabin set is deterministic only to 3.4×10^14, not the documented 3.3×10^24. For ~99.99996% of `uint64`, `IsPrime` is currently **probabilistic with no error bound** — and the smallest counterexample (341,550,071,728,321) is not in the test suite. Fix is 5 LOC: replace witness list with the 12-base Sorenson-Webster set, which is deterministic across all of `uint64`. Add the regression test for the known SPRP. Also rename/repair `MillerRabin(n,k)` (docstring says probabilistic, code is deterministic; `grad` parameter unused in LBFGS is a sibling pattern).
- Slots: 306, 304.

### 2. Fix `prob/mathutil.go:regularizedGammaLowerSeries` — series-only, no continued-fraction branch  [HIGH/HIGH leverage, S effort, 2 votes]
- Numerical Recipes §6.2 / DLMF 8.9: the series converges only for `x < a+1`; outside this regime catastrophic cancellation contaminates the right tail. Three production paths affected: `chiSquaredCDF` (so `ChiSquaredTest` p-values are wrong precisely where they matter most — large χ²), `GammaCDF`, `PoissonCDF`. Fix: add `regularizedGammaUpperCF(a,x)` via Lentz on DLMF 8.9.2 and dispatch by `x < a+1`. Pattern already correctly used in `RegularizedBetaInc` symmetry-flip.
- Slots: 116, 117.

### 3. Adopt CORE-MATH (MIT) as canonical golden-file oracle + encode floats as raw `uint64` bits in JSON  [HIGH/MED leverage, M effort, 6 votes]
- Today's "Go is canonical" policy means goldens inherit Go's libm wobble (≤1 ulp vs the real value); cross-language validators using their own ≤1 ulp libm can disagree by up to 2 ulp — a class of bugs the current decimal-literal JSON cannot even surface. Switch generator to CORE-MATH (MIT-licensed, Inria, ≤0.5 ulp guaranteed) for transcendentals, mpmath+Arb-CR-check for special functions; emit a `bits` field with the raw `uint64` IEEE bits per fixture (Go `math.Float64bits`, Py `struct.pack('<d',…)`, C++ `std::bit_cast`, C# `BitConverter.DoubleToUInt64Bits`). This single change closes an entire class of cross-language drift bugs.
- Slots: 360, 394, 359, 377, 350, 397.

### 4. Actually ship Python/C++/C# validators (the README claim)  [HIGH/MED leverage, L effort, 3 votes]
- Repo has zero non-Go sources today. Stand up `validators/{python,cpp,csharp}/` reading the same JSON goldens, asserting bit-equality on the `bits` field for IEEE-754-class functions and ulp-tolerance against CORE-MATH for transcendentals. Wire one CI matrix cell per (language × OS). Until this lands, every "validates against golden files" sentence in CLAUDE.md is aspirational.
- Slots: 394, 360, 359.

### 5. Land `pgregory.net/rapid` PBT + 30 `FuzzN` targets (one PR)  [HIGH/MED leverage, S-M effort, 4 votes]
- Reality has 0 `FuzzN`, 0 PBT libraries across 97 test files. Highest-yield targets: round-trip pairs (`compression.{Huffman,RLE,Delta,LZ77}`, `info/lz76`, `info/mdl`, `forge.basispoints`, `forge.fnv1a`); numeric invariants (`signal.FFT`/`IFFT` Parseval, `linalg.{LU,QR,Cholesky}` reconstruct, `geometry.QuaternionSlerp` unit-norm); algorithmic equivalences (`graph.Dijkstra` ≡ `BellmanFord` on non-negative weights). Properties: `prob` distributions ∫pdf=1, CDF monotone, F⁻¹(F(x))≈x. PBT/Fuzz are test-only deps and do not violate the zero-runtime-deps invariant.
- Slots: 382, 383, 384.

### 6. Update CLAUDE.md to ground truth (22→41 packages, 1965→real test count, citation style, conventions)  [HIGH/HIGH leverage, S effort, 5 votes]
- Quick Reference still says "Packages (22), Tests: 1,965"; reality has 41 top-level dirs, ~37 production packages, 97 test files, ~2,400 test funcs. README/ARCHITECTURE/CLAUDE all advertise Huffman + LZ77 in `compression/` — neither exists. Add: a one-page style guide (Reference vs Source; Greek vs ASCII; citation format), `Frame:` tag policy (Hamilton [w,x,y,z] + RH but only `geometry/` documents it), `Thread safety:` doc clause for stateful PRNGs/PID/Tape, deprecate `constants/units.go` (zero production callers — `color/` reimplements deg2rad locally).
- Slots: 041, 382, 389, 388, 391, 393, 399.

### 7. Backfill golden vectors to CLAUDE.md targets (20 min, 30 target) + IEEE-754 edge-case suite per function  [HIGH/MED leverage, M effort, 3 votes]
- 636 golden cases across 80 files; median ~7-10/file vs documented "minimum 20, target 30". Three out of 80 files meet target. Many at 3-5 (`em/coulomb_force`, `acoustics/decibel_spl`, `prob/ema`, `fluids/darcy_weisbach`, `orbital/{hohmann,hill}`). 25% of test files reference `NaN/Inf/MaxFloat`; CLAUDE.md mandates this for every function. Subnormals: zero hits in any test. Add `golden_lint.go` CI gate that fails when any JSON has <20 cases; AST lint that every exported `func` has a `Test<Name>_IEEE` or explicit waiver.
- Slots: 382, 303, 302.

### 8. Add `linalg/sum.go` (Kahan/Neumaier/pairwise/KahanDot) and adopt at top-10 hot loops  [HIGH/MED leverage, S effort, 3 votes]
- Only Welford in `audio/fingerprint.go` today. ~30 naive accumulators, 4 with documented "exact for IEEE 754" claims that are factually wrong (slot 303). Highest-leverage replacement sites: linalg dot/Frobenius, signal FFT magnitude sums, prob entropy/KL accumulators, infogeo Bregman, autodiff backward grad-sum. ~250 LOC. Reject xsum/ReproBLAS as over-spec for v1.
- Slots: 302, 349, 377.

### 9. Fix the LBFGS line search in `optim/gradient.go` — claims Wolfe, implements only Armijo (and silently returns step=2^-40)  [HIGH/MED leverage, S effort, 1 vote, but blast radius is large]
- `gradient.go:62` doc says "Wolfe"; code is Armijo backtracking. The `grad` parameter is captured but never used (`_ = grad`) — proof. No descent-direction guard; if `dg ≥ 0` the test becomes non-decrease, line search can return a step that *increases* `f`. After 40 shrinks returns 2^-40 with no failure flag → ghost convergence. Add curvature condition or strong Wolfe; descent guard `if dg ≥ -tiny: d = -g`; emit failure result. Also fix `InteriorPoint` (not actually primal-dual IPM) and `SimplexMethod` latent OOB tie-break.
- Slots: 101, 163.

### 10. Fix `orbital/TrueAnomalyFromMean` for e→1 (Kepler equation) + add input guards  [HIGH/MED leverage, S effort, 1 vote]
- Newton denominator `1 - e·cos(E)` collapses at periapsis when e→1, returning NaN/±Inf. No guard for e≥1, e<0, NaN. Silent non-convergence on `maxIter` exhaustion (no error/flag/counter — caller cannot distinguish converged from oscillating). Fixes: Conway-Laguerre order-5 iteration (no zero-denominator pathology), Danby's "best initial guess", separate hyperbolic (Kepler-H) and Barker parabolic branches, return error on non-convergence. ~80 LOC.
- Slots: 106, 342.

### 11. Fix `queue/erlang.go:ErlangB` (inverted Jagerman direction → +Inf) + `queue/MM1K` overflow path  [HIGH/MED leverage, S effort, 1 vote]
- ErlangB recursion silently underflows to 0 / blows to +Inf for non-pathological argument regions (small A, large N + small A, far-overload A≪N + large N). Jagerman direction is inverted. MM1K overflows to NaN for ρ>1 (`ρ^(K+1)` → +Inf, `1−Inf=−Inf`, then NaN). Splits at ρ=1 use 1e-12 threshold — too tight, lose 3 sig digits at K=100 / ρ=1+1e-9. Also: 5 defects in `sequence/` (JaroWinkler 1±ulp, Soundex first-letter unconditional, NW traceback non-determinism, Hamming int-overflow on large strings, empty-string NaN/Inf — slot 126).
- Slots: 121, 126.

### 12. Day-1 PR: Krylov solver pack (`linalg/krylov` — CG + GMRES(m) + BiCGStab + Jacobi)  [HIGH/MED leverage, M effort, 5 votes]
- Reality has zero Krylov solvers — only dense direct LU/Cholesky/QR. Blocks slots 244 (PDE), 247 (mortar-FEM), 248 (multigrid), 249 (domain-decomp), 102 (optim KKT). Day-1 ~580 LOC ships CG+GMRES(m=30)+BiCGStab+Jacobi preconditioner; defers FGMRES/LGMRES/deflated/IDR(s). Reuses `MatVecMul` (already zero-alloc). Single biggest unlock for the entire PDE/sim cluster.
- Slots: 311, 097, 244, 248, 249.

### 13. Day-1 PR: special functions (`prob/special` — Bessel J/Y/I/K + spherical + elliptic K/E + Lambert W)  [HIGH/MED leverage, M effort, 4 votes]
- Reality ships zero callable Bessel/spherical-harmonic/elliptic/hypergeometric. Blocks `acoustics` cylindrical-room modes, `em` waveguide cutoffs, `orbital` perturbation theory, Pistachio ambisonic spherical harmonics. T0 (Cephes 8-fn) + T1 (Miller downward integer order) + T2 (spherical closed-form) + T3 (Hankel) ~520 LOC. First MIT pure-Go zero-dep DLMF coverage (chapters 5/10/12/13/14/15/16/19/22/23/25/27/28/31/32/33). Pair with `pFq + 1F1 + 0F1 + Carlson + ortho-polys` (slot 298, 299) for ~440 LOC additional → unblocks 5 distributions in slot 117.
- Slots: 300, 298, 299, 399.

### 14. Day-1 PR: Kalman filter family (Joseph-form first, then UD, then SR-UKF, then info-form/PF)  [HIGH/MED leverage, M effort, 3 votes]
- Reality v0.10.0 ships zero Kalman of any flavor. The audit pivots from "harden existing" to "specify the *first* implementation correctly so it never has to be rewritten." Joseph-form linear KF as v0.11.0 entry, UD (Bierman-Thornton) for v0.12.0 hot-path, SR-UKF v0.13.0. 3-way mutual cross-validation pin (naive ≡ Joseph ≡ UD on Linear-Gaussian benchmark) and steady-state ≡ DARE regression. Composable with `signal/pll.go` (PLL is a degenerate Kalman) and `chaos/RK4Step`.
- Slots: 308, 309, 319.

### 15. Day-1 PR: signal-pack catch-up (windows + IIR design + Bluestein + resampler + Hilbert)  [HIGH/MED leverage, M effort, 5 votes]
- Reality ships 3/14 standard windows (Hann, Hamming, Blackman); add Kaiser+Tukey+Gaussian+Welch+Bartlett+Nuttall+BH4+FlatTop one-file ~280 LOC. Zero IIR design (no bilinear/proto/biquad) — ship `signal/iir.go` w/ prewarp+Butter+DF-IIt+RBJ ~480 LOC. Cooley-Tukey panics on non-pow-2 — Bluestein day-1 ~150 LOC composes existing FFT. Zero SRC machinery — Day-1 ship polyphase+sinc ~350 LOC. No Hilbert — Marple FFT path ~140 LOC. Together unblock aicore/Pistachio audio EQ, room equalisation, control discretisation, PLL/Costas.
- Slots: 315, 316, 317, 318, 337, 338, 319.

### 16. Day-1 PR: forward-mode AD (dual numbers) + Hessian/Jacobian helpers + checkpointing  [HIGH/MED leverage, S effort, 2 votes]
- `autodiff/` is reverse-mode-only tape. Forward-mode is the truly streaming variant — naturally fits Pistachio's per-sample 60 FPS gradient where reverse-mode tape allocation is too costly. ~300 LOC dual + ~80 LOC Checkpoint wrapper + ~250 LOC Griewank revolve. Unblocks ESS sensitivity (single-input large-output) in aicore. Pair with `optim/transport` adjoint for OT gradient.
- Slots: 328, 329, 330, 331, 398, 163.

### 17. Day-1 PR: attention/transformer kernels (`attention/` + Adam optimizer)  [HIGH/MED leverage, S-M effort, 2 votes]
- aicore today has to route around reality for transformer inference (no scaled-dot-product, no masked-softmax, no RMSNorm, no RoPE, no GeLU/SiLU/Swish). CLAUDE.md says "be a foundational math library for AI/audio/graphics/control/sim apps" — currently the AI side is the weakest cell. ~400 LOC `attention/` + `optim/sgd.go`+`adam.go`+`adamw.go`+`lion.go` ~250 LOC. Closes a documented consumer-pull gap.
- Slots: 399, 220, 361.

### 18. Eliminate per-call allocations on the 4 named "60 FPS offenders": `chaos/RK4Step`, `audio/spectrogram`, `graph/Dijkstra`, `autodiff/Backward`  [HIGH/MED leverage, S-M effort, 1 vote, named per-file]
- `chaos/ode.go:38-42` allocates 5 slices/call in RK4Step (Pistachio MPPI 60 FPS path); add buffered `RK4StepInto(state, k1, k2, k3, k4 []float64)`. `audio/spectrogram/stft.go:93` allocates per-frame `make([]complex128, frameSize)` in a real-time path; ship `STFTInto`. `graph/shortest.go:31-87` Dijkstra/A* allocate `dist/prev/gScore/cameFrom/inClosed` per call; provide `Workspace` reuse pattern. `autodiff/tape.go:72` allocates `grads := make([]float64, len(t.nodes))` per `Backward`; pool. Pattern is already proven in `optim/proximal/operators.go` (all `Into` variants) and `signal/window.go`.
- Slots: 386, 145 (topology perf), 080 (geometry perf), 250+ "buffered API" pattern.

### 19. Day-1 PR: streaming/sketch package (`sketch/` — Welford-merge + Frequent-Directions + t-digest + HLL + KLL)  [HIGH/MED leverage, M effort, 4 votes]
- 6/22 packages stream; 16/22 are batch-only. Welford+merge already proven in `audio/fingerprint.go` (Chan-Golub-LeVeque) — promote to a generic substrate. Frequent-Directions sketch unblocks streaming PCA/SVD (slot 261). t-digest + KLL for streaming quantiles. HyperLogLog for cardinality. ~600 LOC. Closes the streaming-vs-batch axis without redesigning batch APIs.
- Slots: 398, 224, 261, 388, 302.

### 20. Day-1 PR: NTT + Montgomery + Tonelli-Shanks + CRT (`crypto/modular/`)  [HIGH/MED leverage, S-M effort, 4 votes]
- Zero advanced-modular surface today. Ship Montgomery+Barrett+Tonelli-Shanks+Garner-CRT day-1 (~540 LOC) and NTT next (~320 LOC keystone for slot 211 PQ + slot 293 ECC + slot 290 Galois). Also fix `crypto.mulmod` (currently Russian-peasant O(log n)) → `bits.Mul64+bits.Div64` mulmod for 10× IsPrime speedup. `crypto.MillerRabin` 7-base witness fix (item #1 above) drops here too.
- Slots: 291, 293, 211, 060, 306.

### 21. Day-1 PR: Bayesian optimization + CMA-ES (closes 5 prior deferrals — 102/169/222/227/237)  [HIGH/MED leverage, S-M effort, 2 votes]
- Zero BO surface. ~520 LOC: 4 acquisition fns (EI + UCB + PI + Thompson) + ~280 LOC Matern GP backbone. Closes 5 prior deferrals with one shared math primitive. CMA-ES: zero in repo; aCMA-ES + IPOP restart together ~700 LOC (marginal cost of restart over base loop is ~120 LOC). Together close the derivative-free global-optimization gap and provide aicore a hyperparameter-tuning primitive.
- Slots: 336, 335, 102, 169, 222, 227, 237.

### 22. Day-1 PR: PDE-solver entry (`physics/pde` — FDTD-1D wave + heat-eq generic + Crank-Nicolson)  [HIGH/MED leverage, M effort, 4 votes]
- Reality has only `physics/thermo.go` 1-D heat-eq one-step. No FEM/FDM/FVM/spectral. Blocks any "real" `acoustics` room sim (vs Sabine algebraic), any `em` full-wave, any `fluids` CFD beyond Bernoulli, `physics` heat conduction beyond `HeatEquation1DStep`. Even a 200-LOC 2-D FDTD wave-equation core unlocks 3 packages simultaneously. Closest peer: `gosl` via FFI; reality has a clean shot at first-class pure-Go FDTD with golden-file determinism.
- Slots: 244, 250, 247, 248, 399.

### 23. Day-1 PR: rotation + Lie-group package (`geometry/rotation` — Quat ↔ Euler ↔ DCM ↔ Log/Exp ↔ MRP + SO(3)/SE(3))  [MED/MED leverage, S effort, 3 votes]
- Quaternion-only kernel today; missing `q→Euler`, `q↔DCM`, `Log`/`Exp`, MRP. Day-1 ~150 LOC adds Shepperd extraction + gimbal-aware Euler + Slerp/NLerp/Squad pin. Pair with AHRS (Mahony+Madgwick ~330 LOC) to unblock Pistachio attitude estimation. Lie-group `Exp/Log` for SO(3)/SE(3) is the bridge to slots 205/206 (Riemannian opt). Robotics/IMU consumers want this.
- Slots: 313, 314, 341, 205, 206, 393.

### 24. Day-1 PR: Chebyshev + Pade + DD/QD + numfmt boundary (`numfmt/`, `approx/`)  [MED/MED leverage, M effort, 3 votes]
- Zero Pade/CF surface (T0 Pade-from-Taylor ~120 LOC reuses LUSolve). Zero Chebyshev (T0 DCT-coef + T1 Clenshaw + T2 Lobatto barycentric ~250 LOC; unblocks slot 316 Parks-McClellan). Zero DD/QD/EFT (T0+T1 Dekker/TwoSum/DD ~350 LOC unblocks 7 slots). Add `numfmt/` package with BF16/FP16/FP8 RNE conversion (~250 LOC) for ML interop — fp64 stays canonical, numfmt is boundary only. Reject fp32 surface, generics-over-Float, boxed `Float` interface.
- Slots: 345, 346, 347, 350, 397.

### 25. Move stateful types to thread-safe contracts (PRNGs, PID, autodiff Tape)  [MED/MED leverage, S effort, 2 votes]
- No mutexes in math packages. Stateful types (`crypto.MersenneTwister`, `PCG`, `Xoshiro256`; `control.PIDController`; `autodiff.Tape`) need explicit "Thread safety: not concurrent-safe; one-per-goroutine or external mutex" doc clause. Also: 16+ consumer files bypass `crypto/` PRNGs and use Go-stdlib `math/rand` (which silently breaks bit-identical cross-language reproducibility — Go≠Py≠C++≠C#). Either adopt LXM/Philox/ChaCha20 with seedable cross-lang vectors (slot 376) and require its use, or document the bypass.
- Slots: 388, 304, 376.

### 26. Day-1 PR: ECC/coding-theory pack (CRC + GF(2^8) + Hamming + RS(255,223) CCSDS)  [MED/MED leverage, S-M effort, 2 votes]
- Zero ECC primitives (no Hamming/RS/BCH/CRC/LDPC). Day-1 PR ~760 LOC = CRC + GF(2^8) + Hamming(7,4) + RS(255,223). GF(2^m) absent in v0.10.0 — T0+T1+T2 (~600 LOC) unblocks AES/RS/BCH/ECC-binary/Binius STARK. Pairs with NTT (item #20) and Galois (slot 290) for the algebraic-foundations cluster.
- Slots: 320, 321, 210, 290.

### 27. Day-1 PR: Merkle + h2c + Poseidon + MSM (zkmark prerequisites)  [MED/MED leverage, M effort, 4 votes]
- Zero Merkle/commitment surface — T0+T1+T2 binary Merkle ~310 LOC RFC6962 day-1 atop SHA-256. Zero hash-to-curve — T0 expand_xmd+T1 h2f ~160 LOC. Zero algebraic-hash — T0 Sponge + T2 Poseidon-BN254 ~330 LOC unblocks zkmark Fiat-Shamir/Merkle. Zero MSM — T0 wNAF + T1 Straus + T2 Pippenger ~520 LOC = 30-60× ZK prover speedup keystone. Together turn `zkmark/` from primitive into useful.
- Slots: 326, 323, 325, 324, 322, 292.

### 28. Tighten 4 named API inconsistencies: TF Validate, gametheory game-shape, info bits-vs-nats, sequence rune-policy  [MED/MED leverage, S effort, 4 votes]
- `gametheory` uses 6 mutually inconsistent argument shapes for "describe a game" — pick one (slot 074). `compression/` is bits, `infogeo/` is nats, `prob/` is mixed — pick a default per package or expose `Bits`/`Nats` suffixes (slot 089). `sequence/` silently does `[]rune(s)` with zero docstring acknowledgement; mixes rune-window+byte-hash in Shingling (NFC/NFD-unsafe) (slot 129). `control.TransferFunction` cannot be discretised; add bilinear-transform path (slot 052/315). Also adopt one Result struct shape across optim solvers (currently 6 different shapes for the same concept — slot 104).
- Slots: 074, 089, 129, 052, 104, 075.

### 29. Doc-cohesion + style guide (Reference vs Source, Greek vs ASCII, Frame: tag, Thread safety: tag, godoc Examples)  [MED/LOW leverage, S effort, 5 votes]
- Strong dominant template (Formula/Range/Precision/Reference) but: Reference vs Source mixed; citations in 5+ shapes; Greek vs ASCII per-author; 0 `func Example*` doctests. Greek glyphs in 53 files (368 occurrences). Citation coverage ~70-80% of public functions cite (CLAUDE.md mandate not fully met). Mostly consistent on Distance/Distribution/PDF/CDF naming, but: British vs American split, Mean vs Average, acronym case (URL vs Url), maxIter outlier. One-page style guide; 30-50 worked Examples covering canonical use of each package.
- Slots: 389, 390, 393, 388, 392.

### 30. Build/CI hardening: TinyGo block-list, GOAMD64 policy, FMA fusion policy, coverage telemetry  [MED/LOW leverage, S effort, 4 votes]
- All 10 tested GOOS/GOARCH targets compile clean; only `conduit/` (net/http) and `audio/spectrogram`+`color/visualise` (image/png) block TinyGo. Add `//go:build !tinygo` or split. Go 1.25 silent FMA fusion at GOAMD64=v3 (single rounding via VFMADD231SD) silently changes float results vs Go 1.24/v1 / Python / C++ default — pick a per-package policy (`math.FMA` explicit one-rounding OR `float64(a*b)+c` to defeat fusion) and document. No GOAMD64-split CI matrix today. Wire `go test -coverprofile`; fail PRs that drop pkg coverage below floor (suggested: 85% math, 70% glue).
- Slots: 396, 380, 394, 382.

---

## Phase plan (5 phases)

### Phase 1 — Quick wins, week 1 (correctness + truth-in-advertising)
- **#1** Fix `IsPrime` 7-base → 12-base witnesses (5 LOC + regression test).
- **#2** Add `regularizedGammaUpperCF` continued-fraction branch.
- **#9** Fix LBFGS Wolfe/Armijo doc + descent guard + failure flag.
- **#10** Fix `TrueAnomalyFromMean` Laguerre + hyperbolic branch + input guards.
- **#11** Fix `ErlangB` Jagerman direction + `MM1K` overflow + 5 sequence defects.
- **#6** CLAUDE.md/README/ARCHITECTURE doc-truth pass (22→41 packages; remove Huffman/LZ77 lies; add Frame:/Thread-safety: clauses).

### Phase 2 — Foundation, month 1 (validation infrastructure)
- **#3** CORE-MATH oracle + raw-bits JSON encoding.
- **#4** Python/C++/C# validators wired into CI.
- **#5** `pgregory.net/rapid` PBT + 30 `FuzzN` targets.
- **#7** Backfill goldens to 20-min/30-target + IEEE-754 edge-case suite per fn.
- **#8** `linalg/sum.go` (Kahan/Neumaier/KahanDot) at top-10 hot loops.
- **#25** Thread-safety doc clauses + cross-lang PRNG policy.
- **#30** TinyGo split + GOAMD64/FMA policy + coverage CI.

### Phase 3 — Foundation, month 2 (highest-leverage missing math)
- **#12** Krylov solver pack (CG + GMRES(m) + BiCGStab + Jacobi).
- **#13** Special functions (Bessel + spherical + elliptic + Lambert W).
- **#14** Kalman filter family (Joseph → UD → SR-UKF).
- **#15** Signal-pack catch-up (windows + IIR + Bluestein + resampler + Hilbert).
- **#19** `sketch/` package (Welford-merge + FD + t-digest + HLL + KLL).

### Phase 4 — Consumer-pull additions, month 3 (aicore + Pistachio unblocks)
- **#16** Forward-mode AD + Hessian/Jacobian + checkpointing.
- **#17** Attention/transformer kernels + Adam optimizer.
- **#18** Eliminate alloc on RK4Step + STFT + Dijkstra + Backward.
- **#21** Bayesian opt + CMA-ES.
- **#23** Rotation + Lie-group + AHRS.

### Phase 5 — Algebraic/cryptography depth, quarter 2
- **#20** NTT + Montgomery + Tonelli-Shanks + CRT.
- **#22** PDE entry (FDTD-1D wave + Crank-Nicolson).
- **#24** Chebyshev + Pade + DD/QD + numfmt.
- **#26** ECC pack (CRC + GF(2^8) + Hamming + RS).
- **#27** zkmark prereqs (Merkle + h2c + Poseidon + MSM).
- **#28** API inconsistency pass (gametheory/info/sequence/optim).
- **#29** Style guide + 30-50 godoc Examples.

---

## Themes that didn't make top-30

- **Galois theory / algebraic number theory deep** (slot 290, 294, 295, 296, 297). Beautiful math, low immediate consumer pull. Land after #20 NTT.
- **Topological data analysis depth** (Reeb / Mapper / zigzag / persistence-stats — slots 286-289). `topology/persistent` is functional today; depth gap is real but not consumer-blocked.
- **Optimal transport extensions beyond Sinkhorn** (slot 201). Sinkhorn ships today; Wasserstein-2 / OT-flow / unbalanced-OT are nice-to-have.
- **SDE / SPDE / Fokker-Planck / mean-field-games** (slots 202, 219, 242, 243). Frontier; defer until a consumer surfaces.
- **Tensor networks / TT decomposition** (slot 203, 257). Same — frontier; defer.
- **Bayesian nonparametrics / DP/CRP/PY/HDP** (slot 228). Defer.
- **Lattice/PQ/isogeny crypto** (slots 211, 212, 213). Land after item #20 NTT and item #26 ECC; PQ is in the strategic 2027+ window.
- **Graph-signal-processing / spectral clustering / manifold learning** (slots 270-272). Mostly redundant (slot 273 verdict); pick one and ship.
- **Causal inference / FDR / conformal completion / robust stats / EVT** (slots 229-233). Statistics depth additions; meaningful but smaller blast radius than the top-30.
- **`optim/intopt` ILP/B&B/Gomory + metaheuristics + matroid** (slots 277-279, 285). Rich combinatorial-optimization tier; defer until a consumer needs MIP.
- **`temporal-graphs` / `hypergraphs` / `simplicial-complexes` / `cubical-complexes` / `discrete-Morse`** (slots 281-285). Topological/graph extensions; queue behind item #19 sketches.
- **Riemannian opt + diff-geo + exterior-calc + geometric-algebra** (slots 205-209). Bridge math; defer until robotics consumer (Vocala?) materializes.
- **Compressed-sensing / RMT / free-prob / rough-paths / functional-data / RKHS** (slots 215-218, 235, 236). Theory-heavy; low immediate consumer pull.
- **MCMC / particle-MCMC / SMC / ABC / VI / normalizing-flows / diffusion-models** (slots 238, 265-267, 239-241). Inference engines; large surface; defer until a consumer (likely Pistachio probabilistic-model needs) surfaces.
- **ANN / hyperbolic-embed / spectral-embedding** (slots 225-226, 273). Embedding work; defer until aicore vector-search consumer surfaces.
- **`orbital/` perturbation / SGP4 / Lambert-Izzo extensions** (slots 343, 344). Slot 399 explicitly recommends "freeze, don't extend" `orbital/` until a consumer appears.
- **`em/` and `fluids/` algebraic deepening**. Same diagnosis — over-served vs consumer pull.
- **`gametheory/` cooperative-game depth (Shapley/Banzhaf)**. Same diagnosis.

## What was confirmed strong

- **Zero-deps invariant intact** (slot 395): no go.sum/vendor, no third-party requires, no CGO, all 200 transitive deps are stdlib.
- **All 10 GOOS/GOARCH targets compile clean** (slot 396).
- **7 STRONG MSC cells** (slot 399): probability/copula/conformal (60-62), graph algorithms (68), info-theory/info-geometry (94), audio-DSP, color science, dynamical systems (37), game-theory/OR (90). Each is a defensible moat vs gonum and the gosl/gorgonia/mgl archipelago.
- **Dominant doc template is good** (slot 389): Formula/Range/Precision/Reference. Just inconsistent in labels and citation shape.
- **In-place buffered API pattern is proven** (slot 386): `optim/proximal/operators.go`, `signal/window.go`, `linalg/matrix.go`, `chaos/systems.go` derivative closures, `audio/separation/{wiener,spectral_subtraction}` `…Into` variants. Pattern exists; just needs to be propagated to the 4 named hot-path offenders.
- **Reverse-mode AD tape is correct** in the interior (slot 011); copula-gradient pin at commit `365368a` saturates R-CLOSED-FORM-PIN. Forward-mode is the missing complement, not a fix.
- **Cooley-Tukey FFT is zero-alloc, in-place, correct** (slot 301). Twiddle drift dominates roundoff at ~1e-10 for N=1024 — tighten with twiddle-LUT cache and add Parseval/Plancherel test.
- **PRNGs are sound implementations** (slot 304): MT19937-64, PCG-XSH-RR, xoshiro256** all correct and seedable. Gap is acceptance-gating (TestU01/PractRand/NIST 800-22), not algorithm bugs.
- **CIEDE2000 saturates Sharma-2005 reference** at 4.95e-5 (slot 031). Color science is one of the few subsystems with a published-paper accuracy floor and reality meets it.
- **3-detector audio onset cross-validation** saturates R-MUTUAL-CROSS-VALIDATION 3/3 (commit `6a55bb4`). The cross-validation discipline that should propagate everywhere works here.

---

## Sources

- `C:\limitless\foundation\reality\reviews\overnight-400\MASTER_PLAN.md` — slot definitions
- `C:\limitless\foundation\reality\reviews\overnight-400\PROGRESS.md` — 400 headline lines (1.3 MB)
- `C:\limitless\foundation\reality\CLAUDE.md` — package list (stale; says 22, actual 41)
- Slot 306 — `crypto/prime.go` IsPrime 7-witness bug
- Slot 116 — `prob/mathutil.go` regularizedGammaLowerSeries series-only bug
- Slot 101 — `optim/gradient.go` LBFGS line search bug
- Slot 106 — `orbital/orbital.go` TrueAnomalyFromMean Kepler-equation defects
- Slot 121 — `queue/erlang.go` ErlangB Jagerman direction inverted + MM1K overflow
- Slot 126 — `sequence/` 5 numerical defects
- Slot 081 — `graph/` 6 numerical liabilities + 0 golden vectors
- Slot 041 — `compression/` doc lie (Huffman/LZ77 advertised, not present)
- Slot 304 — PRNG audit (sound impls, no acceptance gates, no CSPRNG)
- Slot 360 — research-validation: CORE-MATH MIT oracle + hex-bits JSON
- Slot 394 — meta-cross-language: zero non-Go sources, decimal JSON drift
- Slot 382 — meta-test-coverage: 0 FuzzN, 0 PBT, 636 vectors / 80 files
- Slot 383 — meta-property-tests: 0 PBTs across 100+ test files
- Slot 384 — meta-fuzzing: ~75 stdlib FuzzN candidates, 0 today
- Slot 386 — meta-allocation-discipline: ~108 alloc sites; 4 named 60-FPS offenders
- Slot 388 — meta-thread-safety: no mutexes; PRNG/PID/Tape need doc clause
- Slot 389 — meta-doc-cohesion: 593 reference markers / 676 public fns
- Slot 393 — meta-frame-conventions: Hamilton + RH everywhere, only `geometry/` documents it
- Slot 396 — meta-build-targets: 10/10 OS/arch clean; TinyGo blocked by 3 files
- Slot 397 — meta-precision-modes: keep fp64-canonical; ratify slot-350 numfmt boundary
- Slot 398 — meta-streaming-vs-batch: 6/22 packages stream; sketch substrate ready
- Slot 399 — meta-domain-coverage: 7 STRONG MSC cells; top gaps special-fns, PDE, forward-AD, attention, RL
- Slot 311 — Krylov solver day-1 pack
- Slot 300 — Bessel/spherical day-1 pack
- Slot 244 — PDE solvers entry
- Slot 308 — Kalman square-root family
- Slot 315 — IIR design (bilinear + Butter + RBJ)
- Slot 317 — windows catch-up (11 missing)
- Slot 337 — Bluestein/CZT FFT
- Slot 318 — resampling
- Slot 338 — Hilbert transform
- Slot 328 — forward-mode AD
- Slot 336 — Bayesian opt acquisition functions
- Slot 335 — CMA-ES + IPOP
- Slot 291 — modular arithmetic (Montgomery/Barrett/Tonelli-Shanks/CRT)
- Slot 293 — NTT
- Slot 320 — ECC pack (CRC + GF + Hamming + RS)
- Slot 321 — GF(2^m) finite field
- Slot 326 — Merkle trees
- Slot 323 — hash-to-curve
- Slot 325 — Poseidon
- Slot 324 — MSM (Pippenger)
- Slot 313 — rotation representations
- Slot 314 — AHRS (Mahony+Madgwick)
- Slot 302 — stable-sums (Kahan/Neumaier/KahanDot)
- Slot 303 — relative-error-bounds (Precision doc lies)
- Slot 350 — fp-precision-modes (numfmt)
- Slot 376 — modern PRNG survey (LXM/Philox/ChaCha20)
- Slot 380 — Go math extras + GOAMD64 + FMA fusion policy
- Slot 357 — research-libs-go (gonum coverage matrix; reality's USP)
