## 384 — meta-fuzzing (FuzzN targets per package)

## Headline
Reality has 0 `func Fuzz*` targets across all 22+ packages (verified via grep); this catalog proposes ~75 Go-1.18-native fuzz targets focused on round-trips, panic-freedom, and ill-conditioned-input survival, executable as `go test -fuzz=Xxx -fuzztime=30s` per package with seed corpora drawn directly from the existing golden-vector JSON.

## Context
- Slot 382 confirmed: `grep -rn "func Fuzz" reality/` returns no Go matches.
- Slot 383 enumerated PBT invariants (universal `∀x` properties). Fuzzing is the *operational* counterpart: `f.Add(seed)` + `f.Fuzz(func(...))` lets the runtime mutate seeds and persist crashing inputs to `testdata/fuzz/FuzzXxx/`. Same invariants, different driver.
- Go 1.18+ stdlib fuzzing — no third-party dep needed (matches reality's zero-dep policy; `_test.go`-only anyway).
- Three classes of bug fuzzing finds that goldens cannot:
  1. **Panics on adversarial input** (NaN propagation through unguarded sqrt, divide-by-zero in `m.At(i,j)/d`, slice OOB on `len(x)<2`).
  2. **Round-trip drift** beyond stated tolerance (encode→decode, FFT→IFFT, color cspace A→B→A).
  3. **Catastrophic numerical loss** (cancellation in `a-b`, overflow in `Factorial(21)`, denormal flush).

## Per-package fuzz targets

### linalg (5)
1. `FuzzInverseRoundTrip` — random n×n (n≤8) → invert → multiply → ‖A·A⁻¹ − I‖_F ≤ κ(A)·N·ε; skip if `det≈0`.
2. `FuzzCholeskySPD` — generate `A = LLᵀ` from random L, lower-triangular; require `Cholesky(A)` recovers L within rel-tol; corrupt one entry → expect typed error not panic.
3. `FuzzQRReconstruction` — `Q,R := QR(A)`; assert `‖QR − A‖ ≤ ε·‖A‖` and `‖QᵀQ − I‖ ≤ ε`.
4. `FuzzSVDOrdering` — feed arbitrary m×n bytes-as-floats → `σ_i ≥ σ_{i+1} ≥ 0`; no NaN σ for finite input.
5. `FuzzShapeMismatch` — fuzz `(rows, cols, len(data))`; functions must return `ErrShape` not panic when `len(data) ≠ rows·cols`.

```go
func FuzzInverseRoundTrip(f *testing.F) {
    f.Add(uint8(3), []byte{1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0})
    f.Fuzz(func(t *testing.T, n8 uint8, raw []byte) {
        n := int(n8%6) + 2; if len(raw) < n*n*8 { t.Skip() }
        A := bytesToMat(raw, n)
        Ainv, err := linalg.Inverse(A); if err != nil { return }
        I := linalg.MatMul(A, Ainv)
        if frobNormDiffI(I) > 1e-8*float64(n) { t.Fatalf("‖AA⁻¹−I‖=%v", I) }
    })
}
```

### prob (4)
1. `FuzzNormalCDFMonotone` — for fuzzed `(μ, σ, x₁, x₂)`: `x₁<x₂ ⇒ CDF(x₁) ≤ CDF(x₂)`; CDF ∈ [0,1] always.
2. `FuzzQuantileInverse` — `CDF(Quantile(p)) ≈ p` for `p ∈ (1e-9, 1−1e-9)`, all 12 distributions.
3. `FuzzPDFNonNegative` — fuzz `(params, x)` including `x = ±Inf, NaN, ±MaxFloat64`; PDF must be ≥0 or `NaN` (never negative).
4. `FuzzExtremeParams` — Beta(α=1e-300, β=1e300), Gamma(k=1e15), Student-t(df=0.5); no panic, return either valid float or NaN.

### signal (4)
1. `FuzzFFTRoundTrip` — random complex128 of fuzzed length N ∈ {0,1,2,…,2¹⁶}; `IFFT(FFT(x))` − x ≤ N·ε. (N=0,1 must not panic.)
2. `FuzzFFTNonPowerOfTwo` — N=3,5,7,…: must produce valid output or typed error, not silent garbage.
3. `FuzzWindowUnity` — fuzz N; Hann/Hamming/Blackman: all `wₙ ∈ [0,1]`, finite.
4. `FuzzFilterStability` — fuzz IIR coeffs; assert no NaN in 1024-sample step response (or detect & report unstable).

### compression (4)
> Slot 320 noted Huffman/LZ77 absence; fuzz the **present** primitives first.
1. `FuzzRLERoundTrip` — `Decode(Encode(b)) == b` for any `[]byte`; never panic, never grow >2× on unencodable input.
2. `FuzzDeltaEncoding` — `DeltaDecode(DeltaEncode(xs)) == xs` for `[]int64` including overflow boundaries (MinInt64, MaxInt64).
3. `FuzzEntropyShannon` — fuzzed prob vector (must sum to 1 within tol); 0 ≤ H ≤ log₂(n); H(uniform) = log₂(n) within ε.
4. `FuzzHuffmanLZ77` *(once added)* — round-trip on arbitrary `[]byte` including all-zero, all-FF, single-byte.

### crypto (3)
1. `FuzzMillerRabinKnownPrimes` — seed with primes up to 1e6; fuzz random uint64; if Miller-Rabin says composite, check `n mod p == 0` for some small p; assert MR(n=0,1,2,3)=correct.
2. `FuzzModExpEdge` — `ModExp(base, exp, mod)`: fuzz including `mod=0` (typed error), `mod=1` (return 0), `exp=0` (return 1 for `base≠0`), `MaxUint64` operands.
3. `FuzzHashDeterminism` — same input twice → same output; differs ≥ 1 bit on 1-bit-flip with high prob (avalanche heuristic, soft-fail).

### combinatorics (3)
1. `FuzzFactorialOverflow` — fuzz `n ∈ [0, 25]`; require typed error/ok-flag for `n ≥ 21` (uint64 overflows at 21!); never silently wrap.
2. `FuzzBinomialSymmetry` — `C(n,k) == C(n,n-k)` for fuzzed `(n,k)`; `C(n,0)=C(n,n)=1`.
3. `FuzzCatalanRecurrence` — `C_n = Σ C_i·C_{n-1-i}` for fuzzed `n ≤ 20`.

### geometry (4)
1. `FuzzQuaternionUnitAfterNormalize` — fuzz 4 floats incl ±0, denorm, ±Inf; `Normalize(q)` must give ‖q‖=1 or typed error for zero/nan input.
2. `FuzzQuaternionMulInverse` — `q · q⁻¹ ≈ identity` for non-zero q; `q · 0` must error not panic.
3. `FuzzSDFTriangleInequality` — for fuzzed point + sphere SDF: `|sdf(p₁) − sdf(p₂)| ≤ ‖p₁ − p₂‖` (1-Lipschitz).
4. `FuzzConvexHullClosure` — fuzz 2D points; `Hull(pts)` is itself convex (every triangle of consecutive vertices same orientation).

### chaos (3)
1. `FuzzRK4Energy` — random Hamiltonian system, fuzz initial condition; energy drift over 1000 steps bounded; no NaN.
2. `FuzzLorenzBoundedness` — Lorenz with classical params: |x|,|y|,|z| < 100 after warmup, ∀ initial in unit ball.
3. `FuzzStiffSystemFallback` — feed van-der-Pol with fuzzed μ ∈ [0, 1e6]; integrator must not silently NaN — should error or pass (validate stiffness handling).

### info (3)
1. `FuzzShannonEntropyBounds` — fuzz prob vector `p` (post-normalization): `0 ≤ H(p) ≤ log₂(n)`; H(δ)=0; H(uniform)=log₂(n).
2. `FuzzKLDivergenceNonNeg` — fuzz `(p,q)` vectors; `D_KL(p‖q) ≥ 0`; `D_KL(p‖p)=0`; `D_KL(p‖q)=∞` when `q_i=0, p_i>0`.
3. `FuzzMutualInfoSymmetry` — `I(X;Y) = I(Y;X)` for fuzzed joint distribution.

### color (3)
1. `FuzzRGBLabRoundTrip` — fuzz RGB ∈ [0,1]³; `Lab→RGB→Lab` ΔE_2000 ≤ 1e-9.
2. `FuzzCIEDE2000Symmetry` — `ΔE(c₁,c₂) = ΔE(c₂,c₁)` always; ≥ 0; = 0 iff c₁=c₂.
3. `FuzzWCAGContrastBounds` — for fuzzed RGB pair: contrast ratio ∈ [1, 21], symmetric.

### orbital (3)
1. `FuzzKeplerSolverConvergence` — fuzz `(M, e)` with `e ∈ [0, 1.5]` (incl hyperbolic); must converge or return typed `ErrEccentricity` for `e<0` or `e=1` (parabolic singular).
2. `FuzzOrbitalEnergyConservation` — propagate Keplerian orbit forward and back fuzzed Δt; specific energy `v²/2 − μ/r` constant within 1e-9.
3. `FuzzHohmannSanity` — fuzz (r₁, r₂) > 0; required Δv > 0; transfer time = π√(a³/μ).

### topology (2)
1. `FuzzVRPersistenceMonotone` — fuzz N≤30 random 2D points; persistence diagram has all `birth ≤ death`; no NaN; degenerate (all same point) handled without panic.
2. `FuzzBottleneckMetric` — `d(D,D)=0`, `d(D₁,D₂)=d(D₂,D₁)`, `d(D₁,D₂) ≤ d(D₁,D₃) + d(D₃,D₂)`.

### sequence (4)
1. `FuzzLevenshteinTriangle` — `d(a,c) ≤ d(a,b) + d(b,c)` for fuzzed UTF-8 strings (incl invalid UTF-8 byte sequences, 0-length, 4-byte runes).
2. `FuzzJaroWinklerBounds` — score ∈ [0,1]; reflexive; symmetric for fuzzed strings.
3. `FuzzNGramDiceCommutative` — `Dice(a,b) = Dice(b,a)`; ∈ [0,1]; Dice(a,a)=1.
4. `FuzzSoundexLength` — output is exactly 4 chars for any ASCII input; non-letter input handled gracefully.

### audio (4)
1. `FuzzSpectrogramSilence` — input all-zero of fuzzed length: spectrogram all zeros; no NaN; no panic at len < frame_size.
2. `FuzzSpectrogramClipped` — input ±1.0 saturated; magnitudes finite; no overflow.
3. `FuzzSpectrogramNaNRejection` — input contains NaN: typed error or sanitized output, never NaN-poisoned MFCC bank.
4. `FuzzMFCCStability` — fuzz signal; MFCC coeffs finite; first coeff (DC/log-energy) monotone-ish with input RMS.

### acoustics (2)
1. `FuzzSpeedOfSoundMonotoneT` — fuzz temperature ∈ [-273.15, 1000]°C: `c(T)` monotone increasing; reject `T < -273.15`.
2. `FuzzDopplerSign` — receding source: `f_observed < f_emitted`; approaching: `>`; never returns negative frequency.

### calculus (2)
1. `FuzzSimpsonExactPoly` — Simpson on cubic poly: result exact within rel-tol (Simpson is exact ≤ degree 3). Fuzz coefficients & bounds.
2. `FuzzRK4OrderConvergence` — y'=−y, y(0)=1: error halves to 16× when h halves (4th order). Fuzz initial condition magnitude.

### control (2)
1. `FuzzPIDBoundedOutput` — fuzz (Kp,Ki,Kd, setpoint, measurement); with anti-windup, output stays in `[u_min,u_max]`.
2. `FuzzBodeNoNaN` — transfer function from fuzzed coeffs: magnitude/phase finite at all frequencies; reject zero-denominator.

### em (2)
1. `FuzzCoulombSymmetry` — `F(q1,q2,r) = F(q2,q1,r)`; opposite signs ⇒ attractive; same ⇒ repulsive; `r=0` errors.
2. `FuzzSeriesParallelDuality` — `parallel(R, R) = R/2`; `series(R, R) = 2R`; resistance ≥ 0 for all positive inputs.

### fluids (2)
1. `FuzzReynoldsScaling` — `Re(2v) = 2·Re(v)`; `Re(v=0)=0`; never negative.
2. `FuzzBernoulliConservation` — for fuzzed (v, h, ρ): total head conserved across two stations; reject ρ ≤ 0.

### gametheory (2)
1. `FuzzShapleyEfficiency` — Σ φ_i = v(N) for fuzzed coalition value functions (≤ 6 players to keep 2ⁿ tractable).
2. `FuzzNashBestResponse` — every Nash strategy is a best response (no unilateral deviation profitable) within ε.

### graph (2)
1. `FuzzDijkstraNonNeg` — fuzz random adjacency with non-negative weights; `dist[s] = 0`; triangle: `dist[v] ≤ dist[u] + w(u,v)`.
2. `FuzzTopoSortAcyclic` — fuzz DAGs (random DAG via permutation+upper-tri); every edge `u→v` has `pos(u) < pos(v)`; cycle ⇒ typed error.

### optim (2)
1. `FuzzBisectionBracket` — fuzz monotone function with sign change in `[a,b]`: returns root; `f(a)·f(b) ≥ 0` ⇒ typed error not infinite loop.
2. `FuzzLBFGSDescent` — on convex quadratic with random PD Hessian: terminate within N≤2n iterations; final ‖∇‖ ≤ tol.

### queue (2)
1. `FuzzMM1UtilizationDomain` — fuzz (λ, μ); ρ=λ/μ; ρ≥1 ⇒ typed error (unbounded queue), ρ<1 ⇒ finite L, W.
2. `FuzzLittleLaw` — for fuzzed queue: `L = λW` within numerical tol.

### autodiff (2)
1. `FuzzReverseModeAdditive` — `∇(f+g) = ∇f + ∇g` for fuzzed expression DAGs.
2. `FuzzGradientFiniteDiff` — autodiff gradient agrees with central-difference within `√ε·‖x‖` for fuzzed smooth ops.

### timeseries / changepoint (2)
1. `FuzzGARCHPositiveVar` — fuzz (ω,α,β) with α+β<1; conditional variance always > 0 for fuzzed return series.
2. `FuzzBOCPDProbabilities` — run-length posterior is proper distribution (sums to 1, all ≥0) at every step for fuzzed data.

### combinatorics seed example
```go
func FuzzFactorialOverflow(f *testing.F) {
    for n := 0; n <= 25; n++ { f.Add(uint8(n)) }
    f.Fuzz(func(t *testing.T, n uint8) {
        v, err := combinatorics.Factorial(int(n))
        if n >= 21 && err == nil { t.Fatalf("n=%d should overflow uint64", n) }
        if n < 21 && err != nil   { t.Fatalf("n=%d unexpected err: %v", n, err) }
        if n < 21 && v == 0       { t.Fatalf("n=%d returned 0", n) }
    })
}
```

## Recommendation

### Standard pattern
```go
func FuzzXxx(f *testing.F) {
    // 1. Seed from existing golden vectors (read testdata/*.json)
    for _, v := range loadGoldenSeeds(t, "testdata/xxx.json") { f.Add(v.Bytes()) }
    // 2. Add boundary seeds: empty, single, max-size, NaN/Inf encodings
    f.Add([]byte{}); f.Add(nanBytes()); f.Add(infBytes())
    // 3. Fuzz function: invariant or round-trip; return early on Skip-worthy input
    f.Fuzz(func(t *testing.T, raw []byte) {
        x, ok := decode(raw); if !ok { t.Skip() }
        defer func() { if r := recover(); r != nil { t.Fatalf("panic: %v", r) } }()
        // assert invariant
    })
}
```

### CI integration
Add to `.github/workflows/fuzz.yml`:
```yaml
- name: Fuzz (smoke)
  run: |
    for pkg in linalg prob signal compression crypto combinatorics geometry chaos info color orbital topology sequence audio acoustics calculus control em fluids gametheory graph optim queue autodiff; do
      for fz in $(go test -list 'Fuzz.*' ./$pkg | grep ^Fuzz); do
        go test -run=^$ -fuzz=^${fz}$ -fuzztime=15s ./$pkg
      done
    done
```
- Per-target 15s in PR CI (≈75 targets × 15s ≈ 19 min single-threaded; parallelize by package matrix → ~2 min).
- Nightly: `-fuzztime=10m` per target on main branch.
- Crashing inputs persisted to `testdata/fuzz/FuzzXxx/` are committed and become regression seeds (gofumpt-stable; deterministic).
- `go test -fuzz` in CI requires Go ≥1.18 (reality is on 1.22).

### Adoption order (highest leverage first)
1. **signal.FuzzFFTRoundTrip, signal.FuzzFFTNonPowerOfTwo** — FFT is consumed by audio, acoustics, control; bugs cascade.
2. **prob.FuzzPDFNonNegative, FuzzQuantileInverse** — used by changepoint, infogeo; NaN propagation costly.
3. **linalg.FuzzInverseRoundTrip, FuzzShapeMismatch** — foundation of PCA, Cholesky in prob, Kalman in chaos.
4. **compression round-trips** — easy wins, 100% coverage achievable.
5. **crypto.FuzzMillerRabin, FuzzModExpEdge** — security-relevant; n=0,1,2,3 boundaries are classic bug sources.

### What dropped at -fuzz (predicted)
- **N=0/N=1 panics** in FFT, window functions, MFCC (slot 6 audio-numerics flagged similar).
- **NaN poisoning** through PDF chains (Beta with α=0).
- **Overflow** in `Factorial(n)` and `Binomial(n,k)` for `n ≥ 21`.
- **Divide-by-zero** in `Quaternion.Normalize(0,0,0,0)`.
- **Infinite loops** in `Bisection` when `f(a)·f(b) > 0` (no bracket).
- **Negative-eccentricity Kepler** silently returning garbage instead of typed error.

## Sources
- Go fuzzing tutorial: <https://go.dev/doc/tutorial/fuzz> (stdlib since Go 1.18, native to `testing`).
- Go 1.22 fuzzing docs: <https://pkg.go.dev/testing#hdr-Fuzzing>.
- Reality CLAUDE.md (zero-deps policy, golden-file infra, 1,965 tests).
- Slot 382 (test-coverage) — confirmed 0 Fuzz targets.
- Slot 383 (property-tests) — invariant catalog this slot operationalizes via stdlib fuzz.
- Slot 320 (compression) — Huffman/LZ77 absent; fuzz round-trip after they land.
