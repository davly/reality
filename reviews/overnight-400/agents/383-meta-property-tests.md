# 383 — meta-property-tests (mathematical invariants for property-based testing)

## Headline
Reality has 0 PBT-based invariant tests across 100+ test files; this catalog enumerates ~150 mathematical invariants across 28 packages that example-based golden vectors structurally cannot cover (universally quantified `∀x` properties), making `gopter`/`rapid` adoption the single highest-leverage testing investment.

## Context
- Confirmed via grep: 0 `func Fuzz*`, 0 `gopter.`, 0 `rapid.`, 0 `quick.Check` in any `*.go` file (all 24 hits are inside `reviews/` markdown).
- Slot 382 found 636 golden vectors / 80 files (median ~8/file). Goldens prove pointwise correctness; PBTs prove *closure under structure* — orthogonal coverage.
- Reality's zero-deps policy permits a `_test`-only PBT dep (test code is build-tag-isolated). `pgregory.net/rapid` is single-file pure-stdlib-style; `leanovate/gopter` is heavier. Recommend `rapid` (matches Go fuzz idioms; shrinking; deterministic seeds → cross-language goldens still work as the *seeds* for rapid runs).

## Package-by-package invariant catalog

### linalg (12 invariants)
1. **Inverse round-trip**: `A · A⁻¹ ≈ I` (within κ(A)·ε) for non-singular `A` of dim ≤ 32.
2. **Determinant multiplicativity**: `det(AB) = det(A)·det(B)` ± rel-tol scaled by ‖A‖·‖B‖.
3. **Determinant via PLU**: `det(A) = sign(P) · ∏ U_ii`.
4. **Cholesky factor**: SPD `A` ⟹ `L Lᵀ = A`, `L_ii > 0`.
5. **QR orthogonality**: `Qᵀ Q = I`, `R` upper-triangular, `QR = A`.
6. **SVD reconstruction**: `U Σ Vᵀ = A`, σ_i ≥ σ_{i+1} ≥ 0, `UᵀU = VᵀV = I`.
7. **Eigvec orthogonality (symmetric)**: `Aᵀ = A` ⟹ eigenvectors mutually orthogonal; `A vᵢ = λᵢ vᵢ`.
8. **Trace = Σ eigenvalues**, **det = ∏ eigenvalues**.
9. **Norm submultiplicativity**: `‖AB‖ ≤ ‖A‖·‖B‖` for induced 2-norm.
10. **Frobenius–trace identity**: `‖A‖_F² = tr(AᵀA)`.
11. **Pseudoinverse Moore–Penrose**: `A A⁺ A = A`, `A⁺ A A⁺ = A⁺`, `(AA⁺)ᵀ = AA⁺`.
12. **PCA recovery**: project + reconstruct on top-k components ⟹ residual ≤ Σ_{i>k} σᵢ².

### prob (10 invariants)
1. **PDF non-negativity** ∀x for every distribution; **CDF monotone non-decreasing**, ∈ [0,1].
2. **CDF→0 at -∞, →1 at +∞**: `CDF(quantile(0)) ≈ 0`, `CDF(quantile(1)) ≈ 1`.
3. **Quantile inverse**: `CDF(quantile(p)) = p`, `quantile(CDF(x)) = x` (continuous distributions).
4. **PDF = d/dx CDF** via numerical derivative (calculus.Derivative cross-check).
5. **PDF integrates to 1** via Simpson over the support (use `calculus.Integrate`).
6. **Var = E[X²] − E[X]²** from analytic moments.
7. **LLN**: mean of N≥10⁵ samples → analytic mean within `5·σ/√N`.
8. **CLT**: standardised sample-mean → N(0,1) by KS statistic ≤ critical at α=0.01.
9. **MGF/CGF identities** (where defined): `cumulant₁ = mean`, `cumulant₂ = var`.
10. **Bayesian conjugate priors**: posterior parameters from sufficient stats match closed-form (Beta–Binomial, Gamma–Poisson, NIG–Normal).

### signal (10 invariants)
1. **FFT round-trip**: `IFFT(FFT(x)) ≈ x` (rel-tol N·ε for N≤2¹⁶).
2. **Parseval/Plancherel**: `Σ|xₙ|² = (1/N) Σ|Xₖ|²`.
3. **Real-input conjugate symmetry**: `x ∈ ℝᴺ` ⟹ `X[k] = conj(X[N−k])`.
4. **Linearity**: `FFT(αx + βy) = α FFT(x) + β FFT(y)`.
5. **Time-shift theorem**: `FFT(shift(x, m))[k] = X[k]·exp(−2πi·k·m/N)`.
6. **Convolution theorem**: `FFT(x ⊛ y) = FFT(x)·FFT(y)` (circular).
7. **Hilbert idempotence**: `Hilbert(Hilbert(x)) = −x`; analytic signal magnitude ≥ |x|.
8. **Window unity DC**: rectangular window `Σwₙ = N`; Hann `Σwₙ ≈ N/2`.
9. **Filter linear-phase**: symmetric FIR ⟹ phase response linear in ω.
10. **Decimation/interpolation**: `down(up(x, L), L) = x` when no aliasing.

### combinatorics (8 invariants)
1. **Symmetry**: `C(n,k) = C(n,n−k)`.
2. **Pascal's rule**: `C(n,k) = C(n−1,k−1) + C(n−1,k)`.
3. **Row sum**: `Σₖ C(n,k) = 2ⁿ`; alternating sum = 0 for n≥1.
4. **Catalan four-way**: `C(2n,n)/(n+1) = (2n)!/((n+1)!n!) = ΣᵢCᵢC_{n−1−i} = ∏ᵢ(n+i)/i`.
5. **Stirling 2nd-kind recurrence**: `S(n,k) = k·S(n−1,k) + S(n−1,k−1)`.
6. **Bell = Σ Stirling-2**: `Bₙ = Σₖ S(n,k)`.
7. **Permutation count**: `P(n,k) = n!/(n−k)! = k!·C(n,k)`.
8. **Partition generating function**: p(n) matches Euler product expansion to n=50.

### geometry (8 invariants)
1. **Quaternion norm-conj inverse**: `q · q* = ‖q‖²`; unit-q ⟹ `q⁻¹ = q*`.
2. **Rotation composition**: `R(q₁q₂) = R(q₁)·R(q₂)` on ℝ³ vectors.
3. **Slerp endpoints**: `slerp(q₀,q₁,0)=q₀`, `slerp(q₀,q₁,1)=q₁`; constant angular velocity.
4. **SDF Lipschitz**: `|sdf(p) − sdf(q)| ≤ ‖p−q‖`.
5. **SDF sign**: inside ⟹ < 0, on surface ⟹ ≈ 0, outside ⟹ > 0.
6. **Convex hull idempotence**: `hull(hull(P)) = hull(P)`; all input points inside hull (sign of cross-product test).
7. **Bezier endpoint interpolation**: `B(0)=P₀`, `B(1)=Pₙ`.
8. **Projective duality**: `cross(line₁, line₂)` returns intersection point; line-through-two-points = `cross(p₁, p₂)`.

### chaos (6 invariants)
1. **Lorenz canonical Lyapunov**: σ=10, ρ=28, β=8/3 ⟹ λ₁ ≈ 0.9056 ± 0.02 over T≥1000.
2. **Conservation under symplectic integrators** (if any added): energy drift bounded.
3. **Time-reversibility of RK4** for Hamiltonian systems within ε·T².
4. **Lyapunov spectrum sums to divergence**: `Σλᵢ = ⟨∇·f⟩` (Lorenz: −σ−1−β = −13.667).
5. **Van der Pol limit cycle period**: μ→0 limit period → 2π.
6. **Logistic map bifurcations**: r=3.5699456 onset of chaos within tabulated digits.

### crypto (8 invariants)
1. **Miller-Rabin soundness**: composite n is reported composite by ≥ ¾ of bases (probabilistic; bound k·witnesses).
2. **Modular exponentiation**: `(a^b mod m)·(a^c mod m) ≡ a^(b+c) mod m`.
3. **Fermat little**: `a^(p−1) ≡ 1 mod p` for prime p, gcd(a,p)=1.
4. **CRT recovery**: solve x ≡ aᵢ mod mᵢ ⟹ unique x mod ∏mᵢ; verify residues.
5. **Extended Euclid**: `gcd(a,b) = sa + tb` and `s,t` returned satisfy this.
6. **Hash determinism + avalanche**: same input ⟹ same output; bit-flip changes ≥ 40% output bits (statistical).
7. **ECC point-on-curve**: every output of scalar mult satisfies `y² = x³ + ax + b mod p`.
8. **PRNG period & uniformity**: χ² over 10⁶ outputs accepts H₀ at α=0.01.

### info (8 invariants)
1. **Entropy bound**: `0 ≤ H(X) ≤ log₂|alphabet|`; equality iff uniform.
2. **Joint entropy**: `H(X,Y) ≤ H(X) + H(Y)`; equality iff independent.
3. **KL ≥ 0**: `D(p‖q) ≥ 0`, =0 iff p=q a.e.
4. **Mutual info symmetry**: `I(X;Y) = I(Y;X)`; `I(X;Y) = H(X)+H(Y)−H(X,Y)`.
5. **Data-processing inequality**: `I(X;Y) ≥ I(X;f(Y))`.
6. **Cross-entropy ≥ entropy**: `H(p,q) ≥ H(p)`.
7. **Chain rule**: `H(X,Y) = H(X) + H(Y|X)`.
8. **LZ76 monotone**: complexity non-decreasing with prefix length.

### color (7 invariants)
1. **RGB↔XYZ round-trip** for in-gamut colours: ‖RGB − XYZ⁻¹(XYZ(RGB))‖∞ ≤ 1e-12.
2. **CIEDE2000 metric axioms**: d(a,a)=0; d(a,b)=d(b,a); ≥ 0.
3. **WCAG contrast**: black/white = 21:1 exactly; symmetric in arguments.
4. **Bradford adaptation invertibility**: `M⁻¹·M·XYZ = XYZ`.
5. **HSL→RGB→HSL** stable away from achromatic axis.
6. **sRGB gamma round-trip**: `linear_to_srgb(srgb_to_linear(c)) = c`.
7. **Luminance monotonicity**: scaling RGB by α ∈ [0,1] scales Y by α (linear-light space).

### orbital (7 invariants)
1. **Kepler period eccentricity-independence**: `T = 2π√(a³/μ)` invariant under e ∈ [0, 0.99).
2. **Vis-viva**: `v² = μ(2/r − 1/a)` along entire orbit.
3. **Specific orbital energy**: `ε = −μ/(2a)`.
4. **Angular momentum conservation**: `‖r×v‖` constant within RK4 drift bound.
5. **Hohmann optimality**: ΔV_total of bi-impulse Hohmann ≤ any other coplanar two-impulse transfer in test.
6. **Escape velocity**: `v_esc(r) = √(2μ/r)`; `ε(v_esc) = 0`.
7. **Hill sphere**: `r_H = a(1−e)·∛(m/3M)` matches Earth/Sun = 1.496e9 m to 3 sig figs.

### acoustics (6 invariants)
1. **Speed of sound**: c(T) monotone increasing in T (kelvin); c(20°C, dry air) = 343.21 m/s.
2. **dB SPL invertibility**: `SPL(P) → P → SPL` round-trip.
3. **A-weighting at 1 kHz** = 0 dB (definitional anchor).
4. **Doppler symmetry**: source-moving and observer-moving formulas agree at v≪c.
5. **Sabine RT60**: doubling absorption halves RT60 (linearity).
6. **Inverse-square**: SPL drops 6 dB per distance doubling in free field.

### fluids (6 invariants)
1. **Reynolds dimensionlessness**: `Re(ρ,v,L,μ)` invariant under consistent unit scaling.
2. **Bernoulli energy**: `½ρv² + ρgh + p` constant along streamline (frictionless).
3. **Darcy–Weisbach scaling**: ΔP ∝ L (linear in pipe length).
4. **Drag quadratic**: `F_d ∝ v²` at fixed C_d, ρ, A.
5. **Terminal velocity** ⟹ net force = 0 (gravity = drag).
6. **Continuity**: `ρ₁A₁v₁ = ρ₂A₂v₂` for steady incompressible.

### em (6 invariants)
1. **Coulomb superposition**: `F(q; q₁,q₂) = F(q;q₁) + F(q;q₂)`.
2. **Newton's third law**: `F(q₁→q₂) = −F(q₂→q₁)`.
3. **Ohm linearity**: `V(αI) = α V(I)`.
4. **Series resistors sum**; **parallel reciprocals sum**.
5. **RC time constant**: charge reaches 1 − e⁻¹ at t=τ exactly.
6. **LC oscillation period**: `T = 2π√(LC)`, frequency-independent of amplitude.

### physics (6 invariants)
1. **Energy conservation in free fall**: `½mv² + mgh = const` along trajectory.
2. **Newton II symmetry under Galilean boost**: forces invariant under uniform velocity shift.
3. **Stress–strain Hookean linearity** below yield.
4. **Thermal equilibrium**: heat flow direction sign matches ΔT sign.
5. **Ideal-gas law cycle**: PV/T constant on closed isothermal/isobaric/isochoric cycle.
6. **Momentum conservation**: 2-body collision Σp_before = Σp_after (elastic and inelastic).

### calculus (6 invariants)
1. **Linearity of integral**: `∫(αf+βg) = α∫f + β∫g`.
2. **Additivity over intervals**: `∫_a^c f = ∫_a^b f + ∫_b^c f`.
3. **FTC**: `d/dx ∫_a^x f(t)dt = f(x)`.
4. **Simpson exactness on cubics**: error = 0 for f ∈ P₃.
5. **RK4 4th-order convergence**: halve h ⟹ error ÷ 16.
6. **Newton root quadratic convergence**: `|x_{n+1}−x*| ≤ C·|x_n−x*|²` near simple root.

### control (5 invariants)
1. **PID linearity in setpoint**: doubling setpoint doubles steady-state output (linear plant).
2. **Bode magnitude/phase consistency**: minimum-phase ⟹ magnitude determines phase (Bode-gain-phase relation).
3. **Stability margins**: gain margin > 1 ⟺ closed-loop stable (Nyquist).
4. **TF series composition**: `TF(G·H) = TF(G)·TF(H)`.
5. **Step-response final-value theorem**: `lim_{t→∞} y(t) = lim_{s→0} sY(s)` for stable systems.

### graph (6 invariants)
1. **Dijkstra ≤ A* ≤ BFS-on-unit-weights** on the same graph (with admissible heuristic).
2. **Triangle inequality**: shortest-path d(u,w) ≤ d(u,v) + d(v,w).
3. **BFS layer monotonicity**: dist non-decreasing as queue advances.
4. **Topological sort**: every edge (u,v) ⟹ pos(u) < pos(v); only on DAGs.
5. **Connectivity symmetry** (undirected): u reaches v ⟺ v reaches u.
6. **Handshake lemma**: Σdeg(v) = 2|E|.

### optim (6 invariants)
1. **Gradient zero at minimiser** for smooth f: `‖∇f(x*)‖ ≤ tol`.
2. **Monotone descent**: `f(x_{k+1}) ≤ f(x_k)` for line-searched methods (Wolfe).
3. **Bisection bracket invariance**: sign(f(a))·sign(f(b)) < 0 maintained each iteration.
4. **Simplex bounded LP**: optimal vertex satisfies all constraints; primal=dual objective.
5. **Simulated annealing acceptance**: `P(accept worse) = exp(−ΔE/T)` empirical match over many trials.
6. **L-BFGS quadratic exactness**: converges in ≤ n steps on n-dim quadratic.

### combinatorics/queue (queue, 6 invariants)
1. **Little's law**: `L = λW` across all M/M/c, M/G/1.
2. **M/M/1 utilisation**: ρ = λ/μ < 1 ⟹ L = ρ/(1−ρ).
3. **Erlang B recursion**: `B(c,a) = aB(c−1,a)/(c+aB(c−1,a))`.
4. **Erlang C ≥ Erlang B** at same (c,a).
5. **PASTA**: arrival-averages = time-averages for Poisson arrivals.
6. **Steady-state existence**: ρ<1 ⟹ all moments finite.

### compression (5 invariants)
1. **Lossless round-trip**: `decode(encode(x)) = x` for RLE/Huffman/LZ77.
2. **Length non-expansion (theoretical)**: optimal Huffman on n symbols ≤ ⌈H(p)⌉ + 1 bits/symbol average.
3. **Kraft inequality**: `Σ 2⁻ℓᵢ ≤ 1` for prefix codes.
4. **Delta encoding telescoping**: `decode(delta(x)) = x`.
5. **Entropy lower bound**: any lossless coder average length ≥ H(X).

### gametheory (5 invariants)
1. **Nash existence (finite game)**: at least one mixed equilibrium exists.
2. **Best-response fixed point**: at NE, no player improves unilaterally.
3. **Shapley efficiency**: `Σᵢ φᵢ(v) = v(N)`.
4. **Shapley symmetry**: equivalent players ⟹ equal payoffs.
5. **Minimax = maximin** in zero-sum (von Neumann).

### autodiff (5 invariants)
1. **Forward-mode = numerical derivative** within central-difference O(h²) on smooth f.
2. **Reverse-mode = forward-mode** to within ε on scalar-out functions.
3. **Chain rule**: `∂(f∘g)/∂x = f'(g(x))·g'(x)` automatic via composition.
4. **Linearity of grad**: `∇(αf+βg) = α∇f + β∇g`.
5. **Hessian symmetry**: `∂²f/∂xᵢ∂xⱼ = ∂²f/∂xⱼ∂xᵢ` for C² f.

### sequence (5 invariants)
1. **Levenshtein metric axioms**: d(a,a)=0; symmetric; triangle inequality.
2. **Jaccard ∈ [0,1]**; J(A,A)=1; J(A,∅)=0.
3. **Dice = 2·|A∩B|/(|A|+|B|)** matches Jaccard via `D = 2J/(1+J)`.
4. **Soundex 4-char invariant**: always returns letter+3 digits.
5. **TokenSetRatio bounded** [0,100]; idempotent on identical strings.

### topology / changepoint / timeseries / infogeo (12 invariants combined)
- **persistent homology stability**: bottleneck distance ≤ Hausdorff distance of inputs.
- **persistence diagram**: birth ≤ death always.
- **BOCPD posterior sums to 1** at every step.
- **changepoint score non-negative**.
- **GARCH(1,1) stationarity**: α+β<1 ⟹ unconditional variance finite.
- **DCC correlation matrix**: PSD at every t; diagonal = 1.
- **Bregman divergence non-negativity** and zero iff x=y.
- **Bregman three-point identity**: `D(x,y) + D(y,z) − D(x,z) = ⟨∇φ(z)−∇φ(y), x−y⟩`.
- **f-divergence ≥ 0**, =0 iff p=q.
- **MMD² ≥ 0** (kernel positive-definite).
- **Wasserstein metric axioms** (in `optim/transport`).
- **infogeo Fisher matrix PSD**.

### audio (5 invariants)
1. **Onset detection idempotence**: detecting on already-detected frames yields same set.
2. **Pitch round-trip**: synthesise sine at f₀ ⟹ pitch-track returns f₀ ± cents tolerance.
3. **Spectrogram + ISTFT** (if any): perfect-reconstruction COLA window passes round-trip.
4. **Tempo doubling-halving ambiguity**: detected BPM ∈ {true, true/2, true·2}.
5. **CQT linearity & log-frequency spacing** invariant under input scaling.

## Recommendation
1. **Adopt `pgregory.net/rapid` test-only** (single dep, only in `_test.go`, doesn't violate zero-deps in shipped binary). Use deterministic seeds and persist failing seeds as new golden vectors → cross-language parity preserved.
2. **Tier-A invariants (must-have, weekend-effort)**: linalg #1–6, prob #1–6, signal #1–4, info #1–4, combinatorics all, geometry #1–3, color #1, orbital #1–3, autodiff #1–3, calculus #1–3 — ~50 properties covering structural correctness across the math core.
3. **Tier-B (week-effort)**: chaos Lyapunov, Erlang/Little, Bregman/MMD, persistent-homology stability — these uncover bugs in numerical code that goldens cannot.
4. **Per-package PBT file convention**: `xxx_property_test.go` build-tagged `//go:build property` so zero-deps purity is preserved on default `go test ./...`. Run nightly via `go test -tags=property ./...`.
5. **Headline target**: 150 invariants × ≥1000 random cases each = 150k checks per CI run; rapid's shrinking will surface minimal counterexamples.
6. Slot 393 (`new-property-tests`) appears to be the implementation slot — this catalog is its spec input.

## Sources
- Slot 382 finding (PROGRESS.md): 0 PBT libs in tree.
- Reality CLAUDE.md tolerance ladder (per-function tol).
- Standard texts implicit: Trefethen & Bau (linalg), Cover & Thomas (info), Knuth TAOCP vol.4 (combinatorics), Strogatz (chaos), Vallado (orbital).
- pgregory.net/rapid (Go PBT lib, single-author, MIT, ~3kLOC, no transitive deps).
