# 208 | new-exterior-calculus

**Summary line 1.** EIGHTH Block-C cutting-edge-math review and FIRST exterior-calculus-as-its-own-axis scoping in 400-sequence (continuous Λ^k(M) side, NOT 194's discrete cochain DEC and NOT 207's broader Riemannian/curvature-tensor surface) covering k-form spaces Λ^k(M)/wedge ω∧η/exterior derivative d:Ω^k→Ω^{k+1}/d²=0/pullback φ*ω/pushforward φ_*X/interior product ι_X/Lie derivative L_X via Cartan magic L_X=dι_X+ι_Xd/Hodge star ★:Λ^k→Λ^{n−k}/codifferential δ=±★d★/Laplace-de Rham Δ=dδ+δd/harmonic forms ker Δ/Hodge decomposition Λ^k=im(d)⊕im(δ)⊕H^k/de Rham cohomology H^k_dR(M)=ker(d)/im(d)/Betti numbers β_k=dim H^k_dR/Stokes ∫_∂M ω=∫_M dω/Poincaré lemma (closed forms locally exact)/volume form √|g| dx¹∧…∧dx^n/integration on oriented manifolds/symplectic 2-form ω=dq^i∧dp_i (cross-204)/connection 1-form ω∈Ω¹(P,𝔤)/curvature 2-form Ω=dω+ω∧ω (Cartan structure)/Maxwell as F=dA, dF=0, d★F=★J/Yang-Mills F=dA+A∧A action ½∫tr(F∧★F)/Chern character ch(F)=tr exp(iF/2π)/Pontryagin classes/generalised Stokes for chains and cochains: reality v0.10.0 ships ZERO continuous-Λ^k surface — `geometry/{quaternion,sdf,curves,polygon}.go` 656 LOC point/curve primitives only; repo-wide grep on `Form\|Wedge\|HodgeStar\|ExteriorDeriv\|deRham\|Pullback\|InteriorProduct\|LieDeriv\|HarmonicForm\|Betti\|VolumeForm\|Symplectic\|ChernCharacter\|Pontryagin\|YangMills\|Connection1Form\|Curvature2Form` returns the same near-empty set as 207 (4 false positives in `autodiff/{vector,ops}.go` + `gametheory/kelly.go`); the closest substrate is `calculus.SimpsonsRule`/`GaussLegendre` (continuous integration), `linalg.MatMul` (tensor contraction), and 207-D6+D7 Christoffel/Riemann machinery proposed but not yet shipped.

**Summary line 2.** Twenty primitives E1–E20 totalling ~3050 LOC across new sub-package `geometry/extcalc/` (mirrors 194 `geometry/dec/` + 207 `geometry/diffgeo/` precedent — three sibling sub-packages of `geometry/`: dec=discrete cochains on meshes, diffgeo=tensors-on-charts, extcalc=Λ^k-as-symbolic-multi-index-arrays). **Versus 194-em-geometry D0–D17:** 194 ships the discrete side (cochain c ∈ ℝ^{n_k}, d as signed incidence matrix, ★ as diagonal primal/dual ratio); this slot ships the continuous side (k-form ω as `func([]float64, *KForm)` evaluator, d symbolic on coefficient functions, ★ as metric-dependent Hodge dual on multi-indices); dual via Whitney's de Rham theorem and both should ship to fully witness `R-MUTUAL-CROSS-VALIDATION 3/3`. **Versus 207-new-diff-geo D1–D28:** 207 incidentally lists D15–D20 (forms/wedge/d/Hodge/Stokes/de Rham) as Tier-3 of five tiers and explicitly punts D20 onto sparse-eigen 097; this slot ELEVATES the same axis to first-class status, expands D15–D20 from 6 entries to 20 (adding pullback/pushforward/interior product/Lie derivative/Cartan magic/codifferential/Laplace-de-Rham/harmonic basis/Hodge decomposition/Poincaré lemma witness/symplectic form/Maxwell-as-F/Yang-Mills/Chern/Pontryagin), and proposes a coordination contract: 207-D15–D20 + 207-D22-D23 (connection-1-form + Cartan-2-form) should be EXTRACTED here, narrowing 207 to Christoffel/Riemann/geodesic/Killing/Einstein scope (~2920 LOC instead of 4100). **Versus 204-symplectic I1–I14:** 204 ships *integrators* preserving the symplectic 2-form ω=dq^i∧dp_i; this slot ships *the form ω itself* as E16 + the test that 204 falsifies non-symplectic methods against. **Versus 194-D6 Maxwell-on-meshes:** complementary — 194-D6 is the numerical solver; this slot's E18 is the axiom statement (F=dA, dF=0, d★F=★J in three lines of code after E1–E10 land). Single highest-leverage 1-day project: **E1+E2+E5+E10 = `KForm` + wedge + d + d²=0 witness ~480 LOC** saturates R-MUTUAL-CROSS-VALIDATION 3/3 (continuous d via `calculus.NumericalGradient` × discrete d via 194-D1 on Whitney-discretised mesh × identity test d²=0 to machine zero). Cutting-edge moat: **E18 Maxwell-as-F + E19 Yang-Mills F=dA+A∧A + E20 Chern/Pontryagin ~610 LOC** — singular competitive piece (no zero-dependency Go library ships symbolic Yang-Mills curvature 2-form with byte-for-byte cross-language golden-file contract; closest are FORM/Cadabra/xAct/Mathematica proprietary, sympy.diffgeo NumPy-bound, tensorial.jl sparse-array-bound). Cross-package blocker hierarchy: 097-T1 SparseEigen gates direct continuous E14 spectral decomposition (194-DEC bridge unblocks practical case); 205-L1+L2 SO(3)/Lie-algebra exp/log gate non-abelian E17/E19/E20 (abelian Maxwell ships); 207-D6+D7 Christoffel/Riemann gate E20 Chern on curved manifold (flat case unblocked).

---

## 0. State at HEAD (2026-05-08, v0.10.0)

`geometry/` (656 LOC, 4 files) — `quaternion.go` SO(3)/SU(2) algebra, `sdf.go` implicit surfaces, `curves.go` Bezier/Catmull-Rom, `polygon.go` 2-D triangle/hull. Closest semantic match: `TriangleArea2D = ½ det(b−a, c−a)` IS the 2-form `dx∧dy` on a 2-simplex but unlabelled; `signal.Convolve` is structurally wedge-with-translates. Neither is exposed as a Λ^k operation.

`infogeo/` (1373 LOC) — f-divergences + Bregman + MMD². No 1-form/metric/volume-form surface. Bregman generator is a closed 0-form on the dual-flat manifold but the dual structure is v2-deferred per 092-T2.

Substrate available: `calculus.SimpsonsRule`/`GaussLegendre` (Stokes validation), `calculus.NumericalGradient` (h ≈ ε^{1/3} ≈ 6e-6, substrate for symbolic d on user-supplied coefficient-functions), `linalg.MatMul`/`MatVec`/`LU` (tensor contraction + metric inversion). Sparse symmetric eigensolver missing → 097-T1 (gates direct continuous spectral E14).

Cross-link audit: 194 ships discrete cochains, complementary; 207 lists D15–D20 as Tier-3, recommend re-home; 204 needs E16 symplectic form to falsify non-symplectic integrators; 205-L1+L2 gates non-abelian E17/E19/E20; 156 topology-persistent cross-validates E15 Betti via simplicial homology over F_2 (integer-equality contract).

Repo-wide grep on `KForm|OneForm|TwoForm|WedgeProduct|HodgeStar|ExteriorDerivative|Codifferential|InteriorProduct|Pullback|Pushforward|LieDerivative|HarmonicForm|deRhamCohomology|BettiNumbers|SymplecticForm|ConnectionOneForm|CurvatureTwoForm|YangMillsAction|ChernCharacter|PontryaginClass` → zero callable surface.

---

## 1. The twenty primitives

For each: capability + composition + LOC + blocker. Numbered E1–E20.

### Tier 1 — Λ^k machinery, ships unblocked (~480 LOC)

**E1 — `KForm` type (multi-index canonical-coefficient storage).** `KForm{Dim, Rank int; Coeffs []float64; Indices [][]int}` where `Indices[i]` is strictly-increasing multi-index of length `Rank`. Length `C(Dim, Rank)`. `Get(idx) float64` returns coefficient with sign for any permutation; `Set(idx, v)` canonicalises. Variable forms: `KFormField func(x []float64, out *KForm)`. **150 LOC** in `geometry/extcalc/forms.go`. Ref: Lee 2012 §14, Bott-Tu §1.

**E2 — Wedge product ω ∧ η : Ω^r × Ω^s → Ω^{r+s}.** `(ω∧η)_{i₀…} = Σ_σ sgn(σ)·ω_{σ(i_left)}·η_{σ(i_right)} / (r! s!)` over (r,s)-shuffles. Tests: antisymmetry `ω∧η = (−1)^{rs} η∧ω`, associativity. **120 LOC** in `extcalc/wedge.go`. Depends E1.

**E5 — Exterior derivative d : Ω^k → Ω^{k+1}, with d²=0 witness.** `(dω)_{i₀…iₖ} = Σ_j (−1)^j ∂_{i_j} ω_{i₀…î_j…iₖ}`. Each ∂ via `calculus.NumericalGradient` (4-digit) or caller-supplied analytic `dOmega KFormField` for round-off. **Flagship test E5':** `dd2Witness(ω, x) → ‖d(dω)‖_∞` should be < 1e-10 (Poincaré, algebraic identity). Companion `IsClosed`, `IsExact`. **210 LOC** in `extcalc/exterior.go`. Ref: Lee §14, do Carmo §6, Spivak Vol. 1 §7.

**E10 — d²=0 witness fixture.** `dd2WitnessSymbolic(ω, x, eps) error` runs 100 random polynomial 0/1/2-forms on R⁴ → all `‖d²ω‖² < 1e-20` (machine-zero). Pure validation. **80 LOC** test-only. Depends E5.

### Tier 2 — Hodge mechanics + Cartan magic (~640 LOC)

**E3 — Pullback φ* : Ω^k(N) → Ω^k(M).** Chain rule on coefficients, `(φ*ω)_{i₀…} = Σ_j ω_{j₀…}(φ(x)) · ∏ ∂(φ^{j_l})/∂x^{i_l}` antisymmetrised. Tests: `φ*(ω∧η) = (φ*ω)∧(φ*η)` (naturality wrt wedge); `φ*(dω) = d(φ*ω)` (the FUNDAMENTAL property — exterior derivative commutes with pullback). **170 LOC** in `extcalc/pullback.go`. Depends E1, E2, E5.

**E4 — Pushforward φ_* : T_pM → T_{φ(p)}N (vector fields only).** `dφ_x(X(x)) = J_φ(x) · X(x)`. Forms only pull back; vectors only push forward in general (forms push forward only when φ is a diffeomorphism). **80 LOC** in `extcalc/pushforward.go`. Doc-pin asymmetry.

**E6 — Interior product ι_X : Ω^k → Ω^{k−1}.** `(ι_X ω)_{i₁…i_{k−1}} = X^j ω_{j i₁…i_{k−1}}`. Tests: graded Leibniz `ι_X(ω∧η) = (ι_Xω)∧η + (−1)^r ω∧(ι_Xη)`, `ι_X² = 0`, anticommutativity `ι_X∘ι_Y + ι_Y∘ι_X = 0`. **110 LOC** in `extcalc/interior.go`. Depends E1, E2.

**E7 — Lie derivative L_X via Cartan magic formula.** `L_X = ι_X∘d + d∘ι_X`. Cross-validates against the coordinate formula `(L_Xω)_{i₀…} = X^j ∂_jω_{i₀…} + Σ_l (∂_{i_l}X^j) ω_{i₀…j…}`. **Flagship cross-validation:** Cartan magic LHS vs RHS to round-off. `[L_X, L_Y] = L_{[X,Y]}` (Jacobi). **140 LOC** in `extcalc/lie_derivative.go`. **Re-home from 207-D10.** Depends E5, E6.

**E8 — Hodge star ★ : Λ^k → Λ^{n−k}.** `(★ω)_{j₁…j_{n−k}} = (1/k!) √|g| ε_{i₁…iₖj₁…j_{n−k}} g^{i₁a₁}…g^{iₖaₖ} ω_{a₁…aₖ}`. Tests: `★★ω = (−1)^{k(n−k)+s}ω` (s = signature of g). On Euclidean R³: `★dx = dy∧dz`, `★(dx∧dy) = dz`, `★★=+1` everywhere. On Minkowski: `★★|_{Λ^2}=−1` (parity inversion that distinguishes Lorentzian from Riemannian). **140 LOC** in `extcalc/hodge.go`. Ref: do Carmo §6, MTW §4.5.

**E9 — Codifferential δ = (−1)^{n(k+1)+1} ★d★ and Laplace-de Rham Δ = dδ + δd.** Restricts to standard Laplacian on 0-forms; pin Lee 2018 sign convention so eigenvalues are non-negative. Tests: self-adjoint under `⟨α,β⟩ = ∫ α∧★β` on closed M; commutes with `★`; spectrum on S²(r) gives `λ_l = l(l+1)/r²`. **120 LOC** in `extcalc/codifferential.go`. Depends E5, E8.

### Tier 3 — Stokes + Poincaré + Hodge decomposition + de Rham (~810 LOC)

**E11 — Volume form vol_g = √|g| dx¹∧…∧dx^n + integration on oriented manifolds.** `IntegrateOverChart(ω, domain, n_quad)` via tensor-product Gauss-Legendre. Tests: ∫_{S²} vol = 4πr², ∫_{R²} (1/π)e^{−r²} dx∧dy = 1. **150 LOC** in `extcalc/integrate.go`. Composes `calculus.GaussLegendre`. Depends E1.

**E12 — Stokes' theorem ∫_∂M ω = ∫_M dω.** `StokesWitness(ω, M, ∂M)` returns `nil` if `|∫_∂M ω − ∫_M dω| < eps`. **The single most important test in the entire continuous-Λ^k canon** — validates that E5 (d) and E11 (∫) are jointly correct. Special cases: 1-form on interval = FTC; 1-form on closed loop in R² = Green's theorem; 2-form on closed surface in R³ = divergence theorem; 2-form on bounded surface in R³ = classical Stokes (curl form). **180 LOC** in `extcalc/stokes.go`. Ref: Spivak 1965 (the entire book is one theorem); Lee §16. Depends E5, E11.

**E13 — Poincaré lemma constructive (closed forms locally exact).** Explicit homotopy formula `α(x) = ∫_0^1 ι_{X_t}ω(F(t,x)) dt` where `F(t,x) = x_★+t(x−x_★)` is the radial contraction. Closes loop with E5 (verify `dα = ω` to round-off after construction). Counterexample: `dθ` on R²\{0} is closed but not exact (obstruction = non-trivial topology, witness via E15). **150 LOC** in `extcalc/poincare.go`. Ref: Madsen-Tornehave §3.3. Depends E5, E6.

**E14 — Hodge decomposition Λ^k = im(d) ⊕ im(δ) ⊕ H^k.** Two Poisson-on-forms solves: `Δf = δω`, `Δβ = dω`, then `h = ω − df − δβ`. Practically: discretise ω via `∫_σω` on each k-simplex → run 194-D8 → reconstruct h via Whitney 194-D11. **Bridge primitive between continuous (this slot) and discrete (194).** Tests: T² with constant 1-form `ω = a dx + b dy` → `df=0`, `δβ=0`, `h=ω` (every constant 1-form harmonic, dim H¹(T²) = 2). **180 LOC** in `extcalc/hodge_decomposition.go`. Sub-blocker on `linalg.SparseEigen` (097-T1) for direct continuous; ships unblocked via 194-DEC bridge. Ref: Warner §6.1, de Rham 1955, Hodge 1941.

**E15 — de Rham cohomology H^k_dR(M) = ker(d_k)/im(d_{k−1}) and Betti numbers β_k.** `dim H^k_dR(M) = dim ker(Δ_k)` (Hodge theorem). For closed orientable n-M: β_k = β_{n−k} (Poincaré duality, witnessed by `★`). Tests: S^n: β = (1,0,…,0,1); T^n: β_k = C(n,k); Σ_g: β = (1, 2g, 1). **The tightest cross-package golden-file contract in the repo:** integer answers, no tolerance, must match byte-for-byte across continuous Hodge (this slot) + discrete Helmholtz-Hodge (194-D8) + simplicial homology over F_2 (156-topology-persistent). **120 LOC** in `extcalc/derham.go`. Ref: Bott-Tu §8, Hatcher §3. Depends E14.

### Tier 4 — Symplectic + Maxwell + Yang-Mills + characteristic classes (~960 LOC)

**E16 — Symplectic form ω = Σ dq^i ∧ dp_i (cross 204).** `SymplecticFormDarboux(n) KForm` returns canonical 2-form on T*Q. `IsSymplectic(ω, x)` validates closed (`dω=0`) and non-degenerate (`ω^n ≠ 0`). `IsSymplecticPreserving(φ, ω)` validates `φ*ω = ω` (used by 204 to test integrators). Liouville volume `vol_ω = ω^n/n!`. **The canonical falsification test for 204:** `LeapfrogStep` preserves ω to round-off over 10⁶ steps; non-symplectic RK4 fails (drift O(h⁴) per step). **110 LOC** in `extcalc/symplectic.go`. Depends E1, E2, E5. Ref: Arnold §8, Marsden-Ratiu §5.

**E17 — Connection 1-form ω ∈ Ω¹(P, 𝔤) on principal G-bundle.** Lie-algebra-valued 1-form. For G = U(1) (electromagnetism), 𝔤 = iR → just a real 1-form (vector potential A). For G = SU(N), 𝔤 = su(N), N²−1 components per spacetime index. **140 LOC** in `extcalc/connection.go`. Soft blocker on 205-L1+L2 for non-abelian; abelian (Maxwell) ships unblocked. Ref: Bleecker §3, Kobayashi-Nomizu Vol. II §II.

**E18 — Maxwell as F = dA, dF = 0, d★F = ★J.** `MaxwellFaraday(A, x)` returns `F = dA` (2-form on R⁴ spacetime). `MaxwellHomogeneous(F, x)` returns `dF` (zero by `F=dA ⟹ dF=d²A=0`, the homogeneous Maxwell pair ∇·B=0 + ∇×E+∂B/∂t=0). `MaxwellInhomogeneous(F, J, g_minkowski, x)` returns `d★F − ★J` (zero by Maxwell, the inhomogeneous pair). **Single most important pedagogical primitive in reality:** demonstrates that the four Maxwell equations collapse to two relations between forms, and the homogeneous pair is automatic from F=dA + d²=0. Tests: plane wave A=(0, A₀cos(kz−ωt), 0, 0) → ω²=c²k² dispersion via d★F=0; static Coulomb A=(φ,0,0,0) with φ=q/(4πε₀r) → d★F=★J reproduces ∇²φ=−ρ/ε₀. **140 LOC** in `extcalc/maxwell.go`. Depends E5, E8 (Hodge ★ on Minkowski). Ref: MTW §3.4+§4.1, Frankel §7, Wald §4.

**E19 — Yang-Mills curvature F = dA + A∧A and action S_YM = ½ ∫ tr(F ∧ ★F).** For Lie-algebra-valued A, `F = dA + A∧A` where the wedge incorporates the Lie bracket on 𝔤 (`(A∧A)_{μν} = [A_μ, A_ν]`). Abelian G=U(1): `[·,·]=0` reduces to Maxwell. Non-abelian G=SU(N): non-zero A∧A produces gluon-self-coupling. `YangMillsAction(A, g, M) float64` gauge-invariant. `YangMillsEulerLagrange`: `D_μ F^{μν} = J^ν` with `Dα = dα + [A,α]`. Tests: Bianchi DF=0, gauge invariance under A → A+Dε, abelian-vs-non-abelian distinguishability via `tr(F∧F)`. **250 LOC** in `extcalc/yangmills.go`. **Re-home from 207-D25.** Soft blocker on 205-L1+L2; abelian ships first. Ref: Bleecker §6, Atiyah-Bott 1983, Donaldson-Kronheimer 1990.

**E20 — Chern character ch(F) = tr exp(iF/2π) and Pontryagin classes p_k(F).** `ChernCharacter(F, k)` returns `ch_k(F) = tr(F^k)/k!`. `PontryaginClass(F, k)` returns `p_k(F) = (−1)^k c_{2k}(F⊗ℂ)`. `EulerClass(M)` for orientable even-dim M: `e(M) = Pf(R)/(2π)^{n/2}`. `ChernGaussBonnet`: `∫_M e(M) = χ(M)`, generalises Gauss-Bonnet from 2-D (E15+207-D26) to all even dims. Tests: trivial bundle F=0 → ch_k=0 except ch_0=rank; G=U(1) monopole on S² → `∫_{S²}F/(2π) = q` integer (Dirac quantisation, exact). **220 LOC** in `extcalc/characteristic_classes.go`. Soft blocker on 205-L1+L2 (Lie-algebra trace) and 207-D6+D7 (curvature on curved manifold for Pontryagin/Euler). Ref: Milnor-Stasheff 1974, Nakahara §11.

---

## 2. Composition graph

```
                E1 KForm
                   │
       ┌───────────┼───────────┬─────────────┐
       │           │           │             │
   E2 wedge   E5 d (keystone) E6 ι_X    E11 vol+∫
       │           │           │             │
       │       ┌───┼───┐       │             │
       │       ▼   ▼   ▼       │             ▼
       │  E3 φ*  E10  E7 L_X   │         E12 Stokes
       │           d²=0  (Cartan)              │
       ▼           witness  │                  ▼
   E8 Hodge★          ┌─────┘              E13 Poincaré
       │              │
       ▼              │
   E9 δ + Δ           │
       └───────┬──────┘
               ▼
          E14 Hodge decomposition ── (194-DEC bridge) ──► E15 de Rham/Betti
               │
               ▼
          E16 Symplectic ω (validates 204)
               │
               ▼
          E17 Connection 1-form (𝔤-valued)
               │
               ▼
          E18 Maxwell-as-F (abelian U(1))
               │
               ▼
          E19 Yang-Mills F = dA + A∧A (non-abelian SU(N), gates 205)
               │
               ▼
          E20 Chern/Pontryagin/Euler classes (Chern-Gauss-Bonnet)

          E4 pushforward (parallel ladder, vector fields only)
```

Tier-1 ships today (E1+E2+E5+E10). Tier-2 Hodge mechanics (E3+E4+E6+E7+E8+E9). Tier-3 (E11+E12+E13+E14+E15) — E14/E15 use 194-DEC bridge. Tier-4 (E16–E20) — E17/E19/E20 partial blocker on 205 for non-abelian; abelian Maxwell ships fully.

---

## 3. Cheapest 1-day standalone PR

**E1 + E2 + E5 + E10 = 480 LOC src + 320 LOC test** delivers:

1. The first continuous Λ^k primitive in reality.
2. R-MUTUAL-CROSS-VALIDATION 3/3 saturation: continuous d on `ω = f dx + g dy` returns `(∂g/∂x − ∂f/∂y) dx∧dy` to round-off (analytic ∂); 194-D1 discrete `d_0` matches to O(h²) on Whitney-discretised triangle mesh (de Rham bridge); E10 d²=0 on 100 random polynomial forms returns ‖d²‖_∞ < 1e-20.
3. Mirrors recent 6a55bb4 audio-onset-3-detector and 365368a copula×autodiff R-MUTUAL-CROSS-VALIDATION saturation pattern.
4. Standalone PR: zero coupling beyond `calculus.NumericalGradient`. New sub-package `geometry/extcalc/`.

---

## 4. Architectural keystone

**E5 ExteriorDerivative** unifies: E10 d²=0 (Poincaré locally exact), E12 Stokes (geometric content of d), E18 Maxwell (F=dA, dF=0 automatic), E19 Yang-Mills (F=dA+A∧A), E14 Hodge (`d` is the boundary operator on forms, kernel-modulo-image IS de Rham cohomology), E15 Betti, E20 Chern. Once E5 ships, the entire continuous-Λ^k canon is one composition graph away.

---

## 5. Placement and import graph

NEW sub-package `geometry/extcalc/` (mirror 194 `geometry/dec/` + 207 `geometry/diffgeo/`). Three sibling sub-packages: `dec/` discrete cochain DEC on meshes (194), `diffgeo/` Christoffel/Riemann on coordinate manifolds (207, narrowed), `extcalc/` continuous Λ^k forms (this slot).

Cycle-free imports:
```
geometry/extcalc/  →  calculus/  (NumericalGradient, GaussLegendre)
                  →  linalg/    (MatMul, LU)
                  →  geometry/  (TriangleArea2D + 205-SO(3))
                  →  geometry/diffgeo/  (Christoffel for E20 on curved M; test-only)
                  →  geometry/dec/      (194-D11 Whitney bridge for E14)
                  →  constants/ (ε₀, μ₀, c for E18 Maxwell)
                  →  topology/persistent/ (test-only Betti cross-check)
```

---

## 6. LOC roll-up

| Tier | Primitives | Source | Test | Cumulative |
|------|-----------|--------|------|------------|
| 1 (1-day) | E1, E2, E5, E10 | 480 | 320 | 480 |
| 2 (Hodge mech) | E3, E4, E6, E7, E8, E9 | 640 | 480 | 1120 |
| 3 (Stokes/dR) | E11, E12, E13, E14, E15 | 810 | 540 | 1930 |
| 4 (gauge+char) | E16, E17, E18, E19, E20 | 960 | 660 | 2890 |

Total: **~3050 LOC src + ~2000 LOC test**. (Re-homes from 207 counted once here, leaving 207 ~1180 LOC lighter.)

---

## 7. Cross-language pinning targets

- **E2 wedge antisymmetry**: random α, β on R⁵ → α∧β + β∧α = 0 to 1e-15 (exact shuffle).
- **E5 d on monomials**: `d(x^i x^j) = j x^i x^{j−1} dx + i x^{i−1} x^j dy` exact 1e-15 with analytic ∂; central-difference variant matches at O(h^{1.5}·ε^{1/3}) ≈ 6e-6.
- **E10 d²=0**: 100 random polynomial 0-/1-/2-forms on R⁴ → `‖d²ω‖² < 1e-20`.
- **E12 Stokes Green**: ∮_circle (x dy − y dx) = 2πr² to 1e-9 with Gauss-Legendre N=20.
- **E12 Stokes divergence**: ∮_sphere (x dy∧dz + y dz∧dx + z dx∧dy) = 4πr³ to 1e-7.
- **E15 Betti S²**: β = (1,0,1) integer-equality across this slot + 194-D8 + 156-topology.
- **E15 Betti T²**: β = (1,2,1) same byte-equality contract.
- **E16 symplectic preservation**: 204's leapfrog-Kepler 10⁶ steps → ‖φ*ω − ω‖ < 1e-12; RK4 drifts ~10^{−6} (falsifies non-symplectic).
- **E18 Maxwell vacuum**: A=(0, A₀cos(k(z−ct)), 0, 0) → d★F = 0 to round-off.
- **E19 abelian = Maxwell**: G=U(1) byte-equality with E18 (independent code paths).
- **E20 Dirac monopole on S²**: F = (q/2)sin θ dθ∧dφ → ∫_{S²} F/(2π) = q exact integer.

---

## 8. Out-of-scope deferrals

Currents/distributional forms (de Rham 1955); spin structures + Dirac forms; non-commutative DG (Connes 1994); stratified/orbifold de Rham; equivariant cohomology + Cartan model 1950; L²-cohomology + Hodge on non-compact; Floer homology; gerbes + 2-bundles. All v2.

---

## 9. Precision hazards

- **E1 multi-index sort invariant**: every `KForm.Set` must canonicalise to strictly-increasing index else wedge picks up wrong sign and silently produces garbage. Mandatory invariant check.
- **E2 wedge factorial convention**: `r! s!` vs `(r+s)!` differs by Choose(r+s,r). Pin Lee §14 (no extra factorial in wedge).
- **E5 finite-difference accuracy**: O(h²) → at h ≈ 6e-6 the d² round-trip hits 1e-9, not 1e-15. For round-off pinning require analytic ∂ω.
- **E8 Hodge sign**: `★★ω = (−1)^{k(n−k)+s}ω`, s=0 Riemannian, s=1 Lorentzian. Pin in docstring.
- **E11 orientation**: `√|g|` has implicit + sign; reverse-orientation flips integral sign. Caller pins.
- **E12 ∂M boundary orientation**: outward-normal convention; sign error of 2× without it.
- **E14 Hodge decomp uniqueness**: closed manifold (compact, no boundary). On manifolds-with-boundary, additional Friedrichs/Morrey 1955-1956 boundary-condition forms enter — beyond v0.
- **E18 Maxwell SI vs Gaussian**: pin SI throughout, consistent with `constants/physics.go` ε₀, μ₀.
- **E19 F = dA + A∧A sign**: physics convention F = dA + A∧A (since `(A∧A)_{μν} = 2[A_μ, A_ν]` halves the bracket). Mathematician's convention sometimes drops the half. Pin docstring.
- **E20 trace normalisation**: fundamental-rep tr vs adjoint-rep differ by 2N for SU(N). Pin "fundamental rep, tr(T_a T_b) = ½δ_ab".

---

## 10. Cross-link summary

- **194-em-geometry**: complementary discrete cochain version on meshes; bridge via Whitney-de Rham (194-D11). Co-ship strongly recommended.
- **207-new-diff-geo**: re-home D15–D20 (forms axis) + D22-D23 (Cartan structure equation) here. 207 narrows to Christoffel/Riemann/geodesic/Killing/Einstein.
- **204-symplectic**: ships integrators preserving E16; this slot ships ω + the test that 204 falsifies non-symplectic against.
- **205-lie-groups**: SO(3)/SE(3) exp/log gates non-abelian E17/E19/E20.
- **156-topology-persistent**: cross-validates E15 Betti via simplicial homology over F_2. Integer-equality contract.
- **097-linalg-missing**: SparseEigen gates direct continuous spectral E14; 194-DEC bridge unblocks practical case.
- **092-infogeo-T2**: E16 symplectic on T*M for RMHMC downstream.

---

## 11. Verdict

**SHIP Tier-1 (E1+E2+E5+E10) ~480 LOC** as 1-sprint standalone PR — saturates R-MUTUAL-CROSS-VALIDATION 3/3 against 194-DEC.

**SHIP Tier-2 (E3-E9 minus E5/E10) ~640 LOC** as 2nd sprint — Hodge mechanics + Cartan magic.

**SHIP Tier-3 (E11-E15) ~810 LOC** as 3rd sprint — Stokes witness + Poincaré constructive + Hodge decomposition + Betti cross-validation.

**SHIP Tier-4 partial (E16+E18 abelian) ~250 LOC** as 4th sprint — symplectic 2-form (validates 204) + Maxwell-as-F (closes em/ field-formulation gap).

**DEFER but design Tier-4 non-abelian (E17+E19+E20) ~860 LOC** — pair with 205-L1+L2 landing.

**DROP**: currents, gerbes, Floer, equivariant, L² Hodge — explicit v2.

Total v0: ~2180 LOC src + 1340 LOC test (Tiers 1–3 + abelian Tier-4). Total v1 with 205: ~3050 LOC src + 2000 LOC test.

Single highest-leverage 1-day project: **E1+E2+E5+E10 ~480 LOC** — first continuous Λ^k primitive in reality, saturates 3/3 cross-validation via Whitney-de Rham bridge to 194-DEC. Single highest-leverage cutting-edge piece: **E18+E19+E20 ~610 LOC** — symbolic Maxwell + Yang-Mills + characteristic classes, the singular reality competitive moat (no zero-dependency Go library ships symbolic Yang-Mills curvature 2-form with byte-for-byte cross-language golden-file contract).

End of report. 20 primitives, ~3050 LOC src, ~2000 LOC test, one new sub-package `geometry/extcalc/`, zero cycles, three soft blockers (097-SparseEigen, 205-Lie-algebra, 207-Christoffel). Keystone E5 ExteriorDerivative collapses Maxwell + Yang-Mills + Stokes + de Rham into one composition graph. Three explicit re-home recommendations against 207 (D15–D20, D22, D23) tighten 207's scope and concentrate the Λ^k axis here.
