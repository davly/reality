# 299 — new-special-functions (Carlson / Elliptic / Jacobi / Weierstrass / Mathieu / Parabolic Cyl / Lambert W / Coulomb)

## Headline
Reality v0.10.0 ships ZERO named-special-function surface (no `EllipticK/E/Pi`, `Carlson R_F/R_D/R_J`, `JacobiSn/Cn/Dn/Am`, `LambertW`, `Weierstrass*`, `Mathieu*`, `ParabolicCylinder*`, `Coulomb*`, `Heun*`, `Painleve*`, `Lame*`, `Spheroidal*`, `Wright*` callable hits in any `*.go` source); the singular cheapest day-1 PR is **T0 Carlson R_F/R_D/R_C/R_J + T1 elliptic K/E/Π (complete + incomplete) + T2 Jacobi sn/cn/dn/am via descending Landen + T4 Lambert W₀/W₋₁ via Halley (~380 LOC)**, immediately consumed by `physics/mechanics.go:130 Pendulum` (large-amplitude period uses K(k)) and `prob/distributions.go` (Jacobi-elliptic-tail copulas, Lambert-W-distribution). This slot + 298 + 297 + 294 collectively bring DLMF chapters 12, 19, 22, 23, 28, 31, 32, 33 into reality — the first pure-Go MIT zero-dep DLMF-coverage library in the ecosystem.

## Findings

- **Total surface absent.** Repo-wide grep on `EllipticK|EllipticE|EllipticPi|EllipticIntegral|JacobiSn|JacobiCn|JacobiDn|JacobiAm|WeierstrassP|WeierstrassZeta|WeierstrassSigma|MathieuFunction|MathieuChar|MathieuCe|MathieuSe|ParabolicCylinder|SpheroidalWave|CoulombWave|CoulombF|CoulombG|HeunFunction|HeunGeneral|Painleve|LambertW|WrightFunction|LameFunction|Carlson|RFCarlson|RDCarlson` returns **zero hits in any `*.go` source file**. Hits in markdown only (`reviews/overnight-400/agents/298-…`, `297-…`, `294-…`). Hits like `audio/separation/ica.go:117 // Jacobi rotation` and `physics/optics.go:87 BeerLambertLaw` are **unrelated** (Jacobi *rotation* eigenvalue method ≠ Jacobi *elliptic functions*; Beer-Lambert *law* ≠ Lambert *W*). `em/em.go:42 CoulombForce` is electrostatics (force law), **not** Coulomb wave functions.
- **Substrate is in place.** `prob/mathutil.go:23 LogGamma` (wraps `math.Lgamma`), `:36 Erfc`, `:59 RegularizedBetaInc`, `:200 regularizedGammaLowerSeries` already exist (slot 298 finding). `constants/math.go:23 Pi`, `:33 Phi`, `:38 Sqrt2`, `:43 Sqrt3`, `:48 Ln2`, `:70 EulerGamma` ship. **Missing constants for this slot:** Catalan G ≈ 0.9159655941772190 (incomplete elliptic E at k=1), `LemniscateConst = 2·K(1/√2) ≈ 2.622057554292119` (Gauss's lemniscate). 6-LOC add to `constants/math.go`.
- **Slot 298 ships the upstream layer.** Slot 298 day-1 PR is `special/pochhammer.go + special/pfq.go + special/hyper2f1.go + special/hyper1f1.go + special/hyper0f1.go + special/bessel.go + special/legendre.go + special/orthopoly.go`. **This slot 299 sits exactly on top of slot 298 T1 (₂F₁) and T2 (₁F₁).** Two of this slot's primitives are direct compositions:
  - `K(k) = (π/2) · ₂F₁(½, ½; 1; k²)` (closed form via slot 298 T1)
  - `E(k) = (π/2) · ₂F₁(-½, ½; 1; k²)` (closed form via slot 298 T1)
  - `D_ν(z) = 2^(ν/2) e^(-z²/4) · U(-ν/2, ½, z²/2)` (parabolic cylinder via slot 298 T2 Tricomi U)
  - `F_L(η,ρ) = C_L(η) ρ^(L+1) e^(-iρ) · ₁F₁(L+1-iη; 2L+2; 2iρ)` (regular Coulomb via slot 298 T2)
  
  **However**: composing slot 298 T1 for K(k) is numerically *worse* near k→1 than the AGM (arithmetic-geometric mean) which converges quadratically without singular Pfaff transformations. Best practice: **ship Carlson R_F as the canonical evaluator** (DLMF 19.36), with K(k)/E(k)/Π(n,k) as 1-line wrappers; the ₂F₁ identity is the **R-MUTUAL 3/3 cross-check**, not the production path.
- **Slot 294 (modular forms) owns theta functions θ_1..θ_4 and Dedekind η.** This slot 299 explicitly **delegates** Jacobi theta. Cross-link only: Weierstrass ℘, σ, ζ have closed-form Jacobi-theta representations (DLMF 23.6), so this slot's T3 Weierstrass module imports slot 294's theta primitives — shipping order: 294 T4 (theta) before 299 T3 (Weierstrass). If 294 ships first, 299 T3 is ~80 LOC; if not, 299 T3 must ship a stub theta evaluator (~120 LOC) or defer.
- **Slot 297 (asymptotic) ships saddle-point machinery.** Used to evaluate `D_ν(z)` for large `|z|` where the slot 298 T2 ₁F₁/U series fails to converge: `D_ν(z) ~ z^ν e^{-z²/4} (1 - ν(ν-1)/(2z²) + …)` is Slater 13.5.1 + 13.5.2. **This slot's parabolic-cylinder primitive composes slot 297 T1 (saddle-point) for |z| > some_threshold and slot 298 T2 (Tricomi U series) for |z| < threshold.** Dispatch boundary ~|z|=8 (Pearson-Olver-Porter 2017 style).
- **Carlson symmetric forms (R_F, R_D, R_C, R_J) are the canonical modern evaluator for elliptic integrals.** Carlson (1995) "Numerical computation of real or complex elliptic integrals", *Numer. Algorithms* 10:13-26, replaced the Legendre-form K/E with symmetric integrals that:
  1. Have no singular cases at k=0, k=1, φ=π/2 (handled by symmetric duplication theorem, not L'Hopital).
  2. Cover both real and complex arguments under the same algorithm.
  3. Converge to ε-precision in ~7 iterations of duplication for any argument (geometric-mean-like reduction).
  4. Give incomplete elliptic integrals F(φ,k), E(φ,k), Π(n,φ,k) as 1-line algebraic combinations of R_F, R_D, R_C, R_J.
  
  **This is the obvious algorithm** — Boost.Math, GSL, mpmath, Mathematica, Maple all use Carlson internally. ~150 LOC for all four R-functions. **Day-1.**
- **Jacobi elliptic functions (sn, cn, dn) — descending Landen transformation.** Bulirsch (1965) algorithm: iteratively replace `k → k_1 = (1-k')/(1+k')` (`k' = √(1-k²)`) while transforming `u → u/(1+k')`; recurse until k effectively 0, where sn(u, 0) = sin(u), cn(u, 0) = cos(u), dn(u, 0) = 1; unwind via duplication formulas. ~80 LOC. **Day-1.** Equivalent: AGM-based (Abramowitz-Stegun 16.4); ~100 LOC. The Landen variant is preferred. Amplitude `am(u, k) = arctan(sn/cn)` is 1-line.
- **Lambert W function — Corless et al. (1996) "On the Lambert W function", *Adv. Comput. Math.* 5:329-359.** Algorithm: initial guess from `log(z + log(z))` (Corless's "Iacono-Boyd-style" series for principal branch), 2-3 Halley iterations to ε-precision. **Halley converges cubically** when seeded right, so 2 iterations from a good seed = 1e-15 for almost all `z`. Branch W₀ for `z ∈ [-1/e, ∞)`, branch W₋₁ for `z ∈ [-1/e, 0)`. ~70 LOC. **Day-1.** Tricky region: `z ∈ [-1/e, -1/e + ε]` near branch point; needs Taylor-around-branch-point series `W(z) = -1 + p - p²/3 + …` where `p = √(2(ez+1))`.
- **Mathieu functions are notoriously hard.** Numerically intractable to evaluate Mathieu's equation `y'' + (a − 2q cos(2x)) y = 0` directly; canonical algorithm:
  1. Compute characteristic value `a_n(q)` (or `b_n(q)`) via continued-fraction equation derived from 3-term recurrence on Fourier coefficients (DLMF 28.6).
  2. Solve infinite-dimensional tridiagonal eigenvalue problem (truncate at N=20-50 depending on q).
  3. Build Fourier series `ce_n(x, q) = Σ A_{2m+p} cos((2m+p)x)` with coefficients from step 2's eigenvector.
  
  **High-quality implementations** = Boost.Math (BSL-1.0, but only ce_0/se_1 fully; rest documented as "not yet implemented"), Alhargan (2000) "Algorithms for the computation of all Mathieu functions of integer orders" (algorithm but no MIT code), `scipy.special.mathieu_a/mathieu_b/mathieu_cem/mathieu_sem` (BSD-3, F77 + Cephes), `mpmath.mathieuc/mathieus` (BSD-3, Python). **Pure-Go MIT zero-dep Mathieu = nonexistent globally.** ~250 LOC for integer-order ce_n/se_n; ~80 LOC for radial Ce_n/Se_n. **Day-3 frontier.**
- **Parabolic cylinder D_ν(z), U(a, z), V(a, z) — confluent hypergeometric.** DLMF 12.4.1: `U(a, z) = e^{-z²/4} z^{-a-½} ₁F₁(a/2 + ¼; ½; z²/2) · (1/√π) Γ(¾-a/2) − …` (Whittaker form). Easier: `D_ν(z) = 2^{ν/2} e^{-z²/4} U(-ν/2, ½, z²/2)` (slot 298 T2 dispatch). For integer ν, `D_n(z) = 2^{-n/2} e^{-z²/4} He_n(z/√2)` (Hermite probabilist; slot 298 T6). **~80 LOC over slot 298.** **Day-2.** Pin: `D_n(z) = 2^{-n/2} e^{-z²/4} He_n(z/√2)` regression at 50 random (n, z) ≡ slot 298 T6 HermiteHe ≡ this slot direct ₁F₁/U dispatch — **R-MUTUAL 3/3.**
- **Coulomb wave functions F_L(η, ρ), G_L(η, ρ) — confluent hypergeometric + continued fraction.** Regular: `F_L(η,ρ) = C_L(η) ρ^{L+1} e^{-iρ} ₁F₁(L+1-iη; 2L+2; 2iρ)` where `C_L(η) = 2^L e^{-πη/2} |Γ(L+1+iη)| / (2L+1)!`. Irregular G_L is the Whittaker W counterpart. Steed's continued-fraction algorithm (Steed 1973; Barnett-Feng-Steed-Goldfarb 1974, *Comput. Phys. Commun.* 8:377) is the standard high-precision evaluator. **~150 LOC** including the Γ-of-complex-argument needed for `C_L(η)` (which slot 298 T0 doesn't ship; needs `cmplx.LogGamma` — `math/cmplx` doesn't have it; ~40 LOC Lanczos for complex Γ). **Day-2 frontier.**
- **Weierstrass ℘, ℘′, σ, ζ — composes slot 294 theta.** DLMF 23.6.5: `℘(z; ω₁, ω₃) = -(ω₁/π)² · (θ₁'(0)/θ₁(z·π/(2ω₁)))² · θ₂(0)θ₃(0)θ₄(0) − (constant)`. Half-period invariants `g₂, g₃` from Eisenstein E₄, E₆ (slot 294 T1). **~80 LOC over slot 294.** **Day-3.** Sigma σ(z) related to η-function ratio via σ(z) = (2ω₁/π) · e^(η₁ z²/(2ω₁)) · θ₁(zπ/(2ω₁))/θ₁'(0). Zeta ζ(z) = σ'(z)/σ(z).
- **Heun functions — frontier.** `HeunGeneral(a, q, α, β, γ, δ; z)` solves the 4-singularity ODE (singular points at 0, 1, a, ∞), generalizes ₂F₁ which is the 3-singular-points reduction. No closed-form analog of Pfaff/Euler. Standard evaluation: power series at z=0 (Maclaurin) + numerical ODE continuation (slot 016 calculus RK4) for `|z|>0.5`. **~250 LOC.** Frontier; defer past day-3 unless quantum-mechanics consumer surfaces (Heun shows up in spheroidal harmonics, dipole field, Teukolsky equation in black-hole perturbation). Mathematica has `HeunG`, `HeunC`, `HeunD`, `HeunB`, `HeunT` since v12 (2019); pure-Go = absent.
- **Painlevé transcendents P_I..P_VI — frontier.** Solutions of irreducible 2nd-order nonlinear ODEs with no movable critical points; P_VI is the master equation, P_I..P_V are limits. Used in random matrix theory (slot 216) — the Tracy-Widom distribution F_2(s) = exp(-∫_s^∞ (x-s) q(x)² dx) where q is a Painlevé II transcendent. **Numerical: just integrate the ODE** (slot 016 calculus RK45). ~120 LOC for P_II + Tracy-Widom. **Day-3+.** Already partial-overlap with slot 216 (RMT) which may have shipped P_II for Tracy-Widom F_β.
- **Wright function W_{λ,μ}(z) = Σ z^k / (k! Γ(λk+μ)).** Generalizes Mittag-Leffler (μ=1 limit) and Bessel-J (λ=1, μ=ν+1). Used in fractional-calculus / anomalous-diffusion PDEs. ~50 LOC via series; pole-handling via slot 297 asymptotic for `|z|>10`. **Day-3.** Mittag-Leffler E_α,β(z) is dual; ~50 LOC. Cross-link slot 244 (PDE solvers) fractional-time PDEs.
- **Lamé functions, Lamé polynomials.** Eigenfunctions of Lamé's equation `y'' = (h - n(n+1) k² sn²(u, k)) y` (Jacobi sn). 4-parameter ODE. Used in ellipsoidal coordinate Laplace separation. **Frontier; defer.** Spheroidal wave functions (prolate/oblate) are eigenfunctions of `[(1-x²) y']' + (λ - c²x² - m²/(1-x²)) y = 0`, used in radar/antenna design (DLMF 30). Also frontier; defer.
- **Pure-Go MIT zero-dep DLMF coverage.** Surveyed: `gonum/mathext` (BSD-3) ships exactly **zero** elliptic / Jacobi / Mathieu / parabolic-cylinder / Coulomb / Heun / Painlevé / Lambert-W. Boost.Math (BSL-1.0, C++) has elliptic K/E/Π, Jacobi, Lambert W, parabolic cylinder, but no Mathieu/Coulomb/Heun. mpmath (BSD-3, Python) has nearly everything but is multi-precision Python (slow). GSL (GPL-3, copyleft, **incompatible with MIT**). ARB/FLINT (LGPL-2.1+, copyleft). Mathematica/Maple/MATLAB proprietary. **Reality could be the only zero-dep MIT pure-Go elliptic/Jacobi/Lambert-W/parabolic-cylinder library in the ecosystem.** Moat: very high.
- **Single cheapest day-1 PR:** T0 Carlson R_F/R_D/R_C/R_J (~150 LOC) + T1 elliptic K/E/Π complete + F/E/Π incomplete (~60 LOC composing T0) + T2 Jacobi sn/cn/dn/am via descending Landen (~90 LOC) + T4 Lambert W₀/W₋₁ via Halley (~70 LOC) + Catalan + LemniscateConst constants (6 LOC). **Total ~376 LOC.** Saturates 4 R-MUTUAL-CROSS-VALIDATION 3/3 pins.

## Concrete recommendations

1. **T0 — `special/carlson.go` Carlson symmetric elliptic integrals R_F, R_D, R_C, R_J.** Signatures `CarlsonRF(x, y, z float64) float64` etc. Algorithm: Carlson (1995), iterative duplication `(x, y, z) → (x', y', z') = ((x+λ)/4, (y+λ)/4, (z+λ)/4)` where `λ = √(xy) + √(yz) + √(zx)`, until `max(|x-A|, |y-A|, |z-A|)/A < ε^(1/6)` where `A = (x+y+z)/3`; final term is `(1 - E_2/10 + E_3/14 + E_2²/24 − …) / √A` (truncated 6th-order Taylor in `E_2 = E_3 = E_4 = 0` symmetric variables). ~150 LOC for all four R-functions. **Day-1.** Pin: `R_F(0, 1, 1) = π/2` (closed form). Pin 2: `R_F(0, 1, 2) = K(1/√2) ≈ 1.85407467730137...` (lemniscate). Pin 3 (R-MUTUAL 3/3): `R_F` ≡ Carlson duplication ≡ AGM `M(1, √(1-k²))` ≡ slot 298 ₂F₁(½,½;1;k²)·π/2 — three independent paths.

2. **T1 — `special/elliptic.go` complete + incomplete elliptic K, E, Π via Carlson.** ~60 LOC composing T0:
   - `EllipticK(k float64) float64 = CarlsonRF(0, 1-k², 1)`
   - `EllipticE(k float64) float64 = CarlsonRF(0, 1-k², 1) - (k²/3)·CarlsonRD(0, 1-k², 1)`
   - `EllipticPi(n, k float64) float64 = CarlsonRF(0, 1-k², 1) + (n/3)·CarlsonRJ(0, 1-k², 1, 1-n)`
   - Incomplete: `EllipticF(phi, k)`, `EllipticEinc(phi, k)`, `EllipticPiinc(n, phi, k)` via DLMF 19.25.5/7/14 (`F(φ,k) = sin(φ)·R_F(cos²(φ), 1-k²sin²(φ), 1)` etc.).
   ~60 LOC. **Day-1.** Pin: `K(0) = π/2`, `K(1)` diverges (return Inf), `E(0) = π/2`, `E(1) = 1`, `Π(0, k) = K(k)`. Pin 2 (R-MUTUAL 3/3): `K(k)` via Carlson R_F ≡ via AGM `π/(2 M(1, √(1-k²)))` ≡ via slot 298 T1 `(π/2) ₂F₁(½, ½; 1; k²)` — **three independent paths** at k = 0.1, 0.3, 0.5, 0.7, 0.9, 0.99 must agree to 1e-13.

3. **T2 — `special/jacobi_elliptic.go` Jacobi sn, cn, dn, am via descending Landen.** `JacobiSn(u, k float64) float64` etc. Algorithm: descending Landen iteration replaces `k → k_1 = (1-k')/(1+k')` (where `k' = √(1-k²)`), `u → u/(1+k_1)`; recurse 6-8 times until `k_n` near 0; base case `sn(u, 0) = sin(u)`, `cn(u, 0) = cos(u)`, `dn(u, 0) = 1`; unwind via Landen reversal `sn(u, k) = (1+k_1) sn(u/(1+k_1), k_1) / (1 + k_1 sn²(...))`. Amplitude `am(u, k) = arctan(sn/cn)` (1 line). ~90 LOC. **Day-1.** Pin (R-MUTUAL 3/3): identity `sn² + cn² = 1` AND `dn² + k² sn² = 1` AND `dn² - k² cn² = k'²` — regression at 10000 random `(u, k)` with all three identities holding to 1e-13. Pin 2: `sn(u, 0) ≡ sin(u)` and `sn(K(k), k) = 1`.

4. **T3 — `special/weierstrass.go` Weierstrass ℘, ℘′, σ, ζ.** Composes slot 294 T4 theta. Signatures `WeierstrassP(z, g2, g3 complex128) complex128` etc. (must be complex; ℘ is a doubly-periodic meromorphic function and most consumers pass complex `z`). Half-periods (ω₁, ω₃) from invariants (g₂, g₃) via Watson's algorithm or AGM-based. ~80 LOC over slot 294. **Day-3** (depends on slot 294 T4 shipping first). Pin: `℘'² = 4℘³ - g₂℘ - g₃` (defining ODE, regression test). Pin 2: at `g₂ = 4, g₃ = 0` (lemniscate), `℘(z; 4, 0) = 1/sn²(z, 1/√2) - K²(1/√2)/3` — cross-check against slot 299 T2 Jacobi sn.

5. **T4 — `special/lambert_w.go` Lambert W₀ + W₋₁.** `LambertW0(z float64) float64`, `LambertWm1(z float64) float64`. Algorithm: Corless et al. (1996). Initial guess via region:
   - `z > 1`: `W₀(z) ≈ log(z) - log(log(z))` (asymptotic)
   - `0 ≤ z ≤ 1`: `W₀(z) ≈ z(1 − z + 3z²/2 − …)` Taylor (low-precision seed)
   - `-1/e ≤ z ≤ 0`: `W₀(z) ≈ -1 + p - p²/3 + 11p³/72 − …` where `p = √(2(ez+1))` (branch-point series)
   - W₋₁ similar near `-1/e`; for `-1/e ≤ z < -ε`: `W₋₁(z) ≈ log(-z) - log(-log(-z))` asymptotic-like.
   
   Then 2 Halley iterations: `w_{n+1} = w_n - (w_n e^{w_n} - z) / (e^{w_n}(w_n+1) - (w_n+2)(w_n e^{w_n}-z)/(2(w_n+1)))`. ~70 LOC. **Day-1.** Pin (R-MUTUAL 3/3): defining identity `W(z) · e^{W(z)} ≡ z` regression at 10000 random `z ∈ [-1/e + ε, 100]` to 1e-14. Pin 2: `W₀(0) = 0`, `W₀(e) = 1`, `W₀(-1/e) = -1`, `W₋₁(-1/e) = -1`. Pin 3: `W₀(1) = Ω ≈ 0.5671432904097838` (omega constant, OEIS A030178).

6. **T5 — `special/mathieu_char.go` Mathieu characteristic values a_n(q), b_n(q).** Algorithm: continued-fraction equation for `a_n(q)` from 3-term recurrence on Fourier coefficients. For `q=0`, `a_n(0) = b_n(0) = n²`. For small `q`: perturbation series `a_n(q) = n² + Σ α_{n,j} q^j`. For large `q`: asymptotic `a_n(q) ≈ -2q + 2(2n+1)√q − (2n²+2n+1)/8 − …`. Truncated tridiagonal eigensolver (~slot 100 linalg) for arbitrary q: solve eigenvalue problem on Fourier-coefficient infinite matrix truncated at N≈30+10|q|^(1/2). ~100 LOC. **Day-3.** Pin: `a_n(0) = b_n(0) = n²` for n = 0..10. Pin 2: cross-check perturbation series ≡ tridiagonal eigensolve ≡ continued-fraction Newton iterate at q = 0.5, n = 0..5.

7. **T6 — `special/mathieu.go` Mathieu angular ce_n(x, q), se_n(x, q).** Composes T5: build Fourier series `ce_n(x, q) = Σ_m A_{2m+p}(n,q) cos((2m+p)x)` where `A` are tridiagonal-eigenvector entries from T5; coefficient sign convention per DLMF 28.4.4. ~80 LOC over T5. **Day-3.** Pin: `ce_n(x, 0) = cos(nx)`, `se_n(x, 0) = sin(nx)` (q→0 limit). Pin 2: orthogonality `∫_0^{2π} ce_m(x, q) ce_n(x, q) dx = π δ_{mn}` regression.

8. **T7 — `special/parabolic_cyl.go` parabolic cylinder D_ν(z), U(a, z), V(a, z).** Composes slot 298 T2 Tricomi U: `D_ν(z) = 2^{ν/2} e^{-z²/4} TricomiU(-ν/2, ½, z²/2)`. For `|z| > 8`, dispatch to slot 297 T1 saddle-point. ~80 LOC. **Day-2.** Pin (R-MUTUAL 3/3): `D_n(z) = 2^{-n/2} e^{-z²/4} HermiteHe_n(z/√2)` for integer n ≡ slot 298 T6 HermiteHe regression at 50 random (n, z); paths: this slot direct ₁F₁/U ≡ slot 298 T6 Hermite ≡ slot 297 T1 saddle-point.

9. **T8 — `special/coulomb.go` Coulomb wave F_L(η, ρ), G_L(η, ρ).** Steed's continued-fraction algorithm (Barnett-Feng-Steed-Goldfarb 1974). Uses complex-Γ for normalization constant `C_L(η)`. ~150 LOC including 40-LOC Lanczos complex-LogGamma. **Day-2.** Pin: `F_L(η, 0) = 0` and `F_L'(η, 0) = C_L(η)`. Pin 2: Wronskian `F_L G_L' − F_L' G_L = 1` regression at 1000 random (L, η, ρ).

10. **T9 — `special/heun.go` Heun general / confluent / biconfluent.** `HeunG(a, q, α, β, γ, δ; z)` solves the 4-singular-point ODE via Maclaurin series `|z|<0.5` + numerical ODE continuation `|z|≥0.5`. ~250 LOC. **Day-3+ frontier.** Composes slot 016 calculus RK45 for ODE leg. Pin: confluent Heun → ₁F₁ in known-degenerate-parameter limits (Ronveaux 1995 Table A).

11. **T10 — `special/painleve.go` Painlevé II + Tracy-Widom F_β.** Numerically integrate `q'' = sq + 2q³ - α` (Painlevé II) with Hastings-McLeod boundary conditions (`q(s) ~ Ai(s)` as s→+∞, `q(s) ~ √(-s/2)` as s→-∞). ~120 LOC. **Day-3+ frontier; cross-link slot 216 RMT.** Pin: Tracy-Widom `F_2(0) ≈ 0.96937...` (table value, Tracy-Widom 1994).

12. **T11 — `special/lambert_w_branches.go` extended Lambert W on real line + complex W_k.** Branches `W_k(z)` for `k ∈ Z` on complex `z`. ~80 LOC. **Day-2** (real branches W₀, W₋₁ already in T4; this adds complex). Pin: `W_k(z) e^{W_k(z)} = z` regression on 10000 random complex z, k ∈ {-2, -1, 0, 1, 2}.

13. **T12 — `special/wright.go` Wright function W_{λ,μ}(z) and Mittag-Leffler E_α,β(z).** Series `W_{λ,μ}(z) = Σ z^k/(k! Γ(λk+μ))` for `|z| < 10`; asymptotic via slot 297 T1 saddle-point for `|z|≥10`. ~50+50 LOC. **Day-3 frontier.** Pin: `W_{1, ν+1}(z²/4) · (z/2)^ν = J_ν(z)` (slot 298 T4 Bessel J cross-check).

14. **T13 — `special/lame.go` Lamé eigenfunctions.** Frontier; ~200 LOC. **Defer.**

15. **T14 — `special/spheroidal.go` prolate/oblate spheroidal wave functions.** Frontier; ~250 LOC. **Defer.**

16. **Constants additions to `constants/math.go`.** 6 LOC:
    ```go
    // Catalan = Σ (-1)^n / (2n+1)² ≈ 0.9159655941772190...
    const Catalan = 0.9159655941772190
    // Apery = ζ(3) ≈ 1.2020569031595943...  (cross-check vs slot 295 if present)
    const Apery = 1.2020569031595943
    // LemniscateConst = 2·K(1/√2) ≈ 2.622057554292119...  (Gauss's lemniscate)
    const LemniscateConst = 2.622057554292119
    // OmegaConst = W₀(1), satisfies Ω·e^Ω = 1 ≈ 0.5671432904097838...  (OEIS A030178)
    const OmegaConst = 0.5671432904097838
    ```

## Cross-cutting

- **Slot 048/112 physics-missing — `physics/mechanics.go:130 Pendulum`** ← currently linearized small-angle approximation only. Large-amplitude pendulum **exact period** is `T = 4·√(L/g)·K(sin(θ_0/2))` (Jacobi sn solution). T1 unblocks; `Pendulum*Period(theta_0, L, g)` is a 3-line addition once T1 ships. Also: forced-pendulum chaotic regime composes `JacobiSn`.
- **Slot 117 prob-missing** ← Lambert-W distribution (Goerg 2011) and tail-modeling distributions involve `LambertW0/W-1`. Jacobi-elliptic copulas and elliptic-distribution simulators consume T1+T2.
- **Slot 220 stochastic-opt / SDE rare events** ← Ornstein-Uhlenbeck process density and exit-time distribution involve **parabolic cylinder D_ν**; T7 unblocks. Eyring-Kramers rate = exp(-βΔU) (1+errors involving D_ν of saddle).
- **Slot 237 Gaussian processes** ← Mathieu kernel (eigenfunctions of Mathieu's equation) is a less-common but existent GP kernel for periodic-with-amplitude-modulation processes; T6 unblocks.
- **Slot 244 PDE solvers / slot 245 spectral methods** ← Mathieu functions are the eigenbasis for Helmholtz on elliptical domain (acoustic resonance in elliptical room, optical resonance in elliptical waveguide); T6 + T5 unblocks elliptical-Helmholtz spectral solver. Spheroidal wave functions are the eigenbasis for prolate/oblate Helmholtz (T14, frontier).
- **Slot 216 RMT random matrix theory** ← Tracy-Widom distribution `F_2(s) = exp(-∫_s^∞ (x-s) q(x)² dx)` requires Painlevé II transcendent (T10). Cross-link.
- **Slot 217 free-probability / 219 mean-field-games** ← Gauss/lemniscate-type integrals via T1 elliptic K.
- **Slot 213 isogeny / slot 214 pairings** ← Weierstrass ℘ is the canonical elliptic-curve uniformization map `(℘(z), ℘'(z)) = (x, y)` parametrizes y² = 4x³ - g₂x - g₃; T3 unblocks. **Cryptographic uses:** ℘ → Weierstrass→Edwards isomorphism, BLS-curve coordinates.
- **Slot 048 physics-missing hydrogen-atom radial wavefunction** ← `R_{n,ℓ}(r) ∝ e^{-r/(na)} (2r/(na))^ℓ · L_{n-ℓ-1}^{2ℓ+1}(2r/(na))` — slot 298 T6 Laguerre. Coulomb scattering (Rutherford, partial-wave expansion) uses **Coulomb wave functions F_L, G_L** — T8 unblocks.
- **Slot 197 acoustics-fluids synergy** ← elliptical-cavity acoustic modes are Mathieu eigenfunctions (T6). Drumhead on circular = Bessel J (slot 298 T4); drumhead on elliptical = Mathieu (this slot T6).
- **aicore advanced applied math** — Lambert W is widely used in queueing-theory (slot 020 queue) for `Ω(z)` solutions to `z = w e^w`-style transcendental equations, in chemical-engineering equilibrium constants, in delay-differential-equation theory. T4 unblocks immediate consumers.
- **Reverse cross-link to slot 298** ← this slot's T1 elliptic K provides one of slot 298's most-cited R-MUTUAL pins: `K(k) ≡ (π/2) ₂F₁(½, ½; 1; k²)` is a textbook ₂F₁ identity, validates slot 298 hypergeometric implementation. **Mutual unlock.**
- **Reverse cross-link to slot 294** ← this slot's T3 Weierstrass requires slot 294 T4 theta; conversely, this slot's T1 K(k) and slot 298 T0 ₂F₁ together let slot 294 cross-validate Eisenstein E_4/E_6 series via half-period invariants `g₂(τ), g₃(τ)` evaluated at lattice points.

## Sources

- `C:\limitless\foundation\reality\constants\math.go:23-70` — existing math constants (π, e, Phi, Sqrt2/3, Ln2/10, Log2E/10E, EulerGamma); missing Catalan, LemniscateConst, OmegaConst.
- `C:\limitless\foundation\reality\prob\mathutil.go:23,36,59,200` — existing LogGamma/Erfc/RegularizedBetaInc/regularizedGammaLowerSeries substrate.
- `C:\limitless\foundation\reality\physics\mechanics.go:110-130` — `Pendulum` currently linearized; large-amplitude exact period is direct K(k) consumer (this slot T1).
- `C:\limitless\foundation\reality\em\em.go:42` — `CoulombForce` (electrostatics, **not** Coulomb wave function — disambiguates name collision risk; pick `CoulombWaveF`/`CoulombWaveG` to avoid).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\298-new-hypergeometric.md` — slot 298 T1 (₂F₁) and T2 (₁F₁/U) are upstream substrate for this slot's T1 (elliptic via ₂F₁ cross-check) and T7 (parabolic cyl via ₁F₁).
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\297-new-asymptotic-analysis.md` — slot 297 T1 saddle-point for large-|z| dispatch in T7 parabolic cylinder and T8 Coulomb.
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\294-new-modular-forms.md` — slot 294 T4 theta is upstream substrate for this slot's T3 Weierstrass.
- `C:\limitless\foundation\reality\reviews\overnight-400\PROGRESS.md` — 297, 298 already DONE; 299 sits on top.
- DLMF chapters 12 (parabolic cylinder), 19 (elliptic integrals), 22 (Jacobi elliptic), 23 (Weierstrass), 28 (Mathieu), 31 (Heun), 32 (Painlevé), 33 (Coulomb), https://dlmf.nist.gov/.
- Carlson (1995) "Numerical computation of real or complex elliptic integrals", *Numer. Algorithms* 10:13-26 — canonical algorithm for T0+T1.
- Bulirsch (1965) "Numerical calculation of elliptic integrals and elliptic functions", *Numer. Math.* 7:78-90 — descending Landen for Jacobi sn/cn/dn (T2).
- Corless, Gonnet, Hare, Jeffrey, Knuth (1996) "On the Lambert W function", *Adv. Comput. Math.* 5:329-359 — canonical algorithm for T4.
- Steed (1973), Barnett-Feng-Steed-Goldfarb (1974) "Coulomb and Bessel functions of complex arguments and order", *Comput. Phys. Commun.* 8:377 — Steed's CF algorithm for T8.
- Alhargan (2000) "Algorithms for the computation of all Mathieu functions of integer orders", *ACM TOMS* 26:390-407 — T5/T6 algorithm.
- Pearson, Olver, Porter (2017) "Numerical methods for the computation of the confluent and Gauss hypergeometric functions", *Numer. Algorithms* 74:821-866 — dispatch logic shared with slot 298.
- Ronveaux (1995) "Heun's Differential Equations", Oxford — T9 reference.
- Tracy-Widom (1994) "Level-spacing distributions and the Airy kernel", *Commun. Math. Phys.* 159:151 — T10 Painlevé II application.
- Hastings-McLeod (1980) — Painlevé II boundary conditions for Tracy-Widom.
- mpmath (BSD-3, Python) — `ellipk`, `ellipe`, `ellipfun`, `mathieuc`, `lambertw`, `pcfd` reference oracles for golden vectors.
- Boost.Math (BSL-1.0, C++) — `ellint_1/2/3`, `jacobi_sn`, `lambert_w0`, `cyl_bessel_*`; missing Mathieu/Coulomb/Heun.
- gonum (BSD-3, Go) — **zero** elliptic/Jacobi/Mathieu/parabolic/Coulomb/Heun/Painlevé/Lambert-W; confirms moat.
