## 250 | new-mortar — Mortar methods, contact, Lagrange multipliers

**Summary line 1.** reality v0.10.0 ships **ZERO** computational-contact-mechanics surface — repo-wide grep on `contact|signorini|hertz|coulomb.*friction|tresca|active.set|semi.smooth|gap.function|stick.slip|kuhn.tucker|complementarity|nonpenetration|ncp|lcp|mlcp|wedge|adhesion|cohesive.zone|reynolds.equation|lubricat|frictionless|asperity|moreau` against `*.go` returns **zero callable matches** anywhere outside review-corpus and false-positive hits (`em/em.go` Coulomb-LAW for electrostatics ≠ Coulomb-FRICTION; `chaos/systems.go` `contact rate` = SIR-epidemic-β name-collision; `physics/materials.go` `TrescaStress` = von-Mises-vs-Tresca yield criterion ≠ Tresca-FRICTION-CONSTANT-FORCE; `physics/mechanics.go` `ElasticCollision` = perfect-elastic-1D-billiard ≠ elastic-CONTACT). The only adjacent pieces of substrate are (a) `physics/materials.go` `HookesLaw` σ = E ε scalar + `VonMisesStress` + `TrescaStress` yield criteria (~30 LOC, scalar, no tensor), (b) `optim/proximal/operators.go` `ProxNonneg` + `ProxBox` projection-prox operators (~30 LOC; these are *literally* the projection operators used in projected-Newton + projected-Gauss-Seidel for NCP/LCP-Signorini-discrete formulations), (c) `optim/proximal/admm.go` consensus-ADMM solver (~80 LOC; the modern split-Bregman backbone for Augmented-Lagrangian-contact, Glowinski-LeTallec-1989). Slot 250 is **STRICT-DOWNSTREAM of 247** (M5-cross-points + M12-MortarInterface + M13-L²-projection + M14-saddle-point-K=[[A B^T];[B 0]] + M16-Wohlmuth-1999-biorthogonal-dual-Lagrange-basis + M18-conservation-witnesses — slot 250 IS the contact-mechanics specialisation that consumes 247's mortar substrate as gap-function discretisation; cannot ship without 247-Tier-2 mortar landing first), **STRICT-DOWNSTREAM of 244** (D25-D26 P1-FE conforming baseline for the bulk-elasticity-stiffness-A inside the saddle-point K), **STRICT-DOWNSTREAM of 097** (T1-SparseMatrix CSR — every realistic 3D-contact problem operates on 10⁴–10⁸-DOF sparse mortar saddle-point), **PARTIAL-IDENTITY with 247-M16** (Wohlmuth-1999 dual-Lagrange basis is the production-grade choice for Hüeber-Wohlmuth-2005-CMAME semi-smooth-Newton-active-set-contact — the "mortar contact" name attaches at slot 250 specifically because 1996-2005 is when mortar-Lagrange-multipliers were applied to contact-Signorini-KKT and the dual-basis collapsed the indefinite saddle-point to a CG-friendly Schur complement *while preserving* the inequality-LCP-structure), **PARTIAL-OVERLAP with 249** (mortar = non-overlapping Schwarz with Lagrange-multiplier interface; contact-mortar adds the *one-sided inequality* λ ≥ 0 + complementarity λ·(g_h−Π u) = 0 making the bilinear-coupling problem a *variational inequality* not equality), **CROSS-LINK 159-em-FDTD** (none — contact is mechanical, not EM), **CROSS-LINK to optim/proximal** (ADMM + Augmented-Lagrangian for contact = Glowinski-LeTallec-1989; ProxNonneg + ProxBox projections directly used in projected-Gauss-Seidel-LCP), **CROSS-LINK 192-fluids-control** (none today; future FSI / lubricated-contact would compose Reynolds-equation-fluids × structural-contact). MASTER_PLAN slot 250 is named "Mortar methods, contact, Lagrange multipliers" and is the **terminal-contact-mechanics entry in the 240–250 cluster**, the canonical Wriggers-2006 *"Computational Contact Mechanics"* + Laursen-2002 *"Computational Contact and Impact Mechanics"* + Wohlmuth-2011-Acta-Numerica-20 *"Variationally consistent discretization schemes and numerical algorithms for contact problems"* discipline-defining-monograph subject area.

**Summary line 2.** Twenty-four contact-mechanics primitives **C1–C24** totalling ~3,200 LOC organized as **(a) Tier-0 Hertz analytic + closed-form contact ~360 LOC** (C1 `contact/hertz.go` Hertz-1882-JdRAM-92 ball-on-ball + ball-on-flat + cylinder-on-cylinder + cylinder-on-flat closed-form pressure-distribution p(r) = p_0·√(1−r²/a²) + contact-radius a = (3FR/4E*)^{1/3} + maximum-pressure p_0 = 3F/(2πa²) + Mindlin-1949-Cattaneo-1938 tangential-contact-stick-slip ~140 LOC, calls `physics.HookesLaw` for E* = E/(1−ν²); C2 `contact/coulomb.go` Coulomb-1781 friction-law |τ| ≤ μσ_n with stick-condition |τ_t| < μσ_n ⇒ velocity = 0 vs slip-condition |τ_t| = μσ_n ⇒ velocity ∥ −τ_t ~80 LOC, the 1781-canonical-static-and-kinetic-friction; C3 `contact/tresca.go` Tresca-1864 constant-friction-force-bound τ_max independent-of-normal-pressure ~60 LOC, the textbook simplification when μσ_n is replaced by a constant; C4 `contact/coulomb_wedge.go` Coulomb-wedge-condition for self-locking when friction-cone-angle exceeds slope-angle ~80 LOC, the analytic-pin for ramp-on-ramp test), **(b) Tier-1 Signorini variational inequality + active-set ~620 LOC** (C5 `contact/signorini.go` Signorini-1933 unilateral-contact-VI: find u ∈ K = {v : v·n ≤ g on Γ_C} such that a(u, v−u) ≥ ℓ(v−u) ∀v ∈ K — the **Signorini problem** = elasticity-with-contact = obstacle-problem-vector-valued, the founding-formulation of computational-contact-mechanics from 1933 ~140 LOC, validates against Brezzi-1972-AnnSNS-Pisa-2 well-posedness; C6 `contact/penalty.go` ε-penalty-method `K_ε(u, v) = a(u, v) + ε⁻¹·∫_{Γ_C} (g − u·n)_+ v·n` with ε → 0 limit recovering Signorini ~120 LOC, the 1970s-pre-mortar-pre-Lagrange-multiplier-method-of-Kikuchi-Oden-1988 still in production for explicit-dynamics; C7 `contact/lagrange_multiplier.go` exact-Lagrange-multiplier-formulation introduces λ = σ_n on Γ_C as primal-dual saddle-point min-max L(u, λ) = ½a(u,u) − ℓ(u) − ⟨λ, u·n − g⟩ with KKT λ ≥ 0, u·n ≤ g, λ(u·n − g) = 0 — the *exact* Signorini reformulation replacing the inequality-constrained-set K with bound-constrained-multiplier ~140 LOC, the production-grade method from Hilbert-1971 / Glowinski-LeTallec-1989; C8 `contact/augmented_lagrangian.go` Glowinski-LeTallec-1989 / Pietrzak-Curnier-1999-CMAME-177 augmented-Lagrangian L_α(u, λ) = L(u, λ) + α/2·(g − u·n)_+² combining penalty-stability with multiplier-exactness ~140 LOC, the post-1999 CMAME-frontier of robust-contact; C9 `contact/active_set.go` Hintermüller-Ito-Kunisch-2003-SISC-13 primal-dual active-set strategy treating contact as *equality* problem on active set A_k = {i : λ_i_k > 0} updating per-iteration A_{k+1} = {i : (λ_i + α(g_i − u_i·n)) > 0} — the textbook-canonical post-2003-active-set-SQP-for-contact ~120 LOC), **(c) Tier-2 mortar contact + dual-basis ~620 LOC** (C10 `contact/mortar/gap.go` discrete-gap function g_h = Π_h(g_master − u_master·n) + Π_h(u_slave·n) using 247-M13 L²-projection + 247-M16 dual-basis ~120 LOC, the post-1996-Hüeber-Wohlmuth-2005-CMAME-194 reformulation; C11 `contact/mortar/dual_lcp.go` LCP discretisation `M λ + N u = r, λ ≥ 0, g_h − Π u_master ≥ 0, λ·(g_h − Π u_master) = 0` with M diagonal-from-247-M16-dual-basis enabling element-wise complementarity solve ~140 LOC, the production-grade Wohlmuth-2011-AN-20 standard; C12 `contact/mortar/sliding.go` master-slave sliding-contact with surface-gap-evolution g_h(t) and search-radius-projection-onto-master-surface ~140 LOC; C13 `contact/mortar/3d_mortar_contact.go` Puso-Laursen-2004-CMAME-193 3D-mortar-contact patch-test-passing element-segmentation + face-pairing on tetrahedral interfaces ~140 LOC, the canonical 3D-contact-mortar mesh-segmentation algorithm; C14 `contact/mortar/contact_patch_test.go` Taylor-Papadopoulos-1991-Crisfield-2000 contact-patch-test = pin-down zero-penetration for uniform-pressure-loading on non-matching meshes ~80 LOC, the keystone-validation for any contact-FE code), **(d) Tier-3 semi-smooth Newton + LCP solvers ~480 LOC** (C15 `contact/ssn.go` Hintermüller-Ito-Kunisch-2003-SISC-13 / Hüeber-Stadler-Wohlmuth-2008-SISC-30 semi-smooth Newton on max(0, λ + α(g_h − Π u))-reformulation `F(u, λ) = 0` with B-subdifferential-Newton-iteration converging Q-superlinearly ~140 LOC, the post-2003 contact-state-of-the-art; C16 `contact/pgs.go` projected-Gauss-Seidel-LCP for `Mλ + b ≥ 0, λ ≥ 0, complementarity` element-wise update λ_i ← max(0, λ_i − ω(M_ii)⁻¹·(M_:i^T λ + b_i)) ~80 LOC, the standard-textbook-LCP-iterative-solver Cottle-Pang-Stone-1992; C17 `contact/lemke.go` Lemke-1965 pivotal-LCP-solver Cottle-Pang-Stone-1992-§4.4 finite-termination for P-matrix LCPs ~140 LOC, the canonical-direct-LCP-solver; C18 `contact/normal_map.go` Robinson-1992 normal-map-NCP-formulation `F(z) = z − Π_K(z − F(z))` with Newton-on-normal-map ~120 LOC, the alternative to ssn for robotic-contact-rigid-body-LCP), **(e) Tier-4 specialised contact frontier ~720 LOC ⊘ DEFER-mostly** (C19 `contact/cohesive.go` Xu-Needleman-1994 / Camacho-Ortiz-1996 cohesive-zone-elements with bilinear traction-separation law for fracture-as-decohesion-contact ~140 LOC ⊘ DEFER blocked-on-247-M14 + brittle-failure consumer; C20 `contact/dynamic.go` Laursen-Chawla-1997 dynamic-contact-Kuhn-Tucker for inertia + Moreau-1988 sweeping-process for impact + Newmark-α-with-contact-projection ~140 LOC ⊘ DEFER blocked-on-244-time-steppers; C21 `contact/lubricated.go` Reynolds-1886-equation coupling — Stokes-flow-thin-film between deformable surfaces, Patir-Cheng-1978 average-Reynolds for rough-surfaces ~140 LOC ⊘ DEFER cross-link-fluids-contact-FSI; C22 `contact/self_contact.go` Wriggers-2002 self-contact-detection via bounding-volume-hierarchy + sphere-tree-collision ~100 LOC ⊘ DEFER algorithmic-niche; C23 `contact/multi_body.go` multi-body-rigid-contact + Stewart-Trinkle-1996-IJNME-39 LCP for rigid-body-impact + Anitescu-Potra-1997 time-stepping ~120 LOC ⊘ DEFER game-engine-niche; C24 `contact/adhesion.go` Frémond-1987 adhesion + Raous-Cangémi-Cocou-1999-CMAME RCC-cohesive-friction-adhesion model ~120 LOC ⊘ DEFER frontier; C25 `contact/dg_contact.go` Wriggers-Heintz-Hesch-2008 DG-with-discontinuous-penalty-for-contact ~120 LOC ⊘ DEFER blocked-on-247-M7-SIP; C26 `contact/shell_contact.go` shell-on-shell-contact Konyukhov-Schweizerhof-2008 mortar-on-shell ~120 LOC ⊘ DEFER blocked-on-shells-FE-substrate-absent; C27 `contact/homogenisation.go` Bonnet-Naimark-1993 contact-homogenisation for periodic-rough-surfaces ~80 LOC ⊘ DEFER frontier; C28 `contact/fictitious_domain.go` Glowinski-Pan-Periaux-1994 fictitious-domain Lagrange-multiplier for embedded-boundary-contact ~100 LOC ⊘ DEFER blocked-on-244-immersed-FE). **SINGULAR-FOUNDATIONAL C5+C7 Signorini-VI + Lagrange-multiplier ~280 LOC** because every other primitive — penalty, augmented-Lagrangian, active-set, mortar-contact, ssn, LCP — is a *discretisation* or *solver* for the C5+C7 continuous-Signorini-Lagrange formulation; cannot ship anything else without these. **SINGULAR-MOAT C10+C11+C15 mortar-gap + dual-LCP + ssn ~400 LOC** because the *Hüeber-Wohlmuth-2005-CMAME-194 ssn-on-dual-mortar-LCP* is **the** production-grade-decision for the FE-contact community since 2005 — used in deal.II-step-41 + MFEM-mfem-lor + FreeFEM-LCP + Code_Aster-mortar-contact + DEAL.II-Trilinos-NOX — and the dual-Lagrange-basis (247-M16) collapsing M to diagonal makes the LCP element-wise solvable + the ssn-active-set converging Q-superlinearly is the engineering-defining decision that made non-conforming-mesh-contact viable beyond research codes; no zero-dep Go library anywhere ships this; the canonical reference is Wohlmuth-2011-Acta-Numerica-20 the discipline-defining monograph. **SINGULAR-CUTTING-EDGE C13 Puso-Laursen-2004 3D-mortar-contact ~140 LOC** because Puso-Laursen-2004-CMAME-193 is **the** post-2004 canonical-3D-mortar-contact algorithm used in commercial codes (ABAQUS-mortar-contact + LS-DYNA-tied-contact + Code_Aster-MORTAR + Sierra-SM-mortar-contact) and zero-dep Go absent everywhere; the original paper has 800+ citations. **SINGULAR-PEDAGOGICAL C1 Hertz-1882 ball-on-ball ~140 LOC** because the Hertz analytical contact-radius `a = (3FR/4E*)^{1/3}` and pressure `p(r) = p_0√(1−r²/a²)` is the textbook 1882-canonical-pin against which every numerical-contact code is validated (Wriggers-2006-§3.1 + Johnson-1985-*"Contact Mechanics"*-Cambridge-§3.4 the discipline-defining textbook), and it is *closed-form* requiring no FE-substrate to ship — making it the cheapest-1-day-shippable Tier-0 demo. **SINGULAR-2024-FRONTIER C18 normal-map-NCP Robinson-1992 + C20 dynamic-contact-Moreau-1988 sweeping-process ~260 LOC** the post-2000 LCP-NCP-frontier deployed in robotic-locomotion (MuJoCo + DART + RaiSim soft-contact + BulletPhysics) and granular-DEM-contact (LIGGGHTS + Yade); zero-dep Go absent. **CANDOR.** Computational-contact-mechanics is the **most-specialised Block-C entry in the 240–250 cluster** — narrower than mortar-FEM (247), narrower than domain-decomposition (249), narrower than multigrid (248). Wriggers-2006 monograph is 540 pages; Laursen-2002 is 460 pages; Wohlmuth-2011-AN-20 is 145 pages. Production codes: ABAQUS-Standard mortar-contact module ~30k LOC, Code_Aster-CONTACT ~25k LOC, MFEM-lor-step-41 ~3k LOC. reality cannot match these in scope, and the consumer-pull inside the reality stack is **near-zero today** — no aicore / Pistachio / Oracle / Sentinel service uses contact-mechanics, and there is no PDE substrate (244 must ship first) + no mortar substrate (247 must ship first) on which to even *demonstrate* these methods. The case for shipping at all rests on: (1) mathematical-completeness — the terminal Block-C-contact entry named in MASTER_PLAN; (2) downstream-ready when 247 ships — directly composes 247-M13/M14/M16 Wohlmuth-dual-mortar; (3) brand-prestige — first zero-dep Go library to ship Hüeber-Wohlmuth-2005 ssn-on-mortar + Puso-Laursen-2004 3D-mortar-contact; (4) Hertz-1882 ball-on-ball is the *cheapest-Tier-0 1-day-shippable* demo independent of all FE substrate.

Recommended placement **NEW package `contact/`** ~2,200 LOC (C1–C18 Tier-0/1/2/3 minus deferred Tier-4 C19–C28). Sub-package layout:

```
contact/
  hertz.go             # C1: Hertz-1882 ball-on-ball + ball-on-flat closed-form
  coulomb.go           # C2: Coulomb friction τ ≤ μσ_n
  tresca.go            # C3: Tresca constant-friction
  coulomb_wedge.go     # C4: Coulomb-wedge self-locking
  signorini.go         # C5: Signorini-1933 VI formulation
  penalty.go           # C6: ε-penalty method Kikuchi-Oden-1988
  lagrange_multiplier.go # C7: exact KKT saddle-point
  augmented_lagrangian.go # C8: Glowinski-LeTallec-1989
  active_set.go        # C9: Hintermüller-Ito-Kunisch-2003 PDAS
  mortar/
    gap.go             # C10: discrete gap g_h via 247-M13
    dual_lcp.go        # C11: LCP via 247-M16 dual-basis
    sliding.go         # C12: master-slave sliding contact
    threed.go          # C13: Puso-Laursen-2004 3D-mortar
    patch_test.go      # C14: Taylor-Papadopoulos contact-patch-test
  ssn.go               # C15: Hüeber-Wohlmuth-2008 ssn
  pgs.go               # C16: projected-Gauss-Seidel-LCP
  lemke.go             # C17: Lemke-1965 pivotal-LCP
  normal_map.go        # C18: Robinson-1992 normal-map NCP
  # DEFER Tier-4
  # cohesive.go        # C19: Xu-Needleman cohesive zone
  # dynamic.go         # C20: Laursen-Chawla dynamic + Moreau sweep
  # lubricated.go      # C21: Reynolds-coupling
  # self_contact.go    # C22: BVH self-detection
  # multi_body.go      # C23: Stewart-Trinkle rigid-LCP
  # adhesion.go        # C24: Frémond-1987 + RCC
  # dg_contact.go      # C25: DG-with-discontinuous-penalty
  # shell_contact.go   # C26: shell-on-shell mortar
  # homogenisation.go  # C27: Bonnet-Naimark periodic
  # fictitious_domain.go # C28: Glowinski-Pan-Periaux
```

Rationale for **NEW** `contact/` rather than nesting under `physics/contact/` or `fem/contact/`: contact-mechanics is a **discipline of its own** spanning physics (Hertz analytic, Coulomb-friction, Tresca, Signorini-VI), optimisation (KKT, Lagrange, Augmented-Lagrangian, PDAS, ssn, LCP, NCP — single-source from `optim/proximal/` for ProxNonneg + ProxBox), FE-discretisation (mortar coupling, 247-M13/M16 reuse), and dynamics (impact, sweeping, time-stepping deferred). Top-level `contact/` matches Wriggers-2006-monograph + Laursen-2002-monograph + ABAQUS / Code_Aster / MFEM / FreeFEM layout where contact is its own top-level subdomain. Sub-package precedent inside reality: `prob/copula/`, `optim/proximal/`, `optim/transport/`, `geometry/dec/` (proposed in 246), `fem/mortar/` (proposed in 247).

**CANDOR.** Recommendation: **ship Tier-0 alone (C1–C4 ~360 LOC) over 1 PR / 2 engineer-days** — this gives reality first-class **closed-form Hertz + Coulomb + Tresca + Coulomb-wedge** validating against the canonical 1882–1864–1781 analytical pins, requires *zero* FE substrate, depends on no other unshipped slot, and saturates R-HERTZ-CLOSED-FORM 4/4 + R-COULOMB-WEDGE-SELF-LOCK 3/3 + R-TRESCA-CONSTANT-FORCE 3/3 immediately. **Defer Tier-1+2+3 (C5–C18 ~1,720 LOC)** strict-blocked-on-247-mortar landing. **Defer Tier-4 (C19–C28 ~720 LOC)** entirely until concrete consumer-pull (no aicore consumer today). **Cheapest-1-day-shippable**: C1 Hertz alone ~140 LOC saturating R-HERTZ-CLOSED-FORM 4/4 against Johnson-1985-§3.4-closed-form pins (sphere-on-sphere a, p_0, max-shear-stress-depth z = 0.48a, contact-area πa²) for steel-on-steel + rubber-on-glass + ball-bearing-on-race + cylinder-on-flat-line-contact. **Highest-leverage architecturally**: PR-3 mortar-contact-Wohlmuth-dual-LCP C10+C11+C15 ~400 LOC, but maximally-blocked on 247-M5+M13+M14+M16 + 244-D25-D26 + 097-T1. **17 of 18 primitives unique to this slot** (C2 Coulomb-friction shares concept with `physics/materials.go` Tresca-yield-criterion via name only — the friction-Tresca and the yield-Tresca are *different* applications of the same shear-bound; cross-link single-source via doc-only).

---

## 0. State at HEAD (2026-05-09, v0.10.0) — verified by direct read

Repo-wide audit for contact / Signorini / Hertz / Coulomb-friction / Tresca-friction / penalty-method / Lagrange-multiplier / augmented-Lagrangian / active-set / semi-smooth-Newton / LCP / NCP / projected-Gauss-Seidel / Lemke / dual-basis-mortar-contact / Wohlmuth-Hüeber / Puso-Laursen / Kikuchi-Oden / Glowinski-LeTallec / Hintermüller-Ito-Kunisch / Pietrzak-Curnier / Wriggers / Laursen / Robinson-normal-map / Moreau-sweeping / cohesive-zone / Xu-Needleman / Camacho-Ortiz / Stewart-Trinkle / Anitescu-Potra / Frémond-adhesion / Reynolds-equation-lubrication / contact-patch-test / coulomb-wedge-self-locking surface — **zero callable matches** anywhere in `*.go` files outside review-corpus and false-positive name-collisions.

| Surface | Path | Contact-mechanics relevance |
|---|---|---|
| `em/em.go` Coulomb-LAW (electrostatics) | `em/em.go:1-25` | NAME-COLLISION ≠ Coulomb-FRICTION-1781 (mechanics) — different Coulomb |
| `physics/materials.go` `TrescaStress` yield-criterion | `physics/materials.go:54` | NAME-COLLISION ≠ Tresca-FRICTION-1864 (constant-shear-bound) — same Tresca but different application |
| `physics/materials.go` `HookesLaw` σ = E ε | `physics/materials.go:9-23` | USABLE — bulk-elasticity scalar for contact-elastic-modulus E* = E/(1−ν²) in C1-Hertz |
| `physics/materials.go` `VonMisesStress` | `physics/materials.go:36` | USABLE — yield criterion for contact-elastic-plastic transition (deferred, frontier) |
| `physics/mechanics.go` `ElasticCollision` | `physics/mechanics.go:103` | NAME-COLLISION ≠ elastic-CONTACT — perfect-billiard-1D-momentum-conservation, no surface-deformation |
| `optim/proximal/operators.go` `ProxNonneg` | `optim/proximal/operators.go:84` | DIRECTLY USABLE — projection onto λ ≥ 0 cone for C7-Lagrange-multiplier-KKT + C16-PGS-LCP element-wise updates |
| `optim/proximal/operators.go` `ProxBox` | `optim/proximal/operators.go:96-103` | DIRECTLY USABLE — projection onto box for C2-Coulomb-friction τ ∈ [−μσ_n, +μσ_n] sliding/sticking |
| `optim/proximal/admm.go` consensus-ADMM | `optim/proximal/admm.go:1-100` | USABLE — Glowinski-LeTallec-1989 augmented-Lagrangian-contact = Douglas-Rachford on f + g where f = elasticity, g = contact-indicator (C8) |
| `optim/rootfind.go` Newton-Raphson + bisection | `optim/rootfind.go` | USABLE — semi-smooth-Newton (C15) needs B-subdifferential-Newton requiring the existing Newton scaffolding |
| `physics/materials.go` `StressIntensityFactor` Irwin-1957 | `physics/materials.go:85` | USABLE — frontier cohesive-zone (C19, deferred) |
| `linalg.LUSolve` / `linalg.Cholesky` | `linalg/decompose.go` | USABLE — direct-solve in saddle-point K=[[A B^T];[B 0]] for small-N C7 |
| `linalg/sparse/` | -- | **ABSENT** — flagged 097-T1; blocks every realistic-N contact problem (dense-LU O((n_h+n_λ)³) blows up at N=10⁴ DOFs) |
| `fem/mortar/projection.go` 247-M13 | not yet shipped | STRICT-UPSTREAM — gap-function discretisation g_h via L²-projection master→slave |
| `fem/mortar/lagrange_mult.go` 247-M14 | not yet shipped | STRICT-UPSTREAM — saddle-point K assembly for C7 |
| `fem/mortar/dual_basis.go` 247-M16 | not yet shipped | STRICT-UPSTREAM — diagonal mass-matrix M enabling C11 element-wise LCP |
| `fem/mortar/cross_points.go` 247-M15 | not yet shipped | STRICT-UPSTREAM — multiplier-space-reduction for cross-points in 2D-contact |
| `pde/fem/p1_2d.go` 244-D26 | not yet shipped | STRICT-UPSTREAM — bulk-elasticity-stiffness A in K=[[A B^T];[B 0]] |
| Hertz-1882 ball-on-ball / cylinder-on-flat closed-form | -- | **ABSENT** — C1 ships |
| Coulomb-1781 friction-law | -- | **ABSENT** — C2 ships |
| Tresca-1864 constant-friction | -- | **ABSENT** — C3 ships |
| Signorini-1933 unilateral-contact-VI | -- | **ABSENT** — C5 ships |
| Penalty-method Kikuchi-Oden-1988 | -- | **ABSENT** — C6 ships |
| Lagrange-multiplier exact-KKT | -- | **ABSENT** — C7 ships |
| Augmented-Lagrangian Glowinski-LeTallec-1989 | -- | **ABSENT** — C8 ships |
| Active-set Hintermüller-Ito-Kunisch-2003 PDAS | -- | **ABSENT** — C9 ships |
| Mortar gap-function discretisation | -- | **ABSENT** — C10 ships |
| Mortar dual-LCP Hüeber-Wohlmuth-2005 | -- | **ABSENT** — C11 ships |
| 3D-mortar-contact Puso-Laursen-2004 | -- | **ABSENT** — C13 ships |
| Contact-patch-test Taylor-Papadopoulos-1991 | -- | **ABSENT** — C14 ships |
| Semi-smooth Newton Hüeber-Stadler-Wohlmuth-2008 | -- | **ABSENT** — C15 ships |
| Projected-Gauss-Seidel-LCP | -- | **ABSENT** — C16 ships |
| Lemke-1965 pivotal-LCP | -- | **ABSENT** — C17 ships |
| Robinson-1992 normal-map NCP | -- | **ABSENT** — C18 ships |
| Cohesive-zone Xu-Needleman-1994 | -- | **ABSENT** — C19 deferred |
| Dynamic-contact Moreau-1988 sweeping | -- | **ABSENT** — C20 deferred |
| Lubricated-contact Reynolds-equation | -- | **ABSENT** — C21 deferred |

**Cross-import edges that this slot creates:**
- `contact → physics` (HookesLaw + VonMises + Tresca-yield) for E* effective-modulus and yield-criteria
- `contact → optim/proximal` (ProxNonneg + ProxBox + ADMM) for projection-onto-cone-for-KKT, projected-Gauss-Seidel-LCP, augmented-Lagrangian
- `contact → optim/rootfind` (Newton-Raphson) for B-subdifferential semi-smooth Newton
- `contact → fem/mortar` (M13 + M14 + M15 + M16) STRICT-UPSTREAM dependency on 247
- `contact → linalg.LUSolve / Cholesky / GMRES` for saddle-point linear-solve
- `contact → linalg/sparse` (097-T1) for realistic-N

**Strict downstream consumers** of `contact/` substrate: NONE in 240–250 cluster (terminal-leaf). Future consumers anticipated:
- aicore-rigid-body-physics (C22 self-contact + C23 multi-body, deferred)
- Pistachio-deformable-character-contact (C13 3D-mortar + C20 dynamic, deferred)
- Sentinel-3D-collision (C18 normal-map NCP for rigid-body, deferred)
- 251-shape-optimisation contact-with-uncertainty (cross-link, deferred)

---

## 1. The eighteen Tier-0/1/2/3 primitives (Tier-4 deferred)

For each: **(a) what reality has (zero, modulo physics + optim/proximal cross-link)**, **(b) what to add (specialised — be candid)**, **(c) connective-tissue LOC**, **(d) blocker**.

### Tier-0 — Hertz analytic + closed-form contact (~360 LOC)

#### C1 — Hertz-1882 ball-on-ball + ball-on-flat + cylinder closed-form
(a) Nothing. `physics.HookesLaw` provides scalar σ = E ε; can compute E* = E/(1−ν²) but no contact-radius / pressure-distribution.
(b) `contact/hertz.go`: `HertzSphereOnSphere(F, R1, R2, E1, ν1, E2, ν2)` returning `(a, p0, δ, max_shear_stress, max_shear_depth)` where a = (3FR/4E*)^{1/3} contact-radius, p_0 = 3F/(2πa²) max-pressure, δ = a²/R indentation-depth, max-shear-stress at z = 0.48a depth (Johnson-1985-§3.4 closed-form). Plus `HertzCylinderOnCylinder` (line-contact), `HertzCylinderOnFlat`, `HertzGeneralEllipsoid` (Greenwood-1997 elliptic-integral). Mindlin-1949 + Cattaneo-1938 tangential-contact-stick-slip extension.
(c) ~140 LOC.
(d) **No blocker.** Cheapest-1-day-shippable.

#### C2 — Coulomb-1781 friction law
(a) `optim/proximal/operators.go` ProxBox is *literally* the projection used in Coulomb-stick-slip for τ ∈ [−μσ_n, +μσ_n].
(b) `contact/coulomb.go`: `CoulombFriction{Mu, NormalPressure}` with `MaxShearStress() = μ·σ_n` + `IsStick(tangential_stress) bool` + `SlipVelocity(tangential_stress, slip_direction) Vec3` enforcing |τ| ≤ μσ_n with stick (|τ| < μσ_n ⇒ v = 0) vs slip (|τ| = μσ_n ⇒ v ∥ −τ). Pin against the canonical Coulomb-cone condition from Wriggers-2006-§4.2.
(c) ~80 LOC.
(d) **No blocker.**

#### C3 — Tresca-1864 constant-friction
(a) Nothing (the same Tresca surname appears in `physics/materials.go` as the *yield* criterion `TrescaStress`; this slot's Tresca is the *friction* application — same shear-bound, different physical context).
(b) `contact/tresca.go`: `TrescaFriction{TauMax}` with constant max-shear-stress independent-of-normal-pressure. Used as textbook simplification of Coulomb when σ_n is approximately uniform. Pin = consistency with C2 in the limit μ·σ_n_avg = τ_max.
(c) ~60 LOC.
(d) **No blocker.**

#### C4 — Coulomb-wedge condition self-locking
(a) Nothing.
(b) `contact/coulomb_wedge.go`: For block-on-ramp at angle θ with friction-coefficient μ = tan(φ), self-locking when θ ≤ φ (no motion regardless of applied tangential force); slip when θ > φ. Closed-form ramp-on-ramp pin against Wriggers-2006-§4.2-Eq-4.40.
(c) ~80 LOC.
(d) **No blocker.**

### Tier-1 — Signorini-VI + active-set (~620 LOC) — STRICT-blocked-on-244+247

#### C5 — Signorini-1933 unilateral-contact variational inequality
(a) Nothing.
(b) `contact/signorini.go`: VI formulation `find u ∈ K such that a(u, v−u) ≥ ℓ(v−u) ∀v ∈ K` where K = {v ∈ V : v·n ≤ g on Γ_C}. The 1933-canonical-formulation (Brezzi-1972-AnnSNS-Pisa-2 well-posedness). Discretisation via P1-FE bulk + nodal-projection-onto-cone on Γ_C.
(c) ~140 LOC.
(d) **Blocker:** 244-D26 P1-FE + bulk-elasticity-stiffness.

#### C6 — Penalty method Kikuchi-Oden-1988
(a) Nothing.
(b) `contact/penalty.go`: Augment elasticity with `K_ε(u, v) = a(u, v) + ε⁻¹·∫_{Γ_C} (g − u·n)_+ v·n dS`. Limit ε → 0 recovers Signorini. Conditioning deteriorates as ε⁻¹ — the well-known penalty-conditioning trade-off. Still in production for explicit-dynamics where solver-conditioning is not the bottleneck.
(c) ~120 LOC.
(d) **Blocker:** 244-D26 + 247-M13.

#### C7 — Exact Lagrange-multiplier saddle-point
(a) Nothing. `optim/proximal/operators.go` ProxNonneg = projection onto λ ≥ 0 cone usable in dual update.
(b) `contact/lagrange_multiplier.go`: KKT min-max `L(u, λ) = ½a(u,u) − ℓ(u) − ⟨λ, u·n − g⟩` with constraints `λ ≥ 0, u·n ≤ g, λ(u·n − g) = 0`. Discrete saddle-point K=[[A B^T];[B 0]] solved by GMRES (or static-condensation Schur complement S = B A⁻¹ B^T positive-definite if 247-M16 dual-basis used).
(c) ~140 LOC.
(d) **Blocker:** 247-M14 + 244-D27 GMRES.

#### C8 — Augmented-Lagrangian Glowinski-LeTallec-1989
(a) `optim/proximal/admm.go` consensus-ADMM is *literally* Glowinski-LeTallec for f + g splitting where f = bulk-elasticity, g = contact-indicator-on-Γ_C — direct reuse with contact-specific g.
(b) `contact/augmented_lagrangian.go`: `L_α(u, λ) = L(u, λ) + α/2·(g − u·n)_+²` Uzawa-iteration combining penalty-stability (α-term ⇒ bounded conditioning) with multiplier-exactness (λ-update). Pietrzak-Curnier-1999-CMAME-177 frictional-extension. Production-grade since 1999.
(c) ~140 LOC.
(d) **Blocker:** C7.

#### C9 — Primal-dual active-set Hintermüller-Ito-Kunisch-2003
(a) Nothing.
(b) `contact/active_set.go`: Treat contact as *equality* problem on active set `A_k = {i : λ_i + α(g_i − u_i·n) > 0}` and solve unconstrained system; update A every iteration. Hintermüller-Ito-Kunisch-2003-SISC-13 prove finite termination + Q-superlinear convergence as semi-smooth Newton.
(c) ~120 LOC.
(d) **Blocker:** C5 + C7.

### Tier-2 — Mortar contact + dual-basis (~620 LOC) — STRICT-blocked-on-247

#### C10 — Discrete gap function via mortar L²-projection
(a) Nothing.
(b) `contact/mortar/gap.go`: `g_h(x) = Π_h(g_master(x) − u_master·n) + Π_h(u_slave·n)` using 247-M13 L²-projection master→slave. Hüeber-Wohlmuth-2005-CMAME-194 reformulation that handles non-matching meshes correctly.
(c) ~120 LOC.
(d) **Blocker:** 247-M13 + 247-M16.

#### C11 — LCP via mortar dual-basis
(a) Nothing. ProxNonneg + ProxBox provide projection-onto-cone for the LCP element-wise update.
(b) `contact/mortar/dual_lcp.go`: Discrete LCP `M λ + N u = r, λ ≥ 0, g_h − Π u_master ≥ 0, λ·(g_h − Π u_master) = 0` with M diagonal-from-247-M16-dual-basis enabling element-wise complementarity update. Solver: PGS (C16) or Lemke (C17) or ssn (C15).
(c) ~140 LOC.
(d) **Blocker:** 247-M16 + C10.

#### C12 — Master-slave sliding contact
(a) Nothing.
(b) `contact/mortar/sliding.go`: Surface-gap-evolution g_h(t) under finite-deformation; search-radius projection of slave-node onto master-surface; tangential-displacement increment for Coulomb-stick-slip. Closest-point-projection algorithm Konyukhov-Schweizerhof-2005.
(c) ~140 LOC.
(d) **Blocker:** C10 + C2.

#### C13 — Puso-Laursen-2004 3D-mortar-contact
(a) Nothing.
(b) `contact/mortar/threed.go`: 3D-mortar-contact patch-test-passing element-segmentation + face-pairing on tetrahedral interfaces. Puso-Laursen-2004-CMAME-193 the canonical 3D-contact-mortar mesh-segmentation algorithm. Used in commercial codes (ABAQUS-mortar-contact / LS-DYNA-tied-contact / Code_Aster-MORTAR / Sierra-SM-mortar-contact).
(c) ~140 LOC.
(d) **Blocker:** 247-M17 3D-mortar + C10.

#### C14 — Contact-patch-test Taylor-Papadopoulos-1991
(a) Nothing.
(b) `contact/mortar/patch_test.go`: Pin-down zero-penetration for uniform-pressure-loading on non-matching meshes. The keystone-validation Taylor-Papadopoulos-1991 + Crisfield-2000 for any contact-FE code. Mandatory for any production-grade contact algorithm.
(c) ~80 LOC.
(d) **Blocker:** C10 + C13.

### Tier-3 — Semi-smooth Newton + LCP solvers (~480 LOC)

#### C15 — Hüeber-Stadler-Wohlmuth-2008 semi-smooth Newton
(a) Nothing. `optim/rootfind.go` Newton-Raphson scaffold reusable.
(b) `contact/ssn.go`: Reformulate KKT as nonsmooth equation `F(u, λ) = max(0, λ + α(g_h − Π u))` and solve by B-subdifferential-Newton iteration. Q-superlinear convergence proved Hintermüller-Ito-Kunisch-2003 + Hüeber-Stadler-Wohlmuth-2008-SISC-30. Post-2003 contact-state-of-the-art.
(c) ~140 LOC.
(d) **Blocker:** C9 + C11.

#### C16 — Projected-Gauss-Seidel-LCP
(a) ProxNonneg directly applicable as the projection in PGS update.
(b) `contact/pgs.go`: For LCP `Mλ + b ≥ 0, λ ≥ 0, complementarity`, element-wise update `λ_i ← max(0, λ_i − ω(M_ii)⁻¹·(M_:i^T λ + b_i))`. Cottle-Pang-Stone-1992-§3.6 standard-textbook-LCP-iterative-solver.
(c) ~80 LOC.
(d) **Blocker:** C11.

#### C17 — Lemke-1965 pivotal-LCP solver
(a) Nothing.
(b) `contact/lemke.go`: Pivotal-LCP-solver (Cottle-Pang-Stone-1992-§4.4) finite-termination for P-matrix LCPs. Used in robotic-locomotion (MuJoCo / DART) and granular-DEM. Direct-LCP-solver alternative to iterative C16.
(c) ~140 LOC.
(d) **Blocker:** None substrate-wise (uses dense linalg).

#### C18 — Robinson-1992 normal-map NCP
(a) Nothing.
(b) `contact/normal_map.go`: NCP-formulation `F(z) = z − Π_K(z − F(z))` with Newton-on-normal-map. Robinson-1992 alternative to ssn for robotic-contact-rigid-body. ProxNonneg-or-ProxBox supplies Π_K.
(c) ~120 LOC.
(d) **Blocker:** None substrate-wise.

---

## 2. Cross-link audit vs neighbouring slots

### 247 (Mortar / non-conforming FEM)
250 is **strict-downstream of 247** — needs M13 L²-projection + M14 saddle-point assembly + M15 cross-points + M16 dual-basis + M17 3D-mortar + M18 conservation-witnesses. The unifying 247–250 architectural contract: M16 dual-basis provides diagonal mass-matrix M making LCP element-wise solvable + ssn-active-set Q-superlinearly-convergent. Without 247 nothing in C10–C15 can ship.

### 249 (Domain decomposition)
250 partial-overlaps 249 because mortar-contact = non-overlap-Schwarz with *one-sided* Lagrange multiplier (instead of equality). FETI-DP-with-contact deferred frontier (cross-link 249-DD12, niche).

### 248 (Multigrid)
250 forward-cross-links 248 only via C5-Signorini-VI as an obstacle-problem (multigrid-for-VI Hoppe-Kornhuber-1994 / Brandt-Cryer-1983 monotone-multigrid). Deferred frontier — single-source ownership: 250 owns the *contact-specific* obstacle-problem-multigrid; 248 owns the generic-VI-multigrid-substrate.

### 244 (PDE solvers / FEM scaffolding)
250 is **strict-downstream of 244** — needs D25-D26 P1-FE conforming baseline for the bulk-elasticity-stiffness-A in K=[[A B^T];[B 0]].

### 251 (Shape optimisation)
250 cross-links 251 via shape-optimisation-with-contact-constraints + Hintermüller-Laurain-2008 contact-shape-derivative. Deferred future-work.

### Cross-link to optim/proximal (PRESENT)
**Strong reuse opportunity** — `optim/proximal/operators.go` ProxNonneg + ProxBox are *literally* the cone-projection operators used in C7-Lagrange-multiplier-KKT, C9-PDAS-active-set, C11-LCP, C15-ssn, C16-PGS. `optim/proximal/admm.go` consensus-ADMM is *literally* Glowinski-LeTallec-1989 augmented-Lagrangian-contact (C8) when f = bulk-elasticity-quadratic, g = contact-indicator-of-K. The 250-slot should *not* duplicate these operators — single-source ownership: `optim/proximal/` owns the generic prox-projection; `contact/` owns the contact-specific assembly that calls them.

### Cross-link to physics/materials (PRESENT)
HookesLaw scalar σ = E ε is the substrate for E* effective-modulus in C1-Hertz. VonMises + Tresca yield-criteria are substrate for elastic-plastic-contact-extension (deferred frontier).

### 097 (Sparse linalg)
**STRONG blocker** for realistic-N — dense-LU on a 100k-DOF contact saddle-point is O(10¹⁵) operations, untenable beyond toy meshes. C5–C18 all need sparse-direct-solve + sparse-preconditioned-Krylov.

---

## 3. Reproducible R-pins (contact-mechanics specific)

### R-HERTZ-CLOSED-FORM 4/4 — Tier-0 cheapest
Hertz sphere-on-sphere for steel (E=210 GPa, ν=0.3) ball-bearing R₁=R₂=10mm, F=100N. Pin (a) contact-radius a = (3FR/4E*)^{1/3} = 0.151mm to 1e-4; (b) max-pressure p_0 = 3F/(2πa²) = 2.10 GPa to 1e-4; (c) max-shear-depth z = 0.48a = 0.0725mm to 1e-4; (d) sphere-on-flat reduces correctly to R₂→∞ limit. Plus rubber-on-glass (Hertz limit) + cylinder-on-flat (Johnson-1985-§4.2 line-contact analytical pressure).

### R-COULOMB-WEDGE-SELF-LOCK 3/3
Block-on-ramp θ = 30°, μ varying. Pin (a) μ ≥ tan(30°) = 0.577 ⇒ no slip; (b) μ < 0.577 ⇒ slip with acceleration g(sin θ − μ cos θ); (c) discontinuity in transition is sharp (Coulomb-cone). Closed-form Wriggers-2006-§4.2-Eq-4.40.

### R-TRESCA-CONSTANT-FORCE 3/3
Block-on-flat under constant tangential force. Pin Tresca-friction prediction equals Coulomb in the limit μ·σ_n_avg = τ_max for uniform σ_n.

### R-SIGNORINI-OBSTACLE 4/4 (Tier-1, blocked-on-244+247)
Elastic block on rigid base under uniform pressure with gap g(x). Pin against analytical-obstacle-problem solution from Glowinski-1984 (book-§I.4-§5). Compare penalty (C6) + Lagrange (C7) + augmented-Lagrangian (C8) + active-set (C9) all converging to the same VI solution to 1e-6.

### R-MORTAR-CONTACT-PATCH 3/3 (Tier-2, blocked-on-247)
Two non-matching mesh blocks under uniform pressure on contact interface. Pin (a) zero-penetration to round-off using 247-M16 dual-basis; (b) uniform-pressure-reproduction Taylor-Papadopoulos-1991 patch-test passes; (c) cross-validation 3 algorithms (C7 + C8 + C15) converge to same solution.

### R-SSN-CONVERGENCE 3/3 (Tier-3)
Hertz-line-contact on a non-matching mortar-interface. Pin (a) ssn (C15) Q-superlinear-convergence rate κ_{k+1} ≤ C·κ_k² for k > k_0; (b) PGS (C16) Q-linear convergence with rate ρ < 1; (c) Lemke (C17) finite-termination ≤ N steps for P-matrix LCP.

### R-HUEBER-WOHLMUTH-CMAME-194 3/3
Cross-validation against the canonical Hüeber-Wohlmuth-2005-CMAME-194 numerical-experiments table — ssn-on-dual-mortar-LCP iteration counts and contact-pressure profiles for the published two-block-elastic test case.

---

## 4. PR sequencing (4 PRs over 9-12 engineer-days)

**PR-1 ~360 LOC / 2 engineer-days** — Tier-0 closed-form C1-C4. Hertz + Coulomb + Tresca + Coulomb-wedge. **No blocker.** Saturates R-HERTZ-CLOSED-FORM 4/4 + R-COULOMB-WEDGE-SELF-LOCK 3/3 + R-TRESCA-CONSTANT-FORCE 3/3 immediately. **Cheapest-shippable**.

**PR-2 ~620 LOC / 4 engineer-days** — Tier-1 Signorini-VI C5-C9. Signorini + penalty + Lagrange-multiplier + augmented-Lagrangian + active-set. Saturates R-SIGNORINI-OBSTACLE 4/4. Strict-blocked-on 244-D26 + 247-M14 landing.

**PR-3 ~620 LOC / 4 engineer-days** — Tier-2 mortar contact C10-C14. Gap-function + dual-LCP + sliding + 3D-mortar + patch-test. Saturates R-MORTAR-CONTACT-PATCH 3/3. Strict-blocked-on 247-M13/M16/M17.

**PR-4 ~480 LOC / 3 engineer-days** — Tier-3 ssn + LCP solvers C15-C18. Semi-smooth Newton + PGS + Lemke + normal-map. Saturates R-SSN-CONVERGENCE 3/3 + R-HUEBER-WOHLMUTH-CMAME-194 3/3.

**Defer (no PR)** — Tier-4 cohesive-zone + dynamic-contact + lubricated + self-contact + multi-body + adhesion + DG-contact + shell-contact + homogenisation + fictitious-domain ~720 LOC C19–C28. Wait for concrete consumer demand (no aicore/Pistachio consumer today).

**Cumulative through PR-4 ~2,080 LOC src + ~1,040 LOC test** saturating six R-pins.

---

## 5. CANDOR — what to ship vs defer

**Ship now (recommended): PR-1 Tier-0 Hertz + Coulomb + Tresca + Coulomb-wedge ~360 LOC over 1 PR / 2 engineer-days.** Closed-form, no blocker, ships against the canonical 1882–1864–1781 analytical pins. Brand-prestige first zero-dep Go library to ship Hertz-1882 / Coulomb-1781 / Tresca-1864 / Coulomb-wedge-self-locking analytic primitives. Direct consumer for any future Pistachio/aicore physics demo.

**Ship-when-247-lands: PR-2 + PR-3 + PR-4 ~1,720 LOC over 3 PRs / 11-13 engineer-days.** First-class Signorini-VI + penalty + Lagrange + augmented-Lagrangian + active-set + mortar-contact-Wohlmuth-dual + ssn-Hüeber-Wohlmuth + LCP-PGS-Lemke + normal-map-NCP. Brand-moat: first zero-dep Go library to ship Hüeber-Wohlmuth-2005-CMAME-194 ssn-on-dual-mortar + Puso-Laursen-2004-CMAME-193 3D-mortar-contact + Hintermüller-Ito-Kunisch-2003-SISC-13 PDAS.

**Defer Tier-4 (no consumer): C19–C28 ~720 LOC.** Cohesive-zone (fracture-mechanics niche), dynamic-contact (game-engine niche), lubricated (FSI), self-contact + multi-body (algorithmic-niche game-engine), adhesion (frontier), DG-contact (post-2008 niche), shell-contact (blocked-on-shells-FE-substrate-absent), homogenisation (frontier), fictitious-domain (frontier). All near-zero consumer-pull today; defer entirely to Q3-Q4 2026 conditional on aicore consumer.

**Single-day-shippable cheapest** — C1 Hertz alone ~140 LOC. Closed-form, no FE substrate required, no blocker. Validates against Johnson-1985-§3.4 ball-on-ball + cylinder-on-flat closed-form pins. The 1882-canonical-textbook-pin every contact-FE code on earth uses for validation.

**Highest-leverage architecturally** — C10+C11+C15 mortar-gap + dual-LCP + ssn ~400 LOC, but maximally-blocked on 247-M13/M14/M16 + 244-D25-D26 + 097-T1.

**Single-source-of-truth concerns:**
- **Friction-vs-yield "Tresca"**: same name, different application (yield criterion in `physics/materials.go`, friction-bound in `contact/tresca.go`). Document both with cross-reference; no functional duplication.
- **ProxNonneg + ProxBox**: lives canonically in `optim/proximal/operators.go`; `contact/` imports + reuses, does NOT duplicate.
- **ADMM consensus**: lives canonically in `optim/proximal/admm.go`; `contact/augmented_lagrangian.go` is a *thin specialisation* configuring f = bulk-elasticity, g = contact-indicator.
- **Newton-Raphson**: lives canonically in `optim/rootfind.go`; `contact/ssn.go` extends to B-subdifferential semi-smooth case.

---

## 6. Summary of LOC + dependency counts

| Tier | Primitives | LOC src | LOC test | Blocks on |
|---|---|---|---|---|
| 0 closed-form | C1-C4 | 360 | 180 | physics.HookesLaw (PRESENT) |
| 1 Signorini-VI | C5-C9 | 620 | 310 | 244-D26, 247-M14, optim/proximal (PRESENT) |
| 2 mortar contact | C10-C14 | 620 | 310 | 247-M13/M15/M16/M17 |
| 3 ssn + LCP | C15-C18 | 480 | 240 | C9, C11, optim/rootfind (PRESENT) |
| 4 frontier ⊘ DEFER | C19-C28 | 720 | 360 | various, near-zero consumer |
| **Total** | **C1-C18 ship** | **2,080** | **1,040** | -- |
| **Total + defer** | **C1-C28** | **2,800** | **1,400** | -- |
| Recommended ship-now | PR-1 only | 360 | 180 | -- |

Eighteen ship-recommended primitives, four primitives (C1-C4) Tier-0-closed-form-no-blocker, fourteen primitives Tier-1+2+3 strict-blocked-on-247-mortar landing.

The Wohlmuth-1999-dual-Lagrange-basis (247-M16) + Hüeber-Wohlmuth-2005-CMAME-194 ssn-on-dual-mortar-LCP is the single-most-important production-grade-decision in computational-contact-mechanics since 2005 and is the architectural keystone of the 247 → 250 strict-downstream chain. Hertz-1882-ball-on-ball is the cheapest-1-day-shippable closed-form primitive and the textbook pin every contact-FE code on earth uses for validation. Ship Tier-0 now, defer Tier-1-2-3 to 247-landing, defer Tier-4 entirely.
