# 087 | info-missing — canonical primitives NOT yet in `info/`

**Scope.** Enumerate canonical information-theory primitives that are
absent from `reality/info/`, `reality/infogeo/`, and
`reality/compression/`.  086 already established that `info/` itself
ships only LZ76 (`info/lz/`) and MDL/NML (`info/mdl/`); all
distribution-level primitives live in the two adjacent packages.

**Reference frame.** Three external libraries set the bar:
- **dit** (Discrete Information Toolbox, James et al., 2018):
  ~50 entropies/divergences, full Williams-Beer PID lattice,
  multivariate `total correlation`, dual total correlation, CAEKL
  mutual information, ~6 conditional MI estimators.
- **JIDT** (Java Information Dynamics Toolkit, Lizier 2014):
  reference for *dynamics* — transfer entropy (Kraskov, Gaussian,
  binned, symbolic), active info storage, predictive info,
  separable info, Wyner common info, conditional TE.
- **NPEET** (Ver Steeg, ~250 LOC Python): canonical KSG
  implementations of `entropy`, `mi`, `cmi`, `entropy_d`,
  `mi_mixed`, `kldiv` for continuous variables.

Pyphi adds the IIT 4.0 `phi` machinery; mpmath gives reference
arbitrary-precision values for golden-file ground truth.

What's already in repo (baseline, per 086):
- `compression/entropy.go`: Shannon, Joint, Conditional, plug-in MI,
  KL, CrossEntropy (all in **bits**, no input validation).
- `infogeo/fdiv.go`: KL, ReverseKL, JS, TotalVariation, Hellinger,
  ChiSquared, Renyi (all in **nats**, with `validatePair`).
- `infogeo/bregman.go`: generic `Bregman` divergence (caller supplies
  `phi`, `gradPhi`).
- `infogeo/mmd.go`: Maximum Mean Discrepancy with Gaussian kernel.
- `info/lz/`: LZ76 production count + Kaspar-Schuster normalization.
- `info/mdl/`: NML, BIC, AIC, Universal-integer codelength, MDL
  selection.
- `optim/transport/wasserstein1d.go`: 1-D Wasserstein-1.
- `optim/transport/sinkhorn.go`: entropic OT (Sinkhorn).

Everything below is **missing**.

---

## Tier 1 — High leverage, ≤ ~30 LOC each, mathematically trivial

These close the headline gaps cited by 086 and are derivable from
primitives already in repo.

### T1.1 — Bhattacharyya coefficient and distance
**Where.** New `infogeo/fdiv.go` additions.
**Math.** `BC(p,q) = Σ √(pᵢ qᵢ)`; `D_B = -log BC`.
**Identity to existing code.** `H²(p,q) = 1 - BC(p,q)`; reuse the
existing `Hellinger` validate path.
**LOC.** ~15.  **Tests.** ~30.
**Reference.** Bhattacharyya (1943).

### T1.2 — Generic Csiszár f-divergence
**Where.** `infogeo/fdiv.go`: `Fdivergence(p, q []float64, f func(t float64) float64) (float64, error)`
where the divergence is `Σ qᵢ · f(pᵢ/qᵢ)`.
**Math.** Recovers KL (`f(t) = t log t`), reverse-KL (`f(t) = -log t`),
TV (`f(t) = ½|t-1|`), Hellinger² (`f(t) = (√t-1)²`), χ² (`f(t) =
(t-1)²`), JS (`f(t) = ½t log t - ½(1+t) log((1+t)/2)`).
**LOC.** ~20 (plus standard `0/0 = 0`, `p>0 ∧ q=0 → +∞` if `f` is
unbounded at 0 conventions).  **Reference.** Csiszár (1967).

### T1.3 — Tsallis entropy and Tsallis divergence
**Math.** `S_q(p) = (1 - Σ pᵢ^q) / (q - 1)` (q ≠ 1, → Shannon at q=1).
`D_q(p||q) = (1/(q-1))(Σ pᵢ^q qᵢ^{1-q} - 1)`.
**LOC.** ~20 each.  **Reference.** Tsallis (1988).

### T1.4 — Min-entropy, max-entropy, collision (Rényi-2) entropy
**Math.**
- `H_∞(p) = -log max_i pᵢ` (Rényi-∞)
- `H_0(p) = log |support(p)|` (Hartley, Rényi-0)
- `H_2(p) = -log Σ pᵢ²` (collision, Rényi-2; basis for security
  reductions and randomness extractors).
**LOC.** ~10 each.  Already implied by Rényi but currently absent
because `infogeo/fdiv.go::Renyi` rejects α≤0 and α=∞.

### T1.5 — Rényi entropy (vs. divergence)
**Math.** `H_α(p) = (1/(1-α)) log Σ pᵢ^α`, with the four limit cases
above.  Currently only `Renyi(p, q, α)` exists (divergence form).
**LOC.** ~15.

### T1.6 — Differential entropy (parametric Gaussian closed form)
**Math.** `h(N(μ, Σ)) = ½ log((2πe)ᵈ |Σ|)`.
**LOC.** ~15 (depends on `linalg.LogDet`, which the linalg/ package
already provides per CLAUDE.md inventory).
**Why now.** Lets callers compute KL / cross-entropy between two
Gaussians without going through samples.

### T1.7 — Closed-form KL between Gaussians
**Math.** For univariate: `½ (σ_p²/σ_q² + (μ_q-μ_p)²/σ_q² - 1 +
log(σ_q²/σ_p²))`.  For multivariate: `½ (tr(Σ_q⁻¹Σ_p) + (μ_q-μ_p)ᵀ
Σ_q⁻¹ (μ_q-μ_p) - d + log(|Σ_q|/|Σ_p|))`.  Standard variational-
inference primitive.
**LOC.** ~25 (univariate) + ~40 (multivariate using existing Cholesky
in `linalg/`).

### T1.8 — Pointwise mutual information (PMI) and NPMI
**Math.** `pmi(x,y) = log p(x,y) / (p(x)p(y))`; normalized
`npmi(x,y) = pmi(x,y) / -log p(x,y)` (in [-1, 1]).
**LOC.** ~15.  **Reference.** Church & Hanks (1990); Bouma (2009).
**Why.** Standard NLP/IR primitive; zero in repo.

### T1.9 — Miller-Madow MI / entropy bias correction
**Math.** Plug-in MI has bias `(K_xy - K_x - K_y + 1) / (2N ln 2)` in
bits (or `... / (2N)` in nats) where `K_*` are observed support
sizes.  086 flagged this as missing.
**LOC.** ~20.  **Reference.** Miller (1955), Madow (1948).

### T1.10 — Direct conditional entropy `Σ p(x) H(Y|x)`
**Math.** Existing `compression.ConditionalEntropy` uses `H(X,Y) -
H(X)` which cancels catastrophically when `H(X,Y) ≈ H(X)`.  Direct
form is more accurate in that regime.
**LOC.** ~25.

### T1.11 — `compression.validatePair` parity with `infogeo`
**Math.** None — pure validation.  086 flagged that
`compression/entropy.go` silently returns 0 on length mismatch and
accepts negatives/NaN.  Adopt the `infogeo/fdiv.go::validatePair`
pattern.
**LOC.** ~30.

### T1.12 — `LogSumExp` and softmax helpers (canonical site)
**Math.** `lse(xs) = max(xs) + log Σ exp(xs - max(xs))`.  086
identified one inline LSE in `info/mdl/nml.go:142-156` that should be
promoted; multiple call sites in autodiff, copula, prob need it.
**Where.** New `prob/mathutil.go` or `info/lse.go`.
**LOC.** ~20 + ~30 tests.

### T1.13 — Jensen inequality witness / Jensen gap
**Math.** Pure utility: given `f`, samples `xs` and weights `ws`,
returns `f(E[X]) - E[f(X)]`.  Not really a primitive but a teaching
hook the docs need.
**LOC.** ~15.

### T1.14 — Total correlation and dual total correlation (multivariate MI)
**Math.**
- `TC(X₁,…,Xₙ) = Σ H(Xᵢ) - H(X₁,…,Xₙ)` (Watanabe 1960; sum of marginal
  entropies minus joint entropy, the canonical multivariate
  MI generalisation; *not* the same as integrated information).
- `DTC(X₁,…,Xₙ) = H(X₁,…,Xₙ) - Σᵢ H(Xᵢ | X_{-i})` (Han 1978).
**LOC.** ~30 each.  Useful precursor for IIT and PID.

### T1.15 — Gini impurity and Theil index
**Math.**
- `Gini(p) = 1 - Σ pᵢ²` (this is a polynomial of `H_2`; classical
  decision-tree split criterion).
- `Theil_T(x) = (1/n) Σᵢ (xᵢ/μ) log(xᵢ/μ)` (KL of empirical to
  uniform, in income/inequality space).
**LOC.** ~10 each.  **Reference.** Theil (1967), Gini (1912).

**Tier-1 total: ~325 LOC of math + ~250 LOC of tests for 15
primitives.**  Every one is shorter than the existing `Renyi`
implementation in `infogeo/fdiv.go`.

---

## Tier 2 — Continuous-variable estimators (~50–250 LOC each)

These require nearest-neighbour search, kernel evaluation, or numeric
integration.  None of them depend on packages outside the existing
linalg/optim/transport/prob set, but they introduce nontrivial
infrastructure.

### T2.1 — Kozachenko-Leonenko differential entropy estimator
**Math.** `Ĥ_KL(X) = -ψ(k) + ψ(N) + log c_d + (d/N) Σ log(2 ε_i)`
where `ε_i` is twice the distance from `x_i` to its kth nearest
neighbour and `c_d` is the unit-ball volume.
**Where.** New `info/diffent/kl.go`.
**LOC.** ~150 (brute-force kNN; ~500 with kd-tree).
**Reference.** Kozachenko & Leonenko (1987).
**Why.** 086 listed this explicitly as a "substantial gap given the
package is named `info/`".

### T2.2 — Kraskov-Stögbauer-Grassberger MI (KSG-1, KSG-2)
**Math.** Two estimators of `I(X;Y)` on continuous variables based on
kNN distances in joint vs. marginal spaces:
- **KSG-1**: uses joint-space ε-radius, marginal counts inside it.
  `Î = ψ(k) - <ψ(n_x+1) + ψ(n_y+1)> + ψ(N)`.
- **KSG-2**: uses marginal-space ε's separately.  Lower bias for
  strongly-dependent variables.
**Where.** New `info/mi/ksg.go`.
**LOC.** ~250 + tests (~150).
**Reference.** Kraskov, Stögbauer, Grassberger (2004) Phys. Rev. E.
**Why.** This is *the* canonical continuous-MI estimator and has
zero implementations in the repo.  npeet's `mi(x, y, k=3)` is the
reference; JIDT's KSG impl matches.

### T2.3 — MIXED-KSG (one continuous, one discrete)
**Math.** Gao et al. (2017, NeurIPS) generalisation of KSG when one
variable is discrete-valued; uses conditional kNN within each
discrete level.
**LOC.** ~120.
**Reference.** Gao, Kannan, Oh, Viswanath (2017).

### T2.4 — Conditional MI (KSG-conditional)
**Math.** `I(X;Y|Z)`; Frenzel-Pompe (2007) and Vejmelka-Paluš (2008)
kNN estimators.  Standard formula `I(X;Y|Z) = H(X,Z) + H(Y,Z) -
H(X,Y,Z) - H(Z)` plus the bias cancellation that makes KSG-style
estimators work on the joint.
**LOC.** ~180.
**Reference.** Frenzel & Pompe (2007) Phys. Rev. Lett.

### T2.5 — Vasicek and Ebrahimi-Pflughoeft-Soofi differential entropy
**Math.** Univariate, sample-spacing-based: `Ĥ_V = (1/N) Σ log((N/2m)(x_{(i+m)} - x_{(i-m)}))`.
**LOC.** ~80 (sort + spacing scan).
**Why.** Cheap O(N log N) baseline against KL kNN for 1-D data.

### T2.6 — Histogram entropy with adaptive binning
**Math.** Scott's rule, Sturges' rule, Freedman-Diaconis bin width;
plug into Shannon.
**LOC.** ~80 (the bin-width math is ~30 each).

### T2.7 — KL between mixture-of-Gaussians (sample-based + variational bound)
**Math.** Closed form does not exist; standard estimators are
(a) Hershey-Olsen 2007 variational lower bound (~80 LOC, requires
`KLGaussian`), (b) Monte Carlo with importance sampling (~50 LOC).
**LOC.** ~150 total.
**Reference.** Hershey & Olsen (2007) ICASSP.

### T2.8 — Stein discrepancy (kernelised KSD)
**Math.** `KSD²(p, q) = E_{x,x'~q}[k_p(x,x')]` where `k_p` is the
Stein kernel `k_p(x,x') = sₚ(x)ᵀ k(x,x') sₚ(x') + sₚ(x)ᵀ ∇_{x'} k +
sₚ(x')ᵀ ∇_x k + Σ ∂² k`.
**Where.** Extends `infogeo/mmd.go` (already has kernels).
**LOC.** ~120 + tests.
**Reference.** Liu, Lee, Jordan (2016).
**Why.** Modern goodness-of-fit primitive used in variational
inference and SVGD.

### T2.9 — Empirical f-divergence with k-NN density ratio
**Math.** Plug-in `f`-divergence on continuous data via Wang-Kulkarni-
Verdú (2009) kNN density-ratio estimator.
**LOC.** ~150.

**Tier-2 total: ~1,150 LOC of math + ~600 LOC of tests for 9
estimators.**  KSG (T2.2) and KL kNN (T2.1) are by far the highest
leverage — `npeet` ships exactly these two as its core API.

---

## Tier 3 — Information dynamics, decomposition, and IIT

These are the JIDT, dit-PID, and Pyphi territories.  Each requires
non-trivial multivariate machinery.

### T3.1 — Transfer entropy (binned, Gaussian, KSG)
**Math.** `TE_{Y→X} = I(X_{t+1}; Y_t^{(l)} | X_t^{(k)})`.  Three
estimators in JIDT:
- **Binned/symbolic** (~80 LOC, requires only joint-distribution
  helpers).
- **Gaussian / linear** (~60 LOC, recovers Granger causality).
- **KSG-based** (~200 LOC, depends on T2.4).
**Reference.** Schreiber (2000) Phys. Rev. Lett.; Kaiser & Schreiber
(2002).
**Total LOC.** ~400.

### T3.2 — Granger causality (linear)
**Math.** F-test on residual variances of two AR fits; equivalent to
TE under joint-Gaussianity.  This is the linear specialisation of
T3.1 and currently has zero hits in repo.
**Where.** New `info/dynamics/granger.go` or `timeseries/`.
**LOC.** ~100 (depends on existing AR fit; if `timeseries/` doesn't
have one, +200 for AR + Yule-Walker).
**Reference.** Granger (1969).

### T3.3 — Active information storage
**Math.** `AIS_X = I(X_{t+1}; X_t^{(k)})`.  Self-MI of past on
future.  JIDT primitive.
**LOC.** ~50 (reuses TE machinery).

### T3.4 — Predictive information / excess entropy
**Math.** `E = lim_{T→∞} I(X_{-T:0}; X_{1:T})`.
**LOC.** ~80 (block-MI estimator with extrapolation).
**Reference.** Bialek, Nemenman, Tishby (2001).

### T3.5 — Entropy rate (HMM / Markov)
**Math.** `h_μ = lim_{n→∞} (1/n) H(X_1,…,X_n) = H(X_2 | X_1)` for a
Markov chain with stationary distribution.  HMM case requires
Birkhoff-style block-entropy extrapolation (Cover-Thomas Ch. 4) or
forward-algorithm-based estimator (Hochwald-Jelenkovic 1999).
**LOC.** ~80 (Markov), ~150 (HMM).

### T3.6 — Williams-Beer Partial Information Decomposition (PID)
**Math.** Decomposes `I(X₁,X₂; Y)` into redundant + unique₁ + unique₂
+ synergistic components via the redundancy lattice.  Two
"redundancy" definitions to ship:
- **`I_min`** (Williams & Beer 2010, original).
- **`I_∩^{MMI}`** (Bertschinger et al. 2014, minimum mutual
  information), which is the dit default.
- Optional: **`I_BROJA`** (Bertschinger et al. 2014, max-entropy
  optimisation).
**LOC.** ~250 (lattice + I_min) + ~200 (BROJA via existing
`optim/`).
**Reference.** Williams & Beer (2010).
**Why.** Currently zero PID anywhere in the repo.

### T3.7 — CAEKL multivariate mutual information
**Math.** Chan-Al-Bashabsheh-Ebrahimi-Kaced-Liu (2015) extension of
MI to ≥3 variables; relates to network-coding capacity bounds.
**LOC.** ~120.
**Reference.** Chan et al. (2015) IEEE Trans. IT.

### T3.8 — Wyner common information
**Math.** `C(X;Y) = inf_W{I(X,Y;W) : X ⊥ Y | W}`.  Subadditive lower
bound on `I(X;Y)`; useful in coding-theoretic settings.
**LOC.** ~100 (variational form via existing `optim/`).
**Reference.** Wyner (1975).

### T3.9 — Integrated information `Φ` (IIT 3.0 and 4.0)
**Math.** Φ measures irreducibility of a system to its parts under
all bipartitions, computed as the EMD between the system's
cause-effect repertoire and the partitioned approximation.  Pyphi is
the reference.
**Components.**
- Cause-effect repertoires (forward & backward perturbational
  conditional distributions).
- Partition enumeration (set partitions of `{1,…,n}` × bipartition
  index).
- EMD on the repertoire space (already partly available via
  `optim/transport/wasserstein1d.go`; multidim needs new code).
- `φ_min` aggregation across all partitions.
**LOC.** ~600–1000 (Pyphi is ~10 kLOC but most is bookkeeping; the
mathematical core is ~700 if we ship only IIT 3.0 small-system,
≤ 8 nodes).
**Reference.** Oizumi, Albantakis, Tononi (2014) PLoS Comp Bio;
Albantakis et al. (2023, IIT 4.0).
**Why.** IIT is the headline use-case the topic asks for; even a
pedagogical `Phi` over 4-node networks is non-trivial and is the
main reason `info/integrated/` should exist as its own subpackage.

### T3.10 — Pearl's d-separation oracle
**Math.** Given a DAG and three disjoint vertex sets `X, Y, Z`,
return whether `Z` d-separates `X` from `Y` (Pearl 1988).
**Where.** Probably `graph/dag.go` or a new `prob/causal/dsep.go` —
not strictly an info primitive but the topic listed it.
**LOC.** ~100 (BFS over moralised graph with Bayes-ball algorithm).
**Reference.** Geiger, Verma, Pearl (1990).

### T3.11 — Sliced Wasserstein and conditional Wasserstein
**Math.** Sliced-W is `E_θ[W_p(P_θ#X, P_θ#Y)]` where `P_θ` is a 1-D
projection, giving an O(n log n) MC approximation that already
leverages the existing `optim/transport/wasserstein1d.go`.
Conditional Sinkhorn is a parametric extension of T2.6 in `optim/`.
**LOC.** ~80 (sliced-W only).
**Reference.** Rabin, Peyré, Delon, Bernot (2012).

**Tier-3 total: ~2,400 LOC of math + ~1,000 LOC of tests across 11
primitives.**  IIT-Φ alone dominates the budget.

---

## Cross-cutting design fixes (mostly named in 086, listed here for closure)

C1.  **Bits vs. nats split** — `compression/` uses bits, `infogeo/`
uses nats.  Decision needed: keep both surfaces (with explicit
`Nats` / `Bits` suffix, like `MutualInformationBits` / `KLNats`) or
unify on nats and provide `BitsToNats` / `NatsToBits` constants in
`constants/`.  086 recommended documenting the split rather than
unifying — agree.

C2.  **Validation parity** — `compression/entropy.go` does no
validation; `infogeo/fdiv.go` validates strictly.  T1.11 closes this.

C3.  **Single canonical LSE site** — T1.12 promotes the `info/mdl/nml.go`
inline LSE.

C4.  **Absolute-continuity boundary tests** — pin `+Inf` returns
under refactors for KL, CrossEntropy, ChiSquared, Renyi(α>1), Tsallis,
T1.2 generic f-divergence.  ~50 LOC of pure tests.

C5.  **Two-language parity (Go/Python golden files)** — every
T1 primitive ships with a `testdata/golden/*.json` per CLAUDE.md §1.
At ~30 vectors per primitive, T1 alone adds ~450 vectors.

---

## What this means for the 4-language port plan

Adding T1 + T2.1 + T2.2 alone (the three highest-leverage clusters)
covers ~80 % of what npeet, dit-core, and JIDT-binned offer.  The
delta to "full JIDT-equivalent" is T3.1 (transfer entropy) + T3.6
(PID) + T3.5 (entropy rate); the delta to "full Pyphi-equivalent"
is T3.9 (Φ).

In dependency order:
1. **T1 first** (~575 LOC).  Pure refactor + closure of obvious gaps.
2. **T2.1, T2.2** (~600 LOC).  Unblocks NPEET parity.
3. **T3.1 binned + T3.2** (~200 LOC).  Unblocks JIDT-binned parity.
4. **T2.4 + T3.1 KSG** (~400 LOC).  JIDT-Kraskov parity.
5. **T3.6 PID** (~450 LOC).  dit parity.
6. **T3.9 Φ small-system** (~700 LOC).  Pyphi parity.

Each step is independently shippable.  T1 + T2.1 + T2.2 is the
**P-1 commit** with the highest leverage-per-LOC in the entire
information-theory layer of the repo.

---

## Files referenced (absolute paths)

- `C:\limitless\foundation\reality\info\lz\` (LZ76)
- `C:\limitless\foundation\reality\info\mdl\` (NML, BIC, AIC)
- `C:\limitless\foundation\reality\infogeo\fdiv.go`
- `C:\limitless\foundation\reality\infogeo\bregman.go`
- `C:\limitless\foundation\reality\infogeo\mmd.go`
- `C:\limitless\foundation\reality\compression\entropy.go`
- `C:\limitless\foundation\reality\optim\transport\wasserstein1d.go`
- `C:\limitless\foundation\reality\optim\transport\sinkhorn.go`
- `C:\limitless\foundation\reality\reviews\overnight-400\agents\086-info-numerics.md`

End report.  ~330 lines.
