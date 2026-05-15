# 255 | new-mrf — Markov random fields: pairwise / higher-order / message passing / Hammersley-Clifford / CRF / TRW / MPLP / AD³ / mean-field / junction-tree / Gibbs / Swendsen-Wang / Markov-Logic / α-expansion-fusion-move

**Summary line 1.** reality v0.10.0 ships **ZERO** Markov-random-field / conditional-random-field / factor-graph / message-passing / Hammersley-Clifford / pseudo-likelihood / partition-function-Z / Bethe-free-energy / Kikuchi-cluster-variation / Markov-Logic-Network / linear-chain-CRF / skip-chain-CRF / loopy-BP / sum-product / max-product / junction-tree / TRW / MPLP / AD³ / Gibbs-MRF / ICM / Geman-Geman-MAP surface — repo-wide grep on `markov.random.field|mrf|conditional.random.field|crf|hammersley.clifford|factor.graph|message.passing|loopy.bp|sum.product|max.product|junction.tree|clique.tree|bethe.free|kikuchi|cluster.variation|trw|tree.reweighted|mplp|globerson.jaakkola|ad3|martins.figueiredo|markov.logic|richardson.domingos|linear.chain.crf|lafferty.mccallum.pereira|skip.chain.crf|tree.crf|pseudo.likelihood|composite.likelihood|piecewise.training|ising|potts|spin.glass|gibbs.sampler|geman.geman|icm|besag.icm|iterated.conditional.modes|swendsen.wang|wolff.cluster|simulated.annealing.mrf|mean.field.vi.mrf|partition.function|log.partition|free.energy|exponential.family.mrf|natural.parameter.mrf` returns **zero callable matches** outside `prob/markov.go::{MarkovSteadyState, MarkovSimulate}` (which are 1-D state-only Markov-chain primitives — NO emission, NO observation, NO MAP-inference, NO partition-function), `optim/metaheuristic.go::SimulatedAnnealing` (PRESENT but the *generic-numerical-optimisation* SA over continuous-`[]float64` neighbours, NOT the Geman-Geman-1984 PAMI-6:721 *pixel-grid-Gibbs-MRF-MAP-with-cooling-schedule* SA), and the false-positive name-collision `audio/segmentation` (segmentation-onset-detection in audio, NOT MRF-segmentation). The entire 1974-2015 Markov-random-field / probabilistic-graphical-model / energy-based-model / discrete-graphical-inference canon (Besag-1974 MRF-introduction-to-statistics; Hammersley-Clifford-1971 unpublished Gibbs-MRF-equivalence; Geman-Geman-1984 PAMI-6:721 Gibbs-sampler-for-image-restoration with simulated-annealing-MAP; Pearl-1988 *Probabilistic Reasoning in Intelligent Systems* belief-propagation-on-trees; Lauritzen-Spiegelhalter-1988 JRSS-50:157 junction-tree; Murphy-Weiss-Jordan-1999 UAI loopy-BP; Lafferty-McCallum-Pereira-2001 ICML CRF-original; Yedidia-Freeman-Weiss-2005 IT-51:2282 Bethe / Kikuchi / generalised-BP; Wainwright-Jaakkola-Willsky-2005 IT-51:3697 TRW + LP-relaxation-of-MAP; Kolmogorov-2006 PAMI-28:1568 TRW-S; Sutton-McCallum-2006 *Introduction to Conditional Random Fields*; Globerson-Jaakkola-2008 NIPS MPLP; Richardson-Domingos-2006 ML-62:107 Markov-Logic-Networks; Sutton-Rohanimanesh-McCallum-2004 ICML skip-chain-CRF; Martins-Figueiredo-Aguiar-Smith-Xing-2011 ICML AD³; Krähenbühl-Koltun-2011 NIPS dense-CRF-permutohedral; Ishikawa-2011 PAMI-33:1234 higher-order-to-pairwise reduction; Fix-Gruber-Boros-Zabih-2011 ICCV higher-order-graph-cuts) is wholly ABSENT — no factor-graph type, no exponential-family-MRF parameterisation, no log-partition-function, no Bethe-free-energy, no pseudo-likelihood training, no Hammersley-Clifford-clique-decomposition primitive, no PGM library at all. **PARTIAL OVERLAP with 254 (graph-cuts) C20 TRW-S + C21 loopy-BP + C22 junction-tree + C23 MPLP + C26 AD³ + C27 Gibbs-sampler + C29 mean-field-Bethe**: slot 255 is the **complementary track** that lives ONE ABSTRACTION LAYER ABOVE 254's graph-cuts — 254 ships the *energy-minimisation* (MAP-inference-as-min-cut) view from the optimisation side, slot 255 ships the *probabilistic-graphical-model* (MRF/CRF/factor-graph/Hammersley-Clifford) view from the probability side; the two converge on TRW-S / loopy-BP / junction-tree / mean-field but diverge on the upstream API (255 needs `Factor`, `FactorGraph`, `LogPotential`, `PartitionFunction`, `Marginal`, `Pseudo-Likelihood-Train` types that 254 has no use for), the marginal-inference primitives (255 needs `sum-product` for marginals; 254 only ships `max-product` for MAP), the training surface (255 ships `BPLearn`, `ContrastiveDivergence`, `PseudoLikelihood`, `PiecewiseLikelihood`, `Persistent-CD` — all absent from 254), and the CRF tier (255-K1-K8 linear-chain-CRF / skip-chain-CRF / tree-CRF / dense-CRF, all absent from 254 which is purely-discrete-graph-cut-MAP). **PARTIAL OVERLAP with 252 (image-segmentation) S14 BoykovKolmogorov + S15 alpha-expansion**: 252 is a *consumer* of MRF-MAP-inference for the image-segmentation API; 255 is the *MRF-substrate-supplier* across all consumers (image-segmentation 252, NLP-tagging via linear-chain-CRF, computer-vision via dense-CRF Krähenbühl-Koltun, statistical-physics via Ising/Potts/spin-glass — collectively the entire PGM-applications-tier). **CROSS-LINK to 165 (synergy-sequence-prob)**: 165 enumerated the missing HMM tier (forward-backward, Viterbi, Baum-Welch in `prob/hmm.go` ~520 LOC) — slot 255 places HMM as a *special case* of linear-chain-CRF where the joint distribution factorises as P(Y, X) = ∏ P(yᵢ|yᵢ₋₁)·P(xᵢ|yᵢ) (generative directed) vs CRF's discriminative undirected P(Y|X) = (1/Z(X))·∏ ψ(yᵢ, yᵢ₋₁, X); recommend 165 ships HMM in `prob/hmm.go` and slot 255 cross-references for the chain-MRF / chain-CRF generalisation. **Block-C verdict:** the entire MRF / CRF / probabilistic-graphical-model tier is absent and is the **second-largest probability-package gap** after 117-prob-missing's PRNG/Dirichlet/Wishart enumeration; this slot enumerates twenty-two primitives M1-M22 ~2,940 LOC (eight of which OVERLAP-COMPATIBLY with 254-graph-cuts and SHIP IN 255 as the *probabilistic-API-thin-wrapper* over 254's optimisation-API-thick-implementation, ~80 LOC of API-bridge per shared algorithm).

**Summary line 2.** Twenty-two primitives M1-M22 totalling ~2,940 LOC organised as **(a) Tier-0 factor-graph + MRF substrate ~440 LOC** (M1 `prob/mrf/factor.go` `Factor{Vars []int; Card []int; LogTable []float64}` + `FactorGraph{Vars []Variable, Factors []Factor}` + `Cliques()` + `MoralGraph()` + `TreeWidth()` ~140 LOC; M2 `prob/mrf/hammersley_clifford.go` Hammersley-Clifford-1971 unpublished / Besag-1974-JRSS-B-36:192 clique-decomposition + Gibbs-distribution-equivalence-checker ~80 LOC; M3 `prob/mrf/exponential_family.go` exponential-family-MRF parameterisation `P(x) = exp(θᵀφ(x) − A(θ))` with `LogPartition`, `Marginals`, `NaturalParameters` ~140 LOC; M4 `prob/mrf/icm.go` Besag-1986-JRSS-D-35:96 Iterated-Conditional-Modes coordinate-descent MAP ~80 LOC), **(b) Tier-1 message-passing inference ~720 LOC** (M5 `prob/mrf/sum_product.go` Pearl-1988 sum-product / belief-propagation on factor-graph for marginal-inference + product-of-incoming-messages-divided-by-outgoing schedule ~140 LOC; M6 `prob/mrf/max_product.go` Pearl-1988 max-product / Viterbi-on-factor-graph for MAP-decoding (degenerate-tie-handling + back-pointers) ~120 LOC; M7 `prob/mrf/junction_tree.go` Lauritzen-Spiegelhalter-1988-JRSS-50:157 triangulate→clique-tree→two-pass-propagation exact-inference ~180 LOC, **shared with 254-C22**; M8 `prob/mrf/loopy_bp.go` Murphy-Weiss-Jordan-1999-UAI loopy-BP both sum-product and max-product flavours, damping-factor-Murphy-2001 to mitigate oscillation ~140 LOC, **shared with 254-C21**; M9 `prob/mrf/trw.go` Wainwright-Jaakkola-Willsky-2005-IT-51:3697 Tree-Reweighted-Message-Passing parallel + Kolmogorov-2006 TRW-S sequential variant, monotone-non-decreasing dual-LP-bound ~140 LOC, **shared with 254-C20**), **(c) Tier-2 variational + cluster ~440 LOC** (M10 `prob/mrf/mean_field.go` mean-field-VI minimising KL(q‖p) over fully-factorised q with coordinate-update; Bethe-free-energy + loopy-BP-as-Bethe-fixed-point connection ~140 LOC, **shared with 254-C29**; M11 `prob/mrf/kikuchi.go` Kikuchi-1951 PR-81:988 cluster-variation-method + Yedidia-Freeman-Weiss-2005 generalised-BP region-graph propagation ~160 LOC; M12 `prob/mrf/mplp.go` Globerson-Jaakkola-2008-NIPS Max-Product-Linear-Programming dual-decomposition with cluster-tightening ~140 LOC, **shared with 254-C23**), **(d) Tier-3 sampling-based MAP + marginals ~360 LOC** (M13 `prob/mrf/gibbs_sampler.go` Geman-Geman-1984-PAMI-6:721 single-site Gibbs sweep + simulated-annealing schedule for MAP ~120 LOC, **shared with 254-C27**; M14 `prob/mrf/swendsen_wang.go` Swendsen-Wang-1987-PRL-58:86 cluster-MC for Ising/Potts ~120 LOC, **shared with 254-C28**; M15 `prob/mrf/metropolis_hastings.go` Metropolis-Hastings on energy-difference acceptance with adaptive-proposal ~120 LOC), **(e) Tier-4 conditional random fields ~620 LOC** (M16 `prob/mrf/linear_chain_crf.go` Lafferty-McCallum-Pereira-2001-ICML linear-chain-CRF with `Forward(x, λ)` + `Backward(x, λ)` + `Viterbi(x, λ)` + `LogLikelihood(x, y, λ)` + L-BFGS-trained gradient `∂ℓ/∂λᵢ = 𝔼_emp[fᵢ] − 𝔼_λ[fᵢ]` ~220 LOC; M17 `prob/mrf/skip_chain_crf.go` Sutton-Rohanimanesh-McCallum-2004-ICML skip-chain-CRF with non-local long-range edges ~80 LOC; M18 `prob/mrf/tree_crf.go` tree-structured-CRF for parse-tree-style outputs ~80 LOC; M19 `prob/mrf/dense_crf.go` Krähenbühl-Koltun-2011-NIPS fully-connected-CRF with permutohedral-lattice-mean-field-BP for image-pixel-labelling at O(N) per iteration vs naive O(N²) ~140 LOC; M20 `prob/mrf/markov_logic.go` Richardson-Domingos-2006-ML-62:107 Markov-Logic-Network with first-order-formula weights `P(x) ∝ exp(Σᵢ wᵢ·n_i(x))` where n_i is grounded-formula-count ~100 LOC), **(f) Tier-5 training + parameter learning ~360 LOC** (M21 `prob/mrf/pseudo_likelihood.go` Besag-1975-Statistician-24:179 pseudo-likelihood `Π_i P(xᵢ | x_{N(i)}, θ)` + L-BFGS-fit (avoids partition-function via local-conditional) + composite-likelihood-Lindsay-1988-Contemp-Math-80:221 ~140 LOC; M22 `prob/mrf/contrastive_divergence.go` Hinton-2002-NeuralComp-14:1771 CD-1 / CD-k Boltzmann-machine-training with Gibbs-step-truncation + Tieleman-2008-ICML PCD persistent-contrastive-divergence ~120 LOC; M23 `prob/mrf/piecewise_training.go` Sutton-McCallum-2007-ML-66:301 piecewise-training decoupling factor-subsets ~100 LOC).

**SINGULAR-FOUNDATIONAL M1 Factor + FactorGraph + M16 LinearChainCRF ~360 LOC** — without M1 there is no PGM API at all (factor-graphs are the lingua-franca of probabilistic-graphical-models since Kschischang-Frey-Loeliger-2001-IT-47:498); without M16 the MRF/CRF distinction collapses to "MRF = unsupervised, CRF = discriminative" with no concrete API surface; together they are the singular-foundational pair without which slot 255 is academic-only. **SINGULAR-CHEAPEST-1-DAY M1 FactorGraph + M4 ICM + M5 sum-product-on-trees + M16 linear-chain-CRF-inference (no training) ~440 LOC** — ICM-Besag-1986 is the simplest MAP-MRF algorithm (coordinate-descent until no-change), sum-product-on-trees is exact and trivial (no loopy schedule needed), linear-chain-CRF-inference reuses sum-product-on-the-chain — ships in one engineer-day against PRESENT `optim/lbfgs` substrate. **SINGULAR-MOAT M16 LinearChainCRF + M19 DenseCRF + M21 PseudoLikelihood ~500 LOC** — linear-chain-CRF is the canonical NLP-tagging primitive (POS tagging, NER, chunking, segmentation; >40,000 citations on Lafferty-McCallum-Pereira-2001), dense-CRF Krähenbühl-Koltun-2011 is the canonical post-processing for semantic-segmentation (>10,000 citations; the DeepLab-v1/v2/v3+ family uses dense-CRF as the final refinement), and pseudo-likelihood is the practical workhorse for MRF-parameter-learning (avoids partition-function-Z). **SINGULAR-2024-FRONTIER M11 Kikuchi + M19 DenseCRF + M22 PCD ~420 LOC** — Kikuchi-cluster-variation is the post-2005 generalisation of Bethe (region-based-free-energy); dense-CRF-permutohedral is the post-2010 GPU-friendly mean-field-MRF; persistent-CD-Tieleman-2008 is the modern Boltzmann-machine training algorithm (ships in PyTorch's `RBM`). **SINGULAR-PEDAGOGICAL M2 Hammersley-Clifford + M3 ExponentialFamily + M4 ICM ~300 LOC** — H-C-1971 is the *fundamental theorem* of MRFs (Markov property ⟺ Gibbs-distribution-with-clique-potentials); exponential-family parameterisation is the modern unified PGM language (Wainwright-Jordan-2008 Foundations-and-Trends-1:1 takes this view); ICM is the simplest MAP algorithm and the right pedagogical entry-point. Five R-MUTUAL-CROSS-VALIDATION 3/3 pins enabled (see §3). Recommended placement **NEW sub-package `prob/mrf/`** under existing `prob/` package — same "consumer-shaped sub-package, not in primitive-supplier-package" precedent (151/153/156/157/158/247/250/252/253/254). Strict-downstream of `prob.MarkovSteadyState` (oracle for trivial chain-marginals via π·P^t), `optim.LBFGS`-PRESENT (CRF parameter-learning), `optim/proximal.Admm`-PRESENT (AD³ ADMM-based dual-decomposition); strict-upstream of 252-S14/S15 segmentation-MAP tier and 165-HMM tier (HMM = chain-MRF special-case) and (a future) `prob/graphical/` Bayesian-network slot.

---

## 0. State at HEAD (2026-05-09, v0.10.0)

Repo-wide audit for MRF / CRF / factor-graph / message-passing / probabilistic-graphical-model surface.

| Surface | Path | MRF relevance |
|---|---|---|
| `prob.MarkovSteadyState` (power-iteration on row-stochastic) | `prob/markov.go:31-70` | 1-D state-only Markov-chain; **PRESENT** but no emission, no observation, no MAP, no partition-function |
| `prob.MarkovSimulate` (LCG-driven trajectory) | `prob/markov.go:99-139` | 1-D state-only sampling; **PRESENT** but not Gibbs-sampler-on-MRF |
| `prob.NormalPDF / Bernoulli / Beta / Gamma` | `prob/distributions.go` | Substrate for emission distributions in HMM/CRF; **PRESENT** |
| `prob.LogGamma / RegularizedBetaInc / RegularizedGammaInc` | `prob/mathutil.go` | Substrate for Dirichlet partition-functions (Markov-Logic + CRF priors); **PARTIAL** — Dirichlet-PDF absent (117 flag) |
| `optim.LBFGS / SimulatedAnnealing` | `optim/*.go` | CRF parameter-learning + MAP-MRF substrate; **PRESENT** |
| `optim/proximal.{Admm, Fbs}` | `optim/proximal/*.go` | AD³ + dual-decomposition substrate; **PRESENT** |
| `graph.MaxFlow` (Edmonds-Karp) | `graph/flow.go` | MAP-MRF-as-min-cut substrate (254 enumerates BK-replacement); **PRESENT but inadequate** (254-C2) |
| `graph/cuts/` package (TRW-S / loopy-BP / junction-tree / Gibbs / mean-field) | -- | **ABSENT** (254 enumerates) |
| `prob/mrf/` package | -- | **ABSENT** — this slot creates |
| `prob/hmm.go` (forward / backward / Viterbi / Baum-Welch) | -- | **ABSENT** (165 enumerates) |
| `LogSumExp` / `Log1mExp` / `Log1pExp` | `changepoint/bocpd.go` (private) | Log-space toolbelt private to `changepoint`; **PARTIAL** (165 + 117 flag — must promote to `prob/mathutil.go`) |
| Pseudo-likelihood / contrastive-divergence / piecewise-training | -- | **ABSENT** |
| Markov-Logic-Network / first-order-MRF | -- | **ABSENT** |
| Dense-CRF / permutohedral-lattice | -- | **ABSENT** |
| M1-M23 MRF/CRF primitives | -- | **ALL ABSENT** |

**False-positive name-collisions audited:**
- `chaos/systems.go` — Lorenz / Van-der-Pol / Rössler ODE systems, NO Ising/Potts/spin-glass. Different math.
- `audio/segmentation` — onset/offset detection in audio signals, NOT MRF-image-segmentation.
- `optim/metaheuristic.SimulatedAnnealing` — generic-numerical-optimisation SA, NOT Geman-Geman-1984 pixel-grid-Gibbs-MRF SA.
- `prob/markov.go::MarkovSteadyState` — 1-D state-only Markov-chain power-iteration; NOT spatial-MRF / 2-D-MRF / factor-graph-MRF.
- `prob/conformal/*.go` — split-conformal / adaptive-conformal prediction; unrelated to MRF.
- `info/mdl/codelength.go` — minimum-description-length code-length; unrelated to MRF-partition-function.

**Cross-import edges that this slot creates:**
- `prob/mrf → prob.{NormalPDF, Bernoulli, BetaPDF}` for emission distributions (M16-M19 CRF + HMM cross-link).
- `prob/mrf → prob.mathutil.{LogSumExp, Log1mExp, Log1pExp}` (165-blocker — must promote first).
- `prob/mrf → optim.LBFGS` for M16-M20 CRF parameter-learning gradient `∇λ ℓ = 𝔼_emp[f] − 𝔼_λ[f]`.
- `prob/mrf → optim/proximal.Admm` for M12 MPLP / AD³ ADMM dual-decomposition.
- `prob/mrf → graph.IntAdjacency` for factor-graph adjacency.
- `prob/mrf → graph/cuts.{BoykovKolmogorov, AlphaExpansion, QPBO}` (254-C2/C11/C15) for MAP-MRF when factors are submodular F²/F³/F⁴ — *bridging consumer of 254*.
- `prob/mrf → linalg.SparseMatrix` (097-flagged) for large-V message-passing schedules.

**Strict downstream consumers of `prob/mrf/`:**
- `image/segment/graph_cut_mrf.go` (252-S14 wrapped as MRF-API) → uses `mrf.FactorGraph` + `mrf.MAP()` instead of raw graph-cut.
- `prob/hmm.go` (165) → linear-chain-MRF special case (M16 generalises HMM via discriminative reparametrisation).
- `info/code/` (a future NLP-tagging consumer) → linear-chain-CRF for POS / NER / chunking.
- `audio/segmentation/onset_offset.go` → Markov-chain-MAP for onset/offset segmentation (current ad-hoc threshold becomes a CRF post-process).
- 251-shape-opt (combinatorial inflation) → MRF prior over discrete shape-labels.

---

## 1. The twenty-three primitives (M1-M23)

Each entry: name, LOC, reference, API sketch.

### Tier 0 — Factor-graph + MRF substrate (~440 LOC)

**M1 — `prob/mrf/factor.go` ~140 LOC.** Core types + Hammersley-Clifford-aware factor-graph machinery.

```go
type Factor struct {
    Vars     []int     // variable indices
    Card     []int     // cardinality of each variable
    LogTable []float64 // log-potential, length = ∏ Card[k], row-major
}

type Variable struct {
    ID   int
    Card int    // alphabet size (2 for binary, 256 for byte, etc.)
}

type FactorGraph struct {
    Vars    []Variable
    Factors []Factor
}

func (g *FactorGraph) Cliques() [][]int     // maximal cliques over variables
func (g *FactorGraph) MoralGraph() Graph    // undirected moralisation
func (g *FactorGraph) TreeWidth() int       // upper-bound via min-degree ordering
func (g *FactorGraph) MarginalCost() int    // bytes for full marginal-inference
```

**Refs.** Kschischang-Frey-Loeliger-2001 *IT-47:498* factor-graph-and-sum-product; Koller-Friedman-2009 *Probabilistic Graphical Models* §4.

**M2 — `prob/mrf/hammersley_clifford.go` ~80 LOC.** Hammersley-Clifford-1971 (unpublished) / Besag-1974-JRSS-B-36:192 *Spatial Interaction and the Statistical Analysis of Lattice Systems*: an MRF with positive everywhere is a Gibbs-distribution iff the joint factorises over cliques `P(x) = (1/Z)·∏_C ψ_C(x_C)`. API:

```go
func IsGibbs(p func(x []int) float64, cliques [][]int, vars []Variable) bool
func ToGibbsFactors(p func(x []int) float64, cliques [][]int, vars []Variable) []Factor
```

The `IsGibbs` implementation: enumerate all configurations (small-V only) and check that `log P(x) − log P(x')` factorises over the `(x, x')` differences-on-cliques. The `ToGibbsFactors` implementation: clique-marginalisation projection for each maximal clique. **Refs.** Hammersley-Clifford-1971 unpublished; Besag-1974 *JRSS-B-36:192*; Lauritzen-1996 *Graphical Models* §3.2.

**M3 — `prob/mrf/exponential_family.go` ~140 LOC.** Exponential-family-MRF parameterisation `P(x; θ) = exp(θᵀφ(x) − A(θ))` where `φ` are sufficient-statistics over cliques and `A(θ) = log Σ_x exp(θᵀφ(x))` is the log-partition-function. API:

```go
type EFMRF struct {
    Vars     []Variable
    Cliques  [][]int
    Theta    []float64                 // natural parameters
    Phi      func(x []int) []float64   // sufficient statistics
}

func (m *EFMRF) LogPartition() float64               // A(θ)
func (m *EFMRF) Marginals() []float64                // ∂A/∂θ via M5 sum-product
func (m *EFMRF) NaturalToMean(theta []float64) []float64
func (m *EFMRF) MeanToNatural(mu []float64, tol float64) []float64  // dual mapping via M10 mean-field
```

**Refs.** Wainwright-Jordan-2008 *Foundations and Trends in Machine Learning 1:1* exponential-family-and-variational-inference (the canonical unified-language reference).

**M4 — `prob/mrf/icm.go` ~80 LOC.** Besag-1986-JRSS-D-35:96 Iterated-Conditional-Modes coordinate-descent MAP. For each variable `i` in turn, set `xᵢ = argmax_{xᵢ} P(xᵢ | x_{N(i)})` (uses only Markov-blanket factors). Iterate until no change. Provably monotone-non-decreasing in `log P`; converges to *local* maximum (no global guarantee). The simplest MAP-MRF algorithm; ICM is to BP what gradient-descent is to Adam — slower and easier to debug. **Refs.** Besag-1986 *JRSS-D-35:96*.

### Tier 1 — Message-passing inference (~720 LOC)

**M5 — `prob/mrf/sum_product.go` ~140 LOC.** Pearl-1988 *Probabilistic Reasoning in Intelligent Systems* sum-product / belief-propagation on factor-graph. Variables → factors and factors → variables messages:
- `μ_{v→f}(x_v) = ∏_{f' ∈ N(v)\{f}} μ_{f'→v}(x_v)`
- `μ_{f→v}(x_v) = Σ_{x_{N(f)\{v}}} f(x_{N(f)}) · ∏_{v' ∈ N(f)\{v}} μ_{v'→f}(x_{v'})`
- Marginals: `b_v(x_v) ∝ ∏_{f ∈ N(v)} μ_{f→v}(x_v)`

Schedule: leaves-to-root + root-to-leaves (exact on trees); damping-factor on loopy. All operations in log-space via `LogSumExp`. Returns marginals + log-partition-Z. **Refs.** Pearl-1988; Kschischang-Frey-Loeliger-2001 *IT-47:498*.

**M6 — `prob/mrf/max_product.go` ~120 LOC.** Pearl-1988 max-product / Viterbi-on-factor-graph: replace `Σ` with `max` in M5; track back-pointers per max-argmax to recover argmax-x globally. Tie-breaking: lex-min on tied configurations for determinism. **Refs.** Pearl-1988; Wainwright-Jaakkola-Willsky-2003 *IT-49:1120* tree-MAP via max-product = LP-tightness.

**M7 — `prob/mrf/junction_tree.go` ~180 LOC — shared with 254-C22.** Lauritzen-Spiegelhalter-1988-JRSS-50:157. Triangulate moralised factor-graph (min-fill heuristic); maximal-cliques of triangulation; build junction-tree (running-intersection property); two-pass message propagation in `2|E|` clique-marginalisation operations. Exact-inference; cost exponential in tree-width. **Refs.** Lauritzen-Spiegelhalter-1988 *JRSS-50:157*; Cowell-Dawid-Lauritzen-Spiegelhalter-1999 *Probabilistic Networks and Expert Systems*.

**M8 — `prob/mrf/loopy_bp.go` ~140 LOC — shared with 254-C21.** Murphy-Weiss-Jordan-1999-UAI loopy-BP. Schedule M5 (sum-product) or M6 (max-product) on a graph with loops; iterate until message-convergence or `maxIter`. Damping: `μ_{new} = α·μ_{computed} + (1−α)·μ_{prev}` with `α = 0.5` default to mitigate oscillation. Bethe-fixed-point connection: loopy-BP fixed points = stationary-points of Bethe-free-energy (Yedidia-Freeman-Weiss-2005). **Refs.** Murphy-Weiss-Jordan-1999 *UAI-1999*; Yedidia-Freeman-Weiss-2005 *IT-51:2282*; Murphy-2001 *PhD-Thesis* damping-analysis.

**M9 — `prob/mrf/trw.go` ~140 LOC — shared with 254-C20.** Wainwright-Jaakkola-Willsky-2005-IT-51:3697 parallel-TRW + Kolmogorov-2006-PAMI-28:1568 sequential TRW-S. Decompose factor-graph into spanning-trees `{T_k, ρ_k}` with `Σ ρ_k = 1`; on each tree run exact M5 sum-product; combine via convex-LP-relaxation. TRW-S: monotone-non-decreasing dual lower-bound on `−log Z`; parallel-TRW: may oscillate. **Refs.** Wainwright-Jaakkola-Willsky-2005 *IT-51:3697*; Kolmogorov-2006 *PAMI-28:1568*.

### Tier 2 — Variational + cluster (~440 LOC)

**M10 — `prob/mrf/mean_field.go` ~140 LOC — shared with 254-C29.** Mean-field-VI: minimise `KL(q‖p)` over fully-factorised `q(x) = ∏_v q_v(x_v)` via coordinate-update `q_v(x_v) ∝ exp(𝔼_{q_{−v}}[log P(x)])`. Converges to local-minimum of free-energy. Always lower-bounds `log Z` from below. The simplest variational-inference algorithm; production-default for fast-approximate-marginals on huge graphs (Krähenbühl-Koltun-2011 dense-CRF uses 5-iter mean-field). Bethe-approximation companion. **Refs.** Wainwright-Jordan-2008 *FTML-1:1*; Yedidia-Freeman-Weiss-2005 *IT-51:2282*; Bethe-1935 *Proc-Royal-Soc-London-A-150*.

**M11 — `prob/mrf/kikuchi.go` ~160 LOC.** Kikuchi-1951-PR-81:988 cluster-variation-method + Yedidia-Freeman-Weiss-2005 generalised-BP region-graph propagation. Choose a *region-graph* with regions larger than maximal-cliques; tighter free-energy approximation than Bethe. Region-based-free-energy: `F_R(b) = Σ_R c_R·F_R(b_R)` with counting-numbers `c_R` (Möbius inversion). Generalised-BP: messages between regions. **Refs.** Kikuchi-1951 *PR-81:988*; Yedidia-Freeman-Weiss-2005 *IT-51:2282*; Welling-Minka-Teh-2005-UAI region-graph-construction.

**M12 — `prob/mrf/mplp.go` ~140 LOC — shared with 254-C23.** Globerson-Jaakkola-2008-NIPS Max-Product-Linear-Programming dual-decomposition + Sontag-Globerson-Jaakkola-2008-NIPS tightening-via-cycle-inequalities. Tighter LP-relaxation than pairwise-LP: add cluster-marginalisation constraints. **Refs.** Globerson-Jaakkola-2008 *NIPS-2008*; Sontag-Globerson-Jaakkola-2008 *NIPS-2008*.

### Tier 3 — Sampling-based MAP + marginals (~360 LOC)

**M13 — `prob/mrf/gibbs_sampler.go` ~120 LOC — shared with 254-C27.** Geman-Geman-1984-PAMI-6:721 single-site Gibbs-sampler: at each step pick variable `i`; sample `xᵢ ~ P(xᵢ | x_{N(i)})` from local-conditional. Sweep order: lexicographic / red-black / random. With cooling schedule `T_k = c/log(k+2)` → simulated-annealing-MAP (Geman-Geman log-cooling guarantees almost-sure convergence to global-min, in infinite time). Practical schedule: geometric `T_k = T_0·α^k` with `α ∈ [0.95, 0.999]`. **Refs.** Geman-Geman-1984 *PAMI-6:721* (5,000+ citations, the original MRF-MAP paper); Kirkpatrick-Gelatt-Vecchi-1983 *Science-220:671*.

**M14 — `prob/mrf/swendsen_wang.go` ~120 LOC — shared with 254-C28.** Swendsen-Wang-1987-PRL-58:86 cluster-Monte-Carlo for Ising/Potts: build random-bond clusters with probability `1 − exp(−β·V)` along each edge where neighbours agree; flip whole clusters at once. Avoids critical-slowing-down near phase-transitions (Wolff-1989 single-cluster variant for further speedup). Barbu-Zhu-2003 generalisation to MRF. **Refs.** Swendsen-Wang-1987 *PRL-58:86*; Wolff-1989 *PRL-62:361*; Barbu-Zhu-2003 *CVPR-2003*.

**M15 — `prob/mrf/metropolis_hastings.go` ~120 LOC.** Metropolis-Hastings-1953/1970: propose `x' ~ q(x'|x)`; accept with `min(1, P(x')·q(x|x') / P(x)·q(x'|x))` = `min(1, exp(ΔE))` for symmetric proposals on energy-MRF. Adaptive-proposal-Haario-Saksman-Tamminen-2001 covariance-adaptation. Burn-in + thinning. Slower convergence than M13 Gibbs on sparse-graphs but works when local-conditional is intractable. **Refs.** Metropolis-Rosenbluth-Rosenbluth-Teller-Teller-1953 *J-Chem-Phys-21:1087*; Hastings-1970 *Biometrika-57:97*; Haario-Saksman-Tamminen-2001 *Bernoulli-7:223*.

### Tier 4 — Conditional random fields (~620 LOC)

**M16 — `prob/mrf/linear_chain_crf.go` ~220 LOC — KEYSTONE-NLP.** Lafferty-McCallum-Pereira-2001-ICML linear-chain-CRF: `P(y|x; λ) = (1/Z(x))·exp(Σᵗ Σⱼ λⱼ·fⱼ(yᵗ⁻¹, yᵗ, x, t))` where `fⱼ` are arbitrary feature-functions. API:

```go
type LinearChainCRF struct {
    NumStates int
    NumFeats  int
    Lambda    []float64                  // weights, length NumFeats
    Features  func(yPrev, y, t int, x []float64) []float64  // length NumFeats
}

func (c *LinearChainCRF) Forward(x [][]float64) (logAlpha [][]float64, logZ float64)
func (c *LinearChainCRF) Backward(x [][]float64) (logBeta [][]float64)
func (c *LinearChainCRF) Viterbi(x [][]float64) (path []int, logScore float64)
func (c *LinearChainCRF) LogLikelihood(x [][]float64, y []int) float64
func (c *LinearChainCRF) Gradient(x [][]float64, y []int) []float64
func (c *LinearChainCRF) Train(data []Sample, maxIter int, lambda2 float64) []float64
```

Forward / backward in log-space (chain is exact-tree under M5/M6); Viterbi for argmax-decoding; gradient `∂ℓ/∂λᵢ = 𝔼_emp[fᵢ] − 𝔼_λ[fᵢ]` is the empirical-vs-model-expectation; L2-regularised L-BFGS training via `optim.LBFGS`-PRESENT. **Refs.** Lafferty-McCallum-Pereira-2001 *ICML-2001*; Sutton-McCallum-2006 *Introduction to Conditional Random Fields*; McCallum-2003 *UAI* MEMM-vs-CRF analysis.

**M17 — `prob/mrf/skip_chain_crf.go` ~80 LOC.** Sutton-Rohanimanesh-McCallum-2004-ICML skip-chain-CRF: linear-chain-CRF augmented with non-local "skip" edges between same-token-mentions (e.g., for entity-coreference). Inference no longer exact (chain becomes loopy) → use M8 loopy-BP. **Refs.** Sutton-Rohanimanesh-McCallum-2004 *ICML-2004*.

**M18 — `prob/mrf/tree_crf.go` ~80 LOC.** Tree-structured-CRF for parse-tree outputs: variables = tree-nodes, factors = parent-child + sibling. Inference exact via M5/M6 on the tree. Application: dependency-parsing, constituency-parsing, RNA-secondary-structure-prediction. **Refs.** Smith-Smith-2007-EMNLP tree-CRF-parsing; Cohn-Blunsom-2005-ACL semantic-role-labelling.

**M19 — `prob/mrf/dense_crf.go` ~140 LOC — KEYSTONE-VISION.** Krähenbühl-Koltun-2011-NIPS fully-connected-CRF: every pixel-pair is an edge with Gaussian-edge-potential. Naive mean-field: `O(N²)` per iteration; permutohedral-lattice-Adams-Gallup-Davis-Kohli-2010 high-dimensional-Gaussian-filtering reduces to `O(N·d)` per iteration where `d` is the feature-dimension. Five mean-field iterations suffice for production-quality. The post-processing module of DeepLab-v1/v2/v3+ semantic-segmentation. **Refs.** Krähenbühl-Koltun-2011 *NIPS-2011*; Adams-Baek-Davis-2010 *Computer Graphics Forum-29:753* permutohedral-lattice-original; Zheng-Jayasumana-Romera-Paredes-Vineet-Su-Du-Huang-Torr-2015 *ICCV-2015* CRF-as-RNN.

**M20 — `prob/mrf/markov_logic.go` ~100 LOC.** Richardson-Domingos-2006-ML-62:107 Markov-Logic-Network: `P(X = x) = (1/Z)·exp(Σᵢ wᵢ·n_i(x))` where `n_i(x)` is the number of true groundings of first-order-formula `Fᵢ` in world `x` and `wᵢ` is a real-valued weight. The bridge between probabilistic graphical models and first-order-logic. Inference: lift the formula to a ground MRF, run M5/M6/M8 on the ground graph; alternatively, use lifted-inference (Singla-Domingos-2008). **Refs.** Richardson-Domingos-2006 *ML-62:107*; Singla-Domingos-2008-AAAI lifted-inference; Kok-Domingos-2005-ICML structure-learning.

### Tier 5 — Training + parameter learning (~360 LOC)

**M21 — `prob/mrf/pseudo_likelihood.go` ~140 LOC — KEYSTONE-LEARNING.** Besag-1975-Statistician-24:179 pseudo-likelihood `PL(θ) = ∏_i P(xᵢ | x_{N(i)}; θ)` — replace global-likelihood (intractable due to Z) with product of local-conditionals (each is a softmax over `xᵢ`'s alphabet). Consistent estimator (asymptotically-unbiased; Lindsay-1988). Composite-likelihood Lindsay-1988 generalises to other factorisations (block-likelihood, pairwise-likelihood). API:

```go
func PseudoLogLikelihood(data [][]int, theta []float64, mrf *EFMRF) float64
func PseudoLogLikelihoodGradient(data [][]int, theta []float64, mrf *EFMRF) []float64
func TrainPL(data [][]int, mrf *EFMRF, maxIter int, l2Reg float64) []float64  // L-BFGS
```

**Refs.** Besag-1975 *Statistician-24:179*; Lindsay-1988 *Contemp-Math-80:221* composite-likelihood; Varin-Reid-Firth-2011 *Stat-Sinica-21:5* composite-likelihood-overview.

**M22 — `prob/mrf/contrastive_divergence.go` ~120 LOC.** Hinton-2002-NeuralComp-14:1771 CD-1 / CD-k Boltzmann-machine training. Approximate `∂ log P / ∂θ = 𝔼_data[φ] − 𝔼_model[φ]` by replacing `𝔼_model[φ]` with `𝔼_{x ~ k-step-Gibbs-from-data}[φ]`. CD-1 = single-step-Gibbs (Hinton's empirical observation: works surprisingly well). Tieleman-2008-ICML Persistent-CD (PCD): keep a persistent-fantasy-particle across mini-batches → better gradient at small extra cost. The standard training algorithm for Restricted-Boltzmann-Machines pre-deep-learning (used in Hinton-Salakhutdinov-2006-Science DBN-pretraining). **Refs.** Hinton-2002 *Neural-Computation-14:1771*; Tieleman-2008 *ICML-2008* PCD; Carreira-Perpiñán-Hinton-2005-AISTATS CD-bias-analysis.

**M23 — `prob/mrf/piecewise_training.go` ~100 LOC.** Sutton-McCallum-2007-ML-66:301 piecewise-training: decompose joint-likelihood into independent factor-likelihoods, train each independently (each factor is a small softmax), no global-Z computation. Asymptotically-biased but trains in `O(|F|)` instead of `O(|F|·|V|)` per epoch. Practical when factors share-structure (template-CRF). **Refs.** Sutton-McCallum-2007 *ML-66:301* piecewise-training; Sutton-Minka-2006 piecewise-pseudo-likelihood.

---

## 2. Connective tissue + cross-package blockers

**Substrate-blocker-1 (HARD)** `prob.LogSumExp` / `Log1mExp` / `Log1pExp` — currently private to `changepoint/bocpd.go`. 165 already flags this as a `prob/mathutil.go` promotion. Every M-primitive uses `LogSumExp` for log-space message-product. Promotion is ~30 LOC + test migration.

**Substrate-blocker-2 (HARD)** `prob/mrf.Factor / FactorGraph` — the type-system foundation. Pure value-types ~120 LOC of M1.

**Substrate-blocker-3 (SOFT)** `prob/hmm.go` (165) — overlap with M16 LinearChainCRF inference (forward / backward / Viterbi). Recommend 165 ships first as the `prob/hmm.go` directed-Markov-emission-model; M16 reuses 165's forward/backward in a *discriminative* chain-CRF wrapper sharing the DP infrastructure.

**Substrate-blocker-4 (SOFT)** `optim.LBFGS` (PRESENT). All CRF-training methods consume this. ZERO BLOCKERS.

**Substrate-blocker-5 (SOFT)** `optim/proximal.Admm` (PRESENT). M12 MPLP / AD³-style consumes. ZERO BLOCKERS.

**Substrate-blocker-6 (SOFT)** `linalg.SparseMatrix` (097-flagged ABSENT). Gates dense-CRF M19 large-image inference at `> 1M` pixels. M19 ships at quality-bar without sparse — permutohedral-lattice is dense O(N·d) without needing sparse-matrix.

**Substrate-blocker-7 (SOFT)** `graph/cuts/{BoykovKolmogorov, AlphaExpansion, QPBO}` (254-C2/C11/C15). Optional — when a factor-graph has only F²-submodular pairwise-potentials, MAP is reducible to graph-cut and `mrf.MAP()` should dispatch to 254 for performance. Soft cross-link, defers to v1.1+.

**Substrate-blocker-8 (NONE)** `optim.SimulatedAnnealing` (PRESENT). M13 Gibbs-MAP via 254 schedule + 117 PRNG. M13 cooling-schedule is a 1-D-temperature parameter that the PRESENT SA infrastructure supplies.

**Total upstream-substrate dependency** for the `prob/mrf/` slot (not counting 165's `prob/mathutil` promotion which is HARD blocker): ~30 LOC of NEW code in non-`prob/mrf/` paths is required before the 22 ship-against-PRESENT-substrate primitives go in. **Cheapest-no-blocker subset** (assuming 165 lands first): **M1 + M4 ICM + M5 sum-product + M6 max-product + M16 LinearChainCRF-inference (no training) ~640 LOC** — the foundational PGM API.

**Recommended PR sequence:**

- **PR-A (Tier-0 substrate ~440 LOC, 1 week)** M1 Factor / FactorGraph + M2 Hammersley-Clifford + M3 ExponentialFamily + M4 ICM. Foundational types + the simplest MAP algorithm. ZERO blockers (`prob.LogSumExp` promotion comes via 165-PR or this PR).
- **PR-B (Tier-1 message-passing ~720 LOC, 2 weeks)** M5 sum-product + M6 max-product + M7 junction-tree + M8 loopy-BP + M9 TRW. Each shared algorithm (M7/M8/M9) ships as the *probabilistic-API thin-wrapper* over 254-C20/C21/C22 implementation if 254 lands first; ships as the *primary implementation* otherwise.
- **PR-C (Tier-2 variational ~440 LOC, 1.5 weeks)** M10 mean-field + M11 Kikuchi + M12 MPLP. Mean-field is the production-default for dense-CRF M19.
- **PR-D (Tier-3 sampling ~360 LOC, 1 week)** M13 Gibbs + M14 Swendsen-Wang + M15 Metropolis-Hastings.
- **PR-E (Tier-4 CRF ~620 LOC, 2 weeks)** M16 LinearChainCRF + M17 SkipChainCRF + M18 TreeCRF + M19 DenseCRF + M20 MarkovLogic. KEYSTONE for NLP and computer-vision consumers.
- **PR-F (Tier-5 training ~360 LOC, 1.5 weeks)** M21 PseudoLikelihood + M22 ContrastiveDivergence + M23 PiecewiseTraining. KEYSTONE for MRF parameter-learning.

Total `prob/mrf/` PR-A through PR-F: ~2,940 LOC, ~9 engineer-weeks. Smaller than 254 (3,640 LOC) because 8 of 23 primitives are *probabilistic-API thin-wrappers* over 254-shared algorithms (~80 LOC API-bridge per shared algo instead of full ~140 LOC reimplementation).

**De-duplication policy with 254:** for the 8 shared algorithms (M7=254-C22, M8=254-C21, M9=254-C20, M10=254-C29, M12=254-C23, M13=254-C27, M14=254-C28, plus M19 dense-CRF which has no 254-counterpart), the implementation lives in `graph/cuts/` (254) keyed by `[]float64` log-potentials over discrete labels; `prob/mrf/` (255) wraps with `Factor`/`FactorGraph` API. ~80 LOC of bridge-LOC per shared algorithm vs ~140 LOC of duplicate full-implementation → save ~480 LOC across the boundary. If 254 ships first the 255 versions are ~80-LOC thin-wrappers; if 255 ships first, 254 is the thin-wrapper around 255's typed-API → either way the math is implemented exactly once.

---

## 3. Five R-MUTUAL-CROSS-VALIDATION 3/3 pins this slot enables

**Pin 1 — Hammersley-Clifford-equivalence on a 4-clique grid-MRF.** Three paths to identical marginals + log-Z on a 3×3 binary grid (9 binary variables → 512 configs, brute-force tractable):
- M2 ToGibbsFactors → M5 sum-product on the resulting factor-graph
- M5 sum-product directly on the user-specified clique-decomposition
- Brute-force enumeration of 2^9 = 512 configurations (oracle)

All three must agree on per-variable marginals to 1e-12 and on log-partition Z to 1e-12. Saturates 3/3 + a Hammersley-Clifford-correctness pin (the equivalence holds iff M2's output matches M5-on-cliques).

**Pin 2 — Linear-chain-CRF on a synthetic 100-token sequence.** Three paths to identical Viterbi-decoding + log-Z + gradient on a synthetic linear-chain-CRF with 5 states + 20 features:
- M16 LinearChainCRF.Viterbi (max-product-on-chain via M6)
- M5 sum-product + greedy-argmax (different code-path, must match Viterbi on uni-modal posterior)
- 165 HMM.Viterbi when the CRF reduces to a generative HMM (set features = log-emission + log-transition)

All three paths must agree on argmax-path. The HMM-vs-CRF reduction validates the *structural-equivalence* of generative-HMM and discriminative-linear-chain-CRF when the CRF is parameterised with the same log-potentials. Saturates 3/3 + bridges to 165.

**Pin 3 — Mean-field free-energy lower-bounds true log-Z.** Three paths on a 4×4 Ising-MRF (16 variables, brute-force tractable):
- M10 mean-field converged → lower-bound `F_MF(q) ≤ −log Z`
- M5 sum-product is exact only on trees → on this loopy graph, M5 produces *Bethe-approximation* `F_Bethe`; Yedidia-Freeman-Weiss-2005 proves `F_Bethe ≤ F_MF` (Bethe tighter than mean-field)
- Brute-force enumeration of 2^16 = 65,536 configurations → exact `−log Z` (oracle)

All three must satisfy `F_MF ≥ F_Bethe ≥ −log Z` and `F_Bethe − (−log Z)` is the Bethe-approximation-error metric. Saturates 3/3 + a variational-inequality-pin.

**Pin 4 — Pseudo-likelihood gradient = exact-likelihood gradient on a tree.** Three paths on a 6-node tree-structured-MRF (binary, 64 configs):
- M21 PseudoLogLikelihood.Gradient (sums of local-conditional log-derivatives)
- Exact MLE gradient `∂ ℓ/∂θ = 𝔼_data[φ] − 𝔼_θ[φ]` with `𝔼_θ[φ]` from M7 junction-tree (exact on tree)
- Brute-force `𝔼_θ[φ]` from enumeration (oracle)

On a tree, pseudo-likelihood is asymptotically-equivalent to exact-likelihood (both consistent estimators); on small data-samples the gradients differ but converge as N→∞. Saturate 3/3 by checking gradient-direction-cosine > 0.99 at large N. Plus a "directed-loopy-pin": on a 6-cycle the pseudo-likelihood gradient differs from exact, certifying the *bias of pseudo-likelihood on loopy MRFs* — a known property.

**Pin 5 — TRW-S monotone-dual on the same input as 254-Pin-4.** Three paths on a synthetic non-submodular factor-graph:
- M9 TRW-S (this slot's wrapper) → monotone-non-decreasing dual-LP-bound sequence
- 254-C20 TRW-S (the underlying graph-cut implementation) → must produce *identical* dual-bound sequence
- M5 sum-product + 254-C25 branch-and-bound LP-relaxation → exact MAP for small instances (oracle)

M9 and 254-C20 must produce *byte-identical* dual-bound sequence (proves the API-bridge is faithful); both must satisfy `dual ≤ MAP-energy` (LP-relaxation property). Saturates 3/3 + a no-drift-across-API-boundary pin certifying the 254↔255 shared-algorithm bridge.

---

## 4. Touchpoints with other agents

- **165 (synergy-sequence-prob):** STRICT-PRECURSOR. 165's `prob/hmm.go` HMM tier (forward / backward / Viterbi / Baum-Welch ~520 LOC) is the *generative directed* analogue of 255-M16's *discriminative undirected* linear-chain-CRF. Recommend 165 ships first; 255-M16 reuses 165's DP-on-chain. The LogSumExp-promotion-from-`changepoint/bocpd.go`-to-`prob/mathutil.go` (165 + 117 flag) is HARD-BLOCKER for both 165 and 255.
- **254 (graph-cuts):** PARTIAL-OVERLAP-COMPATIBLE. 254-C20/C21/C22/C23/C27/C28/C29 = 255-M9/M8/M7/M12/M13/M14/M10. Implementation lives in `graph/cuts/`; 255 wraps with `Factor`/`FactorGraph` API. Net new LOC at the 254↔255 boundary is ~80 LOC of bridge per shared algorithm, not full reimplementation.
- **252 (image-segmentation) S14, S15, S20:** STRICT-DOWNSTREAM-CONSUMER. 252's image-segmentation MRF-MAP variant uses `mrf.MAP(g, FactorGraph{...})` instead of raw graph-cut for the image-shaped consumer-API.
- **253 (active-contours):** SOFT-CROSS-LINK. 253-A22 multi-phase-Chan-Vese can use M16 multi-label-CRF as alternative to log₂(K) level-sets.
- **117 (prob-missing):** STRICT-PREREQUISITE for M13 Gibbs (PRNG primitives) + Dirichlet/Categorical distributions for emission-distribution prior; the Gibbs-sampler relies on a vetted PRNG (currently `math/rand` is acceptable; 117 enumerates a deterministic-cross-language PRNG that will replace).
- **127 (sequence-catalogue):** SOFT-LINK. The sequence-prob synergy 165 enumerated names linear-chain-CRF as the "post-edit-distance" probabilistic layer of the alignment-stack; 255-M16 fulfils that.
- **151 (prob-bayes-missing):** SOFT-LINK. Bayesian-network slot would consume 255-M5/M6 (sum-product / max-product on directed-acyclic-factor-graph is exact and reuses M5/M6 unchanged).
- **244 (pde-solvers) D12 ConjugateGradient:** SOFT-LINK. Required for sparse-LP-MRF inference in M9 TRW-S at large-V (097 / 244 already enumerate).
- **097 (linalg-missing):** Sparse-matrix for large-grid M9 / M10 / M19. Soft-blocker.
- **Nothing currently scheduled for `prob/graphical/` Bayesian-network:** recommend opening `new-bayesian-network`, `new-causal-inference`, `new-do-calculus`, `new-pearl-causality` slots in a future overnight grid (aligned with 165's HMM, 255's MRF/CRF, 254's graph-cuts as the four-pillar PGM tier).

---

## 5. Singular load-bearing recommendation

**Ship PR-A (Tier-0 substrate) FIRST as the SINGULAR-FOUNDATIONAL ~440 LOC, 1 week.** M1 Factor + FactorGraph is the type-system foundation for every downstream MRF/CRF primitive — it is impossible to ship message-passing or CRF-training without first agreeing on the `Factor{Vars, Card, LogTable}` and `FactorGraph{Vars, Factors}` value-types. M2 Hammersley-Clifford is the singular-pedagogical primitive (the *fundamental theorem* of MRFs, 1971-unpublished, Besag-1974-published) and serves as the equivalence-checker that validates any user-supplied clique-decomposition against the ground-truth Gibbs-distribution. M3 ExponentialFamily is the modern unified-language for PGM (Wainwright-Jordan-2008 *Foundations and Trends*). M4 ICM is the simplest MAP-MRF algorithm (Besag-1986) and the right pedagogical entry-point — coordinate-ascent on local-conditionals, monotone-non-decreasing log-P, no message-passing-machinery to debug.

**Then ship PR-E (Tier-4 CRF) ~620 LOC, 2 weeks** because M16 LinearChainCRF is the single most-deployed PGM primitive in NLP (POS-tagging, NER, chunking, segmentation; >40,000 citations on Lafferty-McCallum-Pereira-2001) — and a zero-dep cross-language Go + Python + C++ + C# byte-identical linear-chain-CRF is a **unique reality contribution NO existing library provides at that quality bar**. M19 DenseCRF is the canonical post-processing for semantic-segmentation; together M16 + M19 cover the entire PGM-applications-tier in NLP and vision.

**Then ship PR-B (Tier-1 message-passing) ~720 LOC, 2 weeks** as the inference-engine substrate — once factor-graph + CRFs ship, message-passing is the natural completion. Strongly defer to 254-C20/C21/C22/C23 implementations if 254 lands first; ship as primary if 255 lands first.

**Then ship PR-F (Tier-5 training) ~360 LOC, 1.5 weeks** because M21 PseudoLikelihood + M22 ContrastiveDivergence are the practical-MRF-training algorithms — without them users cannot fit MRFs from data, only do inference on pre-specified models.

**Defer PR-C (Tier-2 variational) and PR-D (Tier-3 sampling)** as v1.1 — both are valuable but the singular-moat is M16 LinearChainCRF + M19 DenseCRF.

**Defer PR-D-Markov-Logic (M20)** — Richardson-Domingos-2006 MLN is a beautiful unification of probability and first-order-logic but the consumer-base is small (mostly-academic AI/symbolic-reasoning research); ship as v1.2.

**Avoid scoping: deep-CRF-as-RNN (Zheng-Jayasumana-Romera-Paredes-Vineet-Su-Du-Huang-Torr-2015 ICCV).** Differentiable-MRF / CRF-as-RNN is a deep-learning primitive — aicore-territory not reality-territory. M19 DenseCRF ships the *non-differentiable* permutohedral-lattice mean-field; the differentiable-CRF-as-RNN is the gradient-through-mean-field-iterations variant.

**Avoid scoping: graph-neural-network message-passing.** GNN message-passing is structurally identical to M5 / M8 but parameterised by learned-tensors not analytic-potentials → aicore-territory.

**Avoid scoping: lifted-inference (Singla-Domingos-2008, Kersting et al.).** Lifted-inference for Markov-logic-networks is academic-niche; defer to v1.2+.

**Final precision-hazards:**
- **(a) LogSumExp underflow/overflow.** Every M-primitive uses log-space. The `LogSumExp(a,b)` from `changepoint/bocpd.go:294-305` handles `−Inf` correctly; vector-form `LogSumExpVec` must subtract max-element first to avoid overflow. Test with `log P` values in `[−1e6, 0]`.
- **(b) Pseudo-likelihood-bias-on-loopy-graphs.** Besag-1975 PL is consistent on tree-MRFs only; on loopy-MRFs the bias is `O(1/N)` and vanishes asymptotically but slow-convergence on small-data. Document explicitly; M21 documentation must enumerate the regime.
- **(c) CD-bias.** Hinton-2002 CD-1 is biased; gradient-direction is correct but magnitude is wrong by `O(1)` factor. Document; recommend M22 with k≥10 for production-quality.
- **(d) Loopy-BP-non-convergence.** M8 may oscillate on loopy graphs; damping-factor `α = 0.5` default is safe; expose for tuning. Adaptive-damping (Elidan-McGraw-Koller-2006-UAI) is v1.1 polish.
- **(e) Junction-tree treewidth-explosion.** M7 is exponential in tree-width; treewidth(W×H grid) = min(W, H) → unusable on > 100×100 grid. M7 documentation must enumerate this regime; consumers should fall through to M8/M9 on grid-graphs.
- **(f) TRW tree-decomposition non-uniqueness.** Different tree-decompositions yield different schedules; standard for grid-MRF: row-trees + column-trees (Wainwright-Jaakkola-Willsky-2005 §IV-C). Pin this for cross-language reproducibility.
- **(g) Gibbs-sampler-cooling-schedule.** Geman-Geman-1984 logarithmic cooling `T_k = c/log(k+2)` guarantees a.s. convergence to global-min in infinite time; geometric cooling `T_k = T_0·α^k` is faster but no guarantee. Document both; default to geometric with a clear-warning that result is local-only.
- **(h) Dense-CRF-permutohedral O(N·d) but d-dependent constant is huge.** For `d=5` (Lab + xy) feature-space the constant is ~32; for `d=10` (Lab + xy + RGB + Lab-gradients) it is ~1024. M19 documentation must enumerate.
- **(i) Markov-Logic-grounding-explosion.** A first-order-formula with k variables over a domain of size N grounds to N^k formulas → N=10000 + k=3 = 10^12 formulas. M20 must support lazy-grounding or fail-fast.
- **(j) Hammersley-Clifford-positive-everywhere requirement.** H-C-1971 holds only for distributions strictly positive on the entire configuration space; M2 must error or warn on zero-probability configurations (these break the equivalence; the Möbius inversion fails).
- **(k) Partition-function-Z computation is #P-hard in general.** M3 LogPartition delegates to M5 (exact on tree) / M9 (LP-bound on loopy) / M11 (Kikuchi-bound on loopy). Document the regime where each is exact.

**Headline:** Twenty-three Markov-random-field / conditional-random-field / probabilistic-graphical-model primitives close the entire 1971-2015 MRF/CRF canon (Hammersley-Clifford-1971 / Besag-1974 + Besag-1986 ICM / Geman-Geman-1984 Gibbs-MAP / Pearl-1988 sum-product + max-product + BP / Lauritzen-Spiegelhalter-1988 junction-tree / Murphy-Weiss-Jordan-1999 loopy-BP / Lafferty-McCallum-Pereira-2001 linear-chain-CRF / Yedidia-Freeman-Weiss-2005 Bethe + Kikuchi + GBP / Wainwright-Jaakkola-Willsky-2005 TRW / Kolmogorov-2006 TRW-S / Sutton-McCallum-2006 CRF-tutorial / Globerson-Jaakkola-2008 MPLP / Sutton-Rohanimanesh-McCallum-2004 skip-chain-CRF / Richardson-Domingos-2006 Markov-Logic / Krähenbühl-Koltun-2011 dense-CRF / Hinton-2002 CD + Tieleman-2008 PCD / Besag-1975 pseudo-likelihood / Sutton-McCallum-2007 piecewise-training / Swendsen-Wang-1987 cluster-MC / Metropolis-1953 + Hastings-1970 MH / Wainwright-Jordan-2008 exponential-family-PGM) in ~2,940 LOC of pure synthesis on top of `prob.MarkovSteadyState`-PRESENT (oracle for chain-marginals via `π·P^t`), `optim.LBFGS`-PRESENT (CRF parameter-learning), `optim/proximal.Admm`-PRESENT (AD³ ADMM-bridge), `optim.SimulatedAnnealing`-PRESENT (Geman-Geman cooling-substrate), `prob.NormalPDF`-PRESENT (emission-distributions), and 8 algorithms shared-with-254 (~80 LOC bridge per shared algo); cheapest-no-blocker subset M1 + M4 + M5 + M6 + M16-inference ~640 LOC; foundational keystone M1 Factor + FactorGraph + M16 LinearChainCRF ~360 LOC; singular-moat M16 + M19 DenseCRF + M21 PseudoLikelihood ~500 LOC; 2024-frontier M11 Kikuchi + M19 DenseCRF + M22 PCD ~420 LOC; pedagogical M2 H-C + M3 ExponentialFamily + M4 ICM ~300 LOC. Five R-MUTUAL-CROSS-VALIDATION 3/3 pins enabled (H-C↔sum-product↔brute-force on 3×3 grid, LinearChainCRF↔HMM↔sum-product on 100-token chain, mean-field↔Bethe↔brute-force on 4×4 Ising, pseudo-likelihood↔exact-MLE↔junction-tree on 6-node tree, M9-TRW-S↔254-C20-TRW-S↔branch-and-bound LP-tightness). Strict-precursor 165 (HMM tier) + LogSumExp-promotion from `changepoint/bocpd.go` to `prob/mathutil.go`; strict-overlap-compatible 254 (graph-cuts) sharing 8 algorithms via API-bridge; strict-downstream consumers 252 (image-segmentation MRF-API), 253 (multi-phase-Chan-Vese discrete-CRF alternative), `audio/segmentation/onset_offset.go` (post-process via Markov-chain-MAP). Recommended placement NEW sub-package `prob/mrf/` under existing `prob/` package. PR-A SINGULAR-FOUNDATIONAL ~440 LOC ships first against PRESENT substrate with ZERO upstream blockers (assuming concurrent 165-PR delivers `LogSumExp`-promotion).
