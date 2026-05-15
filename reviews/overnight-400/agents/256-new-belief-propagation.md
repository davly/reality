# 256 | new-belief-propagation — Pearl-1988 sum-product / max-product / loopy-BP / junction-tree / TRW(-S) / MPLP / generalised-BP / Kikuchi / Bethe-free-energy / mean-field / variable-elimination / bucket-elimination / mini-bucket / damping / async-vs-sync schedules / residual-BP / convex-BP / dual-decomposition / particle-BP / non-parametric-BP / survey-propagation / cavity-method / BP-for-LDPC / BP-as-LP-relaxation / BP-vs-Bethe-fixed-point

**Summary line 1.** reality v0.10.0 ships **ZERO** belief-propagation surface — repo-wide grep on `belief.propagat|sum.product|max.product|max.sum|min.sum|message.passing|factor.graph|junction.tree|clique.tree|tree.decomposition|loopy.bp|loopy.belief|bethe.free|kikuchi|cluster.variation|generalized.bp|generalised.bp|gbp|region.graph|trw|tree.reweighted|wainwright.jaakkola.willsky|kolmogorov.trws|mplp|globerson.jaakkola|ad3|martins.figueiredo|variable.elimination|bucket.elimination|mini.bucket|dechter|residual.bp|elidan.mcgraw.koller|damping.bp|asynchronous.bp|synchronous.bp|particle.bp|nonparametric.bp|sudderth.ihler.freeman.willsky|stochastic.bp|survey.propagation|mezard.parisi.zecchina|approximate.survey.propagation|cavity.method|replica.method|convex.bp|convex.message.passing|dual.decomposition.bp|lp.relaxation.map|local.marginal.polytope|marginal.polytope|reweighted.sum.product|fractional.bp|expectation.propagation|ep.minka|kschischang.frey.loeliger|pearl.1988|yedidia.freeman.weiss|tanaka.cavity|tanner.graph|gallager.bp|mackay.neal|sum.product.ldpc|min.sum.ldpc|forney.factor|kikuchi.cluster|wiegerinck|heskes.convergent|convergent.message.passing|cmp.heskes|residual.belief.propagation|priority.bp|junction.graph|cluster.graph|expectation.maximization.bp` returns **zero callable matches** anywhere in `*.go` outside the false-positive name-collisions `acoustics/acoustics.go` (sound *propagation*, wave-equation, NOT belief-propagation), `autodiff/doc.go` ("backpropagation" = reverse-mode AD on differentiable computations, NOT message-passing on factor-graphs), `physics/materials.go::Paris-Erdogan` (crack-*propagation* fatigue, NOT BP), `graph/centrality.go::Brandes-betweenness` ("back-propagation of dependencies" = Brandes-2001 dependency-accumulation in BFS for betweenness, NOT factor-graph BP), `timeseries/dcc/doc.go::DCC` (volatility-*propagation* / shock-*propagation* in time-series, NOT BP), `signal/window.go::Hamming` (spectral-window, not Hamming-code-BP-LDPC). The entire 1988-2025 belief-propagation / probabilistic-graphical-model-inference canon (Pearl-1988 *Probabilistic Reasoning in Intelligent Systems* Morgan-Kaufmann tree-BP-original; Lauritzen-Spiegelhalter-1988-JRSS-50:157 junction-tree-original; Kschischang-Frey-Loeliger-2001-IT-47:498 factor-graph-and-sum-product-unified-framework; Murphy-Weiss-Jordan-1999-UAI loopy-BP-empirical-success; Yedidia-Freeman-Weiss-2005-IT-51:2282 generalised-BP-and-Bethe-Kikuchi-free-energy; Wainwright-Jaakkola-Willsky-2005-IT-51:3697 TRW-message-passing-and-LP-relaxation; Kolmogorov-2006-PAMI-28:1568 TRW-S-sequential; Heskes-2006-NeurIPS convergent-message-passing-CCCP; Globerson-Jaakkola-2008-NIPS MPLP; Martins-Figueiredo-Aguiar-Smith-Xing-2011-ICML AD³; Sudderth-Ihler-Freeman-Willsky-2003-CVPR particle-BP for continuous-state-spaces; Ihler-McAllester-2009-AISTATS particle-BP-for-continuous; Mezard-Parisi-Zecchina-2002-Science-297:812 survey-propagation-for-random-3-SAT; Mezard-Montanari-2009 *Information, Physics, and Computation* cavity-method-canonical-treatise; Gallager-1963 *Low-Density-Parity-Check-Codes* MIT-PhD original-LDPC-BP; MacKay-Neal-1996-Electron-Lett-32:1645 LDPC-rediscovery-BP; Richardson-Urbanke-2008 *Modern Coding Theory* density-evolution-and-EXIT-charts; Dechter-1999-Artif-Intell-113:41 bucket-elimination; Dechter-Rish-2003-J-ACM-50:107 mini-bucket; Elidan-McGraw-Koller-2006-UAI residual-BP-priority-queue; Wiegerinck-Heskes-2003-NIPS fractional-BP; Minka-2001-MIT-PhD expectation-propagation as one-pass-BP-on-Gaussian-EF) is wholly **ABSENT**. **PARTIAL OVERLAP with 254 (graph-cuts, slot-precursor) ~620 LOC**: 254-C20 TRW-S, 254-C21 loopy-BP, 254-C22 junction-tree, 254-C23 MPLP, 254-C29 mean-field/Bethe — five of the canonical message-passing algorithms ship in 254 as the *energy-minimisation-side-API* over `[]float64` log-potentials on grid-graphs. **PARTIAL OVERLAP with 255 (MRF, slot-twin) ~720 LOC**: 255-M5 sum-product, 255-M6 max-product, 255-M7 junction-tree, 255-M8 loopy-BP, 255-M9 TRW, 255-M10 mean-field, 255-M11 Kikuchi-GBP, 255-M12 MPLP — eight of the canonical message-passing algorithms ship in 255 as the *probabilistic-API* over typed `Factor`/`FactorGraph`. **Slot 256 is the BP-deep-dive that lives ORTHOGONAL to 254 (energy-min-API) and 255 (PGM-API)** — slot 256's value is the BP-SPECIFIC enumeration that NEITHER 254 NOR 255 covered: scheduling theory (synchronous-vs-asynchronous-vs-residual; Elidan-McGraw-Koller-2006 priority-residual-BP gives ≥3× convergence speedup over flooding-schedule, never appeared in 254/255), variable-elimination + bucket-elimination + mini-bucket (Dechter-1999 — exact-inference-via-elimination is a *different* abstraction than message-passing; mini-bucket gives the only practical *bound* on log-Z with controllable accuracy/cost — never appeared), particle-BP / non-parametric-BP / stochastic-BP for continuous state-spaces (Sudderth-Ihler-Freeman-Willsky-2003; Ihler-McAllester-2009 — never appeared because 254/255 are discrete-only), survey-propagation for random-3-SAT (Mezard-Parisi-Zecchina-2002 *Science* — solves random-k-SAT at clause-density >4.2 where DPLL fails; never appeared in 254/255 because SAT is not a probability distribution but a constraint-satisfaction problem), the cavity-method-statistical-physics derivation (Mezard-Montanari-2009 — derives BP from first principles via the Bethe-Peierls-cluster approximation, the *physics* origin of all BP), convergent-message-passing / CCCP / Heskes-2006 (the *convexified* BP that is GUARANTEED to converge unlike loopy-BP), expectation-propagation (Minka-2001 — BP for continuous-exponential-family approximations), tree-decomposition + treewidth (the COMPLEXITY-PARAMETER that determines whether BP is exact-tractable or NP-hard), BP-for-LDPC-decoding (Gallager-1963 + MacKay-Neal-1996 — the SINGLE largest practical deployment of BP, in every WiFi/5G/satellite-comm receiver — *cross-link to 210 coding theory which already enumerated this*), and BP for combinatorial-optimisation (matching, colouring, max-cut). **CROSS-LINK to 210 (coding-theory) ~640 LOC**: 210-`coding/ldpc/` slot enumerates "sum-product / min-sum / offset-min-sum decoders ~640 LOC" — exactly the BP-LDPC instantiation. Slot 256 places `coding/ldpc/sum_product.go` as a *consumer* of the abstract `prob/mrf/sum_product.go` (255-M5) over the GF(2)-Tanner-graph factor-graph: the LDPC-decoder *is* sum-product-BP on the parity-check Tanner-graph with binary variables and parity-check factors. 210-PR delivers the GF(2) substrate; 255-M5 delivers the BP engine; slot 256-T22 BP-LDPC bridge wires them together (~80 LOC). **CROSS-LINK to 165 (synergy-sequence-prob) ~520 LOC**: 165 enumerates HMM forward-backward = BP-on-chain-tree (special-case of sum-product); slot 256-T19 documents this as the chain-MRF / chain-CRF-special-case of M5/M6. **CROSS-LINK to 215 (compressed-sensing) ~280 LOC**: 215 enumerates AMP (Approximate-Message-Passing, Donoho-Maleki-Montanari-2009-PNAS-106:18914) as a BP-DERIVATIVE for compressed-sensing — relaxed-BP (rBP) on dense factor-graphs with Onsager-correction-terms from the cavity-method. Slot 256-T15 names AMP as the cavity-method-derived dense-factor-graph BP variant. **Block-C verdict:** the BP-specific deep-dive is partially covered by 254 (5 algos) and 255 (8 algos) but the BP-SCHEDULING / VARIABLE-ELIMINATION / PARTICLE-BP / SURVEY-PROPAGATION / EXPECTATION-PROPAGATION / CONVERGENT-MP / CAVITY-METHOD / TREE-DECOMPOSITION / LDPC-BRIDGE / AMP-CROSS-LINK tier is the BP-deep-dive that this slot enumerates as ~2,140 LOC of *additive* surface that 254 + 255 + 210 do NOT cover.

**Summary line 2.** Twenty-four primitives B1-B24 totalling ~2,140 LOC organised as **(a) Tier-0 BP foundations 254/255-shared ~0 LOC** (B1-B7 are explicit *cross-references* to 255-M1 Factor/FactorGraph + 255-M5 sum-product + 255-M6 max-product + 255-M7 junction-tree + 255-M8 loopy-BP + 255-M9 TRW + 254-C29/255-M10 mean-field; this slot does NOT re-enumerate them — they ship in 255-PR-B / 254-PR-D and slot 256 imports them via `prob/mrf` typed-API), **(b) Tier-1 BP scheduling theory ~360 LOC NEW** (B8 `prob/mrf/schedule.go` synchronous-flooding-schedule / asynchronous-Gauss-Seidel-schedule / round-robin / random-order schedule taxonomy ~80 LOC; B9 `prob/mrf/residual_bp.go` Elidan-McGraw-Koller-2006-UAI residual-belief-propagation: priority-queue keyed by `‖μ_new − μ_old‖_∞` per edge, always update message with largest-residual first → empirically 3-10× fewer iterations than flooding ~140 LOC; B10 `prob/mrf/damping.go` Murphy-2001 damping `μ_new := α·μ_computed + (1−α)·μ_prev` with `α ∈ [0.1, 0.9]` to mitigate loopy-BP-oscillation + Heskes-2003 fractional-BP `α^t` decay schedule + adaptive-damping with line-search per Pretti-2005 ~80 LOC; B11 `prob/mrf/convergence_check.go` message-residual-norm + KL-divergence-of-marginals + log-Z-stability stopping criteria ~60 LOC), **(c) Tier-2 variable / bucket elimination ~420 LOC NEW** (B12 `prob/mrf/variable_elimination.go` Zhang-Poole-1996 / Dechter-1996 variable-elimination with min-fill / min-degree / weighted-min-fill ordering heuristics — exact-inference, exponential in induced-treewidth, the simplest exact-inference algorithm and the natural pedagogical bridge between brute-force and message-passing ~140 LOC; B13 `prob/mrf/bucket_elimination.go` Dechter-1999 *Artif-Intell-113:41* bucket-elimination unified-framework subsuming variable-elimination + Viterbi + sum-product as a single bucket-process schedule with `combine` and `eliminate` operators ~120 LOC; B14 `prob/mrf/mini_bucket.go` Dechter-Rish-2003 *J-ACM-50:107* mini-bucket: bound-controlled approximate-elimination — partition each bucket into mini-buckets of size ≤ `i` and approximate the elimination, giving log-Z upper-bound (max-version) and log-Z lower-bound (mean-version), with `i` as the speed/accuracy knob; the SINGLE algorithm in this taxonomy that produces a *certified bound* on log-Z with NO loop-iteration ~140 LOC; B15 `prob/mrf/treewidth.go` Bodlaender-2006 / min-degree heuristic / min-fill heuristic upper-bounds + Bodlaender-Koster-2010 lower-bound treewidth-estimation; treewidth IS the complexity-parameter that determines `O(N·d^{tw+1})` exact-inference cost ~80 LOC), **(d) Tier-3 convergent / convex BP ~360 LOC NEW** (B16 `prob/mrf/convergent_mp.go` Heskes-2006 *NeurIPS-2006* convergent-message-passing via CCCP (Concave-Convex-Procedure) — guaranteed-monotone-convergence to a Bethe-stationary-point unlike loopy-BP which may oscillate ~140 LOC; B17 `prob/mrf/convex_bp.go` Wainwright-Jaakkola-Willsky-2005 *IT-51:2697* convex-BP / fractional-BP with edge-weights `ρ_e` such that the Bethe-free-energy becomes convex (every spanning-tree distribution gives `ρ_e = E[T ∋ e]` and the result is convex by Jensen) ~120 LOC; B18 `prob/mrf/expectation_propagation.go` Minka-2001-MIT-PhD expectation-propagation: one-pass-BP-on-Gaussian-EF where each factor is approximated by a Gaussian and product-of-Gaussians is exact — the canonical continuous-EF BP variant; in the discrete-case EP reduces to loopy-BP ~100 LOC), **(e) Tier-4 continuous-state BP ~380 LOC NEW** (B19 `prob/mrf/particle_bp.go` Sudderth-Ihler-Freeman-Willsky-2003 *CVPR-2003* / Ihler-McAllester-2009 *AISTATS-2009* particle-BP: messages are weighted-particle-clouds rather than discrete-tables; resampling at each iteration; works on continuous state-spaces such as articulated-pose-tracking, hand-tracking, sensor-fusion ~160 LOC; B20 `prob/mrf/nonparametric_bp.go` non-parametric-BP / Sudderth-2003 NBP using Gaussian-mixture-models rather than particles — closed-form message-multiplication via Gaussian-mixture-product (combinatorial-explosion, mitigated by KD-tree-pruning) ~140 LOC; B21 `prob/mrf/stochastic_bp.go` Noorshams-Wainwright-2013 stochastic-BP / Mooij-Kappen-2007 randomised message-update with mini-batch sampling for huge factor-graphs where full-iteration is infeasible ~80 LOC), **(f) Tier-5 BP for combinatorial optimisation ~220 LOC NEW** (B22 `prob/mrf/survey_propagation.go` Mezard-Parisi-Zecchina-2002 *Science-297:812* survey-propagation for random-3-SAT — generalises BP to *survey* messages = distributions-over-warning-messages, solves random-3-SAT at clause-density α ∈ [4.0, 4.27] where DPLL/WalkSAT/standard-BP all fail ~140 LOC; B23 `prob/mrf/bp_matching.go` Bayati-Shah-Sharma-2008 *IT-54:1241* BP for max-weight-matching (correct on bipartite-graphs after polynomial iterations); cross-link to graph/matching ~80 LOC), **(g) Tier-6 BP-LDPC bridge ~80 LOC NEW** (B24 `coding/ldpc/bp_decoder.go` thin-wrapper over 255-M5 sum-product on Tanner-graph: LDPC-codeword-decoding = sum-product-BP on `H·c = 0` parity-check factor-graph in the GF(2)-LLR-domain. Implements `BPDecode(received []float64, H *coding.SparseGF2Matrix, maxIter int) (decoded []byte, parity []bool, iters int)` plus `MinSumDecode` (max-product variant — the production-default in 5G-NR LDPC because it avoids `tanh/atanh` non-linearities) plus `OffsetMinSumDecode` (Chen-Fossorier-1999 offset-correction). 80 LOC because the wrapping-bridge code is small; the substrate `255-M5 sum-product + 255-M6 max-product + 210-Tanner-graph + 210-GF(2)-BitMatrix` ships from 210/255). **B25-B27 BP-LP-relaxation extensions ~280 LOC NEW** (B25 `prob/mrf/lp_relax_bp.go` Wainwright-Jaakkola-Willsky-2005-IT-51:3697 BP-as-LP-relaxation on local-marginal-polytope: for tree-structured graphs LP-relaxation is tight; for general graphs LP-relaxation is the dual of TRW-S; documents the BP→LP relationship ~80 LOC; B26 `prob/mrf/dual_decomposition.go` Komodakis-Paragios-Tziritas-2007-ICCV / Sontag-Globerson-Jaakkola-2011 dual-decomposition: decompose factor-graph into tractable sub-problems (trees, matchings, max-flows), solve each independently, enforce agreement via Lagrangian-multiplier-message-passing; SUBSUMES TRW + MPLP as instances ~140 LOC; B27 `prob/mrf/cavity_method.go` Mezard-Parisi-1986 / Mezard-Montanari-2009 cavity-method: derive BP fixed-point equations from the Bethe-Peierls cluster-approximation in statistical physics; documents the BP↔cavity-method bijection that connects PGM-inference to spin-glass-physics ~60 LOC).

**SINGULAR-FOUNDATIONAL B12 VariableElimination + B14 MiniBucket + B15 Treewidth ~360 LOC** — variable-elimination is the SIMPLEST exact-inference algorithm (Pearl himself uses it pedagogically before introducing BP); mini-bucket-Dechter-Rish-2003 is the SINGLE algorithm in this taxonomy that produces a certified-bound on log-Z with controllable cost; treewidth-Bodlaender-2006 is the COMPLEXITY-PARAMETER that determines BP's exact-tractability — without these three, the BP API has no theoretical anchor for `when does BP work?` (treewidth ≤ 2 → exact-on-tree; treewidth = ∞ → loopy → may not converge). **SINGULAR-CHEAPEST-1-DAY B11 ConvergenceCheck + B10 Damping + B8 Schedule ~220 LOC** — these three primitives are pure-API-utility wrapping 255-M5/M6/M8 with no NEW math; ship in one engineer-day against the 255-PR-B substrate. **SINGULAR-MOAT B14 MiniBucket + B19 ParticleBP + B22 SurveyPropagation ~440 LOC** — three BP variants that 254/255 do NOT enumerate and that have NO public Go implementation: mini-bucket is the production-default in libdai / daoopt for log-Z bounds; particle-BP powers articulated-pose-tracking + sensor-fusion; survey-propagation is the canonical algorithm for random-k-SAT at clause-density beyond DPLL's reach. **SINGULAR-2024-FRONTIER B16 ConvergentMP + B18 ExpectationPropagation + B19 ParticleBP ~400 LOC** — convergent-MP-CCCP-Heskes-2006 is the production-quality "loopy-BP-but-guaranteed-to-converge" variant; expectation-propagation is the foundation of Stan's continuous-EF approximate-inference and the Gaussian-process-classification SOTA (Rasmussen-Williams-2006 §3.6); particle-BP is the modern continuous-state-BP. **SINGULAR-PEDAGOGICAL B12 VariableElimination + B14 MiniBucket + B27 CavityMethod ~320 LOC** — VE is the textbook entry-point (Koller-Friedman-2009 §9; Russell-Norvig-2020 §13.4); mini-bucket is the controllable-approximation-bound-bridge between exact-VE and message-passing-BP; cavity-method is the *physics* origin-story that derives BP from first-principles via Bethe-Peierls cluster approximation (Mezard-Montanari-2009 *Information, Physics, Computation* the canonical pedagogical text). **SINGULAR-CROSS-LINK B24 BPLDPCBridge ~80 LOC** — the SINGLE most-deployed BP-instance worldwide (every WiFi-6/5G-NR/satellite-comm-receiver runs sum-product-BP on a Tanner-graph at line-rate); cross-link to 210 (already enumerates `coding/ldpc/`); ships as 80-LOC bridge once 210-T1-T5 GF(2)-BitMatrix-and-SparseGF2Matrix substrate + 255-M5 sum-product engine land. Five R-MUTUAL-CROSS-VALIDATION 3/3 pins enabled (see §3). Recommended placement **EXTEND existing `prob/mrf/` (255-PR-B target sub-package)** rather than NEW package — same precedent as 254-extends-254 (graph/cuts/) and 255-extends-prob/. Strict-downstream of 255-PR-A (Factor/FactorGraph) + 255-PR-B (sum-product / max-product / loopy-BP); strict-upstream of 210-T17/T19 LDPC sum-product/min-sum decoders, 215-T2/T5 AMP / GAMP for compressed-sensing, and a future `solver/sat/` survey-propagation-3-SAT-solver.

---

## 0. State at HEAD (2026-05-09, v0.10.0)

Repo-wide audit for belief-propagation / message-passing / factor-graph / variable-elimination / bucket-elimination / convergent-MP / particle-BP / non-parametric-BP / survey-propagation / cavity-method / convex-BP / expectation-propagation surface.

| Surface | Path | BP relevance |
|---|---|---|
| `prob.MarkovSteadyState` | `prob/markov.go:31-70` | 1-D state-only Markov-chain power-iteration; **PRESENT** but not BP (no factor-graph) |
| `prob.MarkovSimulate` | `prob/markov.go:99-139` | 1-D state-only sampling; **PRESENT** but not particle-BP |
| `prob.NormalPDF / Bernoulli / Beta / Gamma` | `prob/distributions.go` | Substrate for emission distributions in chain-CRF / particle-BP; **PRESENT** |
| `LogSumExp / Log1mExp / Log1pExp` | `changepoint/bocpd.go` (private) | Log-space toolbelt for ALL BP message-products; **PRESENT but private** (165 + 117 + 255 flag must promote to `prob/mathutil.go`) |
| `optim.LBFGS / SimulatedAnnealing` | `optim/*.go` | CRF parameter-learning + MAP-MRF substrate; **PRESENT** |
| `optim/proximal.{Admm, Fbs}` | `optim/proximal/*.go` | AD³ + dual-decomposition substrate; **PRESENT** |
| `graph.{Dijkstra, AStar, BellmanFord, BFS}` | `graph/*.go` | Substrate for B12 VE + B23 BP-matching; **PRESENT** |
| `graph.MaxFlow` | `graph/flow.go` | Substrate for B23 BP-matching cross-validation; **PRESENT** but EK |
| `coding/galois/*` (GF(2)/GF(2^m) substrate) | -- | **ABSENT** (210-T1-T4 enumerates) — gates B24 LDPC-BP bridge |
| `coding/ldpc/*` | -- | **ABSENT** (210-T17 enumerates 640 LOC) — co-target with 256-B24 |
| `prob/mrf/factor.go` (255-M1) | -- | **ABSENT** — 255-PR-A target |
| `prob/mrf/sum_product.go` (255-M5) | -- | **ABSENT** — 255-PR-B target |
| `prob/mrf/max_product.go` (255-M6) | -- | **ABSENT** — 255-PR-B target |
| `prob/mrf/junction_tree.go` (255-M7 / 254-C22) | -- | **ABSENT** — 255-PR-B / 254-PR-E target |
| `prob/mrf/loopy_bp.go` (255-M8 / 254-C21) | -- | **ABSENT** — 255-PR-B / 254-PR-E target |
| `prob/mrf/trw.go` (255-M9 / 254-C20) | -- | **ABSENT** — 255-PR-B / 254-PR-E target |
| `prob/mrf/mean_field.go` (255-M10 / 254-C29) | -- | **ABSENT** — 255-PR-C / 254-PR-E target |
| `prob/mrf/kikuchi.go` (255-M11) | -- | **ABSENT** — 255-PR-C target |
| B8-B27 BP-deep-dive primitives | -- | **ALL ABSENT** — this slot enumerates |

**False-positive name-collisions audited:**
- `acoustics/acoustics.go` — sound-*propagation* / wave-equation, NOT belief-propagation. Different physics, no shared math.
- `autodiff/doc.go` — "back*propagation*" = reverse-mode automatic differentiation on differentiable computations (chain-rule via topological sort), NOT message-passing on factor-graphs (chain-rule via marginalization). Word-collision only; both algorithms are message-passing-on-DAG but the messages and operators differ (gradients-vs-marginals).
- `physics/materials.go::ParisErdogan` — crack-*propagation* fatigue law `da/dN = C·(ΔK)^m`, NOT BP.
- `graph/centrality.go::Brandes-betweenness` — "back-*propagation* of dependencies" comment refers to Brandes-2001 dependency-accumulation in the BFS-based betweenness algorithm (a stack-pop traversal in reverse-BFS-order), NOT factor-graph BP.
- `timeseries/dcc/doc.go::DCC` — volatility-*propagation* / shock-*propagation* in Engle-2002 dynamic-conditional-correlation models; 1-D-time-series math, no factor-graph.
- `signal/window.go::Hamming-window` — spectral-window for FFT, NOT Hamming-code-BP / LDPC-decoding. Word-collision only.
- `chaos/systems.go::Ising` — 1-D-spin-glass / mean-field-thermodynamics dynamical system, NOT 2-D-Ising-MRF-BP-inference.

**Cross-import edges that this slot creates (additive on 254/255):**
- `prob/mrf/{schedule, residual_bp, damping, convergence_check} → prob/mrf.{Factor, FactorGraph, sumProduct, maxProduct, loopyBP}` (255-M1/M5/M6/M8) — internal API extension.
- `prob/mrf/{variable_elimination, bucket_elimination, mini_bucket, treewidth} → prob/mrf.Factor + graph.IntAdjacency + sort` — exact-inference layer; depends on B15 treewidth which uses `graph` for elimination-order heuristics.
- `prob/mrf/convergent_mp → optim/proximal.Fbs` for the CCCP inner-FBS-loop solving the convex-sub-problem of Heskes-2006.
- `prob/mrf/expectation_propagation → prob.NormalPDF + linalg.{Cholesky, MatVec}` for Gaussian-EF closed-form message-products.
- `prob/mrf/particle_bp → prob.NormalPDF + a future prob/random.go PRNG` (117-flagged) — particle-resampling needs unbiased uniform-draws.
- `prob/mrf/nonparametric_bp → prob.GaussianMixture + linalg.KDTree` (097-flagged) for high-dimensional Gaussian-product pruning.
- `prob/mrf/stochastic_bp → a future prob/random.go PRNG` (117-flagged).
- `prob/mrf/survey_propagation → prob.Bernoulli` for survey-message-distributions over warning-messages.
- `prob/mrf/bp_matching → graph.{MaxFlow, BipartiteMatching}` for cross-validation.
- `coding/ldpc/bp_decoder → coding/galois.{BitVector, SparseGF2Matrix} + prob/mrf.{sumProduct, maxProduct}` — the BP-LDPC bridge.

**Strict downstream consumers of slot-256 BP-deep-dive surface:**
- `coding/ldpc/bp_decoder.go` (210-T17) — sum-product / min-sum / offset-min-sum decoders consume slot-256-B24 bridge.
- `coding/polar/sc_decoder.go` (210-T19) — successive-cancellation polar-decoding is BP on a butterfly factor-graph; consumes 255-M5 + slot-256 scheduling theory for pipelined decode.
- A future `solver/sat/` survey-propagation random-3-SAT solver consumes slot-256-B22.
- A future `cv/tracking/articulated_pose.go` particle-BP consumer for articulated body-tracking consumes slot-256-B19.
- A future `solver/cs/amp.go` Approximate-Message-Passing for compressed-sensing (215-T-AMP) consumes slot-256-B27 cavity-method derivation as the AMP-correctness-anchor.
- 252-S14 BP-segmentation alternative: when factors are non-submodular (graph-cut fails), use slot-256-B16 convergent-MP for the MAP-fallback.

---

## 1. The twenty-four primitives (B8-B27 — additive on 254 / 255 / 210)

This slot does NOT re-enumerate the 13 algorithms shared with 254 (5: TRW-S, loopy-BP, junction-tree, MPLP, mean-field) and 255 (8: sum-product, max-product, junction-tree, loopy-BP, TRW, mean-field, Kikuchi-GBP, MPLP). Those ship via 254-PR-E / 255-PR-B / 255-PR-C as enumerated. Slot 256 enumerates the BP-DEEP-DIVE additive surface B8-B27 (the B1-B7 indices reserved as cross-references to the 254/255-shared algorithms).

### Tier 0 — BP foundations cross-references (~0 NEW LOC)

- **B1 → 255-M1** Factor + FactorGraph + Variable types (~140 LOC, 255-PR-A).
- **B2 → 255-M5** sum-product on factor-graph for marginals (~140 LOC, 255-PR-B).
- **B3 → 255-M6** max-product / Viterbi-on-factor-graph for MAP (~120 LOC, 255-PR-B).
- **B4 → 255-M7 / 254-C22** junction-tree exact-inference (~180 LOC, 255-PR-B / 254-PR-E shared).
- **B5 → 255-M8 / 254-C21** loopy-BP sum-product + max-product variants (~140 LOC, 255-PR-B / 254-PR-E shared).
- **B6 → 255-M9 / 254-C20** parallel-TRW + sequential TRW-S (~140 LOC + ~280 LOC = ~420 LOC, 255-PR-B / 254-PR-E shared).
- **B7 → 255-M10 / 254-C29** mean-field VI + Bethe-free-energy (~140 LOC, 255-PR-C / 254-PR-E shared).

### Tier 1 — BP scheduling theory (~360 NEW LOC)

**B8 — `prob/mrf/schedule.go` ~80 LOC.** Schedule taxonomy + dispatcher.

```go
type Schedule int
const (
    Synchronous   Schedule = iota // flooding: all messages updated in parallel each iteration (Pearl-1988 default)
    Asynchronous                  // Gauss-Seidel: messages updated in-place in some fixed order
    RoundRobin                    // deterministic edge-order (lex-sorted by source-id)
    RandomOrder                   // uniformly-random edge-order each iteration
    Residual                      // priority-queue keyed by message-residual (B9)
)

type ScheduleOpts struct {
    Type      Schedule
    Damping   float64 // [0, 1], 0.5 default for loopy-BP
    Tol       float64 // convergence on max message-change
    MaxIter   int
    Seed      int64   // for RandomOrder / stochastic
}

func RunBP(g *FactorGraph, opts ScheduleOpts) (marginals [][]float64, logZ float64, iters int)
```

**Refs.** Pearl-1988; Wainwright-Jordan-2008 §2.5; Murphy-2012 *Machine Learning: a Probabilistic Perspective* §22.4.

**B9 — `prob/mrf/residual_bp.go` ~140 LOC — KEYSTONE-SCHEDULING.** Elidan-McGraw-Koller-2006 *UAI-2006* "Residual Belief Propagation: Informed Scheduling for Asynchronous Message Passing". Maintain a max-heap of edge-residuals `r_e = ‖μ_new(e) − μ_prev(e)‖_∞`; pop edge with largest residual, recompute its message, push affected downstream-edges back into the heap with their updated residuals. Empirical observation: 3-10× fewer message-updates to convergence vs flooding (depends on graph topology — most beneficial on graphs with few high-information cycles). API:

```go
type ResidualBPState struct {
    Heap     *priorityQueue  // edge-id → residual
    Messages map[edgeID][]float64
    G        *FactorGraph
}

func ResidualBP(g *FactorGraph, tol float64, maxUpdates int) (marginals [][]float64, logZ float64)
```

**Refs.** Elidan-McGraw-Koller-2006 *UAI-2006*; Sutton-McCallum-2007-NIPS *Improved Dynamic Schedules for BP*.

**B10 — `prob/mrf/damping.go` ~80 LOC.** Murphy-2001 damping `μ_new := α·μ_computed + (1−α)·μ_prev`; Heskes-2003-NIPS fractional-BP `μ_new := μ_computed^α · μ_prev^(1−α)` log-domain version (geometric-mean instead of arithmetic-mean — preserves probabilistic semantics for log-space messages). Adaptive-damping per Pretti-2005 *Phys-Rev-E-71:066127* with armijo-line-search to reject damping-updates that increase Bethe-free-energy. **Refs.** Murphy-Weiss-Jordan-1999 *UAI*; Heskes-2003 *NIPS*; Pretti-2005 *PRE-71:066127*.

**B11 — `prob/mrf/convergence_check.go` ~60 LOC.** Multiple stopping-criteria:
- `MaxMsgResidual`: `max_e ‖μ_new(e) − μ_prev(e)‖_∞ < tol`
- `MarginalKL`: `Σ_v KL(b_new_v ‖ b_prev_v) < tol`
- `LogZStability`: `|log Z_new − log Z_prev| < tol`
- `MaxIterReached`: hard limit

Returns `(converged bool, reason string, finalResidual float64)` for diagnostics. **Refs.** Mooij-Kappen-2007 *Sufficient conditions for BP convergence on a graph with cycles* — IT-53:4422 derives the convergence-radius bound.

### Tier 2 — Variable / bucket elimination (~420 NEW LOC)

**B12 — `prob/mrf/variable_elimination.go` ~140 LOC.** Zhang-Poole-1996 *AAAI-1996* / Dechter-1996 *UAI-1996* variable-elimination. Algorithm: choose elimination ordering `π = (x_{σ(1)}, ..., x_{σ(n)})`; for each variable `x_i` in turn, multiply all factors mentioning `x_i`, sum (sum-product) or max (max-product) out `x_i`, replace those factors with the resulting reduced-factor. After all variables are eliminated, the remaining factor is the partition-function-Z (sum-product) or the MAP-value (max-product). API:

```go
func EliminationOrder(g *FactorGraph, heuristic OrderingHeuristic) []int
// MinFill, MinDegree, WeightedMinFill, MaxCardinality

func VariableElimination(g *FactorGraph, query []int, evidence map[int]int, ord []int) (logMarg []float64)
func VariableEliminationMAP(g *FactorGraph, evidence map[int]int, ord []int) (assignment []int, logScore float64)
```

The `MinFill` heuristic adds the variable that creates the smallest induced-clique in the elimination-graph (Kjaerulff-1990 PhD analysis); reliably within 2× of optimal-treewidth in benchmarks. Cost: `O(N·d^{w*+1})` where `w*` = induced-treewidth of the ordering. **Refs.** Zhang-Poole-1996 *AAAI*; Dechter-1996 *UAI*; Koller-Friedman-2009 §9.

**B13 — `prob/mrf/bucket_elimination.go` ~120 LOC.** Dechter-1999 *Artif-Intell-113:41* bucket-elimination unified-framework. Each variable's *bucket* is a set of factors plus the elimination-operator (`Σ`, `max`, `argmax`, ∃, ∀ — different operators give different inference tasks). The bucket-tree is the dependency-DAG of buckets along the elimination-ordering. A single bucket-process schedule subsumes:
- `Σ`-bucket → variable-elimination for marginals
- `max`-bucket → MAP-VE (Viterbi)
- `argmax`-bucket → MAP-trace (recover argmax-x)
- `∃`-bucket → propositional satisfiability (Davis-Putnam)
- `count`-bucket → model-counting

API:

```go
type BucketOp int
const (SumOp BucketOp = iota; MaxOp; ArgMaxOp; ExistsOp; CountOp)

func BucketElimination(g *FactorGraph, ord []int, op BucketOp) interface{}
```

**Refs.** Dechter-1999 *Artif-Intell-113:41*; Dechter-2003 *Constraint Processing* §8.

**B14 — `prob/mrf/mini_bucket.go` ~140 LOC — SINGULAR-MOAT.** Dechter-Rish-2003 *J-ACM-50:107* mini-bucket-elimination. Partition each bucket into mini-buckets of size ≤ `i` (the i-bound); approximate the elimination by independently eliminating each mini-bucket. With `max` aggregation across mini-buckets within a bucket, gives `log Z ≤ MBE_i(g)` upper-bound on log-partition-Z. With `mean`-bucket replacement, gives lower-bound. With `i = treewidth`, recovers exact-VE. With `i = 1`, reduces to a fast-but-loose bound. The SINGLE algorithm in BP that produces a *certified bound* on log-Z with controllable cost. Powers the `daoopt` / `mexico-City-ICS-mini-bucket-toolkit` / `libdai`'s WeightedMiniBucketElimination — the production-default bound for partition-function in the UAI Inference Competition 2008-2014. **Refs.** Dechter-Rish-2003 *J-ACM-50:107*; Liu-Ihler-2011 *NeurIPS-2011* weighted-mini-bucket; Dechter-2019 *Reasoning with Probabilistic and Deterministic Graphical Models* §14.

**B15 — `prob/mrf/treewidth.go` ~80 LOC.** Bodlaender-Koster-2010 *Inf-Comput-208:259* treewidth-estimation upper- and lower-bounds:
- Upper: `MinFill`, `MinDegree`, `MaxCardinality`, `WeightedMinFill` heuristics → induced-treewidth of the elimination-ordering.
- Lower: `MMD` (maximum-minimum-degree, Lucena-2003), `MMD+` (Bodlaender-Koster-2010), `Ramachandramurthi-1997` linear-time lower-bound.

API:

```go
func TreewidthLower(g Graph) int       // valid lower bound
func TreewidthUpperFill(g Graph) (int, []int)  // induced-treewidth + ordering, MinFill heuristic
func TreewidthExact(g Graph, deadline time.Duration) (int, error)  // Bodlaender-1996 O(2^O(w³)·n) — only for w ≤ 5
```

Treewidth IS the complexity-parameter that determines whether BP is exact-tractable (`O(N·d^{w+1})`) or NP-hard (general graphs, treewidth = N). Without B15 the BP API has no way to report `won't-be-tractable` ahead of time. **Refs.** Bodlaender-1996 *J-Algorithms-21:358*; Bodlaender-Koster-2010 *Inf-Comput-208:259*; Arnborg-Corneil-Proskurowski-1987 *SIAM-J-Alg-8:277* original treewidth-NP-hardness.

### Tier 3 — Convergent / convex BP (~360 NEW LOC)

**B16 — `prob/mrf/convergent_mp.go` ~140 LOC — KEYSTONE-CONVERGENCE.** Heskes-2006 *NeurIPS-2006* "Convexity arguments for efficient minimization of the Bethe and Kikuchi free energies" + Yuille-2002 *Neural-Comput-14:1691* CCCP (Concave-Convex-Procedure). Loopy-BP fixed-points = stationary-points of Bethe-free-energy `F_Bethe(b) = U(b) − H_Bethe(b)`; loopy-BP itself is *not* guaranteed to converge because its iteration is not a contraction. CCCP decomposes `F_Bethe = F_concave + F_convex` and alternates between solving the convex sub-problem (closed-form) and a concave-update (gradient-step). Guaranteed monotone-decreasing in `F_Bethe` → guaranteed-convergence to a stationary-point. ~3-5× slower per iteration than vanilla-loopy-BP but converges where loopy-BP oscillates. The production-default in `libdai`'s `BP_DUAL` and `Heskes_AsymmetricBP`. **Refs.** Heskes-2006 *NeurIPS-2006*; Yuille-2002 *Neural-Comput-14:1691*; Welling-Teh-2003-UAI Belief-Optimisation.

**B17 — `prob/mrf/convex_bp.go` ~120 LOC.** Wainwright-Jaakkola-Willsky-2005 *IT-51:2697* "A new class of upper bounds on the log partition function" / Hazan-Shashua-2010 *IT-56:6294* convex-BP / fractional-BP. Choose edge-counting-numbers `ρ_e ∈ [0, 1]` with `Σ_e ρ_e ≥ |V|−1` (spanning-tree-cover condition); the resulting weighted-Bethe-free-energy is convex in beliefs and has a unique-minimiser. With `ρ_e = E[T ∋ e]` averaged over a distribution of spanning trees, recovers TRW (Wainwright-Jaakkola-Willsky-2005). With `ρ_e = 1` recovers vanilla-Bethe (non-convex). With smaller `ρ_e` the bound is looser but the optimisation easier. **Refs.** Wainwright-Jaakkola-Willsky-2005 *IT-51:2697*; Hazan-Shashua-2010 *IT-56:6294*; Meshi-Jaakkola-Globerson-2012-ICML convergence-rate-of-convex-BP.

**B18 — `prob/mrf/expectation_propagation.go` ~100 LOC — KEYSTONE-CONTINUOUS.** Minka-2001 *MIT-PhD-Thesis* "A family of algorithms for approximate Bayesian inference" expectation-propagation. EP approximates an intractable posterior `p(θ|D) ∝ ∏_n f_n(θ)` by a product of exponential-family-Gaussian factors `q(θ|D) ∝ ∏_n f̃_n(θ)`, iteratively refining each `f̃_n` to match the moments of the cavity-distribution `q\_n · f_n`. In the discrete-case EP reduces to loopy-BP. In the Gaussian-EF-case EP gives Gaussian-process-classification SOTA (Rasmussen-Williams-2006 §3.6 reports EP > Laplace > MCMC on UCI benchmarks at fixed compute). The continuous-EF generalisation of BP and the foundation of Stan's automatic-differentiation-variational-inference (ADVI). **Refs.** Minka-2001 *MIT-PhD*; Minka-2005 *MSR-TR-2005-173* divergence-measure-overview; Rasmussen-Williams-2006 §3.6.

### Tier 4 — Continuous-state BP (~380 NEW LOC)

**B19 — `prob/mrf/particle_bp.go` ~160 LOC — KEYSTONE-CONTINUOUS.** Sudderth-Ihler-Freeman-Willsky-2003 *CVPR-2003* / Ihler-McAllester-2009 *AISTATS-2009* particle-BP. Each message `μ_{f→v}(x_v)` is represented by a weighted-particle-cloud `{(x_v^{(k)}, w^{(k)})}_{k=1}^K` rather than a discrete-table or a Gaussian. Message product: weighted-particle-multiplication (combinatorial-explosion, mitigated by importance-resampling). Message sum (for sum-product): Monte-Carlo-integration over the particles. Resampling at each iteration keeps `K` constant. Application: articulated-pose-tracking (Sigal-Bhatia-Roth-Black-2004), hand-tracking, sensor-fusion, simultaneous-localization-and-mapping. **Refs.** Sudderth-Ihler-Freeman-Willsky-2003 *CVPR*; Ihler-McAllester-2009 *AISTATS*; Sigal-Bhatia-Roth-Black-2004 *CVPR* loose-limbed-people.

**B20 — `prob/mrf/nonparametric_bp.go` ~140 LOC.** Sudderth-2003 *MIT-MS-Thesis* / Sudderth-Ihler-Isard-Freeman-Willsky-2010 *Comm-ACM-53:95* non-parametric-BP using Gaussian-mixture-models rather than particles. Each message is a GMM with `K` components; message-product is exact (product-of-GMMs is a GMM with `K^|N(f)|` components, then prune to `K` via clustering or particle-resampling on the resulting GMM); message-sum is closed-form Gaussian-marginalization per component. Compared to particle-BP: smoother messages, faster convergence on smooth-state-spaces, but combinatorial-blowup if not pruned aggressively. **Refs.** Sudderth-2003 *MIT-MS*; Sudderth-Ihler-Isard-Freeman-Willsky-2010 *Comm-ACM-53:95*.

**B21 — `prob/mrf/stochastic_bp.go` ~80 LOC.** Noorshams-Wainwright-2013 *IT-59:1981* "Stochastic Belief Propagation" / Mooij-Kappen-2007 *NeurIPS-2007* sub-sampled-BP. Instead of computing the exact message at each step, sample a *subset* of factor-marginalization terms and use the resulting Monte-Carlo estimate as the message-update; over iterations the noise averages out. For huge-`d` factor-graphs (hundreds of states per variable) where exact `O(d^|N(f)|)` factor-elimination is intractable. Provably-convergent on tree-structured graphs (Noorshams-Wainwright-2013); empirically convergent on loopy-BP graphs at the cost of slower convergence rate. **Refs.** Noorshams-Wainwright-2013 *IT-59:1981*; Mooij-Kappen-2007 *NeurIPS*.

### Tier 5 — BP for combinatorial optimisation (~220 NEW LOC)

**B22 — `prob/mrf/survey_propagation.go` ~140 LOC — SINGULAR-MOAT.** Mezard-Parisi-Zecchina-2002 *Science-297:812* "Analytic and algorithmic solution of random satisfiability problems" survey-propagation. For random-3-SAT at clause-density `α = M/N` near the SAT/UNSAT-threshold (`α_c ≈ 4.267` for k=3), standard BP and DPLL-and-WalkSAT all fail — the energy-landscape is glassy with exponentially-many clusters of solutions. Survey-propagation replaces BP-messages (single-warning) with *survey*-messages (probability-distribution over warnings), corresponding to the 1-step replica-symmetry-breaking (1-RSB) cavity-method-Mezard-Montanari-2009 ansatz from spin-glass-physics. Solves random-3-SAT at `α ∈ [4.0, 4.27]` — the only known polynomial-time algorithm in this regime. Generalises to k-colouring and other CSPs. **Refs.** Mezard-Parisi-Zecchina-2002 *Science-297:812*; Braunstein-Mezard-Zecchina-2005 *RSA-27:201*; Mezard-Montanari-2009 *Information, Physics, Computation* §22.

**B23 — `prob/mrf/bp_matching.go` ~80 LOC.** Bayati-Shah-Sharma-2008 *IT-54:1241* "Max-product for maximum weight matching" / Sanghavi-Malioutov-Willsky-2011 *IT-57:7269*. On a bipartite graph, max-product-BP iterated `O(N·log(N/ε))` times converges to the optimal max-weight-matching (provided the optimum is unique). Provides a distributed-/parallel-friendly alternative to Hungarian / auction algorithms. Cross-validation oracle: `graph.MaxFlow` reduction-of-bipartite-matching. **Refs.** Bayati-Shah-Sharma-2008 *IT-54:1241*; Sanghavi-Malioutov-Willsky-2011 *IT-57:7269*; Salez-Shah-2009 *IEEE-J-Sel-Areas-Comm-27:1115* BP-for-stochastic-matching.

### Tier 6 — BP-LDPC bridge (~80 NEW LOC)

**B24 — `coding/ldpc/bp_decoder.go` ~80 LOC — KEYSTONE-CROSS-LINK-TO-210.** Sum-product / min-sum / offset-min-sum LDPC decoders as thin-wrappers over 255-M5 / 255-M6 on the Tanner-graph factor-graph. Tanner-graph: variable-nodes = bits, factor-nodes = parity-check-equations; the factor at parity-check-row `i` equals `1` iff `Σ_j H_{ij}·x_j ≡ 0 (mod 2)`. Inputs = received-channel-LLRs; messages = bit-LLRs. API:

```go
func BPDecode(receivedLLRs []float64, H *coding.SparseGF2Matrix, maxIter int) (decoded []byte, success bool, iters int)
func MinSumDecode(receivedLLRs []float64, H *coding.SparseGF2Matrix, maxIter int) (decoded []byte, success bool, iters int)
func OffsetMinSumDecode(receivedLLRs []float64, H *coding.SparseGF2Matrix, offset float64, maxIter int) (decoded []byte, success bool, iters int)
func NormalizedMinSumDecode(receivedLLRs []float64, H *coding.SparseGF2Matrix, scale float64, maxIter int) (decoded []byte, success bool, iters int)
```

Min-sum is the production-default in 5G-NR LDPC because it avoids `tanh/atanh` non-linearities (Chen-Fossorier-1999 reports ≤0.5dB performance-gap to sum-product at far-lower compute). Offset-min-sum (Chen-Fossorier-1999) and normalized-min-sum (Chen-Lee-Pottie-2002) reduce the gap to ≤0.1dB. The 5G-NR base-graphs BG1 (rate ½) and BG2 (rate ⅓) are specified in 3GPP-TS-38.212; the Wi-Fi-6 802.11ax LDPC codes are specified in IEEE-802.11ax-D8.0. **Refs.** Gallager-1963 *MIT-PhD* original-LDPC-BP; MacKay-Neal-1996 *Electron-Lett-32:1645* LDPC-rediscovery; Richardson-Urbanke-2008 *Modern Coding Theory* density-evolution; Chen-Fossorier-1999 *Comm-Lett-3:165* offset-min-sum; 3GPP-TS-38.212 §5.3 5G-NR-LDPC.

### Tier 7 — BP-LP-relaxation extensions (~280 NEW LOC)

**B25 — `prob/mrf/lp_relax_bp.go` ~80 LOC.** Wainwright-Jaakkola-Willsky-2005 *IT-51:3697* / Sontag-2010 *MIT-PhD-Thesis* "Approximate Inference in Graphical Models using LP Relaxations". LP-relaxation of MAP-MRF over the local-marginal-polytope `M_local`: maximise `Σ_α θ_α^T μ_α` subject to `μ_α ∈ M_local` (consistency on shared variables, non-negativity, normalization). For tree-structured factor-graphs, `M_local = M(G)` (the marginal-polytope) and LP-relaxation is tight. For general graphs, `M_local ⊃ M(G)` and LP-relaxation gives an upper-bound on MAP-energy. Sontag-Globerson-Jaakkola-2008-NIPS "Tightening LP Relaxations for MAP using Message Passing" ships the cycle-tightening protocol. Documents BP↔LP relationship: max-product-BP fixed-points correspond to LP-relaxation vertices. **Refs.** Wainwright-Jaakkola-Willsky-2005 *IT-51:3697*; Sontag-Globerson-Jaakkola-2008 *NIPS*; Sontag-2010 *MIT-PhD*.

**B26 — `prob/mrf/dual_decomposition.go` ~140 LOC.** Komodakis-Paragios-Tziritas-2007 *ICCV-2007* / Sontag-Globerson-Jaakkola-2011 *Optim-for-Mach-Learning* dual-decomposition. Decompose factor-graph into tractable sub-problems `G = G_1 ∪ G_2 ∪ ... ∪ G_K` (e.g., spanning-trees, matchings, max-flows). Solve each sub-problem `G_k` independently (each has a tractable inference algorithm). Enforce agreement on shared variables via Lagrangian multipliers `λ_k`, updated via projected-subgradient or smoothed-dual-coordinate-ascent. SUBSUMES TRW (decompose into spanning-trees) and MPLP (decompose into pairs of factors) as special-instances. The unified-framework that connects discrete-MAP-inference to convex-optimisation. **Refs.** Komodakis-Paragios-Tziritas-2007 *ICCV*; Sontag-Globerson-Jaakkola-2011 *Optim-for-Mach-Learning* §1; Bertsekas-1999 *Nonlinear Programming* §6.4 dual-decomposition-classical.

**B27 — `prob/mrf/cavity_method.go` ~60 LOC.** Mezard-Parisi-1986 *Europhysics-Lett-1:73* / Mezard-Montanari-2009 *Information, Physics, Computation* §14-22. Derives BP fixed-point equations from the Bethe-Peierls cluster-approximation of the Gibbs-distribution: condition on a *cavity* (remove a variable plus its neighbouring factors), compute the marginal of the cavity-graph, then re-add the variable. Equivalently: BP messages = single-occurrence-warnings of the cavity-method. Connects PGM-inference to spin-glass-physics: the 1-RSB cavity-method gives survey-propagation (B22); the 0-RSB cavity-method gives standard BP. Documents the BP↔cavity-method bijection. Critical for understanding AMP (Approximate-Message-Passing-Donoho-Maleki-Montanari-2009-PNAS-106:18914) which is dense-factor-graph-BP with cavity-method-Onsager-correction-terms. **Refs.** Mezard-Parisi-1986 *EPL-1:73*; Mezard-Montanari-2009 *Information, Physics, Computation* §14-22; Donoho-Maleki-Montanari-2009 *PNAS-106:18914* AMP.

---

## 2. Connective tissue + cross-package blockers

**Substrate-blocker-1 (HARD)** `prob/mrf.{Factor, FactorGraph}` (255-M1) + `prob/mrf.sumProduct` (255-M5) + `prob/mrf.maxProduct` (255-M6) — **without 255-PR-A and 255-PR-B landed first, slot 256 has no engine to schedule.** B1-B7 are pure cross-references; B8-B11 (Tier-1 scheduling) wrap 255-M5/M6/M8 with NEW schedule-policy structs.

**Substrate-blocker-2 (HARD)** `prob.LogSumExp` / `Log1mExp` / `Log1pExp` — currently private to `changepoint/bocpd.go`. 165-PR + 117-PR + 255-PR all flag this promotion. Every BP message-product uses `LogSumExp`. Promotion is ~30 LOC + test migration.

**Substrate-blocker-3 (HARD)** `coding/galois.{BitVector, SparseGF2Matrix}` (210-T1-T4) + Tanner-graph constructor — **gates B24 LDPC-BP bridge**. 210-PR-A delivers ~240 LOC of GF(2) substrate; B24 ~80 LOC bridge ships against this.

**Substrate-blocker-4 (SOFT)** `linalg.SparseMatrix / KDTree` (097-flagged) — gates B20 non-parametric-BP at huge-state-space scale. B20 ships at quality-bar without sparse — Gaussian-mixture-product on small-K is dense-O(K^|N(f)|).

**Substrate-blocker-5 (SOFT)** A future `prob/random.go` PRNG (117-flagged) — gates B19 particle-BP, B21 stochastic-BP, B22 survey-propagation. The PRESENT `crypto/rng` PCG-64 is adequate as a fallback (already used by `optim/metaheuristic`).

**Substrate-blocker-6 (SOFT)** `optim/proximal.{Fbs, Admm}` (PRESENT) — used by B16 convergent-MP-CCCP inner-FBS-loop and B26 dual-decomposition. ZERO BLOCKERS.

**Substrate-blocker-7 (NONE)** `optim.LBFGS` (PRESENT) — used by B17 convex-BP edge-weight optimization, B18 EP moment-matching. ZERO BLOCKERS.

**Total NEW upstream-substrate dependency** for slot-256 (assuming 255-PR-A/PR-B and 165-PR-LogSumExp-promotion land first): 0 LOC NEW substrate. Slot-256 is purely additive on the 255-PR-B substrate plus a cross-link bridge to 210-PR-A (~80 LOC for B24).

**Cheapest-no-blocker subset** (assuming 255-PR-B substrate): **B8 Schedule + B11 ConvergenceCheck + B10 Damping ~220 LOC** — pure-API extension over 255-M5/M6/M8, no NEW math. Ships in 1 engineer-day.

**Recommended PR sequence:**

- **PR-G-256 (Tier-1 scheduling ~360 LOC, 1 week)** B8 Schedule + B9 ResidualBP + B10 Damping + B11 ConvergenceCheck. Wraps 255-PR-B sum-product/max-product/loopy-BP with priority-queue-driven async scheduling. Empirical 3-10× speedup on loopy-BP convergence.
- **PR-H-256 (Tier-2 elimination ~420 LOC, 1.5 weeks)** B12 VariableElimination + B13 BucketElimination + B14 MiniBucket + B15 Treewidth. Adds the orthogonal exact-inference-via-elimination tier (NOT message-passing) plus the only certified-bound on log-Z (mini-bucket).
- **PR-I-256 (Tier-3 convergent ~360 LOC, 1.5 weeks)** B16 ConvergentMP-CCCP + B17 ConvexBP + B18 EP. The convergence-guaranteed BP variants. EP is the foundation for continuous-EF inference (Stan-style ADVI).
- **PR-J-256 (Tier-4 continuous ~380 LOC, 1.5 weeks)** B19 ParticleBP + B20 NonParametricBP + B21 StochasticBP. Continuous-state-space BP for vision/sensor-fusion/pose-tracking applications.
- **PR-K-256 (Tier-5 combinatorial ~220 LOC, 1 week)** B22 SurveyPropagation + B23 BPMatching. SP solves random-3-SAT at clause-density beyond DPLL's reach; BP-matching is a max-product-BP textbook application.
- **PR-L-256 (Tier-6 LDPC bridge ~80 LOC, 0.5 week)** B24 BP-LDPC-decoder. Co-target with 210-PR-LDPC. Ships sum-product / min-sum / offset-min-sum / normalized-min-sum decoders.
- **PR-M-256 (Tier-7 LP/dual ~280 LOC, 1 week)** B25 LP-relax-BP + B26 DualDecomposition + B27 CavityMethod. Documents the BP↔LP and BP↔cavity-method bijections; foundation for future AMP-for-compressed-sensing (215-T-AMP).

Total slot-256 PR-G through PR-M: ~2,140 LOC NEW (additive on 254/255/210), ~7 engineer-weeks. Lighter than 254 (3,640 LOC) and 255 (2,940 LOC) because all of the BP-engine is delegated to 255 — slot 256 is the *deep-dive on the BP-specific operations* that 254/255 did not enumerate.

**De-duplication policy with 254 and 255:** all 13 algorithms shared (5 in 254-PR-E, 8 in 255-PR-B/PR-C) ship via the 254/255 PRs. Slot 256's B1-B7 are explicit *cross-references* with NO new code. Slot 256 ships ONLY the additive surface B8-B27 that the BP-deep-dive requires (scheduling, elimination, convergent-MP, continuous-state, combinatorial, LDPC-bridge, LP-relaxation/dual-decomposition).

---

## 3. Five R-MUTUAL-CROSS-VALIDATION 3/3 pins this slot enables

**Pin 1 — Variable-elimination ↔ junction-tree ↔ brute-force on a 5-node binary loop.** Three paths to identical marginals + log-Z on a 5-node binary loop (`x_1 — x_2 — x_3 — x_4 — x_5 — x_1`, all binary, 32 configs):
- B12 VariableElimination with min-fill ordering
- 255-M7 / 254-C22 JunctionTree (triangulate the 5-loop with one chord, treewidth = 2)
- Brute-force enumeration of 2^5 = 32 configurations

All three must agree on per-variable marginals to 1e-12 and on log-partition Z to 1e-12. Saturates 3/3 + a treewidth-correctness pin (B15.TreewidthUpperFill returns 2 on the 5-loop with one chord).

**Pin 2 — Mini-bucket bound ↔ exact-VE ↔ sample-mean on a 4×4 binary grid.** Three paths to a *bound* on log-Z on a 4×4 Ising MRF (16 binary variables, 2^16 = 65,536 configs, brute-force still tractable):
- B14 MiniBucket with i-bound = 4 → upper-bound `log Z ≤ MBE_4`
- B12 VariableElimination → exact `log Z`
- B13 BucketElimination with i-bound = 16 (no-truncation) → exact `log Z`
- Brute-force enumeration → exact `log Z` (oracle)

The MBE_4 result must satisfy `MBE_4 ≥ log Z_exact` strictly, with the gap bounded by the i-bound theory (Dechter-Rish-2003 Theorem 4). Saturates 3/3 + a mini-bucket-bound-correctness pin.

**Pin 3 — Residual-BP ↔ flooding-BP ↔ damped-BP on a 100-node grid loopy-BP convergence-iter-count.** Three paths to the same final marginals on a 10×10 grid Ising MRF with random ±1 couplings:
- B9 ResidualBP with priority-queue scheduling
- B8 Schedule(Synchronous) loopy-BP (255-M8) with damping = 0
- B10 damping = 0.5 + B8 Schedule(Asynchronous) loopy-BP (255-M8)

The final marginals must agree to 1e-8 (loopy-BP doesn't recover the exact marginals on loopy graphs but all three schedules converge to the *same* Bethe-stationary-point). Iteration-count comparison: B9-residual ≤ 0.5× × B8-async iter-count ≤ 0.5× × B8-sync-flooding iter-count (Elidan-McGraw-Koller-2006 empirical result). Saturates 3/3 + a scheduling-efficiency pin.

**Pin 4 — Convergent-MP-CCCP ↔ loopy-BP-with-damping ↔ exact-VE on a 3-loop where loopy-BP oscillates.** Three paths to the same MAP on a frustrated 3-cycle Ising-MRF where standard loopy-BP oscillates:
- B16 ConvergentMP-CCCP (guaranteed-monotone-convergence)
- B10 dampedBP with α = 0.9 (eventually converges, slowly)
- B12 VariableElimination (exact, oracle)

CCCP must converge in ≤ N_iter_max iterations; damped-loopy-BP must agree to 1e-6; exact-VE is the oracle. Saturates 3/3 + a convergence-guarantee pin (the 3-cycle frustrated Ising is a textbook case where vanilla-loopy-BP oscillates indefinitely; Heskes-2006 Fig.1).

**Pin 5 — Sum-product BP-LDPC ↔ min-sum BP-LDPC ↔ syndrome-decode on a (15, 11) Hamming code.** Three paths to the same decoded codeword on the (15, 11)-Hamming-code with single-bit-error received (encoded as a Tanner-graph with 15 variable-nodes, 4 parity-check factors):
- B24 BPDecode (sum-product on Tanner-graph LLRs)
- B24 MinSumDecode (max-product-min-sum approximation)
- 210-T5 SyndromeDecode (lookup-table on the (15, 11)-Hamming syndrome)

For single-bit-error Hamming-(15,11) is a perfect-code → all three must agree on the corrected-codeword bit-for-bit. The LLR-domain BP and the GF(2) syndrome-decode are operating on disjoint code-paths but must produce identical hard-decoded output. Saturates 3/3 + a BP-LDPC-correctness pin spanning 210 + slot-256.

---

## 4. Cross-link map: where slot-256 reaches outside `prob/mrf/`

| Slot | Surface | Cross-link role |
|---|---|---|
| 254 (graph-cuts) | C20 TRW-S, C21 loopy-BP, C22 junction-tree, C23 MPLP, C29 mean-field | 5 algorithms shared; 254 = energy-min-API, slot-256 = BP-deep-dive |
| 255 (MRF) | M1 Factor, M5 sum-product, M6 max-product, M7 junction-tree, M8 loopy-BP, M9 TRW, M10 mean-field, M11 Kikuchi, M12 MPLP | 8 algorithms shared; 255 = PGM-API, slot-256 = BP-scheduling/elimination/continuous-state extensions |
| 210 (coding-theory) | T1-T5 GF(2) substrate, T17 LDPC-decoders | B24 BP-LDPC bridge — sum-product + min-sum + offset-min-sum decoders ship as 80-LOC wrapper over 255-M5/M6 |
| 165 (synergy-sequence-prob) | HMM forward / backward / Viterbi | B19 documents HMM = chain-MRF + chain-CRF special-case of M5/M6; B18 EP generalises forward/backward to continuous-EF |
| 215 (compressed-sensing) | T-AMP / T-GAMP | B27 cavity-method documents AMP-Onsager-correction derivation; future `solver/cs/amp.go` consumes |
| 117 (prob-missing) | LogSumExp, PRNG promotion | Substrate-blocker for slot-256; ships in 165-PR / 117-PR before slot-256 lands |
| 097 (linalg-missing) | SparseMatrix, KDTree | Soft-blocker for B20 non-parametric-BP at large-K |
| A future `solver/sat/` | random-3-SAT solver | B22 survey-propagation is the polynomial-time algorithm at clause-density α ∈ [4.0, 4.27] |
| A future `cv/tracking/` | articulated-pose, hand-tracking | B19 particle-BP is the canonical algorithm |
| A future `solver/lp/` | LP-relaxation of MAP | B25 + B26 document the BP↔LP duality; downstream consumer |

---

## 5. Verdict

**Slot 256 is BP-DEEP-DIVE additive** on top of 254 (5 algos), 255 (8 algos), 210 (LDPC sub-package). The 13 BP algorithms in 254/255 cover the *core* message-passing surface; this slot enumerates the BP-SPECIFIC tier that those slots did NOT cover: **scheduling theory** (B8-B11 ~360 LOC), **variable / bucket / mini-bucket elimination** (B12-B15 ~420 LOC, the orthogonal exact-inference axis), **convergent / convex / EP** (B16-B18 ~360 LOC, the convergence-guaranteed BP variants), **continuous-state BP** (B19-B21 ~380 LOC, particle / non-parametric / stochastic), **combinatorial-optimisation BP** (B22-B23 ~220 LOC, survey-propagation + matching), **BP-LDPC bridge** (B24 ~80 LOC, the cross-link to 210), and **BP-LP / dual-decomposition / cavity-method** (B25-B27 ~280 LOC, the BP↔LP↔physics theoretical-anchor).

Total NEW surface: ~2,140 LOC in 7 PRs over 7 engineer-weeks, all strictly-additive on 254/255/210 substrate.

The SINGULAR-FOUNDATIONAL B12+B14+B15 ~360 LOC (variable-elimination + mini-bucket + treewidth) is the cheapest unblocking-PR because it does NOT depend on 255-PR-B sum-product engine — it's a parallel exact-inference axis using only `prob/mrf.Factor` from 255-PR-A. The SINGULAR-MOAT B14 mini-bucket + B19 particle-BP + B22 survey-propagation ~440 LOC is the differentiator: no zero-dependency Go library worldwide ships these three primitives, and reality would be the only Go shop with golden-file C# / Python / C++ contract on mini-bucket-MBE, particle-BP, and survey-propagation. The SINGULAR-CROSS-LINK B24 BP-LDPC ~80 LOC is the most-deployed BP-instance worldwide (every WiFi-6 / 5G-NR / satellite-comm receiver runs sum-product-BP at line-rate); ships as 80-LOC bridge once 210-PR-A and 255-PR-B land.

Recommended placement: extend `prob/mrf/` (255 target sub-package) with B8-B23, B25-B27 + new `coding/ldpc/bp_decoder.go` for B24 (210 target sub-package). No new top-level package.

Block-C BP-tier verdict: **PARTIAL OVERLAP with 254/255; COMPLEMENTARY DEEP-DIVE on BP-specific axes that 254/255 did NOT cover**. Slot-256 ships LAST in the 254→255→256 sequence, leveraging the substrate from both predecessors.
