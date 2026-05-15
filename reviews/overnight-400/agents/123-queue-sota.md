# 123 — queue: SOTA library comparison

**Topic:** queue-sota — compare with SimPy, queueing-tool, JMT, LINE, BCMP solvers, ML-driven queue control
**Date:** 2026-05-07
**Agent:** 123 of 400
**Scope:** for each SOTA queueing library/paradigm, identify (1) headline algorithm, (2) engineering trick worth importing, (3) zero-dep portability for `reality/queue`.

Slots 121 (numerics) and 122 (missing) already established what `reality/queue` ships (8 functions: M/M/1, M/M/c, M/M/1/K, Erlang-B/C/CWaitTime, JacksonNetwork, BurstinessIndex, OfferedLoad, LittlesLaw). This review explicitly does **not** repeat those gap inventories; it positions the existing surface against the seven leading external systems and extracts what is *transferable* under the zero-dep / golden-file constraint.

---

## Comparator landscape (one-line each)

| System | Paradigm | Lang | Lic | Surface scope vs reality/queue |
|---|---|---|---|---|
| **SimPy** | Process-based DES via Python generators | Py | MIT | Disjoint — DES vs analytic. Useful as oracle for golden-file generation. |
| **Ciw** | DES, generative, networks of queues | Py | MIT | Disjoint — DES; richer features than SimPy (priorities, baulking, schedules, deadlock). |
| **queueing-tool** | DES on NetworkX-graph queueing networks | Py | MIT | Disjoint — DES; graph-routing style closer to Jackson/BCMP networks. |
| **JMT / JMVA** | Analytic + DES, BCMP product-form, GUI suite | Java | GPL-2 | **Direct competitor** for analytic tier — exact + approximate MVA, RECAL, CoMoM, Linearizer, Bard-Schweitzer. |
| **LINE** | Multi-paradigm meta-solver (CTMC, MAM, NC, FLUID, MVA, SSA) | MATLAB/Java | BSD-3 | **Most ambitious** — covers analytic *and* fluid/CTMC layers. Layered queueing networks (LQN). |
| **PMIF / BCMPSolver / pyMVA** | Pure-analytic BCMP / MVA implementations | Various | Various | Pure subset of JMT-analytic; useful as reference for algorithm forms. |
| **ML-driven (DRL admission/scheduling)** | Policy learning over queueing MDPs | PyTorch/RLlib | various | **Orthogonal** — uses analytic queueing as the *environment*, not a competitor; reality could ship the env. |

---

## 1. SimPy

- **Headline algorithm:** **Process-based DES with generator-yield event scheduling.** Each process is a Python generator function; `yield env.timeout(t)` / `yield resource.request()` suspends the process, scheduler advances to the next earliest event in a min-heap event queue. Event sort key is `(time, priority, event_id)` — the increasing event_id is the textbook tie-breaker that keeps the heap stable. ([SimPy docs — Time and Scheduling](https://simpy.readthedocs.io/en/latest/topical_guides/time_and_scheduling.html))
- **Engineering trick worth importing:** **The `(time, priority, monotonic_id)` triple-key for any event-priority queue.** This is the canonical fix to a heap that would otherwise be order-unstable when timestamps tie (e.g., simultaneous arrival/departure ticks). For reality, the trick lands in any future `queue/sim` subpackage and in `chaos`/`signal` tick-based pipelines. The `SortedQueue` for resource requests (priority, request-time, preempt flag) is the same idea generalised.
- **Zero-dep portability:** **Mostly applicable.** Go's `container/heap` is stdlib; the entire SimPy core (~1.2k LOC for env + events + resources + process) ports to Go in roughly the same LOC. The blocker is generators — Go has no `yield`, so processes become either (a) goroutines + channels (idiomatic but heavy at 10⁵+ entities) or (b) explicit continuation-passing via callback closures (lighter, less ergonomic). Recommend (b) for reality if a DES module is added: keeps the package zero-dep-of-runtime and matches the existing `chaos.ODE` callback style. **Verdict: high-fidelity port feasible at ~600 LOC; would be its own package (`queue/sim` or `realsim/`), not part of the analytic queue surface.**

## 2. Ciw

- **Headline algorithm:** **Generative DES with network-aware blocking, baulking, reneging, and priorities baked into the node.** Each node maintains its own active/in-service/blocked lists; routing matrix drives between-node flow; explicit Type-I (manufacturing) blocking via the receiving-node capacity check before service completion. ([Ciw paper, Palmer et al. 2018](https://www.tandfonline.com/doi/full/10.1080/17477778.2018.1473909))
- **Engineering trick worth importing:** **Phase-type and time-dependent inter-arrival/service distributions as first-class samplers.** Ciw's `ciw.dists.PhaseType` accepts an absorbing CTMC `(α, T)` and samples by walking the chain — this is exactly the bridge between an analytic PH/PH/1 (slot 122 T3.1) and a DES that can validate it. Also: **deterministic seeding per node** (each node gets its own `random.Random` instance) so simulation runs are byte-reproducible across platform RNG differences, not just Mersenne-twister-portable.
- **Zero-dep portability:** **Per-node independent RNG streams** is a 5-line idiom in Go using `math/rand/v2`'s `*Rand` per node — directly portable. Phase-type sampling is ~40 LOC and consumes only the absorbing-chain matrix already produced by `linalg`. Type-I blocking semantics need ~30 LOC of bookkeeping. **Verdict: portable as additions to the future DES module; not relevant to the analytic queue surface.**

## 3. queueing-tool

- **Headline algorithm:** **Graph-as-queueing-network DES** — uses NetworkX graph with queues on edges (or nodes) and agents (customers) traversing. The simulation core is event-driven on the same `(time, edge, agent)` heap pattern but the *topology is the graph*, which makes routing matrix construction free. ([queueing-tool docs](https://queueing-tool.readthedocs.io/en/latest/overview.html))
- **Engineering trick worth importing:** **Routing matrix derived from an explicit graph object rather than a dense matrix argument.** For `reality/queue.JacksonNetwork`, the current API takes a dense `P [][]float64` routing matrix. queueing-tool's pattern — accept a `graph.Graph` (which `reality/graph` already provides) with edge weights = routing probabilities — is sparse-friendly, enables BCMP multi-class routing through edge labels, and makes the API match the topology people actually draw. The cross-package coupling is `graph.AdjacencyList` → `(I − Pᵀ)` solve.
- **Zero-dep portability:** **Directly applicable.** A `JacksonNetworkFromGraph(g graph.Graph, lambdaExt []float64, mu []float64, c []int)` overload is ~30 LOC of adapter + the existing solve. No external deps. **Verdict: high-value 30-LOC API improvement; ties queue and graph packages without breaking the existing dense-matrix entry point.**

## 4. JMT / JMVA — the analytic competitor

This is the system whose *analytic tier* most directly competes with `reality/queue`. JMVA is the BCMP-product-form analyzer inside the JMT suite.

- **Headline algorithm — exact MVA (Reiser-Lavenberg 1980):** for closed network with N customers and M load-independent stations, recursively for `n = 1..N`:
  ```
  R_i(n) = (1/μ_i) · (1 + Q_i(n−1))         // arrival theorem: arriving customer sees mean queue minus self
  X(n)   = n / Σ_i V_i · R_i(n)              // throughput (V_i = visit ratio)
  Q_i(n) = X(n) · V_i · R_i(n)               // Little applied per station
  ```
  O(NM) time, O(M) workspace. Exact for product-form (BCMP class-1: M/M/1-FCFS, M/G/1-PS, M/G/1-LCFS-PR, M/M/∞). ([JMVA — JMT docs](https://jmt.sourceforge.net/JMVA.html), [Wikipedia — Mean value analysis](https://en.wikipedia.org/wiki/Mean_value_analysis))
- **Headline algorithm — exact MVA (load-dependent / multi-class):** when stations have load-dependent service rates, the recursion must track the marginal queue-length distribution `P_i(j; n)` for `j = 0..n`, not just the mean — O(N²RM) for R classes. JMVA ships a *stabilized* variant inspired by Seidmann's approximation to fight numerical drift (the load-dependent recursion can produce negative probabilities under naive forward iteration on near-saturated networks).
- **Headline algorithm — approximate MVA (Bard-Schweitzer 1981):** replaces the arrival-theorem assumption `Q_i(n−1) ≈ ((n−1)/n) · Q_i(n)` and iterates the closed system to fixed point — O(M) per sweep, ~5-20 sweeps in practice. Quadratic accuracy in 1/N, suitable for N ≥ 50.
- **Headline algorithm — Linearizer (Chandy-Neuse 1982):** applies a quadratic correction to Bard-Schweitzer using two side problems at populations N−1 and N−2; far more accurate, slightly higher cost. JMVA ships this and the AQL (Aggregate Queue Length) variant.
- **Headline algorithm — RECAL / CoMoM (normalising constant family):** the Convolution algorithm (Buzen 1973) computes the normalising constant `G(N)` of the Gordon-Newell theorem in O(NM) time via the recurrence `g(n,m) = X_m · g(n−1, m) + g(n, m−1)`; RECAL extends to multi-class at O(NM) per chain; **CoMoM (Casale 2008) achieves O(N log N · M)** by Fourier inversion of the generating function and is JMVA's default for large N. ([Buzen's algorithm — Wikipedia](https://en.wikipedia.org/wiki/Buzen's_algorithm))

- **Engineering trick worth importing — the *stabilised* MVA:** the naive load-dependent MVA recursion can produce `Q_i(n) < 0` due to roundoff on near-saturated stations. JMVA's stabilisation pins `Q_i(n) ← max(0, Q_i(n))` and renormalises throughput each step — algebraically a no-op when the recursion is well-conditioned, recovers correctness when it isn't. This is the same family of defence as `reality/queue/basic.go:225`'s `Wq = max(0, …)` clip; the load-dependent MVA needs the same idiom one level up.
- **Engineering trick worth importing — Buzen convolution as the primary closed-network primitive:** for *single-class* closed networks the convolution algorithm is **simpler than MVA** (single 2D table, no per-station vector recursion) and gives `G(N)` directly, from which all marginals fall out by ratios. For reality, ship Buzen first (~50 LOC, single 2D buffer), MVA second.
- **Zero-dep portability:** **All MVA variants are zero-dep.** Exact MVA is ~80 LOC, Bard-Schweitzer ~50, Buzen ~50, Linearizer ~120, CoMoM ~200 (needs `signal.FFT` for the generating-function inversion, already in reality). Every algorithm operates on float64 vectors and matrices; no transcendentals beyond exp/log; golden files generated from `math/big` MVA at 256-bit (multi-class small networks fit in feasible bigfloat budget). **Verdict: full BCMP analytic stack lands in ~500 LOC, no new dependencies. This is the single biggest scope-expansion the queue package has available.**

## 5. LINE — the meta-solver

LINE is the only entry on this list whose *coverage matrix* exceeds JMT's: it has CTMC, MAM, NC, FLUID, MVA, and SSA solvers under one model object. ([LINE Solver](https://line-solver.sourceforge.net/))

- **Headline algorithms:**
  - **CTMC:** explicit infinitesimal-generator construction → uniformisation for transient, sparse linear solve `πQ = 0, π·1 = 1` for stationary. Q matrix is sparse, hence Krylov methods (GMRES, BiCGStab) preferred at scale.
  - **MAM (Matrix-Analytic Methods):** for QBD processes, computes the rate matrix `R` via Latouche-Ramaswami **logarithmic reduction** — `O(log² ε)` iterations vs `O(1/ε)` for naive iteration, this is the standard 1993 result that replaced classical `R = A_0 + R·A_1 + R²·A_2` fixed-point iteration.
  - **NC (Normalising Constant):** generalised Buzen / RECAL / CoMoM for arbitrary BCMP.
  - **FLUID:** mean-field ODE limit — for large N, marginal queue-length probabilities converge to a deterministic fluid trajectory governed by an ODE on the queueing-rate vector. Solves with stiff ODE (Rosenbrock/BDF).
  - **SSA:** Gillespie / next-reaction-method CTMC simulation; state space built lazily from the initial state, no upfront enumeration.
  - **MVA:** as JMVA.
- **Engineering trick worth importing — the meta-solver dispatch idea:** LINE wraps every solver behind a uniform `solve(Model, options)` interface and dispatches on model class (open vs closed; product-form vs not; stationary vs transient; small-state-space vs not). For reality, an analogous `queue.Solve(network)` would look up product-form-eligibility and choose MVA, fall back to CTMC stationary solve when not, and to mean-field fluid when `Σ N_r ≥ threshold`. The dispatch logic itself is ~40 LOC of structural pattern matching on a `Network` struct.
- **Engineering trick worth importing — Latouche-Ramaswami logarithmic reduction for QBD R-matrix:** when slot 122 T3.7 (matrix-analytic methods) lands, this is the algorithm to use, not naive fixed-point iteration. It converges quadratically in `log(1/ε)` and the standard implementation is ~80 LOC over a `linalg.Solve` for matrix inverses. Reference: Latouche & Ramaswami 1993, "A logarithmic reduction algorithm for QBD processes."
- **Engineering trick worth importing — fluid limit as the regime-extension:** at high `N` (closed) or high `λ` (open), MVA cost is O(N²RM) / RECAL grows; the mean-field fluid ODE is **N-independent** — same cost for N=10 or N=10⁹. Couples cleanly into `chaos.ODE` (already in reality). The classical reference is Kurtz 1970; modern queueing usage is from Tsitsiklis-van Roy and the Gast-Gaujal series.
- **Zero-dep portability:** **CTMC stationary solve** = `linalg.Solve(Q^T augmented with 1·1 row, e_n)` — already buildable. **MAM logarithmic reduction** = pure linalg, ~80 LOC. **Fluid ODE** = pure ODE, plugs into `chaos.RK4`. **SSA** = needs `prob.Exponential` sampler (slot 117 T1.3) and a min-heap event queue. **CoMoM/NC** = uses `signal.FFT` for the generating-function inversion. *Every LINE solver class is reachable from existing reality packages* — this is the single most encouraging finding of the review. **Verdict: full LINE-equivalent meta-solver lands in ~1.2k LOC across queue, linalg, chaos, signal hookups. Multi-PR project; lay foundation in ~6 weeks.**

## 6. BCMP solvers (algorithmic notes beyond JMT/LINE)

- **The Convolution algorithm (Buzen 1973, single-class)** — the primitive that *underlies* every closed-network analyser. Recurrence `g(n,m) = X_m · g(n−1, m) + g(n, m−1)` with `g(0, m) = 1, g(n, 0) = X_1^n`. ~50 LOC. Should be the **first** closed-network primitive in reality.
- **Tree-convolution (Lam-Lien 1983)** — for *sparse* routing matrices (many stations with small visit-ratio sets), tree convolution beats flat convolution. Only relevant if reality eventually models LANs / chip interconnects with topology constraints. Defer to Tier 3.
- **Generating-function inversion (Bertozzi-McKenna 1993, Casale's CoMoM 2008)** — flips the closed-network problem into the frequency domain and inverts numerically. The `signal.FFT` already in reality is the right tool. CoMoM is JMVA's default for large N; ~200 LOC.
- **Reduced-load Erlang fixed-point** (Kelly 1986/1991, loss networks) — the multi-rate generalisation of Erlang-B; iterate `B_j(A_j_eff)` where `A_j_eff` accounts for blocking on shared links. Converges in ~10-30 iterations for typical telecom topologies. ~120 LOC, slot 122 T3.9.

**Zero-dep portability of the BCMP family: 100%.** The hardest piece (CoMoM) needs FFT, which is in `signal/`. The arithmetic-stability concern is the convolution recurrence underflowing for large N — same issue as Erlang-B's Jagerman recursion (slot 121's primary finding). Standard fix is the log-domain recurrence: maintain `log g(n,m)` and use `logsumexp` to combine. ~10 extra LOC.

## 7. ML-driven queue control (orthogonal frontier)

The 2022-2026 literature on RL/DRL for queue admission and scheduling is a different *axis* — it uses the queueing model as the *environment* (an MDP whose transitions follow a CTMC), then trains a policy that picks actions (admit/reject, route to server-i, set service rate) to optimise a reward. Key recent work: Roychowdhury et al. 2024 ICML on M/M/k/k+N admission with unknown rates ([arxiv 2202.02419](https://arxiv.org/abs/2202.02419)), Maguluri-Srikant 2025 on heavy-traffic-optimal MaxWeight + replay, Walton-Xu 2024 on deep-RL for scheduling under unknown service distributions ([Frontiers 2025 MARL flexible shop survey](https://www.frontiersin.org/journals/industrial-engineering/articles/10.3389/fieng.2025.1611512/full)).

- **Headline algorithm class:** **policy-gradient + Lyapunov-stability constraint.** Naive DRL on queueing MDPs diverges because the state space is unbounded — queue lengths can grow without limit on bad policies. The 2024-2026 standard is to add a Lyapunov drift constraint on training (`E[V(X_{t+1}) − V(X_t) | X_t] ≤ 0` outside a compact set) to guarantee policy-induced stability. See Lin-Srikant 2024.
- **Engineering trick worth importing — the queueing model *as a Go interface*:** for reality to be useful to the ML/RL ecosystem, it should expose its analytic models as `(state) → (rate matrix Q, reward vector r, action set A)` interfaces. This is a 100-LOC adapter that converts an `MMc` / `JacksonNetwork` into an `MDP` consumable by external RL libraries — without reality itself depending on any RL or ML package. The pattern is identical to OpenAI Gym's `Env` interface.
- **Zero-dep portability:** **Reality's role here is to be the environment, never the agent.** The agent (policy network, replay buffer, optimizer) lives in PyTorch / RLlib downstream. The environment (CTMC transitions, reward shaping, episode termination) is pure float64 + linalg sparse matvec — all in reality already. **Verdict: a `queue/mdp` subpackage exposing `Step(action) (state, reward, done)` over an underlying `MMc` / `Jackson` would let reality plug into the RL ecosystem without ever importing PyTorch.** ~150 LOC; orthogonal to the analytic-tier work.

---

## Tricks worth importing — consolidated table

Ranked by leverage × portability:

| # | Trick | Source | LOC | Tier | Where it lands |
|---|-------|--------|-----|------|----------------|
| 1 | **Forward-direction Buzen convolution** as the primary closed-network primitive | JMT/JMVA, LINE-NC | 50 | 1 | `queue/closed.go:Buzen()` |
| 2 | **Exact MVA (Reiser-Lavenberg)** single-class, load-independent | JMT/JMVA, LINE-MVA | 80 | 1 | `queue/closed.go:MVA()` |
| 3 | **Triple-key event heap `(time, priority, monotonic_id)`** | SimPy | 30 | 2 | future `queue/sim/scheduler.go` |
| 4 | **Routing matrix from `graph.Graph`** API overload | queueing-tool | 30 | 1 | `queue/network.go:JacksonNetworkFromGraph()` |
| 5 | **Stabilised load-dependent MVA** (`max(0, …)` per-station clip + throughput renormalisation) | JMT-SMVA | 20 over MVA | 2 | added when load-dep MVA lands |
| 6 | **Bard-Schweitzer fixed-point AMVA** for large-N approximate | JMT, LINE | 50 | 2 | `queue/closed.go:MVAApproxBS()` |
| 7 | **CTMC stationary solve via `linalg.Solve(Q^T, b)`** | LINE-CTMC | 25 over linalg | 2 | `queue/ctmc.go:Stationary()` |
| 8 | **Latouche-Ramaswami logarithmic reduction** for QBD R-matrix | LINE-MAM | 80 | 3 | `queue/mam.go:RmatrixLR()` |
| 9 | **Mean-field fluid ODE** for large-N regime | LINE-FLUID | 60 over chaos | 3 | `queue/fluid.go:Trajectory()` |
| 10 | **CoMoM via FFT-based generating-function inversion** | JMVA, LINE-NC | 200 over signal | 3 | `queue/closed.go:CoMoM()` |
| 11 | **Phase-type CTMC sampler** for DES | Ciw | 40 | 3 | future `queue/sim/dist.go` |
| 12 | **MDP / RL-environment adapter** over `MMc` / `JacksonNetwork` | (custom, framing) | 150 | 3 | `queue/mdp/env.go` |
| 13 | **Linearizer (Chandy-Neuse)** AMVA, quadratic correction | JMT | 120 | 3 | `queue/closed.go:Linearizer()` |
| 14 | **Per-node independent RNG streams** for reproducible DES | Ciw | 5 | 2 | future `queue/sim/scheduler.go` |
| 15 | **Meta-solver dispatch** (`Solve(network)` → MVA / CTMC / Fluid by class) | LINE | 40 | 3 | `queue/solver.go` |

**Total Tier 1 (immediate):** ~160 LOC — Buzen, single-class MVA, graph-routing overload. Doubles the analytic surface and closes the most-cited textbook gap (closed networks) with bounded risk.

**Total Tier 2 (medium-term):** ~125 LOC additional — stabilised MVA, Bard-Schweitzer, CTMC-stationary, per-node-RNG. Brings parity with JMVA's *non-load-dependent* surface.

**Total Tier 3 (long-term, multi-package):** ~700 LOC additional — MAM, fluid, CoMoM, RL-env, Linearizer, dispatch. Approaches LINE coverage; spans queue + linalg + chaos + signal.

---

## Cross-reference to slots 121, 122

- **Slot 121 (numerics):** the Jagerman-direction defect in `ErlangB` is the same family of issue as the load-dependent MVA stability problem fixed by JMT-SMVA. The fix idiom transfers: stabilise by clipping non-physical negative results and renormalising downstream invariants (Σ probabilities, Little's L=λW). 121's fix to `ErlangB` should land first; SMVA-style clip is the same pattern at a higher level.
- **Slot 122 (missing):** every Tier 1 / Tier 2 / Tier 3 item above lines up with the slot-122 backlog. The mapping:
  - 122-T1.9 (Closed Jackson MVA) ⇄ trick #2 (exact MVA) and #1 (Buzen prerequisite)
  - 122-T2.11 (BCMP open multi-class) ⇄ tricks #1, #2, #6 (multi-class extensions)
  - 122-T2.10 (Heavy-traffic / fluid limit) ⇄ trick #9 (mean-field fluid ODE)
  - 122-T3.4 (Bard-Schweitzer) ⇄ trick #6
  - 122-T3.7 (QBD / matrix-analytic) ⇄ tricks #7, #8
  - 122-T3.12 (Buzen) ⇄ trick #1

Slot 123's contribution beyond 121/122: identifies which **algorithmic forms** to use for those entries (forward Buzen vs naive forward; LR for QBD vs naive iteration; stabilised vs naive load-dep MVA; CoMoM vs naive convolution at large N) and confirms zero-dep portability for every single one against the existing `reality` package set.

---

## Bottom line

The analytic-tier competitor is **JMT/JMVA** (BCMP exact + approximate MVA, Buzen convolution); the meta-solver competitor is **LINE** (adds CTMC, MAM, fluid). Every algorithm either system uses is **zero-dep-portable** to Go on top of reality's existing `linalg`, `signal`, `chaos`, `prob`, `graph` packages — no external libraries required. The simulation-tier systems (SimPy, Ciw, queueing-tool) are *disjoint* from reality's analytic mandate; their main contribution is golden-file *oracle* utility (run external sim → compare against analytic) and a small set of engineering idioms (triple-key event heap, per-node RNG streams, graph-routing API). ML-driven queue control is *orthogonal*: reality's role is to be the environment, not the agent — a thin `queue/mdp` adapter unlocks the ecosystem without taking on RL dependencies. **Highest-leverage immediate trio: Buzen + single-class exact MVA + graph-routing overload — ~160 LOC, doubles the analytic surface, no new package dependencies.**

## Files referenced

- `C:/limitless/foundation/reality/queue/basic.go`
- `C:/limitless/foundation/reality/queue/erlang.go`
- `C:/limitless/foundation/reality/queue/network.go`
- `C:/limitless/foundation/reality/queue/metrics.go`
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/121-queue-numerics.md`
- `C:/limitless/foundation/reality/reviews/overnight-400/agents/122-queue-missing.md`

## Sources

- [SimPy — Time and Scheduling](https://simpy.readthedocs.io/en/latest/topical_guides/time_and_scheduling.html)
- [SimPy — Shared Resources](https://simpy.readthedocs.io/en/latest/topical_guides/resources.html)
- [Ciw — GitHub repo](https://github.com/CiwPython/Ciw)
- [Ciw paper, Palmer et al. 2018](https://www.tandfonline.com/doi/full/10.1080/17477778.2018.1473909)
- [queueing-tool docs](https://queueing-tool.readthedocs.io/en/latest/overview.html)
- [JMT — JSIMgraph](https://jmt.sourceforge.net/JSIMg.html)
- [JMVA — Java Modelling Tools](https://jmt.sourceforge.net/JMVA.html)
- [JMT users manual (PDF)](https://www.ctr.unican.es/asignaturas/dec/doc/jmt_users_manual.pdf)
- [LINE Solver site](https://line-solver.sourceforge.net/)
- [LN: Meta-solver for LQN analysis (Springer)](https://link.springer.com/chapter/10.1007/978-3-031-16336-4_12)
- [Mean value analysis — Wikipedia](https://en.wikipedia.org/wiki/Mean_value_analysis)
- [Buzen's algorithm — Wikipedia](https://en.wikipedia.org/wiki/Buzen%27s_algorithm)
- [SMVA: Stable Mean Value Analysis (Springer)](https://link.springer.com/chapter/10.1007/978-3-319-92378-9_2)
- [Learning to Admit Optimally in M/M/k/k+N (arxiv 2202.02419)](https://arxiv.org/abs/2202.02419)
- [Frontiers 2025 — MARL flexible shop scheduling survey](https://www.frontiersin.org/journals/industrial-engineering/articles/10.3389/fieng.2025.1611512/full)
- [Queue stability-constrained DRL for MEC (MDPI 2025)](https://www.mdpi.com/1999-4893/18/8/498)
