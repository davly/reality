# 122 — queue-missing

**Topic:** queue: missing — M/M/c/K, M/G/c, G/G/1 (Kingman), Jackson networks, BCMP, polling models, fluid limits, fork-join

**Date:** 2026-05-07
**Agent:** 122 of 400
**Scope:** enumerate canonical queueing primitives absent from `reality/queue`

---

## Inventory of present functions (verified)

`C:\limitless\foundation\reality\queue\` ships **8 exported functions** + 0 helpers across 4 source files:

| File | Function | Model |
|------|---------|-------|
| `basic.go:56` | `MM1(λ, μ)` | M/M/1 |
| `basic.go:108` | `MMc(λ, μ, c)` | M/M/c |
| `basic.go:173` | `MM1K(λ, μ, K)` | M/M/1/K (finite buffer, single server) |
| `basic.go:235` | `LittlesLaw(L, λ)` | L = λW |
| `erlang.go:34` | `ErlangB(A, N)` | M/M/N/N blocking probability |
| `erlang.go:70` | `ErlangC(A, N)` | M/M/N delay probability |
| `erlang.go:100` | `ErlangCWaitTime(A, N, μ)` | E[Wq] under Erlang-C |
| `network.go:44` | `JacksonNetwork(λ_ext, P, μ, c)` | open Jackson, fixed-point traffic eqs, M/M/c per node |
| `metrics.go:26` | `BurstinessIndex(t[])` | C² = Var/E² |
| `metrics.go:69` | `OfferedLoad(λ, s)` | A = λ·s |

**Note correcting topic prompt and 121:**
- Erlang-B is **present** (`erlang.go:34`).
- Erlang-C is **present** (`erlang.go:70`) — the prompt's "Erlang-C [present?]" resolves to YES.
- Open Jackson networks are **present** (`network.go:44`) — the topic prompt lists "Jackson networks" as missing, but only the **closed** variant (MVA) and BCMP generalisation are absent.
- A simple two-stage Jackson tandem reduces to two M/M/c calls and is therefore covered by the existing `JacksonNetwork`. "Tandem queues" as a *named* primitive are absent but mathematically subsumed.

---

## Missing primitives — full enumeration

The Kendall A/S/c/K notation has 5 dimensions. The library covers `(M, M, {1,c}, {∞,K})` and three derived quantities (Erlang B/C, Little's Law, open Jackson). Everything below is absent.

### Tier 1 — Canonical, must-ship (zero-dep, ≤120 LOC each, golden-file ready)

These are textbook queueing primitives that any operations-research / capacity-planning consumer expects. All have closed-form or recursive formulas; none require simulation.

| # | Primitive | Formula | LOC | Notes |
|---|-----------|---------|-----|-------|
| T1.1 | **M/M/c/K** (multi-server finite buffer) | Birth-death balance + truncation; reduces to M/M/c when K→∞ and to M/M/1/K when c=1 | ~80 | Fills the gap between the two existing finite/multi-server models. Both Erlang-B (c=K) and M/M/1/K become special cases. |
| T1.2 | **M/M/c/c (Erlang loss / B-formula queue)** | Wraps `ErlangB` and returns the metrics tuple `(Lq=0, Wq=0, L=A·(1-pLoss), W=1/μ, ρ, pLoss)` | ~25 | Currently `ErlangB` returns only the scalar blocking probability — there is no `MMcc` returning the full tuple matching `MMc`/`MM1K`. |
| T1.3 | **M/G/1 (Pollaczek-Khinchine)** | `Lq = ρ²(1+C²_s)/(2(1-ρ))`; takes `(λ, E[S], Var[S])` or `(λ, E[S], C²_s)` | ~40 | The single most-cited queueing formula not yet in the package. Generalises M/M/1 (C²=1) and M/D/1 (C²=0). |
| T1.4 | **M/D/1 (deterministic service)** | Special case of M/G/1 with C²=0: `Lq = ρ²/(2(1-ρ))` | ~30 | Useful as its own export for token buckets, fixed-time pipelines, polled samplers. |
| T1.5 | **M/M/∞ (infinite servers)** | Poisson(L=A) steady state; `L=A`, `W=1/μ`, `Lq=0`, `Wq=0`, no blocking | ~25 | Models server farms where every arrival gets immediate dedicated capacity. Trivial but canonical. |
| T1.6 | **G/G/1 Kingman approximation** | `Wq ≈ (ρ/(1-ρ)) · ((C²_a+C²_s)/2) · (1/μ)` | ~35 | The "VUT formula" — heavy-traffic approximation, the workhorse of factory-physics / capacity planning. Topic prompt names it explicitly. |
| T1.7 | **G/G/c Allen-Cunneen approximation** | `Wq ≈ ErlangC(A,c)/(c·μ-λ) · (C²_a+C²_s)/2` | ~40 | Multi-server Kingman; degenerates to G/G/1 at c=1 and M/M/c at C²_a=C²_s=1. |
| T1.8 | **Erlang loss tuple `MMcc`** | (counted in T1.2) | — | (above) |
| T1.9 | **Closed Jackson / Mean Value Analysis (MVA)** | Reiser-Lavenberg recursion: for n=1..N, `R_i(n) = (1/μ_i)·(1+Q_i(n-1))`; `X(n) = n/Σ V_i·R_i(n)`; `Q_i(n) = X(n)·V_i·R_i(n)` | ~80 | Topic prompt names "Closed Jackson networks (Mean Value Analysis)". The single most useful queueing-network primitive after open Jackson. Single-class, exact, allocation-free with a workspace buffer. |

**Tier 1 total:** ~355 LOC across ~9 functions. Closes the gap to "M/M/* + M/G/1 + G/G/1 + closed/open Jackson + Erlang loss/delay" — the canonical stat-mech-of-queues set that every undergraduate text covers.

### Tier 2 — Standard extensions (still closed-form / recursive, broader coverage)

| # | Primitive | Formula / source | LOC | Notes |
|---|-----------|------------------|-----|-------|
| T2.1 | **Engset formula** (finite-source loss) | Recursive: `B(n,A,S) = (S-n)·A·B(n-1) / (n + (S-n)·A·B(n-1))` | ~40 | Finite-source counterpart to Erlang-B. Models small-population call centres, machine-repair shops. |
| T2.2 | **M/M/1 with priorities (non-preemptive)** | Cobham's formula: `Wq_k = E[S²_eff] / (2·(1-σ_{k-1})·(1-σ_k))` where σ_k = Σ_{j≤k} ρ_j | ~60 | Two-class and K-class variants. Standard queueing-with-classes primitive. |
| T2.3 | **M/M/1 preemptive priorities** | Recursive on remaining-service time: `W_k = (1/μ_k) / (1 - Σ_{j≤k} ρ_j)` | ~50 | Pairs with T2.2; preemptive vs non-preemptive is a doctrinal split. |
| T2.4 | **M/G/1 with vacations** (Takagi decomposition) | `Wq_vac = Wq_M/G/1 + E[V²]/(2·E[V])` | ~30 | Models polling, maintenance windows, server breaks. Classic Takagi 1991 result. |
| T2.5 | **M/G/1 retrials** | Falin-Templeton: `Lq ≈ Lq_M/G/1 · (1 + λ/ν)` for retrial rate ν | ~40 | Models call-back / retry queues common in modern API rate-limiting. |
| T2.6 | **Polling: gated discipline** | Cyclic visit, mean cycle time `E[C] = N·E[V] / (1-Σρ_i)`, gated waiting time formula | ~70 | Classical Takagi result for symmetric polling. |
| T2.7 | **Polling: exhaustive discipline** | Different waiting-time decomposition (gated vs exhaustive change the cycle-time conditioning) | ~70 | Pairs with T2.6. |
| T2.8 | **Burke's theorem departure process** | Returns `(λ_out, C²_out)` for an M/M/c node — for tandem feed-forward analysis | ~25 | Classical 1956 result: M/M/c departures are Poisson(λ) when stable. Useful as a doc/contract function for upstream→downstream chaining. |
| T2.9 | **Fork-join (synchronisation)** | Nelson-Tantawi approximation for K=2 servers: `E[T_FJ] = (12-ρ)/8 · E[T_M/M/1]`; general-K bounds (Varki, Klein) | ~80 | Topic prompt names this. Models scatter-gather, MapReduce barriers, distributed-tracing critical paths. |
| T2.10 | **Heavy-traffic / fluid limit** | `Lq ≈ (C²_a+C²_s)/(2·(1-ρ))` as ρ→1; explicit Brownian-motion approximation for transient response | ~50 | Topic prompt names "Fluid limits (heavy traffic)". Companion to Kingman; provides asymptotics. |
| T2.11 | **BCMP networks (open, multi-class)** | Product-form steady-state for M/M/1-FCFS, M/G/1-PS, M/G/1-LCFS-PR, M/M/∞ stations | ~150 | The natural generalisation of open Jackson to multi-class + non-exponential service. Baskett-Chandy-Muntz-Palacios 1975. |
| T2.12 | **PASTA witness / verifier** | Returns whether a workload satisfies Poisson Arrivals See Time Averages (deterministic check given an interarrival sequence) | ~30 | Companion to `BurstinessIndex`. Useful for validating M/G/1 input assumptions. |

**Tier 2 total:** ~695 LOC across ~12 functions. Covers the working vocabulary of queueing-textbook chapter 4-6 (Gross-Harris) and Kleinrock vol. 2.

### Tier 3 — Specialist / research-tier (advanced, larger LOC, less-frequent demand)

| # | Primitive | Notes |
|---|-----------|-------|
| T3.1 | **PH/PH/1, MAP/PH/1** | Phase-type and Markovian-arrival-process generalisations. Requires linalg matrix-exponential. ~200 LOC. |
| T3.2 | **GI/G/1 transform-based bounds** | Lindley equation, Wiener-Hopf factorisation. Numerically delicate; usually offered alongside Kingman. |
| T3.3 | **Fluid model with batches / batch-arrival M[X]/G/1** | Generalises Pollaczek-Khinchine to batch arrivals via PGF of batch size. ~50 LOC. |
| T3.4 | **Closed BCMP / multi-class MVA** | Bard-Schweitzer iterative approximation when exact MVA is too expensive. ~100 LOC. |
| T3.5 | **Diffusion approximations** | Reflected Brownian motion for G/G/1 transients; Halfin-Whitt regime for G/G/c with c→∞. Larger; hooks into prob/. |
| T3.6 | **Networks with blocking** (manufacturing blocking, communications blocking) | No product-form; requires approximation or simulation. Specialist. |
| T3.7 | **Quasi-birth-death (QBD) processes / matrix-analytic methods** | Latouche-Ramaswami logarithmic reduction. Heavy linalg dependency. ~250 LOC. |
| T3.8 | **G/G/∞ approximations** | Pollaczek formula + Brownian functional. Niche but cited. |
| T3.9 | **Loss networks (multi-rate Erlang)** | Kelly's loss network model; reduced-load (Erlang fixed-point) approximation. ~120 LOC. |
| T3.10 | **Time-dependent / non-stationary M_t/G_t/c** | PSA pointwise-stationary approximation, SIPP simple-infinite-server-pointwise. ~80 LOC. |
| T3.11 | **Generalised Jackson networks (GJN)** | Reiman 1984 heavy-traffic limit theorem; Brownian network approximation. Research-tier. |
| T3.12 | **Buzen's algorithm** (closed-network normalisation constant, mean-value alternative) | Pre-MVA convolution-based product-form. Mostly historical but cited. ~50 LOC. |

---

## Cross-package coupling

- **prob/** owns the distributions; M/G/1 and G/G/1 should accept either `(E[S], Var[S])` or a `prob.Distribution`-like interface. Recommend: pass `(meanS, c2s float64)` to keep zero-dep — coefficient of variation squared is the only moment any of these formulas use. (Pollaczek-Khinchine, Kingman, Allen-Cunneen are all C²-only.)
- **linalg/** is needed for T3.1, T3.7, and T2.11 (BCMP requires matrix solves on the routing matrix); not blocked for Tier 1/2.
- **graph/** is the natural place for routing-matrix construction utilities consumed by Jackson/BCMP; the queue package should accept the matrix pre-built.
- **calculus/** Gauss-Kronrod (slot 017's Tier 1) would be needed for T3.5 / Halfin-Whitt regime integrals.
- The `BurstinessIndex` already in `metrics.go` is the natural input feeder for T1.6/T1.7 — Kingman/Allen-Cunneen take `C²_a` directly. Document this linkage.

## API consistency notes (deferred to slot 124 / queue-api but logged here)

The existing `MM1`/`MMc`/`MM1K` use named-return tuples `(Lq, Wq, L, W, rho [, pLoss])`. Tier 1 additions should match this contract. Recommend a shared `Metrics` struct **only** if 124 reaches the same conclusion package-wide; otherwise keep tuples.

`ErlangB` returns a scalar; `MMcc` (T1.2) should return the full tuple to match `MM1K`'s contract.

## Highest-leverage single PR

**T1.3 + T1.4 + T1.6** as one ~110-LOC patch lands the *canonical* missing trio (Pollaczek-Khinchine + M/D/1 + Kingman) with one shared `c2` parameter idiom and a single doc-block on coefficient-of-variation conventions. Closes the textbook gap most consumers feel first. Each gets ≥20 golden-file vectors driven from the same analytic closed forms cross-checked against the existing `MM1` (C²=1 sanity check) and `MM1K` (with K→∞ and finite ρ).

## Summary

`reality/queue` is a **competent introductory queueing kit** (M/M/* + Erlang B/C + open Jackson + Little) that has the numerically-tricky pieces right (Jagerman recursion, ρ=1 special case in M/M/1/K, fixed-point iteration in Jackson). It is missing **everything past chapter 2 of any standard text** — most critically Pollaczek-Khinchine (M/G/1), Kingman (G/G/1), M/M/c/K, M/M/∞, M/M/c/c-as-tuple, and closed-Jackson MVA. Tier 1's 9 additions (~355 LOC) take the package from "M/M-only" to "covers the canonical queueing-theory undergraduate curriculum"; Tier 2's 12 additions (~695 LOC) add priorities, polling, vacations, retrials, fork-join, BCMP, and heavy-traffic limits — the working vocabulary of capacity planning and operations research. Topic-prompt-named gaps (M/M/c/K, M/G/c, G/G/1 Kingman, BCMP, polling, fluid limits, fork-join) all map cleanly to Tier 1 or Tier 2 line items.
