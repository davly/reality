# 154 | synergy-chaos-timeseries

**Topic:** chaos × timeseries — Takens embedding, Lyapunov forecasting, RQA
**Block:** B (cross-package synergies)
**Date:** 2026-05-08
**Scope:** capabilities that emerge ONLY when `chaos/` and the `prob/timeseries.go`
+ `timeseries/{garch,dcc}` ensemble compose; not what either is missing in
isolation (covered by 026-030 / 136-140).

## Two-line summary

Today `chaos/` ships forward-simulation primitives (RK4, Lorenz, Lyapunov-of-a-known-map,
RecurrencePlot-of-a-known-trajectory) and `prob/timeseries.go` ships
linear-Gaussian forecasters (ExpSmooth, Holt, ARIMA-stub) — but **neither knows how
to read a 1-D scalar series back into a phase-space trajectory**, which is the
single bridge separating synthetic chaos from observed chaos. **Twelve synergy
primitives (T1-T12) totalling ~1240 LOC of glue** stand up Takens embedding,
data-driven Lyapunov estimation (Rosenstein/Kantz), correlation dimension D₂,
RQA metrics, permutation entropy, the 0-1 chaos test, and local-model nonlinear
forecasting on the reconstructed attractor; cheapest first PR is **T1 TakensEmbed**
at ~40 LOC because it is the universal preprocessor every other synergy depends
on, and the highest-leverage one-day unlock is **T9 PermutationEntropy** —
~80 LOC of pure combinatorics that gives a model-free chaos-vs-noise classifier
which competes head-to-head with `prob.ARIMA` on Pulse anomaly detection.

---

## Bases — what each side exposes today

### `chaos/` (588 LOC across 3 source files, agent 026)

`SolveODE, RK4Step, EulerStep` (ode.go) — *forward* integration only.
`LorenzSystem, RosslerSystem, LotkaVolterra, SIRModel, VanDerPol, LogisticMap, GameOfLife` (systems.go) — generators.
`LyapunovExponent(f, x0, n)` — assumes you have **the closed-form map `f`**, not data.
`BifurcationDiagram(f, ...)` — same assumption.
`RecurrencePlot(trajectory, threshold)` — accepts an *N×d* trajectory of state vectors, not a 1-D series; *no metric extraction* (RR/DET/LAM/L_max/ENT) — just the boolean matrix.

### `prob/timeseries.go` (308 LOC, agent 136)

`ExponentialSmoothing, HoltLinear, ARIMA, levinsonDurbin`. All three are linear,
Gaussian, stationary-or-difference-to-stationary. There is no nonlinear
predictor and no notion of state-space reconstruction.

### `timeseries/garch`, `timeseries/dcc` (~580 LOC)

Univariate GARCH(1,1) volatility filter and DCC correlation. Stochastic-volatility
not deterministic-chaos; orthogonal to this synergy.

### What is missing on the seam

- No 1-D-series → m-D-trajectory reconstruction
- No data-driven dimension/delay choice (FNN, AMI)
- No data-driven Lyapunov (Rosenstein, Kantz, Wolf)
- No correlation dimension D₂ (Grassberger-Procaccia)
- No RQA quantification of `RecurrencePlot`
- No permutation entropy, no 0-1 chaos test
- No local-model forecasting on a reconstructed attractor

Every item above is a 30-150 LOC composition of **already-shipped primitives**.
None is new mathematics; the gap is glue.

---

## The conceptual unlock — Takens (1981) embedding theorem

For a deterministic system on a d-dimensional attractor, the delay-coordinate
map

    x(t) ↦ ( s(t), s(t-τ), s(t-2τ), …, s(t-(m-1)τ) ),    m ≥ 2d+1

is generically a smooth embedding of the attractor into ℝᵐ. This single
theorem promotes a 1-D scalar observation `s(t)` to an m-D trajectory on
which **every existing chaos primitive applies**: `RecurrencePlot` works,
distances are meaningful, neighbors index dynamics. Nine of the twelve
synergies below are direct corollaries.

The cost is purely connective tissue because:
- delay-coordinate construction is a one-loop reshape (T1)
- AMI/FNN are histogram + nearest-neighbor scans
- Rosenstein/Kantz are mean-log-divergence over neighbor pairs
- RQA metrics are diagonal/vertical line statistics on the existing `[][]bool`

---

## T1 — `TakensEmbed`: 1-D series → m-D phase space

Foundational; every other synergy depends on it.

```go
func TakensEmbed(series []float64, m, tau int, out [][]float64)
// out[i][k] = series[i + k*tau], i ∈ [0, len(series) - (m-1)*tau)
```

Pure reshape. **~40 LOC.** No new math. Lives in `chaos/embed.go`.
**Saturation witness:** Embed Lorenz x(t) with m=3, τ=8, threshold-RP it,
verify ≥85 % overlap with the RP of the true (x,y,z) trajectory at the same
threshold (Sauer-Yorke-Casdagli 1991 prediction).

## T2 — `AverageMutualInformation`: choose τ from data

The first local minimum of AMI(τ) is the standard Fraser-Swinney 1986 choice
of delay. Composes a 2-D histogram of (s(t), s(t+τ)) with Shannon entropy
which `prob/` already computes elsewhere.

```go
func AverageMutualInformation(series []float64, maxLag, nBins int, out []float64)
func FirstMinimumAMI(ami []float64) int
```

Connective tissue: 2-D histogram (~40 LOC) + scan for first minimum (~10 LOC) +
bin-and-log entropy that mirrors `prob/distribution.go:KLDivergenceNumerical`
patterns. **~110 LOC total.** Saturation: AMI of i.i.d. uniform tails to zero;
AMI of Lorenz x(t) has first min at τ ≈ 8 (dt = 0.01), agreeing with TISEAN's
mutual.

## T3 — `FalseNearestNeighbors`: choose m from data

Kennel-Brown-Abarbanel 1992. For each candidate dimension m, compute the
fraction of nearest neighbors in m-D that "split apart" when promoted to
(m+1)-D — false neighbors caused by unfolding. The dimension at which this
fraction drops below ~1 % is the embedding dimension.

```go
func FalseNearestNeighbors(series []float64, mMax, tau int, rTol float64) []float64
```

Composes `TakensEmbed` (T1) + brute-force NN search (`euclideanDist` already
in `chaos/analysis.go:151`) + the Kennel ratio test. **~120 LOC.** O(N² m_max)
brute force is fine up to N≈5000; KD-tree variant deferred (none of `reality`
ships KD-tree today; tracked in agent 081 graph-numerics). Saturation:
returns ≈0 at m=3 for Lorenz, m=4 for Mackey-Glass τ=17.

## T4 — `LyapunovRosenstein`: data-driven λ₁

Rosenstein-Collins-DeLuca 1993. For each point in the reconstructed
trajectory, find its nearest neighbor (excluding temporally-close points
via the Theiler window), then track the mean log-divergence

    ⟨ ln d_j(i) ⟩  ≈  λ₁ · i · dt  +  const

over forward time `i`. The slope is λ₁ in nats per unit time.

```go
func LyapunovRosenstein(series []float64, m, tau, theiler, iMax int, dt float64,
    divergenceCurve []float64) float64
```

Composes T1 + Theiler-windowed NN + a linear regression on log distance vs `i`.
**~150 LOC.** This is the data-side dual of the existing
`chaos/analysis.go:LyapunovExponent` (which requires the closed-form map):
together they form a **calibration pair** — synthesize a known-λ Lorenz
trajectory, observe only x(t), reconstruct, recover λ ≈ 0.906 ± 0.05.
That round-trip is the saturation witness and goes straight into
golden-file infrastructure as a dimensionless ratio λ_data / λ_true.

## T5 — `LyapunovKantz`: stochastic-extension λ₁

Kantz 1994 generalizes Rosenstein by averaging over a *neighborhood ball*
rather than the single nearest neighbor — more robust to noise and to
small datasets. Same skeleton as T4 but inner loop replaces `argmin d` with
"all neighbors with d < ε". **~80 LOC delta on top of T4** (shares Theiler
helper, distance kernel, regression).

```go
func LyapunovKantz(series []float64, m, tau, theiler, iMax int, eps, dt float64,
    divergenceCurve []float64) float64
```

Saturation: Kantz ≈ Rosenstein on clean Lorenz (within 5 %); Kantz < Rosenstein
on σ=5 % AWGN-corrupted Lorenz (Rosenstein's nearest neighbor is dominated
by noise; Kantz averages it out).

## T6 — `CorrelationDimensionGP`: D₂ from Grassberger-Procaccia

Grassberger-Procaccia 1983. The correlation sum

    C(ε) = (2 / N(N-1)) · #{ (i,j) : i<j, ‖x_i - x_j‖ < ε }

scales as C(ε) ~ ε^D₂ in the scaling region, so D₂ is the slope of
log C vs log ε.

```go
func CorrelationSum(trajectory [][]float64, epsilons, out []float64)
func CorrelationDimensionGP(trajectory [][]float64, epsilons []float64) float64
```

Composes the existing `euclideanDist` + a sweep over ε + linear regression
in log-log. **~140 LOC.** Naive O(N²) per ε; deferred Bentley box-counting
optimization. Saturation: D₂(Lorenz) ≈ 2.05 ± 0.05 (TISEAN reference);
D₂(Hénon) ≈ 1.21.

## T7 — `RQAMetrics`: quantify what `RecurrencePlot` returns

The existing `chaos/analysis.go:RecurrencePlot` returns a binary matrix and
**stops**. The whole point of an RP is the line-statistics layer on top
(Marwan-Romano-Thiel-Kurths 2007). Six standard metrics:

| Metric | Formula | Meaning |
|---|---|---|
| RR (recurrence rate) | (1/N²) Σ R[i][j] | density |
| DET (determinism) | Σ_l>l_min l·P(l) / Σ_i,j R[i][j] | predictability |
| L (avg diag length) | Σ l·P(l) / Σ P(l) | predictability time |
| L_max | longest diagonal | 1 / λ₁ proxy |
| LAM (laminarity) | vertical analog of DET | intermittency |
| ENT (Shannon entropy of P(l)) | -Σ p(l) log p(l) | line-length complexity |

```go
type RQA struct{ RR, DET, L, Lmax, LAM, ENT float64 }
func RecurrenceQuantification(rp [][]bool, lMin, vMin int) RQA
```

Pure histogram-of-line-lengths over an existing matrix. **~180 LOC.**
Saturation: DET(periodic orbit) → 1; DET(white noise) → ~RR; L_max(Lorenz
m=3,τ=8) → 80 ± 5 (Marwan benchmark).

## T8 — `RecurrencePlotFromSeries`: 1-D series → RP without manual embed

Convenience wrapper: T1 + the existing `RecurrencePlot`. Useful because every
applied paper writes "we computed an RP from the EEG channel" — they mean
the embed-then-RP composition.

```go
func RecurrencePlotFromSeries(series []float64, m, tau int, threshold float64) [][]bool
```

**~30 LOC** glue.

## T9 — `PermutationEntropy`: model-free chaos-vs-noise

Bandt-Pompe 2002. For each window of length m, encode the *ordinal pattern*
(which index has the smallest value, second smallest, …) as one of m!
permutations; the Shannon entropy of the empirical distribution over
permutations is permutation entropy H_p ∈ [0, log m!]. Normalised
H_p / log m! is the standard scalar.

```go
func PermutationEntropy(series []float64, m, tau int) float64
```

**~80 LOC.** Hot inner loop is in-place argsort of m elements (m ≤ 7 in
practice → insertion sort optimal). No external dependencies; pure
combinatorics + log. **Highest leverage in this list:**
- model-free (no embedding choice, no metric, no threshold)
- robust to monotone transforms of the series
- O(N · m log m) — cheap enough for streaming
- on i.i.d. uniform → 1.0; on logistic-r=4 → ~0.85; on logistic-r=3.5 (period-4) → ~0.4

This is the one-day unlock that puts a chaos-detector in `prob/anomaly.go`'s
toolbox without ever solving an ODE. **Saturation witness:**
H_p(white noise) / log m! → 1; H_p(sin) / log m! → 0.

## T10 — `ZeroOneChaosTest`: Gottwald-Melbourne 0-1 test

Gottwald-Melbourne 2004/2009. For each c ∈ (0, 2π) draw two translation
variables

    p(n) = Σ_{j=1}^{n} φ(j) cos(jc),     q(n) = Σ_{j=1}^{n} φ(j) sin(jc)

then look at the asymptotic mean-square displacement
M_c(n) = (1/N) Σ_k (p(k+n) - p(k))² + (q(k+n) - q(k))² and compute its
correlation with `n` itself; median-over-c of that correlation is K ∈ [0, 1]:
K → 0 = regular, K → 1 = chaotic.

```go
func ZeroOneChaosTest(series []float64, nC int, rng func() float64) float64
```

**~140 LOC.** Pure cumulative-sum + correlation; no embedding required, no
parameters except the number of c-samples. Complements T9: agreement between
H_p > 0.7 and K > 0.7 is a strong joint witness. Saturation:
K(logistic r=4) ≈ 0.99; K(logistic r=3.2) ≈ 0.02; K(white noise) — known
failure mode of 0-1 test, returns ≈ 1.0, **and that limitation is documented**.

## T11 — `LocalLinearForecast`: nonlinear analog of HoltLinear

The Farmer-Sidorowich 1987 / Sugihara-May 1990 idea: forecast s(t+1) by
finding the k nearest neighbors of the current m-D state in the historical
embedded trajectory, then fitting a local linear model to their next-step
images and evaluating it at the current state.

```go
func LocalLinearForecast(series []float64, m, tau, k int, horizon int,
    out []float64)
```

Composes T1 + brute-force k-NN + `linalg.LinearRegression` (already shipped
in `linalg/`, see agent 096). **~170 LOC** of glue. This is the **direct
nonlinear analog** of `prob/timeseries.go:HoltLinear` — same signature
shape, same `out` discipline — and it should benchmark side-by-side on the
same series via the existing testutil golden-file harness. Saturation:
on a clean Lorenz x(t), 1-step error ratio LocalLinear / HoltLinear < 0.05;
on a random walk, ratio ≈ 1.0 (as expected — random walks have no
deterministic structure to exploit).

## T12 — `SugiharaMay_PredictabilityCurve`: chaos-vs-stochasticity diagnostic

Sugihara-May 1990 noticed that **deterministic chaos has a characteristic
signature**: prediction skill (correlation of forecast vs observed) decays
monotonically with forecast horizon for chaos but stays flat for additive
noise. The full curve `ρ(horizon)` is the diagnostic, not any single number.

```go
func PredictabilityCurve(series []float64, m, tau, k, horizonMax int,
    rho []float64)
```

**~60 LOC** on top of T11 (just sweep horizon and reduce). Saturation:
ρ(Lorenz, h=1) ≈ 0.99, ρ(Lorenz, h=20) ≈ 0.0 (decays to noise floor over
~3 Lyapunov times); ρ(AR(1) ϕ=0.95) ≈ 0.95 *flat* across horizon. The
shape itself is the answer — a chaos-vs-AR(1) classifier from one figure.

---

## Connective-tissue cost summary

| Synergy | LOC | Depends on | New math | Status |
|---|---|---|---|---|
| T1 TakensEmbed | 40 | — | none | reshape |
| T2 AMI / FirstMin | 110 | T1 | none | histogram+log |
| T3 FNN | 120 | T1 | none | NN+ratio test |
| T4 Lyapunov-Rosenstein | 150 | T1 | none | NN+regression |
| T5 Lyapunov-Kantz | 80 | T1, T4 | none | inner-loop swap |
| T6 D₂ Grassberger-Procaccia | 140 | T1 | none | log-log slope |
| T7 RQA metrics | 180 | RecurrencePlot | none | line histogram |
| T8 RP-from-series | 30 | T1, RP | none | glue |
| T9 PermutationEntropy | 80 | — | none | argsort+log |
| T10 0-1 ChaosTest | 140 | — | none | cumsum+corr |
| T11 LocalLinearForecast | 170 | T1, linalg.LinearRegression | none | k-NN+LR |
| T12 PredictabilityCurve | 60 | T11 | none | sweep horizon |
| **Total** | **~1300** | | | |

Eight of the twelve depend on T1. Nothing depends on `timeseries/garch` or
`timeseries/dcc` (those are stochastic volatility, orthogonal). Three are
**independent of `chaos/`'s existing API surface entirely** (T9, T10, the
permutation/cumsum primitives) and could ship in `prob/` alongside ARIMA
without touching `chaos/`.

## Suggested file layout (zero ambiguity)

```
chaos/
  embed.go         # T1 TakensEmbed, T8 RecurrencePlotFromSeries
  ami.go           # T2 AMI, FirstMinimumAMI
  fnn.go           # T3 FalseNearestNeighbors
  lyapunov_data.go # T4 LyapunovRosenstein, T5 LyapunovKantz
  dimension.go     # T6 CorrelationSum, CorrelationDimensionGP
  rqa.go           # T7 RecurrenceQuantification (extends existing analysis.go)
  permutation.go   # T9 PermutationEntropy
  zerone.go        # T10 ZeroOneChaosTest
  forecast.go      # T11 LocalLinearForecast, T12 PredictabilityCurve
```

All under `chaos/` because the *physics* is dynamical-systems theory; the
fact that the input happens to be a 1-D series is incidental, and forcing
this into `prob/timeseries.go` would push `prob/` into territory it doesn't
own (state-space reconstruction is geometric, not probabilistic).

## Cross-cuts not to confuse with this synergy

- **`signal/` × `chaos/`** — windowing the series before AMI is *not* a
  synergy; AMI assumes stationarity, windowing is a pre-step the caller
  already owns. Out of scope.
- **`prob/` × `chaos/`** — bootstrapping CIs around λ₁ via block bootstrap is
  a real synergy but it's a layer above this one; once T4 ships, that
  bootstrap is ~30 LOC against `prob/`'s existing resampling.
- **`timeseries/garch` × `chaos/`** — none. GARCH is stochastic volatility;
  chaos here is deterministic. The Lyapunov literature for GARCH residuals
  exists (Brock-Dechert-Scheinkman BDS test) but that is a separate seam,
  agent-future.

## Cheapest-first sequencing for one PR

1. **T1 TakensEmbed** (40 LOC) — unlocks 8 of 12.
2. **T9 PermutationEntropy** (80 LOC, independent) — ships a usable
   chaos-detector to `prob/anomaly.go` consumers in the same PR.
3. **T7 RQAMetrics** (180 LOC) — completes the existing `RecurrencePlot`
   from a stub matrix into the real Marwan-quantified product. This is the
   "we already half-shipped this" fix.

Total first-PR scope: ~300 LOC + tests. Round-trip Lorenz λ-recovery test
(via T4 in PR 2) is the cross-validation witness that ties this whole
seam back to the round-trip golden-file pattern reality already uses for
copula × autodiff.

## Risks and limits

- **Brute-force NN throughout.** All NN-touching primitives (T3, T4, T5,
  T11) are O(N²) and will hit a wall around N ≈ 10⁴. KD-tree is tracked
  in graph-numerics agent 081 (mentioned for spatial indices); do not block
  this synergy on it — Pulse and Horizon's actual series lengths are
  N ≤ 2000.
- **Theiler window choice (T4, T5, T11).** Standard heuristic is "≥ τ" or
  "≥ first-zero-of-autocorrelation"; expose as a parameter, default to τ,
  document.
- **Stationarity assumption.** AMI/FNN/Lyapunov assume the trajectory comes
  from one attractor. Detrending/changepoint-segmenting before embedding is
  the caller's responsibility (and `changepoint/` already ships Bocpd, so
  the pipeline exists — this is documentation, not code).
- **Permutation-entropy ties.** Equal-valued samples within a window need
  a tie-breaking rule (Bandt-Pompe used "left-first"; Cao 2004 added small
  jitter). Document and test both modes.

---

## Progress line appended to PROGRESS.md

`154 | 2026-05-08 | synergy-chaos-timeseries | chaos has only forward simulation and prob/timeseries has only linear forecasting; twelve composition primitives (~1300 LOC of glue, zero new math) — Takens embed, AMI/FNN delay+dim selection, Rosenstein/Kantz λ₁ from data, Grassberger-Procaccia D₂, RQA on the existing RP, permutation entropy, 0-1 test, local-linear forecasting — turn the seam into a full data-driven nonlinear-dynamics layer with T1 (40 LOC) and T9 (80 LOC) as the cheapest first-PR unlocks.`
