# API surface: reality

> Verified against `7e552be` (origin/master) on 2026-10-07 by the arch-doc sweep (run edc992).

Reality has three kinds of externally reachable interface:

1. the exported Go API of its 71 library packages (the main one);
2. the HTTP routes of one command, `cmd/reality-compute`;
3. one outbound event, posted by `conduit.Emit`.

It has no gRPC service, no MCP server other than the `/mcp/tools/` HTTP routes below, no CLI flags, and it consumes
no events.

## 1. HTTP: `cmd/reality-compute`

Routes are registered on a single `http.ServeMux` pattern, `/mcp/tools/` (`cmd/reality-compute/server.go:163`);
the handler tells the manifest path from tool paths by exact match (`cmd/reality-compute/server.go:165`).

| Method | Path | Handler | Auth | Responses |
|---|---|---|---|---|
| GET | `/mcp/tools/` | inline in `newMux` (`cmd/reality-compute/server.go:163-181`) | `X-Nexus-Service-Token` must equal `NEXUS_SERVICE_TOKEN`; unset means every call gets 401 (`cmd/reality-compute/server.go:337-346`) | 200 `{"tools":[{name, description, input_schema, approval_required}]}`; 401; 405 for non-GET |
| POST | `/mcp/tools/reality.conformal_interval` | `handleToolInvoke` → `handleConformalInterval` (`cmd/reality-compute/server.go:203`, `cmd/reality-compute/server.go:253`) | same token, plus a non-empty `X-User-Id` (400 if absent, `cmd/reality-compute/server.go:221-227`) | 200 envelope; 400 bad JSON, unknown field or more than 1,048,576 residuals; 413 body over 5 MiB; 422 engine domain error; 401; 405 for non-POST |
| POST | `/mcp/tools/<anything else>` | `handleToolInvoke` | same | 404 `unknown tool` (`cmd/reality-compute/server.go:233-237`) |

There is no health route and no other path: any other path gets the standard `http.ServeMux` 404 (or its redirect, for `/mcp/tools` without the trailing slash).

**`reality.conformal_interval`** (`cmd/reality-compute/server.go:72`, schema `cmd/reality-compute/server.go:143-152`):

- Input: `point_estimate` (number), `calibration_residuals` (array of non-negative numbers, oldest first), `alpha`
  (0 < alpha < 1), `half_life_steps` (integer >= 1). All four are required; unknown fields are rejected
  (`cmd/reality-compute/server.go:272`).
- Output (inside `content`): `lo`, `hi` from `conformal.AdaptiveInterval` (`prob/conformal/adaptive.go:137`),
  `coverage` from `conformal.MarginalCoverageBounds` (`prob/conformal/split.go:197`), `effective_n` from
  `conformal.EffectiveSampleSize` (`prob/conformal/adaptive.go:160`). An infinite bound is sent as
  `±math.MaxFloat64` and a NaN bound as 0 (`cmd/reality-compute/server.go:372-382`).
- Envelope: `{"content": <json>, "is_error": bool, "error_message": string}` (`cmd/reality-compute/server.go:104-108`).
- The manifest marks the tool `approval_required: false` (`cmd/reality-compute/server.go:194`).

## 2. Outbound event: `conduit`

| Function | Behaviour |
|---|---|
| `Emit(ctx, Event)` (`conduit/emit.go:66`) | fills `new_status` "OBSERVING", `project_id` "reality" and an RFC 3339 timestamp when empty; POSTs JSON to `CONDUIT_URL` (default `http://localhost:8200/v1/events`) from a goroutine with a 100 ms timeout; all errors are dropped. The caller's `ctx` is not used for the request |
| `EmitSampled(ctx, Event)` (`conduit/emit.go:114`) | emits on every `SampleRate`-th call and sets `observation_count` to the call counter; a rate <= 0 disables it |

Payload fields (`conduit/emit.go:49-60`): `situation_hash` (uint64), `project_id`, `domain`, `old_status`,
`new_status`, `dominance_rate`, `observation_count`, `event_type`, `payload`, `timestamp`. The comment says the tags
must match a struct in a separate repository (`conduit/emit.go:47-48`); that pairing cannot be checked from this
repository. No non-test code in this repository calls `Emit` or `EmitSampled`.

## 3. Go library API

The library API is the exported identifiers of the packages listed in `docs/CODE_MAP.md`. Conventions visible across
the code:

- Most functions take and return `float64` values or slices and return NaN, `false` or an `error` on input outside
  the documented domain; each doc comment states the formula's source, valid range and precision (for example
  `prob/distributions.go:78`).
- Several packages declare sentinel errors in an `errors.go` file (nine: `finance/taxlot/errors.go`, `info/lz/errors.go`,
  `info/mdl/errors.go`, `optim/hrp/errors.go`, `optim/transport/errors.go`, `prob/agreement/errors.go`,
  `prob/copula/errors.go`, `topology/persistent/errors.go`, `trust/errors.go`).
- Deprecated: `copula.StudentTQuantile`, in favour of `prob.StudentTQuantile` (`prob/copula/studentt.go:78`, `prob/trend.go:47`). This is
  the only `Deprecated:` marker in non-test code.
- Overflow-checked variant: `crypto.LCM` wraps on overflow; `crypto.LCMChecked` reports it (`crypto/prime.go:260-275`).
- Process-wide side effects: `forge/session40` panics in `init()` if its constants drift
  (`forge/session40/baked.go:136-138`); importing `conduit` reads `REALITY_CONDUIT_SAMPLE` at init
  (`conduit/emit.go:33`).
- `testutil` is exported but imports `testing`; it is test support, not runtime API (`testutil/golden.go:28-36`).
- Constants only: `constants` (no functions); `pkg/canonical` exports `IsCanonicalSource` and `CanonicalPrimitives`
  (`pkg/canonical/canonical.go:32`, `pkg/canonical/canonical.go:40`); `forge` exports its verdict thresholds
  (`forge/convergence.go:20-37`).

### Exported functions and methods, by package

Generated from the non-test files at `7e552be` with the regex
`^func (?:\((?:\w+\s+)?\*?(Type)(?:\[...\])?\) )?([A-Z]\w*)` (Python, over `git ls-files '*.go'` minus test
files). `Type.Method` is a method; a lower-case type (for example `dijkstraHeap.Len`) is an exported method on an
unexported type, present to satisfy an interface and not callable by name from outside. The count in brackets is
declarations, so a name declared on two types counts twice. Exported types, constants and variables are not listed.
962 declarations in 70 packages (`constants` and `cmd/reality-compute` export no functions).

- **`acoustics`** (9): AWeighting, DecibelFromIntensity, DecibelSPL, DopplerShift, ResonantFrequency, SabineRT60, SoundIntensity, SoundSpeed, WaveLength
- **`audio`** (23): ApplyFilterbank, BaselineStdDev, BestMatch, FingerprintMahalanobis, FingerprintVariance, FrameMFCC, HzToMel, LogMelEnergies, MFCC, MelFilterbank, MelToHz, MergeFingerprints, NewDegradationTracker, NewFingerprint, PowerSpectrum, PushObservation, PushWindowOnly, ResetBaseline, ResetWindow, UpdateBaseline, UpdateFingerprint, WindowMean, ZScore
- **`audio/beat`** (2): DefaultOptions, Track
- **`audio/cqt`** (7): BinFrequencies, BinFrequency, CQT, Magnitude, PeakBin, QualityFactor, WindowLength
- **`audio/idbench`** (1): Evaluate
- **`audio/onset`** (7): ComplexDomainOnset, EnergyOnset, PickPeaks, PickPeaksAdaptive, SpectralFluxOnset, SpectralFluxStrength, SuperFlux
- **`audio/pitch`** (4): AutocorrelationPitch, McLeodPitchMethod, SubharmonicSummation, Yin
- **`audio/segmentation`** (6): FilterByMinDuration, MergeCloseSegments, Segment.Duration, SegmentByEnergy, SegmentByOnsetOffset, SegmentWithMinSilence
- **`audio/separation`** (13): Decompose, EstimateNoiseSpectrum, FastICA, FrameEnergy, FrobeniusError, IsVoiced, IsVoicedAdaptive, Reconstruct, SubtractSpectrum, SubtractSpectrumInto, WienerFilter, WienerFilterInto, ZeroCrossingRate
- **`audio/spectrogram`** (15): Compute, HalfSpectrum, Inferno, Inverse, LogMagnitude, LogMelSpectrogram, Magma, Magnitude, MelSpectrogram, NormaliseTo01, Plasma, PowerSpectrum, ToHeatmap, ToHeatmapWith, Viridis
- **`audio/tempo`** (5): Autocorrelation, BpmToLag, DefaultOptions, Estimate, LagToBpm
- **`audio/vibration`** (2): FundamentalHz, HarmonicEnergyRatio
- **`autodiff`** (21): Add, AddConst, Cos, Div, Dot, Exp, Log, MeanSquaredError, Mul, MulConst, Neg, NewTape, Pow, Sin, Sqrt, Sub, Sum, Tanh, Tape.Backward, Tape.Constant, Tape.Var
- **`calculus`** (6): GaussLegendre, MonteCarloIntegrate, NumericalDerivative, NumericalGradient, SimpsonsRule, TrapezoidalRule
- **`causal`** (3): AdjustedOutcome, BackdoorATE, BackdoorATEWithRefutation
- **`changepoint`** (27): BettingEValue, Bocpd.ChangePointProbability, Bocpd.ChangePointProbabilityWithin, Bocpd.CurrentRegimeMean, Bocpd.CurrentRegimeVariance, Bocpd.ExpectedRunLength, Bocpd.MapRunLength, Bocpd.RunLengthPosterior, Bocpd.Step, Bocpd.Update, DefaultConfig, DefaultNigPrior, EDetector.FireTime, EDetector.Fired, EDetector.LogValue, EDetector.Step, EDetector.Update, EDetector.Value, EProcess.Fired, EProcess.LogValue, EProcess.Step, EProcess.Update, EProcess.Value, New, NewEDetector, NewEProcess, NigPrior.Validate
- **`chaos`** (13): BifurcationDiagram, EulerStep, GameOfLife, LogisticMap, LorenzSystem, LotkaVolterra, LyapunovExponent, RK4Step, RecurrencePlot, RosslerSystem, SIRModel, SolveODE, VanDerPol
- **`color`** (13): BlackbodyToXYZ, BradfordAdapt, DeltaE2000, DeltaE76, HSVToRGB, LabToXYZ, LinearRGBToXYZ, LinearToSRGB, RGBToHSV, SRGBToLinear, ToneMapReinhard, XYZToLab, XYZToLinearRGB
- **`combinatorics`** (23): BarrierOptionReflection, BellNumber, BinomialCoeff, BuildBlocked, CanonicalizeExclusions, CatalanNumber, ConstrainedDerangement, CountDyckPaths, DerangementCount, Factorial, FibonacciNumber, GenerateCombinations, GeneratePermutations, IntegerPartitions, IsValidAssignment, NextPermutation, Permutations, PriceAmericanBinomialTree, PriceEuropeanBinomialTree, RandomSubset, SeedFromCanonical, StirlingFirst, StirlingSecond
- **`compression`** (12): ConditionalEntropy, CrossEntropy, DeltaDecode, DeltaEncode, JointEntropy, KLDivergence, MutualInformation, RunLengthDecode, RunLengthEncode, ScalarDequantize, ScalarQuantize, ShannonEntropy
- **`conduit`** (2): Emit, EmitSampled
- **`control`** (10): ComplementaryFilter, HighPassFilter, LowPassFilter, NewPID, PIDController.Reset, PIDController.Update, RateLimiter, TransferFunction.Evaluate, TransferFunction.IsStable, TransferFunction.Poles
- **`crypto`** (26): ChineseRemainder, ConsistentHash, ExtendedGCD, FNV1a32, FNV1a64, GCD, IsPrime, LCM, LCMChecked, MersenneTwister.Float64, MersenneTwister.Uint64, MillerRabin, ModInverse, ModPow, MurmurHash3_32, NewMersenneTwister, NewPCG, NewXoshiro256, NextPrime, PCG.Float64, PCG.Uint32, PrimeFactors, SituationHashWithStructure, StructuralDescriptor, Xoshiro256.Float64, Xoshiro256.Uint64
- **`em`** (10): CapacitorEnergy, CoulombForce, ElectricField, InductorEnergy, OhmsLaw, PowerElectric, RCTimeConstant, ResistorsInParallel, ResistorsInSeries, ResonantFrequencyLC
- **`evidence`** (5): Grade.String, GradeScore, SampleBackingFactor, Score, Summarize
- **`fairness`** (6): AdverseImpact, AdverseImpactRatio, PassesFourFifths, PassesFourFifthsExact, SelectionRate, WilsonScoreInterval
- **`finance/taxlot`** (12): ApplyWashSale, Classify, D, Date.AddDays, Date.AddYears, Date.After, Date.Before, Date.Days, Date.DaysUntil, Date.Equal, IsLongTerm, Term.String
- **`fluids`** (10): BernoulliPressure, DarcyWeisbach, DragForce, LiftForce, MassFlowRate, PipeFlowFriction, ReynoldsNumber, StokesLaw, TerminalVelocity, VolumetricFlowRate
- **`forge`** (3): Decide, DecideExact, Verdict.String
- **`forge/session40`** (8): AssertBaked, CanonicalDivergence.String, CanonicalPrimitiveSet, CanonicalPrimitives, IsCanonicalSource, Register, Registered, RegisteredByPrimitive
- **`gametheory`** (26): BanzhafIndex, ContinuousGrowthRate, EnsembleMeanReturn, EpsilonGreedy, EpsilonGreedyFromArms, ErgodicityGap, ErgodicityShrinkageExp, ErgodicityShrinkageReciprocal, FractionalKelly, GaleShapley, IsStableMatching, KellyContinuous, KellyContinuousMulti, KellyFraction, KellyFractionMultiple, KellyGrowthRate, Minimax, NashEquilibrium2x2, PriorShrink, ShapleyValue, ShapleyValueWeightedVoting, ThompsonFromArmsBernoulli, ThompsonSampling, TimeAverageGrowthRate, UCB1, UCB1FromArms
- **`geometry`** (27): BezierCubic, BezierCubic3D, CatmullRom, ConvexHull2D, LinearInterpolate, PointInTriangle2D, QuatConjugate, QuatDot, QuatFromAxisAngle, QuatFromEuler, QuatIdentity, QuatMul, QuatNormalize, QuatRotateVec, QuatSlerp, QuatToAxisAngle, SDFBox, SDFCapsule, SDFIntersection, SDFSmoothIntersection, SDFSmoothSubtraction, SDFSmoothUnion, SDFSphere, SDFSubtraction, SDFTorus, SDFUnion, TriangleArea2D
- **`graph`** (51): ADMG.IdentifyConditionalEffect, ADMG.IdentifyEffect, ADMG.IdentifyEffectWithWitness, AStar, AdjacencyList, BFSDownstream, BFSReachable, BackdoorAdjustmentSet, BellmanFord, BetweennessCentrality, ConnectedComponents, DAGDepth, DSeparated, DegreeCentrality, Dijkstra, DiscreteSCM.InterventionalDistribution, DiscreteSCM.ObservationalJoint, EdgeFraction, EigenvectorCentrality, FloydWarshall, Hedge.String, InDegree, KruskalMST, Leaves, LouvainCommunities, MaxFlow, NewADMG, NodeImportance, Nodes, PageRank, PrimMST, RandomSCM, ReachableLeaves, Roots, StronglyConnected, TopologicalSort, dijkstraHeap.Len, dijkstraHeap.Less, dijkstraHeap.Pop, dijkstraHeap.Push, dijkstraHeap.Swap, exprFactor.String, exprMarginal.String, exprP.String, exprProduct.String, idError.Error, minIntHeap.Len, minIntHeap.Less, minIntHeap.Pop, minIntHeap.Push, minIntHeap.Swap
- **`info/lz`** (7): ComplexityFromReturns, CrossComplexity, LempelZivComplexity, NormalizedLZDistance, RollingComplexity, SymbolizeByQuantile, SymbolizeByThreshold
- **`info/mdl`** (11): AICShape, BICShape, BernoulliCodeLength, GaussianCodeLength, ModelCodeLength, NMLBernoulli, NMLMultinomial, SelectMDL, SelectMDLWithMargin, UniversalIntegerCodeLength, UniversalIntegerCodeLengthBits
- **`infogeo`** (17): Bregman, ChiSquared, GaussianKernel, GeneralisedKL, Hellinger, ItakuraSaito, JS, KL, LaplacianKernel, MMD2Biased, MMD2Unbiased, MahalanobisSquared, MedianHeuristicBandwidth, Renyi, ReverseKL, SquaredEuclidean, TotalVariation
- **`linalg`** (39): CholeskyDecompose, CholeskySolve, Clamp, CleanCorrelation, CosineSimilarity, Covariance, CovarianceMatrix, CrossProduct, Determinant, DimensionWeightedDistance, DotProduct, EncodingDistance, Identity, Inverse, JamesSteinShrink, L1Norm, L2Norm, L2Normalize, LInfNorm, LUDecompose, LUSolve, LedoitWolfShrinkageConstantCorr, LedoitWolfShrinkageIdentity, MarchenkoPasturBounds, MatAdd, MatMul, MatScale, MatSub, MatTranspose, MatVecMul, PCA, PearsonCorrelation, QRAlgorithm, SpearmanCorrelation, StructuralOverlap, Trace, VectorAdd, VectorScale, VectorSub
- **`moments`** (19): Merge, NewWelford, NewWelfordVec, Welford.Count, Welford.M2, Welford.Mean, Welford.Merge, Welford.PopVariance, Welford.StdDev, Welford.Update, Welford.Variance, Welford.ZScore, WelfordVec.Count, WelfordVec.Dim, WelfordVec.Mean, WelfordVec.PopVariance, WelfordVec.StdDev, WelfordVec.Update, WelfordVec.Variance
- **`optim`** (14): BisectionMethod, CubicSplineNatural, GeneticAlgorithm, GoldenSectionSearch, GradientDescent, GradientDescentValidated, InteriorPoint, LBFGS, LBFGSValidated, LinearInterpolate, LinearInterpolateRoot, NewtonRaphson, SimplexMethod, SimulatedAnnealing
- **`optim/hrp`** (5): CorrelationDistance, HRPWeights, QuasiDiagonalize, RecursiveBisection, SingleLinkage
- **`optim/portfolio`** (8): BlackLittermanPosterior, BlackLittermanPosteriorCovariance, ContinuousKellyWeights, HeLittermanOmega, ImpliedEquilibriumReturns, MeanVarianceWeights, MeanVarianceWeightsLongOnly, ProjectSimplex
- **`optim/proximal`** (10): Admm, Fbs, ProxBox, ProxL0, ProxL1, ProxL2Ball, ProxLinear, ProxNonNeg, ProxSimplex, ProxSquaredL2
- **`optim/transport`** (6): IQRNormalise, MinPairwiseWasserstein1D, PairwiseWasserstein1D, Sinkhorn, Wasserstein1D, Wasserstein1DDetailed
- **`orbital`** (8): EscapeVelocity, HillSphere, HohmannTransfer, KeplerOrbit, OrbitalPeriod, OrbitalVelocity, SynodicPeriod, TrueAnomalyFromMean
- **`physics`** (30): BeamDeflection, BeerLambertLaw, CarnotEfficiency, CoffinManson, CompositeMixture, CreepArrhenius, ElasticCollision, EulerBuckling, FourierHeatConduction, FresnelReflectance, GravitationalForce, GriffithCriterion, HeatEquation1DStep, HookesLaw, IdealGas, KineticEnergy, NewtonCooling, NewtonSecondLaw, OrbitalVelocity, ParisLaw, Pendulum, PotentialEnergy, ProjectilePosition, SnellRefraction, SpringForce, StefanBoltzmann, StressIntensityFactor, ThermalExpansion, TrescaStress, VonMisesStress
- **`pkg/canonical`** (1): CanonicalPrimitives
- **`prob`** (87): ARIMA, BayesianUpdate, BayesianUpdateChain, BenjaminiHochberg, BetaCDF, BetaDist.CDF, BetaDist.PDF, BetaPDF, BinomialCDF, BinomialPMF, BrierScore, BrierScoreBatch, CatoniMean, ChiSquaredTest, ClampProbability, ConfidenceFromPValue, DeflatedSharpeRatio, EMA, Erfc, ExpectedCalibrationError, ExpectedMaxSharpe, ExponentialCDF, ExponentialDist.CDF, ExponentialDist.PDF, ExponentialPDF, ExponentialQuantile, ExponentialSmoothing, FisherExactTest, GammaCDF, GammaPDF, HoltLinear, IsotonicRegression, JeffreysConfidence, JeffreysKLDivergence, KLDivergenceNumerical, LinearRegression, LogGamma, LogLoss, LogLossBatch, LogOddsPool, LogOddsToProb, MannWhitneyU, MarkovSimulate, MarkovSteadyState, MaximumCalibrationError, Median, MedianOfMeans, MedianOfMeansForConfidence, MinTrackRecordLength, NewBetaDist, NewExponentialDist, NewNormalDist, NewUniformDist, NewVennAbers, NormalCDF, NormalDist.CDF, NormalDist.PDF, NormalPDF, NormalQuantile, Percentile, PoissonCDF, PoissonPMF, ProbToLogOdds, ProbabilisticSharpeRatio, ProportionBayesFactor10, QualityWeightedDominance, Quantile, RegularizedBetaInc, ReliabilityDiagram, SimpleAverage, StudentTQuantile, TTestOneSample, TTestTwoSample, ThreeWayVerdict, TrendCrossing, TrendPredictionInterval, TrimmedMean, UniformCDF, UniformDist.CDF, UniformDist.PDF, UniformPDF, VennAbers.Predict, VennAbers.PredictPoint, VennAbersPoint, WeightedAverage, WilsonConfidenceInterval, WilsonScoreInterval
- **`prob/agreement`** (8): CohenKappa, DiscordantCounts, FleissKappa, KrippendorffAlpha, McNemarExact, McNemarMidP, PairedPermutationTest, WeightedKappa
- **`prob/conformal`** (28): ACI.Level, ACI.RawLevel, ACI.Step, ACI.Update, ACIStream.Level, ACIStream.Observe, ACIStream.Step, ACIStream.Threshold, AbsResidual.Name, AbsResidual.Score, AdaptiveInterval, AdaptiveQuantile, CqrConformityScore, CqrInterval, EffectiveSampleSize, LogResidual.Name, LogResidual.Score, MarginalCoverageBounds, MondrianInterval, MondrianQuantile, NewACI, NewACIStream, NormalizedResidual.Name, NormalizedResidual.Score, ScoreAll, SplitInterval, SplitIntervalSignedResiduals, SplitQuantile
- **`prob/copula`** (33): BivariateNormalCDF, BivariateTCDF, ClaytonCopulaCDFFn, ClaytonHFn, ClaytonLogPDFFn, ClaytonLowerTailDependence, ClaytonPDFFn, DVine.Dim, DVine.EdgeCount, DVine.HFunctionPass, DVine.LogPDF, EmpiricalCdf, GaussianCopulaCDF, GaussianCopulaCDFFn, GaussianCopulaCorrelationFromTau, GumbelCopulaCDFFn, GumbelHFn, GumbelLogPDFFn, GumbelPDFFn, GumbelUpperTailDependence, HFnForFamily, KendallTau, LogPDFFnForFamily, NewDVine, SklarJointFromMarginals, StudentTCDF, StudentTCopulaCDF, StudentTCopulaCDFFn, StudentTQuantile, ThetaFromKendallTau, TrivariateNormalCDF, TrivariateTCDF, VineEdge.Validate
- **`prob/evt`** (24): EvtES, EvtReturnLevel, EvtReturnPeriod, EvtVaR, Exceedances, FitGEVLMoments, FitGEVMLE, FitGPDMLE, FitGPDPWM, FitPOT, GEVCDF, GEVLogLik, GEVPDF, GEVParams.Kind, GEVQuantile, GEVReturnLevel, GPDCDF, GPDLogLik, GPDPDF, GPDQuantile, HillAlpha, HillTailIndex, LMoments3, ThresholdAtRate
- **`prob/hmm`** (9): Backward, BackwardLog, BaumWelch, Forward, ForwardLog, Model.Validate, Posterior, Viterbi, ViterbiLog
- **`prob/numclaim`** (3): ClaimConsistency, DefaultOptions, NumericEquivalent
- **`prob/risk`** (17): AnnualizeReturn, AnnualizeVolatility, Beta, CalmarRatio, CornishFisherVaR, DownsideDeviationFullSample, DownsideDeviationNegativesOnly, HistoricalCVaR, HistoricalVaR, InformationRatio, MaxDrawdownFromPrices, MaxDrawdownFromReturns, OmegaRatio, ParametricCVaR, ParametricVaR, SortinoRatioFullSample, SortinoRatioNegativesOnly
- **`queue`** (10): BurstinessIndex, ErlangB, ErlangC, ErlangCWaitTime, JacksonNetwork, LittlesLaw, MM1, MM1K, MMc, OfferedLoad
- **`reliability`** (8): AvailabilityFromMTBF, BirnbaumImportance, BirnbaumImportances, KofN, LimitingDependency, ParallelAvailability, SeriesAvailability, SystemAvailability
- **`retrymath`** (19): AmplificationFactor, CappedExponentialTerm, DecorrelatedJitter, DecorrelatedUncappedMean, DelayQuantile, DelayVariance, EffectiveArrivalRate, EffectiveUtilization, EqualJitter, ExpectedAttempts, ExpectedDelay, ExpectedReduceOnlyDelay, ExpectedSymmetricDelay, ExpectedTotalDelay, FullJitter, MultiplicativeJitter, ReduceOnlyJitter, StableUnderRetries, SymmetricJitter
- **`sequence`** (16): DamerauLevenshtein, DiffTokens, HammingDistance, JaroWinkler, LevenshteinDistance, LongestCommonSubsequence, LongestCommonSubstring, NGramDiceCoefficient, NGramSimilarity, NGrams, NeedlemanWunsch, Shingling, SmithWaterman, Soundex, TokenSetRatio, WordNGrams
- **`setsim`** (5): MapKeyJaccard, SetDice, SetJaccard, SetOverlapCoefficient, SetOverlapCounts
- **`signal`** (12): ApplyWindow, BlackmanWindow, Convolve, ExponentialMovingAverage, FFT, FFTFrequencies, HammingWindow, HannWindow, IFFT, MedianFilter, MovingAverage, PowerSpectrum
- **`slo`** (12): BudgetFractionConsumed, BurnRate, BurnRateFromErrorRate, DetectionTime, ErrorBudget, Policy.Evaluate, RecommendedWindows, ResetTime, ShortWindow, ThresholdBurnRate, TimeToExhaustion, Window.Fires
- **`spc`** (16): CUSUMARL, CUSUMARLOneSided, CUSUMThresholdForARL, Classify, ClassifyCpk, Compute, Cp, Cpk, DPMO, EWMAARL, EWMAARLGrid, EWMALimits, EWMASteadyStateSigma, OverallSigma, PooledWithinSigma, Rating.String
- **`testutil`** (11): AssertDeterministic, AssertFloat64, AssertFloat64Slice, DistinctOutputs, FloatBits, InputFloat64, InputFloat64Slice, InputInt, LoadGolden, SortedMapBits, Within
- **`timeseries`** (8): EWMoments.Alpha, EWMoments.Count, EWMoments.Mean, EWMoments.StdDev, EWMoments.Update, EWMoments.Variance, EWMoments.ZScore, NewEWMoments
- **`timeseries/dcc`** (6): CorrelationFromQ, EngleDefault, Params.FilterSeries, Params.Update, Params.Validate, SampleQbar
- **`timeseries/garch`** (6): Fit, Model.Filter, Model.ForecastVariance, Model.LogLikelihood, Model.Simulate, Model.Validate
- **`timeseries/statespace`** (7): Filter, KalmanPredict, KalmanUpdate, LocalLevelFilter, LocalLevelSmooth, LocalLevelSteadyState, RTSSmooth
- **`topology/persistent`** (8): Bar.IsEssential, Bar.Persistence, BottleneckDistance, ComputeBarcode, Filtration.Len, Simplex.Dim, Simplex.Equal, VietorisRipsComplex
- **`trust`** (18): AveragingFusion, CumulativeFusion, DempsterCombine, FuseAll, MassFunction.Belief, MassFunction.Plausibility, NewMassFunction, NewOpinion, Opinion.Discount, Opinion.Evidence, Opinion.IsDogmatic, Opinion.IsVacuous, Opinion.ProbabilityProjection, Opinion.ToBinaryMass, Opinion.Validate, OpinionFromBinaryMass, OpinionFromEvidence, YagerCombine
- **`zkmark`** (8): Halo2Prover.Algorithm, Halo2Prover.Prove, HonestProver.Algorithm, HonestProver.Prove, MarkVerifier.VerifyProof, NewHalo2Prover, NewHonestProver, NewMarkVerifier
