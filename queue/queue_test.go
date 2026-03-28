package queue

import (
	"math"
	"testing"

	"github.com/davly/reality/testutil"
)

const tol = 1e-10

// ---------------------------------------------------------------------------
// M/M/1 Tests
// ---------------------------------------------------------------------------

func TestMM1_Rho50(t *testing.T) {
	// Classic textbook: λ=1, μ=2, ρ=0.5
	Lq, Wq, L, W, rho := MM1(1, 2)
	assertClose(t, "rho", rho, 0.5)
	assertClose(t, "L", L, 1.0)           // ρ/(1-ρ) = 0.5/0.5 = 1
	assertClose(t, "W", W, 1.0)           // 1/(μ-λ) = 1/1 = 1
	assertClose(t, "Lq", Lq, 0.5)         // ρ²/(1-ρ) = 0.25/0.5 = 0.5
	assertClose(t, "Wq", Wq, 0.5)         // ρ/(μ-λ) = 0.5/1 = 0.5
}

func TestMM1_Rho80(t *testing.T) {
	Lq, Wq, L, W, rho := MM1(4, 5)
	assertClose(t, "rho", rho, 0.8)
	assertClose(t, "L", L, 4.0)
	assertClose(t, "W", W, 1.0)
	assertClose(t, "Lq", Lq, 3.2)
	assertClose(t, "Wq", Wq, 0.8)
}

func TestMM1_Rho20(t *testing.T) {
	Lq, Wq, L, W, rho := MM1(2, 10)
	assertClose(t, "rho", rho, 0.2)
	assertClose(t, "L", L, 0.25)
	assertClose(t, "W", W, 0.125)
	assertClose(t, "Lq", Lq, 0.05)
	assertClose(t, "Wq", Wq, 0.025)
}

func TestMM1_Rho90(t *testing.T) {
	Lq, Wq, L, W, rho := MM1(9, 10)
	assertClose(t, "rho", rho, 0.9)
	assertClose(t, "L", L, 9.0)
	assertClose(t, "W", W, 1.0)
	assertClose(t, "Lq", Lq, 8.1)
	assertClose(t, "Wq", Wq, 0.9)
}

func TestMM1_LittlesLawHolds(t *testing.T) {
	// Verify L = λ·W for MM1
	lambda := 3.0
	mu := 5.0
	_, _, L, W, _ := MM1(lambda, mu)
	assertClose(t, "Little L=λW", L, lambda*W)
}

func TestMM1_PanicOnUnstable(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("MM1 should panic when lambda >= mu")
		}
	}()
	MM1(5, 5) // λ = μ → unstable
}

func TestMM1_PanicOnEqual(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("MM1 should panic when lambda >= mu")
		}
	}()
	MM1(10, 5) // λ > μ → unstable
}

func TestMM1_PanicOnNegativeLambda(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("MM1 should panic on negative lambda")
		}
	}()
	MM1(-1, 5)
}

func TestMM1_PanicOnZeroMu(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("MM1 should panic on zero mu")
		}
	}()
	MM1(1, 0)
}

// Golden-file tests for MM1.
func TestMM1_Golden(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/queue/mm1.json")
	if gf.Function != "Queue.MM1" {
		t.Fatalf("golden file function = %q, want Queue.MM1", gf.Function)
	}

	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			lambda := testutil.InputFloat64(t, tc, "lambda")
			mu := testutil.InputFloat64(t, tc, "mu")
			Lq, Wq, L, W, rho := MM1(lambda, mu)
			got := []float64{Lq, Wq, L, W, rho}
			// expected is [Lq, Wq, L, W, rho]
			testutil.AssertFloat64Slice(t, tc, got)
			_ = got
		})
	}
}

// ---------------------------------------------------------------------------
// M/M/c Tests
// ---------------------------------------------------------------------------

func TestMMc_SingleServerMatchesMM1(t *testing.T) {
	// M/M/c with c=1 should equal M/M/1
	lambda, mu := 3.0, 5.0
	lq1, wq1, l1, w1, rho1 := MM1(lambda, mu)
	lqc, wqc, lc, wc, rhoc := MMc(lambda, mu, 1)
	assertClose(t, "Lq", lqc, lq1)
	assertClose(t, "Wq", wqc, wq1)
	assertClose(t, "L", lc, l1)
	assertClose(t, "W", wc, w1)
	assertClose(t, "rho", rhoc, rho1)
}

func TestMMc_TwoServers(t *testing.T) {
	// λ=3, μ=2, c=2 → ρ=0.75
	_, _, _, _, rho := MMc(3, 2, 2)
	assertClose(t, "rho", rho, 0.75)
}

func TestMMc_HighCapacity(t *testing.T) {
	// Many servers should give very low wait
	_, Wq, _, _, rho := MMc(5, 2, 10)
	if rho >= 0.5 {
		t.Errorf("expected low utilization, got rho=%v", rho)
	}
	if Wq > 0.001 {
		t.Errorf("expected very low wait time, got Wq=%v", Wq)
	}
}

func TestMMc_LittlesLawHolds(t *testing.T) {
	lambda := 6.0
	mu := 3.0
	c := 3
	_, _, L, W, _ := MMc(lambda, mu, c)
	assertClose(t, "Little L=λW", L, lambda*W)
}

func TestMMc_PanicOnOverload(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("MMc should panic when lambda >= c*mu")
		}
	}()
	MMc(10, 2, 3) // λ=10, c*μ=6 → unstable
}

func TestMMc_PanicOnZeroServers(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("MMc should panic when c < 1")
		}
	}()
	MMc(1, 2, 0)
}

// ---------------------------------------------------------------------------
// M/M/1/K Tests
// ---------------------------------------------------------------------------

func TestMM1K_FiniteCapacity(t *testing.T) {
	// K=10, λ=1, μ=2: should behave close to M/M/1 since loss is tiny
	Lq, Wq, L, W, rho, pLoss := MM1K(1, 2, 10)
	_ = Lq
	_ = Wq
	_ = L
	_ = W
	assertClose(t, "rho", rho, 0.5)
	if pLoss > 0.001 {
		t.Errorf("expected very low loss probability, got %v", pLoss)
	}
}

func TestMM1K_K1MatchesErlangB(t *testing.T) {
	// M/M/1/1 = M/M/1/1: only 1 spot in system. Loss = P(system full).
	// For K=1, P(loss) should be ρ/(1+ρ) for the finite queue.
	lambda, mu := 2.0, 3.0
	_, _, _, _, rho, pLoss := MM1K(lambda, mu, 1)
	expected := rho / (1.0 + rho) // P(1) for M/M/1/1
	assertClose(t, "pLoss K=1", pLoss, expected)
}

func TestMM1K_HighLoad(t *testing.T) {
	// λ > μ is allowed for finite capacity
	_, _, _, _, rho, pLoss := MM1K(10, 2, 5)
	if rho <= 1.0 {
		t.Errorf("expected rho > 1, got %v", rho)
	}
	if pLoss < 0.1 {
		t.Errorf("expected significant loss at high load, got %v", pLoss)
	}
}

func TestMM1K_EqualRates(t *testing.T) {
	// λ = μ (ρ = 1) — the special case with uniform distribution
	_, _, L, _, _, pLoss := MM1K(5, 5, 10)
	assertClose(t, "pLoss rho=1", pLoss, 1.0/11.0)
	assertClose(t, "L rho=1", L, 5.0)
}

func TestMM1K_PanicOnZeroK(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("MM1K should panic when K < 1")
		}
	}()
	MM1K(1, 2, 0)
}

func TestMM1K_LargK_ConvergesToMM1(t *testing.T) {
	// With very large K and ρ<1, should approach M/M/1
	lambda, mu := 1.0, 2.0
	_, _, Lk, Wk, _, pLoss := MM1K(lambda, mu, 1000)
	_, _, L1, W1, _ := MM1(lambda, mu)
	if pLoss > 1e-100 {
		t.Errorf("expected negligible loss for large K, got %v", pLoss)
	}
	if math.Abs(Lk-L1) > 1e-6 {
		t.Errorf("MM1K(K=1000) L=%v, MM1 L=%v", Lk, L1)
	}
	if math.Abs(Wk-W1) > 1e-6 {
		t.Errorf("MM1K(K=1000) W=%v, MM1 W=%v", Wk, W1)
	}
}

// ---------------------------------------------------------------------------
// Little's Law Tests
// ---------------------------------------------------------------------------

func TestLittlesLaw_Basic(t *testing.T) {
	W := LittlesLaw(10, 2)
	assertClose(t, "W", W, 5.0)
}

func TestLittlesLaw_Small(t *testing.T) {
	W := LittlesLaw(0.5, 0.1)
	assertClose(t, "W", W, 5.0)
}

func TestLittlesLaw_PanicOnZeroLambda(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("LittlesLaw should panic on zero lambda")
		}
	}()
	LittlesLaw(10, 0)
}

// ---------------------------------------------------------------------------
// Erlang B Tests
// ---------------------------------------------------------------------------

func TestErlangB_A1N1(t *testing.T) {
	// A=1, N=1: B = A/(1+A) = 0.5
	b := ErlangB(1.0, 1)
	assertClose(t, "ErlangB(1,1)", b, 0.5)
}

func TestErlangB_A1N2(t *testing.T) {
	// A=1, N=2: B = (1/2)/(1 + 1 + 1/2) = 0.5/2.5 = 0.2
	b := ErlangB(1.0, 2)
	assertClose(t, "ErlangB(1,2)", b, 0.2)
}

func TestErlangB_A10N15(t *testing.T) {
	// Known telecom value: ErlangB(10,15) ≈ 0.0365
	b := ErlangB(10.0, 15)
	assertClose(t, "ErlangB(10,15)", b, 0.036496945472371)
}

func TestErlangB_LargeN(t *testing.T) {
	// With many servers, blocking should be very low
	b := ErlangB(5.0, 20)
	if b > 1e-5 {
		t.Errorf("ErlangB(5,20) = %v, expected very low blocking", b)
	}
}

func TestErlangB_HighLoad(t *testing.T) {
	// A >> N: high blocking
	b := ErlangB(20.0, 5)
	if b < 0.5 {
		t.Errorf("ErlangB(20,5) = %v, expected high blocking", b)
	}
}

func TestErlangB_MonotonicInN(t *testing.T) {
	// More servers → less blocking
	A := 5.0
	prev := ErlangB(A, 1)
	for n := 2; n <= 10; n++ {
		cur := ErlangB(A, n)
		if cur >= prev {
			t.Errorf("ErlangB(5,%d) = %v >= ErlangB(5,%d) = %v", n, cur, n-1, prev)
		}
		prev = cur
	}
}

func TestErlangB_PanicOnZeroA(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("ErlangB should panic on non-positive A")
		}
	}()
	ErlangB(0, 5)
}

func TestErlangB_PanicOnZeroN(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("ErlangB should panic on N < 1")
		}
	}()
	ErlangB(5, 0)
}

// ---------------------------------------------------------------------------
// Erlang C Tests
// ---------------------------------------------------------------------------

func TestErlangC_A1N2(t *testing.T) {
	// A=1, N=2: known analytical result
	// ErlangC(1,2) = 1/3
	c := ErlangC(1.0, 2)
	assertClose(t, "ErlangC(1,2)", c, 1.0/3.0)
}

func TestErlangC_A05N1(t *testing.T) {
	// A=0.5, N=1: M/M/1 with ρ=0.5 → P(wait) = ρ = 0.5
	c := ErlangC(0.5, 1)
	assertClose(t, "ErlangC(0.5,1)", c, 0.5)
}

func TestErlangC_A01N1(t *testing.T) {
	// A=0.1, N=1: M/M/1 with ρ=0.1 → P(wait) = 0.1
	c := ErlangC(0.1, 1)
	assertClose(t, "ErlangC(0.1,1)", c, 0.1)
}

func TestErlangC_MonotonicInA(t *testing.T) {
	// Higher load → higher wait probability
	N := 5
	prev := ErlangC(0.5, N)
	for _, A := range []float64{1.0, 2.0, 3.0, 4.0, 4.5} {
		cur := ErlangC(A, N)
		if cur <= prev {
			t.Errorf("ErlangC(%v,%d) = %v <= ErlangC(prev) = %v", A, N, cur, prev)
		}
		prev = cur
	}
}

func TestErlangC_PanicOnUnstable(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("ErlangC should panic when A >= N")
		}
	}()
	ErlangC(5, 5) // A = N → unstable
}

func TestErlangC_PanicOnOverloaded(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("ErlangC should panic when A > N")
		}
	}()
	ErlangC(6, 5) // A > N → unstable
}

// Golden-file tests for Erlang C.
func TestErlangC_Golden(t *testing.T) {
	gf := testutil.LoadGolden(t, "../testdata/queue/erlang_c.json")
	if gf.Function != "Queue.ErlangC" {
		t.Fatalf("golden file function = %q, want Queue.ErlangC", gf.Function)
	}

	for _, tc := range gf.Cases {
		t.Run(tc.Description, func(t *testing.T) {
			A := testutil.InputFloat64(t, tc, "A")
			N := testutil.InputInt(t, tc, "N")
			got := ErlangC(A, N)
			testutil.AssertFloat64(t, tc, got)
		})
	}
}

// ---------------------------------------------------------------------------
// Erlang C Wait Time Tests
// ---------------------------------------------------------------------------

func TestErlangCWaitTime_Basic(t *testing.T) {
	// A=1, N=2, μ=1: E[Wq] = C(1,2) / (2·1·(1-0.5)) = (1/3)/1 = 1/3
	wq := ErlangCWaitTime(1.0, 2, 1.0)
	assertClose(t, "ErlangCWaitTime(1,2,1)", wq, 1.0/3.0)
}

func TestErlangCWaitTime_Higher_Mu(t *testing.T) {
	// Higher service rate → lower wait time
	wq1 := ErlangCWaitTime(2.0, 3, 1.0)
	wq2 := ErlangCWaitTime(2.0, 3, 2.0)
	if wq2 >= wq1 {
		t.Errorf("higher mu should reduce wait: wq1=%v, wq2=%v", wq1, wq2)
	}
}

func TestErlangCWaitTime_PanicOnZeroMu(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("ErlangCWaitTime should panic on zero mu")
		}
	}()
	ErlangCWaitTime(1.0, 2, 0)
}

// ---------------------------------------------------------------------------
// Jackson Network Tests
// ---------------------------------------------------------------------------

func TestJackson_TandemQueue(t *testing.T) {
	// Two nodes in series: arrivals → node0 → node1 → exit
	lambdaExt := []float64{2.0, 0.0}
	routing := [][]float64{
		{0, 1}, // node0 sends 100% to node1
		{0, 0}, // node1 sends 100% out
	}
	mu := []float64{4.0, 4.0}
	servers := []int{1, 1}

	throughput, utilization, queueLength := JacksonNetwork(lambdaExt, routing, mu, servers)

	assertClose(t, "throughput[0]", throughput[0], 2.0)
	assertClose(t, "throughput[1]", throughput[1], 2.0)
	assertClose(t, "utilization[0]", utilization[0], 0.5)
	assertClose(t, "utilization[1]", utilization[1], 0.5)
	// M/M/1 with ρ=0.5: L = 1
	assertClose(t, "queueLength[0]", queueLength[0], 1.0)
	assertClose(t, "queueLength[1]", queueLength[1], 1.0)
}

func TestJackson_FeedbackQueue(t *testing.T) {
	// Single node with 30% feedback: λ_ext=2, 30% loops back
	// Traffic equation: λ = 2 + 0.3·λ → λ = 2/0.7 ≈ 2.857
	lambdaExt := []float64{2.0}
	routing := [][]float64{{0.3}}
	mu := []float64{5.0}
	servers := []int{1}

	throughput, utilization, _ := JacksonNetwork(lambdaExt, routing, mu, servers)

	assertClose(t, "throughput[0]", throughput[0], 2.0/0.7)
	assertClose(t, "utilization[0]", utilization[0], (2.0/0.7)/5.0)
}

func TestJackson_ThreeNodeNetwork(t *testing.T) {
	// 3 nodes: external arrivals to node 0, splits to nodes 1 and 2
	lambdaExt := []float64{4.0, 0.0, 0.0}
	routing := [][]float64{
		{0.0, 0.5, 0.5}, // node0 splits equally
		{0.0, 0.0, 0.0}, // node1 exits
		{0.0, 0.0, 0.0}, // node2 exits
	}
	mu := []float64{10.0, 5.0, 5.0}
	servers := []int{1, 1, 1}

	throughput, _, _ := JacksonNetwork(lambdaExt, routing, mu, servers)

	assertClose(t, "throughput[0]", throughput[0], 4.0)
	assertClose(t, "throughput[1]", throughput[1], 2.0)
	assertClose(t, "throughput[2]", throughput[2], 2.0)
}

func TestJackson_MultiServer(t *testing.T) {
	// Single node with 2 servers
	lambdaExt := []float64{3.0}
	routing := [][]float64{{0}}
	mu := []float64{2.0}
	servers := []int{2}

	throughput, utilization, _ := JacksonNetwork(lambdaExt, routing, mu, servers)

	assertClose(t, "throughput[0]", throughput[0], 3.0)
	assertClose(t, "utilization[0]", utilization[0], 3.0/4.0)
}

func TestJackson_PanicOnEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("JacksonNetwork should panic on empty network")
		}
	}()
	JacksonNetwork([]float64{}, [][]float64{}, []float64{}, []int{})
}

func TestJackson_PanicOnMismatch(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("JacksonNetwork should panic on mismatched lengths")
		}
	}()
	JacksonNetwork([]float64{1}, [][]float64{{0}}, []float64{2, 3}, []int{1})
}

func TestJackson_PanicOnOverload(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("JacksonNetwork should panic on overloaded node")
		}
	}()
	// λ=5, μ=2, c=1 → ρ=2.5 > 1
	JacksonNetwork([]float64{5}, [][]float64{{0}}, []float64{2}, []int{1})
}

// ---------------------------------------------------------------------------
// BurstinessIndex Tests
// ---------------------------------------------------------------------------

func TestBurstinessIndex_Constant(t *testing.T) {
	// Constant inter-arrivals → C² = 0
	times := []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	b := BurstinessIndex(times)
	assertClose(t, "constant burstiness", b, 0.0)
}

func TestBurstinessIndex_Poisson(t *testing.T) {
	// For exponential inter-arrivals (Poisson process), C² = 1.
	// Generate a known exponential distribution with mean 1.
	// Var(Exp(1)) = 1, Mean(Exp(1)) = 1, so C² = 1/1 = 1.
	// Use large sample of known exponential quantiles.
	n := 10000
	times := make([]float64, n)
	for i := 0; i < n; i++ {
		// Use stratified sampling of the exponential CDF to get
		// near-perfect C²=1 without random number generation.
		u := (float64(i) + 0.5) / float64(n)
		times[i] = -math.Log(1 - u) // Inverse CDF of Exp(1)
	}
	b := BurstinessIndex(times)
	if math.Abs(b-1.0) > 0.05 {
		t.Errorf("Poisson burstiness = %v, expected ~1.0", b)
	}
}

func TestBurstinessIndex_Bursty(t *testing.T) {
	// Bursty: many short intervals with occasional long gaps.
	// This models a bursty arrival pattern where requests come in clusters.
	// 8 arrivals at 0.1s apart, then one 100s gap → highly overdispersed.
	times := []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 100, 0.1}
	b := BurstinessIndex(times)
	if b <= 1.0 {
		t.Errorf("bursty burstiness = %v, expected > 1", b)
	}
}

func TestBurstinessIndex_PanicOnTooFew(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("BurstinessIndex should panic on < 2 samples")
		}
	}()
	BurstinessIndex([]float64{1.0})
}

func TestBurstinessIndex_PanicOnZeroMean(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("BurstinessIndex should panic on zero mean")
		}
	}()
	BurstinessIndex([]float64{0, 0, 0})
}

// ---------------------------------------------------------------------------
// OfferedLoad Tests
// ---------------------------------------------------------------------------

func TestOfferedLoad_Basic(t *testing.T) {
	A := OfferedLoad(10, 0.5) // 10 arrivals/s, 0.5s service → 5 erlangs
	assertClose(t, "offered load", A, 5.0)
}

func TestOfferedLoad_One(t *testing.T) {
	A := OfferedLoad(1, 1) // 1 arrival/s, 1s service → 1 erlang
	assertClose(t, "offered load", A, 1.0)
}

func TestOfferedLoad_PanicOnZeroLambda(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("OfferedLoad should panic on zero lambda")
		}
	}()
	OfferedLoad(0, 1)
}

func TestOfferedLoad_PanicOnNegativeServiceTime(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("OfferedLoad should panic on negative service time")
		}
	}()
	OfferedLoad(1, -1)
}

// ---------------------------------------------------------------------------
// Cross-function consistency tests
// ---------------------------------------------------------------------------

func TestMMc_ConsistentWithErlangC(t *testing.T) {
	// The wait probability from MMc should match ErlangC
	lambda := 3.0
	mu := 2.0
	c := 3
	Lq, _, _, _, rho := MMc(lambda, mu, c)

	A := lambda / mu
	pWait := ErlangC(A, c)
	expectedLq := pWait * rho / (1 - rho)
	assertClose(t, "Lq from MMc matches ErlangC", Lq, expectedLq)
}

func TestOfferedLoad_ConsistentWithErlangB(t *testing.T) {
	// Offered load fed into ErlangB should give valid blocking probability
	lambda := 5.0
	serviceTime := 0.5
	A := OfferedLoad(lambda, serviceTime)
	assertClose(t, "offered load", A, 2.5)
	b := ErlangB(A, 3)
	if b < 0 || b > 1 {
		t.Errorf("ErlangB with offered load: blocking = %v, out of [0,1]", b)
	}
}

func TestMM1_ConsistentWithMMc_c1(t *testing.T) {
	// M/M/1 and M/M/c(c=1) must give identical results
	for _, rhoTest := range []float64{0.1, 0.3, 0.5, 0.7, 0.9} {
		mu := 10.0
		lambda := rhoTest * mu
		lq1, wq1, l1, w1, r1 := MM1(lambda, mu)
		lqc, wqc, lc, wc, rc := MMc(lambda, mu, 1)
		assertClose(t, "Lq", lq1, lqc)
		assertClose(t, "Wq", wq1, wqc)
		assertClose(t, "L", l1, lc)
		assertClose(t, "W", w1, wc)
		assertClose(t, "rho", r1, rc)
	}
}

func TestJackson_SingleNode_MatchesMM1(t *testing.T) {
	// Single-node Jackson with no routing = M/M/1
	lambda := 3.0
	mu := 5.0
	_, _, L1, _, _ := MM1(lambda, mu)

	throughput, _, queueLength := JacksonNetwork(
		[]float64{lambda},
		[][]float64{{0}},
		[]float64{mu},
		[]int{1},
	)

	assertClose(t, "throughput", throughput[0], lambda)
	assertClose(t, "queueLength", queueLength[0], L1)
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

func assertClose(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %v, want %v (diff %v)", name, got, want, math.Abs(got-want))
	}
}
