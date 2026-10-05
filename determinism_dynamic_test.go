package reality_test

// Determinism ratchet, dynamic half.
//
// Each function below touches maps internally and is classified
// order-insensitive or sorted-after in testdata/determinism/map_ranges.json.
// Here that classification is executed: 200 identical calls in one process
// must give one bit-exact output. Functions with a measured defect
// (known-nondeterministic) are not asserted here; TestKnownNondeterministic
// only reports their current state so a fix can be noticed and reclassified.

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/davly/reality/graph"
	"github.com/davly/reality/prob/agreement"
	"github.com/davly/reality/prob/conformal"
	"github.com/davly/reality/reliability"
	"github.com/davly/reality/sequence"
	"github.com/davly/reality/testutil"
	"github.com/davly/reality/trust"
)

const determinismCalls = 200

func randomIntAdjacency(rng *rand.Rand, n int, p float64) graph.IntAdjacency {
	adj := graph.IntAdjacency{}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i != j && rng.Float64() < p {
				adj[i] = append(adj[i], j)
			}
		}
	}
	return adj
}

func randomDAGEdges(rng *rand.Rand, n int, p float64) []graph.Edge {
	var edges []graph.Edge
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if rng.Float64() < p {
				edges = append(edges, graph.Edge{fmt.Sprintf("n%02d", i), fmt.Sprintf("n%02d", j)})
			}
		}
	}
	return edges
}

func reliabilityFixture() ([]graph.Edge, map[string]float64) {
	edges := []graph.Edge{
		{"app", "db"}, {"app", "cache"}, {"app", "auth"}, {"auth", "db"},
		{"cache", "disk"}, {"db", "disk"}, {"app", "queue"}, {"queue", "disk"},
	}
	avail := map[string]float64{
		"app": 0.9993, "db": 0.99917, "cache": 0.98731, "auth": 0.99977,
		"disk": 0.999913, "queue": 0.99459,
	}
	return edges, avail
}

func massFunction(t *testing.T, frame int, m map[uint]float64) trust.MassFunction {
	t.Helper()
	mf, err := trust.NewMassFunction(frame, m)
	if err != nil {
		t.Fatalf("NewMassFunction: %v", err)
	}
	return mf
}

func TestDeterministic_MapTouchingFunctions(t *testing.T) {
	rng := rand.New(rand.NewSource(20261005))
	adj := randomIntAdjacency(rng, 40, 0.08)
	dag := randomDAGEdges(rng, 30, 0.12)
	relEdges, avail := reliabilityFixture()

	scores := make([]float64, 300)
	strata := make([]int, 300)
	for i := range scores {
		scores[i] = rng.NormFloat64()
		strata[i] = rng.Intn(5)
	}

	ratings := make([][]float64, 4)
	for r := range ratings {
		ratings[r] = make([]float64, 30)
		for u := range ratings[r] {
			ratings[r][u] = float64(rng.Intn(5))
		}
	}

	m1 := massFunction(t, 4, map[uint]float64{1: 0.1, 2: 0.2, 3: 0.15, 4: 0.05, 5: 0.1, 6: 0.1, 7: 0.1, 9: 0.05, 15: 0.15})
	m2 := massFunction(t, 4, map[uint]float64{1: 0.3, 3: 0.2, 6: 0.25, 12: 0.1, 15: 0.15})

	cases := []struct {
		name string
		f    func() string
	}{
		{"graph.DegreeCentrality", func() string { return testutil.FloatBits(graph.DegreeCentrality(adj, 40)...) }},
		{"graph.ConnectedComponents", func() string { return fmt.Sprint(graph.ConnectedComponents(adj, 40)) }},
		{"graph.DAGDepth", func() string { return fmt.Sprint(graph.DAGDepth(dag)) }},
		{"graph.BackdoorAdjustmentSet", func() string {
			z, ok := graph.BackdoorAdjustmentSet(dag, "n03", "n20")
			return fmt.Sprint(z, ok)
		}},
		{"graph.NodeImportance", func() string { return testutil.SortedMapBits(graph.NodeImportance(dag)) }},
		{"conformal.MondrianQuantile", func() string {
			q, err := conformal.MondrianQuantile(scores, strata, 0.1)
			return testutil.SortedMapBits(q) + fmt.Sprint(err)
		}},
		{"sequence.NGramSimilarity", func() string {
			return testutil.FloatBits(sequence.NGramSimilarity("determinism ratchet", "deterministic ratchets", 3))
		}},
		{"sequence.NGramDiceCoefficient", func() string {
			return testutil.FloatBits(sequence.NGramDiceCoefficient("determinism ratchet", "deterministic ratchets", 2))
		}},
		{"reliability.LimitingDependency", func() string {
			node, a := reliability.LimitingDependency(relEdges, avail, "app")
			return node + testutil.FloatBits(a)
		}},
		{"agreement.KrippendorffAlpha", func() string {
			a, err := agreement.KrippendorffAlpha(ratings, agreement.Nominal)
			return testutil.FloatBits(a) + fmt.Sprint(err)
		}},
		{"trust.DempsterCombine", func() string {
			c, k, err := trust.DempsterCombine(m1, m2)
			return testutil.SortedMapBits(c.Masses) + testutil.FloatBits(k) + fmt.Sprint(err)
		}},
		{"trust.YagerCombine", func() string {
			c, k, err := trust.YagerCombine(m1, m2)
			return testutil.SortedMapBits(c.Masses) + testutil.FloatBits(k) + fmt.Sprint(err)
		}},
	}
	for _, c := range cases {
		testutil.AssertDeterministic(t, c.name, determinismCalls, c.f)
	}
}

// TestKnownNondeterministic reports, without failing, how the measured
// defects behave today. When one reports a single distinct output, look at it:
// if it was fixed, reclassify its sites in testdata/determinism/map_ranges.json
// and move it into TestDeterministic_MapTouchingFunctions.
func TestKnownNondeterministic(t *testing.T) {
	ring8 := graph.IntAdjacency{}
	for i := 0; i < 8; i++ {
		ring8[i] = []int{(i + 1) % 8}
	}
	rng := rand.New(rand.NewSource(1))
	dag := randomDAGEdges(rng, 30, 0.12)
	relEdges, avail := reliabilityFixture()
	m1 := massFunction(t, 4, map[uint]float64{1: 0.1, 2: 0.2, 3: 0.15, 4: 0.05, 5: 0.1, 6: 0.1, 7: 0.1, 9: 0.05, 15: 0.15})

	report := []struct {
		name string
		f    func() string
	}{
		{"graph.LouvainCommunities (ring of 8)", func() string { return fmt.Sprint(graph.LouvainCommunities(ring8, nil, 8)) }},
		{"graph.Roots", func() string { return strings.Join(graph.Roots(dag), ",") }},
		{"trust.MassFunction.Belief(Theta)", func() string { return testutil.FloatBits(m1.Belief(15)) }},
		{"trust.MassFunction.Plausibility", func() string { return testutil.FloatBits(m1.Plausibility(3)) }},
		{"reliability.SystemAvailability", func() string {
			return testutil.FloatBits(reliability.SystemAvailability(relEdges, avail, "app"))
		}},
		{"reliability.BirnbaumImportances", func() string {
			return testutil.SortedMapBits(reliability.BirnbaumImportances(relEdges, avail, "app"))
		}},
	}
	for _, r := range report {
		n := testutil.DistinctOutputs(determinismCalls, r.f)
		t.Logf("%-40s %3d distinct output(s) over %d identical calls", r.name, n, determinismCalls)
	}
}

// TestMaxFlowFractionalCapacities probes the one graph function whose
// map-order dependence is unreviewed: with fractional capacities, different
// augmenting-path orders could round differently. It reports, without failing.
func TestMaxFlowFractionalCapacities(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	adj := randomIntAdjacency(rng, 24, 0.2)
	capacity := map[[2]int]float64{}
	for u, vs := range adj {
		for _, v := range vs {
			capacity[[2]int{u, v}] = rng.Float64() * 10
		}
	}
	n := testutil.DistinctOutputs(determinismCalls, func() string {
		return testutil.FloatBits(graph.MaxFlow(adj, capacity, 0, 23))
	})
	t.Logf("graph.MaxFlow (fractional capacities): %d distinct output(s) over %d identical calls", n, determinismCalls)
}
