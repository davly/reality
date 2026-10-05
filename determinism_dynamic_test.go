package reality_test

// Determinism ratchet, dynamic half.
//
// Each function below touches maps internally and is classified
// order-insensitive or sorted-after in testdata/determinism/map_ranges.json.
// Here that classification is executed: 200 identical calls in one process
// must give one bit-exact output. Several fixtures are inputs on which an
// earlier, map-order-dependent version measurably varied between calls: a
// basic probability assignment whose total sits on the additivity tolerance
// was accepted on some calls and rejected on others, and a max-flow with
// fractional capacities returned two different values.

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

// fractionalFlowFixture builds a graph on which max-flow with fractional
// capacities once returned different values for identical calls, because the
// residual graph, and so the augmenting paths, followed map order.
func fractionalFlowFixture() (graph.IntAdjacency, map[[2]int]float64, int) {
	rng := rand.New(rand.NewSource(1))
	n := 10 + rng.Intn(30)
	adj := graph.IntAdjacency{}
	for u := 0; u < n; u++ {
		for v := 0; v < n; v++ {
			if u != v && rng.Float64() < 0.25 {
				adj[u] = append(adj[u], v)
			}
		}
	}
	capacity := map[[2]int]float64{}
	for u := 0; u < n; u++ {
		for _, v := range adj[u] {
			capacity[[2]int{u, v}] = rng.Float64() * 10
		}
	}
	return adj, capacity, n - 1
}

// boundaryMasses returns a basic probability assignment whose exact total sits
// on the additivity tolerance (1e-9), so whether it is accepted depends on the
// rounding of the sum. Summed in map order, the same input was accepted on 180
// and rejected on 20 of 200 identical calls.
func boundaryMasses() map[uint]float64 {
	rng := rand.New(rand.NewSource(2))
	k := 20 + rng.Intn(40)
	vals := make([]float64, k)
	s := 0.0
	for i := range vals {
		vals[i] = rng.Float64() + 0.01
		s += vals[i]
	}
	scale := (1 + 1e-9) / s
	masses := map[uint]float64{}
	for i := range vals {
		masses[uint(i+1)] = vals[i] * scale
	}
	return masses
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

	// Louvain on a symmetric ring (every move is a tie) and on a weighted
	// graph whose edges often carry different weights in the two directions.
	ring8 := graph.IntAdjacency{}
	for i := 0; i < 8; i++ {
		ring8[i] = []int{(i + 1) % 8}
	}
	louvainWeights := map[[2]int]float64{}
	for u := 0; u < 40; u++ {
		for _, v := range adj[u] {
			louvainWeights[[2]int{u, v}] = 0.5 + rng.Float64()
		}
	}

	flowAdj, flowCap, flowSink := fractionalFlowFixture()
	boundary := boundaryMasses()
	overlap := graph.NewADMG([]string{"a", "b", "c", "d"}, nil, nil)

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
		{"graph.LouvainCommunities (ring of 8)", func() string { return fmt.Sprint(graph.LouvainCommunities(ring8, nil, 8)) }},
		{"graph.LouvainCommunities (weighted)", func() string {
			return fmt.Sprint(graph.LouvainCommunities(adj, louvainWeights, 40))
		}},
		{"graph.Roots", func() string { return strings.Join(graph.Roots(dag), ",") }},
		{"graph.MaxFlow (fractional capacities)", func() string {
			return testutil.FloatBits(graph.MaxFlow(flowAdj, flowCap, 0, flowSink))
		}},
		{"graph.ADMG.IdentifyEffect (overlap error)", func() string {
			_, _, err := overlap.IdentifyEffect([]string{"a", "b", "c", "d"}, []string{"d", "c", "b"})
			return fmt.Sprint(err)
		}},
		{"graph.ADMG.IdentifyEffectWithWitness (overlap error)", func() string {
			_, _, _, err := overlap.IdentifyEffectWithWitness([]string{"a", "b", "c", "d"}, []string{"d", "c", "b"})
			return fmt.Sprint(err)
		}},
		{"trust.NewMassFunction (total on the tolerance)", func() string {
			_, err := trust.NewMassFunction(6, boundary)
			return fmt.Sprint(err)
		}},
		{"trust.MassFunction.Belief(Theta)", func() string { return testutil.FloatBits(m1.Belief(15)) }},
		{"trust.MassFunction.Plausibility", func() string { return testutil.FloatBits(m1.Plausibility(3)) }},
		{"reliability.SystemAvailability", func() string {
			return testutil.FloatBits(reliability.SystemAvailability(relEdges, avail, "app"))
		}},
		{"reliability.BirnbaumImportance", func() string {
			return testutil.FloatBits(reliability.BirnbaumImportance(relEdges, avail, "app", "disk"))
		}},
		{"reliability.BirnbaumImportances", func() string {
			return testutil.SortedMapBits(reliability.BirnbaumImportances(relEdges, avail, "app"))
		}},
	}
	for _, c := range cases {
		testutil.AssertDeterministic(t, c.name, determinismCalls, c.f)
	}
}
