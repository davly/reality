package graph

import (
	"container/heap"
	"math"
)

// Dijkstra computes single-source shortest paths from source to all reachable
// nodes in a weighted directed graph using Dijkstra's algorithm with a binary
// heap priority queue.
//
// Parameters:
//   - adj: directed adjacency list (node -> list of neighbor nodes).
//   - weights: edge weights keyed by [from, to] pairs. Missing edges are
//     treated as absent (not traversable). All weights must be non-negative.
//   - source: the starting node index.
//
// Returns:
//   - dist: shortest distance from source to each node. dist[i] == +Inf means
//     node i is unreachable. The length of dist equals n where n is the
//     maximum node index + 1 seen across adj keys, adj values, and source.
//   - prev: predecessor array for path reconstruction. prev[i] == -1 means
//     node i has no predecessor (either it is the source or unreachable).
//
// Time complexity: O((V + E) log V) with a binary heap.
// Space complexity: O(V).
//
// Source: classic Dijkstra with min-heap, zero external dependencies.
func Dijkstra(adj IntAdjacency, weights map[[2]int]float64, source int) (dist []float64, prev []int) {
	n := graphSize(adj, source)
	dist = make([]float64, n)
	prev = make([]int, n)
	for i := range dist {
		dist[i] = math.Inf(1)
		prev[i] = -1
	}
	dist[source] = 0

	pq := &dijkstraHeap{{node: source, dist: 0}}
	heap.Init(pq)

	for pq.Len() > 0 {
		u := heap.Pop(pq).(dijkstraItem)
		if u.dist > dist[u.node] {
			continue // stale entry
		}
		for _, v := range adj[u.node] {
			w, ok := weights[[2]int{u.node, v}]
			if !ok {
				continue
			}
			newDist := dist[u.node] + w
			if newDist < dist[v] {
				dist[v] = newDist
				prev[v] = u.node
				heap.Push(pq, dijkstraItem{node: v, dist: newDist})
			}
		}
	}

	return dist, prev
}

// AStar finds the shortest path from source to target in a weighted directed
// graph using the A* algorithm with a provided heuristic function.
//
// The heuristic must be admissible (never overestimates the true cost to
// target) for the result to be optimal. When heuristic returns 0 for all
// nodes, A* degenerates to Dijkstra.
//
// Parameters:
//   - adj: directed adjacency list.
//   - weights: edge weights keyed by [from, to]. All weights must be non-negative.
//   - source, target: start and end node indices.
//   - heuristic: estimates cost from a node to target. Must be >= 0.
//
// Returns:
//   - path: ordered list of nodes from source to target, inclusive. nil if
//     target is unreachable.
//   - dist: total cost of the path. +Inf if target is unreachable.
//
// Time complexity: O((V + E) log V) with a good heuristic.
func AStar(adj IntAdjacency, weights map[[2]int]float64, source, target int, heuristic func(int) float64) (path []int, dist float64) {
	n := graphSize3(adj, source, target)
	gScore := make([]float64, n)
	cameFrom := make([]int, n)
	inClosed := make([]bool, n)

	for i := range gScore {
		gScore[i] = math.Inf(1)
		cameFrom[i] = -1
	}
	gScore[source] = 0

	pq := &dijkstraHeap{{node: source, dist: heuristic(source)}}
	heap.Init(pq)

	for pq.Len() > 0 {
		u := heap.Pop(pq).(dijkstraItem)
		if u.node == target {
			// Reconstruct path.
			path = reconstructPath(cameFrom, source, target)
			return path, gScore[target]
		}
		if inClosed[u.node] {
			continue
		}
		inClosed[u.node] = true

		for _, v := range adj[u.node] {
			if inClosed[v] {
				continue
			}
			w, ok := weights[[2]int{u.node, v}]
			if !ok {
				continue
			}
			tentative := gScore[u.node] + w
			if tentative < gScore[v] {
				gScore[v] = tentative
				cameFrom[v] = u.node
				fScore := tentative + heuristic(v)
				heap.Push(pq, dijkstraItem{node: v, dist: fScore})
			}
		}
	}

	return nil, math.Inf(1)
}

// FloydWarshall computes all-pairs shortest paths for a weighted directed
// graph using the Floyd-Warshall algorithm.
//
// Parameters:
//   - n: number of nodes (0 to n-1).
//   - edges: list of weighted directed edges as [from, to, weight] triples.
//
// Returns an n x n distance matrix where dist[i][j] is the shortest distance
// from node i to node j. dist[i][j] == +Inf means j is unreachable from i.
// dist[i][i] == 0 (unless a negative cycle exists, which is undefined behavior).
//
// Time complexity: O(V^3).
// Space complexity: O(V^2).
func FloydWarshall(n int, edges [][3]float64) [][]float64 {
	dist := make([][]float64, n)
	for i := range dist {
		dist[i] = make([]float64, n)
		for j := range dist[i] {
			if i == j {
				dist[i][j] = 0
			} else {
				dist[i][j] = math.Inf(1)
			}
		}
	}

	for _, e := range edges {
		u, v, w := int(e[0]), int(e[1]), e[2]
		if u >= 0 && u < n && v >= 0 && v < n {
			if w < dist[u][v] {
				dist[u][v] = w
			}
		}
	}

	for k := 0; k < n; k++ {
		for i := 0; i < n; i++ {
			if math.IsInf(dist[i][k], 1) {
				continue
			}
			for j := 0; j < n; j++ {
				through := dist[i][k] + dist[k][j]
				if through < dist[i][j] {
					dist[i][j] = through
				}
			}
		}
	}

	return dist
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal: priority queue for Dijkstra / A*
// ═══════════════════════════════════════════════════════════════════════════

type dijkstraItem struct {
	node int
	dist float64
}

type dijkstraHeap []dijkstraItem

func (h dijkstraHeap) Len() int            { return len(h) }
func (h dijkstraHeap) Less(i, j int) bool   { return h[i].dist < h[j].dist }
func (h dijkstraHeap) Swap(i, j int)        { h[i], h[j] = h[j], h[i] }
func (h *dijkstraHeap) Push(x any)          { *h = append(*h, x.(dijkstraItem)) }
func (h *dijkstraHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════════

// graphSize determines the required array size from an adjacency list and
// a source node. Returns max(all node indices) + 1.
func graphSize(adj IntAdjacency, extra int) int {
	n := extra + 1
	for k, vs := range adj {
		if k+1 > n {
			n = k + 1
		}
		for _, v := range vs {
			if v+1 > n {
				n = v + 1
			}
		}
	}
	return n
}

// graphSize3 is like graphSize but considers two extra nodes.
func graphSize3(adj IntAdjacency, a, b int) int {
	n := graphSize(adj, a)
	if b+1 > n {
		n = b + 1
	}
	return n
}

// reconstructPath walks the cameFrom array backward from target to source.
func reconstructPath(cameFrom []int, source, target int) []int {
	var path []int
	for cur := target; cur != -1; cur = cameFrom[cur] {
		path = append(path, cur)
		if cur == source {
			break
		}
	}
	// Reverse the path.
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
	return path
}
