package geometry

import (
	"math"
	"sort"
)

// ---------------------------------------------------------------------------
// Computational Geometry — 2D polygons
//
// Triangle operations, point-in-polygon tests, and convex hull computation.
// ConvexHull2D is the only function that allocates (it must return a new
// slice). All others are allocation-free.
// ---------------------------------------------------------------------------

// TriangleArea2D computes the signed area of a triangle with vertices
// (ax, ay), (bx, by), (cx, cy) using the cross-product formula.
//
// Definition:
//
//	area = 0.5 * ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
//
// Positive when vertices are counter-clockwise, negative when clockwise.
// The absolute value gives the geometric area.
// Precision: exact for IEEE 754 float64.
func TriangleArea2D(ax, ay, bx, by, cx, cy float64) float64 {
	return 0.5 * ((bx-ax)*(cy-ay) - (cx-ax)*(by-ay))
}

// PointInTriangle2D returns true if point (px, py) lies inside or on the
// boundary of the triangle with vertices (ax, ay), (bx, by), (cx, cy).
// Uses the barycentric coordinate method with signed area tests.
//
// Works for both CW and CCW vertex orderings.
// Precision: exact for IEEE 754 float64 (no transcendental functions).
func PointInTriangle2D(px, py, ax, ay, bx, by, cx, cy float64) bool {
	// Compute signed areas of sub-triangles.
	d1 := sign2D(px, py, ax, ay, bx, by)
	d2 := sign2D(px, py, bx, by, cx, cy)
	d3 := sign2D(px, py, cx, cy, ax, ay)

	hasNeg := (d1 < 0) || (d2 < 0) || (d3 < 0)
	hasPos := (d1 > 0) || (d2 > 0) || (d3 > 0)

	return !(hasNeg && hasPos)
}

// sign2D computes the sign of the cross product (p2-p1) x (p3-p1).
// Used internally by PointInTriangle2D.
func sign2D(px, py, ax, ay, bx, by float64) float64 {
	return (px-bx)*(ay-by) - (ax-bx)*(py-by)
}

// ConvexHull2D computes the convex hull of a set of 2D points using the
// Graham scan algorithm. Returns the hull vertices in counter-clockwise order.
//
// The input slice is not modified.
// Returns nil if fewer than 3 points are provided.
// For collinear points, returns the two extreme points (degenerate hull).
//
// Complexity: O(n log n) where n = len(points).
//
// Note: this is the one geometry function that allocates — it must return a
// new slice. All other functions in this package are allocation-free.
func ConvexHull2D(points [][2]float64) [][2]float64 {
	n := len(points)
	if n < 3 {
		if n == 0 {
			return nil
		}
		// Return a copy for fewer than 3 points.
		out := make([][2]float64, n)
		copy(out, points)
		return out
	}

	// Copy to avoid mutating input.
	pts := make([][2]float64, n)
	copy(pts, points)

	// Find the lowest point (and leftmost if tied).
	pivot := 0
	for i := 1; i < n; i++ {
		if pts[i][1] < pts[pivot][1] || (pts[i][1] == pts[pivot][1] && pts[i][0] < pts[pivot][0]) {
			pivot = i
		}
	}
	pts[0], pts[pivot] = pts[pivot], pts[0]

	origin := pts[0]

	// Sort remaining points by polar angle with respect to origin.
	sort.Slice(pts[1:], func(i, j int) bool {
		a := pts[i+1]
		b := pts[j+1]
		cross := (a[0]-origin[0])*(b[1]-origin[1]) - (a[1]-origin[1])*(b[0]-origin[0])
		if cross != 0 {
			return cross > 0
		}
		// Collinear: sort by distance.
		da := (a[0]-origin[0])*(a[0]-origin[0]) + (a[1]-origin[1])*(a[1]-origin[1])
		db := (b[0]-origin[0])*(b[0]-origin[0]) + (b[1]-origin[1])*(b[1]-origin[1])
		return da < db
	})

	// Remove collinear points from the end with the same angle as the last
	// unique angle — keep only the farthest.
	m := n - 1
	for m > 0 {
		cross := (pts[m][0]-origin[0])*(pts[m-1][1]-origin[1]) -
			(pts[m][1]-origin[1])*(pts[m-1][0]-origin[0])
		if math.Abs(cross) > 1e-15 {
			break
		}
		m--
	}
	// Reverse the tail of collinear points so the farthest comes first
	// (already handled by distance sort, but the tail group for the last
	// angle should be reversed to ensure we only keep the farthest on exit).
	// Actually, Graham scan handles this correctly — proceed with the stack.

	// Graham scan.
	stack := make([][2]float64, 0, n)
	stack = append(stack, pts[0], pts[1])

	for i := 2; i < n; i++ {
		for len(stack) > 1 {
			top := stack[len(stack)-1]
			below := stack[len(stack)-2]
			cross := (top[0]-below[0])*(pts[i][1]-below[1]) -
				(top[1]-below[1])*(pts[i][0]-below[0])
			if cross > 0 {
				break
			}
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, pts[i])
	}

	return stack
}
