package agreement

import "sort"

// confusionMatrix builds a k x k contingency matrix for two raters'
// categorical ratings over the same n units. categories holds the sorted
// distinct integer codes observed across either rater; matrix[i][j] counts
// how many units rater 1 coded categories[i] while rater 2 coded
// categories[j].
//
// Shared by CohenKappa and WeightedKappa, which differ only in how they
// weight the off-diagonal cells.
func confusionMatrix(a, b []int) (categories []int, matrix [][]float64, err error) {
	if len(a) != len(b) {
		return nil, nil, ErrLengthMismatch
	}
	if len(a) == 0 {
		return nil, nil, ErrEmptyInput
	}

	seen := make(map[int]struct{})
	for _, v := range a {
		seen[v] = struct{}{}
	}
	for _, v := range b {
		seen[v] = struct{}{}
	}
	categories = make([]int, 0, len(seen))
	for v := range seen {
		categories = append(categories, v)
	}
	sort.Ints(categories)

	index := make(map[int]int, len(categories))
	for i, v := range categories {
		index[v] = i
	}

	k := len(categories)
	matrix = make([][]float64, k)
	for i := range matrix {
		matrix[i] = make([]float64, k)
	}
	for i := range a {
		matrix[index[a[i]]][index[b[i]]]++
	}
	return categories, matrix, nil
}

// marginals returns the row sums, column sums, and grand total of a k x k
// contingency matrix.
func marginals(matrix [][]float64) (rowSum, colSum []float64, n float64) {
	k := len(matrix)
	rowSum = make([]float64, k)
	colSum = make([]float64, k)
	for i := 0; i < k; i++ {
		for j := 0; j < k; j++ {
			rowSum[i] += matrix[i][j]
			colSum[j] += matrix[i][j]
			n += matrix[i][j]
		}
	}
	return rowSum, colSum, n
}
