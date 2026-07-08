package taxlot

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

type washGolden struct {
	Function string `json:"function"`
	Source   string `json:"source"`
	Cases    []struct {
		Description string `json:"description"`
		Sale        struct {
			Shares          int64  `json:"shares"`
			CostBasisCents  int64  `json:"cost_basis_cents"`
			ProceedsCents   int64  `json:"proceeds_cents"`
			AcquisitionDate string `json:"acquisition_date"`
			SaleDate        string `json:"sale_date"`
		} `json:"sale"`
		Replacements []struct {
			Shares          int64  `json:"shares"`
			CostBasisCents  int64  `json:"cost_basis_cents"`
			AcquisitionDate string `json:"acquisition_date"`
		} `json:"replacements"`
		Expected struct {
			SharesSold          int64 `json:"shares_sold"`
			MatchedShares       int64 `json:"matched_shares"`
			TotalLossCents      int64 `json:"total_loss_cents"`
			DisallowedLossCents int64 `json:"disallowed_loss_cents"`
			DeductibleLossCents int64 `json:"deductible_loss_cents"`
			Adjustments         []struct {
				LotIndex               int    `json:"lot_index"`
				MatchedShares          int64  `json:"matched_shares"`
				BasisIncreaseCents     int64  `json:"basis_increase_cents"`
				AdjustedCostBasisCents int64  `json:"adjusted_cost_basis_cents"`
				TackedAcquisitionDate  string `json:"tacked_acquisition_date"`
			} `json:"adjustments"`
		} `json:"expected"`
	} `json:"cases"`
}

// TestGoldenWashSale validates ApplyWashSale against statutory worked examples
// (IRS Pub 550) plus a multi-lot apportionment case exercising acquisition-order
// matching and cent-level cumulative rounding.
func TestGoldenWashSale(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "wash_sale.json"))
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var g washGolden
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatalf("parse golden: %v", err)
	}
	if len(g.Cases) == 0 {
		t.Fatal("golden file has no cases")
	}
	for _, c := range g.Cases {
		sale := LossSale{
			Shares:          c.Sale.Shares,
			CostBasis:       Cents(c.Sale.CostBasisCents),
			Proceeds:        Cents(c.Sale.ProceedsCents),
			AcquisitionDate: parseTestDate(t, c.Sale.AcquisitionDate),
			SaleDate:        parseTestDate(t, c.Sale.SaleDate),
		}
		var reps []ReplacementLot
		for _, r := range c.Replacements {
			reps = append(reps, ReplacementLot{
				Shares:          r.Shares,
				CostBasis:       Cents(r.CostBasisCents),
				AcquisitionDate: parseTestDate(t, r.AcquisitionDate),
			})
		}
		got, err := ApplyWashSale(sale, reps)
		if err != nil {
			t.Fatalf("%s: unexpected error: %v", c.Description, err)
		}
		e := c.Expected
		if got.SharesSold != e.SharesSold || got.MatchedShares != e.MatchedShares ||
			int64(got.TotalLoss) != e.TotalLossCents ||
			int64(got.DisallowedLoss) != e.DisallowedLossCents ||
			int64(got.DeductibleLoss) != e.DeductibleLossCents {
			t.Errorf("%s:\n  scalars got {sold:%d matched:%d total:%d disallowed:%d deductible:%d}\n  want {sold:%d matched:%d total:%d disallowed:%d deductible:%d}",
				c.Description, got.SharesSold, got.MatchedShares, got.TotalLoss, got.DisallowedLoss, got.DeductibleLoss,
				e.SharesSold, e.MatchedShares, e.TotalLossCents, e.DisallowedLossCents, e.DeductibleLossCents)
		}
		if len(got.Adjustments) != len(e.Adjustments) {
			t.Fatalf("%s: got %d adjustments, want %d", c.Description, len(got.Adjustments), len(e.Adjustments))
		}
		for i, ea := range e.Adjustments {
			ga := got.Adjustments[i]
			wantTacked := parseTestDate(t, ea.TackedAcquisitionDate)
			if ga.LotIndex != ea.LotIndex || ga.MatchedShares != ea.MatchedShares ||
				int64(ga.BasisIncrease) != ea.BasisIncreaseCents ||
				int64(ga.AdjustedCostBasis) != ea.AdjustedCostBasisCents ||
				!ga.TackedAcquisitionDate.Equal(wantTacked) {
				t.Errorf("%s adj[%d]:\n  got  {idx:%d matched:%d basis+:%d adj:%d tacked:%v}\n  want {idx:%d matched:%d basis+:%d adj:%d tacked:%s}",
					c.Description, i,
					ga.LotIndex, ga.MatchedShares, ga.BasisIncrease, ga.AdjustedCostBasis, ga.TackedAcquisitionDate,
					ea.LotIndex, ea.MatchedShares, ea.BasisIncreaseCents, ea.AdjustedCostBasisCents, ea.TackedAcquisitionDate)
			}
		}
	}
}

// TestApportionmentSumsExactly is the load-bearing reconciliation invariant:
// across any partition of matched shares, the per-lot basis increases must sum
// EXACTLY to the total disallowed loss — no residual cent may be created or
// destroyed.
func TestApportionmentSumsExactly(t *testing.T) {
	sale := LossSale{Shares: 1000, CostBasis: 1000000, Proceeds: 333333,
		AcquisitionDate: D(2023, 1, 1), SaleDate: D(2023, 6, 1)}
	// Many small odd-sized lots to stress cumulative rounding.
	var reps []ReplacementLot
	sizes := []int64{7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 400}
	for i, s := range sizes {
		reps = append(reps, ReplacementLot{Shares: s, CostBasis: Cents(1000 * s),
			AcquisitionDate: D(2023, 6, 5).AddDays(int64(i))})
	}
	res, err := ApplyWashSale(sale, reps)
	if err != nil {
		t.Fatal(err)
	}
	var sum Cents
	var matchedSum int64
	for _, a := range res.Adjustments {
		sum += a.BasisIncrease
		matchedSum += a.MatchedShares
		if a.AdjustedCostBasis != reps[a.LotIndex].CostBasis+a.BasisIncrease {
			t.Errorf("lot %d adjusted basis inconsistent", a.LotIndex)
		}
	}
	if sum != res.DisallowedLoss {
		t.Errorf("apportioned basis increases sum to %d, want disallowed %d", sum, res.DisallowedLoss)
	}
	if matchedSum != res.MatchedShares {
		t.Errorf("apportioned matched shares sum to %d, want %d", matchedSum, res.MatchedShares)
	}
	if res.DisallowedLoss+res.DeductibleLoss != res.TotalLoss {
		t.Errorf("disallowed + deductible = %d, want total %d", res.DisallowedLoss+res.DeductibleLoss, res.TotalLoss)
	}
}

func TestNoLossIsNotWashSale(t *testing.T) {
	sale := LossSale{Shares: 100, CostBasis: 1000, Proceeds: 1200}
	if _, err := ApplyWashSale(sale, nil); !errors.Is(err, ErrNoLoss) {
		t.Errorf("gain sale: got err %v, want ErrNoLoss", err)
	}
	sale.Proceeds = 1000 // break-even is not a loss
	if _, err := ApplyWashSale(sale, nil); !errors.Is(err, ErrNoLoss) {
		t.Errorf("break-even: got err %v, want ErrNoLoss", err)
	}
}

func TestNoReplacementIsFullyDeductible(t *testing.T) {
	sale := LossSale{Shares: 100, CostBasis: 100000, Proceeds: 70000,
		AcquisitionDate: D(2023, 1, 1), SaleDate: D(2023, 3, 1)}
	res, err := ApplyWashSale(sale, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.DisallowedLoss != 0 || res.DeductibleLoss != 30000 || res.MatchedShares != 0 || len(res.Adjustments) != 0 {
		t.Errorf("no replacement should be fully deductible, got %+v", res)
	}
}

func TestInputValidation(t *testing.T) {
	good := LossSale{Shares: 100, CostBasis: 100000, Proceeds: 70000, AcquisitionDate: D(2023, 1, 1), SaleDate: D(2023, 3, 1)}
	if _, err := ApplyWashSale(LossSale{Shares: 0, CostBasis: 10, Proceeds: 5}, nil); !errors.Is(err, ErrNonPositiveShares) {
		t.Errorf("zero shares: want ErrNonPositiveShares, got %v", err)
	}
	if _, err := ApplyWashSale(LossSale{Shares: 100, CostBasis: -1, Proceeds: 0}, nil); !errors.Is(err, ErrNegativeMoney) {
		t.Errorf("negative basis: want ErrNegativeMoney, got %v", err)
	}
	if _, err := ApplyWashSale(good, []ReplacementLot{{Shares: -1, CostBasis: 10, AcquisitionDate: D(2023, 3, 5)}}); !errors.Is(err, ErrNonPositiveShares) {
		t.Errorf("negative replacement shares: want ErrNonPositiveShares, got %v", err)
	}
	if _, err := ApplyWashSale(good, []ReplacementLot{{Shares: 10, CostBasis: -5, AcquisitionDate: D(2023, 3, 5)}}); !errors.Is(err, ErrNegativeMoney) {
		t.Errorf("negative replacement basis: want ErrNegativeMoney, got %v", err)
	}
}

func TestRoundDivHalfAwayFromZero(t *testing.T) {
	cases := []struct{ num, den, want int64 }{
		{0, 1, 0}, {1, 2, 1}, {3, 2, 2}, {5, 2, 3}, {10, 3, 3}, {11, 3, 4}, {23333 * 30, 70, 10000},
	}
	for _, c := range cases {
		if got := roundDiv(c.num, c.den); got != c.want {
			t.Errorf("roundDiv(%d,%d) = %d, want %d", c.num, c.den, got, c.want)
		}
	}
}
