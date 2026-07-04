package taxlot

import "testing"

// parseTestDate parses a "YYYY-MM-DD" golden-file date string into a Date with
// pure string arithmetic (zero dependency on the time package), so the test
// substrate exercises the same no-time-package discipline as the kernel.
func parseTestDate(t *testing.T, s string) Date {
	t.Helper()
	if len(s) != 10 || s[4] != '-' || s[7] != '-' {
		t.Fatalf("bad date string %q", s)
	}
	atoi := func(b string) int {
		n := 0
		for i := 0; i < len(b); i++ {
			if b[i] < '0' || b[i] > '9' {
				t.Fatalf("non-digit in date %q", s)
			}
			n = n*10 + int(b[i]-'0')
		}
		return n
	}
	return D(atoi(s[0:4]), atoi(s[5:7]), atoi(s[8:10]))
}

func TestDaysEpochRoundTrip(t *testing.T) {
	// Anchor values against the Unix epoch (Hinnant days_from_civil).
	cases := []struct {
		d    Date
		days int64
	}{
		{D(1970, 1, 1), 0},
		{D(1969, 12, 31), -1},
		{D(1970, 1, 2), 1},
		{D(2000, 1, 1), 10957},  // known Unix day count for Y2K
		{D(2020, 2, 29), 18321}, // leap day
		{D(1900, 3, 1), -25508}, // 1900 is NOT a leap year (century rule)
	}
	for _, c := range cases {
		if got := c.d.Days(); got != c.days {
			t.Errorf("%v.Days() = %d, want %d", c.d, got, c.days)
		}
		// civil_from_days inverse must reconstruct the original date.
		if rt := dateFromDays(c.days); !rt.Equal(c.d) {
			t.Errorf("dateFromDays(%d) = %v, want %v", c.days, rt, c.d)
		}
	}
}

func TestDaysUntilAcrossLeapDay(t *testing.T) {
	// 2020 is a leap year: Feb has 29 days, so 2020-02-01 -> 2020-03-01 is 29
	// days; 2019 is common: 2019-02-01 -> 2019-03-01 is 28 days.
	if got := D(2020, 2, 1).DaysUntil(D(2020, 3, 1)); got != 29 {
		t.Errorf("leap Feb span = %d, want 29", got)
	}
	if got := D(2019, 2, 1).DaysUntil(D(2019, 3, 1)); got != 28 {
		t.Errorf("common Feb span = %d, want 28", got)
	}
	// Symmetry: DaysUntil is antisymmetric.
	if a, b := D(2021, 1, 1).DaysUntil(D(2022, 1, 1)), D(2022, 1, 1).DaysUntil(D(2021, 1, 1)); a != -b {
		t.Errorf("DaysUntil not antisymmetric: %d vs %d", a, b)
	}
	// A full non-leap year is 365 days.
	if got := D(2021, 1, 1).DaysUntil(D(2022, 1, 1)); got != 365 {
		t.Errorf("2021 length = %d, want 365", got)
	}
	// A leap year is 366 days.
	if got := D(2020, 1, 1).DaysUntil(D(2021, 1, 1)); got != 366 {
		t.Errorf("2020 length = %d, want 366", got)
	}
}

func TestAddDaysInverse(t *testing.T) {
	d := D(2024, 5, 1)
	for _, n := range []int64{0, 1, 30, 365, 366, -1, -197, 4000} {
		if got := d.AddDays(n).AddDays(-n); !got.Equal(d) {
			t.Errorf("AddDays(%d) not invertible: got %v", n, got)
		}
	}
	// Concrete boundary crossing.
	if got := D(2023, 12, 31).AddDays(1); !got.Equal(D(2024, 1, 1)) {
		t.Errorf("year rollover = %v, want 2024-01-01", got)
	}
}

func TestAddYearsLeapClamp(t *testing.T) {
	// Feb 29 + 1 year lands in a common year -> clamp to Feb 28.
	if got := D(2020, 2, 29).AddYears(1); !got.Equal(D(2021, 2, 28)) {
		t.Errorf("Feb-29 +1yr = %v, want 2021-02-28", got)
	}
	// Feb 29 + 4 years lands on the next leap year -> preserved.
	if got := D(2020, 2, 29).AddYears(4); !got.Equal(D(2024, 2, 29)) {
		t.Errorf("Feb-29 +4yr = %v, want 2024-02-29", got)
	}
	// Non-Feb-29 dates are preserved exactly.
	if got := D(2022, 7, 15).AddYears(1); !got.Equal(D(2023, 7, 15)) {
		t.Errorf("Jul-15 +1yr = %v, want 2023-07-15", got)
	}
}

func TestBeforeAfterEqual(t *testing.T) {
	a, b := D(2023, 1, 1), D(2023, 1, 2)
	if !a.Before(b) || !b.After(a) || a.Equal(b) || !a.Equal(D(2023, 1, 1)) {
		t.Error("ordering predicates wrong")
	}
}
