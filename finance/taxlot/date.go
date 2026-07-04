package taxlot

// Civil-date kernel — pure proleptic-Gregorian date arithmetic with zero
// dependency on the standard library's time package. Tax holding-period and
// wash-sale window math is calendar arithmetic, not wall-clock arithmetic:
// there is no time-of-day, no time zone, and no daylight-saving component to
// a trade date. Encoding dates as (year, month, day) triples with explicit,
// documented leap-year handling keeps the statutory arithmetic deterministic
// and inspectable rather than delegating it to an opaque time.Time whose
// location/monotonic-clock fields are irrelevant here and a source of subtle
// bugs (a Feb-29 anniversary normalises silently inside time.AddDate).
//
// The days-since-epoch conversion is Howard Hinnant's branch-free
// `days_from_civil` algorithm, which is exact for the entire proleptic
// Gregorian calendar and correctly handles every leap year (including the
// century rule: 1900 and 2100 are common years, 2000 is a leap year).
//
// Reference: Howard Hinnant, "chrono-Compatible Low-Level Date Algorithms"
// (2013), http://howardhinnant.github.io/date_algorithms.html — public
// domain. The `days_from_civil` / `civil_from_days` pair is the canonical
// zero-dependency civil-calendar kernel.

// Date is a calendar date in the proleptic Gregorian calendar. Month is
// 1-12, Day is 1-31. A Date carries no time-of-day, zone, or clock: it is a
// pure civil date, the correct unit for trade-date tax arithmetic.
type Date struct {
	Year  int
	Month int
	Day   int
}

// D constructs a Date. It is a convenience for tests and callers; it does not
// validate the (month, day) pair — Days normalises out-of-range components the
// same way the Hinnant algorithm does, but callers are expected to pass real
// calendar dates.
func D(year, month, day int) Date { return Date{Year: year, Month: month, Day: day} }

// Days returns the number of days since the Unix epoch (1970-01-01 = 0),
// negative for earlier dates. This is Hinnant's `days_from_civil`: exact and
// leap-year-correct across the whole proleptic Gregorian range.
//
// Precision: exact integer arithmetic, no floating point. Valid for any year
// representable in int without overflow (the era arithmetic stays well within
// int64 for all realistic tax years).
func (d Date) Days() int64 {
	y := int64(d.Year)
	m := int64(d.Month)
	day := int64(d.Day)
	// Shift the year so that March is the first month: this puts the leap
	// day (Feb 29) at the END of the shifted year, removing the leap-year
	// special case from the day-of-year computation entirely.
	if m <= 2 {
		y--
	}
	era := y
	if era < 0 {
		era -= 399
	}
	era /= 400
	yoe := y - era*400 // [0, 399]
	var mp int64
	if m > 2 {
		mp = m - 3
	} else {
		mp = m + 9
	}
	doy := (153*mp+2)/5 + day - 1          // [0, 365]
	doe := yoe*365 + yoe/4 - yoe/100 + doy // [0, 146096]
	return era*146097 + doe - 719468
}

// dateFromDays is Hinnant's inverse `civil_from_days`: it reconstructs the
// (year, month, day) triple from a Unix-epoch day count. Exact, leap-safe.
func dateFromDays(z int64) Date {
	z += 719468
	var era int64
	if z >= 0 {
		era = z / 146097
	} else {
		era = (z - 146096) / 146097
	}
	doe := z - era*146097                                  // [0, 146096]
	yoe := (doe - doe/1460 + doe/36524 - doe/146096) / 365 // [0, 399]
	y := yoe + era*400
	doy := doe - (365*yoe + yoe/4 - yoe/100) // [0, 365]
	mp := (5*doy + 2) / 153                  // [0, 11]
	day := doy - (153*mp+2)/5 + 1            // [1, 31]
	var m int64
	if mp < 10 {
		m = mp + 3
	} else {
		m = mp - 9
	}
	if m <= 2 {
		y++
	}
	return Date{Year: int(y), Month: int(m), Day: int(day)}
}

// AddDays returns the date n calendar days after d (n may be negative). Exact
// via the epoch-day round trip; correct across month and year boundaries and
// every leap year.
func (d Date) AddDays(n int64) Date { return dateFromDays(d.Days() + n) }

// DaysUntil returns the signed number of days from d to other (other - d).
// Positive when other is later. This is a pure difference of epoch-day counts.
func (d Date) DaysUntil(other Date) int64 { return other.Days() - d.Days() }

// Before reports whether d is strictly earlier than other.
func (d Date) Before(other Date) bool { return d.Days() < other.Days() }

// After reports whether d is strictly later than other.
func (d Date) After(other Date) bool { return d.Days() > other.Days() }

// Equal reports whether d and other are the same calendar date.
func (d Date) Equal(other Date) bool { return d.Days() == other.Days() }

// AddYears returns the date n years after d (n may be negative), holding month
// and day fixed EXCEPT that an invalid result (only Feb 29 in a target common
// year) is clamped to the last valid day of that month, i.e. Feb 28. This is
// the "end-of-month clamp" convention: it keeps the anniversary of a Feb-29
// acquisition on the last day of February in a non-leap year, matching the
// treatment used for last-day-of-month holding periods (Rev. Rul. 66-97).
//
// This clamp only ever fires for a Feb-29 source date landing in a common
// year; every other (month, day) is preserved exactly.
func (d Date) AddYears(n int) Date {
	y := d.Year + n
	day := d.Day
	if d.Month == 2 && d.Day == 29 && !isLeapYear(y) {
		day = 28
	}
	return Date{Year: y, Month: d.Month, Day: day}
}

// isLeapYear applies the Gregorian century rule: divisible by 4, except
// centuries which must be divisible by 400.
func isLeapYear(y int) bool { return (y%4 == 0 && y%100 != 0) || y%400 == 0 }
