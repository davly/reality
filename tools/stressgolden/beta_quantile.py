#!/usr/bin/env python3
"""Generate the BetaQuantile oracle table with mpmath at 800 digits. Dev-time only; the library never imports it.

The precision is this high on purpose: at 60 digits betainc(20000, 30) silently returned a wrong value near 1e-300, and its
own residual check still passed, because cancellation in the series eats the working digits.

For each (a, b, p) it solves I_x(a, b) = p and stores x and 1 - x. The search runs in log space on whichever side of 1/2
holds the root, so x near 0 and x near 1 are both exact to the digits stored.

Usage: python3 generate_beta_quantile.py > beta_quantile.json
"""
import json
import mpmath as mp

mp.mp.dps = 800

SHAPES = [
    ("0.5", "0.5"), ("1", "1"), ("2", "3"), ("0.1", "0.1"), ("0.01", "5"), ("5", "0.01"),
    ("10", "10"), ("100", "0.5"), ("0.5", "100"), ("100", "100"), ("1000", "5000"), ("20000", "30"),
]
PROBS = ["1e-300", "1e-30", "1e-10", "1e-6", "0.001", "0.05", "0.5", "0.95", "0.999"]
ITERATIONS = 240
SMALLEST_DOUBLE = mp.mpf("5e-324")


def cdf(x, a, b):
    return mp.betainc(a, b, 0, x, regularized=True)


def solve_low(target, a, b):
    """Find x in (0, 1/2] with I_x(a, b) = target, bisecting on log x."""
    lo = mp.mpf(-3000)
    hi = mp.log(mp.mpf("0.5"))
    for _ in range(ITERATIONS):
        mid = (lo + hi) / 2
        if cdf(mp.exp(mid), a, b) < target:
            lo = mid
        else:
            hi = mid
    return mp.exp((lo + hi) / 2)


def solve(a, b, p):
    """Return (x, 1 - x, residual), with x and 1 - x each to full relative accuracy and the residual on the solved side."""
    if cdf(mp.mpf("0.5"), a, b) >= p:
        x = solve_low(p, a, b)
        return x, 1 - x, abs(cdf(x, a, b) - p) / p
    q = 1 - p
    y = solve_low(q, b, a)  # I_x(a, b) = 1 - I_{1-x}(b, a)
    return 1 - y, y, abs(cdf(y, b, a) - q) / q


def main():
    rows = []
    for a_s, b_s in SHAPES:
        a, b = mp.mpf(a_s), mp.mpf(b_s)
        for p_s in PROBS:
            p = mp.mpf(p_s)
            x, xc, residual = solve(a, b, p)
            if x < SMALLEST_DOUBLE or xc < SMALLEST_DOUBLE:
                continue  # the answer is not representable as a double, so it cannot be a test of one
            assert residual < mp.mpf("1e-40"), (a_s, b_s, p_s, residual)
            core = mp.mpf("0.5") <= a <= 100 and mp.mpf("0.5") <= b <= 100 and mp.mpf("1e-6") <= p <= mp.mpf("0.999")
            rows.append({"a": a_s, "b": b_s, "p": p_s, "q": mp.nstr(1 - p, 40), "x": mp.nstr(x, 40), "one_minus_x": mp.nstr(xc, 40), "core": bool(core)})
    print(json.dumps({"digits": 800, "rows": rows}, indent=1))


main()
