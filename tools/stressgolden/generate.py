#!/usr/bin/env python3
"""Generate testdata/stress/precision_cases.json: stress-region golden values.

Each case pairs one function with the precision its own docstring claims and
an input in a region where that claim is hardest to keep (distribution tails,
large shapes, ties). Reference values come from mpmath at 60 significant
digits, evaluated at the exact binary64 value of every input, then rounded to
the nearest float64.

This is a development-time generator, not an implementation: Go code never
imports it, and the library keeps zero dependencies. Requirements: Python 3.8+
and mpmath (version recorded in the output). Run from the repository root:

    python tools/stressgolden/generate.py

Tolerance rules (recorded per case in `claim`):
  - Explicit bounds ("at most 6e-15") are enforced exactly as written.
  - Approximate claims ("~1e-14") are enforced with 10x slack.
  - A bound of k ulps is enforced as a relative bound of k * 2^-52, which is
    never smaller than k ulps of the value. "Correctly rounded" is enforced
    as an exact match with the nearest float64 (tolerance 0).
  - A claim stated given an intermediate statistic (a p-value given the test
    statistic) is enforced with the statistic's own rounding added: four ulps
    of the statistic, times the factor by which the docstring says it moves
    the result. Each such case derives its tolerance in `claim`.
  - Claims about p-values and other tail probabilities are enforced as
    RELATIVE error: an absolute reading of "~1e-12" passes a p-value of 0 for
    a true value of 1e-18, which checks nothing.
"""

ULP = 2.0 ** -52
import json
import math
import os
from fractions import Fraction

import mpmath as mp

mp.mp.dps = 60
OUT = os.path.join("testdata", "stress", "precision_cases.json")


def f64(x):
    """Round an mpmath value to the nearest float64."""
    return float(mp.mpf(x))


def ttest_data(n, t):
    # The same construction as the Go side is irrelevant: the data travel in
    # the JSON, so Go uses exactly these binary64 values.
    z = [float(i - n // 2) for i in range(n)]
    ss = 0.0
    for v in z:
        ss += v * v
    s = math.sqrt(ss / (n - 1))
    a = t / math.sqrt(n)
    return [a + v / s for v in z]


def ttest_p(data, mu0):
    xs = [mp.mpf(v) for v in data]
    n = len(xs)
    mean = mp.fsum(xs) / n
    var = mp.fsum((x - mean) ** 2 for x in xs) / (n - 1)
    t = (mean - mu0) / mp.sqrt(var / n)
    df = n - 1
    return mp.betainc(mp.mpf(df) / 2, mp.mpf(1) / 2, 0, df / (df + t * t), regularized=True)


def chisq_p(k, e, d):
    obs = [e + d if i % 2 == 0 else e - d for i in range(k)]  # float64 arithmetic, as in Go
    chi2 = mp.fsum((mp.mpf(o) - mp.mpf(e)) ** 2 / mp.mpf(e) for o in obs)
    df = k - 1
    return mp.gammainc(mp.mpf(df) / 2, chi2 / 2, mp.inf, regularized=True)


def fisher_two_sided(a, b, c, d):
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def pmf(x):
        return Fraction(math.comb(c1, x) * math.comb(n - c1, r1 - x), math.comb(n, r1))

    p_obs = pmf(a)
    lo, hi = max(0, r1 - (n - c1)), min(r1, c1)
    return sum((pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs), Fraction(0))


def bf10(k, n):
    tail = mp.betainc(k + 1, n - k + 1, mp.mpf("0.5"), 1, regularized=True)
    h1 = 2 / mp.mpf(n + 1) * tail
    h0 = mp.binomial(n, k) * mp.mpf("0.5") ** n
    return h1 / h0


def case(cid, func, args, want, tol, kind, claim):
    return {"id": cid, "func": func, "args": args, "want": want, "tol": tol, "tol_kind": kind, "claim": claim}


def main():
    C = []
    tt = ("prob/hypothesis.go TTestOneSample: 'Given the statistic, its relative error is at most 3e-15 for p >= 1e-10; "
          "a smaller p carries an additional error proportional to |ln p|' (taken as 5e-16*|ln p|, the rate the "
          "ChiSquaredTest docstring states) 'The statistic's own rounding (a few ulp) moves p by up to min(t^2, df+1) "
          "times as much' -> ")
    for cid, t, tol, why in (("ttest/moderate-t3", 3.0, 1.1e-14, "3e-15 + 9 * 4 ulp = 1.1e-14"),
                             ("ttest/tail-t20", 20.0, 5.5e-14, "3e-15 + 5e-16*|ln 6.8e-19| + 31 * 4 ulp = 5.2e-14 -> 5.5e-14")):
        data = ttest_data(31, t)
        C.append(case(cid, "TTestOneSample", {"data": data, "mu0": 0.0}, f64(ttest_p(data, 0)), tol, "rel", tt + "relative " + why))

    cs = ("prob/hypothesis.go ChiSquaredTest: 'the statistic is summed with compensation (relative error of a few "
          "ulp)'; 'Given the statistic, its relative error is at most 5e-15 for p >= 1e-10 and below 1e-12 for "
          "p >= 1e-300'. The statistic's rounding moves ln p by d ln Q / d ln chi2 times as much -> ")
    for cid, k, e, d2, tol, why in (
            ("chisq/df5-tail", 6, 33.3333333333, 1066.6666666, 1.1e-12, "1e-12 + (chi2/2 = 96) * 4 ulp = 1.09e-12 -> 1.1e-12"),
            ("chisq/df99999", 100000, 10.0, 9.9, 1.2e-14, "5e-15 + (chi2*f/Q = 7.2) * 4 ulp = 1.14e-14 -> 1.2e-14")):
        d = math.sqrt(d2)
        C.append(case(cid, "ChiSquaredTest", {"k": k, "e": e, "d": d}, f64(chisq_p(k, e, d)), tol, "rel", cs + "relative " + why))

    pc = ("prob/distributions.go PoissonCDF: 'relative error below 1e-12 wherever the result is at least 1e-300 "
          "(... at most 7e-15 for results >= 1e-10 ...)' -> relative ")
    for cid, k, lam, tol in (("poisson/small", 5, 10.0, 7e-15), ("poisson/lower-tail", 0, 40.0, 1e-12), ("poisson/median-1e4", 10000, 10000.0, 7e-15)):
        C.append(case(cid, "PoissonCDF", {"k": k, "lambda": lam}, f64(mp.gammainc(k + 1, mp.mpf(lam), mp.inf, regularized=True)), tol, "rel",
                      pc + ("7e-15 (result >= 1e-10)" if tol == 7e-15 else "1e-12 (result below 1e-10)")))

    gc = ("prob/distributions.go GammaCDF: 'for shapes from 1e-3 to 1e7 (the measured range), relative error at "
          "most 6e-15 for results >= 1e-10' -> relative 6e-15")
    for cid, kk in (("gamma/k10", 10.0), ("gamma/k1e4", 1e4), ("gamma/k1e5", 1e5)):
        C.append(case(cid, "GammaCDF", {"x": kk, "k": kk, "theta": 1.0}, f64(mp.gammainc(mp.mpf(kk), 0, mp.mpf(kk), regularized=True)), 6e-15, "rel", gc))

    nq = "prob/distributions.go NormalQuantile: 'within 2.5 ulps of the exact quantile for every p in (0, 1)' -> relative 2.5 * 2^-52"
    for cid, p in (("normalq/p1e-12", 1e-12), ("normalq/p0.025", 0.025), ("normalq/p0.3", 0.3), ("normalq/p0.975", 0.975), ("normalq/p0.999999", 0.999999)):
        want = mp.sqrt(2) * mp.erfinv(2 * mp.mpf(p) - 1)
        C.append(case(cid, "NormalQuantile", {"p": p}, f64(want), 2.5 * ULP, "rel", nq))
    C.append(case("normalq/full-precision-claim", "NormalQuantile", {"p": 0.05}, f64(mp.sqrt(2) * mp.erfinv(2 * mp.mpf(0.05) - 1)), 2.5 * ULP, "rel",
                  nq + " (this case once bound the docstring's 'full float64 precision' sentence, which contradicted its 1.15e-9 bound)"))

    eq = "prob/distributions.go ExponentialQuantile: 'within 2 ulps of the exact quantile for every p in (0, 1)' -> relative 2 * 2^-52"
    for cid, p in (("expq/p0.5", 0.5), ("expq/p1e-10", 1e-10)):
        C.append(case(cid, "ExponentialQuantile", {"p": p, "lambda": 1.0}, f64(-mp.log(1 - mp.mpf(p))), 2 * ULP, "rel", eq))

    bf = "prob/bayesfactor.go ProportionBayesFactor10: 'the relative error is at most 5.3e-15 for n <= 1000' -> relative 5.3e-15"
    for cid, k, n in (("bf10/k30-n60", 30, 60), ("bf10/k0-n60", 0, 60)):
        C.append(case(cid, "ProportionBayesFactor10", {"k": k, "n": n}, f64(bf10(k, n)), 5.3e-15, "rel", bf))

    bc = "combinatorics/counting.go BinomialCoeff: 'correctly rounded: the result is the float64 nearest to the exact integer C(n,k)' -> exact match"
    for cid, n, k in (("binom/50-21", 50, 21), ("binom/200-100", 200, 100)):
        C.append(case(cid, "BinomialCoeff", {"n": n, "k": k}, float(math.comb(n, k)), 0.0, "rel", bc))

    fc = "combinatorics/counting.go Factorial: 'correctly rounded (round to nearest, ties to even) for every n <= 170, and exact for n <= 22' -> exact match"
    for cid, n in (("factorial/20", 20), ("factorial/100", 100), ("factorial/166", 166)):
        C.append(case(cid, "Factorial", {"n": n}, float(math.factorial(n)), 0.0, "rel", fc))

    fe = "prob/nonparametric.go FisherExactTest: 'relative error at most 7.5e-14 against the exact p-value for every table with n <= 40' -> relative 7.5e-14"
    for cid, t4 in (("fisher/8-2-1-5", (8, 2, 1, 5)), ("fisher/tie-9-11-10-10", (9, 11, 10, 10))):
        a, b, c, d = t4
        C.append(case(cid, "FisherExactTest", {"a": a, "b": b, "c": c, "d": d}, float(fisher_two_sided(a, b, c, d)), 7.5e-14, "rel", fe))

    bt = "prob/distributions.go BetaCDF: 'absolute error at most 2.2e-15 (measured for parameters from 1e-3 to 1e7)' -> absolute 2.2e-15"
    C.append(case("betacdf/0.3-2-5", "BetaCDF", {"x": 0.3, "alpha": 2.0, "beta": 5.0},
                  f64(mp.betainc(2, 5, 0, mp.mpf(0.3), regularized=True)), 2.2e-15, "abs", bt))
    # I_{1/2}(a, a) = 1/2 exactly, by symmetry of Beta(a, a); mpmath's series
    # does not converge at a = 1e6, and the identity is the better oracle.
    C.append(case("betacdf/0.5-1e6-1e6", "BetaCDF", {"x": 0.5, "alpha": 1e6, "beta": 1e6}, 0.5, 2.2e-15, "abs",
                  bt + "; expected value from the symmetry identity I_1/2(a, a) = 1/2"))

    doc = {
        "_comment": "Generated by tools/stressgolden/generate.py; do not edit by hand. See precision_test.go.",
        "generator": {"mpmath": mp.__version__, "dps": mp.mp.dps},
        "cases": C,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    print(f"wrote {OUT}: {len(C)} cases")


if __name__ == "__main__":
    main()
