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
  - Explicit bounds ("< 1.15e-9") are enforced exactly as written.
  - Approximate claims ("~1e-14") are enforced with 10x slack.
  - Claims about p-values and other tail probabilities are enforced as
    RELATIVE error: an absolute reading of "~1e-12" passes a p-value of 0 for
    a true value of 1e-18, which checks nothing.
"""
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
    tt = "prob/hypothesis.go TTestOneSample: 'Precision: ~1e-12 for p-values' -> relative 1e-11"
    for cid, t in (("ttest/moderate-t3", 3.0), ("ttest/tail-t20", 20.0)):
        data = ttest_data(31, t)
        C.append(case(cid, "TTestOneSample", {"data": data, "mu0": 0.0}, f64(ttest_p(data, 0)), 1e-11, "rel", tt))

    cs = "prob/hypothesis.go ChiSquaredTest: 'Precision: ~1e-12 for p-values' -> relative 1e-11"
    for cid, k, e, d2 in (("chisq/df5-tail", 6, 33.3333333333, 1066.6666666), ("chisq/df99999", 100000, 10.0, 9.9)):
        d = math.sqrt(d2)
        C.append(case(cid, "ChiSquaredTest", {"k": k, "e": e, "d": d}, f64(chisq_p(k, e, d)), 1e-11, "rel", cs))

    pc = "prob/distributions.go PoissonCDF: 'Precision: accumulated float64 summation error' -> relative 1e-12 (a CDF below its own PMF, or several percent off, violates any reading)"
    for cid, k, lam in (("poisson/small", 5, 10.0), ("poisson/lower-tail", 0, 40.0), ("poisson/median-1e4", 10000, 10000.0)):
        C.append(case(cid, "PoissonCDF", {"k": k, "lambda": lam}, f64(mp.gammainc(k + 1, mp.mpf(lam), mp.inf, regularized=True)), 1e-12, "rel", pc))

    gc = "prob/distributions.go GammaCDF: 'Precision: ~1e-14' -> relative 1e-13"
    for cid, kk in (("gamma/k10", 10.0), ("gamma/k1e4", 1e4), ("gamma/k1e5", 1e5)):
        C.append(case(cid, "GammaCDF", {"x": kk, "k": kk, "theta": 1.0}, f64(mp.gammainc(mp.mpf(kk), 0, mp.mpf(kk), regularized=True)), 1e-13, "rel", gc))

    nq = "prob/distributions.go NormalQuantile: 'maximum relative error < 1.15e-9 for p bounded away from 1' -> relative 1.15e-9"
    for cid, p in (("normalq/p1e-12", 1e-12), ("normalq/p0.025", 0.025), ("normalq/p0.3", 0.3), ("normalq/p0.975", 0.975), ("normalq/p0.999999", 0.999999)):
        want = mp.sqrt(2) * mp.erfinv(2 * mp.mpf(p) - 1)
        C.append(case(cid, "NormalQuantile", {"p": p}, f64(want), 1.15e-9, "rel", nq))
    C.append(case("normalq/full-precision-claim", "NormalQuantile", {"p": 0.05}, f64(mp.sqrt(2) * mp.erfinv(2 * mp.mpf(0.05) - 1)), 1e-14, "rel",
                  "prob/distributions.go NormalQuantile: 'full float64 precision across the entire range (0, 1)' -> relative 1e-14 (the same docstring also states 1.15e-9; this case binds the contradiction)"))

    eq = "prob/distributions.go ExponentialQuantile: 'Precision: ~15 significant digits' -> relative 1e-14"
    for cid, p in (("expq/p0.5", 0.5), ("expq/p1e-10", 1e-10)):
        C.append(case(cid, "ExponentialQuantile", {"p": p, "lambda": 1.0}, f64(-mp.log(1 - mp.mpf(p))), 1e-14, "rel", eq))

    bf = "prob/bayesfactor.go ProportionBayesFactor10: 'matches a brute-force quadrature oracle to < 1e-6 relative' -> relative 1e-6"
    for cid, k, n in (("bf10/k30-n60", 30, 60), ("bf10/k0-n60", 0, 60)):
        C.append(case(cid, "ProportionBayesFactor10", {"k": k, "n": n}, f64(bf10(k, n)), 1e-6, "rel", bf))

    bc = "combinatorics/counting.go BinomialCoeff: 'relative error < 1e-12 for n <= 200' -> relative 1e-12"
    for cid, n, k in (("binom/50-21", 50, 21), ("binom/200-100", 200, 100)):
        C.append(case(cid, "BinomialCoeff", {"n": n, "k": k}, float(math.comb(n, k)), 1e-12, "rel", bc))

    C.append(case("factorial/20", "Factorial", {"n": 20}, float(math.factorial(20)), 0.0, "rel",
                  "combinatorics/counting.go Factorial: 'exact (bit-exact) for n <= 20' -> relative 0"))
    fc = "combinatorics/counting.go Factorial: 'for 21 <= n <= 170 ... relative error < 1e-13' -> relative 1e-13 (the same docstring cites a worst observed 1.30e-13 at n=166)"
    for cid, n in (("factorial/100", 100), ("factorial/166", 166)):
        C.append(case(cid, "Factorial", {"n": n}, float(math.factorial(n)), 1e-13, "rel", fc))

    fe = "prob/nonparametric.go FisherExactTest: 'Precision: ~1e-12' -> relative 1e-11"
    for cid, t4 in (("fisher/8-2-1-5", (8, 2, 1, 5)), ("fisher/tie-9-11-10-10", (9, 11, 10, 10))):
        a, b, c, d = t4
        C.append(case(cid, "FisherExactTest", {"a": a, "b": b, "c": c, "d": d}, float(fisher_two_sided(a, b, c, d)), 1e-11, "rel", fe))

    bt = "prob/distributions.go BetaCDF: 'Precision: ~1e-14 absolute for typical inputs' -> absolute 1e-13"
    C.append(case("betacdf/0.3-2-5", "BetaCDF", {"x": 0.3, "alpha": 2.0, "beta": 5.0},
                  f64(mp.betainc(2, 5, 0, mp.mpf(0.3), regularized=True)), 1e-13, "abs", bt))
    # I_{1/2}(a, a) = 1/2 exactly, by symmetry of Beta(a, a); mpmath's series
    # does not converge at a = 1e6, and the identity is the better oracle.
    C.append(case("betacdf/0.5-1e6-1e6", "BetaCDF", {"x": 0.5, "alpha": 1e6, "beta": 1e6}, 0.5, 1e-13, "abs",
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
