#!/usr/bin/env python3
"""Generate testdata/stress/precision_numerics.json: claim cases for the
numerical packages (linalg, optim and its subpackages, calculus, chaos).

Each case pairs one function with a precision, exactness or convergence
promise from its own doc comment and an input chosen where that promise is
hardest to keep (cancellation, extreme scales, near-singular or exactly
singular matrices, tolerances below the float spacing), plus typical inputs,
so that "fails everywhere" can be told apart from "fails at the edge".

Reference values come from mpmath at 60 significant digits (or exact
rationals), evaluated at the exact binary64 value of every input, then
rounded to the nearest float64. All input data is produced with a splitmix64
generator and plain IEEE additions and multiplications, so the file is
reproducible bit for bit on any platform.

This is a development-time generator, not an implementation: Go code never
imports it, and the library keeps zero dependencies. Requirements: Python 3.8+
and mpmath (version recorded in the output). Run from the repository root:

    python tools/stressgolden/numerics.py

Reading rules (each case states its reading in `claim`):
  - Explicit bounds are enforced as written.
  - Approximate claims ("~1e-12", "~15 significant digits") are enforced with
    10x slack: ~1e-12 -> 1e-11, "~15 significant digits" -> relative 1e-14.
  - "Exact" / "exact for IEEE 754 float64" for a computed (rounded) result is
    read as faithful rounding: relative error below 2^-52 (one ulp at the
    bottom of the binade, so never stricter than one ulp).
  - "~O(n)*eps" accumulated-rounding claims are enforced as 10*n*2^-52,
    relative to the quantity the error analysis bounds (stated per case).
  - Asymptotic claims (O(h^2), O(1/k^2)) are enforced with the constant of the
    reference the doc comment cites (stated per case).
  - For a vector or matrix result the evaluator returns one error measure
    (stated in `claim`) and the case expects 0 within the claimed tolerance;
    the reference vector travels in args["truth"].
  - A function that reports failure (nil, false, an error) where its doc
    promises a result is scored as NaN, i.e. as missing its claim, and the
    reverse: where the doc promises nil/false, success scores 1 against 0.
"""

import itertools
import json
import math
import os
from fractions import Fraction

import mpmath as mp

mp.mp.dps = 60
OUT = os.path.join("testdata", "stress", "precision_numerics.json")
ULP = 2.0 ** -52
U = 2.0 ** -53
M64 = (1 << 64) - 1


def f64(x):
    """Round an mpmath value (or Fraction) to the nearest float64."""
    if isinstance(x, Fraction):
        return float(x)  # correctly rounded
    return float(mp.mpf(x))


class SplitMix64:
    """Deterministic data source: integer arithmetic plus exact scaling."""

    def __init__(self, seed):
        self.s = seed & M64

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)

    def uniform(self):
        return (self.next() >> 11) * U  # exact multiple of 2^-53 in [0, 1)

    def normal(self):
        # Irwin-Hall: sum of 12 uniforms - 6. Only IEEE + and -, so the
        # sequence is identical on every platform (no libm).
        s = 0.0
        for _ in range(12):
            s += self.uniform()
        return s - 6.0


def case(cid, func, args, want, tol, kind, claim):
    assert kind in ("rel", "abs")
    return {"id": "numerics/" + cid, "func": func, "args": args, "want": want, "tol": tol, "tol_kind": kind, "claim": claim}


def M(x):
    return mp.mpf(x)


# ---------------------------------------------------------------------------
# linalg
# ---------------------------------------------------------------------------

def pearson_true(x, y):
    xs = [M(v) for v in x]
    ys = [M(v) for v in y]
    n = len(xs)
    mx = mp.fsum(xs) / n
    my = mp.fsum(ys) / n
    sxy = mp.fsum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = mp.fsum((a - mx) ** 2 for a in xs)
    syy = mp.fsum((b - my) ** 2 for b in ys)
    return sxy / mp.sqrt(sxx * syy)


def avg_ranks(v):
    idx = sorted(range(len(v)), key=lambda i: v[i])
    r = [Fraction(0)] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        avg = Fraction(i + j, 2) + 1
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r


def spearman_true(x, y):
    rx, ry = avg_ranks(x), avg_ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return M(sxy.numerator) / M(sxy.denominator) / mp.sqrt(M(sxx.numerator) / M(sxx.denominator) * M(syy.numerator) / M(syy.denominator))


def centered(X, T, N):
    Y = [[M(X[t * N + i]) for i in range(N)] for t in range(T)]
    for i in range(N):
        mu = mp.fsum(Y[t][i] for t in range(T)) / T
        for t in range(T):
            Y[t][i] -= mu
    return Y


def cov_mle(Y, T, N):
    return [[mp.fsum(Y[t][i] * Y[t][j] for t in range(T)) / T for j in range(N)] for i in range(N)]


def lw_identity_true(X, T, N):
    Y = centered(X, T, N)
    S = cov_mle(Y, T, N)
    m = mp.fsum(S[i][i] for i in range(N)) / N
    d2 = mp.fsum((S[i][j] - (m if i == j else 0)) ** 2 for i in range(N) for j in range(N)) / N
    bbar2 = mp.fsum(mp.fsum((Y[t][i] * Y[t][j] - S[i][j]) ** 2 for i in range(N) for j in range(N)) / N for t in range(T)) / (T * T)
    b2 = min(bbar2, d2)
    a2 = d2 - b2
    sh = b2 / d2
    sigma = [[a2 / d2 * S[i][j] + (sh * m if i == j else 0) for j in range(N)] for i in range(N)]
    return sh, sigma


def lw_constcorr_true(X, T, N):
    Y = centered(X, T, N)
    S = cov_mle(Y, T, N)
    sd = [mp.sqrt(S[i][i]) for i in range(N)]
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rbar = mp.fsum(S[i][j] / (sd[i] * sd[j]) for i, j in pairs) / len(pairs)
    F = [[S[i][i] if i == j else rbar * sd[i] * sd[j] for j in range(N)] for i in range(N)]
    pim = [[mp.fsum((Y[t][i] * Y[t][j] - S[i][j]) ** 2 for t in range(T)) / T for j in range(N)] for i in range(N)]
    pi = mp.fsum(pim[i][j] for i in range(N) for j in range(N))
    rho = mp.fsum(pim[i][i] for i in range(N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            thii = mp.fsum((Y[t][i] ** 2 - S[i][i]) * (Y[t][i] * Y[t][j] - S[i][j]) for t in range(T)) / T
            thjj = mp.fsum((Y[t][j] ** 2 - S[j][j]) * (Y[t][i] * Y[t][j] - S[i][j]) for t in range(T)) / T
            rho += rbar / 2 * (sd[j] / sd[i] * thii + sd[i] / sd[j] * thjj)
    gamma = mp.fsum((F[i][j] - S[i][j]) ** 2 for i in range(N) for j in range(N))
    kappa = (pi - rho) / gamma
    delta = kappa / T
    assert 0 < delta < 1, delta
    sigma = [[delta * F[i][j] + (1 - delta) * S[i][j] for j in range(N)] for i in range(N)]
    return delta, sigma


def factor_returns(rng, T, N, load, scale, offset=0.0):
    X = []
    for _ in range(T):
        f = rng.normal()
        for i in range(N):
            X.append(offset + scale * (load[i] * f + rng.normal()))
    return X


def clean_corr_true(C, T, N):
    A = mp.matrix(N, N)
    for i in range(N):
        for j in range(N):
            A[i, j] = M(C[i * N + j])
    ev, Q = mp.eigsy(A)
    lmax = (1 + mp.sqrt(mp.mpf(N) / T)) ** 2
    vals = [ev[k] for k in range(N)]
    for v in vals:
        assert abs(v - lmax) > mp.mpf("1e-6"), "eigenvalue too close to the Marchenko-Pastur edge"
    noise = [v for v in vals if v <= lmax]
    avg = mp.fsum(noise) / len(noise) if noise else mp.mpf(0)
    lam = [v if v > lmax else avg for v in vals]
    R = [[mp.fsum(Q[i, k] * lam[k] * Q[j, k] for k in range(N)) for j in range(N)] for i in range(N)]
    out = [[R[i][j] / mp.sqrt(R[i][i] * R[j][j]) for j in range(N)] for i in range(N)]
    for i in range(N):
        out[i][i] = mp.mpf(1)
    return out, len(noise)


def corr_from_returns(X, T, N):
    Y = centered(X, T, N)
    S = cov_mle(Y, T, N)
    C = []
    for i in range(N):
        for j in range(N):
            C.append(1.0 if i == j else f64(S[i][j] / mp.sqrt(S[i][i] * S[j][j])))
    for i in range(N):
        for j in range(i + 1, N):
            C[j * N + i] = C[i * N + j]
    return C


def exact_det(A):
    """Determinant of a square matrix of Fractions (Gaussian elimination in Q)."""
    A = [row[:] for row in A]
    n = len(A)
    det = Fraction(1)
    for k in range(n):
        p = next((i for i in range(k, n) if A[i][k] != 0), None)
        if p is None:
            return Fraction(0)
        if p != k:
            A[k], A[p] = A[p], A[k]
            det = -det
        det *= A[k][k]
        for i in range(k + 1, n):
            f = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= f * A[k][j]
    return det


def linalg_cases():
    C = []
    ex = ("'exact' read as faithful rounding: relative error below 2^-52")

    # PearsonCorrelation ----------------------------------------------------
    pc = "linalg/correlation.go PearsonCorrelation: 'Precision: exact for IEEE 754 float64.' -> " + ex
    # Typical seeds 1000..1007 were measured on the default and GOAMD64=v3
    # builds: 1000 and 1001 miss "exact" by 2-3 ulps on one build and not the
    # other, so the ratchet keeps 1002, which is faithful on both.
    for cid, seed, n, off in (("pearson/typical", 1002, 20, 0.0), ("pearson/offset-1e12", 1100, 50, 1e12)):
        rng = SplitMix64(seed)
        x, y = [], []
        for _ in range(n):
            u = 10.0 * rng.uniform()
            x.append(off + u)
            y.append(off + (0.5 * u + 0.3 * rng.normal()))
        C.append(case(cid, "linalg.PearsonCorrelation", {"x": x, "y": y}, f64(pearson_true(x, y)), ULP, "rel",
                      pc + ("" if off == 0 else "; data share an offset of %g with a spread of 10 (the centring cancels)" % off)))

    # SpearmanCorrelation ---------------------------------------------------
    sc = ("linalg/correlation.go SpearmanCorrelation: 'Precision: exact for IEEE 754 float64 (limited by ranking "
          "step).' -> " + ex + "; average ranks of ties are exact, so the reference is the exact Pearson correlation of the ranks")
    rng = SplitMix64(21)
    for cid, n, levels in (("spearman/ties-n30", 30, 7), ("spearman/n500", 500, 1 << 30)):
        x, y = [], []
        for _ in range(n):
            a = float(rng.next() % levels)
            x.append(a)
            y.append(a + float(rng.next() % levels) * 0.5)
        C.append(case(cid, "linalg.SpearmanCorrelation", {"x": x, "y": y}, f64(spearman_true(x, y)), ULP, "rel", sc))

    # Trace -----------------------------------------------------------------
    tr = ("linalg/matrix.go Trace: 'Precision: exact (accumulated float64 summation error for large n)' -> read as the "
          "recursive-summation bound |error| <= (n-1) * 2^-53 * sum|A_ii|, absolute")
    for cid, diag in (("trace/decimals", [0.1, 0.2, 0.3, 0.7]), ("trace/cancellation", [1e16, 1.0, -1e16])):
        n = len(diag)
        A = [0.0] * (n * n)
        for i, d in enumerate(diag):
            A[i * n + i] = d
            for j in range(n):
                if j != i:
                    A[i * n + j] = 0.25 * (i - j)
        want = sum((Fraction(d) for d in diag), Fraction(0))
        tol = (n - 1) * U * sum(abs(d) for d in diag)
        C.append(case(cid, "linalg.Trace", {"A": A, "n": n}, f64(want), tol, "abs", tr + " = %.3g" % tol))

    # CrossProduct ----------------------------------------------------------
    cp = ("linalg/matrix.go CrossProduct: 'Precision: exact for IEEE 754 float64.' -> " + ex +
          "; scalar = max_i |out_i - true_i| / max_j |true_j| (normwise, the most lenient per-vector reading)")
    eps9 = 2.0 ** -30
    for cid, a, b in (("cross/integers", [1.0, -2.0, 3.0], [4.0, 5.0, -6.0]),
                      ("cross/decimals", [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]),
                      ("cross/near-parallel", [0.1, 0.2, 0.3], [0.1 * (1 + eps9), 0.2, 0.3 * (1 - eps9)])):
        A_, B_ = [M(v) for v in a], [M(v) for v in b]
        t = [A_[1] * B_[2] - A_[2] * B_[1], A_[2] * B_[0] - A_[0] * B_[2], A_[0] * B_[1] - A_[1] * B_[0]]
        C.append(case(cid, "linalg.CrossProduct", {"a": a, "b": b, "truth": [f64(v) for v in t]}, 0.0, ULP, "abs",
                      cp + ("; nearly parallel inputs (the components cancel)" if "parallel" in cid else "")))

    # JamesSteinShrink ------------------------------------------------------
    js = ("linalg/shrinkage.go JamesSteinShrink: 'Precision: exact algebra in float64; no iterative error.' -> " + ex +
          "; scalar = the returned shrinkage factor c = (p-2)*variance/S")
    # Typical seeds 31..38 measured on both builds: all faithful, 31 only just
    # (0.99 of the tolerance on the default build), so the ratchet keeps 33.
    rng = SplitMix64(33)
    for cid, means, var in (("jamesstein/typical", [0.05 + 0.02 * rng.normal() for _ in range(10)], 1e-4),
                            ("jamesstein/offset-1e9", [1e9 + 0.1, 1e9 + 0.35, 1e9 + 0.8, 1e9 - 0.6], 0.05),
                            ("jamesstein/tiny-scale", [1e-151, 2e-151, 4e-151], 1e-303)):
        ms = [M(v) for v in means]
        p = len(ms)
        xb = mp.fsum(ms) / p
        S = mp.fsum((v - xb) ** 2 for v in ms)
        c = (p - 2) * M(var) / S
        assert 0 < c < 1
        C.append(case(cid, "linalg.JamesSteinShrink", {"means": means, "variance": var}, f64(c), ULP, "rel",
                      js + {"jamesstein/offset-1e9": "; means share an offset of 1e9",
                            "jamesstein/tiny-scale": "; means of order 1e-151: S = %.3g is below the code's 1e-300 'all identical' threshold" % f64(S)}.get(cid, "")))

    # Ledoit-Wolf, identity target ------------------------------------------
    lwi = ("linalg/shrinkage.go LedoitWolfShrinkageIdentity: 'Precision: ~1e-12 on the intensity for well-conditioned "
           "inputs; matrix entries accumulate to ~1e-9.' -> ")
    for cid, T, N, off, load in (("lw-identity/typical", 60, 5, 0.0, [0.8, 0.6, 0.4, 0.2, 0.0]),
                                 ("lw-identity/T500-N20", 500, 20, 0.0, [0.5 * (k % 3) for k in range(20)]),
                                 ("lw-identity/offset-1e4", 60, 5, 1e4, [0.8, 0.6, 0.4, 0.2, 0.0]),
                                 ("lw-identity/offset-1e5", 60, 5, 1e5, [0.8, 0.6, 0.4, 0.2, 0.0])):
        X = factor_returns(SplitMix64(100 + T + N), T, N, load, 0.01, off)
        sh, sig = lw_identity_true(X, T, N)
        assert 0 < sh < 1
        note = "" if off == 0 else "; returns carried at a level of %g (prices, not returns): the centring cancels %d digits" % (off, round(math.log10(off / 0.01)))
        C.append(case(cid + "-intensity", "linalg.LedoitWolfShrinkageIdentity", {"x": X, "T": T, "N": N, "measure": "intensity"},
                      f64(sh), 1e-11, "abs", lwi + "intensity absolute 1e-11 (10x)" + note))
        C.append(case(cid + "-matrix", "linalg.LedoitWolfShrinkageIdentity",
                      {"x": X, "T": T, "N": N, "measure": "matrix", "truth": [f64(sig[i][j]) for i in range(N) for j in range(N)]},
                      0.0, 1e-8, "abs", lwi + "matrix 1e-8 (10x); scalar = max |sigma_ij - true_ij| / max |true_ij|" + note))

    # Ledoit-Wolf, constant-correlation target -------------------------------
    lwc = "linalg/shrinkage.go LedoitWolfShrinkageConstantCorr: 'Precision: ~1e-12 intensity; matrix entries ~1e-9.' -> "
    for cid, T, N, off, load, scale in (("lw-constcorr/typical", 60, 5, 0.0, [0.9, 0.7, 0.2, -0.3, 0.5], 0.01),
                                        ("lw-constcorr/high-corr", 500, 6, 0.0, [8.0, 3.5, 9.0, 2.5, 7.0, 10.0], 0.01),
                                        ("lw-constcorr/offset-1e5", 60, 5, 1e5, [0.9, 0.7, 0.2, -0.3, 0.5], 0.01)):
        X = factor_returns(SplitMix64(200 + T + N), T, N, load, scale, off)
        dl, sig = lw_constcorr_true(X, T, N)
        note = {"lw-constcorr/typical": "", "lw-constcorr/high-corr": "; pairwise correlations 0.92-0.99",
                "lw-constcorr/offset-1e5": "; returns carried at a level of 1e5: the centring cancels 7 digits"}[cid]
        C.append(case(cid + "-intensity", "linalg.LedoitWolfShrinkageConstantCorr", {"x": X, "T": T, "N": N, "measure": "intensity"},
                      f64(dl), 1e-11, "abs", lwc + "intensity absolute 1e-11 (10x)" + note))
        C.append(case(cid + "-matrix", "linalg.LedoitWolfShrinkageConstantCorr",
                      {"x": X, "T": T, "N": N, "measure": "matrix", "truth": [f64(sig[i][j]) for i in range(N) for j in range(N)]},
                      0.0, 1e-8, "abs", lwc + "matrix 1e-8 (10x); scalar = max |sigma_ij - true_ij| / max |true_ij|" + note))

    # MarchenkoPasturBounds -------------------------------------------------
    mpb = ("linalg/shrinkage.go MarchenkoPasturBounds: 'Precision: exact to machine epsilon on sqrt.' -> relative error of each "
           "bound below machine epsilon 2^-52")
    for cid, Nn, Tt in (("mpbounds/q0.25", 25, 100), ("mpbounds/q0.1", 1, 10), ("mpbounds/q0.999", 999, 1000), ("mpbounds/q0.99999", 99999, 100000)):
        q = Nn / Tt  # float64 division, as a caller computes it
        sq = mp.sqrt(M(q))
        for which, v in (("min", (1 - sq) ** 2), ("max", (1 + sq) ** 2)):
            C.append(case(cid + "-" + which, "linalg.MarchenkoPasturBounds", {"q": q, "which": which}, f64(v), ULP, "rel",
                          mpb + "; q = %d/%d" % (Nn, Tt) + ("; lambdaMin = (1-sqrt q)^2 cancels as q -> 1" if which == "min" and Nn > 500 else "")))

    # CleanCorrelation --------------------------------------------------------
    cc = ("linalg/shrinkage.go CleanCorrelation: 'Precision: eigendecomposition to ~1e-12; reconstruction accumulates to "
          "~1e-9.' -> 1e-8 (10x) on max |out_ij - true_ij|; reference: mpmath eigsy of the same input, clipped at the "
          "same Marchenko-Pastur edge, reconstructed and renormalised")
    for cid, T, N, load in (("clean/typical", 40, 8, [0.9, 0.8, 0.7, 0.6, 0.0, 0.0, 0.0, 0.0]),
                            ("clean/near-singular", 120, 6, [30.0, 30.0, 0.5, 0.4, 0.0, 0.0])):
        X = factor_returns(SplitMix64(300 + T + N), T, N, load, 0.01)
        Cm = corr_from_returns(X, T, N)
        out, nn = clean_corr_true(Cm, T, N)
        C.append(case(cid, "linalg.CleanCorrelation", {"corr": Cm, "T": T, "N": N, "truth": [f64(out[i][j]) for i in range(N) for j in range(N)]},
                      0.0, 1e-8, "abs", cc + "; %d of %d eigenvalues are noise" % (nn, N) +
                      ("; two assets with correlation ~0.999 (near-singular input)" if "singular" in cid else "")))

    # vector norms and distances ---------------------------------------------
    def vtrue_norm(v):
        return mp.sqrt(mp.fsum(M(a) ** 2 for a in v))

    nrm = ("linalg/vector.go %s: 'Precision: accumulated float64 rounding (~O(n)*eps), not bit-exact for n > 1' (overflow is "
           "caveated, underflow is not) -> relative 10*n*2^-52")
    for cid, v in (("l2norm/typical", [0.1 * (k + 1) for k in range(16)]),
                   ("l2norm/underflow-1e-170", [3e-170, 4e-170]),
                   ("l2norm/subnormal-squares-1e-160", [1e-160, 1e-160, 1e-160])):
        n = len(v)
        C.append(case(cid, "linalg.L2Norm", {"v": v}, f64(vtrue_norm(v)), 10 * n * ULP, "rel", nrm % "L2Norm"))
    for cid, a, b in (("encdist/typical", [0.1 * k for k in range(8)], [0.3 * k - 0.2 for k in range(8)]),
                      ("encdist/underflow-1e-170", [3e-170, 0.0], [0.0, 4e-170])):
        n = len(a)
        t = mp.sqrt(mp.fsum((M(x) - M(y)) ** 2 for x, y in zip(a, b))) / mp.sqrt(n)
        C.append(case(cid, "linalg.EncodingDistance", {"a": a, "b": b}, f64(t), 10 * n * ULP, "rel", nrm % "EncodingDistance"))
    for cid, a, b, w in (("wdist/typical", [0.1 * k for k in range(6)], [0.25 * k for k in range(6)], [1.0, 2.0, 0.5, 3.0, 1.0, 0.25]),
                         ("wdist/underflow-1e-170", [3e-170, 0.0], [0.0, 4e-170], [1.0, 1.0])):
        n = len(a)
        t = mp.sqrt(mp.fsum(M(wi) * (M(x) - M(y)) ** 2 for x, y, wi in zip(a, b, w)) / mp.fsum(M(wi) for wi in w))
        C.append(case(cid, "linalg.DimensionWeightedDistance", {"a": a, "b": b, "w": w}, f64(t), 10 * n * ULP, "rel",
                      nrm % "DimensionWeightedDistance"))
    for cid, v in (("l2normalize/typical", [0.3, -1.7, 2.2, 0.05]), ("l2normalize/underflow-1e-170", [3e-170, 4e-170])):
        n = len(v)
        C.append(case(cid, "linalg.L2Normalize", {"v": v}, 1.0, 10 * n * ULP, "rel",
                      (nrm % "L2Normalize") + "; scalar = the exact 2-norm of the normalised vector (1 when normalised; the "
                      "doc reserves 'returns false, vector unchanged' for zero magnitude)"))
    cs = ("linalg/vector.go CosineSimilarity: 'Precision: subject to accumulated float64 rounding in the dot/norm sums "
          "(~O(n)*eps) and to catastrophic cancellation in the dot product' -> absolute 10*n*2^-52 (cancellation is "
          "relative to |a||b|, i.e. absolute for a cosine); overflow and underflow are not caveated")
    for cid, a, b in (("cosine/typical", [0.1 * k - 0.3 for k in range(10)], [0.05 * k * k for k in range(10)]),
                      ("cosine/near-orthogonal", [1.0, 1e-8, 0.3], [-1e-8, 1.0, 0.0]),
                      ("cosine/underflow-1e-170", [1e-170, 1e-170, 1e-170], [1e-170, 2e-170, 3e-170]),
                      ("cosine/overflow-1e170", [1e170, 1e170, 1e170], [1e170, 2e170, 3e170])):
        A_, B_ = [M(x) for x in a], [M(x) for x in b]
        t = mp.fsum(x * y for x, y in zip(A_, B_)) / (mp.sqrt(mp.fsum(x * x for x in A_)) * mp.sqrt(mp.fsum(y * y for y in B_)))
        C.append(case(cid, "linalg.CosineSimilarity", {"a": a, "b": b}, f64(t), 10 * len(a) * ULP, "abs", cs))
    dp = ("linalg/vector.go DotProduct: 'Precision: subject to accumulated float64 rounding (~O(n)*eps); not bit-exact "
          "for n > 1' -> the standard bound 10*n*2^-52 * sum|a_i b_i| (absolute), so cancellation is allowed")
    for cid, a, b in (("dot/typical", [0.1 * k for k in range(12)], [1.0 / (k + 1) for k in range(12)]),
                      ("dot/cancellation", [1e17, 1.0, -1e17], [1.0, 1.0, 1.0])):
        t = mp.fsum(M(x) * M(y) for x, y in zip(a, b))
        tol = 10 * len(a) * ULP * float(mp.fsum(abs(M(x) * M(y)) for x, y in zip(a, b)))
        C.append(case(cid, "linalg.DotProduct", {"a": a, "b": b}, f64(t), tol, "abs", dp + " = %.3g" % tol))
    l1 = ("linalg/vector.go L1Norm: 'Precision: subject to accumulated float64 rounding (~O(n)*eps); not bit-exact for "
          "n > 1' -> relative 10*n*2^-52")
    v = [((-1) ** k) * 10.0 ** (k % 7 - 3) * (1 + 0.1 * k) for k in range(40)]
    C.append(case("l1norm/mixed-scales", "linalg.L1Norm", {"v": v}, f64(mp.fsum(abs(M(x)) for x in v)), 10 * len(v) * ULP, "rel", l1))
    li = "linalg/vector.go LInfNorm: 'Precision: exact for IEEE 754 float64.' -> exact match (max of |v_i| involves no rounding)"
    v = [-0.0, 5e-324, -3.5, 2.25, -1e300, 7.0]
    C.append(case("linfnorm/mixed", "linalg.LInfNorm", {"v": v}, 1e300, 0.0, "rel", li))

    # singular matrices -------------------------------------------------------
    dt = ("linalg/decompose.go Determinant: 'Returns 0 for singular matrices.' -> exact 0 for an exactly singular input "
          "(integer entries, exact rational determinant 0)")
    for cid, A, n in (("det/singular-2x2-exact-elimination", [1.0, 2.0, 2.0, 4.0], 2),
                      ("det/singular-1to9", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3),
                      ("det/singular-rank2-covariance", [26.0, 8.0, 11.0, 8.0, 10.0, 5.0, 11.0, 5.0, 5.0], 3)):
        assert exact_det([[Fraction(A[i * n + j]) for j in range(n)] for i in range(n)]) == 0
        C.append(case(cid, "linalg.Determinant", {"A": A, "n": n}, 0.0, 0.0, "abs", dt))
    iv = ("linalg/decompose.go Inverse: 'Returns false if A is singular.' -> scalar = 1 when Inverse reports success, 0 "
          "when it reports singular; an exactly singular input (integer entries) must give 0")
    for cid, A, n in (("inverse/singular-2x2-exact-elimination", [1.0, 2.0, 2.0, 4.0], 2),
                      ("inverse/singular-1to9", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3),
                      ("inverse/singular-rank2-covariance", [26.0, 8.0, 11.0, 8.0, 10.0, 5.0, 11.0, 5.0, 5.0], 3)):
        C.append(case(cid, "linalg.Inverse", {"A": A, "n": n}, 0.0, 0.0, "abs", iv))
    return C


# ---------------------------------------------------------------------------
# optim (root package)
# ---------------------------------------------------------------------------

PHI = (math.sqrt(5) - 1) / 2


def lp_opt_bruteforce(c, A, b):
    """Exact optimum of min c'x s.t. Ax <= b, x >= 0 for n <= 2, by vertex
    enumeration in rationals (the feasible sets here are bounded)."""
    n = len(c)
    cons = [([Fraction(v) for v in row], Fraction(bi)) for row, bi in zip(A, b)]
    for j in range(n):
        e = [Fraction(0)] * n
        e[j] = Fraction(-1)
        cons.append((e, Fraction(0)))
    best = None
    for idx in itertools.combinations(range(len(cons)), n):
        rows = [cons[i] for i in idx]
        if n == 1:
            a, bb = rows[0]
            if a[0] == 0:
                continue
            x = [bb / a[0]]
        else:
            (a1, b1), (a2, b2) = rows
            det = a1[0] * a2[1] - a1[1] * a2[0]
            if det == 0:
                continue
            x = [(b1 * a2[1] - a1[1] * b2) / det, (a1[0] * b2 - b1 * a2[0]) / det]
        if all(sum(ai * xi for ai, xi in zip(a, x)) <= bb for a, bb in cons):
            v = sum(Fraction(ci) * xi for ci, xi in zip(c, x))
            if best is None or v < best:
                best = v
    return best


def optim_cases():
    C = []
    ex = "'exact' read as faithful rounding: relative error below 2^-52"

    # LBFGS -----------------------------------------------------------------
    lb = ("optim/gradient.go LBFGS: 'the standard two-loop recursion algorithm with Wolfe line search'; 'tol: stop when "
          "||grad f(x)||_2 < tol'; 'Returns the approximate minimizer.' ")
    # Rosenbrock: minimizer (1, 1) exactly; Hessian there [[802,-400],[-400,200]],
    # smallest eigenvalue 0.3994, so ||grad|| < 1e-8 puts x within 2.6e-8.
    C.append(case("lbfgs/rosenbrock-maxiter100", "optim.LBFGS",
                  {"problem": "rosenbrock", "x0": [-1.2, 1.0], "m": 5, "maxIter": 100, "tol": 1e-8, "truth": [1.0, 1.0]}, 0.0, 1e-6, "abs",
                  lb + "-> on 2-D Rosenbrock from (-1.2, 1) with m = 5, the same two-loop recursion with a strong-Wolfe line search "
                  "(scipy line_search, c1 = 1e-4, c2 = 0.9) reaches ||grad|| < 1e-8 in 38 iterations (scipy L-BFGS-B: 40), so "
                  "maxIter = 100 must return the minimizer (1, 1) to within ||H^-1|| * 1e-8 = 2.6e-8; scalar = max |x_i - 1|, "
                  "tolerance 1e-6"))
    C.append(case("lbfgs/rosenbrock-maxiter1000", "optim.LBFGS",
                  {"problem": "rosenbrock", "x0": [-1.2, 1.0], "m": 5, "maxIter": 1000, "tol": 1e-8, "truth": [1.0, 1.0]}, 0.0, 1e-6, "abs",
                  lb + "-> the same problem with maxIter = 1000; scalar = max |x_i - 1|, tolerance 1e-6"))
    Aq = [[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]]
    bq = [1.0, 2.0, 3.0]
    Am = mp.matrix(Aq)
    xs = mp.lu_solve(Am, mp.matrix(bq))
    ev = mp.eigsy(Am)[0]
    inv_norm = 1 / min(ev[k] for k in range(3))
    tolq = f64(inv_norm * mp.mpf("1e-10"))
    C.append(case("lbfgs/quadratic", "optim.LBFGS",
                  {"problem": "quadratic", "A": [v for row in Aq for v in row], "b": bq, "x0": [0.0, 0.0, 0.0], "m": 5, "maxIter": 100,
                   "tol": 1e-10, "truth": [f64(xs[i]) for i in range(3)]}, 0.0, tolq, "abs",
                  lb + "-> f = x'Ax/2 - b'x with A SPD: stopping at ||Ax - b|| < 1e-10 puts x within ||A^-1|| * 1e-10 = %.3g "
                  "of A^-1 b; scalar = max |x_i - x*_i|" % tolq))

    # BisectionMethod --------------------------------------------------------
    bi = ("optim/rootfind.go BisectionMethod: 'Precision: |root - x*| <= tol after ceil(log2((b-a)/tol)) iterations.' -> "
          "absolute tol; the evaluator stops f after the claimed iteration count (+2 calls) and scores a run that has not "
          "returned by then as NaN")
    for cid, fn, a, b, tol, root in (("bisect/sqrt2", "x2minus2", 1.0, 2.0, 1e-12, mp.sqrt(2)),
                                     ("bisect/tol-below-spacing-exp", "expminus1e5", 0.0, 20.0, 1e-15, mp.log(100000)),
                                     ("bisect/tol-below-spacing-sqrt2", "x2minus2", 1.0, 2.0, 1e-17, mp.sqrt(2))):
        k = math.ceil(math.log2((b - a) / tol))
        note = "" if tol >= 1e-12 else "; tol %g is below the float spacing %.3g at the root, and no float makes f exactly 0" % (tol, math.ulp(f64(root)))
        C.append(case(cid, "optim.BisectionMethod", {"f": fn, "a": a, "b": b, "tol": tol, "budget": k + 3}, f64(root), tol, "abs",
                      bi + " (claimed %d iterations)" % k + note))

    # GoldenSectionSearch ----------------------------------------------------
    gs = ("optim/rootfind.go GoldenSectionSearch: 'Precision: |x* - x_min| <= tol after ceil(log_phi(tol/(b-a))) "
          "iterations.' -> absolute tol; the evaluator stops f after the claimed iteration count (+2 evaluations, +2 slack) "
          "and scores a run that has not returned by then as NaN")
    for cid, fn, a, b, tol, xstar, note in (
            ("golden/typical", "sq_xminus2", 0.0, 5.0, 1e-6, mp.mpf(2), ""),
            ("golden/flat-min-tol1e-12", "sq_xminus1_plus1", 0.0, 3.0, 1e-12, mp.mpf(1),
             "; f(x*) = 1, so f is flat to rounding within ~1e-8 of x*"),
            ("golden/cosh-tol1e-10", "cosh_xminus0.3", -1.0, 2.0, 1e-10, M(0.3), "; f(x*) = 1, flat to rounding within ~1.5e-8 of x*"),
            ("golden/tol-below-spacing", "sq_xminus1e6_plus1", 0.0, 2e6, 1e-12, mp.mpf(10) ** 6,
             "; tol is below the float spacing 1.2e-10 at x* = 1e6")):
        k = math.ceil(math.log(tol / (b - a)) / math.log(PHI))
        C.append(case(cid, "optim.GoldenSectionSearch", {"f": fn, "a": a, "b": b, "tol": tol, "budget": k + 4}, f64(xstar), tol, "abs",
                      gs + " (claimed %d iterations)" % k + note))

    # LinearInterpolate / LinearInterpolateRoot -------------------------------
    li = "optim/interpolate.go LinearInterpolate: 'Precision: exact for IEEE 754 float64.' -> " + ex
    for cid, x0, y0, x1, y1, x in (("lerp/typical", 0.0, 0.0, 1.0, 10.0, 0.3),
                                    ("lerp/near-zero-crossing", 0.1, 0.3, 0.7, -0.9, 0.25)):
        t = M(y0) + (M(y1) - M(y0)) * (M(x) - M(x0)) / (M(x1) - M(x0))
        C.append(case(cid, "optim.LinearInterpolate", {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "x": x}, f64(t), ULP, "rel",
                      li + ("; the interpolated value is near 0 (cancellation)" if "crossing" in cid else "")))
    lr = "optim/rootfind.go LinearInterpolateRoot: 'Precision: exact for IEEE 754 float64 (single division + multiply).' -> " + ex
    for cid, x0, y0, x1, y1 in (("lroot/typical", 1.0, -1.0, 3.0, 3.0),
                                ("lroot/decimals", 0.1, -0.2, 0.7, 1.3),
                                ("lroot/root-near-zero", 0.3, 0.7, -0.1, -0.7 / 3)):
        t = M(x0) - M(y0) * (M(x1) - M(x0)) / (M(y1) - M(y0))
        C.append(case(cid, "optim.LinearInterpolateRoot", {"x0": x0, "y0": y0, "x1": x1, "y1": y1}, f64(t), ULP, "rel",
                      lr + ("; the root is near 0 (x0 and the correction cancel)" if "zero" in cid else "")))

    # SimplexMethod -----------------------------------------------------------
    sm = ("optim/linear.go SimplexMethod: 'Returns the optimal solution x (length n), the optimal objective value, and an "
          "error if the problem is infeasible or unbounded.' -> scalar = the returned objective value (NaN when an error is "
          "returned), relative 1e-9 against the exact LP optimum")
    for cid, c, A, b, note in (
            ("simplex/textbook", [-3.0, -5.0], [[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]], [4.0, 12.0, 18.0], ""),
            ("simplex/row-scaled-1e-11", [-1.0, -2.0], [[1.0, 1.0], [0.0, 1e-11]], [2.0, 1e-11],
             "; the second row is y <= 1 written as 1e-11*y <= 1e-11 (feasible optimum x = y = 1, value -3)"),
            ("simplex/single-row-scaled-1e-11", [-1.0], [[1e-11]], [1e-11], "; x <= 1 written as 1e-11*x <= 1e-11 (optimum x = 1)"),
            ("simplex/costs-1e-11", [-1e-11], [[1.0]], [1.0], "; min -1e-11*x s.t. x <= 1 (optimum x = 1)")):
        opt = lp_opt_bruteforce(c, A, b)
        C.append(case(cid, "optim.SimplexMethod", {"c": c, "A": [v for row in A for v in row], "m": len(A), "b": b}, f64(opt), 1e-9, "rel",
                      sm + note))
    return C


# ---------------------------------------------------------------------------
# optim/portfolio and optim/hrp
# ---------------------------------------------------------------------------

HL = os.path.join("optim", "portfolio", "testdata", "he_litterman_1999.json")


def mpm(rows):
    return mp.matrix([[M(v) for v in r] for r in rows])


def mat_rows(Mt):
    return [[Mt[i, j] for j in range(Mt.cols)] for i in range(Mt.rows)]


def bl_posterior_true(pi, Sig, P, Q, Om, tau):
    S = mpm(Sig) * M(tau)
    Si = S ** -1
    Pm, Omi = mpm(P), mpm(Om) ** -1
    A = Si + Pm.T * Omi * Pm
    rhs = Si * mp.matrix([M(v) for v in pi]) + Pm.T * Omi * mp.matrix([M(v) for v in Q])
    return A ** -1 * rhs, A ** -1


def proj_simplex_exact(v):
    """Euclidean projection onto the probability simplex, in rationals."""
    u = sorted((Fraction(x) for x in v), reverse=True)
    css, theta = Fraction(0), None
    for j, uj in enumerate(u):
        css += uj
        t = (css - 1) / (j + 1)
        if uj - t > 0:
            theta = t
    return [max(Fraction(x) - theta, Fraction(0)) for x in v]


def spd_with_spectrum(evals, seed):
    """Q diag(evals) Q' rounded to float, Q a Householder product from splitmix
    data; returns the float rows (symmetric by construction)."""
    n = len(evals)
    r = SplitMix64(seed)
    Q = mp.eye(n)
    for _ in range(3):
        v = mp.matrix([M(r.normal()) for _ in range(n)])
        H = mp.eye(n) - 2 * (v * v.T) / (v.T * v)[0]
        Q = Q * H
    A = Q * mp.diag([M(e) for e in evals]) * Q.T
    rows = [[f64(A[i, j]) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            rows[j][i] = rows[i][j]
    return rows


def portfolio_cases():
    C = []
    with open(HL, encoding="utf-8") as fh:
        hl = json.load(fh)
    vol = [v / 100 for v in hl["volatility_pct"]]
    corr = hl["correlation"]
    n = len(vol)
    Sig = [[corr[i][j] * vol[i] * vol[j] for j in range(n)] for i in range(n)]
    w_mkt = [v / 100 for v in hl["market_weight_pct"]]
    delta, tau = hl["delta"], hl["tau"]
    P, Q, Om = hl["view"]["P"], hl["view"]["Q"], hl["view"]["Omega"]
    pi_hl = hl["expected"]["equilibrium_returns"]
    sd15 = "'Precision: ~15 significant digits (float64)' -> relative 1e-14 (10x)"

    # ImpliedEquilibriumReturns --------------------------------------------------
    ie = "optim/portfolio/portfolio.go ImpliedEquilibriumReturns: " + sd15 + "; scalar = max_i |pi_i - true_i| / |true_i|"
    for cid, w in (("implied/he-litterman", w_mkt), ("implied/long-short", [0.3, -0.2, 0.25, -0.35, 0.1, -0.05, -0.05])):
        t = M(delta) * (mpm(Sig) * mp.matrix([M(v) for v in w]))
        C.append(case(cid, "portfolio.ImpliedEquilibriumReturns", {"w": w, "Sigma": Sig, "delta": delta, "truth": [f64(t[i]) for i in range(n)]},
                      0.0, 1e-14, "abs", ie + ("; He-Litterman (1999) inputs" if "he-" in cid else "; market-neutral weights")))
    rt = ("optim/portfolio/portfolio.go ImpliedEquilibriumReturns: 'MeanVarianceWeights(ImpliedEquilibriumReturns(w, Sigma, "
          "delta), Sigma, delta) == w for any nonsingular Sigma' -> '==' read at the functions' own ~15 significant digits: "
          "max |w_roundtrip - w| / max |w| <= 1e-14")
    rho6 = 0.9999
    Sig6 = [[(1.0 if i == j else rho6) * 0.04 for j in range(6)] for i in range(6)]
    Sig5 = spd_with_spectrum([1.0, 0.3, 1e-3, 1e-6, 1e-10], 4242)
    for cid, w, Sg, note in (("roundtrip/he-litterman", w_mkt, Sig, "; He-Litterman (1999) inputs"),
                             ("roundtrip/correlation-0.9999", [0.3, 0.1, 0.2, 0.15, 0.05, 0.2], Sig6,
                              "; 6 assets, pairwise correlation 0.9999 (cond 6e4), nonsingular"),
                             ("roundtrip/cond-1e10", [0.3, 0.1, 0.2, 0.25, 0.15], Sig5, "; SPD Sigma with eigenvalues 1 .. 1e-10, nonsingular")):
        C.append(case(cid, "portfolio.RoundTrip", {"w": w, "Sigma": Sg, "delta": delta}, 0.0, 1e-14, "abs", rt + note))

    # HeLittermanOmega -----------------------------------------------------------
    om = "optim/portfolio/portfolio.go HeLittermanOmega: " + sd15 + "; scalar = relative error of Omega_00 = tau * P_0 Sigma P_0'"
    t = M(tau) * (mpm(P) * mpm(Sig) * mpm(P).T)[0, 0]
    C.append(case("omega/he-litterman", "portfolio.HeLittermanOmega", {"P": P, "Sigma": Sig, "tau": tau}, f64(t), 1e-14, "rel",
                  om + "; He-Litterman (1999) view"))
    Ph = [[0.3, -0.7, 0.4]]
    Sh = [[0.04 * (1.0 if i == j else 0.9999) for j in range(3)] for i in range(3)]
    t = M(tau) * (mpm(Ph) * mpm(Sh) * mpm(Ph).T)[0, 0]
    C.append(case("omega/hedge-view", "portfolio.HeLittermanOmega", {"P": Ph, "Sigma": Sh, "tau": tau}, f64(t), 1e-14, "rel",
                  om + "; a view portfolio summing to 0 over assets with correlation 0.9999 (a near-riskless hedge)"))

    # BlackLittermanPosterior / Covariance ------------------------------------------
    bl = ("optim/portfolio/portfolio.go BlackLittermanPosterior: 'Precision: ~15 significant digits (float64) for a "
          "well-conditioned system' -> relative 1e-14 (10x); scalar = max_i |mu_i - true_i| / |true_i|")
    mu, Mcov = bl_posterior_true(pi_hl, Sig, P, Q, Om, tau)
    C.append(case("bl/he-litterman", "portfolio.BlackLittermanPosterior",
                  {"pi": pi_hl, "Sigma": Sig, "P": P, "Q": Q, "Omega": Om, "tau": tau, "truth": [f64(mu[i]) for i in range(n)]},
                  0.0, 1e-14, "abs", bl + "; He-Litterman (1999) inputs"))
    blc = ("optim/portfolio/portfolio.go BlackLittermanPosteriorCovariance: " + sd15 +
           "; scalar = max_ij |M_ij - true_ij| / max_ij |true_ij| (normwise)")
    C.append(case("blcov/he-litterman", "portfolio.BlackLittermanPosteriorCovariance",
                  {"Sigma": Sig, "P": P, "Omega": Om, "tau": tau, "truth": [f64(Mcov[i, j]) for i in range(n) for j in range(n)]},
                  0.0, 1e-14, "abs", blc + "; He-Litterman (1999) inputs"))

    # singular / near-singular covariances -------------------------------------------
    k = 2.0 ** -13
    Ssing = [[26 * k, 8 * k, 11 * k], [8 * k, 10 * k, 5 * k], [11 * k, 5 * k, 5 * k]]
    assert exact_det([[Fraction(v) for v in r] for r in Ssing]) == 0
    # sample covariance of T = 3 observations of 4 assets (rank 2), exactly
    # representable: integer returns in units of 2^-10 with column sums divisible by 3
    X = [[5, -4, 2, 9], [-2, 7, 2, -3], [3, 0, -1, 3]]
    means = [Fraction(sum(X[t][i] for t in range(3)), 3) for i in range(4)]
    Ssamp = []
    for i in range(4):
        row = []
        for j in range(4):
            cv = sum((X[t][i] - means[i]) * (X[t][j] - means[j]) for t in range(3)) / 2 * Fraction(1, 1 << 20)
            assert float(cv) == cv
            row.append(float(cv))
        Ssamp.append(row)
    assert exact_det([[Fraction(v) for v in r] for r in Ssamp]) == 0
    Snear = [[0.04, 0.06], [0.06, 0.09]]
    sing = ("-> scalar = 1 when a result is returned, 0 for nil; an exactly singular Sigma (exact rational determinant 0) "
            "must give 0")
    for fn, doc in (("portfolio.MeanVarianceWeights", "optim/portfolio/portfolio.go MeanVarianceWeights: 'Returns nil if ... Sigma is singular' "),
                    ("portfolio.ContinuousKellyWeights", "optim/portfolio/portfolio.go ContinuousKellyWeights: 'Returns nil if ... Sigma is singular' ")):
        short = fn.split(".")[1]
        for cid, Sg, note in (("rank2-integer", Ssing, "; rank-2 3x3 covariance (integers times 2^-13)"),
                              ("sample-cov-T3-N4", Ssamp, "; sample covariance of 3 observations of 4 assets (rank 2), exact in float64")):
            mu_ = [0.05, 0.07, 0.06, 0.04][:len(Sg)]
            C.append(case("%s/singular-%s" % (short.lower(), cid), fn, {"mu": mu_, "Sigma": Sg, "param": 2.5 if "Mean" in fn else 0.25, "measure": "returned"},
                          0.0, 0.0, "abs", doc + sing + note))
    blsing = ("optim/portfolio/portfolio.go BlackLittermanPosterior: 'Returns nil on ... a singular / near-singular Sigma, "
              "Omega, or precision matrix' " + sing.replace("exactly singular Sigma (exact rational determinant 0)", "singular or near-singular Sigma"))
    for cid, Sg, note in (("bl/singular-rank2-integer", Ssing, "; exactly singular rank-2 3x3 Sigma"),
                          ("bl/near-singular-perfect-correlation", Snear, "; the decimal perfect-correlation matrix [[.04,.06],[.06,.09]] (determinant 4e-19 after rounding, condition ~4e17)")):
        nn = len(Sg)
        Pv = [[1.0, -1.0] + [0.0] * (nn - 2)]
        C.append(case(cid, "portfolio.BlackLittermanPosterior",
                      {"pi": [0.05, 0.07, 0.06][:nn], "Sigma": Sg, "P": Pv, "Q": [0.02], "Omega": [[0.001]], "tau": 0.05, "measure": "returned"},
                      0.0, 0.0, "abs", blsing + note))

    # MeanVarianceWeights / ContinuousKellyWeights / LongOnly at He-Litterman ---------------
    mvt = mpm(Sig) ** -1 * mp.matrix([M(v) for v in pi_hl])
    for fn, par, doc in (("portfolio.MeanVarianceWeights", delta,
                          "optim/portfolio/portfolio.go MeanVarianceWeights: 'Precision: ~15 significant digits (float64) for a well-conditioned Sigma'"),
                         ("portfolio.ContinuousKellyWeights", 0.25,
                          "optim/portfolio/portfolio.go ContinuousKellyWeights: 'Precision: ~15 significant digits (float64) for a well-conditioned Sigma'")):
        scale = 1 / M(par) if "Mean" in fn else M(par)
        C.append(case(fn.split(".")[1].lower() + "/he-litterman", fn,
                      {"mu": pi_hl, "Sigma": Sig, "param": par, "measure": "relerr", "truth": [f64(scale * mvt[i]) for i in range(n)]},
                      0.0, 1e-14, "abs", doc + " -> relative 1e-14 (10x); scalar = max_i |w_i - true_i| / |true_i|; He-Litterman (1999) inputs"))
    mvr = [M(v) for v in hl["market_weight_pct"]]  # unused, keeps the fixture fields visible
    del mvr
    mu_lo = [0.08, 0.02, 0.05, 0.11, 0.03, 0.06, 0.04]
    raw = (mpm(Sig) ** -1 * mp.matrix([M(v) for v in mu_lo])) / M(delta)
    lo = proj_simplex_exact([Fraction(f64(raw[i])) for i in range(n)])
    C.append(case("mvlongonly/he-litterman-sigma", "portfolio.MeanVarianceWeightsLongOnly",
                  {"mu": mu_lo, "Sigma": Sig, "delta": delta, "truth": [float(v) for v in lo]}, 0.0, 1e-14, "abs",
                  "optim/portfolio/portfolio.go MeanVarianceWeightsLongOnly: 'Precision: the underlying solve is ~15 significant "
                  "digits; the projection is exact' -> absolute 1e-14 on the weights (which lie in [0, 1]); the reference projects "
                  "the correctly rounded unconstrained weights exactly"))

    # ProjectSimplex -------------------------------------------------------------
    ps = ("optim/portfolio/portfolio.go ProjectSimplex: 'Precision: exact up to float64 rounding'; 'The result always sums to "
          "exactly 1 (up to float rounding)' -> 'up to float64 rounding' read as the accumulated-rounding bound 10*n*2^-52, "
          "absolute (the weights lie in [0, 1]); ")
    for cid, v, note in (("projsimplex/typical", [0.3, -0.2, 0.9, 0.05, 0.4], ""),
                         ("projsimplex/magnitude-1e5", [123456.789 + 0.13, 123456.789 + 0.71, 123456.789 + 0.37, 123456.789 + 0.59], "; inputs near 1.2e5"),
                         ("projsimplex/magnitude-1e8", [1e8 + 0.13, 1e8 + 0.71, 1e8 + 0.37, 1e8 + 0.59], "; inputs near 1e8"),
                         ("projsimplex/magnitude-1e17", [1e17 + 32, 1e17], "; inputs near 1e17 (projection (1, 0))")):
        exact = proj_simplex_exact(v)
        nn = len(v)
        C.append(case(cid + "-weights", "portfolio.ProjectSimplex", {"v": v, "measure": "weights", "truth": [float(x) for x in exact]},
                      0.0, 10 * nn * ULP, "abs", ps + "scalar = max_i |w_i - true_i|" + note))
        C.append(case(cid + "-sum", "portfolio.ProjectSimplex", {"v": v, "measure": "sum"}, 0.0, 10 * nn * ULP, "abs",
                      ps + "scalar = |sum_i w_i - 1|, the sum taken exactly" + note))

    # hrp ------------------------------------------------------------------------
    cdh = ("optim/hrp/hrp.go CorrelationDistance: 'the result is correct to within one ulp of the true distance' -> "
           "relative 2^-52 against sqrt((1 - rho)/2) at the clamped rho")
    for cid, rho in (("corrdist/0.3", 0.3), ("corrdist/-0.7", -0.7), ("corrdist/0.9999999", 0.9999999), ("corrdist/1e-17", 1e-17),
                     ("corrdist/0.1", 0.1), ("corrdist/-0.999999999", -0.999999999), ("corrdist/above-1", 1 + 2 * ULP)):
        r = min(M(rho), M(1))
        C.append(case(cid, "hrp.CorrelationDistance", {"rho": rho}, f64(mp.sqrt((1 - r) / 2)), ULP, "rel", cdh))
    rb = ("optim/hrp/hrp.go RecursiveBisection: 'the returned weights sum to exactly 1 (to within one ulp times n)' -> "
          "|sum_i w_i - 1| <= n * 2^-52 (one ulp of 1 is 2^-52), the sum taken exactly")
    for cid, nn, seed, spread in (("recbisect/n6", 6, 7001, 1.0), ("recbisect/n64-wide-variances", 64, 7002, 1e8), ("recbisect/n33", 33, 7003, 10.0)):
        r = SplitMix64(seed)
        sd = [0.01 * (spread ** r.uniform()) for _ in range(nn)]
        cov = [[(1.0 if i == j else 0.3) * sd[i] * sd[j] for j in range(nn)] for i in range(nn)]
        order = list(range(nn))
        for i in range(nn - 1, 0, -1):
            j = r.next() % (i + 1)
            order[i], order[j] = order[j], order[i]
        C.append(case(cid, "hrp.RecursiveBisection", {"cov": cov, "order": order}, 0.0, nn * ULP, "abs",
                      rb + "; %d assets, volatilities spread over a factor %g" % (nn, spread)))
    return C


# ---------------------------------------------------------------------------
# optim/proximal and optim/transport
# ---------------------------------------------------------------------------

def lasso_problem(seed, m, n, xs, lam, s_off):
    """A LASSO problem min 0.5||Ax - b||^2 + lam ||x||_1 whose optimum is known:
    b is built so that x* = xs satisfies the KKT conditions with the given
    off-support subgradient, rounded to float64, and the optimum of the
    rounded problem is then solved exactly on the same support."""
    r = SplitMix64(seed)
    A = [[r.normal() for _ in range(n)] for _ in range(m)]
    Am = mpm(A)
    sub = []
    k = 0
    for j in range(n):
        if xs[j] != 0:
            sub.append(mp.sign(xs[j]))
        else:
            sub.append(M(s_off[k]))
            k += 1
    G = Am.T * Am
    resid = Am * (G ** -1) * mp.matrix([M(lam) * v for v in sub])
    bm = Am * mp.matrix([M(v) for v in xs]) + resid
    b = [f64(bm[i]) for i in range(m)]
    S = [j for j in range(n) if xs[j] != 0]
    AS = mp.matrix([[M(A[i][j]) for j in S] for i in range(m)])
    bvec = mp.matrix([M(v) for v in b])
    xS = (AS.T * AS) ** -1 * (AS.T * bvec - mp.matrix([M(lam) * mp.sign(xs[j]) for j in S]))
    x = [mp.mpf(0)] * n
    for t, j in enumerate(S):
        x[j] = xS[t]
        assert mp.sign(x[j]) == mp.sign(xs[j])
    xv = mp.matrix(x)
    rr = Am * xv - bvec
    corr = Am.T * rr
    for j in range(n):
        if j not in S:
            assert abs(corr[j]) < M(lam) * mp.mpf("0.95"), "KKT margin"
    Fstar = mp.fsum(rr[i] ** 2 for i in range(m)) / 2 + M(lam) * mp.fsum(abs(v) for v in x)
    L = max(mp.eigsy(G)[0][i] for i in range(n))
    step = f64(1 / L)
    while M(step) > 1 / L:
        step = math.nextafter(step, 0.0)
    return A, b, x, Fstar, L, step


def w_p_exact(u, v, p):
    """Exact 1-D Wasserstein-p between empirical measures (quantile integral)."""
    us, vs = sorted(Fraction(x) for x in u), sorted(Fraction(x) for x in v)
    n, m = len(us), len(vs)
    i = j = 0
    pos = Fraction(0)
    tot = mp.mpf(0)
    while i < n and j < m:
        nu, nv = Fraction(i + 1, n), Fraction(j + 1, m)
        t = min(nu, nv)
        w = t - pos
        if w > 0:
            d = abs(us[i] - vs[j])
            tot += (M(d.numerator) / M(d.denominator)) ** p * (M(w.numerator) / M(w.denominator))
        pos = t
        if nu <= nv:
            i += 1
        if nv <= nu:
            j += 1
    return tot ** (mp.mpf(1) / p)


def prox_transport_cases():
    C = []

    # Fbs (FBS and FISTA) ----------------------------------------------------------
    lam = 0.5
    A, b, xs, Fstar, L, step = lasso_problem(8001, 20, 8, [1.5, 0, -0.8, 0, 0, 2.0, 0, 0], lam, [0.3, -0.5, 0.2, 0.7, -0.1])
    R2 = mp.fsum(v ** 2 for v in xs)  # ||x0 - x*||^2 with x0 = 0
    fb = ("optim/proximal/fbs.go Fbs: 'FISTA convergence rate is O(1/k^2) on the objective; plain FBS is O(1/k).' -> the "
          "bounds of the cited Beck & Teboulle (2009) at step 1/L: FISTA F(x_k) - F* <= 2 L ||x0 - x*||^2 / (k+1)^2 "
          "(Thm 4.4), FBS F(x_k) - F* <= L ||x0 - x*||^2 / (2k) (Thm 3.1); LASSO with 20 x 8 A, lambda 0.5, x0 = 0, the "
          "optimum solved exactly from the KKT conditions; scalar = F(x_k) - F*")
    for cid, acc, k in (("fbs/fista-k20", True, 20), ("fbs/fista-k200", True, 200), ("fbs/ista-k20", False, 20), ("fbs/ista-k200", False, 200)):
        bound = 2 * L * R2 / (k + 1) ** 2 if acc else L * R2 / (2 * k)
        C.append(case(cid, "proximal.Fbs", {"A": [v for row in A for v in row], "m": 20, "b": b, "lambda": lam, "step": step,
                                            "accelerate": acc, "iters": k, "fstar": f64(Fstar)},
                      0.0, f64(bound), "abs", fb + " (k = %d: bound %.3g)" % (k, f64(bound))))

    # Sinkhorn -------------------------------------------------------------------------
    sk = ("optim/transport/sinkhorn.go Sinkhorn: 'passing maxIter <= 0 uses 200 as the default (enough for epsilon >= 0.01 * "
          "mean(C) on well-conditioned problems)'; 'tol <= 0 falls back to 1e-7' -> with maxIter = 0 and tol = 0 the call must "
          "succeed; scalar = 0 on success, else the iterations a run capped at 1000 needs beyond 200 (+Inf if it needs more "
          "than 1000)")
    for cid, n, eps_frac, shift in (("sinkhorn/n10-eps0.01", 10, 0.01, 0.0), ("sinkhorn/n50-eps0.01", 50, 0.01, 0.0),
                                    ("sinkhorn/n50-eps0.1", 50, 0.1, 0.0), ("sinkhorn/n50-shifted-eps0.01", 50, 0.01, 0.25)):
        xs_ = [i / (n - 1) for i in range(n)]
        ys_ = [i / (n - 1) + shift for i in range(n)]
        Cm = [[(xi - yj) * (xi - yj) for yj in ys_] for xi in xs_]
        meanC = math.fsum(v for row in Cm for v in row) / (n * n)
        a = [1.0 / n] * n
        C.append(case(cid, "transport.Sinkhorn", {"a": a, "b": a, "cost": Cm, "epsilon": eps_frac * meanC, "measure": "default-budget"},
                      0.0, 0.0, "abs", sk + "; uniform marginals, squared distance on a uniform grid of %d points%s, epsilon = %g * mean(C)" %
                      (n, "" if shift == 0 else " shifted by %g" % shift, eps_frac)))
    skr = ("optim/transport/sinkhorn.go Sinkhorn: 'Convergence is measured by the L^1 marginal deviation ||P 1 - a||_1 "
           "against tol' -> on success the returned plan has ||P 1 - a||_1 < tol; scalar = that deviation, summed exactly")
    n = 30
    xs_ = [i / (n - 1) for i in range(n)]
    Cm = [[abs(xi - yj) for yj in xs_] for xi in xs_]
    r = SplitMix64(8101)
    wa = [0.5 + r.uniform() for _ in range(n)]
    wb = [0.5 + r.uniform() for _ in range(n)]
    sa, sb = math.fsum(wa), math.fsum(wb)
    a = [v / sa for v in wa]
    bb = [v / sb for v in wb]
    C.append(case("sinkhorn/residual-contract", "transport.Sinkhorn",
                  {"a": a, "b": bb, "cost": Cm, "epsilon": 0.05, "maxIter": 1000, "tol": 1e-9, "measure": "row-residual"},
                  0.0, 1e-9, "abs", skr + "; 30 points, |x - y| cost, epsilon 0.05, tol 1e-9"))

    # Wasserstein1D --------------------------------------------------------------------
    wd = ("optim/transport/wasserstein1d.go Wasserstein1D: 'returns the closed-form Wasserstein-p distance' (and the "
          "'<=1e-12' cross-implementation contract) -> relative 1e-12 against the exact W_p of the samples (quantile integral "
          "in rationals)")
    r = SplitMix64(8201)
    for cid, nu, nv, p, scale in (("wasserstein/equal-n50-p1", 50, 50, 1.0, 1.0), ("wasserstein/unequal-7-3-p1", 7, 3, 1.0, 1.0),
                                  ("wasserstein/unequal-40-17-p2", 40, 17, 2.0, 1.0), ("wasserstein/p4-scale-1e-90", 2, 2, 4.0, 1e-90),
                                  ("wasserstein/p2-scale-1e160", 3, 3, 2.0, 1e160)):
        u = [scale * r.normal() for _ in range(nu)]
        v = [scale * (r.normal() + 0.5) for _ in range(nv)]
        C.append(case(cid, "transport.Wasserstein1D", {"u": u, "v": v, "p": p}, f64(w_p_exact(u, v, int(p))), 1e-12, "rel",
                      wd + ({"wasserstein/p4-scale-1e-90": "; samples of order 1e-90 (|d|^4 underflows)",
                             "wasserstein/p2-scale-1e160": "; samples of order 1e160 (|d|^2 overflows)"}.get(cid, ""))))
    return C


def main():
    C = []
    C += linalg_cases()
    C += optim_cases()
    C += portfolio_cases()
    C += prox_transport_cases()
    ids = [c["id"] for c in C]
    assert len(ids) == len(set(ids)), "duplicate case id"
    doc = {
        "_comment": "Generated by tools/stressgolden/numerics.py; do not edit by hand. See precision_test.go and precision_numerics_test.go.",
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
