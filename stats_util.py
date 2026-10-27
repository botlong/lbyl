"""Small statistics helpers: mean +- Student-t 95% half-width over seeds (replaces the earlier 1.96 normal
approximation, which is too narrow at n = 5 or 8), and a feature-level train/test overlap check."""
import math, hashlib

_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
         10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 19: 2.093, 29: 2.045}


def t975(df):
    if df in _T975:
        return _T975[df]
    return min((v for k, v in _T975.items() if k <= df), default=1.96) if df < 30 else 1.96


def mean_ci(xs):
    """(mean, 95% half-width, sd) with a t interval over len(xs) runs; half-width 0 for one run."""
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, t975(n - 1) * sd / math.sqrt(n), sd


def fmt(xs, d=2):
    m, h, _ = mean_ci(xs)
    return f"{m:.{d}f} +- {h:.{d}f}" if len(xs) > 1 else f"{m:.{d}f}"


def input_key(f):
    """Hash of one element's full head input (vis and txt feature bytes)."""
    return hashlib.sha1(f["vis"].numpy().tobytes() + f["txt"].numpy().tobytes()).hexdigest()


def overlap(train_rows, test_rows, feats):
    """Number of test rows whose exact (vis, txt) input also occurs in the training rows."""
    tr = {input_key(feats[r["crop"]]) for r in train_rows}
    return sum(input_key(feats[r["crop"]]) in tr for r in test_rows)
