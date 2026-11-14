"""Statistics of the chapter 3 reporting step (dissertation, chapter 3). Pure functions, unit-tested in
revision/tests/test_r3_report_stats.py."""
import math

import numpy as np

MIN_FIRES = 20   # threshold 5: at least 20 firings
Z95 = 1.959963984540054


def wilson(k, n, z=Z95):
    """95 % Wilson score interval for k successes out of n (nan, nan if n = 0)."""
    if n <= 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def choose_threshold(p, y, c_r, min_fires=MIN_FIRES):
    """Threshold 5 (chapter 2, eq:threshold): the SMALLEST theta in [0, 1] such
    that the rows with p > theta (strict, as in eq:rule) are at least `min_fires`
    and their precision is >= 1 - c_r. Returns None if no theta qualifies
    ("недосяжно").

    The firing set {p > theta} is always a top run of the scores in descending
    order. For the run that ends at value v the thetas giving exactly that set
    are [v_next, v), where v_next is the next smaller distinct score (0 if v is
    the smallest score), so the smallest qualifying theta is v_next. A run that
    contains scores equal to 0 cannot fire (p > theta >= 0 fails) and is skipped.
    On the history itself this fires exactly the same rows as the former
    "p >= v" rule; on other rows it differs for scores in (v_next, v)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) < min_fires:
        return None
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys)
    # last index of each run of equal values in descending order
    last = np.r_[np.nonzero(np.diff(ps))[0], len(ps) - 1]
    target = 1.0 - c_r - 1e-12
    best = None
    for i in last:
        n = i + 1
        if ps[i] <= 0.0:
            break              # scores equal to 0 never fire under p > theta
        if n >= min_fires and tp[i] / n >= target:
            best = ps[i + 1] if i + 1 < len(ps) else 0.0   # next smaller score; smallest last hit
    return None if best is None else float(best)


def apply_threshold(p, y, theta):
    """Firings, precision, recall and fire share of `p > theta` (strict, eq:rule)
    with Wilson CIs."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    fire = p > theta
    n_f = int(fire.sum())
    tp = int((y[fire] == 1).sum())
    pos = int((y == 1).sum())
    out = {"n": int(len(p)), "n_fire": n_f, "tp": tp, "n_pos": pos,
           "fire_share": n_f / len(p) if len(p) else math.nan,
           "precision": tp / n_f if n_f else math.nan,
           "recall": tp / pos if pos else math.nan}
    out["prec_lo"], out["prec_hi"] = wilson(tp, n_f)
    out["rec_lo"], out["rec_hi"] = wilson(tp, pos)
    out["fire_lo"], out["fire_hi"] = wilson(n_f, len(p))
    return out


def transfer_check(p_other, y_other, p_proj, y_proj, c_r, min_fires=MIN_FIRES):
    """Threshold 6: theta chosen on other projects; on this project the transfer
    is 'evaluable' if theta fires >= min_fires times and 'sufficient' if the
    POINT precision there is >= 1 - c_r. Returns dict (theta None = unreachable
    on the other projects; then neither evaluable nor sufficient)."""
    theta = choose_threshold(p_other, y_other, c_r, min_fires)
    if theta is None:
        return {"theta": None, "evaluable": False, "sufficient": False, "n_fire": 0,
                "precision": math.nan}
    a = apply_threshold(p_proj, y_proj, theta)
    ev = a["n_fire"] >= min_fires
    return {"theta": theta, "evaluable": ev,
            "sufficient": bool(ev and a["precision"] >= 1.0 - c_r - 1e-12),
            "n_fire": a["n_fire"], "precision": a["precision"],
            "prec_lo": a["prec_lo"], "prec_hi": a["prec_hi"]}


def cluster_boot_mean(values, B=10000, seed=42):
    """Percentile 95 % CI of the mean of per-cluster values (one value per
    repository) by resampling clusters with replacement."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    m = v[idx].mean(axis=1)
    return (float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def paired_boot_diff(a, b, B=10000, seed=42):
    """Mean of a - b over the clusters present in both, paired percentile CI."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return cluster_boot_mean(a[ok] - b[ok], B, seed)


def grouped_boot_mean(values, groups, B=10000, seed=42):
    """CI of the mean of per-repository values when the resampling unit is the
    GROUP (language): groups drawn with replacement, all their repositories
    kept. Point estimate = mean over repositories."""
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    ok = ~np.isnan(v)
    v, g = v[ok], g[ok]
    labels = sorted(set(g))
    by = [v[g == l] for l in labels]
    rng = np.random.default_rng(seed)
    ms = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(by), size=len(by))
        x = np.concatenate([by[i] for i in pick])
        ms[b] = x.mean()
    return (float(v.mean()), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5)))


def strat_boot_diff(v1, v2, B=10000, seed=42):
    """Difference of means of two independent sets of repositories (group 1 −
    group 2), repositories resampled within each set."""
    a = np.asarray(v1, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(v2, dtype=float); b = b[~np.isnan(b)]
    rng = np.random.default_rng(seed)
    d = (a[rng.integers(0, len(a), (B, len(a)))].mean(1) -
         b[rng.integers(0, len(b), (B, len(b)))].mean(1))
    return (float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def between_group_sd(values, groups):
    """Method-of-moments SD between groups for one-way random effects with
    unequal group sizes (repository nested in language):
    tau^2 = max(0, (MSB - MSW) / n0), n0 = (N - sum n_i^2 / N) / (k - 1)."""
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    ok = ~np.isnan(v)
    v, g = v[ok], g[ok]
    labels = sorted(set(g))
    k, N = len(labels), len(v)
    if k < 2 or N <= k:
        return math.nan
    grand = v.mean()
    ns = np.array([np.sum(g == l) for l in labels], dtype=float)
    means = np.array([v[g == l].mean() for l in labels])
    ssb = float(np.sum(ns * (means - grand) ** 2))
    ssw = float(sum(np.sum((v[g == l] - v[g == l].mean()) ** 2) for l in labels))
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - np.sum(ns ** 2) / N) / (k - 1)
    return math.sqrt(max(0.0, (msb - msw) / n0))


def between_group_sd_boot(values, groups, B=2000, seed=42):
    """Two-stage bootstrap CI of between_group_sd: groups with replacement,
    then repositories with replacement within each drawn group."""
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    ok = ~np.isnan(v)
    v, g = v[ok], g[ok]
    labels = sorted(set(g))
    by = [v[g == l] for l in labels]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        vals, grp = [], []
        for j, i in enumerate(rng.integers(0, len(by), size=len(by))):
            x = by[i][rng.integers(0, len(by[i]), size=len(by[i]))]
            vals.append(x)
            grp += [j] * len(x)
        s = between_group_sd(np.concatenate(vals), np.array(grp))
        if not math.isnan(s):
            out.append(s)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def mde_language(sd_d, n_lang=8, mult=3.3):
    """Threshold 3: MDE = 3.3 * sigma_d / sqrt(8)."""
    return mult * sd_d / math.sqrt(n_lang)


def ece(y, p, bins=10):
    """Expected calibration error over `bins` equal-count bins."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(p) == 0:
        return math.nan
    order = np.argsort(p, kind="stable")
    parts = np.array_split(order, min(bins, len(p)))
    return float(sum(len(ix) * abs(y[ix].mean() - p[ix].mean()) for ix in parts if len(ix)) / len(p))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt_fit(p, y):
    """Logistic regression of y on logit(p); returns (a, b) of a + b*logit(p).
    Returns None if y has a single class."""
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return None
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(_logit(p).reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def platt_apply(p, ab):
    a, b = ab
    return 1.0 / (1.0 + np.exp(-(a + b * _logit(p))))


TRANSFER_MIN_EVALUABLE = 10


def verdict_rule_transfer(n_sufficient, n_evaluable):
    """Rule transfer at c_r = 0.2 (evaluation threshold 6)."""
    if n_evaluable < TRANSFER_MIN_EVALUABLE:
        return "не оцінюється"
    return "достатнє" if n_sufficient >= n_evaluable / 2 else "недостатнє"


def verdict_transfer(lo):
    """Threshold 1 by the lower bound of the mean LORO AUC."""
    if lo is None or (isinstance(lo, float) and math.isnan(lo)):
        return "немає даних"
    if lo >= 0.70:
        return "переноситься"
    if lo >= 0.60:
        return "переноситься слабко"
    return "НН3 для мітки спростовано"


def diff_exists(est, lo, hi, min_abs=0.02):
    """Threshold 2: the paired CI excludes 0 and |difference| >= 0.02."""
    return bool((lo > 0 or hi < 0) and abs(est) >= min_abs)
