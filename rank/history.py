"""Point-in-time queries over reviewer and author history.

Everything is answered "as of t0": only events strictly before t0 are
visible. Times are stored as float seconds since the epoch for bisecting.
"""
import math
from collections import defaultdict

import numpy as np

DAY = 86400.0


def _sec(dt):
    return dt.timestamp()


class ReviewerHistory:
    def __init__(self, events):
        per = defaultdict(list)
        for e in events:
            per[e["reviewer"]].append(e)
        self._r = {}
        for login, rows in per.items():
            rows.sort(key=lambda e: e["first_at"])
            self._r[login] = {
                "first": np.array([_sec(e["first_at"]) for e in rows]),
                "created": np.array([_sec(e["pr_created"]) for e in rows]),
                "merged": np.array([_sec(e["pr_merged"]) if e["pr_merged"] else np.inf for e in rows]),
            }
            fam = defaultdict(list)
            for e in rows:
                fam[e["author"]].append(_sec(e["first_at"]))
            self._r[login]["fam"] = {a: np.array(v) for a, v in fam.items()}
        # global event stream for candidate generation
        allev = sorted(((_sec(e["first_at"]), e["reviewer"]) for e in events), key=lambda x: x[0])
        self._all_t = np.array([t for t, _ in allev])
        self._all_r = np.array([r for _, r in allev], dtype=object)

    # ---- candidate generation -------------------------------------------
    def candidates(self, t0, window_days=180):
        t = _sec(t0)
        lo = np.searchsorted(self._all_t, t - window_days * DAY, side="left")
        hi = np.searchsorted(self._all_t, t, side="left")  # strictly before t0
        return sorted(set(self._all_r[lo:hi].tolist()))

    # ---- helpers ---------------------------------------------------------
    def _before(self, login, t0):
        d = self._r.get(login)
        if d is None:
            return None, 0
        return d, int(np.searchsorted(d["first"], _sec(t0), side="left"))

    # ---- activity --------------------------------------------------------
    def activity(self, login, t0, days):
        d, hi = self._before(login, t0)
        if d is None:
            return 0
        lo = int(np.searchsorted(d["first"], _sec(t0) - days * DAY, side="left"))
        return hi - lo

    def total_activity(self, login, t0):
        d, hi = self._before(login, t0)
        return hi

    def days_since_last(self, login, t0):
        d, hi = self._before(login, t0)
        if d is None or hi == 0:
            return math.nan
        return (_sec(t0) - d["first"][hi - 1]) / DAY

    # ---- workload --------------------------------------------------------
    def workload(self, login, t0, days=180):
        """PRs the reviewer engaged with in the last `days` before t0 that are still open at t0."""
        d, hi = self._before(login, t0)
        if d is None or hi == 0:
            return 0
        lo = int(np.searchsorted(d["first"], _sec(t0) - days * DAY, side="left"))
        return int((d["merged"][lo:hi] > _sec(t0)).sum())

    # ---- responsiveness --------------------------------------------------
    def _latencies_hours(self, login, t0, days):
        """Latencies of reviews the candidate submitted in [t0 - days, t0)."""
        d, hi = self._before(login, t0)
        if d is None or hi == 0:
            return np.array([])
        lo = int(np.searchsorted(d["first"], _sec(t0) - days * DAY, side="left"))
        return (d["first"][lo:hi] - d["created"][lo:hi]) / 3600.0

    def latency_median_hours(self, login, t0, days=90):
        lat = self._latencies_hours(login, t0, days)
        return float(np.median(lat)) if lat.size else math.nan

    def fast_share(self, login, t0, days=90, hours=24):
        lat = self._latencies_hours(login, t0, days)
        return float((lat <= hours).mean()) if lat.size else math.nan

    # ---- familiarity -----------------------------------------------------
    def familiarity(self, login, author, t0):
        d = self._r.get(login)
        if d is None:
            return 0
        arr = d["fam"].get(author)
        if arr is None:
            return 0
        return int(np.searchsorted(arr, _sec(t0), side="left"))


class AuthorHistory:
    """PRs an author opened before t0 and how fast they were first reviewed."""

    def __init__(self, prs):
        per = defaultdict(list)
        for p in prs:
            login = (p.get("author") or {}).get("login") or ""
            created = p["createdAt"]
            if hasattr(created, "timestamp"):
                c = _sec(created)
            else:
                from rank.events import parse_ts
                c = _sec(parse_ts(created))
            fr = p.get("first_review_at")
            per[login].append((c, _sec(fr) if fr else np.nan))
        self._a = {}
        for login, rows in per.items():
            rows.sort()
            self._a[login] = {
                "created": np.array([c for c, _ in rows]),
                "first_review": np.array([f for _, f in rows]),
            }

    def prior_prs(self, author, t0):
        d = self._a.get(author)
        if d is None:
            return 0
        return int(np.searchsorted(d["created"], _sec(t0), side="left"))

    def median_hours_to_first_review(self, author, t0):
        d = self._a.get(author)
        if d is None:
            return math.nan
        hi = int(np.searchsorted(d["created"], _sec(t0), side="left"))
        fr = d["first_review"][:hi]
        cr = d["created"][:hi]
        mask = ~np.isnan(fr) & (fr < _sec(t0))
        if not mask.any():
            return math.nan
        return float(np.median((fr[mask] - cr[mask]) / 3600.0))
