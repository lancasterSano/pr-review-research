"""The ranking constants are parameters now; their defaults must stay the AAIT 2026 values."""
import inspect
from datetime import datetime, timedelta, timezone

import pytest

from rank import build, run_e4, run_experiments, split
from rank.history import ReviewerHistory

AAIT = {"warmup_days": 90, "cand_window_days": 180, "fractions": (0.6, 0.2, 0.2), "delta": 0.05,
        "eps_grid": [0.02, 0.05, 0.1, 0.2, 0.3]}


def _default(fn, name):
    return inspect.signature(fn).parameters[name].default


class TestDefaultsAreAait:
    def test_build_repo(self):
        assert _default(build.build_repo, "warmup_days") == AAIT["warmup_days"]
        assert _default(build.build_repo, "cand_window_days") == AAIT["cand_window_days"]
        assert tuple(_default(build.build_repo, "fractions")) == AAIT["fractions"]

    def test_build_main_and_cli(self):
        assert _default(build.main, "warmup_days") == AAIT["warmup_days"]
        assert _default(build.main, "cand_window_days") == AAIT["cand_window_days"]
        assert tuple(_default(build.main, "fractions")) == AAIT["fractions"]
        a = build.parse_args([])
        assert (a.warmup_days, a.cand_window_days, tuple(a.fractions)) == (90, 180, (0.6, 0.2, 0.2))
        assert a.repos == []

    def test_split_and_candidate_window(self):
        assert _default(split.assign_splits, "warmup_days") == AAIT["warmup_days"]
        assert tuple(_default(split.assign_splits, "fractions")) == AAIT["fractions"]
        assert _default(ReviewerHistory.candidates, "window_days") == AAIT["cand_window_days"]

    def test_e4_admissibility_and_tolerance_grid(self):
        assert run_e4.DELTA == AAIT["delta"]
        assert run_e4.EPS_GRID == AAIT["eps_grid"]
        assert _default(run_e4.select_and_headline, "delta") == AAIT["delta"]
        assert _default(run_e4.main, "delta") == AAIT["delta"]
        a = run_e4.parse_args([])
        assert a.delta == AAIT["delta"] and a.select_only is False

    def test_loro_runs_by_default(self):
        assert _default(run_experiments.main, "skip_loro") is False


class TestOverridesReachTheCode:
    def test_cli_parses_variant_values(self):
        a = build.parse_args(["--warmup-days", "30", "--cand-window-days", "365", "--fractions", "0.7,0.15,0.15", "django_django"])
        assert (a.warmup_days, a.cand_window_days, a.fractions, a.repos) == (30, 365, (0.7, 0.15, 0.15), ["django_django"])
        assert run_e4.parse_args(["--select-only", "--delta", "0.1"]).delta == 0.1

    def test_cli_rejects_fractions_not_summing_to_one(self):
        with pytest.raises(SystemExit):
            build.parse_args(["--fractions", "0.7,0.2,0.2"])

    def test_fractions_change_the_split(self):
        created = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(200)]
        s = split.assign_splits(created, warmup_days=0, fractions=(0.7, 0.15, 0.15))
        assert (s.count("train"), s.count("val"), s.count("test")) == (140, 30, 30)

    def test_build_repo_passes_window_and_warmup(self):
        from rank.tests.test_build import _prs
        _, base = build.build_repo("toy", _prs(), warmup_days=30)
        _, short = build.build_repo("toy", _prs(), warmup_days=30, cand_window_days=25)
        # PR 51 (2026-05-20): carol reviewed on 05-01 (inside 25 days), bob last on 04-01 (outside)
        assert base.set_index("pr").loc[51].n_candidates == 2
        assert short.set_index("pr").loc[51].n_candidates == 1
        _, late = build.build_repo("toy", _prs(), warmup_days=200)
        assert (late.split == "warmup").all()
