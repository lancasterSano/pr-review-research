"""Tests of revision/r4_calc/r4_mrr_tol.py: at level 0 every
config is admissible and the largest tolerance is chosen when the objective falls with it; with
the Hit@3 condition at 95 % the saved validation table reproduces e4v2_chosen.json."""
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "r4_calc"))
import r4_mrr_tol as T  # noqa: E402

EPS = [0.02, 0.05, 0.1, 0.2, 0.3, 1.0]
W = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]


def _val():
    """Synthetic validation summary: accuracy and both objectives fall as the tolerance grows;
    the latency weight lowers latency more, the exposure weight lowers Gini more."""
    rows = [{"config": "rel_only", "hit3": 0.70, "mrr": 0.60, "gini_top1_full": 0.95, "top1_lat_med": 5.0}]
    for e in EPS:
        for L, Ew in W:
            rows.append({"config": f"constr_e{e}_L{L}_E{Ew}", "hit3": 0.70 - 0.2 * e - 0.01 * (L + Ew),
                         "mrr": 0.60 - 0.3 * e - 0.01 * (L + Ew),
                         "gini_top1_full": 0.95 - e * (0.5 * Ew + 0.1 * L) / (L + Ew),
                         "top1_lat_med": 5.0 - e * (4.0 * L + 0.5 * Ew) / (L + Ew)})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("acc", ["hit3", "mrr"])
def test_level_zero_picks_largest_tolerance(acc):
    ch = T.select(_val(), acc, delta=1.0)                  # level 0: accuracy >= 0 * reference
    assert T.parse_constr(ch["constr_gini"])[0] == 0.3     # eta = 1 (no tolerance) is outside the family
    assert T.parse_constr(ch["constr_lat"])[0] == 0.3
    assert ch["constr_gini"] == "constr_e0.3_L0.0_E1.0"
    assert ch["constr_lat"] == "constr_e0.3_L1.0_E0.0"


def test_full_level_keeps_reference():
    ch = T.select(_val(), "mrr", delta=0.0)
    assert ch == {"constr_gini": "rel_only", "constr_lat": "rel_only"}


def test_family_excludes_no_tolerance():
    fam = T.constr_family(_val())
    assert not fam.config.str.contains("_e1.0_").any()
    assert "rel_only" in set(fam.config)


def test_parse_constr():
    assert T.parse_constr("constr_e0.05_L1.0_E0.0") == (0.05, 1.0, 0.0)
    assert T.parse_constr("rel_only") == (0.0, 0.0, 0.0)


def test_hit3_condition_reproduces_main_choice():
    res = T.N.RANK_RES
    path = os.path.join(res, "e4v2_val_all.csv")
    if not os.path.exists(path):
        pytest.skip("validation table not available")
    val = pd.read_csv(path)
    val = val[val.repo == T.MICRO]
    ref = json.load(open(os.path.join(res, "e4v2_chosen.json"), encoding="utf-8"))
    ch = T.select(val, "hit3", delta=T.DELTA)
    assert ch["constr_gini"] == ref["constr_gini"]
    assert ch["constr_lat"] == ref["constr_lat"]
