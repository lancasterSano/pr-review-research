"""Tests of revision/r4_calc/r4_negtraces.py (E1 negative traces).

The mutation tests need no pm4py. The alignment tests need the chapter 2 pm4py environment
and are skipped elsewhere; there is no pytest in that environment, so the file also runs as a
script:

    python revision/tests/test_r4_negtraces.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "r4_calc"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import r4_negtraces as N  # noqa: E402

os.environ.setdefault("PM4PY_SHOW_PROGRESS_BAR", "False")
try:
    import pm4py  # noqa: F401
    HAVE_PM4PY = True
except ImportError:
    HAVE_PM4PY = False

TRACES = [
    ["open", "rev_comment", "rev_changes", "revision", "rev_approve", "rev_approve", "merge"],
    ["open", "ai", "revision", "rev_comment", "rev_comment", "rev_approve", "merge", "rev_comment"],
    ["open", "revision", "rev_comment", "merge", "rev_approve"],          # approval after merge
    ["open", "revision", "merge"],                                       # no review
    ["open", "timer", "rev_approve", "ai", "merge"],
]


def _skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        print(f"SKIP: {msg}")
    return True


def _mut(cls, acts, cid="t"):
    return N.mutate(cls, acts, N.rng_for(cls, cid))


def test_m1_mutant_has_no_approvals():
    for k, acts in enumerate(TRACES):
        mut, why = _mut("M1", acts, f"t{k}")
        if "rev_approve" in acts:
            assert mut is not None and "rev_approve" not in mut
            assert [a for a in acts if a != "rev_approve"] == mut
        else:
            assert mut is None and why == "no_approval"


def test_m2_moves_merge_before_first_approval_and_automaton_rejects():
    mut, _ = _mut("M2", TRACES[0])
    assert mut.index("merge") == mut.index("rev_approve") - 1
    assert sorted(mut) == sorted(TRACES[0])
    assert N.automaton_accepts(TRACES[0]) and not N.automaton_accepts(mut)
    assert _mut("M2", TRACES[2])[0] is None      # approval only after merge
    assert _mut("M2", TRACES[3])[0] is None


def test_automaton_one_rule():
    assert [N.automaton_accepts(a) for a in TRACES] == [True, True, False, False, True]
    for k, acts in enumerate(TRACES):            # M3, M4, M6a keep the verdict
        for cls in ("M3", "M4", "M6a"):
            mut, _ = _mut(cls, acts, f"t{k}")
            if mut is not None:
                assert N.automaton_accepts(mut) == N.automaton_accepts(acts)
        assert not N.automaton_accepts(_mut("M6b", acts, f"t{k}")[0])


def test_kplus_positions_and_determinism():
    for k, acts in enumerate(TRACES):
        i_o, i_m, i_a, L = N.limits(acts)
        b, _ = _mut("Kpb", acts, f"t{k}")
        assert len(b) == len(acts) + 1 and b.count("revision") == acts.count("revision") + 1
        assert N.limits(b)[3] == L + 1           # inserted before the first approval / merge
        c, det = _mut("Kpc", acts, f"t{k}")
        assert c.index("merge") == i_m + 1 and det == ("with_ai" if "ai" in acts else "without_ai")
        a, why = _mut("Kpa", acts, f"t{k}")
        if any(x == "rev_comment" for x in acts[i_o + 1:L]):
            assert a.count("rev_comment") == acts.count("rev_comment") + 1
        else:
            assert a is None and why == "no_comment_before_approval"
        for cls in N.CLASSES:
            assert _mut(cls, acts, f"t{k}") == _mut(cls, acts, f"t{k}")


def test_m5_swaps_one_adjacent_pair():
    for k, acts in enumerate(TRACES):
        mut, det = _mut("M5", acts, f"t{k}")
        diff = [i for i in range(len(acts)) if acts[i] != mut[i]]
        assert len(diff) == 2 and diff[1] == diff[0] + 1
        assert det == f"{acts[diff[0]]}|{acts[diff[1]]}"


def test_kplus_on_playout_traces_keep_fitness_one():
    if not HAVE_PM4PY:
        return _skip("pm4py is not installed in this environment")
    import random
    from pm4py.algo.simulation.playout.petri_net import algorithm as playout
    A = N.r2()
    net, im, fm = A.build_net()
    random.seed(7)
    sim = playout.apply(net, im, final_marking=fm,
                        parameters={"noTraces": 150, "maxTraceLength": 40,
                                    "add_only_if_fm_is_reached": True})
    seqs = [[e["concept:name"] for e in tr] for tr in sim]
    req = [("main", s) for s in seqs]
    muts = {}
    for cls in ("Kpa", "Kpb", "Kpc", "M1", "M6a"):
        muts[cls] = [m for m, _ in (N.mutate(cls, s, N.rng_for(cls, f"p{k}"))
                                    for k, s in enumerate(seqs)) if m is not None]
        req += [("main", m) for m in muts[cls]]
    fit = N.align_unique(req, workers=1)
    assert all(fit[("main", tuple(s))][0] == 1.0 for s in seqs)
    for cls in ("Kpa", "Kpb", "Kpc"):
        assert muts[cls], cls
        assert all(fit[("main", tuple(m))][0] == 1.0 for m in muts[cls]), cls
    for cls in ("M1", "M6a"):                    # known-answer negatives
        assert all(fit[("main", tuple(m))][0] < 1.0 for m in muts[cls]), cls


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
