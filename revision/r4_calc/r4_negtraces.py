#!/usr/bin/env python3
"""Chapter 4, E1: negative traces (paired design; mutation classes, seed, measures and criteria
were fixed in the author's notes before the run).

Every one of the 8 000 chapter 2 alignment traces is mutated separately in each class
(M1-M6 must be detected, K+ must not be punished), and the fitness of the mutant on the
chapter 2 workflow net is compared with the fitness of the same original trace (a pair).
The state automaton (one rejection rule: a human approval before the merge) is applied to
the same sequences only as an interpretation.

The chapter 2 code is imported unchanged and only read:
`revision/r2/r2_alignment_v2.py`
(`load_main`, `sample_repo`, `iter_selected`, `build_trace`, `build_net`, `to_event_log`,
`align`, `load_replay`). The mutation functions below do not need pm4py.

Run from the repository root with an interpreter that has requirements-alignment.txt:

    PY=python
    PYTHONDONTWRITEBYTECODE=1 nice -n 10 $PY revision/r4_calc/r4_negtraces.py --dev   # 2 repos x 20
    PYTHONDONTWRITEBYTECODE=1 nice -n 10 $PY revision/r4_calc/r4_negtraces.py         # main run

Output: revision/output/r4/negtraces/{numbers.json, checks.json, pairs.csv, runs.jsonl}
(--dev: revision/output/r4/negtraces/dev/).
"""
import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
CLONE = os.path.dirname(REV)
R2_TABLES = os.path.join(REV, "r2")   # chapter 2 code (r2_alignment_v2.py)
OUT = os.path.join(REV, "output", "r4", "negtraces")

SEED = 20261116
EPS = 1e-9
POST_MERGE = ("rev_approve", "rev_changes", "rev_comment", "revision")
# class -> (criterion fixed before the run on the share of pairs with a fitness drop, net)
CLASSES = {
    "M1": ("ge95", "main"), "M2": ("ge95", "main"), "M3": ("ge95", "main"),
    "M4": ("ge95", "main"), "M4_relaxed": (None, "relaxed"), "M5": (None, "main"),
    "M6a": ("ge95", "main"), "M6b": ("ge95", "main"),
    "Kpa": ("le5", "main"), "Kpb": ("le5", "main"), "Kpc": ("le5", "main"),
}
DEV_REPOS, DEV_N = 2, 20


# ── mutations (pure; no pm4py) ──────────────────────────────────────────────

def _first(acts, a):
    return acts.index(a) if a in acts else None


def limits(acts):
    """(i_o, i_m, i_a, L): first open, first merge, first rev_approve, min(i_a, i_m)."""
    i_o, i_m, i_a = _first(acts, "open"), _first(acts, "merge"), _first(acts, "rev_approve")
    L = i_m if i_a is None else min(i_a, i_m)
    return i_o, i_m, i_a, L


def automaton_accepts(acts):
    """The one rejection rule of replay() on a sequence: accepted iff a human rev_approve
    precedes the first merge."""
    i_m = _first(acts, "merge")
    head = acts if i_m is None else acts[:i_m]
    return "rev_approve" in head


def rng_for(cls, cid, seed=SEED):
    return random.Random(f"{seed}:{cls}:{cid}")


def mutate(cls, acts, rng):
    """(mutant, detail) or (None, reason) when the mutation is impossible for this trace."""
    acts = list(acts)
    i_o, i_m, i_a, L = limits(acts)
    base = cls.replace("_relaxed", "")
    if base == "M1":
        if i_a is None:
            return None, "no_approval"
        n = acts.count("rev_approve")
        return [a for a in acts if a != "rev_approve"], "appr1" if n == 1 else "appr2plus"
    if base == "M2":
        if i_a is None or i_a > i_m:
            return None, "no_approval_before_merge"
        out = acts[:i_m] + acts[i_m + 1:]
        out.insert(i_a, "merge")
        return out, ""
    if base == "M3":
        ev = rng.choice(POST_MERGE)
        out = acts[:]
        out.insert(i_m + 1, ev)
        return out, ev
    if base == "M4":
        if i_a is None or i_a > i_m:
            return None, "no_approval_before_merge"
        out = acts[:]
        out.insert(i_a + 1, "rev_changes")
        return out, ""
    if base == "M5":
        pairs = [i for i in range(len(acts) - 1) if acts[i] != acts[i + 1]]
        if not pairs:
            return None, "no_distinct_pair"
        i = rng.choice(pairs)
        out = acts[:]
        out[i], out[i + 1] = out[i + 1], out[i]
        return out, f"{acts[i]}|{acts[i + 1]}"
    if base == "M6a":
        if i_o is None:
            return None, "no_open"
        return acts[:i_o] + acts[i_o + 1:], ""
    if base == "M6b":
        return ["merge"] + acts[:i_m] + acts[i_m + 1:], ""
    lo = 0 if i_o is None else i_o + 1
    if base == "Kpa":
        cand = [i for i in range(lo, L) if acts[i] == "rev_comment"]
        if not cand:
            return None, "no_comment_before_approval"
        i = rng.choice(cand)
        out = acts[:]
        out.insert(i + 1, "rev_comment")
        return out, ""
    if base == "Kpb":
        out = acts[:]
        out.insert(rng.randint(lo, L), "revision")
        return out, ""
    if base == "Kpc":
        out = acts[:]
        out.insert(rng.randint(lo, i_m), "ai")
        return out, "with_ai" if "ai" in acts else "without_ai"
    raise KeyError(cls)


# ── alignment (pm4py, chapter 2 net) ────────────────────────────────────────

def r2():
    if R2_TABLES not in sys.path:
        sys.path.insert(0, R2_TABLES)
    import r2_alignment_v2 as A
    return A


def _align_chunk(job):
    """[(fitness, cost units)] for sequences on one net (worker process)."""
    net_name, seqs = job
    A = r2()
    net, im, fm = A.build_net(relaxed=(net_name == "relaxed"))
    t0 = datetime(2000, 1, 1, tzinfo=timezone.utc)
    cases = [(f"s{k}", [(t0 + timedelta(seconds=i), a) for i, a in enumerate(s)])
             for k, s in enumerate(seqs)]
    res = A.align(A.to_event_log(cases), net, im, fm)
    return [(float(r["fitness"]), int(r["cost"] // 10000)) for r in res]


def align_unique(requests, workers, chunk=150):
    """{(net, seq tuple): (fitness, cost)}; every distinct sequence is aligned once."""
    todo = defaultdict(set)
    for net, seq in requests:
        todo[net].add(tuple(seq))
    jobs, keys = [], []
    for net in sorted(todo):
        seqs = sorted(todo[net], key=lambda s: (len(s), s))
        for i in range(0, len(seqs), chunk):
            jobs.append((net, seqs[i:i + chunk]))
            keys.append([(net, s) for s in seqs[i:i + chunk]])
    out = {}
    t0 = time.time()
    if workers <= 1:
        results = map(_align_chunk, jobs)
    else:
        from concurrent.futures import ProcessPoolExecutor
        ex = ProcessPoolExecutor(max_workers=workers)
        results = ex.map(_align_chunk, jobs)
    for n, (ks, rs) in enumerate(zip(keys, results), 1):
        out.update(zip(ks, rs))
        if n % 20 == 0 or n == len(jobs):
            print(f"  aligned chunks {n}/{len(jobs)} ({len(out)} sequences, {time.time() - t0:.0f} s)",
                  flush=True)
    if workers > 1:
        ex.shutdown()
    return out


# ── traces ──────────────────────────────────────────────────────────────────

def load_traces(dev=False):
    """[(case id, repo, acts)] exactly as r2_alignment_v2.main builds them."""
    A = r2()
    main_rows = A.load_main()
    repos = sorted(main_rows)[:DEV_REPOS] if dev else sorted(main_rows)
    counts = Counter()
    out = []
    for repo in repos:
        nums = A.sample_repo(repo, main_rows[repo].keys(), A.N_PER_REPO)
        got = []
        for pr in A.iter_selected(repo, nums):
            num = int(pr["number"])
            evs = A.build_trace(pr, main_rows[repo][num], counts)
            got.append((f"{repo}#{num}", repo, [a for _, a in evs]))
        if len(got) != len(nums):
            raise SystemExit(f"{repo}: {len(got)} of {len(nums)} PRs found")
        out += got[:DEV_N] if dev else got
        print(f"traces {repo}: {len(got)}", flush=True)
    return out


def playout_traces():
    """The chapter 2 positive control: 1 000 traces generated by the main net.

    The pm4py playout is not reproducible run to run even with random.seed fixed, so the playout
    numbers (neg.playout.*) vary slightly between runs; the 8 000 alignment traces do not."""
    A = r2()
    from pm4py.algo.simulation.playout.petri_net import algorithm as playout
    net, im, fm = A.build_net()
    random.seed(A.SEED)
    sim = playout.apply(net, im, final_marking=fm,
                        parameters={"noTraces": A.N_PLAYOUT, "maxTraceLength": 40,
                                    "add_only_if_fm_is_reached": True})
    return [(f"playout#{k}", "playout", [e["concept:name"] for e in tr]) for k, tr in enumerate(sim)]


# ── summary ─────────────────────────────────────────────────────────────────

def pct(a, b):
    return 100.0 * a / b if b else None


def summarise_class(rows, crit):
    poss = [r for r in rows if r["possible"]]
    n, n_all = len(poss), len(rows)
    drop = [r for r in poss if r["mut_fit"] < r["orig_fit"] - EPS]
    rise = [r for r in poss if r["mut_fit"] > r["orig_fit"] + EPS]
    d = [r["orig_fit"] - r["mut_fit"] for r in poss]
    s = {
        "n_traces": n_all, "n_pairs": n, "n_impossible": n_all - n,
        "impossible_pct": pct(n_all - n, n_all),
        "impossible_reasons": dict(Counter(r["detail"] for r in rows if not r["possible"])),
        "drop_n": len(drop), "drop_pct": pct(len(drop), n),
        "same_pct": pct(n - len(drop) - len(rise), n), "rise_pct": pct(len(rise), n),
        "mean_drop": statistics.fmean(d) if d else None,
        "mean_drop_dropped": statistics.fmean(r["orig_fit"] - r["mut_fit"] for r in drop) if drop else None,
        "mean_dcost": statistics.fmean(r["mut_cost"] - r["orig_cost"] for r in poss) if poss else None,
        "orig_fit_mean": statistics.fmean(r["orig_fit"] for r in poss) if poss else None,
        "mut_fit_mean": statistics.fmean(r["mut_fit"] for r in poss) if poss else None,
        "auto_orig_acc_pct": pct(sum(r["orig_acc"] for r in poss), n),
        "auto_mut_acc_pct": pct(sum(r["mut_acc"] for r in poss), n),
        "auto_mut_rej_pct": pct(sum(1 - r["mut_acc"] for r in poss), n),
        "auto_flip_pct": pct(sum(r["orig_acc"] and not r["mut_acc"] for r in poss), n),
        "criterion": {"ge95": ">= 95", "le5": "<= 5", None: None}[crit],
    }
    if crit == "ge95":
        s["criterion_met"] = int(s["drop_pct"] is not None and s["drop_pct"] >= 95.0)
    elif crit == "le5":
        s["criterion_met"] = int(s["drop_pct"] is not None and s["drop_pct"] <= 5.0)
    by_repo = defaultdict(lambda: [0, 0])
    for r in poss:
        by_repo[r["repo"]][0] += 1
        by_repo[r["repo"]][1] += int(r["mut_fit"] < r["orig_fit"] - EPS)
    rp = [pct(b, a) for a, b in by_repo.values() if a]
    if rp:
        s["drop_pct_repo_min"], s["drop_pct_repo_med"] = min(rp), statistics.median(rp)
    # descriptive: originals that fit the net fully vs the rest
    for name, sub in (("orig_fit1", [r for r in poss if r["orig_fit"] == 1.0]),
                      ("orig_nonfit", [r for r in poss if r["orig_fit"] < 1.0])):
        s[f"{name}_n"] = len(sub)
        s[f"{name}_drop_pct"] = pct(sum(r["mut_fit"] < r["orig_fit"] - EPS for r in sub), len(sub))
    det = defaultdict(lambda: [0, 0])
    for r in poss:
        if r["detail"]:
            det[r["detail"]][0] += 1
            det[r["detail"]][1] += int(r["mut_fit"] < r["orig_fit"] - EPS)
    if det:
        s["by_detail"] = {k: {"n": a, "drop_pct": pct(b, a)} for k, (a, b) in
                          sorted(det.items(), key=lambda kv: -kv[1][0])}
    return s


def git_head():
    try:
        return subprocess.check_output(["git", "-C", REV, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


def make_pairs(traces, classes):
    """Pair rows (original, mutant) for every trace and class, before any alignment."""
    rows = defaultdict(list)
    for cls in classes:
        net = CLASSES[cls][1]
        for cid, repo, acts in traces:
            mut, detail = mutate(cls, acts, rng_for(cls, cid))
            rows[cls].append({"case": cid, "repo": repo, "cls": cls, "net": net,
                              "possible": mut is not None, "detail": detail, "acts": list(acts),
                              "orig_acc": int(automaton_accepts(acts)), "mut": mut})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help=f"{DEV_REPOS} repos x {DEV_N} traces")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    out_dir = os.path.join(OUT, "dev") if args.dev else OUT
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()
    A = r2()

    traces = load_traces(args.dev)
    play = playout_traces()
    print(f"{len(traces)} traces, {len(play)} playout traces", flush=True)
    classes = list(CLASSES)

    # mutate first (pure), then align every distinct sequence once
    req = []
    for tset in (traces, play):
        for _, _, acts in tset:
            req += [("main", acts), ("relaxed", acts)]
    rows = make_pairs(traces, classes)
    prow = make_pairs(play, classes)
    for R in (rows, prow):
        for cls, rs in R.items():
            req += [(r["net"], r["mut"]) for r in rs if r["mut"] is not None]
    fit = align_unique(req, args.workers)
    for R in (rows, prow):
        for cls, rs in R.items():
            for r in rs:
                r["orig_fit"], r["orig_cost"] = fit[(r["net"], tuple(r["acts"]))]
                if r["mut"] is not None:
                    r["mut_fit"], r["mut_cost"] = fit[(r["net"], tuple(r["mut"]))]
                    r["mut_acc"] = int(automaton_accepts(r["mut"]))

    # ── checks with a known answer ──
    checks = {}
    ref = {}
    with open(os.path.join(A.OUT, "r2_alignment_v2_traces.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ref[r["case"]] = r
    seq_ok = all(ref[c]["trace"] == " ".join(a) for c, _, a in traces)
    dm = max(abs(fit[("main", tuple(a))][0] - float(ref[c]["main_fitness"])) for c, _, a in traces)
    dr = max(abs(fit[("relaxed", tuple(a))][0] - float(ref[c]["relaxed_fitness"])) for c, _, a in traces)
    main_mean = statistics.fmean(fit[("main", tuple(a))][0] for _, _, a in traces)
    rep = A.load_replay([c for c, _, _ in traces])
    agree = sum(rep[c][0] == int(automaton_accepts(a)) for c, _, a in traces)
    checks["k1_r2_reproduction"] = {"n_traces": len(traces), "sequences_equal": seq_ok,
                                    "max_abs_diff_main_fitness": dm,
                                    "max_abs_diff_relaxed_fitness": dr,
                                    "main_fitness_mean": main_mean,
                                    "ok": bool(seq_ok and dm <= 1e-6 and dr <= 1e-6)}
    checks["k2_automaton_equals_replay"] = {"agree": agree, "n": len(traces),
                                            "ok": agree == len(traces)}
    m1_no_appr = all("rev_approve" not in r["mut"] for r in rows["M1"] if r["mut"] is not None)
    checks["k3_m1_has_no_approvals"] = m1_no_appr
    play_orig = [fit[("main", tuple(a))][0] for _, _, a in play]
    kp_play = {cls: [r["mut_fit"] for r in prow[cls] if r["mut"] is not None]
               for cls in ("Kpa", "Kpb", "Kpc")}
    checks["k4_playout"] = {
        "n": len(play), "orig_all_fit": all(x == 1.0 for x in play_orig),
        **{f"{cls}_n": len(v) for cls, v in kp_play.items()},
        **{f"{cls}_all_fit": all(x == 1.0 for x in v) for cls, v in kp_play.items()},
        "M_drop_pct": {cls: summarise_class(prow[cls], None)["drop_pct"]
                       for cls in classes if not cls.startswith("Kp")},
    }
    checks["k4_playout"]["ok"] = bool(checks["k4_playout"]["orig_all_fit"]
                                      and all(checks["k4_playout"][f"{c}_all_fit"] for c in kp_play))
    if not args.dev and not (checks["k1_r2_reproduction"]["ok"] and checks["k2_automaton_equals_replay"]["ok"]):
        json.dump(checks, open(os.path.join(out_dir, "checks.json"), "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False)
        raise SystemExit("reproduction of chapter 2 failed: see checks.json")

    # ── numbers ──
    nums = {"neg.seed": SEED, "neg.n_traces": len(traces), "neg.n_repos": len({r for _, r, _ in traces}),
            "neg.orig.main.fitness": main_mean,
            "neg.orig.relaxed.fitness": statistics.fmean(fit[("relaxed", tuple(a))][0] for _, _, a in traces),
            "neg.orig.auto_acc_pct": pct(sum(automaton_accepts(a) for _, _, a in traces), len(traces))}
    summ = {}
    for cls in classes:
        s = summarise_class(rows[cls], CLASSES[cls][0])
        summ[cls] = s
        k = cls.lower()
        for key, v in s.items():
            if isinstance(v, dict):
                if key == "by_detail":
                    for dk, dv in v.items():
                        nums[f"neg.{k}.by_detail.{dk}.n"] = dv["n"]
                        nums[f"neg.{k}.by_detail.{dk}.drop_pct"] = dv["drop_pct"]
                elif key == "impossible_reasons":
                    for dk, dv in v.items():
                        nums[f"neg.{k}.impossible.{dk}"] = dv
                continue
            nums[f"neg.{k}.{key}"] = v
        ps = summarise_class(prow[cls], None)
        nums[f"neg.playout.{k}.n_pairs"] = ps["n_pairs"]
        nums[f"neg.playout.{k}.drop_pct"] = ps["drop_pct"]
    crit = [summ[c]["criterion_met"] for c in classes if CLASSES[c][0]]
    nums["neg.criteria_n"] = len(crit)
    nums["neg.criteria_met_n"] = sum(crit)
    nums["neg.criteria_all_met"] = int(all(crit))
    nums["neg.playout.kp_all_fit"] = int(checks["k4_playout"]["ok"])

    seconds = round(time.time() - t_start, 1)
    run = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "kind": "dev" if args.dev else "main",
           "seed": SEED, "n_traces": len(traces), "git": git_head(), "seconds": seconds,
           "workers": args.workers,
           "checks": {"k1_r2_reproduction_ok": checks["k1_r2_reproduction"]["ok"],
                      "k2_automaton_equals_replay_ok": checks["k2_automaton_equals_replay"]["ok"],
                      "k3_m1_has_no_approvals": m1_no_appr, "k4_playout_ok": checks["k4_playout"]["ok"]},
           "summary": {f"neg.{c.lower()}.drop_pct": summ[c]["drop_pct"] for c in classes}}
    run["summary"]["neg.criteria_all_met"] = nums["neg.criteria_all_met"]
    with open(os.path.join(out_dir, "runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")
    checks["runs"] = [json.loads(x) for x in open(os.path.join(out_dir, "runs.jsonl"), encoding="utf-8")]
    checks["meta"] = {"script": "revision/r4_calc/r4_negtraces.py", "r2_script": "r2_alignment_v2.py",
                      "pm4py": __import__("pm4py").__version__, "python": sys.version.split()[0],
                      "pr_table": A.PR_TABLE.replace(os.path.expanduser("~"), "~")}
    checks["class_summaries"] = summ
    json.dump(checks, open(os.path.join(out_dir, "checks.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    json.dump({k: nums[k] for k in sorted(nums)}, open(os.path.join(out_dir, "numbers.json"), "w",
                                                     encoding="utf-8"), indent=1, ensure_ascii=False)
    fields = ["case", "repo", "cls", "net", "possible", "detail", "orig_fit", "mut_fit", "orig_cost",
              "mut_cost", "orig_acc", "mut_acc", "mutant"]
    with open(os.path.join(out_dir, "pairs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for cls in classes:
            for r in rows[cls]:
                w.writerow({k: r.get(k, "") for k in fields[:-1]} |
                           {"possible": int(r["possible"]),
                            "mutant": " ".join(r["mut"]) if r["mut"] is not None else ""})
    print(json.dumps(run, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
