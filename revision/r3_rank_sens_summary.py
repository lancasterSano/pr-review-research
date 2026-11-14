"""Summary of the ranking-constant sensitivity run (dissertation, chapter 3, part
"sensitivity of the ranking constants").

Reads the base (rank_shared/results) and every variant under rank_sens/<v>/results
and writes rank_sens/summary.csv, rank_sens/eta_grid.csv and
rank_sens/rank_sens_numbers.json. Criteria, fixed before the run:
  (a) pooled STRICT beats "most familiar" on test Hit@3, paired 95 % CI above 0;
  (b) validation picks a tolerance re-ranking (constr_*) for both objectives, and
      no weighted penalty (wsum_*) is admissible at the variant's bound;
  (c) relative test Hit@3 loss of the chosen latency re-ranking <= delta.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

MICRO = "ALL (micro)"
# variant -> (constant, value, delta, own model?)  own model = pairs rebuilt and models retrained
VARIANTS = {
    "base":    ("AAIT 2026", "", 0.05, True),
    "pool90":  ("cand_window_days", "90", 0.05, True),
    "pool365": ("cand_window_days", "365", 0.05, True),
    "warm30":  ("warmup_days", "30", 0.05, True),
    "warm180": ("warmup_days", "180", 0.05, True),
    "split70": ("fractions", "0.7/0.15/0.15", 0.05, True),
    "adm90":   ("delta (admissible %)", "90", 0.10, False),
    "adm98":   ("delta (admissible %)", "98", 0.02, False),
}


def eps_of(config):
    return float(config.split("_e")[1].split("_L")[0]) if config.startswith("constr_") else float("nan")


def variant_row(name, res, model_res):
    constant, value, delta, _ = VARIANTS[name]
    micro = pd.read_csv(model_res / "r1_micro.csv")
    model = micro[micro.config == "ranker_pooled_STRICT"].iloc[0]
    paired = pd.read_csv(model_res / "r1_familiar_paired.csv")
    g = paired[(paired.repo == "ALL") & (paired.model == "pooled") & (paired.baseline == "most_familiar")
               & (paired.metric == "hit3")].iloc[0]
    chosen = json.loads((res / "e4v2_chosen.json").read_text())
    val = pd.read_csv(res / "e4v2_val_all.csv"); val = val[val.repo == MICRO].set_index("config")
    test = pd.read_csv(res / "e4v2_test_all.csv"); test = test[test.repo == MICRO].set_index("config")

    ref_val = val.loc["rel_only", "hit3"]
    wsum = val[val.index.str.startswith("wsum_")]
    n_wsum_adm = int((wsum.hit3 >= (1 - delta) * ref_val).sum())
    crit_a = bool(g.delta > 0 and g.lo > 0)
    crit_b = bool(chosen["constr_gini"].startswith("constr_") and chosen["constr_lat"].startswith("constr_")
                  and n_wsum_adm == 0)
    rel_t = test.loc["rel_only", "hit3"]
    lat_t = test.loc[chosen["constr_lat"], "hit3"]
    rel_loss = 1 - lat_t / rel_t
    crit_c = bool(rel_loss <= delta + 1e-12)
    return {
        "variant": name, "constant": constant, "value": value, "admissible_pct": round(100 * (1 - delta)),
        "n_test_prs": int(model.n), "hit3": float(model.hit3), "mrr": float(model.mrr),
        "gain_vs_familiar_hit3": float(g.delta), "gain_lo": float(g.lo), "gain_hi": float(g.hi), "crit_a": int(crit_a),
        "chosen_constr_gini": chosen["constr_gini"], "chosen_constr_lat": chosen["constr_lat"],
        "chosen_wsum_gini": chosen["wsum_gini"], "chosen_wsum_lat": chosen["wsum_lat"],
        "val_hit3_rel_only": float(ref_val), "val_hit3_best_wsum": float(wsum.hit3.max()),
        "val_best_wsum": wsum.hit3.idxmax(), "n_wsum_admissible": n_wsum_adm, "crit_b": int(crit_b),
        "tol_value": eps_of(chosen["constr_lat"]),
        "test_hit3_rel_only": float(rel_t), "test_hit3_constr_lat": float(lat_t),
        "tol_latency_hit3_cost": float(rel_t - lat_t), "tol_latency_rel_loss_pct": float(100 * rel_loss),
        "crit_c": int(crit_c), "holds": int(crit_a and crit_b and crit_c),
    }


def eta_grid(base_res):
    """The tolerance grid of the base, reported as is (no selection)."""
    val = pd.read_csv(base_res / "e4v2_val_all.csv"); test = pd.read_csv(base_res / "e4v2_test_all.csv")
    rows = []
    for split, d in (("val", val), ("test", test)):
        d = d[d.repo == MICRO].set_index("config")
        ref = d.loc["rel_only", "hit3"]
        for cfg, r in d[d.index.str.startswith("constr_")].iterrows():
            rows.append({"split": split, "config": cfg, "eps": eps_of(cfg),
                         "w_lat": float(cfg.split("_L")[1].split("_E")[0]), "w_exp": float(cfg.split("_E")[1]),
                         "hit3": r.hit3, "hit3_share_of_rel_only_pct": 100 * r.hit3 / ref,
                         "gini_top1_full": r.gini_top1_full, "top1_lat_med": r.top1_lat_med, "top1_unique": r.top1_unique})
    return pd.DataFrame(rows).sort_values(["split", "w_lat", "w_exp", "eps"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    base_res = a.base / "results"
    rows = []
    for name, (_, _, _, own) in VARIANTS.items():
        res = base_res if name == "base" else a.out / name / "results"
        rows.append(variant_row(name, res, res if own else base_res))
    s = pd.DataFrame(rows)
    b = s[s.variant == "base"].iloc[0]
    s["dhit3"] = s.hit3 - b.hit3
    s["dmrr"] = s.mrr - b.mrr
    s.to_csv(a.out / "summary.csv", index=False)
    eta_grid(base_res).to_csv(a.out / "eta_grid.csv", index=False)

    v = s[s.variant != "base"]
    nums = {"rank.sens.n_variants": int(len(v)), "rank.sens.n_hold": int(v.holds.sum()),
            "rank.sens.max_abs_dhit3": float(v.dhit3.abs().max()), "rank.sens.max_abs_dmrr": float(v.dmrr.abs().max())}
    for _, r in v.iterrows():
        k = f"rank.sens.{r.variant}"
        nums.update({f"{k}.hit3": r.hit3, f"{k}.mrr": r.mrr, f"{k}.gain_vs_familiar_hit3": r.gain_vs_familiar_hit3,
                     f"{k}.gain_lo": r.gain_lo, f"{k}.gain_hi": r.gain_hi, f"{k}.tol_value": r.tol_value,
                     f"{k}.tol_latency_hit3_cost": r.tol_latency_hit3_cost, f"{k}.holds": int(r.holds)})
    ints = (".holds", ".n_variants", ".n_hold")
    nums = {k: int(x) if k.endswith(ints) else float(x) for k, x in nums.items()}
    (a.out / "rank_sens_numbers.json").write_text(json.dumps(nums, indent=2, ensure_ascii=False) + "\n")
    print(s[["variant", "hit3", "mrr", "dhit3", "gain_vs_familiar_hit3", "gain_lo", "crit_a", "n_wsum_admissible",
             "chosen_constr_lat", "crit_b", "tol_latency_rel_loss_pct", "crit_c", "holds"]].to_string(index=False))


if __name__ == "__main__":
    main()
