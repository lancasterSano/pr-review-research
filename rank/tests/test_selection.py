import pandas as pd

from rank.selection import select_on_validation


class TestSelectOnValidation:
    def test_picks_least_concentrated_config_within_accuracy_tolerance(self):
        val = pd.DataFrame({
            "config": ["rel_only", "c1", "c2", "c3"],
            "hit3": [0.50, 0.49, 0.44, 0.48],
            "gini_top1": [0.60, 0.50, 0.20, 0.30],
            "top1_lat_med": [5.0, 4.0, 1.0, 6.0],
        })
        # tolerance 5% of rel_only hit3 -> threshold 0.475: c1 and c3 admissible; c3 has lower gini
        assert select_on_validation(val, reference="rel_only", delta=0.05, objective="gini_top1") == "c3"

    def test_objective_latency(self):
        val = pd.DataFrame({"config": ["rel_only", "c1", "c3"], "hit3": [0.5, 0.49, 0.48],
                            "gini_top1": [0.6, 0.5, 0.3], "top1_lat_med": [5.0, 4.0, 6.0]})
        assert select_on_validation(val, reference="rel_only", delta=0.05, objective="top1_lat_med") == "c1"

    def test_reference_returned_when_nothing_admissible(self):
        val = pd.DataFrame({"config": ["rel_only", "c1"], "hit3": [0.5, 0.1], "gini_top1": [0.6, 0.0], "top1_lat_med": [5.0, 1.0]})
        assert select_on_validation(val, reference="rel_only", delta=0.05, objective="gini_top1") == "rel_only"

    def test_reference_is_admissible_and_wins_when_it_has_the_best_objective(self):
        val = pd.DataFrame({"config": ["rel_only", "c1"], "hit3": [0.5, 0.49], "gini_top1": [0.2, 0.5], "top1_lat_med": [5.0, 4.0]})
        assert select_on_validation(val, reference="rel_only", delta=0.05, objective="gini_top1") == "rel_only"
