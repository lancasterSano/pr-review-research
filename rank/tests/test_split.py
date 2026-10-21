from datetime import datetime, timedelta, timezone


from rank.split import assign_splits


def d(i):
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)


class TestAssignSplits:
    def test_warmup_prs_are_excluded_and_rest_is_60_20_20_in_time_order(self):
        created = [d(i) for i in range(0, 200)]  # 200 days of PRs, one per day
        s = assign_splits(created, warmup_days=90)
        assert s[:90] == ["warmup"] * 90
        rest = s[90:]
        assert rest.count("train") == 66 and rest.count("val") == 22 and rest.count("test") == 22
        assert rest == ["train"] * 66 + ["val"] * 22 + ["test"] * 22

    def test_unsorted_input_is_split_by_time_not_position(self):
        created = [d(150), d(0), d(199)]
        s = assign_splits(created, warmup_days=90)
        assert s[1] == "warmup" and s[0] == "train" and s[2] == "test"
