"""Offline tests for the corpus v2 collector (no network).

Run from the project root:  .venv/bin/python -m pytest fetch/tests -q
"""
import json
import os
import sys
from datetime import datetime, timezone

import pytest

_FETCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _FETCH)

import fetch_prs_v2 as fp  # noqa: E402
import run_corpus_v2 as rc  # noqa: E402

START, END, CI = "2024-10-01T00:00:00Z", "2026-08-31T23:59:59Z", "2025-10-01T00:00:00Z"


# ── helpers ─────────────────────────────────────────────────────────────────

def conn(nodes, has_next=False, cursor=None, total=None):
    return {"totalCount": len(nodes) if total is None else total,
            "pageInfo": {"hasNextPage": has_next, "endCursor": cursor}, "nodes": nodes}


def pr(number, created, commits=None, commits_next=False, threads=None):
    return {
        "id": "PR_%d" % number, "number": number, "createdAt": created, "state": "MERGED",
        "labels": conn([]),
        "commits": conn(commits or [{"commit": {"oid": "c%d" % number}}],
                        has_next=commits_next, cursor="cc1" if commits_next else None),
        "reviews": conn([]),
        "reviewThreads": conn(threads or []),
        "reviewRequests": conn([]),
        "timelineItems": conn([]),
    }


RATE = {"cost": 14, "remaining": 4000, "resetAt": "2026-10-14T15:00:00Z", "limit": 5000}


class FakeClient:
    """Scripted GraphQL client. `pages` maps cursor -> list of PR nodes (+ next cursor)."""

    def __init__(self, pages, locate_pages=None, more=None, fail_on=None, ci_fail_over=None,
                 page_fail_over=None, stats_timeout=False, bad_oids=(), review_link_error=False):
        self.ci_fail_over = ci_fail_over   # CI detail chunks larger than this time out
        self.page_fail_over = page_fail_over   # main pages larger than this time out
        self.stats_timeout = stats_timeout     # pages with per-commit stats time out
        self.bad_oids = set(bad_oids)          # commits whose stats query times out
        self.review_link_error = review_link_error  # pages with comment review links fail
        self.ci_chunks = []
        self.pages = pages            # cursor -> (nodes, next_cursor)
        self.locate_pages = locate_pages or {}
        self.more = more or {}
        self.fail_on = fail_on or {}  # cursor -> exception (raised once)
        self.calls = []
        self.last_rate = RATE
        self.log = lambda m: None

    def execute(self, query, variables, max_attempts=6):
        self.calls.append((query, dict(variables)))
        cur = variables.get("cursor")
        if "edges { cursor" in query:
            edges, nxt = self.locate_pages[cur]
            return {"rateLimit": dict(RATE, cost=1), "repository": {"pullRequests": {
                "pageInfo": {"hasNextPage": nxt is not None, "endCursor": nxt},
                "edges": edges}}}
        if "nodes(ids: $ids)" in query:
            ids = variables["ids"]
            if self.ci_fail_over is not None and len(ids) > self.ci_fail_over:
                raise fp.ServerTimeoutError("HTTP 502")
            self.ci_chunks.append(list(ids))
            return {"rateLimit": dict(RATE, cost=2),
                    "nodes": [{"id": i, "ciHead": {"nodes": [{"commit": {"oid": "h" + i}}]}} for i in ids]}
        if "object(oid: $oid)" in query:
            if variables["oid"] in self.bad_oids:
                raise fp.ServerTimeoutError("HTTP 502")
            return {"rateLimit": dict(RATE, cost=1), "repository": {"object": {
                "additions": 3, "deletions": 1, "changedFilesIfAvailable": 2}}}
        if "node(id: $id)" in query:
            return {"rateLimit": dict(RATE, cost=1), "node": self.more[(variables["id"], cur)]}
        if cur in self.fail_on:
            exc = self.fail_on.pop(cur)
            raise exc
        if self.page_fail_over is not None and variables["pageSize"] > self.page_fail_over:
            raise fp.ServerTimeoutError("HTTP 502")
        if self.stats_timeout and "changedFilesIfAvailable" in query:
            raise fp.ServerTimeoutError("HTTP 502")
        if self.review_link_error and fp.REVIEW_LINK in query:
            raise fp.ServerTimeoutError("GraphQL: Something went wrong")
        nodes, nxt = self.pages[cur]
        nodes = json.loads(json.dumps(nodes))  # fresh copy each time
        return {"rateLimit": RATE, "repository": {"pullRequests": {
            "totalCount": 999, "pageInfo": {"hasNextPage": nxt is not None, "endCursor": nxt},
            "nodes": nodes}}}


def make_fetcher(client, tmp_path, **kw):
    return fp.RepoFetcher(client, "o", "r", raw_dir=str(tmp_path), window_start=START,
                          window_end=END, ci_since=CI, echo=False, **kw)


def read_jsonl(tmp_path):
    with open(os.path.join(str(tmp_path), "o_r", "prs.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


def locate_top():
    """Locate scan whose very first PR is already inside the window."""
    return {None: ([{"cursor": "L1", "node": {"number": 9, "createdAt": "2026-08-01T00:00:00Z"}}], None)}


# ── window boundaries ───────────────────────────────────────────────────────

class TestWindowPosition:
    @pytest.mark.parametrize("created,expected", [
        ("2024-10-01T00:00:00Z", "in"),       # start is inclusive
        ("2024-09-30T23:59:59Z", "before"),
        ("2026-08-31T23:59:59Z", "in"),       # end is inclusive to the second
        ("2026-09-01T00:00:00Z", "after"),
        ("2025-06-15T12:00:00Z", "in"),
    ])
    def test_boundaries(self, created, expected):
        assert fp.window_position(created, START, END) == expected


# ── language classification ─────────────────────────────────────────────────

def langs(*pairs):
    total = sum(s for _, s in pairs)
    return {"totalSize": total, "edges": [{"size": s, "node": {"name": n}} for n, s in pairs]}


class TestLanguages:
    def test_primary_is_largest_byte_share_and_shares_sum_to_one(self):
        out = fp.summarize_languages(langs(("Go", 47), ("TypeScript", 48), ("PLpgSQL", 5)))
        assert out["primary"] == "TypeScript"
        assert out["primary_share"] == 0.48
        assert out["mixed"] is True
        assert abs(sum(s for _, _, s in out["shares"]) - 1.0) < 1e-9

    def test_exactly_sixty_percent_is_not_mixed(self):
        assert fp.summarize_languages(langs(("PHP", 60), ("TypeScript", 40)))["mixed"] is False

    def test_just_below_sixty_percent_is_mixed(self):
        assert fp.summarize_languages(langs(("Ruby", 599), ("JavaScript", 401)))["mixed"] is True

    def test_empty_repository_has_no_label(self):
        out = fp.summarize_languages({"totalSize": 0, "edges": []})
        assert out["primary"] is None and out["mixed"] is None

    def test_repos_csv_shares_format(self):
        shares = fp.summarize_languages(langs(("Python", 970), ("Jinja", 20), ("CSS", 10)))["shares"]
        assert rc.format_shares(shares) == "Python:0.9700|Jinja:0.0200|CSS:0.0100"


# ── page parsing and exact date stop ───────────────────────────────────────

class TestPageProcessing:
    def test_skips_after_window_keeps_inside_and_stops_at_first_older(self, tmp_path):
        f = make_fetcher(FakeClient({}), tmp_path)
        cp = f._new_checkpoint()
        page = conn([pr(5, "2026-09-01T00:00:00Z"), pr(4, "2026-08-31T23:59:59Z"),
                     pr(3, "2024-10-01T00:00:00Z"), pr(2, "2024-09-30T23:59:59Z"),
                     pr(1, "2024-09-01T00:00:00Z")])
        records, stop = f.process_page(page, cp, seen={}, with_ci=True)
        assert [r["number"] for r in records] == [4, 3]
        assert stop is True
        assert cp["skipped_after_window"] == 1

    def test_meta_marks_ci_window_and_fetch(self, tmp_path):
        f = make_fetcher(FakeClient({}), tmp_path)
        cp = f._new_checkpoint()
        records, _ = f.process_page(conn([pr(2, "2025-10-01T00:00:00Z"), pr(1, "2025-09-30T23:59:59Z")]),
                                    cp, seen={}, with_ci=False)
        assert [r["_meta"]["ci_window"] for r in records] == [True, False]
        assert all(r["_meta"]["ci_fetched"] is False for r in records)

    def test_already_written_numbers_are_not_duplicated(self, tmp_path):
        f = make_fetcher(FakeClient({}), tmp_path)
        records, _ = f.process_page(conn([pr(2, "2025-01-02T00:00:00Z"), pr(1, "2025-01-01T00:00:00Z")]),
                                    f._new_checkpoint(), seen={2: "x"}, with_ci=False)
        assert [r["number"] for r in records] == [1]


# ── nested pagination ───────────────────────────────────────────────────────

class TestNestedPagination:
    def test_commits_beyond_first_page_are_appended(self, tmp_path):
        first = [{"commit": {"oid": "a"}}]
        more = {("PR_1", "cc1"): {"commits": conn([{"commit": {"oid": "b"}}, {"commit": {"oid": "c"}}])}}
        client = FakeClient({}, more=more)
        f = make_fetcher(client, tmp_path)
        cp = f._new_checkpoint()
        node = f.complete_nested(pr(1, "2025-01-01T00:00:00Z", commits=first, commits_next=True), cp)
        assert [c["commit"]["oid"] for c in node["commits"]["nodes"]] == ["a", "b", "c"]
        assert node["commits"]["pageInfo"]["hasNextPage"] is False
        assert cp["followup_queries"] == 1
        q, v = client.calls[0]
        assert "commits(first: 100, after: $cursor)" in q and v == {"id": "PR_1", "cursor": "cc1"}

    def test_timeline_follow_up_keeps_item_type_filter(self):
        q = fp.build_more_query("timelineItems")
        assert "timelineItems(first: 100, after: $cursor, itemTypes: [READY_FOR_REVIEW_EVENT" in q

    def test_thread_comments_beyond_20_are_fetched(self, tmp_path):
        thread = {"id": "T1", "comments": conn([{"body": "1"}], has_next=True, cursor="k")}
        more = {("T1", "k"): {"comments": conn([{"body": "2"}])}}
        f = make_fetcher(FakeClient({}, more=more), tmp_path)
        node = f.complete_nested(pr(1, "2025-01-01T00:00:00Z", threads=[thread]), f._new_checkpoint())
        assert [c["body"] for c in node["reviewThreads"]["nodes"][0]["comments"]["nodes"]] == ["1", "2"]


# ── full run, checkpoint and resume ─────────────────────────────────────────

def three_pages():
    return {
        "L1": ([pr(9, "2026-08-01T00:00:00Z"), pr(8, "2026-07-01T00:00:00Z")], "P1"),
        "P1": ([pr(7, "2025-06-01T00:00:00Z"), pr(6, "2025-05-01T00:00:00Z")], "P2"),
        "P2": ([pr(5, "2024-10-02T00:00:00Z"), pr(4, "2024-09-01T00:00:00Z")], "P3"),
        "P3": ([pr(3, "2024-08-01T00:00:00Z")], None),
    }


class TestRunAndResume:
    def test_locate_skips_newer_prs_and_starts_at_first_in_window(self, tmp_path):
        locate = {None: ([{"cursor": "L0", "node": {"number": 11, "createdAt": "2026-10-01T00:00:00Z"}},
                          {"cursor": "L1", "node": {"number": 10, "createdAt": "2026-09-05T00:00:00Z"}}], "LP"),
                  "LP": ([{"cursor": "L2", "node": {"number": 9, "createdAt": "2026-08-01T00:00:00Z"}}], None)}
        pages = {"L1": ([pr(9, "2026-08-01T00:00:00Z")], "P1"),
                 "P1": ([pr(8, "2024-01-01T00:00:00Z")], None)}
        client = FakeClient(pages, locate_pages=locate)
        cp = make_fetcher(client, tmp_path).run()
        assert cp["status"] == "done" and cp["n_written"] == 1
        assert cp["skipped_after_window"] == 2
        first_page_call = [v for q, v in client.calls if "pageSize" in v][0]
        assert first_page_call["cursor"] == "L1"

    def test_full_run_stops_exactly_at_window_start(self, tmp_path):
        pages = {None: ([pr(9, "2026-08-01T00:00:00Z"), pr(8, "2026-07-01T00:00:00Z")], "P1")}
        pages.update({k: v for k, v in three_pages().items() if k != "L1"})
        client = FakeClient(pages, locate_pages=locate_top())
        cp = make_fetcher(client, tmp_path).run()
        assert [r["number"] for r in read_jsonl(tmp_path)] == [9, 8, 7, 6, 5]
        assert cp["phase"] == "done" and cp["pages"] == 3
        # P3 (older than the window) is never requested
        assert all(v.get("cursor") != "P3" for _, v in client.calls)

    def test_error_keeps_checkpoint_and_resume_continues_without_duplicates(self, tmp_path):
        pages = {None: ([pr(9, "2026-08-01T00:00:00Z"), pr(8, "2026-07-01T00:00:00Z")], "P1")}
        pages.update({k: v for k, v in three_pages().items() if k != "L1"})
        client = FakeClient(pages, locate_pages=locate_top(),
                            fail_on={"P1": fp.TransientError("HTTP 500")})
        with pytest.raises(fp.TransientError):   # already at the smallest page size -> error
            make_fetcher(client, tmp_path, page_size=1).run()
        cp = json.load(open(os.path.join(str(tmp_path), "o_r", "checkpoint.json")))
        assert cp["status"] == "error" and cp["cursor"] == "P1" and cp["n_written"] == 2
        progress = open(os.path.join(str(tmp_path), "progress.csv")).read()
        assert "o/r,error,2," in progress

        client2 = FakeClient(pages, locate_pages=locate_top())
        cp2 = make_fetcher(client2, tmp_path).run()
        assert cp2["status"] == "done"
        assert [r["number"] for r in read_jsonl(tmp_path)] == [9, 8, 7, 6, 5]
        assert [v["cursor"] for _, v in client2.calls if "pageSize" in v] == ["P1", "P2"]  # no locate, no page 1
        assert not any("edges { cursor" in q for q, _ in client2.calls)

    def test_partial_last_line_is_repaired_and_page_refetched(self, tmp_path):
        pages = {None: ([pr(9, "2026-08-01T00:00:00Z")], "P1"),
                 "P1": ([pr(8, "2026-07-01T00:00:00Z")], "P2"),
                 "P2": ([pr(1, "2023-01-01T00:00:00Z")], None)}
        client = FakeClient(pages, locate_pages=locate_top(), fail_on={"P1": fp.TransientError("x")})
        with pytest.raises(fp.TransientError):
            make_fetcher(client, tmp_path, page_size=1).run()
        path = os.path.join(str(tmp_path), "o_r", "prs.jsonl")
        with open(path, "a") as f:
            f.write('{"number": 8, "createdAt": "2026-07')   # crash mid-write
        cp = make_fetcher(FakeClient(pages, locate_pages=locate_top()), tmp_path).run()
        assert [r["number"] for r in read_jsonl(tmp_path)] == [9, 8]
        assert cp["n_written"] == 2

    def test_done_repository_is_not_refetched(self, tmp_path):
        pages = {None: ([pr(1, "2023-01-01T00:00:00Z")], None)}
        make_fetcher(FakeClient(pages, locate_pages=locate_top()), tmp_path).run()
        client = FakeClient(pages, locate_pages=locate_top())
        make_fetcher(client, tmp_path).run()
        assert client.calls == []

    def test_checkpoint_with_another_window_is_refused(self, tmp_path):
        f = make_fetcher(FakeClient({}), tmp_path)
        cp = f._new_checkpoint()
        cp["window"] = {"start": "2023-01-01T00:00:00Z", "end": END}
        fp.atomic_write_json(f.cp_path, cp)
        with pytest.raises(fp.GraphQLError):
            f.run()

    @pytest.mark.parametrize("key, value", [
        ("ci_since", "2025-01-01T00:00:00Z"),
        ("states", ["MERGED"]),
        ("collector", "v1.0"),
    ])
    def test_checkpoint_with_other_corpus_parameters_is_refused(self, tmp_path, key, value):
        f = make_fetcher(FakeClient({}), tmp_path)
        cp = f._new_checkpoint()
        cp[key] = value
        fp.atomic_write_json(f.cp_path, cp)
        client = FakeClient({})
        f = make_fetcher(client, tmp_path)
        with pytest.raises(fp.GraphQLError, match="refusing to mix") as exc:
            f.run()
        assert key in str(exc.value)
        assert client.calls == []
        with open(f.cp_path) as fh:
            assert json.load(fh)[key] == value   # the checkpoint is left as it was

    def test_checkpoint_without_newer_fields_still_resumes(self, tmp_path):
        pages = {None: ([pr(9, "2026-08-01T00:00:00Z")], "P1"),
                 "P1": ([pr(8, "2026-07-01T00:00:00Z")], None)}
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return calls["n"] > 1   # one page, then stop

        f = make_fetcher(FakeClient(pages, locate_pages=locate_top()), tmp_path,
                         stop_requested=stop)
        f.run()
        with open(f.cp_path) as fh:
            cp = json.load(fh)
        for key in ("ci_since", "states", "collector"):
            del cp[key]
        fp.atomic_write_json(f.cp_path, cp)
        cp = make_fetcher(FakeClient(pages, locate_pages=locate_top()), tmp_path).run()
        assert cp["status"] == "done" and cp["n_written"] == 2

    def test_stop_request_saves_and_exits_after_page(self, tmp_path):
        pages = {None: ([pr(9, "2026-08-01T00:00:00Z")], "P1"),
                 "P1": ([pr(8, "2026-07-01T00:00:00Z")], None)}
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return calls["n"] > 1   # allow exactly one page

        cp = make_fetcher(FakeClient(pages, locate_pages=locate_top()), tmp_path,
                          stop_requested=stop).run()
        assert cp["status"] == "stopped" and cp["cursor"] == "P1" and cp["n_written"] == 1


# ── page size fallback and CI switch ────────────────────────────────────────

class TestPageSizeAndCi:
    def test_repeated_502_drops_page_size_to_10_for_the_same_cursor(self, tmp_path):
        pages = {None: ([pr(9, "2026-08-01T00:00:00Z")], None)}
        client = FakeClient(pages, locate_pages=locate_top(),
                            fail_on={None: fp.ServerTimeoutError("HTTP 502")})
        cp = make_fetcher(client, tmp_path).run()
        sizes = [v["pageSize"] for _, v in client.calls if "pageSize" in v]
        assert sizes == [25, 10]
        assert cp["page_size"] == 10 and cp["n_written"] == 1

    def test_page_size_steps_down_to_one_when_smaller_pages_still_time_out(self, tmp_path):
        pages = {None: ([pr(9, "2026-08-01T00:00:00Z")], None)}
        client = FakeClient(pages, locate_pages=locate_top(), page_fail_over=1)
        cp = make_fetcher(client, tmp_path).run()
        sizes = [v["pageSize"] for _, v in client.calls if "pageSize" in v]
        assert sizes == [25, 10, 5, 2, 1]
        assert cp["status"] == "done" and cp["page_size"] == 1 and cp["n_written"] == 1

    def test_timeout_at_smallest_page_size_is_an_error(self, tmp_path):
        pages = {None: ([pr(9, "2026-08-01T00:00:00Z")], None)}
        client = FakeClient(pages, locate_pages=locate_top(), page_fail_over=0)
        with pytest.raises(fp.ServerTimeoutError):
            make_fetcher(client, tmp_path).run()
        cp = json.load(open(os.path.join(str(tmp_path), "o_r", "checkpoint.json")))
        assert cp["status"] == "error" and cp["page_size"] == 1 and cp["n_written"] == 0

    def test_one_pr_timeout_falls_back_to_per_commit_stats(self, tmp_path):
        commits = [{"commit": {"oid": "good"}}, {"commit": {"oid": "huge"}}]
        pages = {None: ([pr(9, "2026-08-01T00:00:00Z", commits=commits)], "P1"),
                 "P1": ([pr(8, "2026-07-01T00:00:00Z")], None)}
        client = FakeClient(pages, locate_pages=locate_top(), stats_timeout=True,
                            bad_oids={"huge"})
        cp = make_fetcher(client, tmp_path, page_size=1).run()
        assert cp["status"] == "done" and cp["n_written"] == 2
        recs = {r["number"]: r for r in read_jsonl(tmp_path)}
        got = {n["commit"]["oid"]: n["commit"] for n in recs[9]["commits"]["nodes"]}
        assert got["good"]["additions"] == 3 and got["good"]["changedFilesIfAvailable"] == 2
        assert got["huge"]["additions"] is None and got["huge"]["deletions"] is None
        assert recs[9]["_meta"]["commit_stats_per_commit"]
        assert recs[9]["_meta"]["commit_stats_missing"] == 1
        # the next page asks for per-commit stats in the page query again
        page_queries = [q for q, v in client.calls if "pageSize" in v]
        assert ["changedFilesIfAvailable" in q for q in page_queries] == [True, False, True, False]

    def test_timeout_without_commit_stats_is_still_an_error(self, tmp_path):
        pages = {None: ([pr(9, "2026-08-01T00:00:00Z")], None)}
        client = FakeClient(pages, locate_pages=locate_top(), page_fail_over=0)
        with pytest.raises(fp.ServerTimeoutError):
            make_fetcher(client, tmp_path, page_size=1).run()
        page_queries = [q for q, v in client.calls if "pageSize" in v]
        assert [("changedFilesIfAvailable" in q, fp.REVIEW_LINK in q) for q in page_queries] == \
            [(True, True), (False, True), (False, False)]

    def test_failing_review_link_falls_back_to_comments_without_it(self, tmp_path):
        pages = {None: ([pr(9, "2026-08-01T00:00:00Z")], "P1"),
                 "P1": ([pr(8, "2026-07-01T00:00:00Z")], None)}
        client = FakeClient(pages, locate_pages=locate_top(), review_link_error=True)
        cp = make_fetcher(client, tmp_path, page_size=1).run()
        assert cp["status"] == "done" and cp["n_written"] == 2
        recs = {r["number"]: r for r in read_jsonl(tmp_path)}
        assert recs[9]["_meta"]["thread_review_link_missing"]
        assert recs[9]["_meta"]["commit_stats_per_commit"]
        # each page first asks for everything, then drops stats, then the review link
        page_queries = [q for q, v in client.calls if "pageSize" in v]
        assert [("changedFilesIfAvailable" in q, fp.REVIEW_LINK in q) for q in page_queries] == \
            [(True, True), (False, True), (False, False)] * 2

    def test_review_link_query_variants(self):
        assert fp.REVIEW_LINK in fp.build_page_query(False)
        assert fp.REVIEW_LINK not in fp.build_page_query(False, review_link=False)
        assert fp.REVIEW_LINK not in fp.build_more_query("reviewThreads", review_link=False)
        assert fp.REVIEW_LINK not in fp.THREAD_COMMENTS_QUERY_NO_REVIEW_LINK
        assert fp.REVIEW_LINK in fp.THREAD_COMMENTS_QUERY

    def test_page_size_rises_one_step_after_successful_reduced_pages(self, tmp_path):
        n = fp.RESTORE_AFTER_PAGES + 2
        pages = {}
        for i in range(n):
            nxt = "P%d" % (i + 1) if i + 1 < n else None
            pages[None if i == 0 else "P%d" % i] = ([pr(100 - i, "2026-08-01T00:00:00Z")], nxt)
        client = FakeClient(pages, locate_pages=locate_top(), page_fail_over=5)
        make_fetcher(client, tmp_path, page_size=10).run()
        sizes = [v["pageSize"] for _, v in client.calls if "pageSize" in v]
        r = fp.RESTORE_AFTER_PAGES
        # 10 fails, then r pages at 5, one try at 10 fails, back to 5
        assert sizes[:r + 3] == [10] + [5] * r + [10, 5]

    def test_ci_fields_requested_only_while_pages_are_recent(self, tmp_path):
        pages = {None: ([pr(9, "2025-10-02T00:00:00Z"), pr(8, "2025-09-01T00:00:00Z")], "P1"),
                 "P1": ([pr(7, "2025-06-01T00:00:00Z")], "P2"),
                 "P2": ([pr(1, "2023-01-01T00:00:00Z")], None)}
        client = FakeClient(pages, locate_pages=locate_top())
        make_fetcher(client, tmp_path).run()
        with_ci = ["ciCommits" in q for q, v in client.calls if "pageSize" in v]
        assert all("ciHead" not in q for q, v in client.calls if "pageSize" in v)
        assert with_ci == [True, False, False]
        recs = {r["number"]: r["_meta"] for r in read_jsonl(tmp_path)}
        assert recs[9]["ci_fetched"] and recs[8]["ci_fetched"] and not recs[7]["ci_fetched"]
        assert recs[9]["ci_window"] and not recs[8]["ci_window"]


# ── HTTP client: retries and rate limit ─────────────────────────────────────

class FakeResp:
    def __init__(self, status, payload=None, headers=None, text=""):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = text or json.dumps(payload or {})

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.posts = 0

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


NOW = datetime(2026, 10, 14, 14, 0, 0, tzinfo=timezone.utc)


def ok(remaining=4000, reset="2026-10-14T14:30:00Z"):
    return FakeResp(200, {"data": {"rateLimit": {"cost": 14, "remaining": remaining,
                                                 "resetAt": reset, "limit": 5000},
                                   "repository": {"x": 1}}})


def client_with(responses):
    sleeps = []
    c = fp.GitHubClient("t0k3n", session=FakeSession(responses), sleep=sleeps.append,
                        clock=lambda: NOW)
    return c, sleeps


class TestClient:
    def test_502_is_retried_with_backoff_then_succeeds(self):
        c, sleeps = client_with([FakeResp(502, text="bad gateway"), ok()])
        assert c.execute("q", {})["repository"] == {"x": 1}
        assert len(sleeps) == 1 and sleeps[0] >= 5

    def test_persistent_502_raises_server_timeout(self):
        c, _ = client_with([FakeResp(502, text="x"), FakeResp(502, text="x")])
        with pytest.raises(fp.ServerTimeoutError):
            c.execute("q", {}, max_attempts=2)

    def test_network_timeout_counts_as_server_timeout(self):
        import requests
        c, _ = client_with([requests.Timeout(), requests.Timeout()])
        with pytest.raises(fp.ServerTimeoutError):
            c.execute("q", {}, max_attempts=2)

    def test_low_remaining_waits_until_reset_plus_margin(self):
        c, sleeps = client_with([ok(remaining=10, reset="2026-10-14T14:20:00Z")])
        c.execute("q", {})
        assert sleeps == [20 * 60 + fp.RESET_MARGIN_S]

    def test_graphql_rate_limited_waits_and_does_not_use_attempts(self):
        limited = FakeResp(200, {"data": {"rateLimit": {"cost": 1, "remaining": 0,
                                                        "resetAt": "2026-10-14T14:10:00Z", "limit": 5000}},
                                 "errors": [{"type": "RATE_LIMITED", "message": "API rate limit exceeded"}]})
        c, sleeps = client_with([limited, limited, ok()])
        assert c.execute("q", {}, max_attempts=1)["repository"] == {"x": 1}
        assert sleeps == [600 + fp.RESET_MARGIN_S] * 2

    def test_secondary_rate_limit_uses_retry_after(self):
        c, sleeps = client_with([FakeResp(403, text="You have exceeded a secondary rate limit",
                                          headers={"Retry-After": "61"}), ok()])
        c.execute("q", {})
        assert sleeps == [66]

    def test_401_is_fatal_and_token_not_in_message_or_repr(self):
        c, _ = client_with([FakeResp(401, text="Bad credentials")])
        with pytest.raises(fp.GraphQLError) as ei:
            c.execute("q", {})
        assert "t0k3n" not in str(ei.value) and "t0k3n" not in repr(c)

    def test_graphql_timeout_error_is_retried(self):
        err = FakeResp(200, {"data": None, "errors": [{"message": "Something went wrong while executing your query. This may be the result of a timeout"}]})
        c, sleeps = client_with([err, ok()])
        assert c.execute("q", {})["repository"] == {"x": 1}

    def test_query_error_is_fatal(self):
        err = FakeResp(200, {"errors": [{"type": "undefinedField", "message": "Field 'x' doesn't exist"}]})
        c, _ = client_with([err])
        with pytest.raises(fp.GraphQLError):
            c.execute("q", {})


# ── orchestrator ────────────────────────────────────────────────────────────

class TestOrchestrator:
    def test_plan_has_40_unique_repos_in_plan_order(self):
        repos = [p[0] for p in rc.REPO_PLAN]
        assert len(repos) == 40 == len(set(repos))
        assert repos[:3] == ["eslint/eslint", "django/django", "matomo-org/matomo"]
        assert repos.index("apache/airflow") < repos.index("strapi/strapi") < repos.index("getsentry/relay")
        assert sum(1 for p in rc.REPO_PLAN if "P1" in p[1]) == 20
        assert sum(1 for p in rc.REPO_PLAN if "R12" in p[1]) == 12
        assert sum(1 for p in rc.REPO_PLAN if p[1] == "P2") == 8
        assert {p[2] for p in rc.REPO_PLAN} == {rc.COMPANY, rc.COMMUNITY}

    def test_select_only_keeps_given_order(self):
        assert [p[0] for p in rc.select_plan("django/django,eslint/eslint")] == \
            ["django/django", "eslint/eslint"]

    def test_failed_repo_is_retried_in_next_pass(self, tmp_path, monkeypatch):
        attempts = {}

        class Fetcher:
            def __init__(self, client, owner, name, raw_dir, stop_requested):
                self.repo, self.raw_dir = "%s/%s" % (owner, name), raw_dir

            def run(self):
                attempts[self.repo] = attempts.get(self.repo, 0) + 1
                if self.repo == "a/b" and attempts[self.repo] == 1:
                    raise fp.TransientError("boom")
                d = rc.repo_dir(self.raw_dir, self.repo)
                os.makedirs(d, exist_ok=True)
                fp.atomic_write_json(os.path.join(d, "checkpoint.json"), {"phase": "done"})
                return {"status": "done"}

        plan = [("a/b", "P1", rc.COMPANY, "", ""), ("c/d", "P1", rc.COMMUNITY, "", "")]
        ok_ = rc.collect(None, str(tmp_path), plan, max_passes=2, retry_pause_s=0,
                         sleep=lambda s: None, fetcher_cls=Fetcher)
        assert ok_ is True and attempts == {"a/b": 2, "c/d": 1}


class TestPartialData:
    def test_nested_forbidden_keeps_data(self):
        resp = FakeResp(200, {"data": {"rateLimit": {"cost": 1, "remaining": 4000,
                                                     "resetAt": "2026-10-14T15:00:00Z", "limit": 5000},
                                       "repository": {"x": 1}},
                              "errors": [{"type": "FORBIDDEN", "message": "Resource not accessible",
                                          "path": ["repository", "pullRequests", "nodes", 0,
                                                   "reviewRequests", "nodes", 0, "requestedReviewer"]}]})
        c, _ = client_with([resp])
        assert c.execute("q", {})["repository"] == {"x": 1}
        assert c.partial_errors == 1

    def test_top_level_not_found_is_fatal(self):
        resp = FakeResp(200, {"data": {"repository": None},
                              "errors": [{"type": "NOT_FOUND", "message": "Could not resolve",
                                          "path": ["repository"]}]})
        c, _ = client_with([resp])
        with pytest.raises(fp.GraphQLError):
            c.execute("q", {})


class TestCiDetails:
    def test_head_check_runs_attached_only_to_ci_pages(self, tmp_path):
        pages = {None: ([pr(9, "2025-10-02T00:00:00Z"), pr(8, "2025-09-01T00:00:00Z")], "P1"),
                 "P1": ([pr(7, "2025-06-01T00:00:00Z")], "P2"),
                 "P2": ([pr(1, "2023-01-01T00:00:00Z")], None)}
        client = FakeClient(pages, locate_pages=locate_top())
        make_fetcher(client, tmp_path).run()
        recs = {r["number"]: r for r in read_jsonl(tmp_path)}
        assert recs[9]["ciHead"]["nodes"][0]["commit"]["oid"] == "hPR_9"
        assert "ciHead" in recs[8] and "ciHead" not in recs[7]
        assert client.ci_chunks == [["PR_9", "PR_8"]]

    def test_timeout_splits_chunk_down_to_single_prs(self, tmp_path):
        nodes = [pr(i, "2026-01-%02dT00:00:00Z" % (30 - i)) for i in range(1, 6)]
        client = FakeClient({}, ci_fail_over=2)
        f = make_fetcher(client, tmp_path)
        cp = f._new_checkpoint()
        records, _ = f.process_page(conn(nodes), cp, seen={}, with_ci=True)
        assert all("ciHead" in r for r in records)
        assert sorted(sum(client.ci_chunks, [])) == sorted("PR_%d" % i for i in range(1, 6))
        assert max(len(c) for c in client.ci_chunks) <= 2

    def test_resource_limits_exceeded_is_retried_not_accepted_as_partial(self):
        bad = FakeResp(200, {"data": {"repository": {"x": "partial"}},
                             "errors": [{"type": "RESOURCE_LIMITS_EXCEEDED",
                                         "message": "Resource limits for this query exceeded.",
                                         "path": ["repository", "pullRequests", "nodes", 4, "ciHead"]}]})
        c, _ = client_with([bad, bad])
        with pytest.raises(fp.ServerTimeoutError):
            c.execute("q", {}, max_attempts=2)


class TestInvalidJson:
    def test_truncated_200_body_is_a_server_timeout(self):
        c, sleeps = client_with([FakeResp(200, None, text='{"data": {"repos'), ok()])
        assert c.execute("q", {})["repository"] == {"x": 1}
        c2, _ = client_with([FakeResp(200, None, text="{"), FakeResp(200, None, text="{")])
        with pytest.raises(fp.ServerTimeoutError):
            c2.execute("q", {}, max_attempts=2)

    def test_any_transient_failure_on_page_25_drops_to_10(self, tmp_path):
        pages = {None: ([pr(9, "2026-08-01T00:00:00Z")], None)}
        client = FakeClient(pages, locate_pages=locate_top(), fail_on={None: fp.TransientError("HTTP 500")})
        cp = make_fetcher(client, tmp_path).run()
        assert [v["pageSize"] for _, v in client.calls if "pageSize" in v] == [25, 10]
        assert cp["status"] == "done"
