# Analysis outputs (git-ignored) may hold extracted copies of tagged releases
# with their own tests; never collect them.
collect_ignore_glob = ["revision/output/*", "rank/output/*", "predict/output/*"]
