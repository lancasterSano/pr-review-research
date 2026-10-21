"""Where the analyses in revision/ write their results.

Results go under one output root, in the same sub-folders the papers use
(journal_articles/HAIT_2026/revision_results, conf_papers/october_2026/...).
The root is taken from DISSERTATION_DIR, first from the environment, then from
the repository's .env file. Without it, results go to revision/output/, which
is git-ignored. A relative DISSERTATION_DIR is resolved against the repository
root.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT, ".env")
DEFAULT_OUTPUT = os.path.join(ROOT, "revision", "output")


def _from_env_file(env_path, key):
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):]
                if line.startswith(key + "="):
                    return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


def output_root(env_path=ENV_PATH):
    value = (os.environ.get("DISSERTATION_DIR") or "").strip() or _from_env_file(env_path, "DISSERTATION_DIR")
    if not value:
        return DEFAULT_OUTPUT
    return os.path.normpath(os.path.join(ROOT, os.path.expanduser(value)))


DISSERTATION = output_root()
