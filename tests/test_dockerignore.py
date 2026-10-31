"""Local secret files stay out of the image (#133).

The Dockerfile copies the whole build context (`COPY . /app`), so anything
`.dockerignore` lets through is baked in, including a local `.env` with keys.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_env_files_are_excluded_from_the_build_context():
    patterns = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    }
    assert {".env", ".env.*"} <= patterns
