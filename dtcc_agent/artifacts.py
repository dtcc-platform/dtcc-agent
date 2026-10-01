"""Files a Session produces, and the only files a tool may read (T7, #21).

Each Session writes into its own directory under ARTIFACTS_DIR, named by a
random token, so no caller-supplied path ever reaches the filesystem (U1,
#10). The MCP server writes artifacts; the chatbot serves them back at
/artifacts/<session>/<name>. Both import this module, so they agree on the
layout: in the two-service deployment (T13) they share the directory.

GeoJSON input is read only from SHARED_RESULTS_DIR, where dtcc-sim writes
its results.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import tempfile
from pathlib import Path

# Chatbot Session ids are hex; "local" is the stdio default. Anything else
# never becomes a directory name.
_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
# <32 hex token>_<stem>.<ext>; the stem is what a download is saved as.
_NAME = re.compile(r"[0-9a-f]{32}_[A-Za-z0-9_.-]{1,64}\.[a-z0-9]{1,8}")
_STEM_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")

IMAGE_SUFFIXES = {".png"}


def root() -> Path:
    return Path(
        os.getenv("DTCC_AGENT_ARTIFACTS_DIR")
        or Path(tempfile.gettempdir()) / "dtcc_agent_artifacts"
    )


def _session_dir(session_id: str) -> Path:
    if not _SESSION_ID.fullmatch(session_id):
        raise ValueError(f"Refused: {session_id!r} is not a valid Session id.")
    return root() / session_id


def new_path(session_id: str, stem: str, suffix: str) -> Path:
    """A fresh, unguessable file path in this Session's directory."""
    directory = _session_dir(session_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    stem = _STEM_UNSAFE.sub("_", stem)[:64] or "artifact"
    return directory / f"{secrets.token_hex(16)}_{stem}{suffix.lower()}"


def describe(path: Path) -> dict[str, str]:
    """What a tool returns for an artifact: never the filesystem path."""
    kind = "image" if path.suffix in IMAGE_SUFFIXES else "file"
    return {"name": path.name, "kind": kind}


def find(session_id: str, name: str) -> Path | None:
    """This Session's artifact called `name`, or None."""
    if not _NAME.fullmatch(name) or not _SESSION_ID.fullmatch(session_id):
        return None
    path = _session_dir(session_id) / name
    return path if path.is_file() and not path.is_symlink() else None


def download_name(name: str) -> str:
    """The name a download is saved as: the artifact name without its token."""
    return name.split("_", 1)[1]


def remove_session(session_id: str) -> None:
    if _SESSION_ID.fullmatch(session_id):
        shutil.rmtree(_session_dir(session_id), ignore_errors=True)


def shared_result(relative: str) -> Path | None:
    """A file under SHARED_RESULTS_DIR, or None if unset, absent, or outside it."""
    base = os.getenv("SHARED_RESULTS_DIR")
    if not base or not relative or Path(relative).is_absolute():
        return None
    base_path = Path(base).resolve()
    path = (base_path / relative).resolve()
    if not path.is_relative_to(base_path) or not path.is_file():
        return None
    return path
