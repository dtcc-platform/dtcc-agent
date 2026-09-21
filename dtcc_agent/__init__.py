"""dtcc-agent: LLM agent interface for the DTCC digital twin platform."""

__version__ = "0.1.0"

# dtcc-core is a hard dependency, and its absence must be loud.
#
# Historically it was not declared in pyproject.toml, and every module that
# needed it caught the ImportError and degraded silently: registry.py skipped
# dataset registration, serializers.py built an empty dispatch table, and
# dispatcher.py stopped coercing Bounds. The result was a server that started
# cleanly, answered requests, and served an empty catalogue — a failure that
# looks exactly like success until someone asks why there are no operations.
#
# The dependency is now declared and pinned, so a missing Core means a broken
# install rather than a supported degraded mode. Fail here, at import, with a
# message that says what to do.
try:  # pragma: no cover - exercised by the missing-Core test via subprocess
    import dtcc_core as _dtcc_core  # noqa: F401
except ImportError as exc:  # pragma: no cover - see above
    raise ImportError(
        "dtcc-agent requires dtcc-core, which is not importable.\n"
        "\n"
        "dtcc-core is a pinned git dependency, so a plain `pip install dtcc-agent`\n"
        "from a stale environment can leave it missing. Reinstall with:\n"
        "\n"
        "    uv sync --locked\n"
        "\n"
        "or, for an editable checkout:\n"
        "\n"
        "    uv pip install -e ../dtcc-core -e .\n"
        "\n"
        "Do not work around this by catching the error: without dtcc-core the\n"
        "operation catalogue is empty and every geometry tool is inert."
    ) from exc

del _dtcc_core
