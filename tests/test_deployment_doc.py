"""docs/deployment.md and docker-compose.yml name the same variables (T43, #112).

The doc is what the deployment design reads, so a variable it names that
compose doesn't set is a setting that doesn't exist, and a variable compose
substitutes that the doc leaves out is a knob the design never hears about.
Both directions are checked.

The doc names variables in backticks, in capitals. It refers to code constants
by file instead, so they aren't mistaken for variables.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = (ROOT / "docs" / "deployment.md").read_text(encoding="utf-8")
COMPOSE = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")


def _doc_variables() -> set[str]:
    return set(re.findall(r"`([A-Z][A-Z0-9_]{2,})`", DOC))


def _compose_set() -> set[str]:
    """Every variable compose substitutes or sets in a container."""
    substituted = re.findall(r"\$\{([A-Z][A-Z0-9_]*)", COMPOSE)
    set_in_env = re.findall(r"^\s*-\s*([A-Z][A-Z0-9_]*)=", COMPOSE, re.M)
    build_args = re.findall(r"^\s*([A-Z][A-Z0-9_]*):\s*\$\{", COMPOSE, re.M)
    return set(substituted) | set(set_in_env) | set(build_args)


def test_doc_names_variables():
    """Guards the regex: a doc that names nothing would pass vacuously."""
    assert {"DTCC_MCP_SECRET", "DTCC_ENGINE_URL", "AWS_REGION"} <= _doc_variables()


def test_every_variable_the_doc_names_is_in_compose():
    missing = _doc_variables() - _compose_set()
    assert not missing, (
        f"docs/deployment.md names {sorted(missing)}, which docker-compose.yml doesn't set; "
        "add them to compose or take them out of the doc"
    )


def test_every_compose_setting_is_in_the_doc():
    substituted = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)", COMPOSE))
    missing = substituted - _doc_variables()
    assert not missing, (
        f"docker-compose.yml takes {sorted(missing)} from the environment, "
        "but docs/deployment.md doesn't describe them"
    )
