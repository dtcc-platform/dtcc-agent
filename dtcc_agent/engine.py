"""Client for the DTCC Engine API (dtcc-twin ``apps/engine``).

With DTCC_ENGINE_URL set, simulations run as Engine jobs (#109): submit, poll
until the job finishes, and download its result as a ``.dtccpkg`` that Core
reads. Results come back over HTTP, so nothing is shared on disk with the
service that computed them, unlike the dtcc-sim mini-service.

Engine API (dtcc-twin ``apps/engine/dtcc_engine/api.py``), every route but
health behind one bearer token:

- ``GET  /api/v1/datasets``: ``{"datasets": {name: describe()}, ...}``
- ``POST /api/v1/jobs`` ``{"dataset", "parameters"}``: 202 ``{"job_id", "status_url"}``
- ``GET  /api/v1/jobs/{id}``: ``{"state": queued|unconfirmed|running|completed|failed, ...}``
- ``GET  /api/v1/jobs/{id}/package``: the result package
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from typing import Any, Callable

import httpx

# A package larger than this is refused before Core reads it: the Engine is a
# separate service, and Core asks callers of untrusted packages for a budget.
# The default object budget (server.OBJECT_BUDGET_BYTES) is 2 GiB too.
MAX_PACKAGE_BYTES = 2 * 1024**3
POLL_FIRST_SECONDS = 1.0
POLL_MAX_SECONDS = 5.0
DEFAULT_TIMEOUT_SECONDS = 900.0
REQUEST_TIMEOUT_SECONDS = 30.0


class EngineError(RuntimeError):
    """An Engine job that did not produce a result, said so the model can relay it."""


def url() -> str | None:
    """The configured Engine base URL, or None when simulations don't use it."""
    value = os.getenv("DTCC_ENGINE_URL", "").strip().rstrip("/")
    return value or None


# Experimental (T41, #110): with DTCC_ENGINE_CORE=1 these downloads run as
# Engine jobs too, without the disk cache, so the two can be measured against
# each other. T42 (#111) keeps or removes the switch.
CORE_DATASETS = frozenset({"datasets.buildings", "datasets.point_cloud"})


def core_enabled() -> bool:
    """Whether CORE_DATASETS download on the Engine (DTCC_ENGINE_CORE=1)."""
    return os.getenv("DTCC_ENGINE_CORE", "").strip() == "1"


def check_config() -> None:
    """Refuse to start with an Engine URL and no token, or with the Core switch
    and no Engine: every job would fail."""
    if url() and not os.getenv("DTCC_ENGINE_TOKEN", "").strip():
        raise RuntimeError("DTCC_ENGINE_URL is set but DTCC_ENGINE_TOKEN is not; "
                           "set the Engine's API token, or unset DTCC_ENGINE_URL")
    if core_enabled() and not url():
        raise RuntimeError("DTCC_ENGINE_CORE=1 needs DTCC_ENGINE_URL; "
                           "set the Engine, or unset DTCC_ENGINE_CORE")


def download(name: str, params: dict[str, Any]) -> Any:
    """Run one of CORE_DATASETS as an Engine job and return its model.
    Raises EngineError, which callers return to the model as an error."""
    model, _job_id, _size = client().run(name.removeprefix("datasets."), params)
    return model


class Engine:
    """One Engine, reached over HTTP. Thread-safe: each call makes its own requests."""

    def __init__(self, base_url: str, token: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.base_url = base_url
        self.timeout = timeout
        self._http = httpx.Client(base_url=base_url, transport=transport,
                                  headers={"Authorization": f"Bearer {token}"},
                                  timeout=REQUEST_TIMEOUT_SECONDS)
        self._sleep = sleep
        self._clock = clock
        self._datasets: dict[str, Any] | None = None
        self._lock = threading.Lock()

    # -- Requests ------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise EngineError(f"The Engine at {self.base_url} is unreachable: {exc}") from exc
        if response.status_code == 401:
            raise EngineError("The Engine rejected the agent's token (DTCC_ENGINE_TOKEN); "
                              "tell the administrator.")
        return response

    @staticmethod
    def _detail(response: httpx.Response) -> Any:
        try:
            return response.json().get("detail", response.text)
        except ValueError:
            return response.text

    # -- Discovery -----------------------------------------------------------

    def datasets(self) -> dict[str, Any]:
        """Every dataset's Core ``describe()``, keyed by name; asked once per process."""
        with self._lock:
            if self._datasets is None:
                response = self._request("GET", "/api/v1/datasets")
                if response.status_code != 200:
                    raise EngineError(f"The Engine could not list its datasets "
                                      f"(HTTP {response.status_code}): {self._detail(response)}")
                self._datasets = response.json()["datasets"]
            return self._datasets

    def describe(self, name: str) -> dict[str, Any]:
        """One dataset's ``describe()``; KeyError when the Engine doesn't offer it."""
        return self.datasets()[name]

    # -- Jobs ----------------------------------------------------------------

    def submit(self, dataset: str, parameters: dict[str, Any]) -> str:
        """Submit a job and return its id. The Engine always answers with a
        package, and refuses a ``format`` parameter, so any is dropped."""
        body = {"dataset": dataset,
                "parameters": {k: v for k, v in parameters.items() if k != "format"}}
        response = self._request("POST", "/api/v1/jobs", json=body)
        if response.status_code == 202:
            return response.json()["job_id"]
        detail = self._detail(response)
        if response.status_code == 503 and isinstance(detail, dict) and detail.get("job_id"):
            # The queue didn't confirm the job, which may still run: poll it,
            # never submit it again (the Engine's own instruction).
            return detail["job_id"]
        if response.status_code == 422:
            raise EngineError(f"The Engine refused the parameters for {dataset}: {detail}")
        if response.status_code == 503:
            raise EngineError(f"The Engine's job queue is unavailable; try again shortly ({detail})")
        raise EngineError(f"The Engine did not accept the job (HTTP {response.status_code}): {detail}")

    def wait(self, job_id: str) -> dict[str, Any]:
        """Poll until the job completes, and return its status.

        Polls after 1 s, doubling to every 5 s, until ``timeout`` seconds pass.
        """
        deadline = self._clock() + self.timeout
        delay = POLL_FIRST_SECONDS
        while True:
            response = self._request("GET", f"/api/v1/jobs/{job_id}")
            if response.status_code == 404:
                raise EngineError(f"Engine job {job_id} is unknown or expired; run it again.")
            if response.status_code == 200:
                job = response.json()
                if job["state"] == "completed":
                    return job
                if job["state"] == "failed":
                    kind = (job.get("error") or {}).get("type") or "unknown error"
                    raise EngineError(f"Engine job {job_id} failed: {kind}")
            # 503 (job state briefly unknown) or still queued/running: wait.
            if self._clock() + delay > deadline:
                raise EngineError(f"Engine job {job_id} did not finish within "
                                  f"{self.timeout:.0f} s; it may still be running.")
            self._sleep(delay)
            delay = min(delay * 2, POLL_MAX_SECONDS)

    def package(self, job_id: str) -> tuple[Any, int]:
        """Download a completed job's package and return its model and size in bytes."""
        from dtcc_core.datasets import load_model_package

        with tempfile.NamedTemporaryFile(suffix=".dtccpkg") as f:
            try:
                with self._http.stream("GET", f"/api/v1/jobs/{job_id}/package") as response:
                    if response.status_code == 401:
                        raise EngineError("The Engine rejected the agent's token "
                                          "(DTCC_ENGINE_TOKEN); tell the administrator.")
                    if response.status_code != 200:
                        response.read()
                        raise EngineError(f"Engine job {job_id}'s result could not be downloaded "
                                          f"(HTTP {response.status_code}): {self._detail(response)}")
                    for chunk in response.iter_bytes():
                        f.write(chunk)
                        if f.tell() > MAX_PACKAGE_BYTES:
                            raise EngineError(f"Engine job {job_id}'s result is larger than "
                                              f"{MAX_PACKAGE_BYTES // 1024**3} GiB; try a smaller area.")
            except httpx.HTTPError as exc:
                raise EngineError(f"Engine job {job_id}'s result could not be downloaded: {exc}") from exc
            f.flush()
            size = f.tell()
            return load_model_package(f.name, max_bytes=MAX_PACKAGE_BYTES), size

    def run(self, dataset: str, parameters: dict[str, Any]) -> tuple[Any, str, int]:
        """Run ``dataset`` as a job; return its model, job id and package size."""
        job_id = self.submit(dataset, parameters)
        self.wait(job_id)
        model, size = self.package(job_id)
        return model, job_id, size


_client: Engine | None = None
_client_lock = threading.Lock()


def client() -> Engine:
    """The configured Engine, made on first use. Call only when ``url()`` is set."""
    global _client
    base = url()
    if base is None:
        raise RuntimeError("DTCC_ENGINE_URL is not set")
    check_config()
    with _client_lock:
        if _client is None or _client.base_url != base:
            timeout = float(os.getenv("DTCC_ENGINE_TIMEOUT", "") or DEFAULT_TIMEOUT_SECONDS)
            _client = Engine(base, os.getenv("DTCC_ENGINE_TOKEN", "").strip(), timeout=timeout)
        return _client
