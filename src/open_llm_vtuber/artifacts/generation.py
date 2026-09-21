"""Resource controls for artifact/document generation jobs."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import monotonic
from typing import AsyncIterator, Callable, TypeVar

T = TypeVar("T")

DEFAULT_MAX_CONCURRENT_JOBS = 2
DEFAULT_MAX_JOB_SECONDS = 60.0
ABSOLUTE_MAX_CONCURRENT_JOBS = 8
ABSOLUTE_MAX_JOB_SECONDS = 300.0


@dataclass(slots=True)
class GenerationLimits:
    max_concurrent_jobs: int = DEFAULT_MAX_CONCURRENT_JOBS
    max_job_seconds: float = DEFAULT_MAX_JOB_SECONDS

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_concurrent_jobs, int)
            or not 0 < self.max_concurrent_jobs <= ABSOLUTE_MAX_CONCURRENT_JOBS
        ):
            raise ValueError(
                f"max_concurrent_jobs must be between 1 and {ABSOLUTE_MAX_CONCURRENT_JOBS}"
            )
        if (
            not isinstance(self.max_job_seconds, (int, float))
            or not 0 < float(self.max_job_seconds) <= ABSOLUTE_MAX_JOB_SECONDS
        ):
            raise ValueError(
                f"max_job_seconds must be between 0 and {ABSOLUTE_MAX_JOB_SECONDS} seconds"
            )


class ArtifactGenerationManager:
    """Bound concurrent generation work and keep blocking libraries off the event loop.

    Timeouts are soft: asyncio cannot safely kill a Python thread. The semaphore still
    bounds new work, and the generator functions perform their own input-size limits.
    """

    def __init__(self, limits: GenerationLimits | None = None) -> None:
        self.limits = limits or GenerationLimits()
        self._semaphore = asyncio.Semaphore(self.limits.max_concurrent_jobs)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        await self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()

    async def run(self, func: Callable[..., T], *args: object, **kwargs: object) -> T:
        started = monotonic()
        async with self.slot():
            result = await asyncio.to_thread(func, *args, **kwargs)
        elapsed = monotonic() - started
        if elapsed > self.limits.max_job_seconds:
            raise TimeoutError(
                f"Artifact generation exceeded the soft time limit ({self.limits.max_job_seconds:.1f}s)"
            )
        return result
