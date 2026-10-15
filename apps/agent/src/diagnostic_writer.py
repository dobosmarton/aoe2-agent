"""Bounded, nonblocking screenshot writes owned by one game run."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

log = structlog.stdlib.get_logger()
_MAX_PENDING_WRITES = 24


@dataclass(frozen=True, slots=True)
class _Write:
    path: Path
    data: bytes
    operation_evidence: bool


def _write_file(job: _Write) -> None:
    job.path.parent.mkdir(parents=True, exist_ok=True)
    job.path.write_bytes(job.data)


class DiagnosticWriter:
    """Drop sampled frames before operation evidence; never block perception."""

    def __init__(self, limit: int = _MAX_PENDING_WRITES) -> None:
        self._limit = limit
        self._pending: deque[_Write] = deque()
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self.dropped_samples = 0
        self.dropped_evidence = 0

    def submit(self, path: Path, data: bytes, *, operation_evidence: bool = False) -> bool:
        if self._closed:
            return False
        if len(self._pending) >= self._limit:
            if not operation_evidence:
                self.dropped_samples += 1
                log.warning("diagnostic_write_dropped", kind="sample", count=self.dropped_samples)
                return False
            sample = next((job for job in self._pending if not job.operation_evidence), None)
            if sample is None:
                self.dropped_evidence += 1
                log.error("diagnostic_write_dropped", kind="operation", count=self.dropped_evidence)
                return False
            self._pending.remove(sample)
            self.dropped_samples += 1
        self._pending.append(_Write(path, data, operation_evidence))
        if self._worker is None:
            self._worker = asyncio.create_task(self._drain())
        return True

    async def _drain(self) -> None:
        try:
            while self._pending:
                job = self._pending.popleft()
                try:
                    await asyncio.to_thread(_write_file, job)
                except OSError as exc:
                    log.error("diagnostic_write_failed", path=str(job.path), error=repr(exc))
        finally:
            self._worker = None

    async def close(self) -> None:
        self._closed = True
        if self._worker is not None:
            await self._worker


__all__ = ["DiagnosticWriter"]
