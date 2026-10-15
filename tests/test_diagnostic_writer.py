"""Diagnostic writes do not hold the game loop or displace action evidence."""

from __future__ import annotations

import asyncio

from gameplay_agent.diagnostic_writer import DiagnosticWriter


def test_operation_evidence_evictions_are_bounded(tmp_path) -> None:
    writer = DiagnosticWriter(limit=2)
    sample_a = tmp_path / "sample_a.jpg"
    sample_b = tmp_path / "sample_b.jpg"
    action = tmp_path / "action.jpg"

    async def run() -> None:
        assert writer.submit(sample_a, b"a")
        assert writer.submit(sample_b, b"b")
        assert writer.submit(action, b"evidence", operation_evidence=True)
        await writer.close()

    asyncio.run(run())
    assert writer.dropped_samples == 1
    assert not sample_a.exists()
    assert sample_b.read_bytes() == b"b"
    assert action.read_bytes() == b"evidence"


def test_full_operation_queue_reports_drop(tmp_path) -> None:
    writer = DiagnosticWriter(limit=1)

    async def run() -> None:
        assert writer.submit(tmp_path / "first.jpg", b"a", operation_evidence=True)
        assert not writer.submit(tmp_path / "second.jpg", b"b", operation_evidence=True)
        await writer.close()

    asyncio.run(run())
    assert writer.dropped_evidence == 1
