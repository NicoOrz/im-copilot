"""Timing collector for LangGraph node execution profiling.

Uses contextvars so the collector is scoped to the current async task or
thread without polluting the LangGraph state schema.
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import asdict, dataclass, field


@dataclass
class NodeTiming:
    node: str
    start_ms: float
    duration_ms: float
    status: str


@dataclass
class TimingCollector:
    graph_start_ms: float = 0.0
    graph_duration_ms: float = 0.0
    entries: list[NodeTiming] = field(default_factory=list)

    def record(self, node: str, start_ms: float, duration_ms: float, status: str) -> None:
        self.entries.append(NodeTiming(
            node=node,
            start_ms=start_ms - self.graph_start_ms,
            duration_ms=duration_ms,
            status=status,
        ))

    def to_dict(self) -> dict:
        return {
            "graph_duration_ms": self.graph_duration_ms,
            "nodes": [asdict(e) for e in self.entries],
        }


_current_collector: contextvars.ContextVar[TimingCollector | None] = contextvars.ContextVar(
    "timing_collector", default=None,
)


def start_collecting() -> TimingCollector:
    collector = TimingCollector(graph_start_ms=time.perf_counter() * 1000)
    _current_collector.set(collector)
    return collector


def get_collector() -> TimingCollector | None:
    return _current_collector.get()


def stop_collecting() -> TimingCollector | None:
    collector = _current_collector.get()
    if collector:
        collector.graph_duration_ms = (time.perf_counter() * 1000) - collector.graph_start_ms
    _current_collector.set(None)
    return collector
