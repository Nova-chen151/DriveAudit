from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Tuple

from driveaudit_eval.video.runtime_state import RuntimeState


@dataclass(frozen=True)
class EvaluationNode:
    name: str
    fn: Callable[[RuntimeState], None]
    depends_on: Tuple[str, ...] = field(default_factory=tuple)
    kind: str = "node"
    outputs: Tuple[str, ...] = field(default_factory=tuple)


class EvaluationGraph:
    """Small deterministic DAG runner for benchmark evaluation."""

    def __init__(self) -> None:
        self.nodes: "OrderedDict[str, EvaluationNode]" = OrderedDict()

    def add_node(
        self,
        name: str,
        fn: Callable[[RuntimeState], None],
        *,
        depends_on: Iterable[str] = (),
        kind: str = "node",
        outputs: Iterable[str] = (),
    ) -> None:
        if name in self.nodes:
            raise ValueError(f"duplicate evaluation graph node: {name}")
        deps = tuple(depends_on)
        for dep in deps:
            if dep not in self.nodes:
                raise ValueError(f"node {name} depends on unknown node {dep}")
        self.nodes[name] = EvaluationNode(name, fn, deps, kind, tuple(outputs))

    def describe(self) -> Dict[str, List[Dict[str, object]]]:
        return {
            "nodes": [
                {
                    "index": idx,
                    "name": node.name,
                    "kind": node.kind,
                    "depends_on": list(node.depends_on),
                    "outputs": list(node.outputs),
                }
                for idx, node in enumerate(self.nodes.values())
            ]
        }

    def execute(self, state: RuntimeState) -> None:
        executed = set()
        failed = {}
        pending = OrderedDict(self.nodes)
        while pending:
            progressed = False
            for name, node in list(pending.items()):
                if not set(node.depends_on).issubset(executed):
                    continue
                record = {
                    "index": len(state.execution_trace),
                    "name": node.name,
                    "kind": node.kind,
                    "depends_on": list(node.depends_on),
                    "outputs": list(node.outputs),
                    "status": "started",
                }
                state.register_trace(record)
                try:
                    if node.kind == "aggregation":
                        record["status"] = "completed"
                    blocked = [dep for dep in node.depends_on if dep in failed]
                    if blocked and node.kind != "aggregation":
                        raise ImportError(f"dependency nodes failed: {blocked}")
                    node.fn(state)
                except Exception as exc:
                    record["status"] = "failed"
                    record["error_type"] = type(exc).__name__
                    record["error"] = str(exc)
                    from driveaudit_eval.video.evaluator import RESULT_KEYS
                    metric_name = next((key for key, result in RESULT_KEYS.items()
                                        if key in node.name.split(".") or result in node.name.split(".")), None)
                    if metric_name is None:
                        raise
                    failed[name] = str(exc)
                    skipped = isinstance(exc, (ImportError, FileNotFoundError))
                    result = {"status": "SKIPPED" if skipped else "FAILED", "enabled": True,
                              "skipped": True, "reason": f"{type(exc).__name__}: {exc}",
                              "metric_validity": "unavailable", "metric_confidence": "unavailable"}
                    state.results[RESULT_KEYS[metric_name]] = result
                    state.cfg["metrics"][metric_name]["enabled"] = False
                if name not in failed:
                    record["status"] = "completed"
                executed.add(name)
                del pending[name]
                progressed = True
            if not progressed:
                unresolved = {name: list(node.depends_on) for name, node in pending.items()}
                raise RuntimeError(f"evaluation graph has unresolved dependencies: {unresolved}")
