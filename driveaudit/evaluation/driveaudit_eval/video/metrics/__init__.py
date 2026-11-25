from __future__ import annotations

from typing import Any, Dict


class BaseMetric:
    name = "base"

    def __init__(self, cfg: Dict[str, Any], state) -> None:
        self.cfg = cfg
        self.state = state
        self.metric_cfg = cfg["metrics"].get(self.name, {})
        self.enabled = bool(self.metric_cfg.get("enabled", True))
        self.skipped = False
        self.reason = None
        self.summary: Dict[str, Any] = {
            "enabled": self.enabled,
            "skipped": False,
            "reason": None,
            "backend": None,
            "metric_confidence": None,
            "metric_validity": None,
            "comparison_validity": None,
        }

    def disable(self):
        self.enabled = False
        self.skipped = True
        self.reason = "disabled"
        self.summary.update(
            {
                "enabled": False,
                "skipped": True,
                "reason": "disabled",
                "backend": "disabled",
                "metric_confidence": "disabled",
                "metric_validity": "disabled",
                "comparison_validity": "invalid",
            }
        )

    def warn(self, warning_type: str, message: str, scene_id=None, severity="warning"):
        self.state.warnings.add(self.name, warning_type, message, scene_id, severity)

    def skip(self, reason: str, scene_id=None, message=None):
        self.skipped = True
        self.reason = reason
        self.summary["skipped"] = True
        self.summary["reason"] = reason
        self.summary["backend"] = self.summary.get("backend") or "skipped"
        self.summary["metric_confidence"] = "skipped"
        self.summary["metric_validity"] = "skipped"
        self.summary["comparison_validity"] = "invalid"
        self.warn(reason, message or reason, scene_id=scene_id)

    def prepare(self):
        return None

    def update(self, scene_sample):
        return None

    def compute(self):
        self.summary["enabled"] = self.enabled
        self.summary["skipped"] = self.skipped
        self.summary["reason"] = self.reason
        if not self.enabled:
            self.disable()
        else:
            self.summary.setdefault("backend", None)
            self.summary.setdefault("metric_confidence", None)
            self.summary.setdefault("metric_validity", None)
            self.summary.setdefault("comparison_validity", None)
        return self.summary

    def save(self, output_dir):
        return None
