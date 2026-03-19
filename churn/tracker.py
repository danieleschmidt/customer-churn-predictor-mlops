"""
Simple JSON-based experiment tracker.

No MLflow, no extra deps — just a flat JSON log that accumulates run
records under experiments/runs.json.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_RUNS_PATH = Path("experiments/runs.json")


class ExperimentTracker:
    def __init__(self, path: Path | str = DEFAULT_RUNS_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._runs: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._runs, indent=2))

    def log_run(
        self,
        model_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Record one training run and return its run_id."""
        run_id = str(uuid.uuid4())[:8]
        record = {
            "run_id": run_id,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "model": model_name,
            "params": params,
            "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
            "tags": tags or {},
        }
        self._runs.append(record)
        self._save()
        return run_id

    def list_runs(self) -> List[Dict[str, Any]]:
        return list(self._runs)

    def best_run(self, metric: str = "roc_auc", higher_is_better: bool = True) -> Optional[Dict[str, Any]]:
        """Return the run with the highest (or lowest) value for *metric*."""
        valid = [r for r in self._runs if metric in r.get("metrics", {})]
        if not valid:
            return None
        return max(valid, key=lambda r: r["metrics"][metric]) if higher_is_better else min(
            valid, key=lambda r: r["metrics"][metric]
        )

    def summary(self) -> str:
        lines = [f"{'run_id':10s} {'model':30s} {'roc_auc':>10s} {'f1':>8s}  timestamp"]
        lines.append("-" * 80)
        for r in self._runs:
            m = r.get("metrics", {})
            lines.append(
                f"{r['run_id']:10s} {r['model']:30s} "
                f"{m.get('roc_auc', float('nan')):>10.4f} "
                f"{m.get('f1', float('nan')):>8.4f}  "
                f"{r['timestamp'][:19]}"
            )
        return "\n".join(lines)
