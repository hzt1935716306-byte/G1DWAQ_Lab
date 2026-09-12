"""Offline certificate evaluation at TD0; never changes policy observations."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class OfflineCertificate:
    def __init__(self, env: Any, *, capability_path: Path, nominal_path: Path):
        from legged_lab.recovery.plane_certificate_runtime import PlaneCalibratedG1CertificateEvaluator, query_diagnostic
        from legged_lab.recovery.state_extractor import G1PrivilegedStateExtractor, G1StateExtractorCfg

        self._query_diagnostic = query_diagnostic
        self.extractor = G1PrivilegedStateExtractor(
            env,
            G1StateExtractorCfg(
                contact_force_threshold=5.0,
                min_touchdown_interval=0.08,
                use_terrain_plane_geometry=True,
            ),
        )
        self.evaluator = PlaneCalibratedG1CertificateEvaluator(
            capability_path, nominal_path, workers=1, executor_type="sequential", use_state_b=True,
        )

    def extract(self) -> Any:
        return self.extractor.extract()

    def evaluate(self, state: Any, env_ids: Any) -> list[dict[str, Any]]:
        pending = self.evaluator.submit(state, env_ids)
        raw = self.evaluator.resolve_raw_results(pending)
        results = []
        for query, value in zip(pending.queries, raw):
            diagnostic = self._query_diagnostic(query, value)
            valid = diagnostic["category"] == "success"
            results.append({
                "valid": valid,
                "n_min": int(value.n_min) if valid and value.n_min is not None else None,
                "margin": float(value.margin) if valid and value.margin is not None else None,
                "diagnostic": diagnostic,
            })
        return results

    def close(self) -> None:
        self.evaluator.close()
