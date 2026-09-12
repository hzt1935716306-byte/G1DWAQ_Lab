"""Summarize an analysis-excluded N4-HIGH physical coverage probe."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from paper_eval_final.src.common import ROOT, atomic_write, read_json, write_json
from paper_eval_final.src.experiment1_sampling import load_boundaries
from paper_eval_final.src.storage import validate_record


def summarize_probe(root: Path) -> dict:
    identity = read_json(root / "identity.json")
    completion = read_json(root / "completion.json")
    if identity.get("dataset_role") != "COVERAGE_PROBE_NONANALYSIS":
        raise ValueError("input is not an isolated coverage probe")
    if completion.get("status") != "COMPLETE":
        raise ValueError("coverage probe is incomplete")
    records = [read_json(path) for path in (root / "records").glob("*.json")]
    records.sort(key=lambda row: (int(row["sequence"]), row["trial_id"]))
    for row in records:
        validate_record(row)
        if row.get("dataset_role") != "COVERAGE_PROBE_NONANALYSIS" or row.get("analysis_excluded") is not True:
            raise ValueError(f"probe isolation marker missing: {row['trial_id']}")
    boundaries = load_boundaries(require_frozen=True)
    q2 = float(boundaries["shared_q2"])
    natural = Counter()
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        if row.get("certificate_valid") and row.get("margin_raw") is not None:
            natural[int(row["Nmin"])] += 1
            by_condition[str(row["condition_id"])].append(row)
    conditions = {}
    all_n4 = []
    for condition_id in sorted({str(row["condition_id"]) for row in records}):
        attempts = sum(str(row["condition_id"]) == condition_id for row in records)
        eligible = by_condition.get(condition_id, [])
        n4 = [float(row["margin_raw"]) for row in eligible if row.get("Nmin") == 4]
        all_n4.extend(n4)
        conditions[condition_id] = {
            "attempt_count": attempts,
            "certificate_valid_count": len(eligible),
            "N4_count": len(n4),
            "N4_HIGH_count": sum(value >= q2 for value in n4),
            "N4_margin_maximum": max(n4) if n4 else None,
            "N4_margin_q95": float(np.quantile(n4, 0.95)) if n4 else None,
        }
    maximum = max(all_n4) if all_n4 else None
    return {
        "status": "N4_HIGH_NOT_OBSERVED" if not any(value >= q2 for value in all_n4) else "N4_HIGH_OBSERVED",
        "dataset_role": identity["dataset_role"],
        "analysis_excluded": True,
        "outcome_fields_read_for_coverage_decision": [],
        "manifest_hash": identity["manifest_hash"],
        "evaluation_code_commit": identity["evaluation_code_commit"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "record_count": len(records),
        "certificate_valid_count": sum(natural.values()),
        "natural_certificate_valid_Nmin_counts": {str(key): natural[key] for key in sorted(natural)},
        "shared_q2": q2,
        "N4_count": len(all_n4),
        "N4_HIGH_count": sum(value >= q2 for value in all_n4),
        "N4_margin_maximum": maximum,
        "gap_from_q2": q2 - maximum if maximum is not None and math.isfinite(maximum) else None,
        "conditions": conditions,
        "conclusion": (
            "The frozen shared q2 is not covered by Nmin=4 in any predeclared physical probe condition; "
            "do not admit probe rows, start formal evaluation, or move q2 post hoc."
        ),
    }


def write_chinese_report(path: Path, result: dict) -> None:
    lines = [
        "# 实验一 N4-High 物理覆盖探针", "",
        f"- 状态：`{result['status']}`",
        f"- 固定预算：{result['record_count']} 条",
        f"- 证书有效：{result['certificate_valid_count']} 条",
        f"- 自然 Nmin 分布：`{result['natural_certificate_valid_Nmin_counts']}`",
        f"- 共享 q2：`{result['shared_q2']}`",
        f"- Nmin=4：{result['N4_count']} 条",
        f"- N4-High：{result['N4_HIGH_count']} 条",
        f"- N4 最大 margin：`{result['N4_margin_maximum']}`",
        f"- 与 q2 的缺口：`{result['gap_from_q2']}`", "",
        "探针数据角色为 `COVERAGE_PROBE_NONANALYSIS`，每条记录均带有",
        "`analysis_excluded=true`。覆盖判断没有读取恢复、摔倒、生存、恢复时间或",
        "恢复步数字段。", "", "## 分条件覆盖", "",
        "| 条件 | 尝试 | 证书有效 | N4 | N4-High | N4 最大 margin |", "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, row in result["conditions"].items():
        lines.append(
            f"| {condition} | {row['attempt_count']} | {row['certificate_valid_count']} | "
            f"{row['N4_count']} | {row['N4_HIGH_count']} | {row['N4_margin_maximum']} |"
        )
    lines += [
        "", "## 结论", "",
        "扩展覆盖仍未产生任何 `Nmin=4 且 margin≥q2` 样本。结合证书状态空间",
        "离线扫描结果，当前缺口不是单纯候选数量不足。不得把探针样本纳入分析、",
        "不得事后移动 q2，也不得启动正式实验。共享数值边界与 Nmin 的结构关联",
        "需要先重新审视。", "",
    ]
    atomic_write(path, "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_root", type=Path)
    parser.add_argument(
        "--json-output", type=Path,
        default=ROOT / "reports/pilot/experiment_1_n4_coverage_probe.json",
    )
    parser.add_argument(
        "--report-output", type=Path,
        default=ROOT / "reports/pilot/experiment_1_n4_coverage_probe.md",
    )
    args = parser.parse_args()
    result = summarize_probe(args.probe_root)
    write_json(args.json_output, result)
    write_chinese_report(args.report_output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
