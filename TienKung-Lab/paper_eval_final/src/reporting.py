"""Human-readable technical reports; raw records remain immutable."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import ROOT, atomic_write


def write_experiment_report(stage: str, experiment: int, aggregate: dict[str, Any]) -> Path:
    stage_name = {"screening": "筛选", "pilot": "预实验", "formal": "正式实验"}.get(stage, stage)
    lines = [f"# 实验 {experiment} — {stage_name}", "", f"记录数：{aggregate['record_count']}", ""]
    for model_id, payload in aggregate["models"].items():
        summary = payload["summary"]
        lines.extend([
            f"## {model_id}", "", f"正式训练种子状态：`{payload['formal_status']}`", "",
            f"- 分母统计：`{summary['denominators']}`",
            f"- 生存率：`{summary['survival_rate']}`",
            f"- 持续恢复率：`{summary['recovery_rate']}`",
            f"- 恢复时间 Trec：`{summary['Trec']}`",
            f"- Spearman(Nmin, nTD0)：`{summary['spearman_Nmin_nTD0']}`", "",
        ])
        relation = payload.get("stratified_relation_set")
        if relation is not None:
            lines.extend([
                f"- 各 Nmin 条件 margin 边界：`{relation.get('boundaries')}`",
                f"- 校准 manifest hash：`{relation.get('calibration_manifest_hash')}`",
                f"- 分析集 manifest hash：`{relation.get('evaluation_manifest_hash')}`",
                f"- 分层关系集是否完整：`{relation['complete']}`",
                f"- 分层关系集接纳 trial 数：{len(relation['accepted_trial_ids'])}",
                f"- 分层单元统计：`{relation['cells']}`",
                f"- 分层 Spearman(Nmin, nTD0)：`{relation['summary']['spearman_Nmin_nTD0']}`",
                "",
            ])
    path = ROOT / "reports" / stage / f"experiment_{experiment}.md"
    atomic_write(path, "\n".join(lines) + "\n")
    return path


def write_screening_report(aggregate: dict[str, Any]) -> Path:
    from .model_loader import registry

    lines = ["# 实验一基线筛选", "",
             "筛选结果仅用于探索，不得与预实验或正式实验的证据合并。",
             "系统不会自动选择基线；评审结果后必须显式冻结所选基线。", ""]
    baseline_registry = registry()["screening_baselines"]
    model_to_baselines: dict[str, list[str]] = {}
    for baseline_id, row in baseline_registry.items():
        if row.get("model"):
            model_to_baselines.setdefault(row["model"], []).append(baseline_id)
    for model_id, payload in aggregate["models"].items():
        summary = payload["summary"]
        detail = payload["screening_detail"]
        occupancy = detail["N_margin_occupancy"]
        occupied = sum(value["n"] > 0 for value in occupancy.values())
        d = summary["denominators"]
        lines.extend([
            f"## {model_id} ({', '.join(model_to_baselines.get(model_id, []))})", "",
            f"- checkpoint SHA-256：`{', '.join(payload['checkpoint_sha256'])}`",
            f"- 有效 trial：{d['valid']}",
            f"- 物理覆盖：{d['valid']}/{d['executed']}",
            f"- 扰动前失败：{d['pre_push_physical_failure']}",
            f"- 已施加扰动：{d['push_applied']}",
            f"- TD0 覆盖：{detail['TD0_count']}/{d['valid']}",
            f"- 有效证书覆盖：{detail['certificate_valid_count']}/{detail['TD0_count']}",
            f"- 主关系分析 N=2/3/4 数量：`{detail['relation_Nmin_counts']}`",
            f"- 固定预算自然 Nmin 分布（含 N=5）：`{detail['natural_Nmin_counts']}`",
            f"- 已覆盖 N-margin 单元：{occupied}/9",
            f"- 九宫格覆盖：`{occupancy}`",
            f"- 恢复率：{summary['recovery_rate']}",
            f"- Trec：`{summary['Trec']}`",
            f"- nTD0：`{summary['nTD0']}`",
            f"- Spearman(Nmin,nTD0)：`{summary['spearman_Nmin_nTD0']}`", "",
        ])
    missing = [(baseline_id, row) for baseline_id, row in baseline_registry.items() if not row.get("model")]
    if missing:
        lines.extend(["## 不可用的已注册候选", ""])
        for baseline_id, row in missing:
            lines.append(f"- `{baseline_id}` / `{row['task_name']}`: `{row.get('status', 'CHECKPOINT_MISSING')}`")
        lines.append("")
    lines.extend(["## 决策门禁", "",
                  "停止：进入预实验或正式实验前，必须显式选择实验一基线。", ""])
    path = ROOT / "reports/screening/EXP1_BASELINE_SCREENING.md"
    atomic_write(path, "\n".join(lines) + "\n")
    return path
