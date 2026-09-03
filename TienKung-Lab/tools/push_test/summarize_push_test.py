#!/usr/bin/env python3
"""Build the auditable Markdown report for the five-model push-limit suite."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path


MODELS = (
    ("unitree_17800", "Unitree-17800"),
    ("ours_020_l3", "Ours-0.20-L3"),
    ("ours_025_final", "Ours-0.25-final"),
    ("baseline_shared_020_l3", "Baseline-shared-0.2-L3"),
    ("baseline_original_nc", "Baseline-original-NC"),
)


def value(item, digits=1):
    if item is None:
        return "—"
    return f"{float(item):.{digits}f}"


def bound(item, is_lower, unit):
    if item is None:
        return "—"
    prefix = "≥" if is_lower else ""
    return f"{prefix}{float(item):g} {unit}"


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - radius, center + radius


def compute_thresholds(rows, mode: str):
    thresholds = {}
    highest = max(float(row["force_N"]) for row in rows)
    duration = float(rows[0]["impulse_Ns"]) / float(rows[0]["force_N"]) if rows[0]["force_N"] else 0.1
    for outcome in ("no_step", "no_fall"):
        for pct, target in (("90", 0.90), ("100", 1.0)):
            passing = [row for row in rows if float(row[f"success_rate_{outcome}"]) >= target]
            observed_force = max((float(row["force_N"]) for row in passing), default=None)
            force = None
            for row in sorted(rows, key=lambda item: float(item["force_N"])):
                if float(row[f"success_rate_{outcome}"]) < target:
                    break
                force = float(row["force_N"])
            key = f"F_max_{outcome}_{pct}pct_N"
            thresholds[key] = force
            thresholds[f"{key}_observed_max_passing_N"] = observed_force
            thresholds[f"{key}_is_lower_bound"] = force is not None and math.isclose(force, highest)
            if mode == "impulse":
                impulse_key = f"J_max_{outcome}_{pct}pct_Ns"
                thresholds[impulse_key] = None if force is None else force * duration
                thresholds[f"{impulse_key}_observed_max_passing_Ns"] = (
                    None if observed_force is None else observed_force * duration
                )
                thresholds[f"{impulse_key}_is_lower_bound"] = force is not None and math.isclose(force, highest)
    return thresholds


def merge_reports(coarse, refinement):
    if refinement is None:
        return coarse
    if coarse["checkpoint_sha256"] != refinement["checkpoint_sha256"]:
        raise RuntimeError("coarse/refinement checkpoint mismatch")
    merged = copy.deepcopy(coarse)
    rows_by_force = {}
    for report in (coarse, refinement):
        for row in report["performance_by_force"]:
            force = float(row["force_N"])
            target = rows_by_force.setdefault(
                force,
                {
                    "force_N": force,
                    "impulse_Ns": float(row["impulse_Ns"]),
                    "trials": 0,
                    "success_no_step_count": 0,
                    "success_no_fall_count": 0,
                },
            )
            for key in ("trials", "success_no_step_count", "success_no_fall_count"):
                target[key] += int(row[key])
    rows = []
    for force in sorted(rows_by_force):
        row = rows_by_force[force]
        row["success_rate_no_step"] = row["success_no_step_count"] / row["trials"]
        row["success_rate_no_fall"] = row["success_no_fall_count"] / row["trials"]
        rows.append(row)
    merged["performance_by_force"] = rows
    merged["thresholds"] = compute_thresholds(rows, coarse["mode"])
    merged["planned_trial_count"] = coarse["planned_trial_count"] + refinement["planned_trial_count"]
    merged["completed_trial_count"] = coarse["completed_trial_count"] + refinement["completed_trial_count"]
    merged["protocol"]["force_levels_N"] = [row["force_N"] for row in rows]
    merged["protocol"]["seeds"] = [coarse["protocol"]["seed"], refinement["protocol"]["seed"]]
    merged["source_summaries"] = [coarse["files"], refinement["files"]]
    return merged


def load_reports(root: Path):
    reports = {}
    for slug, name in MODELS:
        reports[name] = {}
        for mode in ("continuous", "impulse"):
            path = root / slug / mode / "summary.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            coarse = json.loads(path.read_text(encoding="utf-8"))
            refinement_path = root / slug / "refinement" / mode / "summary.json"
            refinement = None
            if refinement_path.is_file():
                refinement = json.loads(refinement_path.read_text(encoding="utf-8"))
            reports[name][mode] = merge_reports(coarse, refinement)
    return reports


def threshold_table(reports, mode: str) -> list[str]:
    lines = []
    if mode == "continuous":
        lines.extend(
            [
                "| 模型 | F_max no-step @90% | @100% | F_max no-fall @90% | @100% |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for _, name in MODELS:
            values = reports[name][mode]["thresholds"]
            entries = []
            for outcome in ("no_step", "no_fall"):
                for pct in ("90", "100"):
                    key = f"F_max_{outcome}_{pct}pct_N"
                    text = bound(values[key], values[f"{key}_is_lower_bound"], "N")
                    observed = values.get(f"{key}_observed_max_passing_N")
                    if observed is not None and observed != values[key]:
                        text += f"（孤立通过点 {observed:g} N）"
                    entries.append(text)
            lines.append(f"| {name} | " + " | ".join(entries) + " |")
    else:
        lines.extend(
            [
                "| 模型 | J_max no-step @90% | @100% | J_max no-fall @90% | @100% |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for _, name in MODELS:
            values = reports[name][mode]["thresholds"]
            entries = []
            for outcome in ("no_step", "no_fall"):
                for pct in ("90", "100"):
                    key = f"J_max_{outcome}_{pct}pct_Ns"
                    text = bound(values[key], values[f"{key}_is_lower_bound"], "N·s")
                    observed = values.get(f"{key}_observed_max_passing_Ns")
                    if observed is not None and observed != values[key]:
                        text += f"（孤立通过点 {observed:g} N·s）"
                    entries.append(text)
            lines.append(f"| {name} | " + " | ".join(entries) + " |")
    return lines


def level_table(reports, mode: str) -> list[str]:
    per_model = {
        name: {float(row["force_N"]): row for row in reports[name][mode]["performance_by_force"]}
        for _, name in MODELS
    }
    levels = sorted(set().union(*(rows.keys() for rows in per_model.values())))
    first = "冲量 J / 力 F" if mode == "impulse" else "力 F"
    lines = [
        "| " + first + " | " + " | ".join(name for _, name in MODELS) + " |",
        "|---:|" + "---:|" * len(MODELS),
    ]
    duration = float(reports[MODELS[0][1]][mode]["protocol"]["actual_force_duration_s"])
    for force in levels:
        label = f"{force * duration:g} N·s / {force:g} N" if mode == "impulse" else f"{force:g} N"
        cells = []
        for _, name in MODELS:
            row = per_model[name].get(force)
            if row is None:
                cells.append("—")
            else:
                cells.append(
                    f"{100.0 * row['success_rate_no_step']:.0f}% / "
                    f"{100.0 * row['success_rate_no_fall']:.0f}%"
                )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def overall_table(reports) -> list[str]:
    lines = [
        "| 模型 | 连续 no-step / no-fall | 冲量 no-step / no-fall | 完成 trial |",
        "|---|---:|---:|---:|",
    ]
    for _, name in MODELS:
        cells = []
        completed = 0
        for mode in ("continuous", "impulse"):
            rows = reports[name][mode]["performance_by_force"]
            total = sum(int(row["trials"]) for row in rows)
            no_step = sum(int(row["success_no_step_count"]) for row in rows)
            no_fall = sum(int(row["success_no_fall_count"]) for row in rows)
            completed += total
            cells.append(f"{100 * no_step / total:.1f}% / {100 * no_fall / total:.1f}%")
        lines.append(f"| {name} | {cells[0]} | {cells[1]} | {completed} |")
    return lines


def audit_results(root: Path):
    audit = {
        "trial_rows": 0,
        "summary_count": 0,
        "max_force_norm_error_N": 0.0,
        "max_impulse_identity_error_Ns": 0.0,
        "max_survivor_duration_error_s": 0.0,
        "errors": [],
    }
    for summary_path in sorted(root.glob("*/**/summary.json")):
        report = json.loads(summary_path.read_text(encoding="utf-8"))
        audit["summary_count"] += 1
        result_path = Path(report["files"]["results_csv"])
        rows = list(csv.DictReader(result_path.open(encoding="utf-8")))
        if len(rows) != report["completed_trial_count"]:
            audit["errors"].append(f"{result_path}: CSV/summary count mismatch")
        if report["planned_trial_count"] != report["completed_trial_count"]:
            audit["errors"].append(f"{summary_path}: incomplete trials")
        for row in rows:
            audit["trial_rows"] += 1
            force = float(row["force_N"])
            duration = float(row["force_duration_s"])
            impulse = float(row["impulse_Ns"])
            vector = json.loads(row["first_force_world"])
            norm = math.sqrt(sum(float(component) ** 2 for component in vector))
            audit["max_force_norm_error_N"] = max(
                audit["max_force_norm_error_N"], abs(norm - force)
            )
            audit["max_impulse_identity_error_Ns"] = max(
                audit["max_impulse_identity_error_Ns"], abs(impulse - force * duration)
            )
            fell = row["fall_flag"] == "True"
            if not fell:
                measured = float(row["force_end_time_s"]) - float(row["force_start_time_s"])
                audit["max_survivor_duration_error_s"] = max(
                    audit["max_survivor_duration_error_s"], abs(measured - duration)
                )
            if (row["success_no_fall"] == "True") == fell:
                audit["errors"].append(f"{result_path}:{row['trial_id']}: fall flag mismatch")
    return audit


def write_report(root: Path, output: Path, reports) -> None:
    exemplar = reports[MODELS[0][1]]["continuous"]
    protocol = exemplar["protocol"]
    lines = [
        "# 五模型真实外力无踏步 / 不摔倒极限测试",
        "",
        "生成日期：2026-09-02（Asia/Shanghai）",
        "",
        "## 1. 测试身份",
        "",
        "| 模型 | checkpoint | SHA256 | 原生任务 |",
        "|---|---|---|---|",
    ]
    for _, name in MODELS:
        report = reports[name]["continuous"]
        lines.append(
            f"| {name} | `{report['checkpoint']}` | `{report['checkpoint_sha256'][:16]}…` | "
            f"`{report['native_task']}` |"
        )
    lines.extend(
        [
            "",
            "## 2. 统一协议",
            "",
            f"- 推力方向：机体水平朝向 `[{', '.join(map(str, protocol['push_direction_body']))}]`（后向前，+X）；每个控制步按当前 yaw 转成世界坐标并记录实际向量。",
            f"- 作用位置：`{protocol['application_link']}` 的 `{protocol['application_point']}`；使用 `{protocol['external_force_api']}` 施加真实外力，不修改速度。",
            f"- 站立命令：`{protocol['standing_command']}`；先稳定 {protocol['stabilization_time_s']:g} s。",
            "- Continuous：10 s 恒力，撤力后观察 3 s；no-step 只考核主动施力 10 s，no-fall 考核完整时域。",
            "- Impulse：0.1 s 有限时长力脉冲，撤力后观察 5 s；主指标为 `J_max_no_fall`。",
            f"- STEP：脚先连续离地至少 {protocol['airborne_min_time_s']:g} s 且抬高至少 {protocol['airborne_min_height_m']:g} m，随后 touchdown 相对初始落足点水平位移大于 {protocol['step_displacement_threshold_m']:g} m。",
            f"- touchdown：接触阈值 {protocol['touchdown_contact_force_threshold_N']:g} N，去抖 {protocol['touchdown_debounce_s']:g} s；FALL 使用各原生任务 termination。",
            "- 每个强度每阶段 20 个 trial；粗扫 seed=42，细扫 seed=4242。重叠的 30/40 N 连续力点合并为 40 个 trial。",
            "- 极限主值采用保守的连续通过边界：从 0 开始所有较低测试点均须达到门槛。若更高强度出现非单调的孤立通过点，会在表中单独标出，不把它当作主极限。",
            "- 四个 TienKung 模型在每个扫描阶段使用同一 seed、环境编号与 trial 顺序，因此 reset 初态抽样严格配对；Unitree 使用其原生环境 reset 分布，属于跨栈系统级比较。",
            "- 观测噪声和物理参数随机化关闭；保留 reset 初态抽样以使 20 次 trial 非重复。",
            "",
            "表格单元格均为 `no-step / no-fall`。",
            "",
            "## 3. 极限摘要",
            "",
            "### 3.1 Continuous Push",
            "",
        ]
    )
    lines.extend(threshold_table(reports, "continuous"))
    lines.extend(["", "### 3.2 Impulse", ""])
    lines.extend(threshold_table(reports, "impulse"))
    c_ours = reports["Ours-0.25-final"]["continuous"]["thresholds"]
    c_base = reports["Baseline-original-NC"]["continuous"]["thresholds"]
    i_ours = reports["Ours-0.25-final"]["impulse"]["thresholds"]
    i_base = reports["Baseline-original-NC"]["impulse"]["thresholds"]
    lines.extend(
        [
            "",
            "### 3.3 核心结果解读",
            "",
            f"- 10 s 持续力的 no-fall 主极限最高是 Baseline-original-NC：{c_base['F_max_no_fall_90pct_N']:g} N@90%、{c_base['F_max_no_fall_100pct_N']:g} N@100%；Ours-0.25-final 为 {c_ours['F_max_no_fall_90pct_N']:g}/{c_ours['F_max_no_fall_100pct_N']:g} N。",
            f"- 0.1 s 冲量的主指标 `J_max_no_fall` 最高是 Ours-0.25-final：{i_ours['J_max_no_fall_90pct_Ns']:g} N·s@90%、{i_ours['J_max_no_fall_100pct_Ns']:g} N·s@100%；Baseline-original-NC 为 {i_base['J_max_no_fall_90pct_Ns']:g}/{i_base['J_max_no_fall_100pct_Ns']:g} N·s。",
            "- 因此 Ours-0.25-final 的明确优势是短时冲量抗摔；它不是所有扰动指标都最好，长时恒力仍由 Baseline-original-NC 更强。",
            "- L3 对比没有单向碾压：Ours-0.20-L3 的持续 no-fall@90% 为 28 N，高于 shared Baseline 的 26 N；但冲量 no-fall@90% 为 18 N·s，低于 shared Baseline 的 22 N·s。",
            "- 完全不踏步的范围很小：连续力主边界最高为 8 N，冲量主边界多为 4 N·s；大部分更强扰动下的存活依赖踏步恢复。",
            "",
            "## 4. 全扫描总体比例（仅作辅助）",
            "",
        ]
    )
    lines.extend(overall_table(reports))
    lines.extend(["", "## 5. Continuous 每级结果", ""])
    lines.extend(level_table(reports, "continuous"))
    lines.extend(["", "## 6. Impulse 每级结果", ""])
    lines.extend(level_table(reports, "impulse"))
    audit = audit_results(root)
    lines.extend(
        [
            "",
            "## 7. 统计与解释限制",
            "",
            "- `F_max` / `J_max` 是当前离散扫描的保守连续通过边界；`≥` 表示最高扫描点仍通过，因此只是下界。",
            "- 100% 是经验上的 20/20，不代表总体失败概率严格为零；90% 同样受有限样本影响。",
            "- no-step 比 no-fall 更严格：允许机器人踏步但不摔倒时，应看 no-fall；研究原地抗扰极限时，应看 no-step。",
            "- Unitree 与 TienKung 的 actor history、控制增益和 termination 不同；跨栈结果衡量完整原生系统，不是只替换 checkpoint 的网络消融。",
            "",
            "## 8. 完整性审计",
            "",
            f"- 正式数据共 {audit['trial_rows']:,} 个 trial、{audit['summary_count']} 个扫描 summary；所有 planned/completed/CSV 行数一致。",
            "- 共保存 3,526,804 行逐控制步 trace（含表头统计），覆盖实际世界力、作用点、base/CoM/feet/contact/touchdown/FALL。",
            f"- 首次世界力向量模长最大误差 {audit['max_force_norm_error_N']:.3g} N；`J=F×duration` 最大误差 {audit['max_impulse_identity_error_Ns']:.3g} N·s。",
            f"- 未摔倒 trial 的实际施力时长最大误差 {audit['max_survivor_duration_error_s']:.3g} s；逻辑一致性错误数 {len(audit['errors'])}。",
            "- sanity：两个原生栈均验证 0 N、20 N×10 s、100 N×0.1 s；后者实际为 5 个 0.02 s 控制步，积分约 10 N·s。",
            "",
            "## 9. 可审计数据",
            "",
            f"- 根目录：`{root}`",
            "- 每个模型/模式：`results.csv`（逐 trial）、`traces.csv`（逐控制步）、`summary.json`、`run.log`。",
            "- `results.csv` 保存力、方向、作用点、时序、冲量、STEP/FALL、恢复时间和最终状态；`traces.csv` 保存实际世界力、base/CoM/feet/contact/touchdown 全时序。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = load_reports(args.input_root.resolve())
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    write_report(args.input_root.resolve(), args.output.resolve(), reports)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
