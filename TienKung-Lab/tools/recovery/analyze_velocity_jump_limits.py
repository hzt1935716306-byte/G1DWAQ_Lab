#!/usr/bin/env python3
"""Summarize fixed-horizon root-velocity-jump limits for the strongest policies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MODELS = (
    ("baseline_original_nc", "Baseline-original-NC"),
    ("dwaq_flat_new", "DWAQ-flat-new"),
    ("ours_025_final", "Ours-0.25-final"),
    ("input_context_final", "Input-context-final-L6"),
    ("stage1_symmetric_4999", "Stage1-Symmetric-4999"),
    ("stage1_flat_4999", "Stage1-Flat-4999"),
    ("slope_nosys_d_final", "Slope-NoSys-D-9999"),
    ("dwaq_slope_d_final", "DWAQ-Slope-D-9999"),
    ("slope_sys_d_final", "Slope-Sys-D-9999"),
)


def _load(input_dir: Path) -> dict[str, dict]:
    reports = {}
    for slug, name in MODELS:
        path = input_dir / f"{slug}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["planned_episode_count"] != report["completed_episode_count"]:
            raise RuntimeError(f"incomplete report: {path}")
        if report["pending_episode_count"] != 0:
            raise RuntimeError(f"pending trials: {path}")
        outcomes = {episode["outcome"] for episode in report["episodes"]}
        if not outcomes.issubset({"FALL", "SURVIVED"}):
            raise RuntimeError(f"unexpected fixed-horizon outcomes in {path}: {outcomes}")
        reports[name] = report
    hashes = {report["common_protocol"]["trial_plan_sha256"] for report in reports.values()}
    if len(hashes) != 1:
        raise RuntimeError(f"trial plans are not paired: {hashes}")
    return reports


def _groups(report: dict) -> list[dict]:
    rows = []
    magnitudes = report["common_protocol"]["velocity_magnitudes_mps"]
    for magnitude in magnitudes:
        episodes = [
            episode
            for episode in report["episodes"]
            if math.isclose(float(episode["push_magnitude_mps"]), float(magnitude))
        ]
        survived = sum(episode["outcome"] == "SURVIVED" for episode in episodes)
        rows.append(
            {
                "magnitude": float(magnitude),
                "trials": len(episodes),
                "survived": survived,
                "rate": survived / len(episodes),
            }
        )
    return rows


def _threshold(rows: list[dict], target: float) -> float | None:
    boundary = None
    for row in rows:
        if row["rate"] + 1.0e-12 < target:
            break
        boundary = row["magnitude"]
    return boundary


def _first_fall(rows: list[dict]) -> float | None:
    return next((row["magnitude"] for row in rows if row["survived"] < row["trials"]), None)


def _direction_threshold(report: dict, direction_index: int, target: float) -> float | None:
    rows = []
    for magnitude in report["common_protocol"]["velocity_magnitudes_mps"]:
        episodes = [
            episode
            for episode in report["episodes"]
            if episode["direction_index"] == direction_index
            and math.isclose(float(episode["push_magnitude_mps"]), float(magnitude))
        ]
        survived = sum(episode["outcome"] == "SURVIVED" for episode in episodes)
        rows.append({"magnitude": float(magnitude), "rate": survived / len(episodes)})
    return _threshold(rows, target)


def _format(value: float | None) -> str:
    return "—" if value is None else f"{value:g} m/s"


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _mcnemar_exact(first_only: int, second_only: int) -> float:
    total = first_only + second_only
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(min(first_only, second_only) + 1))
    return min(1.0, 2.0 * tail / (2**total))


def _paired_at_magnitude(first: dict, second: dict, magnitude: float) -> dict:
    first_map = {
        episode["trial_id"]: episode["outcome"] == "SURVIVED"
        for episode in first["episodes"]
        if math.isclose(float(episode["push_magnitude_mps"]), magnitude)
    }
    second_map = {
        episode["trial_id"]: episode["outcome"] == "SURVIVED"
        for episode in second["episodes"]
        if math.isclose(float(episode["push_magnitude_mps"]), magnitude)
    }
    if first_map.keys() != second_map.keys():
        raise RuntimeError(f"unpaired Stage1 trials at {magnitude:g} m/s")
    first_only = sum(first_map[key] and not second_map[key] for key in first_map)
    second_only = sum(second_map[key] and not first_map[key] for key in first_map)
    return {
        "first_rate": sum(first_map.values()) / len(first_map),
        "second_rate": sum(second_map.values()) / len(second_map),
        "first_only": first_only,
        "second_only": second_only,
        "p": _mcnemar_exact(first_only, second_only),
    }


def write_report(input_dir: Path, output: Path, reports: dict[str, dict]) -> None:
    grouped = {name: _groups(report) for name, report in reports.items()}
    thresholds = {
        name: {
            "p90": _threshold(rows, 0.90),
            "p100": _threshold(rows, 1.0),
            "first_fall": _first_fall(rows),
        }
        for name, rows in grouped.items()
    }
    best_90 = max(thresholds, key=lambda name: thresholds[name]["p90"] or -1.0)
    exemplar = next(iter(reports.values()))
    protocol = exemplar["common_protocol"]
    highest_magnitude = float(protocol["velocity_magnitudes_mps"][-1])
    highest_ranking = sorted(
        (
            (name, grouped[name][-1]["rate"])
            for _, name in MODELS
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    all_episodes = [episode for report in reports.values() for episode in report["episodes"]]
    max_norm_error = max(
        abs(
            math.hypot(*episode["delta_v_world_xy"])
            - float(episode["push_magnitude_mps"])
        )
        for episode in all_episodes
    )
    lines = [
        "# 强模型训练同型速度突变极限测试",
        "",
        "生成日期：2026-09-04（Asia/Shanghai）",
        "",
        "## 1. 模型身份",
        "",
        "| 模型 | checkpoint | SHA256 | task |",
        "|---|---|---|---|",
    ]
    for name, report in reports.items():
        lines.append(
            f"| {name} | `{report['checkpoint']}` | `{report['checkpoint_sha256'][:16]}…` | "
            f"`{report['native_task']}` |"
        )
    lines += [
        "",
        "## 2. 协议",
        "",
        "- 扰动接口与Stage2训练一致：直接给当前root世界系水平速度叠加一次 `Δv_xy`，不是施加外力。",
        f"- 固定模长：{', '.join(f'{value:g}' for value in protocol['velocity_magnitudes_mps'])} m/s。",
        "- 方向：世界系水平面8个等间隔方向（含±X、±Y和四个对角方向）。",
        "- 命令：8种训练域内命令；每个方向/命令组合使用两个扰动时刻，共128个trial/模长。",
        f"- 每次速度突变后固定观察 {protocol['survival_horizon_s']:g} s；即使提前恢复也不结束，只有原生termination才记FALL。",
        "- SURVIVED后同一环境继续走并准备下一次扰动；只有FALL才由环境reset，这是连续反复扰动而非每个trial独立冷启动。",
        f"- 观测噪声、随机物理参数和环境自动push关闭；{len(MODELS)}个模型使用相同trial plan。",
        "- 这里的模长是 `||Δv_xy||₂`。训练范围是每个世界系分量独立处于[-1,1] m/s，因此训练域是方形而不是固定半径圆。",
        "",
        "## 3. 保守存活极限",
        "",
        "主极限要求从0开始的所有较低扫描点都达到门槛，避免把非单调的孤立通过点当作极限。",
        "",
        "| 模型 | no-fall @90% | no-fall @100% | 首次观测到FALL的模长 |",
        "|---|---:|---:|---:|",
    ]
    for _, name in MODELS:
        item = thresholds[name]
        lines.append(
            f"| {name} | {_format(item['p90'])} | {_format(item['p100'])} | "
            f"{_format(item['first_fall'])} |"
        )
    lines += [
        "",
        f"- 当前扫描中，90%固定时域存活极限最高的是 **{best_90}**（{_format(thresholds[best_90]['p90'])}）。",
        f"- 在最高扫描点{highest_magnitude:g} m/s，{highest_ranking[0][0]}存活率最高"
        f"（{_percent(highest_ranking[0][1])}），其次是{highest_ranking[1][0]}"
        f"（{_percent(highest_ranking[1][1])}）。",
        "- `首次FALL`只表示128个有限样本中第一次出现失败，不应单独当作稳定极限；主结论看90%边界。",
        "",
        "## 4. 两个Stage1模型的相同trial直接比较",
        "",
        "差值为Stage1-Symmetric减Stage1-Flat；p值为双侧exact McNemar。",
        "",
        "| `||Δv_xy||₂` | Symmetric | Flat | 差值 | Sym-only/Flat-only | p |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    stage1_symmetric = reports["Stage1-Symmetric-4999"]
    stage1_flat = reports["Stage1-Flat-4999"]
    for magnitude in protocol["velocity_magnitudes_mps"]:
        if magnitude < 1.5:
            continue
        paired = _paired_at_magnitude(stage1_symmetric, stage1_flat, float(magnitude))
        lines.append(
            f"| {magnitude:g} m/s | {_percent(paired['first_rate'])} | "
            f"{_percent(paired['second_rate'])} | "
            f"{100.0 * (paired['first_rate'] - paired['second_rate']):+.1f} pp | "
            f"{paired['first_only']}/{paired['second_only']} | {paired['p']:.4g} |"
        )
    lines += [
        "",
        "- 两者的90%保守边界同为2.25 m/s；Flat的100%边界较高（1.75 vs 1.25 m/s），但这由Symmetric在1.5 m/s的单个失败决定。",
        "- 在3.0 m/s极端OOD点，Symmetric为73.4%，Flat为58.6%，差+14.8 pp，配对p=0.01445。",
        "- 因此不能概括成某一个全程更好：Flat的低端零失败边界更稳，Symmetric的极端扰动尾部更强。",
        "",
        "## 5. 每个速度模长的10 s存活率",
        "",
        "| `||Δv_xy||₂` | " + " | ".join(name for _, name in MODELS) + " |",
        "|---:|" + "---:|" * len(MODELS),
    ]
    for row_index, magnitude in enumerate(protocol["velocity_magnitudes_mps"]):
        cells = []
        for _, name in MODELS:
            row = grouped[name][row_index]
            cells.append(f"{_percent(row['rate'])} ({row['survived']}/{row['trials']})")
        lines.append(f"| {magnitude:g} m/s | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 6. 分方向no-fall @90%边界",
        "",
        "| 世界系方向角 | " + " | ".join(name for _, name in MODELS) + " |",
        "|---:|" + "---:|" * len(MODELS),
    ]
    for direction_index in range(int(protocol["direction_count"])):
        angle = 360.0 * direction_index / int(protocol["direction_count"])
        cells = [
            _format(_direction_threshold(reports[name], direction_index, 0.90))
            for _, name in MODELS
        ]
        lines.append(f"| {angle:g}° | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 7. 完整性审计",
        "",
        f"- 共 {len(all_episodes):,} 个trial；每模型 {exemplar['completed_episode_count']} 个，全部完成且无pending。",
        f"- {len(MODELS)}个报告的trial plan SHA256完全一致：`{protocol['trial_plan_sha256']}`。",
        f"- 记录的 `||Δv_xy||₂` 与计划模长最大误差：{max_norm_error:.3g} m/s。",
        "- 每个trial均保留方向、命令、扰动时刻、FALL/SURVIVED和恢复期间touchdown计数。",
        "",
        f"原始JSON与日志：`{input_dir}`",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    reports = _load(input_dir)
    write_report(input_dir, args.output.resolve(), reports)
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
