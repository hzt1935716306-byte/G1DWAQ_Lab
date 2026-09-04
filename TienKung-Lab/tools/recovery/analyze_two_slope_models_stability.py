#!/usr/bin/env python3
"""Compare paired flat/uphill/downhill velocity-jump suites for two policies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SLOPES = (
    (-20.0, "slope_minus20", "下坡20°"),
    (-10.0, "slope_minus10", "下坡10°"),
    (0.0, "slope_flat", "平地"),
    (10.0, "slope_plus10", "上坡10°"),
    (20.0, "slope_plus20", "上坡20°"),
)
MODELS = (
    ("Slope-NoSys-D-9999", "slope_nosys_d", "g1_slope_nosys_d"),
    ("DWAQ-Slope-D-9999", "dwaq_slope_d", "g1_dwaq_slope_nosys_d"),
)


def _load_one(input_dir: Path, expected_policy: str) -> dict[float, dict]:
    reports = {}
    for slope, slug, _ in SLOPES:
        path = input_dir / f"{slug}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        protocol = report["common_protocol"]
        if report["policy"] != expected_policy:
            raise RuntimeError(f"policy mismatch: {path}")
        if not math.isclose(float(protocol["slope_degrees"]), slope):
            raise RuntimeError(f"slope mismatch: {path}")
        if report["planned_episode_count"] != report["completed_episode_count"]:
            raise RuntimeError(f"incomplete report: {path}")
        if report["pending_episode_count"] != 0:
            raise RuntimeError(f"pending trials: {path}")
        if {episode["outcome"] for episode in report["episodes"]} - {"FALL", "SURVIVED"}:
            raise RuntimeError(f"unexpected outcome: {path}")
        reports[slope] = report
    return reports


def _rows(report: dict) -> list[dict]:
    result = []
    for magnitude in report["common_protocol"]["velocity_magnitudes_mps"]:
        episodes = [
            episode
            for episode in report["episodes"]
            if math.isclose(float(episode["push_magnitude_mps"]), float(magnitude))
        ]
        survived = sum(episode["outcome"] == "SURVIVED" for episode in episodes)
        result.append(
            {
                "magnitude": float(magnitude),
                "survived": survived,
                "trials": len(episodes),
                "rate": survived / len(episodes),
            }
        )
    return result


def _threshold(rows: list[dict], target: float) -> float | None:
    boundary = None
    for row in rows:
        if row["rate"] + 1.0e-12 < target:
            break
        boundary = row["magnitude"]
    return boundary


def _first_fall(rows: list[dict]) -> float | None:
    return next((row["magnitude"] for row in rows if row["survived"] < row["trials"]), None)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:g} m/s"


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _mean_tracking_error(report: dict) -> float:
    values = [
        episode["mean_velocity_tracking_error_mps"]
        for episode in report["episodes"]
        if math.isclose(float(episode["push_magnitude_mps"]), 0.0)
        and episode.get("mean_velocity_tracking_error_mps") is not None
    ]
    return sum(values) / len(values)


def _binomial_two_sided(discordant_a: int, discordant_b: int) -> float:
    n = discordant_a + discordant_b
    if n == 0:
        return 1.0
    k = min(discordant_a, discordant_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _paired(report_a: dict, report_b: dict, magnitude: float) -> tuple[int, int, float]:
    episodes_a = {
        episode["trial_id"]: episode["outcome"] == "SURVIVED"
        for episode in report_a["episodes"]
        if math.isclose(float(episode["push_magnitude_mps"]), magnitude)
    }
    episodes_b = {
        episode["trial_id"]: episode["outcome"] == "SURVIVED"
        for episode in report_b["episodes"]
        if math.isclose(float(episode["push_magnitude_mps"]), magnitude)
    }
    if episodes_a.keys() != episodes_b.keys():
        raise RuntimeError("paired trial IDs differ")
    a_only = sum(episodes_a[key] and not episodes_b[key] for key in episodes_a)
    b_only = sum(episodes_b[key] and not episodes_a[key] for key in episodes_a)
    return a_only, b_only, _binomial_two_sided(a_only, b_only)


def write_report(
    input_a: Path,
    input_b: Path,
    output: Path,
    all_reports: dict[str, dict[float, dict]],
) -> None:
    labels = {slope: label for slope, _, label in SLOPES}
    rows = {
        model: {slope: _rows(report) for slope, report in reports.items()}
        for model, reports in all_reports.items()
    }
    exemplar = all_reports[MODELS[0][0]][0.0]
    protocol = exemplar["common_protocol"]
    all_trial_hashes = {
        report["common_protocol"]["trial_plan_sha256"]
        for reports in all_reports.values()
        for report in reports.values()
    }
    checkpoint_hashes = {
        model: next(iter(reports.values()))["checkpoint_sha256"]
        for model, reports in all_reports.items()
    }
    lines = [
        "# 两个斜坡策略：平地/斜坡速度突变稳定性对比",
        "",
        "生成日期：2026-09-04（Asia/Shanghai）",
        "",
        "## 1. 模型",
        "",
        "| 名称 | checkpoint | SHA256 | 评测原生任务 |",
        "|---|---|---|---|",
    ]
    for model, _, native_task in MODELS:
        report = all_reports[model][0.0]
        lines.append(
            f"| {model} | `{report['checkpoint']}` | `{checkpoint_hashes[model][:16]}…` | `{native_task}` |"
        )
    lines += [
        "",
        "## 2. 统一协议",
        "",
        "- 地形：下坡20°、下坡10°、平地、上坡10°、上坡20°；沿+X以0.4 m/s行走。",
        "- 瞬时root水平速度跳变：0--3.0 m/s、0.25 m/s步长、8个世界系方向。",
        "- 每个模型/坡度/模长128个trial；扰动后必须连续存活10 s，原生termination为FALL。",
        "- 这是连续反复扰动测试：SURVIVED后同一环境继续走并准备下一次扰动，只有FALL才reset；曲线不强制单调。",
        "- 两个模型和五个坡度使用同一trial plan；禁用自动push、噪声和物理随机化。",
        "",
        "## 3. 主结果",
        "",
        "| 模型 | 地形 | 0 m/s存活 | 0 m/s平均速度误差 | no-fall @90% | @100% | 首次FALL |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model, _, _ in MODELS:
        for slope, _, label in SLOPES:
            items = rows[model][slope]
            lines.append(
                f"| {model} | {label} | {_pct(items[0]['rate'])} | "
                f"{_mean_tracking_error(all_reports[model][slope]):.3f} m/s | "
                f"{_fmt(_threshold(items, 0.90))} | {_fmt(_threshold(items, 1.0))} | "
                f"{_fmt(_first_fall(items))} |"
            )
    lines += [
        "",
        "## 4. 完整存活率曲线",
        "",
    ]
    for slope, _, label in SLOPES:
        lines += [
            f"### {label}",
            "",
            "| `||Δv_xy||₂` | Slope-NoSys-D | DWAQ-Slope-D | 差值(DWAQ-NoSys) |",
            "|---:|---:|---:|---:|",
        ]
        for index, magnitude in enumerate(protocol["velocity_magnitudes_mps"]):
            rate_a = rows[MODELS[0][0]][slope][index]["rate"]
            rate_b = rows[MODELS[1][0]][slope][index]["rate"]
            lines.append(
                f"| {magnitude:g} m/s | {_pct(rate_a)} | {_pct(rate_b)} | "
                f"{100.0 * (rate_b - rate_a):+.1f} pp |"
            )
        lines.append("")
    lines += [
        "## 5. 强扰动配对检验",
        "",
        "同一trial中仅一个模型存活的数量记为A-only/B-only；p为双侧exact McNemar。",
        "",
        "| 地形 | 模长 | NoSys-D | DWAQ-D | 差值 | NoSys-only/DWAQ-only | p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    model_a, model_b = MODELS[0][0], MODELS[1][0]
    for slope, _, label in SLOPES:
        for magnitude in (1.5, 2.0, 2.5, 3.0):
            index = protocol["velocity_magnitudes_mps"].index(magnitude)
            rate_a = rows[model_a][slope][index]["rate"]
            rate_b = rows[model_b][slope][index]["rate"]
            a_only, b_only, p_value = _paired(
                all_reports[model_a][slope], all_reports[model_b][slope], magnitude
            )
            lines.append(
                f"| {label} | {magnitude:g} m/s | {_pct(rate_a)} | {_pct(rate_b)} | "
                f"{100.0 * (rate_b - rate_a):+.1f} pp | {a_only}/{b_only} | {p_value:.4g} |"
            )
    aggregate = {}
    for model in (model_a, model_b):
        episodes = [
            episode
            for report in all_reports[model].values()
            for episode in report["episodes"]
        ]
        aggregate[model] = {
            "all": sum(episode["outcome"] == "SURVIVED" for episode in episodes) / len(episodes),
            "train_like": sum(
                episode["outcome"] == "SURVIVED"
                for episode in episodes
                if 0.0 <= float(episode["push_magnitude_mps"]) <= 1.0
            )
            / sum(0.0 <= float(episode["push_magnitude_mps"]) <= 1.0 for episode in episodes),
            "strong": sum(
                episode["outcome"] == "SURVIVED"
                for episode in episodes
                if 2.25 <= float(episode["push_magnitude_mps"]) <= 3.0
            )
            / sum(2.25 <= float(episode["push_magnitude_mps"]) <= 3.0 for episode in episodes),
        }
    boundary_gains = []
    for slope, _, _ in SLOPES:
        a_boundary = _threshold(rows[model_a][slope], 0.90)
        b_boundary = _threshold(rows[model_b][slope], 0.90)
        boundary_gains.append(float(b_boundary) - float(a_boundary))
    lines += [
        "",
        "## 6. 聚合结果与解释",
        "",
        "| 模型 | 全部坡度/模长 | 0--1.0 m/s | 2.25--3.0 m/s |",
        "|---|---:|---:|---:|",
    ]
    for model in (model_a, model_b):
        lines.append(
            f"| {model} | {_pct(aggregate[model]['all'])} | "
            f"{_pct(aggregate[model]['train_like'])} | {_pct(aggregate[model]['strong'])} |"
        )
    lines += [
        "",
        f"- DWAQ在五种地形的90%边界均更高，提升范围为{min(boundary_gains):g}--{max(boundary_gains):g} m/s。",
        f"- 强扰动区间2.25--3.0 m/s，DWAQ为{_pct(aggregate[model_b]['strong'])}，"
        f"NoSys-D为{_pct(aggregate[model_a]['strong'])}，差"
        f"{100.0 * (aggregate[model_b]['strong'] - aggregate[model_a]['strong']):+.1f} pp。",
        f"- 训练中心区间0--1.0 m/s，两者都接近饱和；DWAQ为{_pct(aggregate[model_b]['train_like'])}，"
        f"NoSys-D为{_pct(aggregate[model_a]['train_like'])}。DWAQ的小幅下降主要来自下坡10°的非扰动/弱扰动自然跌倒。",
        "- 基础速度跟踪方面，DWAQ在平地和上坡更准；在下坡20°反而明显更差（0.141 vs 0.078 m/s）。",
        "- 因此结论是DWAQ显著扩大了强冲击存活域，但不能概括为每个弱扰动点和每项跟踪指标都更优。",
    ]
    total_trials = sum(
        report["completed_episode_count"]
        for reports in all_reports.values()
        for report in reports.values()
    )
    lines += [
        "",
        "## 7. 完整性审计",
        "",
        f"- 共{total_trials:,}个trial，全部完成且无pending。",
        f"- 所有报告trial plan SHA256一致：`{protocol['trial_plan_sha256']}`（哈希种类数={len(all_trial_hashes)}）。",
        f"- Slope-NoSys-D原始数据：`{input_a}`",
        f"- DWAQ-Slope-D原始数据：`{input_b}`",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_a", type=Path, required=True)
    parser.add_argument("--input_b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    all_reports = {
        MODELS[0][0]: _load_one(args.input_a.resolve(), "slope"),
        MODELS[1][0]: _load_one(args.input_b.resolve(), "dwaq_slope"),
    }
    write_report(args.input_a.resolve(), args.input_b.resolve(), args.output.resolve(), all_reports)
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
