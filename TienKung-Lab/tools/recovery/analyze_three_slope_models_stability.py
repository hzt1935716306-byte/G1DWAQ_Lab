#!/usr/bin/env python3
"""Compare three paired flat/uphill/downhill velocity-jump suites."""

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
    ("Slope-NoSys-D-9999", "slope", "g1_slope_nosys_d"),
    ("DWAQ-Slope-D-9999", "dwaq_slope", "g1_dwaq_slope_nosys_d"),
    ("Slope-Sys-D-9999", "slope_sys_d", "g1_slope_sys_d"),
)


def _load_suite(path: Path, expected_policy: str) -> dict[float, dict]:
    reports = {}
    for slope, slug, _ in SLOPES:
        report_path = path / f"{slug}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        protocol = report["common_protocol"]
        valid = (
            report["policy"] == expected_policy
            and report["planned_episode_count"] == report["completed_episode_count"] == 1664
            and report["pending_episode_count"] == 0
            and math.isclose(float(protocol["slope_degrees"]), slope)
            and not ({episode["outcome"] for episode in report["episodes"]} - {"FALL", "SURVIVED"})
        )
        if not valid:
            raise RuntimeError(f"invalid report: {report_path}")
        reports[slope] = report
    return reports


def _rows(report: dict) -> list[dict]:
    result = []
    for magnitude in report["common_protocol"]["velocity_magnitudes_mps"]:
        episodes = [
            episode for episode in report["episodes"]
            if math.isclose(float(episode["push_magnitude_mps"]), float(magnitude))
        ]
        survived = sum(episode["outcome"] == "SURVIVED" for episode in episodes)
        result.append({
            "magnitude": float(magnitude),
            "survived": survived,
            "trials": len(episodes),
            "rate": survived / len(episodes),
        })
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


def _direction_threshold(report: dict, direction: int, target: float = 0.90) -> float | None:
    rows = []
    for magnitude in report["common_protocol"]["velocity_magnitudes_mps"]:
        episodes = [
            episode for episode in report["episodes"]
            if episode["direction_index"] == direction
            and math.isclose(float(episode["push_magnitude_mps"]), float(magnitude))
        ]
        rows.append({
            "magnitude": float(magnitude),
            "rate": sum(episode["outcome"] == "SURVIVED" for episode in episodes) / len(episodes),
        })
    return _threshold(rows, target)


def _mean_tracking(report: dict) -> float:
    values = [
        episode["mean_velocity_tracking_error_mps"]
        for episode in report["episodes"]
        if math.isclose(float(episode["push_magnitude_mps"]), 0.0)
        and episode.get("mean_velocity_tracking_error_mps") is not None
    ]
    return sum(values) / len(values)


def _paired(first: dict, second: dict, magnitude: float) -> tuple[int, int, float]:
    def outcomes(report):
        return {
            episode["trial_id"]: episode["outcome"] == "SURVIVED"
            for episode in report["episodes"]
            if math.isclose(float(episode["push_magnitude_mps"]), magnitude)
        }
    first_map, second_map = outcomes(first), outcomes(second)
    if first_map.keys() != second_map.keys():
        raise RuntimeError("paired trial IDs differ")
    first_only = sum(first_map[key] and not second_map[key] for key in first_map)
    second_only = sum(second_map[key] and not first_map[key] for key in first_map)
    n = first_only + second_only
    if n == 0:
        p_value = 1.0
    else:
        k = min(first_only, second_only)
        p_value = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))
    return first_only, second_only, p_value


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:g} m/s"


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _aggregate(reports: dict[float, dict], low: float, high: float) -> float:
    episodes = [
        episode for report in reports.values() for episode in report["episodes"]
        if low <= float(episode["push_magnitude_mps"]) <= high
    ]
    return sum(episode["outcome"] == "SURVIVED" for episode in episodes) / len(episodes)


def write_report(output: Path, inputs: list[Path], suites: dict[str, dict[float, dict]]) -> None:
    model_names = [spec[0] for spec in MODELS]
    rows = {
        model: {slope: _rows(report) for slope, report in reports.items()}
        for model, reports in suites.items()
    }
    exemplar = suites[model_names[0]][0.0]
    protocol = exemplar["common_protocol"]
    hashes = {
        report["common_protocol"]["trial_plan_sha256"]
        for reports in suites.values() for report in reports.values()
    }
    if len(hashes) != 1:
        raise RuntimeError(f"unpaired trial plans: {hashes}")
    aggregate = {
        model: {
            "all": _aggregate(suites[model], 0.0, 3.0),
            "train": _aggregate(suites[model], 0.0, 1.0),
            "strong": _aggregate(suites[model], 2.25, 3.0),
        }
        for model in model_names
    }
    lines = [
        "# 三个斜坡策略：平地/斜坡速度突变稳定性对比",
        "",
        "生成日期：2026-09-04（Asia/Shanghai）",
        "",
        "## 1. 模型与统一协议",
        "",
        "| 模型 | checkpoint | SHA256 | 评测任务 |",
        "|---|---|---|---|",
    ]
    for model, _, task in MODELS:
        report = suites[model][0.0]
        lines.append(
            f"| {model} | `{report['checkpoint']}` | `{report['checkpoint_sha256'][:16]}…` | `{task}` |"
        )
    lines += [
        "",
        "- 地形为下坡20°、下坡10°、平地、上坡10°、上坡20°，统一沿+X以0.4 m/s行走。",
        "- 瞬时root水平速度跳变0--3.0 m/s、步长0.25 m/s、8个世界系方向、128 trial/坡度/模长。",
        "- 每次扰动后必须连续存活10 s；SURVIVED后继续走入下一trial，只有原生FALL才reset。",
        "- 三个模型使用同一trial plan；自动push、观测噪声和物理随机化均关闭。",
        "",
        "## 2. 主结果",
        "",
        "| 模型 | 地形 | 0 m/s存活 | 0 m/s速度误差 | no-fall @90% | @100% | 首次FALL |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in model_names:
        for slope, _, label in SLOPES:
            items = rows[model][slope]
            lines.append(
                f"| {model} | {label} | {_pct(items[0]['rate'])} | "
                f"{_mean_tracking(suites[model][slope]):.3f} m/s | {_fmt(_threshold(items, .90))} | "
                f"{_fmt(_threshold(items, 1.0))} | {_fmt(_first_fall(items))} |"
            )
    lines += [
        "",
        "## 3. 聚合存活率",
        "",
        "| 模型 | 全部0--3 m/s | 0--1 m/s | 强扰动2.25--3 m/s |",
        "|---|---:|---:|---:|",
    ]
    for model in model_names:
        lines.append(
            f"| {model} | {_pct(aggregate[model]['all'])} | {_pct(aggregate[model]['train'])} | "
            f"{_pct(aggregate[model]['strong'])} |"
        )
    lines += [
        "",
        "## 4. 结论",
        "",
        "- Sys-D在五种地形的90%连续边界都比NoSys-D高0.5 m/s。",
        "- 相比DWAQ-D，Sys-D在下坡20°边界持平，其余四种地形均低0.25 m/s。",
        f"- 强扰动2.25--3.0 m/s聚合存活率：Sys-D {_pct(aggregate[model_names[2]]['strong'])}，"
        f"NoSys-D {_pct(aggregate[model_names[0]]['strong'])}，DWAQ-D {_pct(aggregate[model_names[1]]['strong'])}。",
        f"- 全部0--3.0 m/s聚合：Sys-D {_pct(aggregate[model_names[2]]['all'])}，"
        f"位于NoSys-D的{_pct(aggregate[model_names[0]]['all'])}和DWAQ-D的{_pct(aggregate[model_names[1]]['all'])}之间。",
        "- Sys-D的特点是弱扰动稳定、零失败区间较宽；DWAQ-D在极端OOD尾部总体更强，但部分下坡基础行走/跟踪不占优。",
        "- 这些模型除是否保留固定步态系统外还存在策略结构或对称训练差异，因此该测试不能把全部差异单独归因于system。",
        "",
        "## 5. 完整存活率曲线",
        "",
    ]
    for slope, _, label in SLOPES:
        lines += [
            f"### {label}", "",
            "| `||Δv_xy||₂` | NoSys-D | DWAQ-D | Sys-D | Sys-NoSys | Sys-DWAQ |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for index, magnitude in enumerate(protocol["velocity_magnitudes_mps"]):
            rates = [rows[model][slope][index]["rate"] for model in model_names]
            lines.append(
                f"| {magnitude:g} m/s | {_pct(rates[0])} | {_pct(rates[1])} | {_pct(rates[2])} | "
                f"{100*(rates[2]-rates[0]):+.1f} pp | {100*(rates[2]-rates[1]):+.1f} pp |"
            )
        lines.append("")
    lines += [
        "## 6. Sys-D强扰动配对检验", "",
        "差值均为Sys-D减对照；p为双侧exact McNemar。", "",
        "| 对照 | 地形 | 模长 | 对照 | Sys-D | 差值 | 对照-only/Sys-only | p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for reference in model_names[:2]:
        for slope, _, label in SLOPES:
            for magnitude in (1.5, 2.0, 2.5, 3.0):
                index = protocol["velocity_magnitudes_mps"].index(magnitude)
                ref_rate = rows[reference][slope][index]["rate"]
                sys_rate = rows[model_names[2]][slope][index]["rate"]
                ref_only, sys_only, p_value = _paired(
                    suites[reference][slope], suites[model_names[2]][slope], magnitude
                )
                lines.append(
                    f"| {reference} | {label} | {magnitude:g} m/s | {_pct(ref_rate)} | {_pct(sys_rate)} | "
                    f"{100*(sys_rate-ref_rate):+.1f} pp | {ref_only}/{sys_only} | {p_value:.4g} |"
                )
    lines += ["", "## 7. 分方向no-fall @90%边界", ""]
    for slope, _, label in SLOPES:
        lines += [
            f"### {label}", "",
            "| 世界系方向角 | NoSys-D | DWAQ-D | Sys-D |",
            "|---:|---:|---:|---:|",
        ]
        for direction in range(8):
            values = [_direction_threshold(suites[model][slope], direction) for model in model_names]
            lines.append(
                f"| {45*direction}° | {_fmt(values[0])} | {_fmt(values[1])} | {_fmt(values[2])} |"
            )
        lines.append("")
    total = sum(
        report["completed_episode_count"]
        for reports in suites.values() for report in reports.values()
    )
    lines += [
        "## 8. 完整性审计", "",
        f"- 共{total:,}个trial，全部完成且无pending。",
        f"- 15份报告trial plan SHA256一致：`{next(iter(hashes))}`。",
    ]
    for model, input_path in zip(model_names, inputs, strict=True):
        lines.append(f"- {model}原始数据：`{input_path}`")
    lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_nosys", type=Path, required=True)
    parser.add_argument("--input_dwaq", type=Path, required=True)
    parser.add_argument("--input_sys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [args.input_nosys.resolve(), args.input_dwaq.resolve(), args.input_sys.resolve()]
    suites = {
        model: _load_suite(path, policy)
        for (model, policy, _), path in zip(MODELS, inputs, strict=True)
    }
    write_report(args.output.resolve(), inputs, suites)
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
