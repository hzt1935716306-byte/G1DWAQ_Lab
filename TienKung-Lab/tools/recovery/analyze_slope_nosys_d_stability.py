#!/usr/bin/env python3
"""Summarize flat/uphill/downhill fixed-horizon velocity-jump stability."""

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


def _load(input_dir: Path) -> dict[float, dict]:
    reports = {}
    for slope, slug, _ in SLOPES:
        path = input_dir / f"{slug}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        protocol = report["common_protocol"]
        if not math.isclose(float(protocol["slope_degrees"]), slope):
            raise RuntimeError(f"slope identity mismatch: {path}")
        if report["planned_episode_count"] != report["completed_episode_count"]:
            raise RuntimeError(f"incomplete report: {path}")
        if report["pending_episode_count"] != 0:
            raise RuntimeError(f"pending trials: {path}")
        if {episode["outcome"] for episode in report["episodes"]} - {"FALL", "SURVIVED"}:
            raise RuntimeError(f"unexpected outcome in {path}")
        reports[slope] = report
    hashes = {report["common_protocol"]["trial_plan_sha256"] for report in reports.values()}
    if len(hashes) != 1:
        raise RuntimeError(f"slope trial plans are not paired: {hashes}")
    return reports


def _rows(report: dict) -> list[dict]:
    rows = []
    for magnitude in report["common_protocol"]["velocity_magnitudes_mps"]:
        episodes = [
            episode
            for episode in report["episodes"]
            if math.isclose(float(episode["push_magnitude_mps"]), float(magnitude))
        ]
        survived = sum(episode["outcome"] == "SURVIVED" for episode in episodes)
        rows.append(
            {
                "magnitude": float(magnitude),
                "survived": survived,
                "trials": len(episodes),
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
    grouped = []
    for magnitude in report["common_protocol"]["velocity_magnitudes_mps"]:
        episodes = [
            episode
            for episode in report["episodes"]
            if episode["direction_index"] == direction_index
            and math.isclose(float(episode["push_magnitude_mps"]), float(magnitude))
        ]
        grouped.append(
            {
                "magnitude": float(magnitude),
                "rate": sum(episode["outcome"] == "SURVIVED" for episode in episodes)
                / len(episodes),
            }
        )
    return _threshold(grouped, target)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:g} m/s"


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def write_report(input_dir: Path, output: Path, reports: dict[float, dict]) -> None:
    rows = {slope: _rows(report) for slope, report in reports.items()}
    threshold = {
        slope: {
            "p90": _threshold(items, 0.90),
            "p100": _threshold(items, 1.0),
            "first_fall": _first_fall(items),
        }
        for slope, items in rows.items()
    }
    exemplar = reports[0.0]
    protocol = exemplar["common_protocol"]
    all_episodes = [episode for report in reports.values() for episode in report["episodes"]]
    nominal = {slope: rows[slope][0]["rate"] for slope in reports}
    nominal_tracking_error = {
        slope: sum(
            episode["mean_velocity_tracking_error_mps"]
            for episode in report["episodes"]
            if math.isclose(float(episode["push_magnitude_mps"]), 0.0)
            and episode.get("mean_velocity_tracking_error_mps") is not None
        )
        / sum(
            math.isclose(float(episode["push_magnitude_mps"]), 0.0)
            and episode.get("mean_velocity_tracking_error_mps") is not None
            for episode in report["episodes"]
        )
        for slope, report in reports.items()
    }
    max_norm_error = max(
        abs(math.hypot(*episode["delta_v_world_xy"]) - episode["push_magnitude_mps"])
        for episode in all_episodes
    )
    lines = [
        "# G1 Slope-NoSys-D：平地/斜坡速度突变稳定性",
        "",
        "生成日期：2026-09-04（Asia/Shanghai）",
        "",
        "## 1. 模型与训练域",
        "",
        f"- checkpoint：`{exemplar['checkpoint']}`",
        f"- SHA256：`{exemplar['checkpoint_sha256']}`",
        "- 原生任务：`g1_slope_nosys_d`，960维Actor，10000轮。",
        "- 训练地形：40%平地、30%上坡、30%下坡；坡度参数0--0.364，约等于0--20°。",
        "- 训练扰动：每10--15 s给root水平速度叠加一次每轴[-1,1] m/s随机跳变。",
        "",
        "## 2. 测试协议",
        "",
        "- 连续x对齐平面：下坡20°、下坡10°、平地、上坡10°、上坡20°。+X命令在正坡度时为上坡，在负坡度时为下坡。",
        "- 行走命令固定为 `[0.4, 0, 0]` m/s，符合该NoSys训练任务关闭转向后的命令域。",
        "- 与训练相同的瞬时root速度增量接口；固定模长0--3.0 m/s、步长0.25 m/s、8个世界系方向。",
        "- 每个坡度/模长128个trial，覆盖16个扰动时刻；突变后固定观察10 s，提前恢复不结束，原生termination记FALL。",
        "- 这是连续反复扰动测试：SURVIVED后同一环境继续走并准备下一次扰动；只有原生FALL才触发环境reset。因此各模长曲线不强制单调。",
        "- 关闭自动push、观测噪声和物理随机化；五个坡度使用同一trial plan，可逐trial配对。",
        "",
        "## 3. 无扰动行走与速度突变极限",
        "",
        "| 地形 | 0 m/s下10 s存活 | 0 m/s平均速度跟踪误差 | no-fall @90% | @100% | 首次FALL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {slope: label for slope, _, label in SLOPES}
    for slope, _, label in SLOPES:
        item = threshold[slope]
        lines.append(
            f"| {label} | {_pct(nominal[slope])} | {nominal_tracking_error[slope]:.3f} m/s | {_fmt(item['p90'])} | "
            f"{_fmt(item['p100'])} | {_fmt(item['first_fall'])} |"
        )
    lines += [
        "",
        "## 4. 每个速度模长的10 s存活率",
        "",
        "| `||Δv_xy||₂` | " + " | ".join(label for _, _, label in SLOPES) + " |",
        "|---:|" + "---:|" * len(SLOPES),
    ]
    for row_index, magnitude in enumerate(protocol["velocity_magnitudes_mps"]):
        cells = [
            f"{_pct(rows[slope][row_index]['rate'])} "
            f"({rows[slope][row_index]['survived']}/{rows[slope][row_index]['trials']})"
            for slope, _, _ in SLOPES
        ]
        lines.append(f"| {magnitude:g} m/s | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 5. 坡度 × 扰动方向的no-fall @90%边界",
        "",
        "方向角0°为+X（正坡上坡方向），180°为-X。",
        "",
        "| 方向角 | " + " | ".join(label for _, _, label in SLOPES) + " |",
        "|---:|" + "---:|" * len(SLOPES),
    ]
    for direction_index in range(8):
        angle = 45 * direction_index
        cells = [
            _fmt(_direction_threshold(reports[slope], direction_index, 0.90))
            for slope, _, _ in SLOPES
        ]
        lines.append(f"| {angle}° | " + " | ".join(cells) + " |")
    best_slope = max(threshold, key=lambda slope: threshold[slope]["p90"] or -1.0)
    worst_slope = min(threshold, key=lambda slope: threshold[slope]["p90"] or -1.0)
    lines += [
        "",
        "## 6. 结论",
        "",
        f"- 90%主极限最高地形：{labels[best_slope]}（{_fmt(threshold[best_slope]['p90'])}）。",
        f"- 90%主极限最低地形：{labels[worst_slope]}（{_fmt(threshold[worst_slope]['p90'])}）。",
        "- 0 m/s行只判断该策略是否能在对应坡度按0.4 m/s命令连续行走10 s，不代表速度跟踪误差为零。",
        "- 90%边界采用从0开始连续通过的保守定义；首次FALL只是有限样本观测，不单独作为稳定极限。",
        "",
        "## 7. 完整性审计",
        "",
        f"- 共{len(all_episodes):,}个trial；每坡度{exemplar['completed_episode_count']}个，全部完成且无pending。",
        f"- 五个坡度trial plan SHA256一致：`{protocol['trial_plan_sha256']}`。",
        f"- `||Δv_xy||₂`与计划模长最大误差：{max_norm_error:.3g} m/s。",
        "",
        f"原始JSON和日志：`{input_dir}`",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = _load(args.input_dir.resolve())
    write_report(args.input_dir.resolve(), args.output.resolve(), reports)
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
