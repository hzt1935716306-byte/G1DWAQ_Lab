#!/usr/bin/env python3
"""Summarize the matched L2-boundary comprehensive policy sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


MODELS = (
    {
        "slug": "input_context_l2",
        "name": "Input-context（L2边界）",
        "transition": 3600,
        "checkpoint_curriculum_iteration": 3500,
        "identity": "963维Actor；真实certificate context；shared events scale=0.5",
    },
    {
        "slug": "baseline_original_l2",
        "name": "Baseline-original（L2边界）",
        "transition": 2723,
        "checkpoint_curriculum_iteration": 2700,
        "identity": "960维Actor；原始locomotion reward",
    },
    {
        "slug": "baseline_shared020_l2",
        "name": "Baseline-shared-0.2（L2边界）",
        "transition": 1425,
        "checkpoint_curriculum_iteration": 1400,
        "identity": "960维Actor；三个shared events，scale=0.2",
    },
    {
        "slug": "ours_shared_cert020_l2",
        "name": "Ours-shared+cert-0.2（L2边界）",
        "transition": 1020,
        "checkpoint_curriculum_iteration": 1000,
        "identity": "960维Actor；shared events + certificate reward，scale=0.2",
    },
    {
        "slug": "ours_cert020_l2",
        "name": "Ours-cert-only-0.20（L2边界）",
        "transition": 2531,
        "checkpoint_curriculum_iteration": 2500,
        "identity": "960维Actor；certificate-only，scale=0.20",
    },
    {
        "slug": "ours_cert050_l2",
        "name": "Ours-cert-only-0.50（L2边界）",
        "transition": 3600,
        "checkpoint_curriculum_iteration": 3500,
        "identity": "960维Actor；certificate-only，scale=0.50",
    },
    {
        "slug": "baseline_original_nc",
        "name": "Baseline-original（无课程最终）",
        "transition": None,
        "checkpoint_curriculum_iteration": None,
        "identity": "960维Actor；无课程；固定L6；原始locomotion reward",
    },
    {
        "slug": "ours_cert020_nc",
        "name": "Ours-cert-only-0.20（无课程最终）",
        "transition": None,
        "checkpoint_curriculum_iteration": None,
        "identity": "960维Actor；无课程；固定L6；certificate-only，scale=0.20",
    },
    {
        "slug": "dwaq_new",
        "name": "DWAQ-flat-new",
        "transition": None,
        "checkpoint_curriculum_iteration": None,
        "identity": "100维DWAQ Actor；新纯平地策略",
    },
    {
        "slug": "dwaq_old",
        "name": "DWAQ-old",
        "transition": None,
        "checkpoint_curriculum_iteration": None,
        "identity": "100维DWAQ Actor；历史参考策略",
    },
)
SEEDS = (42, 123, 2026)


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _pvalue(value: float) -> str:
    return "<1e-300" if value == 0.0 else f"{value:.4g}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _performance(episodes: list[dict]) -> dict:
    total = len(episodes)
    counts = {name: sum(row["outcome"] == name for row in episodes) for name in ("SUCCESS", "TIMEOUT", "FALL")}
    steps = [int(row["practical_enter_step"]) for row in episodes if row["outcome"] == "SUCCESS"]
    times = [float(row["recovery_time_s"]) for row in episodes if row["outcome"] == "SUCCESS"]
    return {
        "count": total,
        "counts": counts,
        "P5": counts["SUCCESS"] / total,
        "timeout": counts["TIMEOUT"] / total,
        "fall": counts["FALL"] / total,
        "step_mean": float(np.mean(steps)) if steps else math.nan,
        "step_median": float(np.median(steps)) if steps else math.nan,
        "time_mean": float(np.mean(times)) if times else math.nan,
    }


def _load(input_dir: Path) -> dict:
    reports = {}
    for model in MODELS:
        model_reports = []
        for seed in SEEDS:
            path = input_dir / f"{model['slug']}_seed{seed}.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["completed_episode_count"] != 1536 or report["pending_episode_count"] != 0:
                raise RuntimeError(f"incomplete report: {path}")
            model_reports.append(report)
        reports[model["slug"]] = model_reports
    for seed_index, seed in enumerate(SEEDS):
        hashes = {
            reports[model["slug"]][seed_index]["common_protocol"]["trial_plan_sha256"]
            for model in MODELS
        }
        if len(hashes) != 1:
            raise RuntimeError(f"trial plan mismatch for seed {seed}: {hashes}")
    return reports


def _episodes(reports: dict, slug: str) -> list[dict]:
    result = []
    for report in reports[slug]:
        seed = int(report["common_protocol"]["seed"])
        for row in report["episodes"]:
            item = dict(row)
            item["seed"] = seed
            result.append(item)
    return result


def _paired(current: list[dict], other: list[dict]) -> dict:
    a = {(row["seed"], row["trial_id"]): row for row in current}
    b = {(row["seed"], row["trial_id"]): row for row in other}
    if a.keys() != b.keys():
        raise RuntimeError("paired trial identities do not match")
    a_only = b_only = 0
    for key in a:
        a_success = a[key]["outcome"] == "SUCCESS"
        b_success = b[key]["outcome"] == "SUCCESS"
        a_only += int(a_success and not b_success)
        b_only += int(b_success and not a_success)
    discordant = a_only + b_only
    p = float(binomtest(a_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {"a_only": a_only, "b_only": b_only, "p": p}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = _load(args.input_dir.resolve())
    all_episodes = {model["slug"]: _episodes(reports, model["slug"]) for model in MODELS}
    overall = {slug: _performance(rows) for slug, rows in all_episodes.items()}
    per_level = {
        slug: {level: _performance([row for row in rows if int(row["level"]) == level]) for level in range(1, 7)}
        for slug, rows in all_episodes.items()
    }

    lines = [
        "# L2边界 checkpoint 与无课程/DWAQ：统一六等级恢复测试",
        "",
        "## 1. 对比范围",
        "",
        "有课程模型取各自训练中仍处于L2的最后一个已保存checkpoint；另外加入两个无课程最终模型和新旧DWAQ作为参考。测试协议与 `ALL_TRAINED_MODEL_COMPREHENSIVE_COMPARISON.md` 相同：seed=42/123/2026，L1--L6，每级每seed 256 episode，每模型共4608 episode。",
        "",
        "0.15和0.25只有独立最终权重，缺少L2中间checkpoint，因此未作为L2边界模型纳入。无课程模型和DWAQ没有课程阶段，结果仅作为最终能力参考。",
        "",
        "## 2. checkpoint身份",
        "",
        "| 模型 | checkpoint | Actor输入 | L2→L3迭代 | 距升级轮数 | SHA256 | 训练身份 |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for model in MODELS:
        report = reports[model["slug"]][0]
        checkpoint = Path(report["checkpoint"])
        actor_dim = int(report["actor_observation_shape"][-1])
        if model["transition"] is None:
            transition = "--"
            remaining = "--"
        else:
            transition = str(model["transition"])
            remaining = str(model["transition"] - model["checkpoint_curriculum_iteration"])
        lines.append(
            f"| {model['name']} | `{checkpoint}` | {actor_dim} | {transition} | {remaining} | `{_sha256(checkpoint)[:12]}…` | {model['identity']} |"
        )

    lines += [
        "",
        "## 3. 总体结果",
        "",
        "| 模型 | SUCCESS/TIMEOUT/FALL | P5 | timeout | fall | 成功步数mean/median | 成功时间mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in sorted(MODELS, key=lambda item: overall[item["slug"]]["P5"], reverse=True):
        metric = overall[model["slug"]]
        counts = metric["counts"]
        lines.append(
            f"| {model['name']} | {counts['SUCCESS']}/{counts['TIMEOUT']}/{counts['FALL']} | {_percent(metric['P5'])} | {_percent(metric['timeout'])} | {_percent(metric['fall'])} | {metric['step_mean']:.3f}/{metric['step_median']:.1f} | {metric['time_mean']:.3f} s |"
        )

    lines += [
        "",
        "## 4. 逐等级P5",
        "",
        "| Level | " + " | ".join(model["name"] for model in MODELS) + " |",
        "|---:|" + "---:|" * len(MODELS),
    ]
    for level in range(1, 7):
        lines.append(
            f"| L{level} | " + " | ".join(_percent(per_level[model["slug"]][level]["P5"]) for model in MODELS) + " |"
        )

    current_slug = MODELS[0]["slug"]
    lines += [
        "",
        "## 5. Input-context与其他模型的相同trial配对",
        "",
        "| 对照模型 | delta P5（Input-context - 对照） | Input-only/对照-only成功 | exact McNemar p |",
        "|---|---:|---:|---:|",
    ]
    for model in MODELS[1:]:
        paired = _paired(all_episodes[current_slug], all_episodes[model["slug"]])
        delta = overall[current_slug]["P5"] - overall[model["slug"]]["P5"]
        lines.append(
            f"| {model['name']} | {100.0 * delta:+.2f} pp | {paired['a_only']}/{paired['b_only']} | {_pvalue(paired['p'])} |"
        )

    lines += [
        "",
        "## 6. 解释边界",
        "",
        "- 所有测试均为inference-only，训练reward关闭。Input-context模型仍运行certificate solver，因为这是其Actor输入的一部分；其余960维模型不运行solver。",
        "- 有课程checkpoint都属于L2，但到达L2边界所经历的训练轮数不同，这是各自自适应课程轨迹的一部分；无课程模型和DWAQ只作最终参考。",
        "- L2边界组回答‘各方法在各自L2结束时的策略能力’，不等于相同训练样本预算的单变量消融；与无课程/DWAQ的差异更不能归因于单一机制。",
        "- 每个seed的trial ID、command和速度跳变完全一致，可进行相同trial配对比较。",
        "",
        f"原始结果：`{args.input_dir.resolve()}`",
        "",
    ]
    args.output.resolve().write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
