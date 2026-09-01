#!/usr/bin/env python3
"""Build the fixed-protocol report for certificate weights and no-curriculum ablations."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats


PROJECT = Path(__file__).resolve().parents[2]
RESULT_DIR = Path(__file__).resolve().parent / "generated/five_policy_no_shared_comprehensive_2026-09-01"
OUTPUT = Path(__file__).resolve().parent / "CERTIFICATE_WEIGHT_NO_CURRICULUM_COMPARISON.md"
SEEDS = (42, 123, 2026)
LEVELS = (1, 2, 3, 4, 5, 6)
MODELS = {
    "ours015_curriculum": {
        "name": "Ours-0.15（有课程）",
        "short": "Ours-0.15-C",
        "policy": "ours",
        "checkpoint": PROJECT / "logs/our0.15_model_14998_no_sharereware.pt",
        "iteration": 14998,
        "curriculum": "有（用户提供的训练身份；独立 checkpoint 无 params）",
        "reward": "certificate-only，event_scale=0.15；无 shared rewards",
    },
    "ours020_curriculum": {
        "name": "Ours-0.20（有课程）",
        "short": "Ours-0.20-C",
        "policy": "ours",
        "checkpoint": PROJECT / "logs/g1_flat_symmetric/2026-08-31_21-43-19_stage2_ours_certonly020_resume_L5_from11700/model_14998.pt",
        "iteration": 14998,
        "curriculum": "有；resume 时 L5，iterations_in_level=570",
        "reward": "certificate-only，event_scale=0.20；无 shared rewards",
    },
    "ours025_curriculum": {
        "name": "Ours-0.25（有课程）",
        "short": "Ours-0.25-C",
        "policy": "ours",
        "checkpoint": PROJECT / "logs/our0.25_model_14998_no_sharereward.pt",
        "iteration": 14998,
        "curriculum": "有（用户提供的训练身份；独立 checkpoint 无 params）",
        "reward": "certificate-only，event_scale=0.25；无 shared rewards",
    },
    "ours020_no_curriculum": {
        "name": "Ours-0.20（无课程）",
        "short": "Ours-0.20-NC",
        "policy": "ours",
        "checkpoint": PROJECT / "logs/g1_flat_symmetric/2026-08-31_23-44-06_stage2_ours_certonly020_no_curriculum_from4999/model_9998.pt",
        "iteration": 9998,
        "curriculum": "无；全程固定完整 L6 随机扰动",
        "reward": "certificate-only，event_scale=0.20；无 shared rewards",
    },
    "baseline_no_curriculum": {
        "name": "Baseline（无课程）",
        "short": "Baseline-NC",
        "policy": "baseline",
        "checkpoint": PROJECT / "logs/g1_flat_symmetric/2026-08-31_23-48-44_stage2_baseline_no_curriculum_from4999/model_9998.pt",
        "iteration": 9998,
        "curriculum": "无；全程固定完整 L6 随机扰动",
        "reward": "原始 locomotion reward；无 shared、无 certificate",
    },
}


def _read_reports() -> dict[str, dict[int, dict]]:
    reports: dict[str, dict[int, dict]] = {}
    for label, model in MODELS.items():
        reports[label] = {}
        for seed in SEEDS:
            path = RESULT_DIR / f"{label}_seed{seed}.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["policy"] != model["policy"]:
                raise ValueError(f"policy mismatch: {path}")
            if Path(report["checkpoint"]).resolve() != model["checkpoint"].resolve():
                raise ValueError(f"checkpoint mismatch: {path}")
            if report["planned_episode_count"] != 1536 or report["completed_episode_count"] != 1536:
                raise ValueError(f"incomplete report: {path}")
            if report["pending_episode_count"] != 0:
                raise ValueError(f"pending trials: {path}")
            if int(report["common_protocol"]["seed"]) != seed:
                raise ValueError(f"seed mismatch: {path}")
            reports[label][seed] = report

    for seed in SEEDS:
        seed_reports = [reports[label][seed] for label in MODELS]
        hashes = {report["common_protocol"]["trial_plan_sha256"] for report in seed_reports}
        trial_sets = [{item["trial_id"] for item in report["episodes"]} for report in seed_reports]
        if len(hashes) != 1 or any(value != trial_sets[0] for value in trial_sets[1:]):
            raise ValueError(f"unpaired trial plans for seed {seed}")
    return reports


def _episodes(reports, label: str, *, level=None, command=None, seed=None, strength_bin=None):
    selected = []
    seeds = (seed,) if seed is not None else SEEDS
    for current_seed in seeds:
        for episode in reports[label][current_seed]["episodes"]:
            if level is not None and episode["level"] != level:
                continue
            if command is not None and tuple(episode["command_velocity"]) != tuple(command):
                continue
            if strength_bin is not None and _strength_bin(episode) != strength_bin:
                continue
            selected.append({**episode, "seed": current_seed})
    return selected


def _q(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: None for key in ("mean", "median", "p75", "p90")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _perf(episodes):
    counts = Counter(item["outcome"] for item in episodes)
    successes = [item for item in episodes if item["outcome"] == "SUCCESS"]
    return {
        "n": len(episodes),
        "success": counts["SUCCESS"],
        "timeout": counts["TIMEOUT"],
        "fall": counts["FALL"],
        "rate": counts["SUCCESS"] / len(episodes),
        "steps": _q([item["practical_enter_step"] for item in successes]),
        "success_time": _q([item["recovery_time_s"] for item in successes]),
        "all_time": _q([item["recovery_time_s"] for item in episodes]),
        "step_distribution": Counter(item["practical_enter_step"] for item in successes),
    }


def _wilson(successes: int, total: int):
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - half, center + half


def _paired(reports, first: str, second: str, *, level=None):
    first_items = {
        (item["seed"], item["trial_id"]): item
        for item in _episodes(reports, first, level=level)
    }
    second_items = {
        (item["seed"], item["trial_id"]): item
        for item in _episodes(reports, second, level=level)
    }
    if first_items.keys() != second_items.keys():
        raise ValueError(f"unpaired episodes: {first}, {second}, L{level}")
    first_only = 0
    second_only = 0
    joint_step_diffs = []
    joint_time_diffs = []
    for key, a in first_items.items():
        b = second_items[key]
        a_success = a["outcome"] == "SUCCESS"
        b_success = b["outcome"] == "SUCCESS"
        first_only += int(a_success and not b_success)
        second_only += int(b_success and not a_success)
        if a_success and b_success:
            joint_step_diffs.append(a["practical_enter_step"] - b["practical_enter_step"])
            joint_time_diffs.append(a["recovery_time_s"] - b["recovery_time_s"])
    discordant = first_only + second_only
    mcnemar_p = stats.binomtest(first_only, discordant, 0.5).pvalue if discordant else 1.0

    def wilcoxon(values):
        array = np.asarray(values, dtype=np.float64)
        if not array.size or np.allclose(array, 0.0):
            return 1.0
        return float(stats.wilcoxon(array, alternative="two-sided").pvalue)

    a_rate = sum(item["outcome"] == "SUCCESS" for item in first_items.values()) / len(first_items)
    b_rate = sum(item["outcome"] == "SUCCESS" for item in second_items.values()) / len(second_items)
    return {
        "first_only": first_only,
        "second_only": second_only,
        "delta_rate": a_rate - b_rate,
        "mcnemar_p": float(mcnemar_p),
        "joint": len(joint_step_diffs),
        "step_delta": float(np.mean(joint_step_diffs)) if joint_step_diffs else None,
        "step_p": wilcoxon(joint_step_diffs),
        "time_delta": float(np.mean(joint_time_diffs)) if joint_time_diffs else None,
        "time_p": wilcoxon(joint_time_diffs),
    }


def _holm(p_values):
    count = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted = [0.0] * count
    running = 0.0
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (count - rank)))
        adjusted[index] = running
    return adjusted


def _p(value):
    if value < 1.0e-300:
        return "<1e-300"
    if value < 1.0e-3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _percent(value):
    return f"{100.0 * value:.2f}%"


def _number(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def _command_name(command):
    return "[" + ", ".join(f"{value:g}" for value in command) + "]"


STRENGTH_THRESHOLDS = None


def _strength_bin(episode):
    magnitude = float(np.linalg.norm(episode["normalized_delta_xy"]))
    return int(np.searchsorted(STRENGTH_THRESHOLDS, magnitude, side="right")) + 1


def _file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    global STRENGTH_THRESHOLDS
    reports = _read_reports()
    reference = _episodes(reports, "ours020_curriculum")
    strengths = np.asarray(
        [np.linalg.norm(item["normalized_delta_xy"]) for item in reference], dtype=np.float64
    )
    STRENGTH_THRESHOLDS = np.quantile(strengths, (0.25, 0.50, 0.75))
    overall = {label: _perf(_episodes(reports, label)) for label in MODELS}
    ranking = sorted(MODELS, key=lambda label: overall[label]["rate"], reverse=True)

    key_pairs = [
        ("ours020_curriculum", "ours015_curriculum"),
        ("ours020_curriculum", "ours025_curriculum"),
        ("ours020_no_curriculum", "baseline_no_curriculum"),
        ("ours020_curriculum", "ours020_no_curriculum"),
    ]
    key_results = {(a, b): _paired(reports, a, b) for a, b in key_pairs}
    weight_best = max(
        ("ours015_curriculum", "ours020_curriculum", "ours025_curriculum"),
        key=lambda label: overall[label]["rate"],
    )
    nc = key_results[("ours020_no_curriculum", "baseline_no_curriculum")]
    curriculum = key_results[("ours020_curriculum", "ours020_no_curriculum")]
    weight_015 = key_results[("ours020_curriculum", "ours015_curriculum")]
    weight_025 = key_results[("ours020_curriculum", "ours025_curriculum")]

    lines = [
        "# Certificate 权重与无课程训练全面对比",
        "",
        "更新时间：2026-09-01（Asia/Shanghai）",
        "",
        "> 范围说明：本文件只记录 certificate 权重与无课程5模型消融。包含全部11个正式最终模型的总报告见 "
        "[`ALL_TRAINED_MODEL_COMPREHENSIVE_COMPARISON.md`](ALL_TRAINED_MODEL_COMPREHENSIVE_COMPARISON.md)。",
        "",
        "## 1. 结论摘要",
        "",
        f"1. 固定协议总体 P5 最高的是 **{MODELS[ranking[0]]['name']}**："
        f"{_percent(overall[ranking[0]]['rate'])}；三个有课程权重中最高的是 "
        f"**{MODELS[weight_best]['name']}**（{_percent(overall[weight_best]['rate'])}）。",
        f"2. 在有课程权重消融中，0.25 相对 0.20 的 P5 高 "
        f"{-100 * weight_025['delta_rate']:.2f} pp，但双方成功 trial 上平均多 "
        f"{-weight_025['step_delta']:.3f} 步；0.20 表现为更强的步数效率，0.25 表现为更强的成功率。",
        f"3. 在两份无课程模型的直接对照中，Ours-0.20 相对 Baseline 的 P5 差为 "
        f"{100 * nc['delta_rate']:+.2f} pp（McNemar p={_p(nc['mcnemar_p'])}），"
        f"双方成功 trial 上的平均 enter-step 差为 {nc['step_delta']:+.3f}。",
        f"4. 有课程 Ours-0.20 相对无课程 Ours-0.20 的 P5 差为 "
        f"{100 * curriculum['delta_rate']:+.2f} pp（p={_p(curriculum['mcnemar_p'])}）。"
        "该比较同时包含 checkpoint iteration/训练历史差异，不能解释为纯课程因果效应。",
        "5. 五个模型均为 0 FALL；差异全部来自能否在五次 touchdown/10 s 内进入严格合格窗口，"
        "不是避免倒地能力的差异。",
        "6. 所有 enter-step 均只在 SUCCESS episode 上统计；它必须与 TIMEOUT/FALL/P5 一起看，"
        "不能用更低的条件均值掩盖更多失败。",
        "",
        "## 2. 模型身份与可比性",
        "",
        "下表 reward 指训练期间的 reward；本次评测为 inference-only，不计算训练 reward，也不运行 certificate/LIPM solver。",
        "",
        "| 模型 | checkpoint | iter 字段 | 训练课程 | 训练 reward | SHA256 |",
        "|---|---|---:|---|---|---|",
    ]
    for label, model in MODELS.items():
        relative = model["checkpoint"].resolve().relative_to(PROJECT.resolve())
        lines.append(
            f"| {model['name']} | `{relative}` | {model['iteration']} | {model['curriculum']} | "
            f"{model['reward']} | `{_file_sha256(model['checkpoint'])[:12]}…` |"
        )
    lines.extend(
        [
            "",
            "两份独立的 0.15/0.25 checkpoint 不附带对应 `params/env.yaml`；其权重和课程身份来自用户提供的训练记录。"
            "网络结构、checkpoint iter 和可加载性已经验证，但无法仅从 `.pt` 反向证明其完整课程轨迹。",
            "",
            "两份无课程 checkpoint 的 `iter=9998`，三份有课程 checkpoint 的 `iter=14998`。按用户说明可正常横向评测，"
            "但涉及有/无课程的差异必须同时注明训练轮次/历史不完全相同。",
            "",
            "## 3. 固定测试协议与完整性",
            "",
            "- seed：42、123、2026；L1--L6；每等级每 seed 256 个 episode；",
            "- 每模型 4608 个 episode，五模型共 23040 个；",
            "- 每个 seed 内五个模型共享完全相同的 command、扰动、`trial_id` 和 trial-plan hash；",
            "- 8 种 command；flat plane；关闭 observation noise 和 physics randomization；",
            "- 最大 5 次 recovery touchdown，最大恢复时间 10 s；",
            "- actor 使用各自 checkpoint，但本组均为 symmetric policy，actor observation dim 均为 960。",
            "",
            "| Seed | 每模型 planned/completed/pending | trial hash | 五模型异常 reset 合计 |",
            "|---:|---:|---|---:|",
        ]
    )
    for seed in SEEDS:
        sample = reports[next(iter(MODELS))][seed]
        resets = sum(reports[label][seed]["nominal_reset_count"] for label in MODELS)
        lines.append(
            f"| {seed} | 1536 / 1536 / 0 | `{sample['common_protocol']['trial_plan_sha256']}` | {resets} |"
        )

    lines.extend(
        [
            "",
            "## 4. 总体结果",
            "",
            "P5 的 95% CI 为 Wilson interval。步数和时间为三个 seed 合并后的 SUCCESS episode 统计。",
            "",
            "| 模型 | SUCCESS / TIMEOUT / FALL | P5 [95% CI] | 成功步数 mean | median/P75/P90 | 成功时间 mean | 全 episode 时间 mean |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ranking:
        perf = overall[label]
        low, high = _wilson(perf["success"], perf["n"])
        steps = perf["steps"]
        lines.append(
            f"| {MODELS[label]['name']} | {perf['success']} / {perf['timeout']} / {perf['fall']} | "
            f"**{_percent(perf['rate'])}** [{_percent(low)}, {_percent(high)}] | {_number(steps['mean'])} | "
            f"{_number(steps['median'], 1)} / {_number(steps['p75'], 1)} / {_number(steps['p90'], 1)} | "
            f"{_number(perf['success_time']['mean'])} s | {_number(perf['all_time']['mean'])} s |"
        )

    lines.extend(
        [
            "",
            "### 4.1 跨 seed 稳定性",
            "",
            "| Seed | " + " | ".join(model["short"] for model in MODELS.values()) + " |",
            "|---:|" + "---:|" * len(MODELS),
        ]
    )
    for seed in SEEDS:
        values = [_percent(_perf(_episodes(reports, label, seed=seed))["rate"]) for label in MODELS]
        lines.append(f"| {seed} | " + " | ".join(values) + " |")
    sd_values = []
    for label in MODELS:
        rates = [_perf(_episodes(reports, label, seed=seed))["rate"] for seed in SEEDS]
        sd_values.append(f"{100 * np.std(rates):.2f} pp")
    lines.append("| population SD | " + " | ".join(sd_values) + " |")

    lines.extend(
        [
            "",
            "### 4.2 成功 episode 的 enter-step 分布",
            "",
            "| 模型 | 1 步 | 2 步 | 3 步 | 4 步 | 5 步 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in MODELS:
        distribution = overall[label]["step_distribution"]
        lines.append(
            f"| {MODELS[label]['name']} | " + " | ".join(str(distribution[step]) for step in range(1, 6)) + " |"
        )

    lines.extend(
        [
            "",
            "## 5. 逐等级结果",
            "",
            "### 5.1 严格恢复成功率 P5",
            "",
            "每格为三个 seed 合并后的成功率。",
            "",
            "| Level | " + " | ".join(model["short"] for model in MODELS.values()) + " |",
            "|---|" + "---:|" * len(MODELS),
        ]
    )
    for level in LEVELS:
        values = [_percent(_perf(_episodes(reports, label, level=level))["rate"]) for label in MODELS]
        lines.append(f"| L{level} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "### 5.2 成功 episode 平均 enter step",
            "",
            "| Level | " + " | ".join(model["short"] for model in MODELS.values()) + " |",
            "|---|" + "---:|" * len(MODELS),
        ]
    )
    for level in LEVELS:
        values = [_number(_perf(_episodes(reports, label, level=level))["steps"]["mean"]) for label in MODELS]
        lines.append(f"| L{level} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "### 5.3 成功 episode 平均恢复时间",
            "",
            "| Level | " + " | ".join(model["short"] for model in MODELS.values()) + " |",
            "|---|" + "---:|" * len(MODELS),
        ]
    )
    for level in LEVELS:
        values = [
            _number(_perf(_episodes(reports, label, level=level))["success_time"]["mean"])
            for label in MODELS
        ]
        lines.append(f"| L{level} | " + " | ".join(f"{value} s" for value in values) + " |")

    commands = [tuple(value) for value in reports[next(iter(MODELS))][SEEDS[0]]["common_protocol"]["commands"]]
    lines.extend(
        [
            "",
            "## 6. 分 command 的严格 P5",
            "",
            "每种 command 每模型合计 576 个 episode，并混合 L1--L6。",
            "",
            "| Command `[vx, vy, wz]` | " + " | ".join(model["short"] for model in MODELS.values()) + " |",
            "|---|" + "---:|" * len(MODELS),
        ]
    )
    for command in commands:
        values = [_percent(_perf(_episodes(reports, label, command=command))["rate"]) for label in MODELS]
        lines.append(f"| `{_command_name(command)}` | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## 7. 分归一化扰动强度四分位的 P5",
            "",
            "强度定义为 `sqrt(nx^2 + ny^2)`，其中 `nx, ny` 是各等级范围内的归一化扰动。"
            f"Q1/Q2/Q3 边界为 {STRENGTH_THRESHOLDS[0]:.3f} / {STRENGTH_THRESHOLDS[1]:.3f} / "
            f"{STRENGTH_THRESHOLDS[2]:.3f}。",
            "",
            "| 强度组 | " + " | ".join(model["short"] for model in MODELS.values()) + " |",
            "|---|" + "---:|" * len(MODELS),
        ]
    )
    for strength_bin in range(1, 5):
        values = [
            _percent(_perf(_episodes(reports, label, strength_bin=strength_bin))["rate"])
            for label in MODELS
        ]
        lines.append(f"| Q{strength_bin} | " + " | ".join(values) + " |")

    all_pairs = list(itertools.combinations(MODELS, 2))
    all_results = [_paired(reports, first, second) for first, second in all_pairs]
    adjusted = _holm([result["mcnemar_p"] for result in all_results])
    lines.extend(
        [
            "",
            "## 8. 相同 trial 的全 pair 配对统计",
            "",
            "delta 均为 A-B。`A only/B only` 是成功结果不一致的 trial；McNemar 为双侧 exact test，"
            "并报告 10 个模型对比的 Holm 校正 p。step/time 只在双方都 SUCCESS 的 trial 上做双侧 Wilcoxon。",
            "",
            "| A | B | delta P5 | A only / B only | McNemar p | Holm p | joint success | delta step | step p | delta time | time p |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for (first, second), result, adjusted_p in zip(all_pairs, all_results, adjusted):
        lines.append(
            f"| {MODELS[first]['short']} | {MODELS[second]['short']} | {100 * result['delta_rate']:+.2f} pp | "
            f"{result['first_only']} / {result['second_only']} | {_p(result['mcnemar_p'])} | {_p(adjusted_p)} | "
            f"{result['joint']} | {result['step_delta']:+.3f} | {_p(result['step_p'])} | "
            f"{result['time_delta']:+.3f} s | {_p(result['time_p'])} |"
        )

    lines.extend(
        [
            "",
            "### 8.1 主要假设的逐等级配对结果",
            "",
            "这里不做逐等级多重比较校正，因此 p 值是探索性的。",
            "",
            "| A | B | Level | delta P5 | A only / B only | McNemar p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for first, second in key_pairs:
        for level in LEVELS:
            result = _paired(reports, first, second, level=level)
            lines.append(
                f"| {MODELS[first]['short']} | {MODELS[second]['short']} | L{level} | "
                f"{100 * result['delta_rate']:+.2f} pp | {result['first_only']} / {result['second_only']} | "
                f"{_p(result['mcnemar_p'])} |"
            )

    lines.extend(
        [
            "",
            "## 9. 解释边界与后续价值",
            "",
            "### 9.1 这组实验实际说明了什么",
            "",
            f"- **certificate reward 有可测的正作用，但首先体现在成功率。**无课程 Ours-0.20 比同条件 Baseline "
            f"高 {100 * nc['delta_rate']:.2f} pp，少 {overall['baseline_no_curriculum']['timeout'] - overall['ours020_no_curriculum']['timeout']} 个 TIMEOUT；"
            f"但配对 enter-step 只差 {nc['step_delta']:+.3f}（p={_p(nc['step_p'])}），没有证据表明它在这组无课程训练中减少踏步。",
            f"- 无课程 Ours 相对 Baseline 的增益并非覆盖所有区域：L1/L2/L5 分别为 "
            f"{100 * (_perf(_episodes(reports, 'ours020_no_curriculum', level=1))['rate'] - _perf(_episodes(reports, 'baseline_no_curriculum', level=1))['rate']):+.2f} / "
            f"{100 * (_perf(_episodes(reports, 'ours020_no_curriculum', level=2))['rate'] - _perf(_episodes(reports, 'baseline_no_curriculum', level=2))['rate']):+.2f} / "
            f"{100 * (_perf(_episodes(reports, 'ours020_no_curriculum', level=5))['rate'] - _perf(_episodes(reports, 'baseline_no_curriculum', level=5))['rate']):+.2f} pp，"
            f"但 L6 仅 {100 * (_perf(_episodes(reports, 'ours020_no_curriculum', level=6))['rate'] - _perf(_episodes(reports, 'baseline_no_curriculum', level=6))['rate']):+.2f} pp 且不显著。"
            f"按 command，后退 `[-0.3,0,0]` 和静止 `[0,0,0]` 分别提高 "
            f"{100 * (_perf(_episodes(reports, 'ours020_no_curriculum', command=(-0.3, 0.0, 0.0)))['rate'] - _perf(_episodes(reports, 'baseline_no_curriculum', command=(-0.3, 0.0, 0.0)))['rate']):.2f} / "
            f"{100 * (_perf(_episodes(reports, 'ours020_no_curriculum', command=(0.0, 0.0, 0.0)))['rate'] - _perf(_episodes(reports, 'baseline_no_curriculum', command=(0.0, 0.0, 0.0)))['rate']):.2f} pp。",
            f"- **0.25 最值得作为高扰动鲁棒性方向继续拓展。**它相对有课程 0.20 在 L5/L6 分别高 "
            f"{100 * (_perf(_episodes(reports, 'ours025_curriculum', level=5))['rate'] - _perf(_episodes(reports, 'ours020_curriculum', level=5))['rate']):.2f} / "
            f"{100 * (_perf(_episodes(reports, 'ours025_curriculum', level=6))['rate'] - _perf(_episodes(reports, 'ours020_curriculum', level=6))['rate']):.2f} pp，"
            f"强扰动 Q4 高 {100 * (_perf(_episodes(reports, 'ours025_curriculum', strength_bin=4))['rate'] - _perf(_episodes(reports, 'ours020_curriculum', strength_bin=4))['rate']):.2f} pp。",
            f"- **0.20 最值得作为少踏步方向保留。**它相对 0.15 同时提高 P5 "
            f"{100 * weight_015['delta_rate']:.2f} pp，并在双方成功 trial 上少 {-weight_015['step_delta']:.3f} 步；"
            f"相对 0.25 则少 {-weight_025['step_delta']:.3f} 步，但牺牲 {-100 * weight_025['delta_rate']:.2f} pp P5。",
            f"- **无课程训练呈现成功率与步数效率的取舍。**无课程 0.20 比有课程 0.20 高 "
            f"{-100 * curriculum['delta_rate']:.2f} pp P5，但双方成功时平均多 {-curriculum['step_delta']:.3f} 步、"
            f"多 {-curriculum['time_delta']:.3f} s；由于训练迭代和历史不同，这只是现象，不是纯课程因果结论。",
            f"- **时间效率仍是 Ours 的短板。**无课程 Ours 与 Baseline 的成功 enter-step 基本相同，"
            f"但成功恢复时间平均多 {nc['time_delta']:.3f} s（p={_p(nc['time_p'])}）。certificate 奖励按 touchdown 推动进展，"
            "未直接鼓励缩短两次 touchdown 之间的物理时间。",
            "- 所有模型的 1 步成功数均为 0，是因为 practical-good-cycle 指标必须先形成一个完整 touchdown interval 才能判定，"
            "不是证明机器人在动力学上绝对不可能一步恢复。",
            "",
            "### 9.2 建议优先保留的指标",
            "",
            "1. 主指标：总体 P5，并单独报告 L5/L6 和强扰动 Q4 P5；它最能体现 0.25 的价值。",
            "2. 第二主指标：双方成功 trial 的配对 enter-step，而不是各模型各自成功样本的非配对均值；它最能体现 0.20 的步数效率。",
            "3. 必须同时报告 SUCCESS/TIMEOUT/FALL，当前所有差异都来自 TIMEOUT，不能写成防摔提升。",
            "4. 保留成功恢复时间；无课程 Ours 虽提高 P5，但比 Baseline 慢，后续若要真实快速恢复需要单独解决。",
            "5. 分 command 重点看静止和后退命令，分等级重点看 L5/L6；这些区域对 certificate 权重最敏感。",
            "",
            "### 9.3 解释边界",
            "",
            "不能直接得出：",
            "",
            "- 有课程与无课程模型的差异不能全部归因于课程，因为 checkpoint iter 和训练历史不同；",
            "- 0.15/0.25 的 standalone `.pt` 没有 env/agent YAML，课程轨迹身份不能从权重文件独立复核；",
            "- 本测试关闭观测噪声和物理随机化，不能代替 sim-to-real 或随机动力学泛化测试；",
            "- 成功样本中的低 enter-step 不能代替总体 P5，二者必须联合解释；",
            "- 统计 p 值不等于实际效应大小，尤其在 4608 个配对 trial 下，小差异也可能显著。",
            "",
            "## 10. 原始数据与复现",
            "",
            f"原始目录：[`{RESULT_DIR.relative_to(Path(__file__).resolve().parent)}/`]"
            f"({RESULT_DIR.relative_to(Path(__file__).resolve().parent)}/)",
            "",
            "每个模型包含 `seed42/123/2026.json` 和对应 Isaac 运行日志。",
            "",
            "复现脚本：[`run_five_policy_no_shared_comprehensive_sweep.sh`](run_five_policy_no_shared_comprehensive_sweep.sh)",
            "",
            "汇总脚本：[`analyze_five_policy_no_shared_comparison.py`](analyze_five_policy_no_shared_comparison.py)",
            "",
        ]
    )
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
