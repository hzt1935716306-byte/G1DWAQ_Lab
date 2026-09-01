#!/usr/bin/env python3
"""Analyze the native-system fixed-push comparison at component-wise +/-0.5 m/s."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats


PROJECT = Path(__file__).resolve().parents[2]
RESULT_DIR = Path(__file__).resolve().parent / "generated/cross_project_fixed_push_b050_2026-09-01"
OUTPUT = Path(__file__).resolve().parent / "CROSS_PROJECT_FIXED_PUSH_B050_COMPARISON.md"
SEEDS = (42, 123, 2026)
MODELS = {
    "unitree_latest_timefix": {
        "name": "Unitree 原生最新模型",
        "short": "Unitree-17800",
        "checkpoint": Path(
            "/home/zt/project/g1_base/unitree_rl_lab/logs/rsl_rl/"
            "unitree_g1_29dof_velocity/2026-04-24_06-32-29/model_17800.pt"
        ),
        "training": "原生 Unitree；训练 push 每 5 s，x/y 独立均匀于 [-0.5, 0.5] m/s",
    },
    "ours_cert020_L3_matched": {
        "name": "Ours-0.20 L3 范围匹配",
        "short": "Ours-0.20-L3",
        "checkpoint": PROJECT
        / "logs/g1_flat_symmetric/2026-08-31_00-20-09_stage2_ours_certonly020_async_from4999/model_9300.pt",
        "training": "certificate-only 0.20；L3；训练 push x/y 最大 +/-0.55 m/s；进入 L4 前最后整百 checkpoint",
    },
    "ours_cert020_curriculum": {
        "name": "Ours-0.20 最终模型",
        "short": "Ours-0.20-final",
        "checkpoint": PROJECT
        / "logs/g1_flat_symmetric/2026-08-31_21-43-19_stage2_ours_certonly020_resume_L5_from11700/model_14998.pt",
        "training": "certificate-only 0.20；完整课程最终模型；补充对照，不是训练范围匹配模型",
    },
    "ours_cert025_curriculum": {
        "name": "Ours-0.25 最终模型",
        "short": "Ours-0.25-final",
        "checkpoint": PROJECT / "logs/our0.25_model_14998_no_sharereward.pt",
        "training": "certificate-only 0.25；完整课程最终模型；本机没有对应 L3 中间 checkpoint",
    },
}


def read_reports():
    reports = {label: {} for label in MODELS}
    for label, model in MODELS.items():
        for seed in SEEDS:
            path = RESULT_DIR / f"{label}_seed{seed}.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            if Path(report["checkpoint"]).resolve() != model["checkpoint"].resolve():
                raise ValueError(f"checkpoint mismatch: {path}")
            if report["completed_episode_count"] != 256 or report["pending_episode_count"] != 0:
                raise ValueError(f"incomplete result: {path}")
            protocol = report["common_protocol"]
            if protocol["seed"] != seed or protocol["component_bound_mps"] != 0.5:
                raise ValueError(f"protocol mismatch: {path}")
            reports[label][seed] = report
    for seed in SEEDS:
        seed_reports = [reports[label][seed] for label in MODELS]
        hashes = {report["common_protocol"]["trial_plan_sha256"] for report in seed_reports}
        ids = [{episode["trial_id"] for episode in report["episodes"]} for report in seed_reports]
        if len(hashes) != 1 or any(value != ids[0] for value in ids[1:]):
            raise ValueError(f"trial plans are not paired at seed {seed}")
    return reports


def episodes(reports, label, *, seed=None, command=None, severity=None):
    result = []
    for current_seed in ((seed,) if seed is not None else SEEDS):
        for episode in reports[label][current_seed]["episodes"]:
            if command is not None and tuple(episode["command_velocity"]) != tuple(command):
                continue
            if severity is not None and severity_bin(episode) != severity:
                continue
            result.append({**episode, "seed": current_seed})
    return result


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {key: None for key in ("mean", "median", "p75", "p90")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def performance(items):
    count = Counter(item["outcome"] for item in items)
    success = [item for item in items if item["outcome"] == "SUCCESS"]
    failure_aware_steps = [
        item["practical_enter_step"]
        if item["outcome"] == "SUCCESS" and item["practical_enter_step"] is not None
        else 6
        for item in items
    ]
    return {
        "n": len(items),
        "success": count["SUCCESS"],
        "timeout": count["TIMEOUT"],
        "fall": count["FALL"],
        "p5": count["SUCCESS"] / len(items),
        "nonfall": 1.0 - count["FALL"] / len(items),
        "steps": quantiles([item["practical_enter_step"] for item in success]),
        "failure_aware_steps": quantiles(failure_aware_steps),
        "time": quantiles([item["recovery_time_s"] for item in success]),
    }


def wilson(success, total):
    z = 1.959963984540054
    p = success / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def paired(reports, first, second):
    a = {(item["seed"], item["trial_id"]): item for item in episodes(reports, first)}
    b = {(item["seed"], item["trial_id"]): item for item in episodes(reports, second)}
    if a.keys() != b.keys():
        raise ValueError(f"unpaired results: {first}, {second}")
    first_only = second_only = 0
    step_diffs = []
    time_diffs = []
    for key in a:
        a_success = a[key]["outcome"] == "SUCCESS"
        b_success = b[key]["outcome"] == "SUCCESS"
        first_only += int(a_success and not b_success)
        second_only += int(b_success and not a_success)
        if a_success and b_success:
            step_diffs.append(a[key]["practical_enter_step"] - b[key]["practical_enter_step"])
            time_diffs.append(a[key]["recovery_time_s"] - b[key]["recovery_time_s"])
    discordant = first_only + second_only
    p_value = stats.binomtest(first_only, discordant, 0.5).pvalue if discordant else 1.0

    def signed_rank(values):
        values = np.asarray(values, dtype=np.float64)
        if not values.size or np.allclose(values, 0.0):
            return 1.0
        return float(stats.wilcoxon(values).pvalue)

    first_p5 = performance(list(a.values()))["p5"]
    second_p5 = performance(list(b.values()))["p5"]
    return {
        "delta": first_p5 - second_p5,
        "first_only": first_only,
        "second_only": second_only,
        "p": float(p_value),
        "joint": len(step_diffs),
        "step_delta": float(np.mean(step_diffs)) if step_diffs else None,
        "step_p": signed_rank(step_diffs),
        "time_delta": float(np.mean(time_diffs)) if time_diffs else None,
        "time_p": signed_rank(time_diffs),
    }


def severity_bin(episode):
    normalized = max(abs(value) for value in episode["normalized_delta_xy"])
    return min(4, int(normalized * 4) + 1)


def percent(value):
    return f"{100 * value:.2f}%"


def p_text(value):
    if value < 1.0e-300:
        return "<1e-300"
    if value < 1.0e-3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def number(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def command_text(command):
    return "[" + ", ".join(f"{value:g}" for value in command) + "]"


def main():
    reports = read_reports()
    overall = {label: performance(episodes(reports, label)) for label in MODELS}
    ranking = sorted(MODELS, key=lambda label: overall[label]["p5"], reverse=True)
    comparisons = [
        ("ours_cert020_L3_matched", "unitree_latest_timefix", "主对比：Ours-0.20-L3 - Unitree"),
        ("ours_cert020_curriculum", "unitree_latest_timefix", "补充：Ours-0.20-final - Unitree"),
        ("ours_cert025_curriculum", "unitree_latest_timefix", "补充：Ours-0.25-final - Unitree"),
        ("ours_cert025_curriculum", "ours_cert020_curriculum", "权重：Ours-0.25-final - Ours-0.20-final"),
        ("ours_cert020_curriculum", "ours_cert020_L3_matched", "训练阶段：Ours-0.20-final - Ours-0.20-L3"),
    ]
    paired_results = [(title, paired(reports, first, second)) for first, second, title in comparisons]
    primary = paired_results[0][1]
    weight = paired_results[3][1]

    lines = [
        "# Unitree 最新模型与 Ours 课程模型：±0.5 m/s 统一冲击对比",
        "",
        "更新时间：2026-09-01（Asia/Shanghai）",
        "",
        "## 1. 结论",
        "",
        f"1. **训练范围最接近的主对比中，Ours-0.20-L3 的严格 P5 为 "
        f"{percent(overall['ours_cert020_L3_matched']['p5'])}，Unitree 最新模型为 "
        f"{percent(overall['unitree_latest_timefix']['p5'])}，差 {100 * primary['delta']:+.2f} pp**"
        f"（按相同 trial 描述配对的 McNemar p={p_text(primary['p'])}）。",
        f"2. 四个模型的 non-fall 都是 {percent(min(value['nonfall'] for value in overall.values()))}；"
        "这里拉开差距的是能否在 5 次 touchdown / 10 s 内进入严格稳定窗口，不是是否摔倒。",
        f"3. 最终权重对照中，0.25 相对 0.20 的 P5 差为 {100 * weight['delta']:+.2f} pp；"
        f"双方都成功的 trial 上，0.25 的平均 enter-step 差为 {weight['step_delta']:+.3f} 步。",
        "4. Unitree 的成功 episode 条件步数较低，但它有大量 TIMEOUT；条件步数不能脱离 P5 单独用于排名。",
        "5. Unitree 与 Ours 使用不同的原生 actor history、资产/执行器增益和 termination，"
        "所以这是统一刺激下的**原生系统级比较**；只有 Ours-0.20-final 与 Ours-0.25-final 更接近同栈权重消融。",
        "",
        "## 2. 模型与训练范围",
        "",
        "| 标识 | checkpoint | iter | actor obs | 训练身份 | SHA256 |",
        "|---|---|---:|---:|---|---|",
    ]
    for label, model in MODELS.items():
        report = reports[label][SEEDS[0]]
        try:
            path_text = str(model["checkpoint"].resolve().relative_to(PROJECT.resolve()))
        except ValueError:
            path_text = str(model["checkpoint"].resolve())
        lines.append(
            f"| {model['name']} | `{path_text}` | {report['checkpoint_iteration']} | "
            f"{report['actor_observation_shape'][-1]} | {model['training']} | "
            f"`{report['checkpoint_sha256'][:12]}...` |"
        )
    lines.extend(
        [
            "",
            "课程范围说明：Ours 课程档位为 ±0.25、±0.40、±0.55、±0.70、±0.85、±1.00 m/s，"
            "没有精确 ±0.50 档。`model_9300.pt` 是 L3→L4 升级前最后一个整百 checkpoint，"
            "因此以 ±0.55 作为与 Unitree ±0.50 最接近的主对比。0.25 在本机没有 L3 中间权重，"
            "故只能使用最终模型并明确标作补充对比。",
            "",
            "## 3. 测试协议与完整性",
            "",
            "- 每个模型 3 seeds × 256 = 768 个 episode；四模型共 3072 个；",
            "- 单次速度跳变：Δvx、Δvy 独立均匀于 [-0.5, 0.5] m/s；",
            "- 8 种共同 command；flat plane；关闭 observation noise、push event 和 physics randomization；",
            "- 每 seed 四模型共享同一 command/扰动列表、trial ID 和 plan hash；",
            "- 严格 P5：最多 5 次 recovery touchdown、10 s，并满足既有速度误差与 roll/pitch 合格窗口；",
            "- non-fall：10 s 内没有触发各自原生环境的跌倒 termination；",
            "- 这是单次冲击恢复测试，不等同于 Unitree 训练时每 5 s 重复一次 push 的长时存活测试。",
            "",
            "| Seed | 每模型 completed/pending | 共同 trial-plan SHA256 |",
            "|---:|---:|---|",
        ]
    )
    for seed in SEEDS:
        sample = reports["unitree_latest_timefix"][seed]
        lines.append(
            f"| {seed} | {sample['completed_episode_count']} / {sample['pending_episode_count']} | "
            f"`{sample['common_protocol']['trial_plan_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## 4. 总体结果",
            "",
            "P5 区间为合并 768 次的 Wilson 95% CI。成功步数只对 SUCCESS episode 统计；"
            "失败感知步数则把 TIMEOUT/FALL 记为 6 步。",
            "",
            "| 排名 | 模型 | S / T / F | P5 [95% CI] | non-fall | 成功步数 mean | median/P75/P90 | 失败感知步数 mean | 成功时间 mean |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, label in enumerate(ranking, 1):
        perf = overall[label]
        low, high = wilson(perf["success"], perf["n"])
        lines.append(
            f"| {rank} | {MODELS[label]['name']} | {perf['success']} / {perf['timeout']} / {perf['fall']} | "
            f"**{percent(perf['p5'])}** [{percent(low)}, {percent(high)}] | {percent(perf['nonfall'])} | "
            f"{number(perf['steps']['mean'])} | {number(perf['steps']['median'], 1)} / "
            f"{number(perf['steps']['p75'], 1)} / {number(perf['steps']['p90'], 1)} | "
            f"{number(perf['failure_aware_steps']['mean'])} | "
            f"{number(perf['time']['mean'])} s |"
        )
    lines.extend(
        [
            "",
            "## 5. 跨 seed 稳定性",
            "",
            "| 模型 | seed42 | seed123 | seed2026 | seed P5 mean ± population SD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in MODELS:
        values = [performance(episodes(reports, label, seed=seed))["p5"] for seed in SEEDS]
        lines.append(
            f"| {MODELS[label]['name']} | {percent(values[0])} | {percent(values[1])} | {percent(values[2])} | "
            f"{percent(float(np.mean(values)))} ± {100 * float(np.std(values)):.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## 6. 扰动强度分层",
            "",
            "强度使用 `max(|Δvx|, |Δvy|)`；Q1--Q4 固定为 (0,0.125]、(0.125,0.25]、"
            "(0.25,0.375]、(0.375,0.5] m/s。表内是 P5，括号内是该模型 trial 数。",
            "",
            "| 强度 | Unitree-17800 | Ours-0.20-L3 | Ours-0.20-final | Ours-0.25-final |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for severity in range(1, 5):
        cells = []
        for label in MODELS:
            perf = performance(episodes(reports, label, severity=severity))
            cells.append(f"{percent(perf['p5'])} ({perf['n']})")
        lines.append(f"| Q{severity} | " + " | ".join(cells) + " |")
    commands = reports["unitree_latest_timefix"][SEEDS[0]]["common_protocol"]["commands"]
    lines.extend(
        [
            "",
            "## 7. 按 command 的严格 P5",
            "",
            "| command [vx, vy, wz] | Unitree-17800 | Ours-0.20-L3 | Ours-0.20-final | Ours-0.25-final |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for command in commands:
        cells = [percent(performance(episodes(reports, label, command=command))["p5"]) for label in MODELS]
        lines.append(f"| `{command_text(command)}` | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## 8. 配对检验",
            "",
            "P5 使用 exact McNemar；`A-only/B-only` 是 A 成功 B 失败 / B 成功 A 失败。"
            "步数和时间差只在双方均 SUCCESS 的 trial 上统计，使用 Wilcoxon signed-rank。",
            "",
            "| 比较 A-B | P5 差 | A-only / B-only | McNemar p | 共同成功 n | enter-step 差 | p | 时间差 | p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for title, result in paired_results:
        lines.append(
            f"| {title} | {100 * result['delta']:+.2f} pp | {result['first_only']} / {result['second_only']} | "
            f"{p_text(result['p'])} | {result['joint']} | {result['step_delta']:+.3f} | "
            f"{p_text(result['step_p'])} | {result['time_delta']:+.3f} s | {p_text(result['time_p'])} |"
        )
    lines.extend(
        [
            "",
            "## 9. 可解释边界",
            "",
            "- Unitree 原生 actor 为 480 维（5 帧历史），Ours 为 960 维（10 帧历史）；",
            "- 两个工程的机器人资产族相近，但初始姿态、执行器 PD/限制及 termination 不完全相同；",
            "- 因此 Unitree 对 Ours 的差异包含整套原生系统差异，不能全部归因于 certificate reward；",
            "- Ours-0.20-L3 的训练上限是 ±0.55，不是精确 ±0.50；这是现有离散课程里误差最小的可用匹配；",
            "- Ours-0.25 只有最终 checkpoint，不能做严格的同训练范围 L3 对照；",
            "- 所有模型本次均 0 FALL，所以结论主要针对严格恢复窗口，而非更强 OOD 下的倒地边界。",
            "",
            "## 10. 复现文件",
            "",
            "- 评测器：[`evaluate_cross_project_fixed_push.py`](evaluate_cross_project_fixed_push.py)",
            "- 批处理：[`run_cross_project_fixed_push_b050.sh`](run_cross_project_fixed_push_b050.sh)",
            "- 原始 JSON/日志：[`generated/cross_project_fixed_push_b050_2026-09-01/`](generated/cross_project_fixed_push_b050_2026-09-01/)",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
