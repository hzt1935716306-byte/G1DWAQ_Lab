#!/usr/bin/env python3
"""Analyze the paired cross-project disturbance-survival suite."""

from __future__ import annotations

import json
import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats


PROJECT = Path(__file__).resolve().parents[2]
RESULT_DIR = Path(__file__).resolve().parent / "generated/cross_project_survival_suite_2026-09-01"
OUTPUT = Path(__file__).resolve().parent / "CROSS_PROJECT_SURVIVAL_SUITE_COMPARISON.md"
SEEDS = (42, 123, 2026)
FAILURE_STEP = 6
MODELS = {
    "unitree_latest": {
        "name": "Unitree-17800",
        "checkpoint": Path(
            "/home/zt/project/g1_base/unitree_rl_lab/logs/rsl_rl/"
            "unitree_g1_29dof_velocity/2026-04-24_06-32-29/model_17800.pt"
        ),
        "training": "Unitree 原生速度策略；训练 push x/y 独立均匀于 [-0.5, 0.5] m/s",
    },
    "ours_cert020_L3_matched": {
        "name": "Ours-0.20-L3",
        "checkpoint": PROJECT
        / "logs/g1_flat_symmetric/2026-08-31_00-20-09_stage2_ours_certonly020_async_from4999/model_9300.pt",
        "training": "certificate-only 0.20；课程 L3；训练 push 最大约 +/-0.55 m/s",
    },
    "ours_cert025_final": {
        "name": "Ours-0.25-final",
        "checkpoint": PROJECT / "logs/our0.25_model_14998_no_sharereward.pt",
        "training": "certificate-only 0.25；完整课程最终模型；训练范围不是与 Unitree 严格匹配",
    },
    "baseline_shared020_L3": {
        "name": "Baseline-shared-0.2-L3",
        "checkpoint": PROJECT
        / "logs/g1_flat_symmetric/2026-08-29_13-15-09_stage2_baseline_scale02_solverfix_resume/model_8200.pt",
        "training": "三个 shared recovery events，scale=0.2；课程 L3；L3->L4（global 8225）前最后整百 checkpoint",
    },
    "baseline_original_no_curriculum": {
        "name": "Baseline-original-NC",
        "checkpoint": PROJECT
        / "logs/g1_flat_symmetric/2026-08-31_23-48-44_stage2_baseline_no_curriculum_from4999/model_9998.pt",
        "training": "原始 locomotion reward；无课程；全程固定 L6 x/y +/-1.0 m/s",
    },
}
FAMILIES = (
    "velocity_ood",
    "force_pulse",
    "constant_force",
    "repeated_impulse",
    "random_force",
    "wrench_pulse",
)
FAMILY_NAMES = {
    "velocity_ood": "瞬时速度冲击（OOD）",
    "force_pulse": "有限时长外力脉冲",
    "constant_force": "5 s 持续外力",
    "repeated_impulse": "8 s 重复冲击",
    "random_force": "10 s 随机外力",
    "wrench_pulse": "力/力矩脉冲",
}


def read_reports():
    reports = {label: {} for label in MODELS}
    condition_specs = None
    for label, model in MODELS.items():
        for seed in SEEDS:
            path = RESULT_DIR / f"{label}_seed{seed}.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            if Path(report["checkpoint"]).resolve() != model["checkpoint"].resolve():
                raise ValueError(f"checkpoint mismatch: {path}")
            if report["completed_episode_count"] != 2048 or report["pending_episode_count"] != 0:
                raise ValueError(f"incomplete report: {path}")
            protocol = report["common_protocol"]
            if protocol["seed"] != seed or not protocol["fixed_survival_horizon"]:
                raise ValueError(f"protocol mismatch: {path}")
            if condition_specs is None:
                condition_specs = report["condition_specs"]
            elif report["condition_specs"] != condition_specs:
                raise ValueError(f"condition spec mismatch: {path}")
            reports[label][seed] = report

    for seed in SEEDS:
        current = [reports[label][seed] for label in MODELS]
        hashes = {report["common_protocol"]["trial_plan_sha256"] for report in current}
        trial_ids = [{item["trial_id"] for item in report["episodes"]} for report in current]
        if len(hashes) != 1 or any(ids != trial_ids[0] for ids in trial_ids[1:]):
            raise ValueError(f"unpaired trial plan for seed {seed}")
    return reports, condition_specs


def all_episodes(reports, label, *, family=None, condition=None, seed=None):
    values = []
    selected_seeds = (seed,) if seed is not None else SEEDS
    for current_seed in selected_seeds:
        for item in reports[label][current_seed]["episodes"]:
            if family is not None and item["family"] != family:
                continue
            if condition is not None and item["condition_id"] != condition:
                continue
            values.append({**item, "seed": current_seed})
    return values


def quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {key: None for key in ("mean", "median", "p75", "p90")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
    }


def performance(items):
    outcomes = Counter(item["outcome"] for item in items)
    success = [item for item in items if item["outcome"] == "SUCCESS"]
    survival_count = sum(bool(item["survived_full_horizon"]) for item in items)
    success_steps = [item["practical_enter_step"] for item in success]
    failure_aware = [
        item["practical_enter_step"]
        if item["outcome"] == "SUCCESS" and item["practical_enter_step"] is not None
        else FAILURE_STEP
        for item in items
    ]
    return {
        "n": len(items),
        "survival_count": survival_count,
        "survival": survival_count / len(items),
        "success": outcomes["SUCCESS"],
        "timeout": outcomes["TIMEOUT"],
        "fall": outcomes["FALL"],
        "p5": outcomes["SUCCESS"] / len(items),
        "success_steps": quantiles(success_steps),
        "failure_aware_steps": quantiles(failure_aware),
        "response_velocity_rms": quantiles(
            [item["response_velocity_error_rms"] for item in items if item["response_velocity_error_rms"] is not None]
        ),
        "max_roll": quantiles([item["max_abs_roll"] for item in items]),
        "max_pitch": quantiles([item["max_abs_pitch"] for item in items]),
    }


def wilson(success, total):
    z = 1.959963984540054
    p = success / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def paired_binary(reports, first, second, key):
    a = {(item["seed"], item["trial_id"]): bool(item[key]) for item in all_episodes(reports, first)}
    b = {(item["seed"], item["trial_id"]): bool(item[key]) for item in all_episodes(reports, second)}
    if a.keys() != b.keys():
        raise ValueError(f"unpaired results: {first}, {second}")
    first_only = sum(a[k] and not b[k] for k in a)
    second_only = sum(b[k] and not a[k] for k in a)
    discordant = first_only + second_only
    p_value = stats.binomtest(first_only, discordant, 0.5).pvalue if discordant else 1.0
    return {
        "delta": np.mean(list(a.values())) - np.mean(list(b.values())),
        "first_only": first_only,
        "second_only": second_only,
        "p": float(p_value),
    }


def paired_p5(reports, first, second):
    converted = {label: {} for label in (first, second)}
    for label in converted:
        for seed in SEEDS:
            report = reports[label][seed]
            converted[label][seed] = {
                **report,
                "episodes": [
                    {**item, "p5_success": item["outcome"] == "SUCCESS"} for item in report["episodes"]
                ],
            }
    return paired_binary(converted, first, second, "p5_success")


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


def path_text(path):
    try:
        return str(path.resolve().relative_to(PROJECT.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    reports, condition_specs = read_reports()
    overall = {label: performance(all_episodes(reports, label)) for label in MODELS}
    rankings = sorted(MODELS, key=lambda label: overall[label]["survival"], reverse=True)
    comparisons = [
        ("ours_cert020_L3_matched", "unitree_latest", "Ours-0.20-L3 - Unitree-17800"),
        ("baseline_shared020_L3", "unitree_latest", "Baseline-shared-0.2-L3 - Unitree-17800"),
        ("baseline_original_no_curriculum", "unitree_latest", "Baseline-original-NC - Unitree-17800"),
        ("ours_cert025_final", "unitree_latest", "Ours-0.25-final - Unitree-17800"),
        ("ours_cert025_final", "ours_cert020_L3_matched", "Ours-0.25-final - Ours-0.20-L3"),
        ("ours_cert020_L3_matched", "baseline_shared020_L3", "Ours-0.20-L3 - Baseline-shared-0.2-L3"),
        ("ours_cert025_final", "baseline_original_no_curriculum", "Ours-0.25-final - Baseline-original-NC"),
    ]
    paired = [
        (title, paired_binary(reports, first, second, "survived_full_horizon"), paired_p5(reports, first, second))
        for first, second, title in comparisons
    ]
    primary = paired[0][1]
    l3_ablation = paired[5]
    full_range_ablation = paired[6]

    lines = [
        "# Unitree、Ours 与 Baseline：多类型扰动存活率全面对比",
        "",
        "更新时间：2026-09-01（Asia/Shanghai）",
        "",
        "## 1. 核心结论",
        "",
        f"1. 总存活率排名为：" + " > ".join(
            f"**{MODELS[label]['name']} {percent(overall[label]['survival'])}**" for label in rankings
        ) + "。",
        "2. 训练扰动范围近似匹配（Unitree +/-0.50、两个 L3 约 +/-0.55）时，Unitree 存活率为 "
        f"{percent(overall['unitree_latest']['survival'])}，Ours-0.20-L3 为 "
        f"{percent(overall['ours_cert020_L3_matched']['survival'])}，Baseline-shared-0.2-L3 为 "
        f"{percent(overall['baseline_shared020_L3']['survival'])}；Ours 相对 Unitree 差 "
        f"**{100 * primary['delta']:+.2f} pp**（p={p_text(primary['p'])}）。",
        "3. 同为完整 +/-1.0 训练范围时，Baseline-original-NC 相对 Ours-0.25-final 的存活率高 "
        f"**{-100 * full_range_ablation[1]['delta']:+.2f} pp**"
        f"（p={p_text(full_range_ablation[1]['p'])}），P5 高 "
        f"**{-100 * full_range_ablation[2]['delta']:+.2f} pp**"
        f"（p={p_text(full_range_ablation[2]['p'])}）。这组结果不支持把 Ours-0.25 的高存活率主要归因于 certificate reward。",
        "4. L3 同范围下，Ours-0.20 相对 shared Baseline 的存活率差为 "
        f"{100 * l3_ablation[1]['delta']:+.2f} pp（p={p_text(l3_ablation[1]['p'])}），"
        f"但 P5 低 {-100 * l3_ablation[2]['delta']:.2f} pp（p={p_text(l3_ablation[2]['p'])}）。",
        "5. 完整时域存活的定义是：机器人不只要扛过扰动，还要在扰动结束后继续存活 10 s；"
        "P5（5 次 touchdown / 10 s 内进入严格窗口）单独报告，二者不能混用。",
        "6. 恢复步数同时报告两种口径：`成功样本均值` 会排除 TIMEOUT/FALL；"
        f"`失败感知均值` 对 SUCCESS 用实际步数，对 TIMEOUT/FALL 统一记为 {FAILURE_STEP} 步。",
        "7. Unitree 与其余四个模型使用不同的原生 actor history、资产、控制增益和 termination。"
        "外力按各自机器人质量归一化，因此这是同一刺激协议下的原生系统级比较，不是只替换网络权重的同栈消融。",
        "",
        "## 2. 模型身份",
        "",
        "| 标识 | checkpoint | actor obs | 训练身份 | SHA256 |",
        "|---|---|---:|---|---|",
    ]
    for label, model in MODELS.items():
        report = reports[label][SEEDS[0]]
        lines.append(
            f"| {model['name']} | `{path_text(model['checkpoint'])}` | "
            f"{report['actor_observation_shape'][-1]} | {model['training']} | "
            f"`{sha256_file(model['checkpoint'])[:12]}...` |"
        )

    lines.extend(
        [
            "",
            "## 3. 统一协议与完整性",
            "",
            f"- 3 个 seed：42、123、2026；每模型每 seed 2,048 次；每模型共 6,144 次；全部 {len(MODELS)} 个模型共 {len(MODELS) * 6144:,} 次。",
            "- 6 类、32 个扰动条件；每条件每 seed 64 次；8 个共同速度命令；相同 seed 下 trial ID、方向、命令和扰动参数严格配对。",
            "- 扰动作用于 `torso_link`，世界坐标系；力按模型总质量换算成相同等效加速度/速度增量。",
            "- 关闭观测噪声和物理随机化，平地，control dt=0.02 s；扰动后固定观察 10 s。",
            "",
            "## 4. 总体结果",
            "",
            "| 模型 | 完整时域存活率 [95% CI] | P5 | SUCCESS / TIMEOUT / FALL | 成功样本恢复步数 mean/median/P75/P90 | 失败感知步数 mean/median/P75/P90 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in MODELS:
        value = overall[label]
        lo, hi = wilson(value["survival_count"], value["n"])
        success_steps = value["success_steps"]
        capped_steps = value["failure_aware_steps"]
        lines.append(
            f"| {MODELS[label]['name']} | {percent(value['survival'])} "
            f"[{percent(lo)}, {percent(hi)}] | {percent(value['p5'])} | "
            f"{value['success']} / {value['timeout']} / {value['fall']} | "
            f"{number(success_steps['mean'])}/{number(success_steps['median'])}/"
            f"{number(success_steps['p75'])}/{number(success_steps['p90'])} | "
            f"{number(capped_steps['mean'])}/{number(capped_steps['median'])}/"
            f"{number(capped_steps['p75'])}/{number(capped_steps['p90'])} |"
        )

    lines.extend(
        [
            "",
            "这里的 `成功样本恢复步数` 只回答“已经成功的 trial 通常用了几步”；"
            "不能用它代替总体恢复能力，因为失败样本没有 enter-step。",
            "",
            "### 配对显著性",
            "",
            "| 对比（前者-后者） | 存活率差 | 存活 discordant 前胜/后胜 | 存活 p | P5 差 | P5 discordant 前胜/后胜 | P5 p |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for title, survival, p5 in paired:
        lines.append(
            f"| {title} | {100 * survival['delta']:+.2f} pp | "
            f"{survival['first_only']}/{survival['second_only']} | {p_text(survival['p'])} | "
            f"{100 * p5['delta']:+.2f} pp | {p5['first_only']}/{p5['second_only']} | {p_text(p5['p'])} |"
        )

    lines.extend(
        [
            "",
            "## 5. 分扰动类别结果",
            "",
            "| 扰动类别 | 模型 | n | 完整时域存活率 | P5 | S/T/F | 成功步数 mean | 失败感知步数 mean | 响应速度误差 RMS mean |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for family in FAMILIES:
        for label in MODELS:
            value = performance(all_episodes(reports, label, family=family))
            lines.append(
                f"| {FAMILY_NAMES[family]} | {MODELS[label]['name']} | {value['n']} | "
                f"{percent(value['survival'])} | {percent(value['p5'])} | "
                f"{value['success']}/{value['timeout']}/{value['fall']} | "
                f"{number(value['success_steps']['mean'])} | "
                f"{number(value['failure_aware_steps']['mean'])} | "
                f"{number(value['response_velocity_rms']['mean'])} |"
            )

    lines.extend(
        [
            "",
            "## 6. 全部 32 个扰动条件",
            "",
            "单元格为 `完整时域存活率 / P5`。每个单元格 n=192（3 seed x 64）。",
            "",
            "| 类别 | 条件 | Unitree-17800 | Ours-0.20-L3 | Ours-0.25-final | Baseline-shared-0.2-L3 | Baseline-original-NC | 存活率最佳 |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for spec in condition_specs:
        values = {
            label: performance(
                all_episodes(reports, label, family=spec["family"], condition=spec["condition_id"])
            )
            for label in MODELS
        }
        best_value = max(value["survival"] for value in values.values())
        best = ", ".join(
            MODELS[label]["name"] for label, value in values.items() if value["survival"] == best_value
        )
        cells = [f"{percent(values[label]['survival'])} / {percent(values[label]['p5'])}" for label in MODELS]
        lines.append(
            f"| {FAMILY_NAMES[spec['family']]} | `{spec['condition_id']}` | "
            + " | ".join(cells)
            + f" | {best} |"
        )

    lines.extend(
        [
            "",
            "## 7. 连续响应指标（全部条件）",
            "",
            "| 模型 | response velocity RMS mean/P90 | max |roll| mean/P90 (rad) | max |pitch| mean/P90 (rad) |",
            "|---|---:|---:|---:|",
        ]
    )
    for label in MODELS:
        value = overall[label]
        lines.append(
            f"| {MODELS[label]['name']} | {number(value['response_velocity_rms']['mean'])}/"
            f"{number(value['response_velocity_rms']['p90'])} | "
            f"{number(value['max_roll']['mean'])}/{number(value['max_roll']['p90'])} | "
            f"{number(value['max_pitch']['mean'])}/{number(value['max_pitch']['p90'])} |"
        )

    lines.extend(
        [
            "",
            "## 8. 指标口径",
            "",
            "- **完整时域存活**：扰动阶段及释放后 10 s 内均没有触发原生 termination。",
            "- **P5**：释放后 10 s 内、最多 5 次 recovery touchdown，进入严格稳定窗口。",
            "- **成功样本恢复步数**：只对 `outcome == SUCCESS` 的 `practical_enter_step` 求统计；TIMEOUT/FALL 不进入分母。",
            f"- **失败感知步数**：SUCCESS 使用实际 1--5 步，TIMEOUT/FALL 记为 {FAILURE_STEP}；"
            "用于避免低成功率模型因为只留下少量容易成功的样本而显得恢复更快。",
            "- **注意**：失败感知 6 步是明确的有限惩罚编码，不代表失败真的在第 6 步恢复。"
            "若只关心成功概率，应优先看 P5；若只关心不摔倒，应优先看完整时域存活率。",
            "",
            "## 9. 数据文件",
            "",
            f"- 原始 JSON/日志：`{RESULT_DIR.relative_to(PROJECT)}/`",
            f"- 分析脚本：`{Path(__file__).resolve().relative_to(PROJECT)}`",
        ]
    )

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
