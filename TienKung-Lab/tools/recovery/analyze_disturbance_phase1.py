#!/usr/bin/env python3
"""Aggregate the paired all-model Phase-1 disturbance suite."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

from analyze_all_trained_models_comparison import MODELS


TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
INPUT = TOOLS / "generated/all_models_disturbance_phase1_2026-09-01"
OUTPUT = TOOLS / "ALL_MODELS_DISTURBANCE_PHASE1.md"
SEEDS = (42, 123, 2026)
FAMILIES = (
    "velocity_ood",
    "force_pulse",
    "constant_force",
    "repeated_impulse",
    "random_force",
    "wrench_pulse",
)
FAMILY_NAMES = {
    "velocity_ood": "OOD瞬时Δv",
    "force_pulse": "有限时长外力脉冲",
    "constant_force": "5 s恒定外力",
    "repeated_impulse": "8 s重复冲击",
    "random_force": "10 s OU随机外力",
    "wrench_pulse": "外力+力矩脉冲",
}
PERSISTENT_FAMILIES = FAMILIES[1:]
EXPECTED_EPISODES = 4480
REPORT_STEMS = {
    "baseline_original": "baseline_original_curriculum",
    "baseline_shared020": "baseline_shared020_curriculum",
    "baseline_no_curriculum": "baseline_original_no_curriculum",
    "ours_shared_cert020": "ours_shared_cert020_curriculum",
    "ours_cert015": "ours_cert015_curriculum",
    "ours_cert020": "ours_cert020_curriculum",
    "ours_cert025": "ours_cert025_curriculum",
    "ours_cert050": "ours_cert050_curriculum",
    "ours_cert020_no_curriculum": "ours_cert020_no_curriculum",
    "dwaq_flat_new": "dwaq_flat_new",
    "dwaq_old": "dwaq_old",
}


def pct(value):
    return "--" if value is None else f"{100.0 * value:.2f}%"


def number(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def pvalue(value):
    if value < 1.0e-300:
        return "<1e-300"
    if value < 1.0e-3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {key: None for key in ("mean", "median", "p75", "p90", "max")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def wilson(successes, total):
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
    ) / denominator
    return center - half, center + half


def load_reports():
    reports = {}
    for label, model in MODELS.items():
        reports[label] = {}
        for seed in SEEDS:
            path = INPUT / f"{REPORT_STEMS[label]}_seed{seed}.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            if Path(report["checkpoint"]).resolve() != model["checkpoint"].resolve():
                raise ValueError(f"checkpoint mismatch: {path}")
            if report["policy"] != model["policy"]:
                raise ValueError(f"policy mismatch: {path}")
            if (
                report["planned_episode_count"] != EXPECTED_EPISODES
                or report["completed_episode_count"] != EXPECTED_EPISODES
                or report["pending_episode_count"] != 0
            ):
                raise ValueError(f"incomplete report: {path}")
            if report["common_protocol"]["seed"] != seed:
                raise ValueError(f"seed mismatch: {path}")
            reports[label][seed] = report
    for seed in SEEDS:
        seed_reports = [reports[label][seed] for label in MODELS]
        hashes = {item["common_protocol"]["trial_plan_sha256"] for item in seed_reports}
        trial_sets = [{episode["trial_id"] for episode in item["episodes"]} for item in seed_reports]
        if len(hashes) != 1 or any(value != trial_sets[0] for value in trial_sets[1:]):
            raise ValueError(f"unpaired trial plans for seed {seed}")
    return reports


def episodes(reports, label, *, family=None, condition=None, seed=None):
    selected = []
    for current_seed in ((seed,) if seed is not None else SEEDS):
        for item in reports[label][current_seed]["episodes"]:
            if family is not None and item["family"] != family:
                continue
            if condition is not None and item["condition_id"] != condition:
                continue
            selected.append({**item, "seed": current_seed})
    return selected


def performance(items):
    counts = Counter(item["outcome"] for item in items)
    successes = [item for item in items if item["outcome"] == "SUCCESS"]
    return {
        "n": len(items),
        "success": counts["SUCCESS"],
        "timeout": counts["TIMEOUT"],
        "fall": counts["FALL"],
        "p5": counts["SUCCESS"] / len(items),
        "survival": np.mean([item["survived_disturbance"] for item in items]),
        "hold": np.mean([item["functional_hold"] for item in items]),
        "steps": quantiles([item["practical_enter_step"] for item in successes]),
        "time": quantiles([item["post_release_recovery_time_s"] for item in successes]),
        "response_rms": quantiles(
            [item["response_velocity_error_rms"] for item in items if item["response_velocity_error_rms"] is not None]
        ),
        "tilt_peak": quantiles(
            [max(item["max_abs_roll"], item["max_abs_pitch"]) for item in items]
        ),
        "displacement": quantiles([item["max_com_displacement"] for item in items]),
        "contact": quantiles([item["max_contact_force"] for item in items]),
        "slip": quantiles([item["foot_slip_distance"] for item in items]),
        "max_action": quantiles([item["max_abs_action"] for item in items]),
        "action_clip": quantiles(
            [item["action_saturation_fraction"] for item in items if item["action_saturation_fraction"] is not None]
        ),
    }


def keyed(reports, label, family=None):
    return {
        (item["seed"], item["trial_id"]): item
        for item in episodes(reports, label, family=family)
    }


def paired(reports, first, second, family=None):
    first_items = keyed(reports, first, family)
    second_items = keyed(reports, second, family)
    if first_items.keys() != second_items.keys():
        raise ValueError(f"unpaired data: {first}, {second}, {family}")

    def exact_binary(field):
        first_only = sum(bool(a[field]) and not bool(second_items[key][field]) for key, a in first_items.items())
        second_only = sum(not bool(a[field]) and bool(second_items[key][field]) for key, a in first_items.items())
        discordant = first_only + second_only
        p = stats.binomtest(first_only, discordant, 0.5).pvalue if discordant else 1.0
        return first_only, second_only, float(p)

    success_first = {
        key: {**item, "binary_success": item["outcome"] == "SUCCESS"}
        for key, item in first_items.items()
    }
    success_second = {
        key: {**item, "binary_success": item["outcome"] == "SUCCESS"}
        for key, item in second_items.items()
    }
    original_first, original_second = first_items, second_items
    first_items, second_items = success_first, success_second
    success_only_a, success_only_b, success_p = exact_binary("binary_success")
    first_items, second_items = original_first, original_second
    survival_only_a, survival_only_b, survival_p = exact_binary("survived_disturbance")

    joint = [
        (item, second_items[key])
        for key, item in first_items.items()
        if item["outcome"] == "SUCCESS" and second_items[key]["outcome"] == "SUCCESS"
    ]
    step_diff = np.asarray(
        [a["practical_enter_step"] - b["practical_enter_step"] for a, b in joint],
        dtype=np.float64,
    )
    response_pairs = [
        (a["response_velocity_error_rms"], second_items[key]["response_velocity_error_rms"])
        for key, a in first_items.items()
        if a["response_velocity_error_rms"] is not None
        and second_items[key]["response_velocity_error_rms"] is not None
    ]
    response_diff = np.asarray([a - b for a, b in response_pairs], dtype=np.float64)

    def signed_rank(values):
        if not values.size or np.allclose(values, 0.0):
            return 1.0
        return float(stats.wilcoxon(values).pvalue)

    p5_a = np.mean([item["outcome"] == "SUCCESS" for item in first_items.values()])
    p5_b = np.mean([item["outcome"] == "SUCCESS" for item in second_items.values()])
    survival_a = np.mean([item["survived_disturbance"] for item in first_items.values()])
    survival_b = np.mean([item["survived_disturbance"] for item in second_items.values()])
    return {
        "delta_p5": p5_a - p5_b,
        "success_only_a": success_only_a,
        "success_only_b": success_only_b,
        "success_p": success_p,
        "delta_survival": survival_a - survival_b,
        "survival_only_a": survival_only_a,
        "survival_only_b": survival_only_b,
        "survival_p": survival_p,
        "joint_success": len(joint),
        "step_delta": float(np.mean(step_diff)) if step_diff.size else None,
        "step_p": signed_rank(step_diff),
        "response_rms_delta": float(np.mean(response_diff)) if response_diff.size else None,
        "response_rms_p": signed_rank(response_diff),
        "response_joint": len(response_diff),
    }


def main():
    reports = load_reports()
    overall = {label: performance(episodes(reports, label)) for label in MODELS}
    by_family = {
        label: {family: performance(episodes(reports, label, family=family)) for family in FAMILIES}
        for label in MODELS
    }
    balanced = {
        label: float(np.mean([by_family[label][family]["p5"] for family in FAMILIES]))
        for label in MODELS
    }
    persistent_survival = {
        label: float(np.mean([by_family[label][family]["survival"] for family in PERSISTENT_FAMILIES]))
        for label in MODELS
    }
    persistent_hold = {
        label: float(np.mean([by_family[label][family]["hold"] for family in PERSISTENT_FAMILIES]))
        for label in MODELS
    }
    seed_balanced = {
        label: {
            seed: float(
                np.mean(
                    [
                        performance(episodes(reports, label, family=family, seed=seed))["p5"]
                        for family in FAMILIES
                    ]
                )
            )
            for seed in SEEDS
        }
        for label in MODELS
    }
    ranking = sorted(MODELS, key=lambda label: balanced[label], reverse=True)
    pure_labels = ("ours_cert015", "ours_cert020", "ours_cert025", "ours_cert050")
    best_pure = max(pure_labels, key=lambda label: balanced[label])
    family_gain = {
        family: by_family[best_pure][family]["p5"] - by_family["baseline_original"][family]["p5"]
        for family in FAMILIES
    }
    best_gain_family = max(FAMILIES, key=lambda family: family_gain[family])

    key_pairs = [
        ("baseline_shared020", "baseline_original", "shared reward 对原始 Baseline"),
        ("ours_shared_cert020", "baseline_shared020", "旧 shared+certificate 对 shared Baseline"),
        (best_pure, "baseline_original", "最佳 pure-certificate 对原始 Baseline"),
        ("ours_cert020_no_curriculum", "baseline_no_curriculum", "无课程 certificate 直接消融"),
        ("ours_cert020", "ours_cert015", "certificate 0.20 对 0.15"),
        ("ours_cert025", "ours_cert020", "certificate 0.25 对 0.20"),
        ("ours_cert020", "ours_cert050", "certificate 0.20 对 0.50"),
        ("dwaq_flat_new", "baseline_original", "新 DWAQ 对原始 Baseline"),
    ]
    pair_results = {(a, b): paired(reports, a, b) for a, b, _ in key_pairs}
    family_pair_specs = (
        (best_pure, "baseline_original", "最佳 pure-certificate（有课程）"),
        (
            "ours_cert020_no_curriculum",
            "baseline_no_curriculum",
            "certificate 0.20（无课程）",
        ),
    )
    family_pair_results = {
        (first, second, family): paired(reports, first, second, family)
        for first, second, _ in family_pair_specs
        for family in FAMILIES
    }

    top = ranking[0]
    lines = [
        "# G1 全模型持续外力/冲击抗扰动测试：Phase 1",
        "",
        "更新时间：2026-09-01（Asia/Shanghai）",
        "",
        "> 训练内 `[-1,1]` 扰动的固定协议结果见 "
        "[`ALL_TRAINED_MODEL_COMPREHENSIVE_COMPARISON.md`](ALL_TRAINED_MODEL_COMPREHENSIVE_COMPARISON.md)；"
        "本报告只回答新的持续力、重复冲击、随机力、力矩和 OOD Δv 鲁棒性。",
        "",
        "## 1. 结论摘要",
        "",
        f"1. 六类扰动等权后的第一名是 **{MODELS[top]['name']}**，family-balanced P5 为 {pct(balanced[top])}。",
        f"2. pure-certificate 权重中第一名是 **{MODELS[best_pure]['name']}**，但其总体 P5 比原始有课程 Baseline "
        f"低 {100 * (overall['baseline_original']['p5'] - overall[best_pure]['p5']):.2f} pp；"
        f"唯一显著的正向 family 是 **{FAMILY_NAMES[best_gain_family]}**（{100 * family_gain[best_gain_family]:+.2f} pp）。",
        "3. 无课程的 certificate 0.20 对无课程 Baseline 在重复冲击和 OU 随机外力上分别提高 "
        f"{100 * (by_family['ours_cert020_no_curriculum']['repeated_impulse']['p5'] - by_family['baseline_no_curriculum']['repeated_impulse']['p5']):.2f} pp 和 "
        f"{100 * (by_family['ours_cert020_no_curriculum']['random_force']['p5'] - by_family['baseline_no_curriculum']['random_force']['p5']):.2f} pp，"
        "但总体 P5 仍更低，且持续生存率略有下降。",
        f"4. 持续扰动期间生存率最高的是 **{MODELS[max(MODELS, key=persistent_survival.get)]['name']}**"
        f"（五类外力 family 等权 {pct(max(persistent_survival.values()))}）；严格在线保持率最高的是 "
        f"**{MODELS[max(MODELS, key=persistent_hold.get)]['name']}**（{pct(max(persistent_hold.values()))}）。",
        "5. 本阶段只做 inference；reward、certificate/LIPM solver、curriculum 均未在评测中运行。"
        "所以这里测的是训练后策略能力，不是测试时 certificate 帮机器人决策。",
        "",
        "## 2. 模型与 checkpoint",
        "",
        "| 缩写 | 模型 | 精确 checkpoint | 训练身份 |",
        "|---|---|---|---|",
    ]
    for label, model in MODELS.items():
        relative = model["checkpoint"].resolve().relative_to(ROOT.resolve())
        lines.append(
            f"| {model['short']} | {model['name']} | `{relative}` | {model['training']} |"
        )
    lines.extend(
        [
            "",
            "## 3. 协议与完整性",
            "",
            "- 每模型每 seed 4480 条：OOD Δv 768、外力脉冲 1152、恒力 640、重复冲击 768、OU 随机力 768、力矩脉冲 384；",
            "- seed=42/123/2026；每模型 13440 条，11 模型共 147840 条；",
            "- 相同 seed 内所有模型使用相同 trial ID、command、方向、扰动波形和 plan hash；",
            "- Δv 为 world-frame x/y 分量；外力施加于 `torso_link`，按机器人真实总质量归一化；",
            "- flat plane，关闭 observation noise 与 physics randomization；正式运行统一使用256 env；",
            "- 持续力施加期间不判定恢复成功；释放后重新计 recovery touchdown，最多5步/10 s；",
            "- `survival`：扰动释放前不倒；`hold`：扰动末端最多1 s内满足原 practical 窗口；`P5`：释放后5次 touchdown 内成功；",
            "- peak/动作指标窗口为扰动开始至 outcome；`disturbance_velocity_error` 只覆盖外力有效期。",
            "",
            "| Seed | 每模型 planned/completed/pending | trial plan hash | 11模型 nominal reset |",
            "|---:|---:|---|---:|",
        ]
    )
    for seed in SEEDS:
        sample = reports[next(iter(MODELS))][seed]
        resets = sum(reports[label][seed]["nominal_reset_count"] for label in MODELS)
        lines.append(
            f"| {seed} | 4480 / 4480 / 0 | `{sample['common_protocol']['trial_plan_sha256']}` | {resets} |"
        )

    lines.extend(
        [
            "",
            "## 4. 总体排名",
            "",
            "总体 P5 按设计样本数加权；balanced P5 对六个 family 等权。CI 为总体 P5 的 Wilson 95%。",
            "",
            "| 排名 | 模型 | SUCCESS/TIMEOUT/FALL | 总体P5 [95% CI] | balanced P5 | 持续生存 | 在线hold | 成功步数 mean | median/P75/P90 | 成功时间 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, label in enumerate(ranking, 1):
        item = overall[label]
        lo, hi = wilson(item["success"], item["n"])
        lines.append(
            f"| {rank} | {MODELS[label]['name']} | {item['success']}/{item['timeout']}/{item['fall']} | "
            f"{pct(item['p5'])} [{pct(lo)}, {pct(hi)}] | {pct(balanced[label])} | "
            f"{pct(persistent_survival[label])} | {pct(persistent_hold[label])} | {number(item['steps']['mean'])} | "
            f"{number(item['steps']['median'], 1)}/{number(item['steps']['p75'], 1)}/{number(item['steps']['p90'], 1)} | "
            f"{number(item['time']['mean'])} s |"
        )

    lines.extend(
        [
            "",
            "### 4.1 family-balanced P5 的跨 seed 稳定性",
            "",
            "| 模型 | seed42 | seed123 | seed2026 | population SD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in MODELS:
        values = [seed_balanced[label][seed] for seed in SEEDS]
        lines.append(
            f"| {MODELS[label]['short']} | {pct(values[0])} | {pct(values[1])} | {pct(values[2])} | "
            f"{100 * np.std(values):.2f} pp |"
        )

    for title, field in (
        ("逐 family P5", "p5"),
        ("逐 family 扰动期间 survival", "survival"),
        ("逐 family 在线 functional hold", "hold"),
    ):
        lines.extend(
            [
                "",
                f"## {5 if field == 'p5' else 6 if field == 'survival' else 7}. {title}",
                "",
                "| 模型 | " + " | ".join(FAMILY_NAMES[family] for family in FAMILIES) + " |",
                "|---|" + "---:|" * len(FAMILIES),
            ]
        )
        for label in MODELS:
            values = [
                "--" if field == "hold" and family == "velocity_ood" else pct(by_family[label][family][field])
                for family in FAMILIES
            ]
            lines.append(f"| {MODELS[label]['short']} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## 8. 响应质量与安全代理指标",
            "",
            "以下均为每 episode 指标的 median；tilt peak=max(|roll|,|pitch|)。COM位移在机器人 heading frame 统计；"
            "action 是 actor 原始输出（关节目标偏移还会乘0.25），clip 阈值为 |a|≥99。",
            "",
            "| 模型 | response速度RMS | tilt peak(rad) | COM位移(m) | 峰值足端接触力(N) | 足端滑移(m) | max abs(action) | clip比例 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label in MODELS:
        item = overall[label]
        lines.append(
            f"| {MODELS[label]['short']} | {number(item['response_rms']['median'])} | "
            f"{number(item['tilt_peak']['median'])} | {number(item['displacement']['median'])} | "
            f"{number(item['contact']['median'], 1)} | {number(item['slip']['median'])} | "
            f"{number(item['max_action']['median'])} | {pct(item['action_clip']['mean'])} |"
        )

    lines.extend(
        [
            "",
            "## 9. 主要相同-trial配对检验",
            "",
            "success/survival 使用双侧 exact McNemar；成功双方的步数和全 trial response RMS 使用双侧 Wilcoxon。delta 均为 A-B。",
            "",
            "| 对比 | A | B | ΔP5 | A-only/B-only | P5 p | Δsurvival | survival p | joint success | Δstep | step p | RMS joint | Δresponse RMS | RMS p |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for first, second, description in key_pairs:
        item = pair_results[(first, second)]
        lines.append(
            f"| {description} | {MODELS[first]['short']} | {MODELS[second]['short']} | "
            f"{100 * item['delta_p5']:+.2f} pp | {item['success_only_a']}/{item['success_only_b']} | "
            f"{pvalue(item['success_p'])} | {100 * item['delta_survival']:+.2f} pp | "
            f"{pvalue(item['survival_p'])} | {item['joint_success']} | {number(item['step_delta'])} | "
            f"{pvalue(item['step_p'])} | {item['response_joint']} | {number(item['response_rms_delta'])} | "
            f"{pvalue(item['response_rms_p'])} |"
        )

    lines.extend(
        [
            "",
            "### 9.1 certificate 相对同类 Baseline 的逐 family 配对检验",
            "",
            "这里把有课程和无课程的 clean ablation 分开；delta 仍为 certificate-Baseline。"
            " P5 p 与 survival p 均为双侧 exact McNemar。",
            "",
            "| 对比 | Family | ΔP5 | P5 p | Δsurvival | survival p | Δ成功步数 | step p | Δresponse RMS | RMS p |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for first, second, description in family_pair_specs:
        for family in FAMILIES:
            item = family_pair_results[(first, second, family)]
            lines.append(
                f"| {description} | {FAMILY_NAMES[family]} | {100 * item['delta_p5']:+.2f} pp | "
                f"{pvalue(item['success_p'])} | {100 * item['delta_survival']:+.2f} pp | "
                f"{pvalue(item['survival_p'])} | {number(item['step_delta'])} | "
                f"{pvalue(item['step_p'])} | {number(item['response_rms_delta'])} | "
                f"{pvalue(item['response_rms_p'])} |"
            )

    condition_specs = reports[next(iter(MODELS))][SEEDS[0]]["condition_specs"]
    lines.extend(
        [
            "",
            "## 10. 全32个条件的P5附录",
            "",
            "每格合并3个 seed；OOD 条件每模型768条/条件，其他条件384条/条件。",
            "",
            "| Family / condition | " + " | ".join(MODELS[label]["short"] for label in MODELS) + " |",
            "|---|" + "---:|" * len(MODELS),
        ]
    )
    for condition in condition_specs:
        family = condition["family"]
        condition_id = condition["condition_id"]
        values = [
            pct(performance(episodes(reports, label, family=family, condition=condition_id))["p5"])
            for label in MODELS
        ]
        lines.append(
            f"| {FAMILY_NAMES[family]} / `{condition_id}` | " + " | ".join(values) + " |"
        )

    lines.extend(
        [
            "",
            "## 11. 解读边界与后续拓展依据",
            "",
            "- 本阶段的“持续外力”是质量归一化后的等效水平加速度，不把不同模型/环境质量差异误当成能力差异。",
            "- OOD Δv 的 square bound 是每个 x/y 分量的采样边界，不是向量模长固定值；分析时不可把 `s=2` 解释成恒定2 m/s。",
            "- functional hold 使用训练课程原有 practical 阈值，严格且不等于“不倒”；应与 survival 和释放后 P5 同看。",
            "- max contact force、slip 和 action 是仿真安全代理指标，不直接等价于真机冲击载荷、摩擦裕量或电机饱和。",
            "- 是否值得继续扩展 certificate，应优先看：相对原始 Baseline 的相同-trial P5/survival 增益是否集中在持续/重复扰动，"
            "以及这种增益是否伴随更大的 tilt、slip 或动作需求；不能只看总体平均成功率。",
            "",
            "## 12. 原始数据与复现",
            "",
            "- 原始 JSON/log：[`generated/all_models_disturbance_phase1_2026-09-01`](generated/all_models_disturbance_phase1_2026-09-01)",
            "- 评测器：[`evaluate_policy_disturbance_suite.py`](evaluate_policy_disturbance_suite.py)",
            "- 启动器：[`run_all_models_disturbance_phase1.sh`](run_all_models_disturbance_phase1.sh)",
            "- 汇总器：[`analyze_disturbance_phase1.py`](analyze_disturbance_phase1.py)",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
