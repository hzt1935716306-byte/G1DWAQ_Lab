#!/usr/bin/env python3
"""Generate one report covering every formal final policy evaluated so far."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

import analyze_five_policy_no_shared_comparison as metrics


ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
HISTORICAL = TOOLS / "generated/three_policy_comprehensive_final"
CURRENT = TOOLS / "generated/five_policy_no_shared_comprehensive_2026-09-01"
OUTPUT = TOOLS / "ALL_TRAINED_MODEL_COMPREHENSIVE_COMPARISON.md"
SEEDS = metrics.SEEDS
LEVELS = metrics.LEVELS

MODELS = {
    "baseline_original": {
        "name": "Baseline-original（有课程）",
        "short": "B-Orig-C",
        "family": "Symmetric",
        "policy": "baseline",
        "report_dir": HISTORICAL,
        "report_stem": "baseline",
        "checkpoint": ROOT / "logs/g1_flat_symmetric/2026-08-30_02-35-28_stage2_baseline_original_from4999/model_14998.pt",
        "training": "有课程；原始 locomotion；无 shared、无 certificate",
    },
    "baseline_shared020": {
        "name": "Baseline-shared-0.2（有课程）",
        "short": "B-Shared02-C",
        "family": "Symmetric",
        "policy": "baseline",
        "report_dir": HISTORICAL,
        "report_stem": "baseline_shared02",
        "checkpoint": ROOT / "logs/g1_flat_symmetric/2026-08-29_13-15-09_stage2_baseline_scale02_solverfix_resume/model_14999.pt",
        "training": "有课程；三个 shared events，scale=0.2；无 certificate",
    },
    "baseline_no_curriculum": {
        "name": "Baseline-original（无课程）",
        "short": "B-Orig-NC",
        "family": "Symmetric",
        "policy": "baseline",
        "report_dir": CURRENT,
        "report_stem": "baseline_no_curriculum",
        "checkpoint": ROOT / "logs/g1_flat_symmetric/2026-08-31_23-48-44_stage2_baseline_no_curriculum_from4999/model_9998.pt",
        "training": "无课程；完整L6随机扰动；原始 locomotion reward",
    },
    "ours_shared_cert020": {
        "name": "Ours-shared+cert-0.2（有课程）",
        "short": "O-Shared+Cert02-C",
        "family": "Symmetric",
        "policy": "ours",
        "report_dir": HISTORICAL,
        "report_stem": "ours02",
        "checkpoint": ROOT / "logs/g1_flat_symmetric/2026-08-29_13-12-16_stage2_ours_scale02_solverfix_resume/model_14999.pt",
        "training": "有课程；三个 shared events + certificate，scale=0.2；曾从7400恢复并重置课程",
    },
    "ours_cert015": {
        "name": "Ours-cert-only-0.15（有课程）",
        "short": "O-Cert015-C",
        "family": "Symmetric",
        "policy": "ours",
        "report_dir": CURRENT,
        "report_stem": "ours015_curriculum",
        "checkpoint": ROOT / "logs/our0.15_model_14998_no_sharereware.pt",
        "training": "有课程；certificate-only，scale=0.15（身份来自用户记录）",
    },
    "ours_cert020": {
        "name": "Ours-cert-only-0.20（有课程）",
        "short": "O-Cert020-C",
        "family": "Symmetric",
        "policy": "ours",
        "report_dir": CURRENT,
        "report_stem": "ours020_curriculum",
        "checkpoint": ROOT / "logs/g1_flat_symmetric/2026-08-31_21-43-19_stage2_ours_certonly020_resume_L5_from11700/model_14998.pt",
        "training": "有课程；certificate-only，scale=0.20；resume L5/570",
    },
    "ours_cert025": {
        "name": "Ours-cert-only-0.25（有课程）",
        "short": "O-Cert025-C",
        "family": "Symmetric",
        "policy": "ours",
        "report_dir": CURRENT,
        "report_stem": "ours025_curriculum",
        "checkpoint": ROOT / "logs/our0.25_model_14998_no_sharereward.pt",
        "training": "有课程；certificate-only，scale=0.25（身份来自用户记录）",
    },
    "ours_cert050": {
        "name": "Ours-cert-only-0.50（有课程）",
        "short": "O-Cert050-C",
        "family": "Symmetric",
        "policy": "ours",
        "report_dir": HISTORICAL,
        "report_stem": "ours",
        "checkpoint": ROOT / "logs/g1_flat_symmetric/2026-08-30_02-39-40_stage2_ours_cert050_async_from4999/model_14998.pt",
        "training": "有课程；certificate-only，scale=0.50",
    },
    "ours_cert020_no_curriculum": {
        "name": "Ours-cert-only-0.20（无课程）",
        "short": "O-Cert020-NC",
        "family": "Symmetric",
        "policy": "ours",
        "report_dir": CURRENT,
        "report_stem": "ours020_no_curriculum",
        "checkpoint": ROOT / "logs/g1_flat_symmetric/2026-08-31_23-44-06_stage2_ours_certonly020_no_curriculum_from4999/model_9998.pt",
        "training": "无课程；完整L6随机扰动；certificate-only，scale=0.20",
    },
    "dwaq_flat_new": {
        "name": "DWAQ-flat-new",
        "short": "DWAQ-New",
        "family": "DWAQ",
        "policy": "dwaq",
        "report_dir": HISTORICAL,
        "report_stem": "dwaq_flat_new",
        "checkpoint": ROOT / "logs/model_9999.pt",
        "training": "新纯平地 DWAQ；无 Stage2 certificate",
    },
    "dwaq_old": {
        "name": "DWAQ-old",
        "short": "DWAQ-Old",
        "family": "DWAQ",
        "policy": "dwaq",
        "report_dir": HISTORICAL,
        "report_stem": "dwaq",
        "checkpoint": ROOT / "logs/g1_dwaq/2026-01-16_00-46-00/model_9999.pt",
        "training": "旧 DWAQ 训练；历史参考",
    },
}


def load_reports():
    reports = {}
    for label, model in MODELS.items():
        reports[label] = {}
        for seed in SEEDS:
            path = model["report_dir"] / f"{model['report_stem']}_seed{seed}.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["policy"] != model["policy"]:
                raise ValueError(f"policy mismatch: {path}")
            if Path(report["checkpoint"]).resolve() != model["checkpoint"].resolve():
                raise ValueError(f"checkpoint mismatch: {path}")
            if (
                report["planned_episode_count"] != 1536
                or report["completed_episode_count"] != 1536
                or report["pending_episode_count"] != 0
            ):
                raise ValueError(f"incomplete report: {path}")
            if report["common_protocol"]["seed"] != seed:
                raise ValueError(f"seed mismatch: {path}")
            reports[label][seed] = report

    for seed in SEEDS:
        seed_reports = [reports[label][seed] for label in MODELS]
        hashes = {report["common_protocol"]["trial_plan_sha256"] for report in seed_reports}
        ids = [{episode["trial_id"] for episode in report["episodes"]} for report in seed_reports]
        if len(hashes) != 1 or any(current != ids[0] for current in ids[1:]):
            raise ValueError(f"trial mismatch for seed {seed}")
    return reports


def episodes(reports, label, **filters):
    return metrics._episodes(reports, label, **filters)


def performance(reports, label, **filters):
    return metrics._perf(episodes(reports, label, **filters))


def paired(reports, first, second, level=None):
    return metrics._paired(reports, first, second, level=level)


def pct(value):
    return metrics._percent(value)


def num(value, digits=3):
    return metrics._number(value, digits)


def pvalue(value):
    return metrics._p(value)


def sha(path):
    return metrics._file_sha256(path)[:12] + "…"


def main():
    reports = load_reports()
    reference = episodes(reports, "baseline_original")
    strengths = np.asarray(
        [np.linalg.norm(item["normalized_delta_xy"]) for item in reference], dtype=np.float64
    )
    metrics.STRENGTH_THRESHOLDS = np.quantile(strengths, (0.25, 0.50, 0.75))
    overall = {label: performance(reports, label) for label in MODELS}
    ranking = sorted(MODELS, key=lambda label: overall[label]["rate"], reverse=True)

    key_pairs = [
        ("baseline_shared020", "baseline_original", "shared events 对原始 Baseline"),
        ("ours_shared_cert020", "baseline_shared020", "旧混合 Ours 对 shared Baseline"),
        ("ours_cert025", "baseline_original", "最佳 pure-certificate 对原始 Baseline"),
        ("ours_cert020_no_curriculum", "baseline_no_curriculum", "无课程 pure-certificate 直接消融"),
        ("ours_cert020", "ours_cert015", "0.20 对 0.15"),
        ("ours_cert025", "ours_cert020", "0.25 对 0.20"),
        ("ours_cert020", "ours_cert050", "0.20 对 0.50"),
        ("ours_cert020", "ours_cert020_no_curriculum", "有课程/无课程 0.20（有混杂）"),
        ("baseline_shared020", "dwaq_flat_new", "最佳 Baseline 对新 DWAQ"),
        ("ours_cert025", "dwaq_flat_new", "最佳 pure-certificate 对新 DWAQ"),
    ]
    key = {(a, b): paired(reports, a, b) for a, b, _ in key_pairs}

    lines = [
        "# G1 Stage2 全部正式训练版本全面对比",
        "",
        "更新时间：2026-09-01（Asia/Shanghai）",
        "",
        "## 1. 范围与结论摘要",
        "",
        "本报告合并此前6个正式最终版本和本次5个版本，共11个模型。Stage1起点、smoke、"
        "未完成运行和每100轮中间 checkpoint 不作为独立方法版本。",
        "",
        f"1. 总体 P5 第一名是 **{MODELS[ranking[0]]['name']}**（{pct(overall[ranking[0]]['rate'])}），"
        f"第二名是 **{MODELS[ranking[1]]['name']}**（{pct(overall[ranking[1]]['rate'])}）；"
        f"二者差异不显著（paired McNemar p={pvalue(key[('baseline_shared020', 'dwaq_flat_new')]['mcnemar_p'])}）。",
        f"2. pure-certificate 第一名是 **Ours-cert-only-0.25（有课程）**（{pct(overall['ours_cert025']['rate'])}）；"
        f"它与原始有课程 Baseline 相差 {100 * key[('ours_cert025', 'baseline_original')]['delta_rate']:+.2f} pp，"
        f"差异不显著（p={pvalue(key[('ours_cert025', 'baseline_original')]['mcnemar_p'])}）。",
        f"3. 最干净的 certificate 直接证据来自无课程对照：Ours-0.20 比 Baseline 高 "
        f"{100 * key[('ours_cert020_no_curriculum', 'baseline_no_curriculum')]['delta_rate']:+.2f} pp "
        f"（p={pvalue(key[('ours_cert020_no_curriculum', 'baseline_no_curriculum')]['mcnemar_p'])}），但没有减少成功 trial 的踏步数。",
        f"4. shared events 对 Baseline 的 P5 提升为 "
        f"{100 * key[('baseline_shared020', 'baseline_original')]['delta_rate']:+.2f} pp；"
        "这是目前成功率最强的 reward 版本，但它不是纯 certificate 方法。",
        "5. 旧shared+certificate Ours排第三且成功平均步数较少，但它包含三个shared rewards，不能当作pure-certificate结果。",
        "6. 0.25 偏向困难扰动成功率，0.20 偏向成功后的少踏步；0.50 权重明显过大，0.15 也弱于0.20。",
        "7. 新 DWAQ 成功率很高且平均踏步少，但恢复时间明显慢于 symmetric policies；旧 DWAQ 仅作为历史失败参考。",
        "",
        "## 2. 全部模型身份",
        "",
        "| 模型 | Actor族/输入维度 | 精确 checkpoint | 训练身份 | SHA256 |",
        "|---|---:|---|---|---|",
    ]
    for label, model in MODELS.items():
        obs_dim = reports[label][SEEDS[0]]["actor_observation_shape"][-1]
        path = model["checkpoint"].resolve().relative_to(ROOT.resolve())
        lines.append(
            f"| {model['name']} | {model['family']} / {obs_dim} | `{path}` | {model['training']} | `{sha(model['checkpoint'])}` |"
        )
    lines.extend(
        [
            "",
            "0.15/0.25 的独立 `.pt` 没有配套 params YAML；其训练身份来自用户记录。旧 shared+certificate Ours "
            "在7400恢复时课程曾重置，因此不能作为纯 certificate 因果消融。",
            "",
            "## 3. 统一测试协议与完整性",
            "",
            "- seed：42、123、2026；L1--L6；每级每 seed 256 episode；",
            "- 每模型4608 episode；11模型合计 **50,688 episode**；",
            "- 每个 seed 内11模型拥有完全相同的 command、扰动、trial ID 和 trial-plan hash；",
            "- flat plane，关闭 observation noise 和 physics randomization；最多5次 touchdown/10 s；",
            "- inference-only；不计算训练 reward，不运行 certificate/LIPM solver；",
            "- DWAQ 保留100维原生 actor 输入，symmetric policies 保留960维原生输入。",
            "",
            "| Seed | 每模型 planned/completed/pending | trial hash | 11模型异常reset合计 |",
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
            "## 4. 11模型总体排名",
            "",
            "步数/成功时间只统计 SUCCESS；全episode时间包含TIMEOUT。P5 CI为Wilson 95%区间。",
            "",
            "| 排名 | 模型 | SUCCESS/TIMEOUT/FALL | P5 [95% CI] | 成功步数 mean | median/P75/P90 | 成功时间 | 全episode时间 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, label in enumerate(ranking, start=1):
        perf = overall[label]
        low, high = metrics._wilson(perf["success"], perf["n"])
        steps = perf["steps"]
        lines.append(
            f"| {rank} | {MODELS[label]['name']} | {perf['success']}/{perf['timeout']}/{perf['fall']} | "
            f"**{pct(perf['rate'])}** [{pct(low)}, {pct(high)}] | {num(steps['mean'])} | "
            f"{num(steps['median'], 1)}/{num(steps['p75'], 1)}/{num(steps['p90'], 1)} | "
            f"{num(perf['success_time']['mean'])} s | {num(perf['all_time']['mean'])} s |"
        )

    lines.extend(
        [
            "",
            "## 5. 跨 seed P5 稳定性",
            "",
            "| 模型 | seed42 | seed123 | seed2026 | population SD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in ranking:
        rates = [performance(reports, label, seed=seed)["rate"] for seed in SEEDS]
        lines.append(
            f"| {MODELS[label]['name']} | " + " | ".join(pct(value) for value in rates) + f" | {100*np.std(rates):.2f} pp |"
        )

    def wide_table(title, value_fn, suffix=""):
        lines.extend(
            [
                "",
                title,
                "",
                "| Level | " + " | ".join(MODELS[label]["short"] for label in MODELS) + " |",
                "|---|" + "---:|" * len(MODELS),
            ]
        )
        for level in LEVELS:
            values = [value_fn(label, level) + suffix for label in MODELS]
            lines.append(f"| L{level} | " + " | ".join(values) + " |")

    wide_table(
        "## 6. 逐等级严格P5",
        lambda label, level: pct(performance(reports, label, level=level)["rate"]),
    )
    wide_table(
        "## 7. 逐等级成功episode平均enter-step",
        lambda label, level: num(performance(reports, label, level=level)["steps"]["mean"]),
    )
    wide_table(
        "## 8. 逐等级成功episode平均恢复时间",
        lambda label, level: num(performance(reports, label, level=level)["success_time"]["mean"]),
        " s",
    )

    commands = [tuple(value) for value in reports["baseline_original"][SEEDS[0]]["common_protocol"]["commands"]]
    lines.extend(
        [
            "",
            "## 9. 分command严格P5",
            "",
            "| Command | " + " | ".join(MODELS[label]["short"] for label in MODELS) + " |",
            "|---|" + "---:|" * len(MODELS),
        ]
    )
    for command in commands:
        values = [pct(performance(reports, label, command=command)["rate"]) for label in MODELS]
        lines.append(f"| `{metrics._command_name(command)}` | " + " | ".join(values) + " |")

    thresholds = metrics.STRENGTH_THRESHOLDS
    lines.extend(
        [
            "",
            "## 10. 分归一化扰动强度四分位P5",
            "",
            f"边界：{thresholds[0]:.3f}/{thresholds[1]:.3f}/{thresholds[2]:.3f}。",
            "",
            "| 强度 | " + " | ".join(MODELS[label]["short"] for label in MODELS) + " |",
            "|---|" + "---:|" * len(MODELS),
        ]
    )
    for strength_bin in range(1, 5):
        values = [
            pct(performance(reports, label, strength_bin=strength_bin)["rate"])
            for label in MODELS
        ]
        lines.append(f"| Q{strength_bin} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## 11. 成功episode的enter-step分布",
            "",
            "| 模型 | 1步 | 2步 | 3步 | 4步 | 5步 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in ranking:
        distribution = overall[label]["step_distribution"]
        lines.append(
            f"| {MODELS[label]['name']} | " + " | ".join(str(distribution[step]) for step in range(1, 6)) + " |"
        )

    lines.extend(
        [
            "",
            "## 12. 主要假设的相同trial配对检验",
            "",
            "delta均为A-B。McNemar为双侧exact；step/time为双方都成功trial上的双侧Wilcoxon。",
            "",
            "| 对比 | A | B | delta P5 | A only/B only | McNemar p | joint | delta step | step p | delta time | time p |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for first, second, description in key_pairs:
        result = key[(first, second)]
        lines.append(
            f"| {description} | {MODELS[first]['short']} | {MODELS[second]['short']} | "
            f"{100*result['delta_rate']:+.2f} pp | {result['first_only']}/{result['second_only']} | "
            f"{pvalue(result['mcnemar_p'])} | {result['joint']} | {result['step_delta']:+.3f} | "
            f"{pvalue(result['step_p'])} | {result['time_delta']:+.3f} s | {pvalue(result['time_p'])} |"
        )

    all_pairs = list(itertools.combinations(MODELS, 2))
    all_results = [paired(reports, first, second) for first, second in all_pairs]
    holm = metrics._holm([result["mcnemar_p"] for result in all_results])
    lines.extend(
        [
            "",
            "## 13. 全55组模型对的P5配对附录",
            "",
            "Holm p对本节55个McNemar检验进行校正。完整step/time配对结果保留在汇总脚本中，主假设见上一节。",
            "",
            "| A | B | delta P5 | A only/B only | raw p | Holm p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for (first, second), result, adjusted in zip(all_pairs, all_results, holm):
        lines.append(
            f"| {MODELS[first]['short']} | {MODELS[second]['short']} | {100*result['delta_rate']:+.2f} pp | "
            f"{result['first_only']}/{result['second_only']} | {pvalue(result['mcnemar_p'])} | {pvalue(adjusted)} |"
        )

    lines.extend(
        [
            "",
            "## 14. 总体解释",
            "",
            "### 14.1 Reward结论",
            "",
            "- 三个shared events对严格P5的提升最稳定：shared Baseline是总体第一，但成功步数并未改善。",
            "- 旧shared+certificate Ours的P5低于shared Baseline，但双方成功时踏步更少；由于它同时含shared reward且课程恢复历史不同，不能代表纯certificate收益。",
            "- pure-certificate存在合理权重区间：0.15偏弱，0.20更强调少踏步，0.25更强调L5/L6/Q4成功率，0.50出现明显性能退化。",
            "- 无课程Ours/Baseline是最干净的certificate消融：certificate提高P5，但不减少成功trial踏步，并让恢复时间稍慢。",
            "",
            "### 14.2 哪些方向值得继续",
            "",
            "1. 若论文主张是困难扰动鲁棒性，优先扩展0.25，并主报L5/L6、Q4和TIMEOUT下降。",
            "2. 若主张是更少恢复踏步，保留0.20，并使用双方成功trial的配对enter-step；不要只报各自成功样本均值。",
            "3. 若希望得到最强总体P5，shared reward仍是当前最强工程方案，但它不能证明certificate理论本身有效。",
            "4. 新DWAQ应作为强参考基线：成功率接近第一且步数少，但其100维输入和较慢恢复时间必须单独说明。",
            "5. 后续应补一组相同训练预算、相同课程轨迹、多个训练seed的Baseline与pure-certificate0.25，才能做最干净的因果结论。",
            "",
            "### 14.3 解释限制",
            "",
            "- 有/无课程模型训练轮次和历史不同；不能把差异全部归因于课程。",
            "- 0.15/0.25缺少params YAML；身份依赖训练记录。",
            "- 固定协议关闭噪声和物理随机化，不是sim-to-real结论。",
            "- DWAQ与symmetric actor输入结构不同，只能比较最终物理trial表现。",
            "- 1步成功为0与practical-good-cycle判据需要完整touchdown interval有关，不能解释为动力学绝对不可能。",
            "",
            "## 15. 原始数据与复现",
            "",
            "历史6模型：[`generated/three_policy_comprehensive_final/`](generated/three_policy_comprehensive_final/)",
            "",
            "本次5模型：[`generated/five_policy_no_shared_comprehensive_2026-09-01/`](generated/five_policy_no_shared_comprehensive_2026-09-01/)",
            "",
            "统一汇总脚本：[`analyze_all_trained_models_comparison.py`](analyze_all_trained_models_comparison.py)",
            "",
        ]
    )
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
