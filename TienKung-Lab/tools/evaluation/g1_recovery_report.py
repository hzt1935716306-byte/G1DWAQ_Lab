"""Rebuild one bilingual Markdown/Word report from validated structured local results.

Word uses the standard OOXML container directly, avoiding a simulator or office
Python dependency. LibreOffice can render the resulting .docx for visual QA.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import io
import itertools
import json
import re
from pathlib import Path
import shutil
import struct
import zipfile
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

import numpy as np
import yaml

from g1_recovery_protocol import (
    COMPATIBILITY, TASKS, RunStore, atomic_write, code_identity, compatible, digest, evaluation_key, jsonl,
    load_prepared, load_yaml, lock, nominal_reference, read_json, read_jsonl, sha256, write_json,
)
from g1_recovery_metrics import METRICS_VERSION, RecoveryDetector, TrialMachine, paired_comparison, summarize


def fmt(value, percent=False):
    if value is None:
        return 'N/A'
    if isinstance(value, float):
        return f'{value * 100:.1f}%' if percent else f'{value:.3f}'
    return str(value)


def display_quantile(q):
    return f"{fmt(q['median'])} / {fmt(q['p90'])}; IQR={fmt(q['iqr'])} [{fmt(q['q1'])}, {fmt(q['q3'])}] (n={q['count']})"


def display_rate(s):
    interval = s['wilson_95']
    ci = f"[{fmt(interval[0], True)}, {fmt(interval[1], True)}]" if interval else 'N/A'
    return f"{s['numerator']}/{s['denominator']} {s['denominator_name']}; {fmt(s['rate'], True)}; Wilson95% {ci}"


def table(headers, rows):
    return ('table', headers, [[str(x) for x in row] for row in rows])


def is_full_lite_run(run, full_manifest):
    """Formal tables require the exact prepared 150 + 240 trial assignment."""
    manifest = run['manifest']
    return (run['identity'].get('checkpoint_stage') == 'final'
            and run['identity'].get('subset') == 'lite_full'
            and len(manifest) == 390
            and sum(row['experiment'] == 'E0' for row in manifest) == 150
            and sum(row['experiment'] == 'E1' for row in manifest) == 240
            and manifest == full_manifest)


def is_development_subset(run):
    return bool(re.fullmatch(r'first_[1-9][0-9]*_per_experiment', run['identity'].get('subset', '')))


def load_runs(root, formal_only=True):
    _, full_manifest, _ = load_prepared(root)
    runs, excluded, seen = [], [], set()
    for path in sorted((root / 'runs').glob('*')):
        if not path.is_dir() or path.name.startswith('.'):
            continue
        try:
            if not (path / 'completion.json').exists():
                status = read_json(path / 'summary.json').get('status', 'PARTIAL') if (path / 'summary.json').exists() else 'PARTIAL'
                _, partial_records = RunStore(path).validate(require_complete=False)
                partial = summarize(partial_records, read_jsonl(path / 'manifest_snapshot.jsonl'))
                s = partial['E1']
                excluded.append((path.name, status, f"未完成；不进入主表；E1 执行/指定 {s['executed']}/{s['designated']}，"
                                 f"成功 {s['successes']}，已执行失败 {s['failures_executed']}，pending {s['pending']}"))
                continue
            identity, records = RunStore(path).validate()
            if not identity.get('evaluation_runtime_sha256'):
                raise ValueError('Legacy result lacks evaluation_runtime_sha256; reevaluation required')
            if identity['method'].startswith('context') and identity.get('identity_schema_version', 1) < 2:
                raise ValueError('Legacy Plane result lacks SHA-bound native solver inputs; reevaluation required')
            content = sha256(path / 'completion.json')
            if content in seen:
                continue
            seen.add(content)
            manifest = read_jsonl(path / 'manifest_snapshot.jsonl')
            if 'actual_physics_hash' not in identity:
                raise ValueError('Missing measured physics profile')
            runs.append({'id': path.name, 'path': path, 'identity': identity, 'records': records,
                         'manifest': manifest, 'summary': summarize(records, manifest)})
        except (ValueError, KeyError, OSError) as exc:
            excluded.append((path.name, 'INVALID', str(exc)))
    # An explicit retest is retained in history, but never counted as additional samples.
    selected = {}
    for run in runs:
        key = (evaluation_key(run['identity']), run['identity']['actual_physics_hash'])
        def rank(item):
            completion = read_json(item['path'] / 'completion.json')
            return (item['identity'].get('attempt_index', 0), completion.get('completed_at_utc', ''), item['id'])
        if key not in selected or rank(run) > rank(selected[key]):
            selected[key] = run
    selected_runs = list(selected.values())
    return ([r for r in selected_runs if is_full_lite_run(r, full_manifest)]
            if formal_only else selected_runs), excluded, runs


def compatible_run(a, b):
    return compatible(a['identity'], b['identity']) and a['identity']['actual_physics_hash'] == b['identity']['actual_physics_hash']


def budget_relation(a, b):
    aa, bb = a['identity'].get('training_transitions'), b['identity'].get('training_transitions')
    if type(aa) is not int or type(bb) is not int or aa <= 0 or bb <= 0:
        return '预算未知（非等预算比较）'
    return '等预算' if aa == bb else f'不同预算（{aa} / {bb} transitions）'


def matched_ablation(a, b):
    seed = a['identity'].get('training_seed')
    return (compatible_run(a, b) and type(seed) is int and seed == b['identity'].get('training_seed')
            and a['identity'].get('checkpoint_stage') == b['identity'].get('checkpoint_stage') == 'final'
            and budget_relation(a, b) == '等预算')


def previous_checkpoint(run, history):
    i = run['identity']
    run_id, budget = i.get('training_run_id'), i.get('training_transitions')
    if not run_id or type(budget) is not int:
        return None
    candidates = [r for r in history if compatible_run(run, r)
                  and r['identity'].get('training_run_id') == run_id
                  and r['identity']['method'] == i['method']
                  and r['identity'].get('training_seed') == i.get('training_seed')
                  and type(r['identity'].get('training_transitions')) is int
                  and r['identity']['training_transitions'] < budget]
    return max(candidates, key=lambda r: (r['identity']['training_transitions'],
               r['identity'].get('attempt_index', 0), r['id'])) if candidates else None


def curve_plots(root, group, group_id):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths = []
    for name, xlabel in [('time', 'Recovery entry time (s)'), ('steps', 'New touchdowns to confirmed entry')]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        for ax, denominator in zip(axes, ('designated', 'pushed')):
            for run in group:
                points = run['summary']['curves'][name]
                y = [x[denominator] if x[denominator] is not None else np.nan for x in points]
                ax.step([x['x'] for x in points], y, where='post', label=run['identity']['model_alias'])
            ax.set(xlabel=xlabel, ylabel='Recovered AND survived fraction', ylim=(0, 1.02),
                   title=f'Denominator: {denominator} trials')
            ax.grid(alpha=0.25)
            ax.legend(fontsize=6)
        fig.tight_layout()
        path = root / 'report/figures' / f'{group_id}_{name}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = io.BytesIO()
        fig.savefig(stream, format='png', dpi=160)
        plt.close(fig)
        atomic_write(path, stream.getvalue())
        paths.append(path)
    return paths


def report_blocks(root, runs, excluded, history, plots=True):
    p, manifest, prepared = load_prepared(root)
    runs = [r for r in runs if is_full_lite_run(r, manifest)]
    development = [r for r in history if is_development_subset(r)]
    models = load_yaml(root / 'models.yaml')['models']
    blocks = [('heading', 1, 'G1 可重复评测实验记录'),
              ('heading', 2, '当前结论摘要'),
              ('text', f"正式 full-Lite final 评测：{len(runs)}；开发子集：{len(development)}；全部完成记录：{len(history)}。" +
               ('当前没有可进入正式主表的结果。' if not runs else
                '以下仅为当前 checkpoint 的观察结果，不代表多 seed 稳定优势或理论证明。')),
              ('heading', 2, '协议与版本'),
              ('text', f"{p['protocol_id']} / {p['protocol_version']}；指标 {p['metrics']['version']}。"),
              ('text', 'E0：−10°、0°、+10°，四方向 0.4 m/s 与 standing；每格 10 次，暖机 2 秒、记录 10 秒。'
               'E1：+x 0.4 m/s，四方向加性速度跳变 0.5/1.0 m/s，参考步间隔 0.25/0.75，每格 5 次。'),
              ('text', '恢复 entry 为首个达标的完整扰动后触地间隔结束；紧邻下一间隔也达标才确认。'
               '主 entry 截止 8 秒，确认须在 10 秒观察截止前完成，并存活到截止。该定义含完整间隔观测延迟。'
               '任务成功率分母为全部指定 E1；条件成功率分母为实际施加扰动 E1。Standing 单列。'),
              ('text', '每格参考脚实际 2/3 或 3/2，整体左右各 120；详情见 manifests/reference_foot_counts.json。'),
              ('text', '检测器实测验收未完成前保持开发版；DETECTOR_VALIDATION.md 记录验收状态。'),
              ('heading', 2, '模型清单')]
    rows = []
    partial_models = {}
    for path in (root / 'runs').glob('*/identity.json'):
        if not (path.parent / 'completion.json').exists():
            raw = read_json(path)
            status_path = path.parent / 'summary.json'
            status = read_json(status_path).get('status', 'PARTIAL') if status_path.exists() else 'PARTIAL'
            partial_models[raw['checkpoint_sha256']] = 'INVALID' if status == 'INVALID' else 'PARTIAL'
    for model in models:
        full_complete = [r for r in runs if r['identity']['checkpoint_sha256'] == model.get('checkpoint_sha256')]
        dev_complete = [r for r in development if r['identity']['checkpoint_sha256'] == model.get('checkpoint_sha256')]
        status = ('FULL_EVAL_COMPLETE' if full_complete else
                  partial_models.get(model.get('checkpoint_sha256')) or
                  ('DEV_SUBSET_COMPLETE' if dev_complete else 'PENDING_FULL_EVAL'))
        rows.append([model['model_alias'], model['task_name'], model.get('training_iteration', 'unknown'),
                     model.get('training_run_id') or '未知', fmt(model.get('training_transitions')),
                     model.get('training_seed', 'unknown'), model.get('checkpoint_stage', 'unknown'),
                     status, (model.get('checkpoint_sha256') or 'N/A')[:12]])
    blocks.append(table(['模型', 'Task', '迭代', '训练批次 ID', 'Transitions', 'Seed', '阶段', '状态', 'SHA 前缀'], rows))
    blocks += [('heading', 2, 'Development / Smoke Evaluation'),
               ('text', '开发子集仅用于流水线和检测器人工检查，不进入 E0/E1 正式主表、累计曲线、baseline 比较或内部消融排名。')]
    if development:
        blocks.append(table(['模型', '评测 ID', '阶段', 'Subset', 'E0 执行/指定', 'E1 执行/指定', '状态'], [[
            run['identity']['model_alias'], run['id'], run['identity'].get('checkpoint_stage', 'unknown'),
            run['identity']['subset'],
            f"{run['summary']['E0']['moving']['executed'] + run['summary']['E0']['standing']['executed']}/"
            f"{run['summary']['E0']['moving']['designated'] + run['summary']['E0']['standing']['designated']}",
            f"{run['summary']['E1']['executed']}/{run['summary']['E1']['designated']}", 'DEV_SUBSET_COMPLETE']
            for run in development]))
    else:
        blocks.append(('text', '尚无完成的开发子集。'))
    groups = defaultdict(list)
    for run in runs:
        i = run['identity']
        group_id = digest({k: i[k] for k in COMPATIBILITY} | {'actual_physics_hash': i['actual_physics_hash']})[:12]
        groups[group_id].append(run)
    blocks += [('heading', 2, 'E0 正常运动总表'), ('text', '主表仅包含显式 final 且完整执行 150 个 E0 + 240 个 E1 的 Lite 模型；开发子集和 intermediate/unknown 不参与。Moving 与 standing 分开；RMSE 单位 m/s，姿态波动单位 rad。')]
    if not runs:
        blocks.append(('text', '待评测：E0 每模型 150 个指定 trial。'))
    for gid, group in groups.items():
        blocks.append(('heading', 3, f'兼容组 {gid}'))
        rows = []
        for run in group:
            for label, stats in run['summary']['E0'].items():
                rows.append([run['identity']['model_alias'], fmt(run['identity'].get('training_transitions')),
                             label, f"{stats['executed']}/{stats['designated']}",
                             display_rate(stats['survival_statistics']), display_quantile(stats['root_xy_velocity_rmse']),
                             display_quantile(stats['com_xy_velocity_rmse']), display_quantile(stats['heading_drift']),
                             f"{display_quantile(stats['roll_std'])} / {display_quantile(stats['pitch_std'])}"])
        blocks.append(table(['模型', 'Transitions', '命令', '执行/指定', '存活率', 'Root RMSE', 'CoM RMSE', '航向漂移', 'Roll/Pitch σ'], rows))
        blocks.append(table(['模型', '命令', '触地数 median/P90', '步间隔秒 median/P90'],
                            [[r['identity']['model_alias'], label, display_quantile(stats['touchdown_count']),
                              display_quantile(stats['step_interval_s'])] for r in group for label, stats in r['summary']['E0'].items()]))
    blocks.append(('heading', 2, 'E1 抗扰动总表'))
    if not runs:
        blocks.append(('text', '待评测：E1 每模型 240 个指定 trial。无可排名数据。'))
    for gid, group in groups.items():
        blocks += [('heading', 3, f'兼容组 {gid}'), ('text', '各兼容组独立呈现；不同 manifest、指标、物理配置或推理模式不混排。')]
        rows, durations, extras = [], [], []
        for run in group:
            i, s = run['identity'], run['summary']['E1']
            rows.append([i['model_alias'], fmt(i.get('training_transitions')), f"{s['successes']}/{s['designated']} ({s['pushed']} pushed)",
                         fmt(s['precondition_pass_rate'], True), fmt(s['task_recovery_success_rate'], True),
                         fmt(s['conditional_recovery_success_rate'], True), fmt(s['pushed_10s_survival_rate'], True),
                         fmt(s['recovery_within_3_touchdowns'], True), fmt(s['recovery_within_5_touchdowns'], True)])
            durations.append([i['model_alias'], display_quantile(s['recovery_time']), display_quantile(s['recovery_steps']),
                              f"{s['successes']}/{s['designated']}", fmt(s['failure_rate'], True)])
            extras.append([i['model_alias'], display_quantile(s['peak_tilt']), display_quantile(s['peak_velocity_error']),
                           display_quantile(s['integrated_velocity_error']), display_quantile(s['heading_drift'])])
        blocks += [table(['模型', 'Transitions', '成功/指定 (已推)', '准备通过', '任务成功', '条件成功', '10s 存活', '≤3 次落脚', '≤5 次落脚'], rows),
                   table(['模型', '恢复秒 median/P90', '落脚数 median/P90', '成功/指定', '失败率'], durations),
                   table(['模型', '峰值倾角 rad', '峰值速度误差 m/s', '误差积分 m', '航向漂移 rad'], extras)]
        blocks.append(table(['模型', '比例指标', '分子/分母、比例及 Wilson 95% 区间'],
                            [[r['identity']['model_alias'], key, display_rate(value)] for r in group
                             for key, value in r['summary']['E1']['rates'].items()]))
    blocks.append(('heading', 2, '分坡度、方向、时相结果'))
    for gid, group in groups.items():
        for field in ('slope_deg', 'push_direction', 'target_phase'):
            rows = []
            for run in group:
                conditions = sorted({str(r[field]) for r in run['manifest'] if r['experiment'] == 'E1'})
                for condition in conditions:
                    plans = [r for r in run['manifest'] if r['experiment'] == 'E1' and str(r[field]) == condition]
                    records = [r for r in run['records'] if r['experiment'] == 'E1' and str(r[field]) == condition]
                    s = summarize(records, plans)['E1']
                    rows.append([run['identity']['model_alias'], condition, f"{s['successes']}/{s['designated']}",
                                 display_rate(s['rates']['task_recovery_success_rate']), display_rate(s['rates']['recovery_within_5_touchdowns'])])
            blocks += [('text', f'{gid} / {field}'), table(['模型', '条件', '成功/指定', '任务恢复率', '五次落脚内'], rows)]
    if not runs:
        blocks.append(('text', '待评测。'))
    blocks.append(('heading', 2, '恢复时间与落脚数累计曲线'))
    blocks.append(('text', 'F_T(t)、F_N(k) 仅把确认恢复且观察期存活计入分子；失败保留在指定试验分母。右图为已施加扰动条件分母。'))
    for gid, group in groups.items():
        if plots:
            for path in curve_plots(root, group, gid):
                blocks.append(('image', str(path), f'兼容组 {gid}：累计恢复且存活比例。'))
    if not runs:
        blocks.append(('text', '尚无真实轨迹，不绘制虚构曲线。'))
    blocks.append(('heading', 2, '内部消融链与自动配对比较'))
    blocks.append(('text', '固定消融链：rl_only → context_only → context_reward。'))
    for method in ('rl_only', 'context_only', 'context_reward'):
        found = [r for r in runs if r['identity']['method'] == method]
        blocks.append(('text', f"{method}：" + (', '.join(r['identity']['model_alias'] for r in found) if found else '待评测')))
    compare_rows = []
    for run in runs:
        i = run['identity']
        candidates = [r for r in runs if r is not run and compatible_run(run, r)]
        targets = []
        for baseline in ('ppo_plain', 'dwaq'):
            matches = [r for r in candidates if r['identity']['method'] == baseline]
            if matches:
                targets.extend((f'baseline:{baseline}；{budget_relation(run, r)}', r) for r in matches)
            elif i['method'] != baseline:
                blocks.append(('text', f"{i['model_alias']}：缺少兼容 {baseline} 结果。"))
        if i['method'] in ('rl_only', 'context_only', 'context_reward'):
            for method in ('rl_only', 'context_only', 'context_reward'):
                if method == i['method']:
                    continue
                matches = [r for r in candidates if r['identity']['method'] == method]
                targets.extend(('matched ablation（同 seed / final / 等预算）' if matched_ablation(run, r)
                                else '非控制变量消融；' + budget_relation(run, r), r) for r in matches)
                if not matches:
                    blocks.append(('text', f"{i['model_alias']}：缺少兼容的 {method}。"))
        for relation, other in targets:
            comp = paired_comparison(run['records'], other['records'], run['summary']['E1']['designated'])
            blocks.append(('text', f"当前 checkpoint {i['model_alias']} 相比 {other['identity']['model_alias']}：在 {run['summary']['E1']['designated']} 个指定 E1 trial 上，五次落脚内恢复率变化 {fmt(comp['five_touchdown_delta_pp'])} 个百分点；双方共同成功 {comp['paired_success_count']} 条，配对平均恢复时间变化 {fmt(comp['paired_mean_time_delta_s'])} 秒（{fmt(comp['paired_relative_time_delta'], True)}）。"))
            compare_rows.append([i['model_alias'], other['identity']['model_alias'], relation, fmt(comp['five_touchdown_delta_pp']),
                                 comp['paired_success_count'], fmt(comp['paired_mean_time_delta_s']), fmt(comp['paired_relative_time_delta'], True)])
    if compare_rows:
        blocks.append(table(['模型 A', '模型 B', '关系', '≤5 差 pp (A−B)', '共同成功 n', '时间差 s', '相对时间差'], compare_rows))
    blocks.append(('heading', 2, '各模型历次 checkpoint 的变化'))
    blocks.append(('text', '仅在 training_run_id、方法、seed 和评测协议一致时连接轨迹；未知训练身份或预算不自动配对。'))
    blocks.append(table(['模型', '训练批次 ID', '迭代', 'Transitions', 'Seed', '阶段', '评测/attempt', '是否主表', 'E1 成功/指定'],
                        [[r['identity']['model_alias'], r['identity'].get('training_run_id') or '未知',
                          r['identity']['training_iteration'], fmt(r['identity'].get('training_transitions')),
                          r['identity']['training_seed'], r['identity']['checkpoint_stage'], r['id'],
                          '是' if r in runs else '演进/历史',
                          f"{r['summary']['E1']['successes']}/{r['summary']['E1']['designated']}"] for r in history]))
    for run in history:
        previous = previous_checkpoint(run, history)
        if previous:
            blocks.append(('text', f"训练批次 {run['identity']['training_run_id']}：{previous['id']} → {run['id']}；"
                           f"{budget_relation(run, previous)}，仅展示训练演进。"))
    blocks.append(('heading', 2, 'Plane 专项诊断'))
    for run in runs:
        values = [r.get('plane_diagnostics') for r in run['records'] if r.get('plane_diagnostics')]
        blocks.append(('text', run['identity']['model_alias'] + ('：N/A（不创建 shadow certificate）。' if not values else
                       '：逐 trial context valid、standing N/A、heading/geometry invalid、N、margin、solver failure 与恢复奖励分量见 plane_diagnostics.json。')))
        if values:
            frame_count = sum(v.get('frames', 1) for v in values)
            available = all(v.get('diagnostics_available', False) for v in values)
            blocks.append(table(['模型', '诊断帧数', 'Context valid', 'Standing N/A', '无效 context 帧', '查询失败事件', '数值失败事件'], [[
                run['identity']['model_alias'], frame_count,
                fmt(sum(v['context_valid_rate'] * v.get('frames', 1) for v in values) / frame_count, True),
                fmt(sum(v['standing_na_rate'] * v.get('frames', 1) for v in values) / frame_count, True),
                sum(v['invalid_context_frames'] for v in values) if available else 'N/A（旧诊断证据不足）',
                sum(v['query_failures'] for v in values) if available else 'N/A（旧诊断证据不足）',
                sum(v['numerical_failure_events'] for v in values) if available else 'N/A（旧诊断证据不足）']]))
            write_json(root / 'report' / (run['id'] + '_plane_diagnostics.json'), values)
    if not runs:
        blocks.append(('text', 'Baseline / RL-only：N/A；两个 estimator Plane 模型：待评测。'))
    blocks += [('heading', 2, '失败案例与结果局限'),
               ('text', 'PRECONDITION_FAILED、FELL、OUT_OF_TEST_AREA 等物理终态保留；执行错误为 INVALID，缺失 trial 为 PARTIAL。'
                '发生过确认恢复但后来跌倒仍计主失败，确认事件保留。单 seed 只能说明当前 checkpoint 观察。'
                '两份 baseline 为检测器开发验证对象，不能据此称为独立最终测试集。')]
    if excluded:
        blocks.append(table(['评测', '状态', '排除原因'], excluded))
    blocks.append(('text', 'PARTIAL 不参与排名。指定分母中的 pending 单列，未执行不算失败；指定分母尚有 pending 时不计算 Wilson 区间，已执行/已推分母的区间仅描述当前样本。'))
    for run in runs:
        failures = [r for r in run['records'] if r['status'] not in ('RECOVERED_AND_SURVIVED', 'ALIVE_NOT_RECOVERED')]
        blocks.append(('text', f"{run['identity']['model_alias']}：失败记录 {len(failures)}，全部轨迹保存在 runs/{run['id']}/traces。"))
    blocks += [('heading', 2, '历史参考'), ('text', '旧 ALL_MODELS_PUSH_LIMITS.md、Gate B 和单间隔恢复报告属于不同协议；不导入本主表。'),
               ('heading', 2, '结果文件及 SHA 索引')]
    for k, value in prepared.items():
        blocks.append(('text', f'{k}: {value}'))
    for run in runs:
        blocks.append(('text', f"{run['id']}：runs/{run['id']}；completion SHA256={sha256(run['path'] / 'completion.json')}。完整文件 SHA 见 completion.json。"))
    blocks.append(('heading', 2, '人工备注'))
    notes = load_yaml(root / 'report/notes.yaml')
    blocks.append(('text', yaml.safe_dump(notes, allow_unicode=True, sort_keys=False)))
    return blocks


def markdown(blocks, report_dir):
    lines = []
    for b in blocks:
        if b[0] == 'heading':
            lines.append('#' * b[1] + ' ' + b[2])
        elif b[0] == 'text':
            lines.append(b[1])
        elif b[0] == 'table':
            clean = lambda x: str(x).replace('|', '\\|').replace('\n', '<br>')
            lines += ['| ' + ' | '.join(map(clean, b[1])) + ' |', '| ' + ' | '.join(['---'] * len(b[1])) + ' |']
            lines += ['| ' + ' | '.join(map(clean, row)) + ' |' for row in b[2]]
        elif b[0] == 'image':
            lines.append(f'![{b[2]}]({Path(b[1]).relative_to(report_dir)})\n\n{b[2]}')
        lines.append('')
    return '\n'.join(lines)


def table_widths(headers, weights=None):
    if weights is None:
        weights = ([1.7, 2.6, .6, 1.4, .8, .55, .65, 1.35, 1.0]
                   if headers[0:2] == ['模型', 'Task'] else [1] * len(headers))
    if len(weights) != len(headers):
        raise ValueError(f'Table column width count {len(weights)} does not match header count {len(headers)}')
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError('Table column weights must be positive')
    return [int(14500 * weight / sum(weights)) for weight in weights]


def word_document(blocks):
    """OOXML with landscape A4, fixed table width, CJK font, image captions and PAGE field."""
    ns = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    relationships = []
    media = []
    def paragraph(text, style=None):
        properties = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ''
        chunks = str(text).split('\n')
        return '<w:p>' + properties + '<w:r><w:t xml:space="preserve">' + '</w:t><w:br/><w:t xml:space="preserve">'.join(escape(x) for x in chunks) + '</w:t></w:r></w:p>'
    body = []
    for b in blocks:
        if b[0] == 'heading':
            body.append(paragraph(b[2], f'Heading{b[1]}'))
        elif b[0] == 'text':
            body.append(paragraph(b[1]))
        elif b[0] == 'table':
            widths = table_widths(b[1])
            rows = []
            for j, row in enumerate([b[1]] + b[2]):
                if len(row) != len(b[1]):
                    raise ValueError(f'Table row has {len(row)} cells but header has {len(b[1])}')
                cells = ''.join(f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/></w:tcPr>{paragraph(v)}</w:tc>' for v, width in zip(row, widths))
                rows.append('<w:tr>' + ('<w:trPr><w:tblHeader/></w:trPr>' if j == 0 else '') + cells + '</w:tr>')
            body.append('<w:tbl><w:tblPr><w:tblW w:w="14500" w:type="dxa"/><w:tblLayout w:type="fixed"/>'
                        '<w:tblBorders>' + ''.join(f'<w:{x} w:val="single" w:sz="4" w:color="BBBBBB"/>' for x in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']) +
                        '</w:tblBorders></w:tblPr><w:tblGrid>' + ''.join(f'<w:gridCol w:w="{width}"/>' for width in widths) + '</w:tblGrid>' + ''.join(rows) + '</w:tbl>')
        elif b[0] == 'image':
            index = len(media) + 1
            data = Path(b[1]).read_bytes()
            w, h = struct.unpack('>II', data[16:24])
            cx, cy = 8700000, int(8700000 * h / w)
            media.append((f'word/media/image{index}.png', data))
            relationships.append(f'<Relationship Id="img{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image{index}.png"/>')
            body.append(f'<w:p><w:r><w:drawing><wp:inline><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{index}" name="Curve {index}"/>'
                        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr>'
                        f'<pic:cNvPr id="{index}" name="curve.png"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="img{index}"/>'
                        '<a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/>'
                        f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
                        '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')
            body.append(paragraph(b[2], 'Caption'))
    body.append('<w:sectPr><w:footerReference w:type="default" r:id="footer"/><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/>'
                '<w:pgMar w:top="900" w:right="1100" w:bottom="900" w:left="1100" w:header="400" w:footer="400"/></w:sectPr>')
    document = f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    document += 'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>' + ''.join(body) + '</w:body></w:document>'
    styles = f'<w:styles xmlns:w="{ns}"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Liberation Sans" w:eastAsia="Noto Sans CJK SC"/><w:sz w:val="18"/></w:rPr></w:rPrDefault></w:docDefaults>'
    for i, size in [(1, 32), (2, 25), (3, 21)]:
        styles += f'<w:style w:type="paragraph" w:styleId="Heading{i}"><w:name w:val="heading {i}"/><w:pPr><w:keepNext/><w:spacing w:before="180" w:after="100"/></w:pPr><w:rPr><w:b/><w:sz w:val="{size}"/></w:rPr></w:style>'
    styles += '<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/><w:rPr><w:i/></w:rPr></w:style></w:styles>'
    rels = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + ''.join(relationships)
    rels += '<Relationship Id="styles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    rels += '<Relationship Id="footer" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/></Relationships>'
    types = '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'
    for part, content in [('document', 'document.main'), ('styles', 'styles'), ('footer1', 'footer')]:
        types += f'<Override PartName="/word/{part}.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.{content}+xml"/>'
    types += '</Types>'
    files = {'[Content_Types].xml': types,
             '_rels/.rels': '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="doc" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>',
             'word/document.xml': document, 'word/styles.xml': styles, 'word/_rels/document.xml.rels': rels,
             'word/footer1.xml': f'<w:ftr xmlns:w="{ns}"><w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:t>G1 Recovery Eval · </w:t></w:r><w:fldSimple w:instr="PAGE"/></w:p></w:ftr>'}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in list(files.items()) + media:
            z.writestr(name, data)
    return stream.getvalue()


def preserve_existing(report):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    meta_path = report / 'generated_files.json'
    prior = read_json(meta_path) if meta_path.exists() else {}
    notes = load_yaml(report / 'notes.yaml')
    changed = False
    for ext in ('md', 'docx'):
        path = report / f'G1_RECOVERY_EXPERIMENTS.{ext}'
        if not path.exists():
            continue
        dest = report / 'history' / stamp / path.name
        if ext == 'md':
            def archive_image(match):
                target = match.group(2).strip('<>')
                if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target):
                    return match.group(0)
                source = (report / target).resolve(strict=True)
                if not source.is_relative_to(report.resolve()):
                    raise ValueError('Historical Markdown image must be inside report directory')
                relative = Path('assets') / (sha256(source) + source.suffix)
                atomic_write(dest.parent / relative, source.read_bytes())
                return f'![{match.group(1)}]({relative.as_posix()})'
            archived = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', archive_image, path.read_text())
            atomic_write(dest, archived)
        else:
            atomic_write(dest, path.read_bytes())
        if prior.get(path.name) != sha256(path):
            if ext == 'docx':
                with zipfile.ZipFile(path) as z:
                    xml = ET.fromstring(z.read('word/document.xml'))
                text = '\n'.join(''.join(n.itertext()) for n in xml.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'))
            else:
                text = path.read_text()
            notes.setdefault('migrated_document_edits', []).append({'timestamp': stamp, 'source': dest.relative_to(report).as_posix(),
                                                                  'sha256': sha256(path), 'preserved_text': text})
            changed = True
    if changed:
        atomic_write(report / 'notes.yaml', yaml.safe_dump(notes, allow_unicode=True, sort_keys=False))


def build_report(root, formats=('md', 'docx')):
    root = Path(root).resolve()
    report = root / 'report'
    report.mkdir(parents=True, exist_ok=True)
    with lock(report / '.report.lock'):
        preserve_existing(report)
        runs, excluded, history = load_runs(root)
        blocks = report_blocks(root, runs, excluded, history)
        outputs = {}
        for ext in formats:
            path = report / f'G1_RECOVERY_EXPERIMENTS.{ext}'
            data = markdown(blocks, report) if ext == 'md' else word_document(blocks)
            atomic_write(path, data)
            outputs[path.name] = sha256(path)
        old = read_json(report / 'generated_files.json') if (report / 'generated_files.json').exists() else {}
        write_json(report / 'generated_files.json', {**old, **outputs})
    return {'outputs': {str(report / k): v for k, v in outputs.items()}, 'compatible_complete_runs': len(runs), 'excluded': excluded}


def analyze_traces(source, protocol_path):
    identity, records = RunStore(source).validate()
    p = load_yaml(protocol_path)
    old_p = load_yaml(source / 'protocol_snapshot.yaml')
    for key in old_p.keys() - {'metrics', 'protocol_version'}:
        if old_p[key] != p[key]:
            raise ValueError('Offline reanalysis may only change metrics, not physical trials')
    if p['metrics']['version'] != METRICS_VERSION:
        raise ValueError('This implementation supports practical_interval_confirm2_v1 only; implement a new detector for a new definition')
    reference, nodes = nominal_reference(p)
    if (digest(p['metrics']) != identity['metrics_config_hash'] or sha256(reference) != identity['metrics_reference_sha256']) and p['protocol_version'] == old_p['protocol_version'] and p['metrics']['version'] == old_p['metrics']['version']:
        raise ValueError('Changed metrics/reference requires a new protocol or metrics version')
    analysis_id = digest({'metrics': p['metrics'], 'reference': sha256(reference), 'source': sha256(source / 'completion.json')})[:24]
    dest = source / 'analyses' / analysis_id
    if dest.exists():
        return {'status': 'ALREADY_ANALYZED', 'path': str(dest)}
    new_rows = []
    for old in records:
        plan = {k: old[k] for k in read_jsonl(source / 'manifest_snapshot.jsonl')[0]}
        machine = TrialMachine(plan, p, nodes.get(f"{plan['slope_deg']}:{plan['command_name']}"))
        machine.trigger = {key: old[key] for key in ('actual_push_time', 'scheduled_push_time', 'reference_touchdown_time',
                          'T_used_for_trigger', 'trigger_timing_error', 'actual_contact_state', 'velocity_before', 'velocity_after', 'push_heading_yaw') if key in old}
        if plan['experiment'] == 'E1':
            machine.state = 'OBSERVE'
            machine.push_time = float('inf')  # original physical trigger, never rerun readiness with new thresholds
            machine.precondition_passed = old['precondition_passed']
        with np.load(source / 'traces' / (old['trial_id'] + '.npz'), allow_pickle=False) as trace:
            for index, t in enumerate(trace['time']):
                frame = {k: trace[k][index].tolist() for k in trace.files if k != 'plane_json'}
                frame['plane'] = json.loads(str(trace['plane_json'][index]))
                if plan['experiment'] == 'E1' and old['push_applied'] and t >= old['actual_push_time'] and machine.recovery is None:
                    machine.push_time = old['actual_push_time']
                    machine.recovery = RecoveryDetector(machine.push_time, p['e1']['recovery_deadline_s'], p['e1']['observation_s'])
                machine.feed(frame)
        if not old['push_applied'] and plan['experiment'] == 'E1':
            machine.push_time = None
        if not machine.status:
            machine.finish(old['status'] if old['status'] != 'RECOVERED_AND_SURVIVED' else 'ALIVE_NOT_RECOVERED')
        new_rows.append(machine.result())
    dest.mkdir(parents=True)
    atomic_write(dest / 'trials.jsonl', jsonl(new_rows))
    write_json(dest / 'summary.json', summarize(new_rows, read_jsonl(source / 'manifest_snapshot.jsonl')))
    atomic_write(dest / 'protocol.yaml', Path(protocol_path).read_bytes())
    atomic_write(dest / 'metrics_reference.yaml', reference.read_bytes())
    write_json(dest / 'analysis.json', {'source_completion_sha256': sha256(source / 'completion.json'),
                                      'metrics_version': p['metrics']['version'], 'metrics_config_hash': digest(p['metrics']),
                                      'metrics_reference_sha256': sha256(reference), 'independent_new_analysis': True})
    return {'status': 'ANALYZED', 'path': str(dest)}


def upload_wandb(run, protocol):
    """Optional display only; failure leaves complete local results untouched."""
    try:
        import wandb
        i = read_json(run / 'identity.json')
        summary = read_json(run / 'summary.json')
        name = f"{i['method']}_iter{i['training_iteration']}_seed{i['training_seed']}_{i['checkpoint_sha256'][:10]}"
        with wandb.init(project=protocol['wandb']['project'], group=protocol['wandb']['group'], name=name,
                        job_type='evaluation', reinit=True, config=i) as tracking:
            tracking.log({f'Eval/E1/{k}': v for k, v in summary['E1'].items() if isinstance(v, (int, float))})
        return True
    except Exception as exc:
        # Outside the sealed run so the completion SHA remains immutable.
        write_json(run.parent.parent / 'wandb_status' / (run.name + '.json'), {'uploaded': False, 'error': str(exc)})
        return False


DETECTOR_COMPATIBILITY = ('protocol_hash', 'metrics_version', 'metrics_config_hash',
                          'metrics_reference_sha256', 'physics_profile_hash', 'inference_mode')


def detector_protocol_hashes(protocol, prepared):
    hashes = {prepared['protocol_hash']}
    if protocol['protocol_version'] == '1.0':
        development = json.loads(json.dumps(protocol))
        development['protocol_version'] = '1.0-dev'
        hashes.add(digest(development))
    return hashes


def validate_detector_source(run, full_manifest, prepared, evaluation_runtime_sha256,
                             allowed_protocol_hashes=None):
    identity, source_manifest = run['identity'], run['manifest']
    if identity.get('synthetic'):
        raise ValueError('Synthetic run is not detector evidence')
    if identity.get('method') not in ('ppo_plain', 'dwaq'):
        raise ValueError('Detector evidence must come from a baseline method')
    if identity.get('checkpoint_stage') != 'final':
        raise ValueError('Detector evidence must use an explicitly final baseline checkpoint')
    allowed_protocol_hashes = allowed_protocol_hashes or {prepared['protocol_hash']}
    if identity.get('protocol_hash') not in allowed_protocol_hashes:
        raise ValueError('Detector source protocol_hash mismatch')
    for field in DETECTOR_COMPATIBILITY[1:]:
        if identity.get(field) != prepared[field]:
            raise ValueError(f'Detector source {field} mismatch')
    if identity.get('evaluation_runtime_sha256') != evaluation_runtime_sha256:
        raise ValueError('Detector source evaluation_runtime_sha256 mismatch')
    if not identity.get('actual_physics_hash'):
        raise ValueError('Detector source lacks actual_physics_hash')
    if identity.get('manifest_hash') != digest(source_manifest):
        raise ValueError('Detector source manifest identity mismatch')
    subset = identity.get('subset')
    if subset == 'lite_full':
        if source_manifest != full_manifest:
            raise ValueError('Detector full manifest differs from current prepared manifest')
    else:
        match = re.fullmatch(r'first_([1-9][0-9]*)_per_experiment', subset or '')
        if not match:
            raise ValueError('Detector source subset is not a supported deterministic subset')
        count = int(match.group(1))
        expected = [row for experiment in ('E0', 'E1')
                    for row in [item for item in full_manifest if item['experiment'] == experiment][:count]]
        if source_manifest != expected:
            raise ValueError('Detector development manifest is not the exact prepared deterministic subset')
    return {'evaluation_id': run['id'], 'method': identity['method'], 'subset': subset,
            'manifest_hash': identity['manifest_hash'], 'actual_physics_hash': identity['actual_physics_hash'],
            'protocol_hash': identity['protocol_hash'],
            'evaluation_runtime_sha256': identity['evaluation_runtime_sha256']}


def validate_detector_gate(root, protocol, full_manifest, prepared, identity, gate):
    if gate.get('status') != 'PASSED' or not gate.get('reviewer'):
        raise ValueError('Detector gate must include reviewer and at least 20 reviewed trace SHAs')
    runtime_hash = identity['evaluation_runtime_sha256']
    allowed = detector_protocol_hashes(protocol, prepared)
    if gate.get('protocol_hash') not in allowed:
        raise ValueError('Detector gate belongs to a different protocol')
    for field in ('manifest_hash', 'metrics_version', 'metrics_config_hash',
                  'metrics_reference_sha256', 'physics_profile_hash', 'inference_mode'):
        if gate.get(field) != prepared[field]:
            raise ValueError('Detector gate belongs to a different protocol/metrics/physics configuration')
    if gate.get('evaluation_runtime_sha256') != runtime_hash:
        raise ValueError('Detector gate belongs to a different evaluation runtime')
    source_entries = gate.get('sources')
    if not source_entries:
        raise ValueError('Detector gate has no revalidatable source evaluations')
    available_traces, validated = set(), []
    for claimed in source_entries:
        if Path(claimed['evaluation_id']).name != claimed['evaluation_id']:
            raise ValueError('Detector source evaluation ID is unsafe')
        source_path = Path(root) / 'runs' / claimed['evaluation_id']
        if not (source_path / 'completion.json').exists():
            raise ValueError(f'Detector source is not sealed COMPLETE: {source_path.name}')
        source_identity, records = RunStore(source_path).validate()
        source = {'id': source_path.name, 'path': source_path, 'identity': source_identity,
                  'records': records, 'manifest': read_jsonl(source_path / 'manifest_snapshot.jsonl')}
        actual = validate_detector_source(source, full_manifest, prepared, runtime_hash, allowed)
        if actual != claimed:
            raise ValueError(f'Detector source metadata changed: {source_path.name}')
        validated.append(actual)
        available_traces.update(record['trace_sha256'] for record in records)
    reviewed = {item.get('trace_sha256') if isinstance(item, dict) else item
                for item in (gate.get('reviewed_traces') or [])}
    if len(reviewed) < 20:
        raise ValueError('Detector gate must include at least 20 distinct reviewed trace SHAs')
    if None in reviewed or not reviewed.issubset(available_traces):
        raise ValueError('Detector gate references a trace outside its validated sources')
    if {source['method'] for source in validated} != {'ppo_plain', 'dwaq'}:
        raise ValueError('Detector gate must include both baseline methods')
    return validated


def detector_validation_report(root):
    """Evidence inventory for manual detector acceptance; never auto-relax thresholds/freeze."""
    root = Path(root)
    p, manifest, info = load_prepared(root)
    runs, excluded, _ = load_runs(root, formal_only=False)
    runtime_hash = code_identity()['evaluation_runtime_sha256']
    allowed = detector_protocol_hashes(p, info)
    candidates, rejected, sources = [], [], []
    for run in runs:
        if run['identity'].get('method') not in ('ppo_plain', 'dwaq'):
            continue
        try:
            sources.append(validate_detector_source(run, manifest, info, runtime_hash, allowed))
            candidates.append(run)
        except ValueError as exc:
            rejected.append({'evaluation_id': run['id'], 'reason': str(exc)})
    rows, push_ids = [], []
    for run in candidates:
        intervals, passed = 0, 0
        for trial in run['records']:
            events = read_jsonl(run['path'] / 'trial_events' / (trial['trial_id'] + '.jsonl'))
            if trial['experiment'] == 'E0' and trial['command_name'] != 'standing':
                for event in events:
                    if event['event'] == 'practical_interval':
                        intervals += 1
                        passed += int(event['passed'])
            if trial['experiment'] == 'E1' and trial['push_applied']:
                push_ids.append({'evaluation_id': run['id'], 'trial_id': trial['trial_id'],
                                 'magnitude': trial['push_magnitude'], 'status': trial['status'],
                                 'trace_sha256': trial['trace_sha256']})
        rows.append([run['identity']['model_alias'], intervals, passed,
                     fmt(passed / intervals if intervals else None, True)])
    evidence = {k: info[k] for k in ('protocol_hash', 'manifest_hash', 'metrics_version', 'metrics_config_hash',
                                     'metrics_reference_sha256', 'physics_profile_hash', 'inference_mode')}
    evidence.update(status='PENDING_MANUAL_VALIDATION' if push_ids else 'NOT_RUN',
                    evaluation_runtime_sha256=runtime_hash, sources=sources,
                    source_evaluation_ids=[source['evaluation_id'] for source in sources],
                    source_subsets={source['evaluation_id']: source['subset'] for source in sources},
                    source_manifest_hashes={source['evaluation_id']: source['manifest_hash'] for source in sources},
                    actual_physics_hashes={source['evaluation_id']: source['actual_physics_hash'] for source in sources},
                    rejected_sources=rejected, load_exclusions=excluded,
                    pushed_trajectories=push_ids, minimum_required=20, automatic_freeze=False)
    blocks = [('heading', 1, 'DETECTOR VALIDATION / 恢复检测器验收'),
              ('text', '状态：' + evidence['status']),
              ('text', '本轮按用户要求未启动实验。离线单元测试不等价于真实轨迹验收；协议保持 1.0-dev。' if not push_ids else
               '本文件只汇总可审计证据；需人工逐条审查正常行走、轻/重扰动与失败，才能确认检测器适用性。'),
              ('text', f'已有 baseline 推扰轨迹 {len(push_ids)}，最低需检查 20 条，且覆盖两个 baseline、轻/重扰动和失败。'),
              table(['Baseline', 'E0 moving 完整间隔', '达标间隔', '达标比例'], rows),
              ('heading', 2, '待验收事项'),
              ('text', '检查无扰动正常片段识别率、假恢复、长时间不恢复、接触抖动及重复同脚计步。'
               '检查 entry/confirmed 时刻，确认后跌倒不得算主成功。若正常片段大量不达标，停止冻结并报告证据，不自动放宽阈值。'),
              ('text', '人工验收记录写入 report/notes.yaml 的 detector_validation 项，逐条关联 evaluation_id / trial_id / trace SHA。'),
              ('text', json.dumps(evidence, ensure_ascii=False, indent=2))]
    with lock(root / 'report/.report.lock'):
        atomic_write(root / 'report/DETECTOR_VALIDATION.md', markdown(blocks, root / 'report'))
        write_json(root / 'detector_evidence.json', evidence)
    return evidence
