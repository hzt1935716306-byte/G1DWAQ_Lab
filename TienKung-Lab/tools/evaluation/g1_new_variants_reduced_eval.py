#!/usr/bin/env python3
"""Evaluate three new checkpoints on the frozen standard390 + reduced1960 plan."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

from g1_recovery_protocol import (
    COMMON_PROTOCOL,
    LAB,
    atomic_write,
    code_identity,
    digest,
    inspect_checkpoint,
    jsonl,
    load_prepared as load_standard,
    prepare,
    read_json,
    read_jsonl,
    sha256,
    write_json,
)
from g1_reduced_budget import ROOT as REDUCED, SOURCE, TARGET_COUNTS, immutable
from g1_robustness_protocol import load_prepared as load_robustness


ROOT = LAB / 'experiments/g1_new_variants_reduced_eval_v1'
ESTIMATOR = 'logs/g1_com_velocity_estimator/v2_iteration_long_5000_random_init_fixed/com_velocity_estimator_v2_long_best.pt'
RESOURCE_PATHS = {
    'native_nominal': 'tools/recovery/generated/g1_plane_nominal_params_g1_slope_sys_d_candidate.yaml',
    'native_capability': 'tools/recovery/generated/g1_recovery_params.yaml',
    'estimator': ESTIMATOR,
}
MODELS = {
    'dwaq_v4': {
        'label': 'DWAQ V4',
        'native_model': 'dwaq',
        'task': 'g1_dwaq_slope_nosys_d_matched_v4',
        'checkpoint': 'logs/g1_dwaq_slope_nosys_d_matched_v4/2026-09-11_02-24-46_dwaq_nosys_matched_no_idle_no_swing/model_9999.pt',
        'estimator': None,
        'dimensions': (115, 307, 29, 1, 1),
        'iteration': 9999,
    },
    'context_only_v3': {
        'label': 'Context-only V3 (fine-tuned)',
        'native_model': 'context_only',
        'task': 'g1_plane_v1_estimator_context_no_reward_matched_v3',
        'checkpoint': 'logs/g1_plane_v1_estimator_context_no_reward_matched_v3/2026-09-11_02-07-23_estimator_context_no_reward_matched_v3_idle2p0_swing0p2_ft2000/model_11999.pt',
        'estimator': ESTIMATOR,
        'dimensions': (483, 1010, 29, 5, 10),
        'iteration': 11999,
    },
    'context_reward_v2': {
        'label': 'Context+Reward V2',
        'native_model': 'context_reward',
        'task': 'g1_plane_v1_estimator_context_reward_matched_v2',
        'checkpoint': 'logs/g1_plane_v1_estimator_context_reward_matched_v2/2026-09-09_16-51-41_estimator_context_reward_matched_v2_idle01_swing02/model_9999.pt',
        'estimator': ESTIMATOR,
        'dimensions': (483, 1010, 29, 5, 10),
        'iteration': 9999,
    },
}
PHYSICS_FIELDS = (
    'actual_physics_hash',
    'realized_environment_hash',
    'candidate_parameters_sha256',
    'protocol_hash',
    'metrics_config_hash',
)


def identity_manifest(alias: str, spec: dict) -> Path | None:
    if not spec['estimator']:
        return None
    checkpoint = LAB / spec['checkpoint']
    value = {
        'task_name': spec['task'],
        'task_confirmed': True,
        'checkpoint_sha256': sha256(checkpoint),
        'agent_config_sha256': sha256(checkpoint.parent / 'params/agent.yaml'),
        'env_config_sha256': sha256(checkpoint.parent / 'params/env.yaml'),
        'resources': {
            role: {'path': path, 'sha256': sha256(LAB / path)}
            for role, path in RESOURCE_PATHS.items()
        },
    }
    path = ROOT / alias / 'identity_manifest.json'
    immutable(path, value)
    return path


def inspect_models() -> dict:
    identities = {}
    for alias, spec in MODELS.items():
        declaration = identity_manifest(alias, spec)
        identity = inspect_checkpoint(
            spec['task'], LAB / spec['checkpoint'], alias, 'final',
            LAB / spec['estimator'] if spec['estimator'] else None, declaration,
        )
        dimensions = tuple(identity[k] for k in (
            'actor_input_dimension', 'critic_input_dimension', 'action_dimension',
            'actor_history_length', 'critic_history_length'))
        if dimensions != spec['dimensions']:
            raise ValueError(f'{alias}: native dimensions/history differ: {dimensions}')
        if identity['method'] != spec['native_model'] or identity['training_iteration'] != spec['iteration']:
            raise ValueError(f'{alias}: checkpoint task/method/iteration differs')
        identities[alias] = dict(identity)
        immutable(ROOT / alias / 'checkpoint_identity.json', dict(identity))
    immutable(ROOT / 'checkpoint_identities.json', identities)
    return identities


def completed_run(root: Path) -> Path | None:
    runs = list((root / 'runs').glob('*')) if (root / 'runs').exists() else []
    invalid = [run for run in runs if (run / 'failure.json').exists()]
    if invalid:
        raise ValueError(f'INVALID attempt retained: {invalid[0]}')
    complete = [run for run in runs if (run / 'completion.json').exists()
                and read_json(run / 'completion.json')['status'] == 'COMPLETE']
    if len(complete) > 1:
        raise ValueError(f'Ambiguous completed runs: {root}')
    return complete[0] if complete else None


def reference_standard(native_model: str) -> Path:
    index = read_json(REDUCED / 'standard_benchmark_five_model_index.json')
    row = next(item for item in index['models'] if item['model'] == native_model)
    return LAB / row.get('run_path', 'experiments/g1_recovery_eval_v2/runs/' + row['evaluation_id'])


def check_physics(run: Path, reference: Path) -> None:
    actual, expected = read_json(run / 'identity.json'), read_json(reference / 'identity.json')
    for field in PHYSICS_FIELDS:
        if actual[field] != expected[field]:
            raise ValueError(f'Physical/detector identity differs for {field}')


def prepare_standard(alias: str, spec: dict) -> Path:
    root = ROOT / alias / 'standard'
    prepare(root, COMMON_PROTOCOL)
    reference = reference_standard(spec['native_model'])
    if load_standard(root)[1] != read_jsonl(reference / 'manifest_snapshot.jsonl'):
        raise ValueError(f'{alias}: standard390 trial plan differs')
    return root


def prepare_robustness(alias: str, spec: dict, standard: Path) -> Path:
    master = ROOT / alias
    target = master / 'robustness' / 'execution' / spec['native_model']
    selection = read_json(REDUCED / 'reduced_budget_manifest_v1.json')
    immutable(master / 'reduced_budget_manifest_v1.json', selection)
    rows = read_jsonl(SOURCE / 'execution/context_only/manifest.jsonl')
    selected_ids = {item['trial_id'] for item in selection['trials']}
    if len(rows) != 1960 or {row['trial_id'] for row in rows} != selected_ids:
        raise ValueError('Frozen reduced1960 source differs')
    ids = [row['trial_id'] for row in rows]
    immutable(master / 'pending_execution_queues.json', {spec['native_model']: ids})
    immutable(master / 'reduced_budget_reuse_index.json', {'entries': [
        {'model': spec['native_model'], 'target_trial_id': trial_id, 'reuse_decision': 'PENDING_EXECUTION'}
        for trial_id in ids
    ]})
    identity = read_json(standard / 'identity.json')
    index = {'models': [{
        'model': spec['native_model'], 'task': identity['task_name'],
        'evaluation_id': standard.name, 'run_path': str(standard.relative_to(LAB)),
        'checkpoint_path': identity['checkpoint_path'],
        'checkpoint_sha256': identity['checkpoint_sha256'],
        'completion_sha256': sha256(standard / 'completion.json'),
    }]}
    immutable(master / 'standard_benchmark_five_model_index.json', index)
    protocol, _, source_info = load_robustness(SOURCE)
    info = {
        **source_info,
        'budget_mode': 'reduced_budget_v1',
        'reduced_model': spec['native_model'],
        'reduced_budget_root': str(master),
        'standard_index_sha256': sha256(master / 'standard_benchmark_five_model_index.json'),
        'reduced_manifest_sha256': sha256(master / 'reduced_budget_manifest_v1.json'),
        'execution_batch_size': 64,
        'execution_queue_sha256': sha256(master / 'pending_execution_queues.json'),
        'reuse_index_sha256': sha256(master / 'reduced_budget_reuse_index.json'),
        'manifest_hash': digest(rows),
        'counts_per_model': dict(Counter(row['suite'] for row in rows)),
        'trials_per_model': 1960,
        'total_new_robustness_trials': 1960,
    }
    files = {
        'protocol.yaml': yaml.safe_dump(protocol, sort_keys=False).encode(),
        'manifest.jsonl': jsonl(rows).encode(),
        'standard_benchmark_five_model_index.json': (master / 'standard_benchmark_five_model_index.json').read_bytes(),
        'common_detector_config.yaml': (SOURCE / 'common_detector_config.yaml').read_bytes(),
    }
    for name, data in files.items():
        path = target / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f'{alias}: immutable robustness input changed: {name}')
        if not path.exists():
            atomic_write(path, data)
    immutable(target / 'prepared.json', info)
    for trial_id in ids:
        immutable(target / 'paired_initial_states' / f'{trial_id}.json',
                  read_json(SOURCE / 'paired_initial_states' / f'{trial_id}.json'))
    load_robustness(target)
    return target


def run_owned(command: list[str], root: Path, label: str, state: dict) -> Path:
    existing = completed_run(root)
    if existing:
        return existing
    log = ROOT / f'{label}.log'
    if log.exists():
        raise ValueError(f'Incomplete prior attempt requires inspection: {log}')
    environment = os.environ.copy()
    environment['PYTHONPATH'] = ':'.join((str(LAB / 'tools/evaluation'), str(LAB / 'rsl_rl'),
                                          str(LAB), environment.get('PYTHONPATH', '')))
    with log.open('xb', buffering=0) as stream:
        process = subprocess.Popen(command, cwd=LAB, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(status='RUNNING', current=label, pid=process.pid, log=str(log))
        write_json(ROOT / 'status.json', state)
        sealed_at = None
        try:
            while True:
                time.sleep(10)
                run = completed_run(root)
                if run:
                    sealed_at = sealed_at or time.monotonic()
                    if process.poll() is not None or time.monotonic() - sealed_at > 30:
                        return run
                elif process.poll() is not None:
                    raise ValueError(f'Evaluator exited before COMPLETE: {label}')
                state['completed_current'] = sum(1 for _ in root.glob('runs/*/trial_records/*.json'))
                write_json(ROOT / 'status.json', state)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def read_sealed(run: Path, robustness: bool) -> list[dict]:
    from g1_recovery_protocol import RunStore
    from g1_robustness_store import RobustnessStore
    if read_json(run / 'completion.json')['status'] != 'COMPLETE':
        raise ValueError(f'Unsealed result: {run}')
    (RobustnessStore(run) if robustness else RunStore(run)).validate(validation_level='light')
    return [read_json(path) for path in sorted((run / 'trial_records').glob('*.json'))]


def old_four() -> tuple[dict, dict]:
    from g1_reduced_sources import verify_reference
    sources = read_json(LAB / 'experiments/g1_context_v2_final_eval/report/sources.json')
    aliases = {
        'ppo_plain': 'ppo_plain',
        'dwaq_v2': 'dwaq_v2',
        'dwaq_v3': 'dwaq_v3',
        'context_only_v2': 'context_only_v2_final',
    }
    standard, robust = {}, {}
    for alias, source_alias in aliases.items():
        ref = sources[source_alias]['standard']
        run = Path(ref['run'])
        if sha256(run / 'completion.json') != ref['completion_sha256']:
            raise ValueError(f'Old standard seal changed: {alias}')
        standard[alias] = read_sealed(run, False)
        if 'robustness' in sources[source_alias]:
            ref = sources[source_alias]['robustness']
            run = Path(ref['run'])
            if sha256(run / 'completion.json') != ref['completion_sha256']:
                raise ValueError(f'Old robustness seal changed: {alias}')
            robust[alias] = read_sealed(run, True)
    cache = set()
    index = read_json(REDUCED / 'primary_matched_source_index.json')['entries']
    robust['ppo_plain'] = [verify_reference(ref, cache) for ref in index if ref['model'] == 'ppo_plain']
    return standard, robust


def build_report(new_runs: dict | None = None) -> dict:
    from g1_dwaq_ablation_report import paired_records
    from g1_robustness_analysis import aggregate
    standard, robust = old_four()
    if new_runs is None:
        new_runs = {}
        for alias, spec in MODELS.items():
            standard_run = completed_run(ROOT / alias / 'standard')
            robustness_run = completed_run(ROOT / alias / 'robustness/execution' / spec['native_model'])
            if not standard_run or not robustness_run:
                raise ValueError(f'{alias}: incomplete result')
            new_runs[alias] = (standard_run, robustness_run)
    for alias, (standard_run, robustness_run) in new_runs.items():
        standard[alias] = read_sealed(standard_run, False)
        robust[alias] = read_sealed(robustness_run, True)
    base = standard['ppo_plain']
    robust_base = robust['ppo_plain']
    for alias in standard:
        paired_records(standard[alias], base)
        paired_records(robust[alias], robust_base)
        if len(standard[alias]) != 390 or dict(Counter(row['suite'] for row in robust[alias])) != TARGET_COUNTS:
            raise ValueError(f'{alias}: frozen trial counts differ')
    labels = {
        'ppo_plain': 'PPO', 'dwaq_v2': 'DWAQ V2', 'dwaq_v3': 'DWAQ V3',
        'context_only_v2': 'Context-only V2',
        **{alias: spec['label'] for alias, spec in MODELS.items()},
    }
    groups = {
        'standard_recovery': {m: [r for r in rows if r['experiment'] == 'E1'] for m, rows in standard.items()},
        'in_domain_robustness': {m: [r for r in rows if r['suite'] != 'extreme_ood'] for m, rows in robust.items()},
        'extreme_main': {m: [r for r in rows if r['suite'] == 'extreme_main'] for m, rows in robust.items()},
        'extreme_ood': {m: [r for r in rows if r['suite'] == 'extreme_ood'] for m, rows in robust.items()},
        'velocity_ood': {m: [r for r in rows if r['suite'] == 'velocity_ood'] for m, rows in robust.items()},
        'force_pulse': {m: [r for r in rows if r['suite'] == 'force_pulse'] for m, rows in robust.items()},
        'constant_force': {m: [r for r in rows if r['suite'] == 'constant_force'] for m, rows in robust.items()},
        'repeated_impulse': {m: [r for r in rows if r['suite'] == 'repeated_impulse'] for m, rows in robust.items()},
        'random_force': {m: [r for r in rows if r['suite'] == 'random_force'] for m, rows in robust.items()},
        'wrench_pulse': {m: [r for r in rows if r['suite'] == 'wrench_pulse'] for m, rows in robust.items()},
    }
    summary = {group: {model: aggregate(rows) for model, rows in values.items()}
               for group, values in groups.items()}
    report = ROOT / 'report'
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / 'summary.json', summary)
    def pct(count, total):
        return 'N/A' if not total else f'{100 * count / total:.2f}%'
    def num(value):
        return 'N/A' if value is None else f'{value:.2f}'
    lines = [
        '# 七模型 Common Task 固定方案对比', '',
        '每个模型使用相同的 390 条标准清单和 1,960 条缩减 robustness 清单。'
        '恢复率分母包含指定的真实扰动试验；时间与物理落脚次数只统计成功持续恢复样本。', '',
        '| 模型 | 标准持续恢复 | 域内持续恢复 | 域内生存 | ≤5落脚恢复 | 恢复时间中位数(s) | 平均落脚 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for model in labels:
        std = summary['standard_recovery'][model]
        domain = summary['in_domain_robustness'][model]
        lines.append('| ' + ' | '.join((
            labels[model],
            pct(std['recovered_sustained_and_survived']['count'], std['designated_real']),
            pct(domain['recovered_sustained_and_survived']['count'], domain['designated_real']),
            pct(domain['survived']['count'], domain['actually_pushed']),
            pct(domain['recovery_within_5_touchdowns']['count'], domain['designated_real']),
            num(domain['recovery_time']['median']), num(domain['recovery_steps']['mean']),
        )) + ' |')
    lines += ['', '## 分扰动族', '']
    for group, title in (
        ('extreme_main', '域内瞬时速度扰动'), ('extreme_ood', 'OOD ±20°'),
        ('velocity_ood', '随机二维速度突变'), ('force_pulse', '有限时长力脉冲'),
        ('constant_force', '5秒持续力'), ('repeated_impulse', '8秒重复冲击'),
        ('random_force', '10秒随机力'), ('wrench_pulse', '力/力矩脉冲')):
        lines += [f'### {title}', '', '| 模型 | 持续恢复/指定 | 生存/实际施扰 | T中位数(s) | K均值 | K中位数 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for model in labels:
            value = summary[group][model]
            lines.append('| ' + ' | '.join((labels[model],
                pct(value['recovered_sustained_and_survived']['count'], value['designated_real']),
                pct(value['survived']['count'], value['actually_pushed']),
                num(value['recovery_time']['median']), num(value['recovery_steps']['mean']),
                num(value['recovery_steps']['median']))) + ' |')
        lines.append('')
    lines += [
        '候选 Common Task 判据及 W/H 未修改。OOD ±20° 单独列示。未恢复样本的时间与落脚次数保持为空，未用 0 或超时值代替。',
        '旧四模型 sealed 数据只读复用；没有执行 D50 或新的边界探索。',
    ]
    atomic_write(report / 'SEVEN_MODEL_COMPARISON_ZH.md', ('\n'.join(lines) + '\n').encode())
    result = {
        'status': 'PASS', 'models': list(labels), 'episodes_per_new_model': 2350,
        'new_episodes': 2350 * len(MODELS), 'evaluation_errors': 0,
        'standard_trial_ids_paired': True, 'robustness_trial_ids_paired': True,
        'threshold_changed': False, 'boundary_exploration_run': False,
    }
    write_json(report / 'integrity.json', result)
    return result


def run() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    code = code_identity()
    if code['evaluation_runtime_dirty']:
        raise ValueError('Commit evaluation runtime before physical execution')
    immutable(ROOT / 'runtime_identity.json', code)
    identities = inspect_models()
    state = {'status': 'PREPARED', 'driver_pid': os.getpid(),
             'commit': code['evaluation_code_commit'], 'completed': [],
             'planned_per_model': 2350, 'planned_total': 7050}
    write_json(ROOT / 'status.json', state)
    runs = {}
    try:
        for alias, spec in MODELS.items():
            if code_identity()['evaluation_code_commit'] != state['commit']:
                raise ValueError('Evaluation commit changed during queue')
            standard_root = prepare_standard(alias, spec)
            command = [
                sys.executable, '-u', '-B', 'tools/evaluation/g1_recovery_eval.py', 'run',
                '--protocol', str(COMMON_PROTOCOL), '--task', spec['task'],
                '--checkpoint', str(LAB / spec['checkpoint']), '--model_alias', alias,
                '--checkpoint_stage', 'final', '--suite', 'lite', '--num_envs', '8',
                '--headless', '--device', 'cuda:0', '--output_root', str(standard_root),
            ]
            if spec['estimator']:
                command += ['--estimator_checkpoint', str(LAB / spec['estimator']),
                            '--identity_manifest', str(ROOT / alias / 'identity_manifest.json')]
            standard_run = run_owned(command, standard_root, alias + '_standard', state)
            check_physics(standard_run, reference_standard(spec['native_model']))
            state['completed'].append(alias + '_standard')
            write_json(ROOT / 'status.json', state)

            robustness_root = prepare_robustness(alias, spec, standard_run)
            command = [
                sys.executable, '-u', '-B', 'tools/evaluation/g1_complete_robustness.py', 'run',
                '--output_root', str(robustness_root), '--model', spec['native_model'], '--device', 'cuda:0',
            ]
            robustness_run = run_owned(command, robustness_root, alias + '_robustness', state)
            reference = read_json(SOURCE / 'completed_models' / f"{spec['native_model']}.json")
            check_physics(robustness_run, SOURCE / 'runs' / reference['evaluation_id'])
            state['completed'].append(alias + '_robustness')
            runs[alias] = (standard_run, robustness_run)
            write_json(ROOT / 'status.json', state)
        result = build_report(runs)
        state.update(status='COMPLETE', current=None, pid=None, completed_current=0,
                     total_completed=result['new_episodes'], evaluation_errors=0,
                     report=str(ROOT / 'report/SEVEN_MODEL_COMPARISON_ZH.md'))
        write_json(ROOT / 'status.json', state)
    except BaseException as exc:
        state.update(status='STOPPED_ERROR', error=repr(exc), pid=None)
        write_json(ROOT / 'status.json', state)
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    build_report() if args.report_only else run()
