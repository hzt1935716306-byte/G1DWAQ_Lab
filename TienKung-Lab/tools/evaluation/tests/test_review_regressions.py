"""E01–E08 acceptance with synthetic weights, CPU solvers and temporary artifacts."""
import ast
import copy
import re
import shutil
import sys
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
import yaml

import g1_recovery_protocol as protocol
import g1_recovery_metrics as metrics
from g1_recovery_eval import snapshot_checkpoint, bind_native_inputs, assert_native_inputs, verify_native_snapshot
from g1_recovery_report import (budget_relation, matched_ablation, previous_checkpoint, load_runs,
                                preserve_existing, report_blocks, markdown, curve_plots)
from g1_recovery_protocol import LAB, TASKS, atomic_write, sha256, digest, load_yaml, inspect_checkpoint, write_json
from test_protocol_report import synthetic_run
from test_metrics import frame, P, PLANS, NODES


@pytest.fixture
def checkpoint_factory(tmp_path):
    sys.path.insert(0, str(LAB / 'rsl_rl'))
    from rsl_rl.modules.actor_critic import ActorCritic
    from legged_lab.estimation.com_velocity_estimator import ComVelocityEstimator
    def make(method='context_only'):
        directory = tmp_path / method
        directory.mkdir(exist_ok=True)
        context = method.startswith('context')
        agent = {'runner_class_name': 'OnPolicyRunner', 'experiment_name': 'g1_plane_v1_matched',
                 'run_name': '任意展示名 my-run', 'seed': 42, 'empirical_normalization': False,
                 'policy': {'actor_hidden_dims': [16, 8], 'critic_hidden_dims': [16, 8],
                            'activation': 'elu', 'init_noise_std': 1.}}
        env = {'robot': {'actor_obs_history_length': 5, 'critic_obs_history_length': 10}}
        resources = {}
        if context:
            for role in ('native_nominal', 'native_capability'):
                path = directory / (role + '.yaml')
                atomic_write(path, f'{role}: A\n')
                resources[role] = {'path': path.name, 'sha256': sha256(path)}
            env.update(com_velocity_source='estimator', recovery_context={'enabled': True, 'mode': 'certificate'},
                       plane_v1_reward={'enabled': method == 'context_reward'},
                       plane_recovery={'nominal_parameters_path': str(directory / 'native_nominal.yaml'), 'z_sole': -.045},
                       stage2_reward={'certificate_parameters_path': str(directory / 'native_capability.yaml')})
        atomic_write(directory / 'params/agent.yaml', yaml.safe_dump(agent))
        atomic_write(directory / 'params/env.yaml', yaml.safe_dump(env))
        model = ActorCritic(483 if context else 480, 1010, 29, **agent['policy'])
        checkpoint = directory / 'model.pt'
        torch.save({'model_state_dict': model.state_dict(), 'iter': 1000}, checkpoint)
        est = None
        if context:
            est = directory / 'estimator.pt'
            estimator = ComVelocityEstimator(495, [256, 128, 64], 2)
            torch.save({'model_state_dict': estimator.state_dict(), 'input_dim': 495, 'history_length': 5,
                        'actor_per_frame_obs_dim': 96, 'per_frame_obs_dim': 99, 'imu_input_dim': 3,
                        'imu_acceleration_scale': .05, 'hidden_dims': [256, 128, 64], 'output_dim': 2,
                        'output_frame': 'heading', 'output_quantity': 'whole_body_com_velocity_xy', 'output_unit': 'm/s'}, est)
        manifest = {'checkpoint_sha256': sha256(checkpoint), 'agent_config_sha256': sha256(directory / 'params/agent.yaml'),
                    'env_config_sha256': sha256(directory / 'params/env.yaml'), 'resources': resources,
                    'training_run_id': 'synthetic-training-A', 'training_transitions': 240000}
        evidence = directory / 'identity.yaml'
        atomic_write(evidence, yaml.safe_dump(manifest))
        return NS(path=checkpoint, estimator=est, manifest=evidence, env=env, agent=agent, method=method)
    return make


def inspect(fixture, **kw):
    return inspect_checkpoint(TASKS[fixture.method], fixture.path, 'synthetic', estimator=fixture.estimator,
                              identity_manifest=fixture.manifest, **kw)


def rewrite_fixture(fixture, **payload_changes):
    atomic_write(fixture.path.parent / 'params/env.yaml', yaml.safe_dump(fixture.env))
    atomic_write(fixture.path.parent / 'params/agent.yaml', yaml.safe_dump(fixture.agent))
    if payload_changes:
        payload = torch.load(fixture.path, weights_only=False)
        payload.update(payload_changes)
        torch.save(payload, fixture.path)
    declaration = load_yaml(fixture.manifest)
    declaration.update(checkpoint_sha256=sha256(fixture.path),
                       agent_config_sha256=sha256(fixture.path.parent / 'params/agent.yaml'),
                       env_config_sha256=sha256(fixture.path.parent / 'params/env.yaml'))
    atomic_write(fixture.manifest, yaml.safe_dump(declaration))


@pytest.mark.parametrize('method', ['rl_only', 'context_only', 'context_reward'])
def test_custom_run_name_and_strict_weights(checkpoint_factory, method):
    f = checkpoint_factory(method)
    first = inspect(f)
    f.agent['run_name'] = 'another name with no fixed prefix'
    rewrite_fixture(f)
    second = inspect(f)
    assert first['checkpoint_sha256'] == second['checkpoint_sha256']
    state = torch.load(f.path, weights_only=False)['model_state_dict']
    state.pop('critic.0.bias')
    rewrite_fixture(f, model_state_dict=state)
    with pytest.raises(RuntimeError, match='Missing key'):
        inspect(f)


@pytest.mark.parametrize('wrong', ['source', 'reward', 'context', 'shape', 'task'])
def test_incorrect_identity_still_rejected(checkpoint_factory, wrong):
    f = checkpoint_factory()
    if wrong == 'source':
        f.env['com_velocity_source'] = 'privileged'
    elif wrong == 'reward':
        f.env['plane_v1_reward']['enabled'] = True
    elif wrong == 'context':
        f.env['recovery_context']['mode'] = 'zero'
    elif wrong == 'task':
        f.env['task_name'] = TASKS['context_reward']
    else:
        state = torch.load(f.path, weights_only=False)['model_state_dict']
        state['actor.0.weight'] = torch.zeros(16, 482)
        rewrite_fixture(f, model_state_dict=state)
    rewrite_fixture(f)
    with pytest.raises(ValueError, match='mismatch'):
        inspect(f)


def test_legacy_confirmation_is_explicit_and_sha_bound(checkpoint_factory):
    f = checkpoint_factory('rl_only')
    f.agent['experiment_name'] = 'legacy'
    rewrite_fixture(f)
    with pytest.raises(ValueError, match='task_confirmed'):
        inspect(f)
    declaration = load_yaml(f.manifest)
    declaration.update(task_name=TASKS['rl_only'], task_confirmed=True)
    atomic_write(f.manifest, yaml.safe_dump(declaration))
    assert inspect(f)['task_confirmation'] is True
    declaration['checkpoint_sha256'] = 'wrong'
    atomic_write(f.manifest, yaml.safe_dump(declaration))
    with pytest.raises(ValueError, match='manifest checkpoint_sha256 mismatch'):
        inspect(f)


def test_relocation_preserves_raw_config_and_content(checkpoint_factory, tmp_path):
    f = checkpoint_factory()
    before = inspect(f)
    raw = (f.path.parent / 'params/env.yaml').read_bytes()
    moved = tmp_path / 'another-checkout'
    shutil.copytree(f.path.parent, moved)
    shutil.rmtree(f.path.parent)
    f.path, f.estimator, f.manifest = moved / 'model.pt', moved / 'estimator.pt', moved / 'identity.yaml'
    # Content identity also survives an explicitly mapped filename/extension change.
    resource = moved / 'renamed-native-content.data'
    (moved / 'native_nominal.yaml').rename(resource)
    declaration = load_yaml(f.manifest)
    declaration['resources']['native_nominal']['path'] = resource.name
    atomic_write(f.manifest, yaml.safe_dump(declaration))
    after = inspect(f)
    assert before['resources'] == after['resources']
    assert before['env_config_sha256'] == after['env_config_sha256']
    assert before['training_run_id'] == after['training_run_id']
    snapshot = snapshot_checkpoint(tmp_path / 'snapshots', after)
    assert (snapshot.parent / 'params/env.yaml').read_bytes() == raw
    assert verify_native_snapshot(after, snapshot)['native_nominal'].read_text() == 'native_nominal: A\n'
    with pytest.raises(ValueError, match='explicit path and sha256'):
        inspect_checkpoint(TASKS[f.method], f.path, 'missing-map', estimator=f.estimator)
    resource.write_text('native_nominal: B\n')
    with pytest.raises(ValueError, match='resource SHA mismatch'):
        inspect(f)
    resource.unlink()
    with pytest.raises(FileNotFoundError):
        inspect(f)


def test_actual_solver_uses_snapshot_and_detects_drift(checkpoint_factory, tmp_path):
    f = checkpoint_factory()
    identity = inspect(f)
    snapshot = snapshot_checkpoint(tmp_path / 'output', identity)
    b = tmp_path / 'registry-B.yaml'
    b.write_text('native_nominal: B\n')
    current = copy.deepcopy(f.env)
    current['plane_recovery']['nominal_parameters_path'] = str(b)
    cfg = NS(**{k: NS(**v) if isinstance(v, dict) else v for k, v in current.items()})
    cfg.to_dict = lambda: current
    paths = bind_native_inputs(cfg, identity, snapshot)
    inconsistent = {**identity, 'native_nominal_sha256': sha256(b)}
    with pytest.raises(ValueError, match='identity SHA conflicts'):
        verify_native_snapshot(inconsistent, snapshot)
    assert cfg.plane_recovery.nominal_parameters_path == str(paths['native_nominal'])
    assert sha256(cfg.plane_recovery.nominal_parameters_path) == identity['native_nominal_sha256']
    cfg.plane_recovery.nominal_parameters_path = str(b)
    with pytest.raises(ValueError, match='actual solver input SHA'):
        assert_native_inputs(cfg, identity)
    current['plane_recovery']['z_sole'] = -.1
    with pytest.raises(ValueError, match='configuration differs'):
        bind_native_inputs(cfg, identity, snapshot)
    # Old source changes cannot mutate the files already supplied to workers.
    (f.path.parent / 'native_nominal.yaml').write_text('changed source')
    assert verify_native_snapshot(identity, snapshot)
    with pytest.raises(ValueError, match='source changed'):
        snapshot_checkpoint(tmp_path / 'output', identity)
    paths['native_nominal'].chmod(0o644)
    paths['native_nominal'].write_text('tampered snapshot')
    with pytest.raises(ValueError, match='snapshot SHA mismatch'):
        verify_native_snapshot(identity, snapshot)


def test_run_archive_revalidates_native_files_and_effective_inputs(checkpoint_factory, tmp_path):
    f = checkpoint_factory()
    identity = inspect(f)
    run = synthetic_run(tmp_path / 'reports')
    path = run['path']
    (path / 'completion.json').unlink()
    identity.update({k: run['identity'][k] for k in protocol.COMPATIBILITY})
    identity['synthetic'] = True
    write_json(path / 'identity.json', identity)
    write_json(path / 'native_configuration.json', identity['native_configuration'])
    for role, resource in identity['resources'].items():
        if role != 'estimator':
            atomic_write(path / resource['snapshot'], identity.inputs[role].read_bytes())
    store = protocol.RunStore(path)
    store.validate(require_complete=False, allow_synthetic=True)
    physics = {'synthetic': True}
    identity['actual_physics_hash'] = digest(physics)
    write_json(path / 'identity.json', identity)
    effective = {'actual_physics': physics, 'native_configuration_sha256': identity['native_configuration_sha256'],
                 'native_inputs': {k: v['sha256'] for k, v in identity['resources'].items()}}
    effective['native_inputs']['native_nominal'] = 'different-runtime-B'
    atomic_write(path / 'effective_env_config.yaml', yaml.safe_dump(effective))
    with pytest.raises(ValueError, match='Effective native inputs'):
        store.validate(require_complete=False, allow_synthetic=True)
    effective['native_inputs']['native_nominal'] = identity['native_nominal_sha256']
    atomic_write(path / 'effective_env_config.yaml', yaml.safe_dump(effective))
    store.validate(require_complete=False, allow_synthetic=True)
    atomic_write(path / identity['resources']['native_capability']['snapshot'], b'tampered')
    with pytest.raises(ValueError, match='native_capability snapshot mismatch'):
        store.validate(require_complete=False, allow_synthetic=True)


def test_main_table_and_budget_lineage_selection(tmp_path, monkeypatch):
    early = synthetic_run(tmp_path, 'early', full=True)
    early['identity']['checkpoint_stage'] = 'intermediate'
    final = synthetic_run(tmp_path, 'final', full=True)
    final['identity'].update(training_iteration=9999, training_run_id='training-A', training_transitions=2400000)
    early['identity'].update(training_run_id='training-A', training_transitions=240000, training_iteration=1000)
    repeat = copy.deepcopy(early)
    repeat['id'] = 'independent'; repeat['identity']['training_run_id'] = 'training-B'
    assert previous_checkpoint(final, [repeat]) is None
    assert previous_checkpoint(final, [early, repeat]) is early
    assert not matched_ablation(early, final)
    assert '不同预算' in budget_relation(early, final)
    equal = copy.deepcopy(final)
    equal['identity']['method'] = 'rl_only'
    assert matched_ablation(final, equal)
    equal['identity']['training_transitions'] = None
    assert not matched_ablation(final, equal)
    for run in (early, final):
        path = tmp_path / 'runs' / run['id']
        write_json(path / 'identity.json', run['identity'])
        write_json(path / 'completion.json', {'synthetic': run['id']})
        atomic_write(path / 'manifest_snapshot.jsonl', protocol.jsonl(run['manifest']))
    monkeypatch.setattr(protocol.RunStore, 'validate', lambda self: (protocol.read_json(self.path / 'identity.json'), early['records']))
    selected, _, history = load_runs(tmp_path)
    assert [r['id'] for r in selected] == ['final']
    text = markdown(report_blocks(tmp_path, [early, final], [], [early, final], plots=False), tmp_path / 'report')
    main, evolution = text.split('## 各模型历次 checkpoint 的变化')
    assert '| early |' not in main and '| early |' in evolution


def test_numpy_old_and_new_interfaces_and_cleanup(monkeypatch):
    expected = 7.5  # trapezoids through (0,1), (1,2), (3,4)
    def integrate(y, x):
        return sum((a + b) * (right - left) / 2 for a, b, left, right in zip(y, y[1:], x, x[1:]))
    for api in (NS(trapz=integrate), NS(trapezoid=integrate)):
        with monkeypatch.context() as patch:
            patch.setattr(metrics, 'np', api)
            assert metrics.trapezoid_integral([1, 2, 4], [0, 1, 3]) == expected
    plan = next(r for r in PLANS if r['command_name'] == 'standing')
    machine = metrics.TrialMachine(plan, P, None)
    machine.feed(frame(0)); machine.feed(frame(2)); machine.feed(frame(12))
    def failing(*args):
        raise RuntimeError('synthetic integration error')
    monkeypatch.setattr(metrics, 'trajectory_metrics', failing)
    with pytest.raises(RuntimeError):
        machine.result()
    machine.finish('EVALUATION_ERROR')
    result = machine.result(include_metrics=False)
    assert result['status'] == 'EVALUATION_ERROR' and result['metrics_unavailable']


def test_iqr_wilson_and_partial_denominators(tmp_path):
    q = metrics.quantiles([1, 2, 3, 4, 5])
    assert (q['q1'], q['q3'], q['iqr'], q['median']) == (2, 4, 2, 3)
    ci = metrics.binomial_rate(5, 10, 'executed')['wilson_95']
    assert ci == pytest.approx([.2365930905, .7634069095])
    assert metrics.binomial_rate(0, 0, 'executed')['wilson_95'] is None
    run = synthetic_run(tmp_path)
    plans = [r for r in run['manifest'] if r['experiment'] == 'E1']
    plans += [{**plans[0], 'trial_id': 'pending'}]
    failure = {**run['records'][1], 'status': 'PRECONDITION_FAILED', 'survived': False, 'push_applied': False,
               'precondition_passed': False}
    s = metrics.summarize([failure], plans)
    assert s['status'] == 'PARTIAL'
    assert s['E1']['pending'] == s['E1']['failures_executed'] == 1
    assert s['E1']['failure_rate'] == .5  # pending is not a second failure
    assert s['E1']['rates']['failure_rate']['wilson_95'] is None
    assert s['E1']['rates']['executed_failure_rate']['rate'] == 1


def test_historical_markdown_images_survive_current_deletion(tmp_path):
    run = synthetic_run(tmp_path)
    report = tmp_path / 'report'
    path = report / 'G1_RECOVERY_EXPERIMENTS.md'
    for value in (1., 2., 3.):
        preserve_existing(report)
        run['records'][1]['recovery_time'] = value
        run['summary'] = metrics.summarize(run['records'], run['manifest'])
        images = curve_plots(tmp_path, [run], 'same-group')
        atomic_write(path, markdown([('image', str(p), 'synthetic curve') for p in images], report))
        write_json(report / 'generated_files.json', {path.name: sha256(path)})
    shutil.rmtree(report / 'figures')
    archived = list((report / 'history').glob('*/*.md'))
    assert len(archived) == 2
    image_hashes = []
    for old in archived:
        for link in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', old.read_text()):
            image = old.parent / link
            assert image.read_bytes().startswith(b'\x89PNG')
            image_hashes.append(sha256(image))
    assert len(set(image_hashes)) >= 3


@pytest.mark.parametrize('category,expected', [('lookup', 0), ('adapter', 0), ('communication', 0), ('numerical', 1)])
def test_query_events_distinct_from_ten_held_frames(category, expected):
    plane = {'context_valid': False, 'standing_na': False, 'invalid_reason': category, 'N': None, 'margin': None,
             'query_id': 3, 'query_failed': True, 'query_category': category,
             **{k: 0. for k in ('reward_progress', 'reward_step_cost', 'reward_td5', 'reward_total')}}
    frames = [frame(i*.02, plane={**plane, 'query_event': i == 0}) for i in range(10)]
    result = metrics.trajectory_metrics(frames, PLANS[0])['plane_diagnostics']
    assert result['query_failures'] == 1 and result['invalid_context_frames'] == 10
    assert result['numerical_failure_events'] == expected
    for f in frames:
        f['plane'].update(standing_na=True, invalid_reason='standing', query_event=False)
    result = metrics.trajectory_metrics(frames, PLANS[0])['plane_diagnostics']
    assert result['query_failures'] == result['numerical_failure_events'] == result['invalid_context_frames'] == 0


def test_training_budget_counter_is_independent_of_logging_and_run_name(tmp_path):
    sys.path.insert(0, str(LAB / 'rsl_rl'))
    from rsl_rl.utils.training_provenance import advance_training_transitions, new_training_provenance
    params = tmp_path / 'params'
    atomic_write(params / 'agent.yaml', 'run_name: arbitrary\n')
    atomic_write(params / 'env.yaml', 'synthetic: true\n')
    first = new_training_provenance('task', params, {})
    second = new_training_provenance('task', params, {}, first)
    assert first['training_run_id'] != second['training_run_id']
    assert second['parent_training_run_id'] == first['training_run_id']
    runner = NS(training_transitions=24, num_steps_per_env=24, env=NS(num_envs=2), gpu_world_size=2)
    advance_training_transitions(runner)
    assert runner.training_transitions == 120
    runner.training_transitions = None
    advance_training_transitions(runner)
    assert runner.training_transitions is None


@pytest.mark.parametrize('runner_name', ['OnPolicyRunner', 'DWAQOnPolicyRunner'])
def test_runner_save_and_resume_preserve_actual_budget(tmp_path, runner_name):
    sys.path.insert(0, str(LAB / 'rsl_rl'))
    import rsl_rl.runners as runners
    from rsl_rl.modules.actor_critic import ActorCritic
    cls = getattr(runners, runner_name)
    runner = cls.__new__(cls)
    model = ActorCritic(4, 5, 2, actor_hidden_dims=[8], critic_hidden_dims=[8])
    runner.alg = NS(policy=model, optimizer=torch.optim.Adam(model.parameters()), rnd=None)
    runner.empirical_normalization = False
    runner.logger_type = 'tensorboard'; runner.disable_logs = True
    runner._save_run_metadata = lambda: None
    runner.current_learning_iteration = 1000
    runner.training_provenance = {'task_name': 'synthetic', 'training_run_id': 'training-A'}
    runner.training_transitions = 24000
    target = tmp_path / 'model.pt'
    runner.save(str(target))
    runner.training_transitions = 0
    runner.load(str(target))
    assert runner.training_transitions == 24000
    assert runner.training_provenance['training_run_id'] == 'training-A'
    legacy = torch.load(target, weights_only=False)
    del legacy['training_provenance']
    torch.save(legacy, target)
    runner.load(str(target))
    assert runner.training_transitions is None


@pytest.mark.parametrize('category', ['lookup', 'adapter', 'communication', 'numerical'])
def test_native_refresh_counts_only_real_numerical_queries(category):
    # Execute the native method with CPU tensors, retaining its ZOH early return.
    tree = ast.parse((LAB / 'legged_lab/envs/g1/g1_plane_v1_env.py').read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_refresh_plane_v1_context')
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {'torch': torch, 'time': time,
                 'matched_command_standing_mask': lambda command: command[:, :2].norm(dim=1) < 1e-8,
                 'normalize_recovery_context': lambda n, margin, valid: torch.zeros(len(n), 3)}
    exec(compile(ast.fix_missing_locations(module), 'native_refresh', 'exec'), namespace)
    env = NS()
    for key in ('_context_touchdown_mask', '_context_refresh_mask', '_touchdown_certificate_cache_mask',
                '_touchdown_certificate_cache_valid', 'current_certificate_valid', '_v1_last_geometry_valid',
                '_v1_last_solver_valid', '_v1_intentional_not_applicable'):
        setattr(env, key, torch.zeros(1, dtype=torch.bool))
    for key in ('_touchdown_certificate_cache_n', 'current_n_min'):
        setattr(env, key, torch.zeros(1, dtype=torch.long))
    for key in ('_touchdown_certificate_cache_margin', 'current_margin'):
        setattr(env, key, torch.zeros(1))
    env._recovery_context = torch.zeros(1, 3)
    env._context_n_counts = torch.zeros(7, dtype=torch.long)
    for key in ('_v1_intentional_not_applicable_count', '_v1_geometry_invalid_count', '_v1_query_failure_count',
                '_v1_solver_failure_count', '_context_refresh_batches', '_context_refresh_evaluations',
                '_context_valid_evaluations', '_context_refresh_total_seconds'):
        setattr(env, key, 0)
    env._context_refresh_latencies_s = []; env._context_refresh_batch_sizes = []
    env._v1_context_diagnostics = [{}]; env._v1_query_failure_categories = {}
    env._certificate_evaluator = NS(_record_profile=lambda *args: None,
        evaluate_with_validity=lambda *args: (torch.tensor([6]), torch.tensor([-3.]), torch.tensor([False])),
        last_query_diagnostics={0: {'query_id': 1, 'failed': True, 'category': category}})
    state = NS(touchdown=torch.tensor([True]), terrain_plane_valid=torch.tensor([True]),
               command_velocity=torch.tensor([[.4, 0., 0.]]))
    refresh = namespace['_refresh_plane_v1_context']
    refresh(env, state, torch.tensor([False]), torch.tensor([True]))
    state.touchdown[:] = False
    for _ in range(9):
        refresh(env, state, torch.tensor([False]), torch.tensor([True]))
    assert env._v1_query_failure_count == 1
    assert env._v1_solver_failure_count == int(category == 'numerical')
    assert env._v1_context_diagnostics[0]['category'] == category
    assert not bool(env.current_certificate_valid[0])
