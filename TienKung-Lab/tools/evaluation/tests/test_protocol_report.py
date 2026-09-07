"""All fixtures are explicitly synthetic and confined to pytest temporary directories."""
import ast
import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
from xml.etree import ElementTree as ET
import numpy as np
import pytest
import yaml
from g1_recovery_protocol import *
from g1_recovery_metrics import summarize
from g1_recovery_report import build_report, report_blocks, markdown, word_document, compatible_run, analyze_traces


def test_manifest_count_balance_and_idempotence(tmp_path):
    info = prepare(tmp_path)
    p, rows, _ = load_prepared(tmp_path)
    assert len(rows) == 390
    assert len({r['trial_id'] for r in rows}) == 390
    assert sum(r['experiment'] == 'E0' for r in rows) == 150
    assert sum(r['reference_touchdown_foot'] == 'left' for r in rows) == 120
    assert sum(r['reference_touchdown_foot'] == 'right' for r in rows) == 120
    files = {str(f): sha256(f) for f in (tmp_path / 'manifests').glob('*')}
    assert prepare(tmp_path) == info
    assert files == {str(f): sha256(f) for f in (tmp_path / 'manifests').glob('*')}
    assert all(len(r['initial_joint_parameters']['position_scale']) == 29 for r in rows)
    balances = read_json(tmp_path / 'manifests/reference_foot_counts.json')
    assert all(sorted(x.values()) == [2, 3] for x in balances.values())
    assert rows == generate_manifest(p)  # no runtime RNG/reset branch dependence


def test_manifest_tampering_and_changed_protocol_fail(tmp_path):
    prepare(tmp_path)
    path = tmp_path / 'manifests/nominal_lite_v1.jsonl'
    rows = read_jsonl(path); rows[0]['initial_pose_parameters']['x'] += .1
    atomic_write(path, jsonl(rows))
    with pytest.raises(ValueError, match='integrity'):
        load_prepared(tmp_path)
    p = load_yaml(DEFAULT_PROTOCOL); p['e1']['magnitudes'] = [.6, 1.]
    config = tmp_path / 'changed.yaml'; atomic_write(config, yaml.safe_dump(p))
    with pytest.raises(ValueError, match='changed'):
        prepare(tmp_path, config)


def test_reference_exact_lookup_no_clamp(tmp_path):
    bad = load_yaml(DEFAULT_PROTOCOL); bad['metrics']['version'] = 'unimplemented_detector'
    config = tmp_path / 'bad-version.yaml'; atomic_write(config, yaml.safe_dump(bad))
    with pytest.raises(ValueError, match='Unsupported metrics definition'):
        prepare(tmp_path / 'bad-output', config)
    p = load_yaml(DEFAULT_PROTOCOL)
    p['slopes_deg'] = [10.1]
    with pytest.raises(ValueError, match='exact node'):
        nominal_reference(p)
    p = load_yaml(DEFAULT_PROTOCOL)
    source, _ = nominal_reference(p)
    ref = load_yaml(source)
    node = next(n for n in ref['nominal_plane_gait']['nodes'] if n['slope_degrees'] == -10 and n['direction'] == '+x' and n['speed'] == .4)
    del node['roll_star']
    atomic_write(tmp_path / 'ref.yaml', yaml.safe_dump(ref))
    p['metrics']['reference'] = str(tmp_path / 'ref.yaml')
    with pytest.raises(KeyError):
        nominal_reference(p)


def test_yaml_python_tags_are_data(tmp_path):
    path = tmp_path / 'data.yaml'
    path.write_text('x: !!python/object/apply:os.system ["exit 99"]\n')
    assert load_yaml(path)['x'] == ['exit 99']


def test_offline_commands_do_not_import_isaac(tmp_path):
    code = '''
import builtins, sys
real = builtins.__import__
def guarded(name, *args, **kw):
    if name.startswith(('isaac', 'omni', 'torch')):
        raise AssertionError('offline command imported ' + name)
    return real(name, *args, **kw)
builtins.__import__ = guarded
from g1_recovery_eval import main
main(['prepare','--output_root',sys.argv[1]])
main(['report','--output_root',sys.argv[1]])
'''
    subprocess.run([sys.executable, '-c', code, str(tmp_path)], env={**os.environ, 'PYTHONPATH': str(LAB / 'tools/evaluation')}, check=True, capture_output=True)


def synthetic_run(tmp_path, alias='synthetic_A', full=False):
    """Construct a marked synthetic in-memory result for report presentation only."""
    prepare(tmp_path)
    p, rows, info = load_prepared(tmp_path)
    plans = rows if full else [rows[0], next(r for r in rows if r['experiment'] == 'E1')]
    i = {**info, 'synthetic': True, 'model_alias': alias, 'method': 'ppo_plain', 'task_name': TASKS['ppo_plain'],
         'checkpoint_sha256': digest(alias), 'estimator_sha256': None, 'native_nominal_sha256': None,
         'agent_config_sha256': 'test', 'env_config_sha256': 'test',
         'evaluation_runtime_sha256': 'test_runtime', 'report_code_sha256': 'test_report',
         'actual_physics_hash': 'synthetic_physics', 'training_iteration': 100, 'training_seed': 42,
         'checkpoint_stage': 'final' if full else 'intermediate',
         'subset': 'lite_full' if full else 'first_1_per_experiment'}
    i['manifest_hash'] = digest(plans)
    path = tmp_path / 'synthetic_only' / alias
    path.mkdir(parents=True)
    write_json(path / 'identity.json', i)
    atomic_write(path / 'protocol_snapshot.yaml', (tmp_path / 'protocol.yaml').read_bytes())
    atomic_write(path / 'metrics_reference.yaml', (tmp_path / 'metrics_reference.yaml').read_bytes())
    atomic_write(path / 'manifest_snapshot.jsonl', jsonl(plans))
    records = []
    for plan in plans:
        records.append({**plan, 'status': 'RECOVERED_AND_SURVIVED' if plan['experiment'] == 'E1' else 'ALIVE_NOT_RECOVERED',
                        'survived': True, 'precondition_passed': True, 'push_applied': plan['experiment'] == 'E1',
                        'recovery_time': 1., 'recovery_steps': 3, 'recovery_within_3_touchdowns': True, 'recovery_within_5_touchdowns': True})
    # Never register these fixtures in the formal registry.
    write_json(path / 'completion.json', {'synthetic': True, 'status': 'COMPLETE', 'files': {}})
    return dict(id=alias, identity=i, records=records, manifest=plans, summary=summarize(records, plans), path=path)


def test_synthetic_cannot_import_or_register(tmp_path):
    run = synthetic_run(tmp_path / 'fixture')
    with pytest.raises(ValueError, match='Synthetic'):
        RunStore(run['path']).validate()
    target = tmp_path / 'formal'; prepare(target)
    with pytest.raises(ValueError, match='Synthetic'):
        import_results(run['path'], target)
    assert not list((target / 'runs').glob('*'))


def test_synthetic_report_tables_pairing_and_docx(tmp_path):
    run = synthetic_run(tmp_path, full=True)
    other = copy.deepcopy(run)
    other['id'] = 'synthetic_B'; other['identity']['model_alias'] = 'synthetic_B'; other['identity']['method'] = 'dwaq'
    blocks = [('text', 'SYNTHETIC TEST DATA — NOT EXPERIMENT RESULTS')] + report_blocks(tmp_path, [run, other], [], [run, other])
    text = markdown(blocks, tmp_path / 'report')
    assert '共同成功 n' in text and '240/240 (240 pushed)' in text and '待评测' in text
    assert 'SYNTHETIC TEST DATA' in text
    data = word_document(blocks)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith('.xml'):
                ET.fromstring(z.read(name))
        assert b'PAGE' in z.read('word/footer1.xml')
        assert any(name.endswith('.png') for name in z.namelist())
    atomic_write(tmp_path / 'report/SYNTHETIC_QA.docx', data)


def test_incompatible_metrics_manifest_and_physics_isolated(tmp_path):
    a = synthetic_run(tmp_path)
    for key in COMPATIBILITY:
        b = copy.deepcopy(a); b['identity'][key] = 'different'
        assert not compatible_run(a, b)
    b = copy.deepcopy(a); b['identity']['actual_physics_hash'] = 'different'
    assert not compatible_run(a, b)


def test_report_notes_and_manual_word_migration(tmp_path):
    prepare(tmp_path)
    note = {'notes': {'ppo_plain': '人工记录：这条备注必须保留'}}
    atomic_write(tmp_path / 'report/notes.yaml', yaml.safe_dump(note, allow_unicode=True))
    build_report(tmp_path)
    path = tmp_path / 'report/G1_RECOVERY_EXPERIMENTS.docx'
    old = sha256(path)
    atomic_write(path, word_document([('text', '用户在 Word 中直接写入的分析')]))
    build_report(tmp_path)
    notes = load_yaml(tmp_path / 'report/notes.yaml')
    assert notes['notes'] == note['notes']
    assert '用户在 Word' in notes['migrated_document_edits'][0]['preserved_text']
    assert list((tmp_path / 'report/history').glob('*/*.docx'))
    text = (tmp_path / 'report/G1_RECOVERY_EXPERIMENTS.md').read_text()
    assert '人工记录' in text and '用户在 Word' in text
    assert old != sha256(path)


def test_per_trial_commit_resume_and_no_overwrite(tmp_path):
    run = synthetic_run(tmp_path)
    (run['path'] / 'completion.json').unlink()
    store = RunStore(run['path'])
    r = run['records'][0]
    store.save_trial(r, {'time': np.array([0., .02])}, [{'event': 'synthetic'}])
    assert len(store.records()) == 1
    with pytest.raises(ValueError, match='already committed'):
        store.save_trial(r, {'time': np.array([0., .02])}, [])
    # Crash after trace but before JSON commit must leave this trial missing.
    atomic_write(run['path'] / 'traces/orphan.npz', b'incomplete')
    assert {r['trial_id'] for r in store.records()} == {run['manifest'][0]['trial_id']}
    store.rebuild_tables()
    assert len(list(csv.DictReader((run['path'] / 'trials.csv').open()))) == 1


def test_evaluation_key_ignores_path_alias_but_not_weights_or_metrics(tmp_path):
    run = synthetic_run(tmp_path)
    a = run['identity']; b = {**a, 'model_alias': 'renamed', 'checkpoint_path': 'logs/another/model.pt'}
    assert evaluation_key(a) == evaluation_key(b)
    assert evaluation_key(a) != evaluation_key({**b, 'checkpoint_sha256': 'other'})
    assert evaluation_key(a) != evaluation_key({**b, 'metrics_version': 'other'})


def test_real_baselines_identity_cpu_and_wrong_task():
    path = LAB / 'logs/g1_slope_nosys_d_matched/2026-09-05_18-18-07_slope_nosys_d_matched_seed42/model_9999.pt'
    if not path.exists():
        pytest.skip('local baseline not supplied on this machine')
    i = inspect_checkpoint(TASKS['ppo_plain'], path, 'cpu_audit', 'final')
    assert i['training_iteration'] == 9999 and i['training_seed'] == 42
    assert i['actor_input_dimension'] == 960 and i['action_dimension'] == 29
    assert i['checkpoint_path'].startswith('logs/')
    assert not Path(i['checkpoint_path']).is_absolute()
    with pytest.raises(ValueError, match='task mismatch'):
        inspect_checkpoint(TASKS['ppo_symmetric'], path, 'wrong')


def test_persisted_paths_are_project_relative(tmp_path):
    inside = LAB / 'tools/evaluation/configs/g1_recovery_eval_lite_v1.yaml'
    assert portable_path(inside) == 'tools/evaluation/configs/g1_recovery_eval_lite_v1.yaml'
    with pytest.raises(ValueError, match='inside the project'):
        portable_path(tmp_path)


@pytest.mark.parametrize('mutation', ['dimension', 'normalizer', 'estimator'])
def test_bad_checkpoint_contracts(tmp_path, mutation):
    import torch
    src = LAB / 'logs/g1_slope_nosys_d_matched/2026-09-05_18-18-07_slope_nosys_d_matched_seed42'
    if not src.exists():
        pytest.skip('local baseline absent')
    shutil.copytree(src / 'params', tmp_path / 'params')
    c = torch.load(src / 'model_9999.pt', map_location='cpu', weights_only=False)
    a = load_yaml(tmp_path / 'params/agent.yaml')
    e = load_yaml(tmp_path / 'params/env.yaml')
    task = TASKS['ppo_plain']
    if mutation == 'dimension':
        c['model_state_dict']['actor.0.weight'] = torch.zeros(512, 959)
    elif mutation == 'normalizer':
        a['empirical_normalization'] = True
    else:
        task = TASKS['context_only']
        a['experiment_name'] = 'g1_plane_v1_matched'
        a['run_name'] = 'estimator_context_no_reward_matched'
        e['com_velocity_source'] = 'estimator'; e['plane_v1_reward'] = {'enabled': False}
        e['recovery_context'] = {'enabled': True, 'mode': 'certificate'}
        e['robot']['actor_obs_history_length'] = 5
        c['model_state_dict']['actor.0.weight'] = torch.zeros(512, 483)
    atomic_write(tmp_path / 'params/agent.yaml', yaml.safe_dump(a))
    atomic_write(tmp_path / 'params/env.yaml', yaml.safe_dump(e))
    torch.save(c, tmp_path / 'model.pt')
    with pytest.raises(ValueError, match={'dimension': 'dimensions', 'normalizer': 'normalizer', 'estimator': 'requires --estimator'}[mutation]):
        inspect_checkpoint(task, tmp_path / 'model.pt', 'bad')


def test_import_dedup_collision_and_integrity_with_synthetic_validator(tmp_path, monkeypatch):
    """Use a test-only validator for the marked fixture; production rejection tested above."""
    run = synthetic_run(tmp_path / 'fixture')
    def fixture_validate(self, require_complete=True):
        assert self.path.is_relative_to(tmp_path)
        assert read_json(self.path / 'identity.json')['synthetic'] is True
        return read_json(self.path / 'identity.json'), run['records']
    monkeypatch.setattr(RunStore, 'validate', fixture_validate)
    root = tmp_path / 'target'; prepare(root)
    first = import_results(run['path'], root)
    second = import_results(run['path'], root)
    assert first == second
    assert len(list((root / 'runs').glob('*'))) == 1
    registry = read_json(root / 'registry.json')
    assert len(registry['evaluations']) == 1
    assert next(iter(registry['evaluations'].values()))['identity']['synthetic'] is True


def test_native_cpu_strict_forward_and_missing_parameter():
    import torch
    sys.path.insert(0, str(LAB / 'rsl_rl'))
    from rsl_rl.modules.actor_critic import ActorCritic
    from rsl_rl.modules.actor_critic_DWAQ import ActorCritic_DWAQ
    paths = [LAB / 'logs/g1_slope_nosys_d_matched/2026-09-05_18-18-07_slope_nosys_d_matched_seed42/model_9999.pt',
             LAB / 'logs/g1_dwaq_slope_nosys_d_matched/2026-09-05_18-18-46_dwaq_slope_nosys_d_matched_seed42/model_9999.pt']
    if not all(p.exists() for p in paths):
        pytest.skip('supplied baseline paths not available')
    models = [ActorCritic(960, 1010, 29, actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128]),
              ActorCritic_DWAQ(115, 307, 29, 480, 19, 96)]
    for index, (model, path) in enumerate(zip(models, paths)):
        state = torch.load(path, weights_only=False, map_location='cpu')['model_state_dict']
        model.load_state_dict(state, strict=True)
        model.eval(); model.requires_grad_(False)
        with torch.inference_mode():
            if index == 0:
                action = model.act_inference(torch.zeros(2, 960))
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(42)
                    action = model.act_inference(torch.zeros(2, 96), torch.zeros(2, 480))
        assert action.shape == (2, 29) and torch.isfinite(action).all()
        bad = dict(state); bad.pop('actor.0.bias')
        with pytest.raises(RuntimeError, match='Missing key'):
            model.load_state_dict(bad, strict=True)


def test_complete_storage_integrity_uses_real_validator(tmp_path, monkeypatch):
    run = synthetic_run(tmp_path)
    path = run['path']; (path / 'completion.json').unlink()
    store = RunStore(path)
    t = np.arange(601) * .02
    trace = {'time': t, 'root_velocity': np.zeros((601, 3)), 'root_velocity_world': np.zeros((601, 6)),
             'com_velocity': np.zeros((601, 3)), 'roll_pitch': np.zeros((601, 2)), 'yaw': np.zeros(601),
             'forces': np.zeros((601, 2)), 'root_position': np.zeros((601, 3)), 'fell': np.zeros(601, bool),
             'timeout': np.zeros(601, bool), 'out_of_test_area': np.zeros(601, bool), 'plane_json': np.full(601, 'null')}
    for r in run['records']:
        store.save_trial({**r, 'actual_push_time': 2}, trace, [{'event': 'synthetic_fixture'}])
    physics = {'synthetic': True}
    run['identity']['actual_physics_hash'] = digest(physics)
    write_json(path / 'identity.json', run['identity'])
    atomic_write(path / 'effective_env_config.yaml', yaml.safe_dump({'actual_physics': physics}))
    atomic_write(path / 'run.log', 'SYNTHETIC TEST ONLY\n')
    original_validate = RunStore.validate
    def test_validate(self, require_complete=True):
        assert self.path.is_relative_to(tmp_path)
        return original_validate(self, require_complete, allow_synthetic=True)
    monkeypatch.setattr(RunStore, 'validate', test_validate)
    store.complete(summarize(store.records(), run['manifest']))
    store.validate()
    target = path / 'traces' / (run['records'][0]['trial_id'] + '.npz')
    atomic_write(target, b'corrupt')
    with pytest.raises(ValueError, match='Corrupt trace'):
        store.validate()
