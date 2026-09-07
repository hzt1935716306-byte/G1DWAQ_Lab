"""Simulator-free acceptance tests for the final evaluation isolation polish."""
import ast
import copy
import io
import zipfile
from types import SimpleNamespace as NS
from xml.etree import ElementTree as ET

import pytest
import torch

from g1_recovery_protocol import (LAB, TASKS, code_identity, digest, evaluation_key,
                                  load_prepared, load_yaml, prepare, atomic_write)
from g1_recovery_report import (compatible_run, is_development_subset, is_full_lite_run,
                                markdown, report_blocks, table, table_widths,
                                validate_detector_source, word_document)
from test_protocol_report import synthetic_run


def test_formal_main_table_requires_final_full_lite_and_keeps_smoke(tmp_path):
    formal = synthetic_run(tmp_path, 'formal', full=True)
    smoke = synthetic_run(tmp_path, 'smoke')
    smoke['identity']['checkpoint_stage'] = 'final'
    intermediate = copy.deepcopy(formal)
    intermediate['id'] = 'intermediate'
    intermediate['identity']['model_alias'] = 'intermediate'
    intermediate['identity']['checkpoint_stage'] = 'intermediate'
    _, full_manifest, _ = load_prepared(tmp_path)

    assert is_full_lite_run(formal, full_manifest)
    assert not is_full_lite_run(smoke, full_manifest)
    assert not is_full_lite_run(intermediate, full_manifest)
    assert is_development_subset(smoke)

    models = load_yaml(tmp_path / 'models.yaml')
    models['models'].append({'model_alias': 'smoke', 'method': 'ppo_plain',
                            'task_name': TASKS['ppo_plain'],
                            'checkpoint_sha256': smoke['identity']['checkpoint_sha256'],
                            'checkpoint_stage': 'final'})
    import yaml
    atomic_write(tmp_path / 'models.yaml', yaml.safe_dump(models, sort_keys=False))
    blocks = report_blocks(tmp_path, [formal, smoke, intermediate], [],
                           [formal, smoke, intermediate], plots=False)
    rendered = markdown(blocks, tmp_path / 'report')
    development = rendered.split('## Development / Smoke Evaluation', 1)[1].split('## E0 正常运动总表', 1)[0]
    formal_tables = rendered.split('## E0 正常运动总表', 1)[1].split('## 各模型历次 checkpoint 的变化', 1)[0]
    assert 'smoke' in development and 'DEV_SUBSET_COMPLETE' in rendered
    assert 'formal' in formal_tables
    assert '| smoke |' not in formal_tables and '| intermediate |' not in formal_tables


def test_docx_model_table_keeps_all_nine_columns():
    headers = ['模型', 'Task', '迭代', '训练批次 ID', 'Transitions', 'Seed', '阶段', '状态', 'SHA 前缀']
    row = ['model', 'task', '9999', 'training-A', '240000', '42', 'final',
           'FULL_EVAL_COMPLETE', 'abcdef123456']
    data = word_document([table(headers, [row])])
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        root = ET.fromstring(archive.read('word/document.xml'))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    rows = root.findall('.//w:tbl[1]/w:tr', ns)
    assert [len(item.findall('w:tc', ns)) for item in rows] == [9, 9]
    text = ''.join(root.itertext())
    assert '状态' in text and 'SHA 前缀' in text


def test_docx_rejects_width_or_row_column_mismatch():
    with pytest.raises(ValueError, match='width count'):
        table_widths(['A', 'B'], [1])
    with pytest.raises(ValueError, match='row has 1 cells'):
        word_document([table(['A', 'B'], [['only-A']])])


def test_begin_batch_synchronizes_independent_origin_tensors():
    tree = ast.parse((LAB / 'tools/evaluation/g1_recovery_eval.py').read_text())
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == 'EvaluationEnv')
    native = type('Native', (), {})
    namespace = {'native': native, 'torch': torch, 'p': {'slopes_deg': [-10, 0, 10]},
                 'metrics_cfg': {'test_area_half_extent_m': 24}}
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), 'evaluation_env', 'exec'), namespace)
    env = namespace['EvaluationEnv'].__new__(namespace['EvaluationEnv'])
    env.num_envs = 2
    env.device = 'cpu'
    terrain_origins = torch.full((2, 3), -100.)
    scene_origins = torch.full((2, 3), 100.)
    assert terrain_origins.data_ptr() != scene_origins.data_ptr()
    env.scene = NS(terrain=NS(terrain_levels=torch.zeros(2, dtype=torch.long),
                              terrain_types=torch.zeros(2, dtype=torch.long),
                              env_origins=terrain_origins),
                   env_origins=scene_origins)
    env.eval_origin_table = torch.tensor([[[1., 0., 10.], [2., 0., 20.], [3., 0., 30.]]])
    env.eval_slopes = torch.zeros(2)
    env.eval_commands = torch.zeros(2, 3)
    env.eval_history_calls = 0
    env.extras = {'observations': {}}
    reset_checked = []
    env.reset = lambda ids: reset_checked.append(torch.equal(env.scene.env_origins,
                                                               env.scene.terrain.env_origins))
    def observations():
        env.eval_history_calls += 1
        return torch.zeros(2, 1), env.extras
    env.get_observations = observations
    plans = [{'slope_deg': -10, 'command_vx': .4, 'command_vy': 0., 'command_yaw': 0.},
             {'slope_deg': 10, 'command_vx': 0., 'command_vy': .4, 'command_yaw': 0.}]
    env.begin_batch(plans)
    assert reset_checked == [True]
    assert torch.equal(env.scene.env_origins, torch.tensor([[1., 0., 10.], [3., 0., 30.]]))
    assert torch.equal(env.scene.env_origins, env.scene.terrain.env_origins)
    assert env.scene.env_origins.data_ptr() != env.scene.terrain.env_origins.data_ptr()


def _identity():
    return {'protocol_hash': 'protocol', 'manifest_hash': 'manifest', 'metrics_version': 'metrics',
            'metrics_config_hash': 'metrics-config', 'metrics_reference_sha256': 'reference',
            'physics_profile_hash': 'physics', 'inference_mode': 'native',
            'evaluation_runtime_sha256': 'runtime-A', 'report_code_sha256': 'report-A',
            'task_name': 'task', 'checkpoint_sha256': 'checkpoint', 'estimator_sha256': None,
            'native_nominal_sha256': None, 'agent_config_sha256': 'agent', 'env_config_sha256': 'env',
            'identity_schema_version': 2, 'native_capability_sha256': None,
            'native_configuration_sha256': 'native-config', 'training_run_id': 'training-A',
            'training_transitions': 100, 'checkpoint_stage': 'final', 'actual_physics_hash': 'actual',
            'realized_environment_hash': 'realized'}


def test_report_hash_does_not_change_evaluation_key_but_runtime_does():
    identity = _identity()
    report_changed = {**identity, 'report_code_sha256': 'report-B'}
    runtime_changed = {**identity, 'evaluation_runtime_sha256': 'runtime-B'}
    assert evaluation_key(identity) == evaluation_key(report_changed)
    assert evaluation_key(identity) != evaluation_key(runtime_changed)
    run = {'identity': identity}
    assert compatible_run(run, {'identity': report_changed})
    assert not compatible_run(run, {'identity': runtime_changed})
    sources = code_identity()
    assert 'tools/evaluation/g1_recovery_report.py' not in sources['evaluation_runtime_sources']
    assert not any('/tests/' in name for name in sources['evaluation_runtime_sources'])
    assert 'tools/evaluation/g1_recovery_report.py' in sources['report_code_sources']


def detector_run(tmp_path, count=2):
    run = synthetic_run(tmp_path, 'detector-source')
    _, full, prepared = load_prepared(tmp_path)
    subset = copy.deepcopy([row for experiment in ('E0', 'E1')
                            for row in [item for item in full if item['experiment'] == experiment][:count]])
    run['manifest'] = subset
    run['identity'].update({**prepared, 'synthetic': False, 'checkpoint_stage': 'final',
                            'subset': f'first_{count}_per_experiment',
                            'manifest_hash': digest(subset), 'evaluation_runtime_sha256': 'runtime-current',
                            'actual_physics_hash': 'actual-physics'})
    return run, full, prepared


def test_detector_accepts_exact_current_development_subset(tmp_path):
    run, full, prepared = detector_run(tmp_path)
    source = validate_detector_source(run, full, prepared, 'runtime-current')
    assert source['evaluation_id'] == run['id']
    assert source['subset'] == 'first_2_per_experiment'
    assert source['manifest_hash'] == digest(run['manifest'])
    assert source['actual_physics_hash'] == 'actual-physics'


@pytest.mark.parametrize('mismatch', ['manifest', 'metrics', 'physics', 'runtime', 'protocol'])
def test_detector_rejects_incompatible_or_modified_sources(tmp_path, mismatch):
    run, full, prepared = detector_run(tmp_path)
    if mismatch == 'manifest':
        run['manifest'][0]['command_vx'] += .01
        run['identity']['manifest_hash'] = digest(run['manifest'])
    elif mismatch == 'runtime':
        run['identity']['evaluation_runtime_sha256'] = 'runtime-other'
    elif mismatch == 'protocol':
        run['identity']['protocol_hash'] = 'protocol-other'
    else:
        field = {'metrics': 'metrics_config_hash', 'physics': 'physics_profile_hash'}[mismatch]
        run['identity'][field] = field + '-other'
    with pytest.raises(ValueError, match={'manifest': 'deterministic subset', 'runtime': 'runtime',
                                          'protocol': 'protocol_hash', 'metrics': 'metrics_config_hash',
                                          'physics': 'physics_profile_hash'}[mismatch]):
        validate_detector_source(run, full, prepared, 'runtime-current')
