"""Training identity bookkeeping; independent of display names and log counters."""
import hashlib
from pathlib import Path
import uuid


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def new_training_provenance(task, params_dir, resource_paths, parent=None):
    params = Path(params_dir)
    return {'schema_version': 1, 'task_name': task, 'training_run_id': str(uuid.uuid4()),
            'parent_training_run_id': (parent or {}).get('training_run_id'),
            'agent_config_sha256': file_sha256(params / 'agent.yaml'),
            'env_config_sha256': file_sha256(params / 'env.yaml'),
            'resources': {role: {'sha256': file_sha256(path)} for role, path in resource_paths.items()}}


def advance_training_transitions(runner):
    # Unknown legacy/warm-start pretraining budgets must remain unknown.
    if getattr(runner, "training_transitions", None) is not None:
        runner.training_transitions += (
            runner.num_steps_per_env * runner.env.num_envs * getattr(runner, "gpu_world_size", 1)
        )
