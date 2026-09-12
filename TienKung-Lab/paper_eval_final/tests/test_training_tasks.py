import copy

import pytest


def test_all_registered_training_task_configs_import_and_instantiate():
    pytest.importorskip("pxr", reason="Isaac task import is validated after AppLauncher in the simulator smoke")
    import legged_lab.envs  # noqa: F401
    from legged_lab.utils import task_registry

    assert task_registry.env_cfgs
    for name in sorted(task_registry.env_cfgs):
        env_cfg, agent_cfg = task_registry.get_cfgs(name)
        assert copy.deepcopy(env_cfg).to_dict()
        assert copy.deepcopy(agent_cfg).to_dict()
