import pytest

torch = pytest.importorskip("torch")

from paper_eval_final.src.model_loader import dwaq_history_from_step_payload, resolve_baseline


def test_registered_dwaq_history_contracts_are_both_explicitly_supported():
    history = torch.zeros((4, 395))
    assert dwaq_history_from_step_payload(history) is history
    assert dwaq_history_from_step_payload({"observations": {"obs_hist": history}}) is history
    with pytest.raises(ValueError, match="history contract"):
        dwaq_history_from_step_payload({"observations": {"critic": history}})


@pytest.mark.parametrize(
    ("baseline_id", "model_id", "task_name"),
    (
        ("slope_sys_d", "E1_SYS_D", "g1_slope_sys_d"),
        ("slope_nosys_d_matched", "M1", "g1_slope_nosys_d_matched"),
        ("dwaq_slope", "E1_DWAQ_SLOPE", "g1_dwaq_slope"),
    ),
)
def test_requested_experiment1_candidates_are_registered(baseline_id, model_id, task_name):
    identity = resolve_baseline(baseline_id, train_seed=42)
    assert identity.model_id == model_id
    assert identity.task_name == task_name
    assert identity.actor_certificate_context == ()
