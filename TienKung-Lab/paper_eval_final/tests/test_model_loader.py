import pytest

torch = pytest.importorskip("torch")

from paper_eval_final.src.common import sha256_file
from paper_eval_final.src.model_loader import (
    dwaq_history_from_step_payload,
    registry,
    resolve_model,
)


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
    data = registry()
    baseline = data["screening_baselines"][baseline_id]
    assert baseline["model"] == model_id
    assert baseline["task_name"] == task_name
    assert data["models"][model_id]["actor_certificate_context"] == []


def test_user_selected_checkpoint_is_hashed_not_globally_frozen(tmp_path):
    checkpoint = tmp_path / "model.pt"
    estimator = tmp_path / "estimator.pt"
    checkpoint.write_bytes(b"checkpoint variant")
    estimator.write_bytes(b"estimator variant")

    identity = resolve_model(
        "M3", train_seed=17, checkpoint_path=checkpoint,
        estimator_path=estimator, training_commit="training-revision",
    )

    assert identity.train_seed == 17
    assert identity.checkpoint_sha256 == sha256_file(checkpoint)
    assert identity.estimator_sha256 == sha256_file(estimator)
    assert identity.training_commit == "training-revision"
    assert identity.formal_status == "READY_USER_SELECTED_CHECKPOINT"


def test_user_selected_checkpoint_requires_train_seed(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="train-seed"):
        resolve_model("M3", checkpoint_path=checkpoint)
