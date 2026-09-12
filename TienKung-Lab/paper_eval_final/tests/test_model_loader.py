import pytest

torch = pytest.importorskip("torch")

from paper_eval_final.src.model_loader import dwaq_history_from_step_payload


def test_registered_dwaq_history_contracts_are_both_explicitly_supported():
    history = torch.zeros((4, 395))
    assert dwaq_history_from_step_payload(history) is history
    assert dwaq_history_from_step_payload({"observations": {"obs_hist": history}}) is history
    with pytest.raises(ValueError, match="history contract"):
        dwaq_history_from_step_payload({"observations": {"critic": history}})
