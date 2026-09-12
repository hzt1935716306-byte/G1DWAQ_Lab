from pathlib import Path

import pytest

from paper_eval_final import run
from paper_eval_final.src import executor, storage
from paper_eval_final.src.model_loader import ModelIdentity


def _model(checkpoint_sha: str, estimator_sha: str | None = None) -> ModelIdentity:
    return ModelIdentity(
        model_id="M3", method="method", task_name="task",
        actor_certificate_context=("margin_norm", "valid"), train_seed=42,
        checkpoint_path=Path("/tmp/model.pt"), checkpoint_sha256=checkpoint_sha,
        training_commit="training", estimator_path=None,
        estimator_sha256=estimator_sha, formal_status="READY_USER_SELECTED_CHECKPOINT",
    )


def test_cli_selected_checkpoint_does_not_require_registry_entry():
    args = run.parser().parse_args([
        "--stage", "formal", "--experiment", "2", "--model", "M3",
        "--checkpoint", "/models/m3.pt", "--train-seed", "7",
    ])
    assert run.registered_seeds(args, 2) == [7]


def test_cli_selected_checkpoint_requires_training_seed():
    args = run.parser().parse_args([
        "--stage", "formal", "--experiment", "2", "--model", "M3",
        "--checkpoint", "/models/m3.pt",
    ])
    with pytest.raises(ValueError, match="train-seed"):
        run.registered_seeds(args, 2)


def test_new_shard_path_contains_checkpoint_and_estimator_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(executor, "ROOT", tmp_path)
    model = _model("a" * 64, "b" * 64)
    path = executor._run_path("formal", 2, model, smoke=False)
    assert path.name == "checkpoint_aaaaaaaaaaaaaaaa_estimator_bbbbbbbbbbbbbbbb"


def test_existing_legacy_shard_remains_resumable(monkeypatch, tmp_path):
    monkeypatch.setattr(executor, "ROOT", tmp_path)
    model = _model("a" * 64)
    legacy = (tmp_path / "results/formal/protocol_1_3_b71a1ddf9293"
              / "experiment_1/M3/train_seed_42")
    legacy.mkdir(parents=True)
    (legacy / "identity.json").write_text("{}", encoding="utf-8")
    assert executor._run_path("formal", 1, model, smoke=False) == legacy


def test_documentation_commit_can_resume_same_code_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    path = tmp_path / "results/formal/shard"
    first = {"evaluation_code_commit": "commit-a", "evaluation_code_hash": "same-code",
             "checkpoint_sha256": "checkpoint"}
    storage.RunStore(path, first)
    resumed = storage.RunStore(path, {
        **first, "evaluation_code_commit": "commit-b", "training_commit": "new-provenance",
    })
    assert resumed.identity["evaluation_code_commit"] == "commit-a"
    with pytest.raises(ValueError, match="identity differs"):
        storage.RunStore(path, {**first, "evaluation_code_hash": "different-code"})
