from argparse import Namespace

from paper_eval_final import run
from paper_eval_final.src.common import protocol_bundle


def test_v1_3_uses_one_frozen_checkpoint_per_method():
    protocol, _, _ = protocol_bundle()
    assert protocol["protocol_version"] == "1.3"
    assert protocol["checkpoint_repetition"] == {
        "checkpoints_per_method": 1,
        "independent_training_seed_repetition": False,
        "paired_eval_seeds": True,
        "inference_scope": "fixed_checkpoint_evaluation_condition_variability",
        "prohibited_claim": "training_seed_stability",
    }


def test_formal_experiment1_accepts_the_single_registered_seed42_checkpoint():
    args = Namespace(
        stage="formal", experiment=1, experiments=None, model=None,
        baseline="slope_nosys_d_matched", smoke=False, train_seed=None,
    )
    assert run.registered_seeds(args, 1) == [42]
