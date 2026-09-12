from argparse import Namespace

from paper_eval_final import run
from paper_eval_final.src.common import protocol_bundle


def test_v1_3_does_not_claim_training_seed_stability():
    protocol, _, _ = protocol_bundle()
    assert protocol["protocol_version"] == "1.3"
    assert protocol["checkpoint_repetition"]["paired_eval_seeds"] is True
    assert protocol["checkpoint_repetition"]["prohibited_claim"] == "training_seed_stability"


def test_formal_experiment1_accepts_the_single_registered_seed42_checkpoint():
    args = Namespace(
        stage="formal", experiment=1, experiments=None, model=None,
        baseline="slope_nosys_d_matched", smoke=False, train_seed=None,
    )
    assert run.registered_seeds(args, 1) == [42]
