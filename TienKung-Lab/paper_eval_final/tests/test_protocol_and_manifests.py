from collections import Counter

from paper_eval_final.src.common import protocol_bundle
from paper_eval_final.src.environment_adapter import _plain
from paper_eval_final.src.trial_generator import generate_trials, manifest_payload, validate_manifest


def test_twelve_seconds_is_600_policy_and_2400_physics_steps():
    protocol, _, _ = protocol_bundle()
    assert protocol["timing"]["experiment2_horizon_s"] / protocol["physics"]["dt_policy_s"] == 600
    assert protocol["timing"]["experiment2_horizon_s"] / protocol["physics"]["dt_phys_s"] == 2400


def test_pilot_and_formal_manifests_do_not_overlap():
    for experiment in (1, 2, 3):
        pilot = generate_trials("pilot", experiment)
        formal = generate_trials("formal", experiment)
        assert {row["eval_seed"] for row in pilot}.isdisjoint({row["eval_seed"] for row in formal})
        assert {row["trial_id"] for row in pilot}.isdisjoint({row["trial_id"] for row in formal})


def test_common_manifest_is_model_independent_and_has_exact_budgets():
    expected = {
        ("screening", 1): 80, ("screening", 2): 20, ("screening", 3): 160,
        ("pilot", 1): 500, ("pilot", 2): 500, ("pilot", 3): 1000,
        ("formal", 1): 2000, ("formal", 2): 2000, ("formal", 3): 4000,
    }
    for key, count in expected.items():
        first = generate_trials(*key)
        second = generate_trials(*key)
        assert first == second
        assert len(first) == count


def test_experiment2_has_no_external_disturbance():
    assert {row["disturbance"]["type"] for row in generate_trials("formal", 2)} == {"none"}


def test_pilot_balanced_rotation_counts():
    exp1 = Counter(row["condition_id"] for row in generate_trials("pilot", 1))
    assert Counter(exp1.values()) == {6: 60, 7: 20}
    exp3 = generate_trials("pilot", 3)
    for family, expected in (("velocity_jump", {6: 60, 7: 20}),
                             ("low_frequency_random_wrench", {6: 30, 7: 10}),
                             ("constant_wrench", {6: 30, 7: 10})):
        counts = Counter(row["condition_id"] for row in exp3 if row["disturbance"]["type"] == family)
        assert Counter(counts.values()) == expected


def test_manifest_identity_verifies():
    payload = manifest_payload("pilot", 1)
    validate_manifest(payload)
    payload["trials"][0]["command_vx"] += 0.01
    try:
        validate_manifest(payload)
    except ValueError as exc:
        assert "hash" in str(exc).lower()
    else:
        raise AssertionError("tampered manifest accepted")


def test_effective_config_snapshot_encodes_nonfinite_config_constants_as_strings():
    assert _plain({"upper": float("inf"), "lower": float("-inf")}) == {
        "upper": "+Infinity", "lower": "-Infinity",
    }
