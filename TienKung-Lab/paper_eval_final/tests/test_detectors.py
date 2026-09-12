import math

from paper_eval_final.src.common import protocol_bundle
from paper_eval_final.src.recovery_detector import RecoveryDetector
from paper_eval_final.src.touchdown_detector import TouchdownDetector
from paper_eval_final.src.trial_machine import TrialMachine
from paper_eval_final.src.trial_generator import generate_trials


def test_same_foot_and_simultaneous_physical_events_are_retained():
    detector = TouchdownDetector(contact_on_N=5, contact_off_N=3, stable_frames=2, debounce_s=.08)
    sequence = [([10, 10], 0), ([0, 10], .01), ([0, 10], .02), ([10, 10], .10), ([10, 10], .11),
                ([0, 10], .20), ([0, 10], .21), ([10, 10], .30), ([10, 10], .31),
                ([0, 0], .40), ([0, 0], .41), ([10, 10], .50), ([10, 10], .51)]
    for forces, timestamp in sequence:
        detector.update(timestamp, forces)
    assert detector.same_foot_count == 1
    assert detector.simultaneous_count == 1
    assert sum(event["event"] == "physical_touchdown" for event in detector.events) == 4


def _sample(timestamp, good=True):
    return {"timestamp": timestamp, "velocity_error": 0.05 if good else 0.5,
            "gravity_tilt": math.radians(2), "world_z_angular_error": .05,
            "clearance": .8, "physical_failure": False, "out_of_area": False, "valid": True}


def test_recovery_must_be_sustained_to_end_and_final_relapse_reenters():
    config = protocol_bundle()[0]["metrics"]["recovery"]
    detector = RecoveryDetector(0.0, config)
    t = .02
    while t <= 1.3:
        detector.update(_sample(t))
        t += .02
    detector.update(_sample(1.32, good=False))
    t = 1.34
    while t <= 10.001:
        detector.update(_sample(t))
        t += .02
    result = detector.result(survived_to_end=True)
    assert result["recovered_sustained"]
    assert len(result["relapse_events"]) == 1
    assert result["final_entry"] > result["first_entry"]


def test_unrecovered_trec_is_na_and_td0_is_not_counted_in_ntd0():
    protocol = protocol_bundle()[0]
    machine = TrialMachine(generate_trials("screening", 1)[0], protocol)
    machine.disturbance_start = 1.0
    machine.disturbance_end = 1.0
    machine.push_applied = True
    machine.recovery = RecoveryDetector(1.0, protocol["metrics"]["recovery"])
    machine.td0_time = 2.0
    machine.touchdowns.events = [
        {"event": "alternating_touchdown", "timestamp": 2.0},
        {"event": "alternating_touchdown", "timestamp": 2.5},
    ]
    machine.finish("HORIZON_REACHED", "FAILURE", "NOT_RECOVERED")
    result = machine.result()
    assert result["recovery_time"] is None and result["nTD0"] is None


def test_pre_push_failure_stays_in_prescribed_denominator():
    protocol = protocol_bundle()[0]
    machine = TrialMachine(generate_trials("screening", 3)[0], protocol)
    machine.feed_physics(.005, [0, 0], "ROLL_LIMIT")
    result = machine.result()
    assert result["failure_reason"] == "PRE_PUSH_PHYSICAL_FAILURE"
    assert result["scheduled"] == result["valid"] == 1
    assert result["push_applied"] is False


def test_certificate_after_recovery_is_retained_but_excluded_from_ntd0_relation():
    protocol = protocol_bundle()[0]
    machine = TrialMachine(generate_trials("screening", 1)[0], protocol)
    machine.disturbance_start = machine.disturbance_end = 0.0
    machine.push_applied = True
    machine.recovery = RecoveryDetector(0.0, protocol["metrics"]["recovery"])
    timestamp = 0.02
    while timestamp <= 10.001:
        machine.recovery.update(_sample(timestamp))
        timestamp += 0.02
    machine.td0_time = 2.0
    machine.touchdowns.events = [
        {"event": "alternating_touchdown", "timestamp": 2.0},
        {"event": "alternating_touchdown", "timestamp": 2.5},
    ]
    machine.finish("HORIZON_REACHED", "SUCCESS", None)
    result = machine.result()
    assert result["recovered_sustained"] is True
    assert result["final_entry"] < result["TD0_time"]
    assert result["CERT_AFTER_RECOVERY"] is True
    assert result["nTD0"] is None
