import copy
import inspect

import pytest

from paper_eval_final.src.common import digest, load_yaml
from paper_eval_final.src.common import protocol_bundle
from paper_eval_final.src.trial_generator import (
    coverage_probe_manifest,
    generate_experiment1_continuation,
    generate_trials,
)
import paper_eval_final.src.experiment1_sampling as sampling_module
from paper_eval_final.src.experiment1_sampling import (
    MARGIN_GROUPS,
    TARGET_NMIN,
    classify_margin,
    fit_pilot_boundaries,
    pilot_continuation_condition_weights,
    select_formal_samples,
    select_pilot_analysis_samples,
    select_pilot_samples,
)


def _row(sequence, n_min, margin, *, stage="pilot", outcome="SUCCESS"):
    condition_index = sequence % 80
    slope = (-15.0, -5.0, 0.0, 5.0, 15.0)[condition_index % 5]
    speed = ("LOW", "HIGH")[(condition_index // 5) % 2]
    intensity = ("LOW", "HIGH")[(condition_index // 10) % 2]
    direction = (0, 90, 180, 270)[(condition_index // 20) % 4]
    return {
        "trial_id": f"trial-{sequence:06d}",
        "eval_seed": 9_000_000 + sequence,
        "sequence": sequence,
        "stage": stage,
        "condition_id": f"s{slope}_{speed}_{intensity}_d{direction}",
        "slope_deg": slope, "speed_group": speed,
        "disturbance": {"type": "velocity_jump", "intensity_group": intensity,
                        "direction_Hpush_deg": direction},
        "certificate_valid": True,
        "invalid_kind": None,
        "Nmin": n_min,
        "margin_raw": float(margin),
        "margin_storage_dtype": "float64",
        "margin_was_rounded": False,
        "certificate_calculation": {
            "margin_saturated": False,
            "margin_clamped": False,
            "input": {"b": [sequence, 0.0]},
        },
        "task_outcome": outcome,
        "recovered_sustained": outcome == "SUCCESS",
        "recovery_time": 1.0 if outcome == "SUCCESS" else None,
        "nTD0": sequence % 7,
        "Krec": sequence % 7 + 1, "push_applied": True,
        "termination_kind": "HORIZON_REACHED", "failure_reason": None,
        "survived_post_observation": True, "first_entry": None, "final_entry": None,
        "confirmations": [], "relapse_events": [], "events_path": "events.json",
        "trace_path": "trace.npz",
    }


def _complete_pilot():
    rows = []
    for n_min in TARGET_NMIN:
        for index in range(120):
            rows.append(_row(index, n_min, (n_min - 2) * 0.2 + index / 1000.0,
                             outcome="SUCCESS" if index % 3 else "FAILURE"))
    return rows


def test_pilot_selects_exactly_120_per_nmin_without_outcome_filtering():
    rows = _complete_pilot()
    rows += [_row(1000 + i, 3, 9.0, outcome="FAILURE") for i in range(10)]
    selection = select_pilot_samples(rows)
    assert selection["complete"]
    assert selection["counts"] == {"2": 120, "3": 120, "4": 120}
    selected = [row for group in selection["selected"].values() for row in group]
    assert {row["task_outcome"] for row in selected} == {"SUCCESS", "FAILURE"}


def test_exact_per_n_order_statistics_create_distinct_pairs_and_40_each_cell(tmp_path):
    boundaries = fit_pilot_boundaries(
        _complete_pilot(), source_identity={"checkpoint_sha256": "a" * 64},
        output_path=tmp_path / "boundaries.yaml",
    )
    assert boundaries["status"] == "FROZEN"
    assert boundaries["n2_q1"] == pytest.approx(0.0395)
    assert boundaries["n3_q1"] == pytest.approx(0.2395)
    assert boundaries["n4_q1"] == pytest.approx(0.4395)
    reloaded = load_yaml(tmp_path / "boundaries.yaml")
    assert reloaded["boundary_id"] == digest({key: value for key, value in reloaded.items() if key != "boundary_id"})
    for n_min in TARGET_NMIN:
        counts = {group: 0 for group in MARGIN_GROUPS}
        for index in range(120):
            counts[classify_margin(n_min, (n_min - 2) * 0.2 + index / 1000.0, boundaries)] += 1
        assert counts == {"LOWER": 40, "MIDDLE": 40, "UPPER": 40}


def test_boundary_tie_is_margin_degenerate_and_raw_trace_is_emitted(tmp_path):
    rows = _complete_pilot()
    n2 = [row for row in rows if row["Nmin"] == 2]
    n2[40]["margin_raw"] = n2[39]["margin_raw"]
    boundaries = fit_pilot_boundaries(
        rows, source_identity={}, output_path=tmp_path / "boundaries.yaml",
    )
    assert boundaries["status"] == "MARGIN_DEGENERATE"
    diagnostic = boundaries["diagnostic"]["per_Nmin"]["2"]
    assert "M40_EQUALS_M41" in diagnostic["degeneracy_reasons"]
    assert len(diagnostic["raw_margin_and_intermediates"]) == 120


def test_nmin2_narrow_margin_range_blocks_boundary_freeze(tmp_path):
    rows = _complete_pilot()
    for index, row in enumerate(item for item in rows if item["Nmin"] == 2):
        row["margin_raw"] = 0.1 + index * 1.0e-10
    boundaries = fit_pilot_boundaries(
        rows, source_identity={}, output_path=tmp_path / "boundaries.yaml",
    )
    assert boundaries["status"] == "MARGIN_DEGENERATE"
    assert "N2_MARGIN_VARIATION_INSUFFICIENT" in boundaries["diagnostic"]["per_Nmin"]["2"]["degeneracy_reasons"]


def test_formal_acceptance_is_outcome_blind_and_exact_per_cell(tmp_path):
    boundaries = fit_pilot_boundaries(
        _complete_pilot(), source_identity={}, output_path=tmp_path / "boundaries.yaml",
    )
    rows = []
    centers = {"LOWER": 0.01, "MIDDLE": 0.06, "UPPER": 0.10}
    for n_min in TARGET_NMIN:
        for group in MARGIN_GROUPS:
            for index in range(170):
                rows.append(_row(
                    len(rows), n_min, centers[group] + (n_min - 2) * 0.2, stage="formal",
                    outcome="FAILURE" if index < 160 else "SUCCESS",
                ))
    selected = select_formal_samples(rows, boundaries)
    assert selected["complete"]
    assert len(selected["accepted_trial_ids"]) == 1440
    assert all(cell["accepted"] == 160 for cell in selected["cells"].values())
    assert {item["sample_role"] for item in selected["assignments"].values()} == {
        "formal_analysis", None,
    }
    assert "shared_q1" not in inspect.getsource(select_formal_samples)
    assert "shared_q2" not in inspect.getsource(select_formal_samples)


def test_calibration_rows_form_exact_per_n_tertiles(tmp_path):
    rows = []
    for n_min in TARGET_NMIN:
        for index in range(120):
            rows.append(_row(len(rows), n_min, (n_min - 2) + index / 1000.0))
    boundaries = fit_pilot_boundaries(rows, source_identity={}, output_path=tmp_path / "boundaries.yaml")
    analysis = select_pilot_analysis_samples(rows, boundaries)
    calibration_only = [item for item in analysis["assignments"].values() if item["calibration_only"]]
    assert len(calibration_only) == 0


def test_continuation_ranking_does_not_change_when_outcomes_change():
    rows = _complete_pilot()
    first = pilot_continuation_condition_weights(rows)
    changed = copy.deepcopy(rows)
    for row in changed:
        row["task_outcome"] = "INVALID" if row["task_outcome"] == "SUCCESS" else "SUCCESS"
        row["recovered_sustained"] = not row["recovered_sustained"]
        row["recovery_time"] = 99.0
        row["nTD0"] = 999
    assert pilot_continuation_condition_weights(changed) == first


def test_complete_conditional_cells_need_no_continuation(tmp_path, monkeypatch):
    plans = generate_trials("pilot", 1)
    records = []
    for index, plan in enumerate(plans):
        row = dict(plan)
        if index < 360:
            n_min = 2 + index % 3
            row.update(
                certificate_valid=True, invalid_kind=None, Nmin=n_min,
                margin_raw=(n_min - 2) * 0.05 + (index // 3) / 1000.0,
                margin_storage_dtype="float64", margin_was_rounded=False,
                certificate_calculation={"margin_saturated": False, "margin_clamped": False,
                                         "margin_truncated": False},
                push_applied=True, termination_kind="HORIZON_REACHED", task_outcome="SUCCESS",
                failure_reason=None, recovered_sustained=True, survived_post_observation=True,
                recovery_time=1.0, nTD0=2, Krec=3, first_entry=1.0, final_entry=1.0,
                confirmations=[], relapse_events=[], events_path="events.json", trace_path="trace.npz",
            )
        else:
            row.update(certificate_valid=False, invalid_kind=None, Nmin=None, margin_raw=None,
                       margin_storage_dtype=None, margin_was_rounded=None)
        records.append(row)
    cfg = sampling_module.experiment1_config()
    boundary_path = tmp_path / "boundaries.yaml"
    fit_pilot_boundaries(records, source_identity={
        "stage": "pilot", "protocol_hash": protocol_bundle()[2]["protocol_hash"],
        "manifest_hash": "m", "checkpoint_sha256": cfg["checkpoint_sha256"],
        "model_id": "M1", "train_seed": 42,
    }, output_path=boundary_path)
    monkeypatch.setattr(sampling_module, "BOUNDARY_PATH", boundary_path)
    continuation = generate_experiment1_continuation("pilot", 500, 24, records)
    assert continuation == []


def test_n4_high_coverage_probe_is_fixed_and_analysis_excluded():
    with pytest.raises(ValueError, match="retired"):
        coverage_probe_manifest(count=32)
