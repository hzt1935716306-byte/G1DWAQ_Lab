import copy

import pytest

from paper_eval_final.src.experiment1_sampling import (
    MARGIN_GROUPS,
    TARGET_NMIN,
    classify_margin,
    fit_pilot_boundaries,
    pilot_continuation_condition_weights,
    select_formal_samples,
    select_pilot_samples,
)


def _row(sequence, n_min, margin, *, stage="pilot", outcome="SUCCESS"):
    return {
        "trial_id": f"trial-{sequence:06d}",
        "sequence": sequence,
        "stage": stage,
        "condition_id": f"layer-{sequence % 5}",
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
    }


def _complete_pilot():
    rows = []
    for n_min in TARGET_NMIN:
        for index in range(120):
            rows.append(_row(len(rows), n_min, n_min + index / 1000.0,
                             outcome="SUCCESS" if index % 3 else "FAILURE"))
    return rows


def test_pilot_selects_exactly_120_per_nmin_without_outcome_filtering():
    rows = _complete_pilot()
    rows += [_row(1000 + i, 3, 9.0, outcome="FAILURE") for i in range(10)]
    selection = select_pilot_samples(rows)
    assert selection["complete"]
    assert selection["counts"] == {"3": 120, "4": 120, "5": 120}
    selected = [row for group in selection["selected"].values() for row in group]
    assert {row["task_outcome"] for row in selected} == {"SUCCESS", "FAILURE"}


def test_exact_order_statistics_and_40_40_40_cells(tmp_path):
    boundaries = fit_pilot_boundaries(
        _complete_pilot(), source_identity={"checkpoint_sha256": "a" * 64},
        output_path=tmp_path / "boundaries.yaml",
    )
    assert boundaries["status"] == "FROZEN"
    for n_min in TARGET_NMIN:
        assert boundaries["boundaries"][str(n_min)]["q1"] == pytest.approx(n_min + 0.0395)
        assert boundaries["boundaries"][str(n_min)]["q2"] == pytest.approx(n_min + 0.0795)
        counts = {group: 0 for group in MARGIN_GROUPS}
        for index in range(120):
            counts[classify_margin(n_min, n_min + index / 1000.0, boundaries)] += 1
        assert counts == {"LOW": 40, "MEDIUM": 40, "HIGH": 40}


def test_boundary_tie_is_margin_degenerate_and_raw_trace_is_emitted(tmp_path):
    rows = _complete_pilot()
    n3 = [row for row in rows if row["Nmin"] == 3]
    n3[39]["margin_raw"] = n3[40]["margin_raw"]
    boundaries = fit_pilot_boundaries(
        rows, source_identity={}, output_path=tmp_path / "boundaries.yaml",
    )
    assert boundaries["status"] == "MARGIN_DEGENERATE"
    diagnostic = boundaries["diagnostics"]["3"]
    assert "M40_EQUALS_M41" in diagnostic["degeneracy_reasons"]
    assert len(diagnostic["raw_margin_and_intermediates"]) == 120


def test_formal_acceptance_is_outcome_blind_and_exact_per_cell(tmp_path):
    boundaries = fit_pilot_boundaries(
        _complete_pilot(), source_identity={}, output_path=tmp_path / "boundaries.yaml",
    )
    rows = []
    centers = {"LOW": 0.01, "MEDIUM": 0.06, "HIGH": 0.10}
    for n_min in TARGET_NMIN:
        for group in MARGIN_GROUPS:
            for index in range(170):
                rows.append(_row(
                    len(rows), n_min, n_min + centers[group], stage="formal",
                    outcome="FAILURE" if index < 160 else "SUCCESS",
                ))
    selected = select_formal_samples(rows, boundaries)
    assert selected["complete"]
    assert len(selected["accepted_trial_ids"]) == 1440
    assert all(cell["accepted"] == 160 for cell in selected["cells"].values())


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
