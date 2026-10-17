import os
from pathlib import Path
import pandas as pd
import pytest
from pymast.formatter import time_to_event

class DummyProject:
    def __init__(self, db, tags):
        self.db = db
        self.tags = tags

@pytest.mark.unit
def test_tte_adjacency_filter_removes_illegal_transition():
    temp_dir = Path(".pytest_cache")
    temp_dir.mkdir(exist_ok=True)
    db_path = temp_dir / "tte_recap.h5"
    recaptures = pd.DataFrame([
        {
            "freq_code": "F1",
            "rec_id": "R1",
            "epoch": 10.0,
            "time_stamp": pd.Timestamp("2020-01-01T00:00:10"),
            "lag": 1.0,
            "overlapping": 0,
            "ambiguous_overlap": 0,
            "bout_no": 1,
            "power": 50.0,
            "noise_ratio": 0.0,
            "det_hist": 0,
            "hit_ratio": 1.0,
            "cons_det": 1,
            "cons_length": 1,
            "likelihood_T": 0.9,
            "likelihood_F": 0.1,
        },
        {
            "freq_code": "F1",
            "rec_id": "R2",
            "epoch": 20.0,
            "time_stamp": pd.Timestamp("2020-01-01T00:00:20"),
            "lag": 1.0,
            "overlapping": 0,
            "ambiguous_overlap": 0,
            "bout_no": 1,
            "power": 50.0,
            "noise_ratio": 0.0,
            "det_hist": 0,
            "hit_ratio": 1.0,
            "cons_det": 1,
            "cons_length": 1,
            "likelihood_T": 0.9,
            "likelihood_F": 0.1,
        },
        {
            "freq_code": "F1",
            "rec_id": "R1",
            "epoch": 30.0,
            "time_stamp": pd.Timestamp("2020-01-01T00:00:30"),
            "lag": 1.0,
            "overlapping": 0,
            "ambiguous_overlap": 0,
            "bout_no": 1,
            "power": 50.0,
            "noise_ratio": 0.0,
            "det_hist": 0,
            "hit_ratio": 1.0,
            "cons_det": 1,
            "cons_length": 1,
            "likelihood_T": 0.9,
            "likelihood_F": 0.1,
        },
        {
            "freq_code": "F1",
            "rec_id": "R3",
            "epoch": 40.0,
            "time_stamp": pd.Timestamp("2020-01-01T00:00:40"),
            "lag": 1.0,
            "overlapping": 0,
            "ambiguous_overlap": 0,
            "bout_no": 1,
            "power": 50.0,
            "noise_ratio": 0.0,
            "det_hist": 0,
            "hit_ratio": 1.0,
            "cons_det": 1,
            "cons_length": 1,
            "likelihood_T": 0.9,
            "likelihood_F": 0.1,
        },
    ])

    with pd.HDFStore(str(db_path), "w") as store:
        store.append("recaptures", recaptures, format="table", data_columns=True)

    tags = pd.DataFrame(
        {
            "rel_date": [pd.Timestamp("2020-01-01T00:00:00")],
            "cap_loc": ["A"],
            "rel_loc": ["A"],
            "tag_type": ["study"],
            "pulse_rate": [5.0],
            "length": [100],
            "species": ["TEST"],
        },
        index=pd.Index(["F1"], name="freq_code"),
    )

    project = DummyProject(str(db_path), tags)

    receiver_to_state = {"R1": 1, "R2": 9, "R3": 2}
    tte = time_to_event(receiver_to_state, project)
    tte.data_prep(project, adjacency_filter=[(9, 1)])

    transitions = set(tte.master_state_table["transition"].tolist())
    assert (9, 1) not in transitions
    assert (1, 9) in transitions
    assert (9, 2) in transitions

    row = tte.master_state_table[tte.master_state_table["end_state"] == 2].iloc[0]
    assert row["start_state"] == 9
    assert row["time_0"] == 20.0


def _recaps(rows):
    base = {"overlapping": 0, "ambiguous_overlap": 0, "bout_no": 1, "power": 50.0, "noise_ratio": 0.0, "det_hist": 0,
            "hit_ratio": 1.0, "cons_det": 1, "cons_length": 1, "likelihood_T": 0.9, "likelihood_F": 0.1, "lag": 1.0}
    return pd.DataFrame([{**base, **r} for r in rows])


@pytest.mark.unit
def test_tte_keeps_one_second_resolution_at_real_epoch_values_and_tolerates_a_minimal_tag_table(tmp_path):
    # epoch around 1.56e9 s: float32 would round these to multiples of 128 s and make the 3 s and 7 s steps 0 or 128
    t0 = pd.Timestamp("2019-07-12T08:26:27")
    stamps = [t0, t0 + pd.Timedelta(seconds=3), t0 + pd.Timedelta(seconds=10)]
    recaptures = _recaps([{"freq_code": "F1", "rec_id": rec, "time_stamp": stamp,
                           "epoch": float((stamp - pd.Timestamp(0)).total_seconds())}
                          for rec, stamp in zip(("R1", "R2", "R3"), stamps)])
    temp_dir = Path(".pytest_cache")
    temp_dir.mkdir(exist_ok=True)
    db_path = temp_dir / "tte_precision.h5"
    with pd.HDFStore(str(db_path), "w") as store:
        store.append("recaptures", recaptures, format="table", data_columns=True)
    # no length, no species, no rel_loc, no cap_loc, no tag_type, no pulse_rate
    tags = pd.DataFrame({"rel_date": [pd.Timestamp("2019-07-01")]}, index=pd.Index(["F1"], name="freq_code"))
    project = DummyProject(str(db_path), tags)
    project.output_dir = str(tmp_path)

    tte = time_to_event({"R1": 1, "R2": 2, "R3": 3}, project)
    tte.data_prep(project)
    table = tte.master_state_table
    assert table["time_delta"].tolist() == [3.0, 7.0]
    assert (table["time_1"] - table["time_0"]).tolist() == [3.0, 7.0]
    assert table["time_1"].iloc[0] == (stamps[1] - pd.Timestamp(0)).total_seconds()      # exact, not rounded to 128 s
    tte.summary(print_summary=False)
    assert tte.master_state_table["dur"].tolist() == [3, 7]


@pytest.mark.unit
def test_tte_refuses_a_location_filter_the_tag_table_cannot_answer():
    t0 = pd.Timestamp("2019-07-12T08:26:27")
    recaptures = _recaps([{"freq_code": "F1", "rec_id": "R1", "time_stamp": t0, "epoch": float((t0 - pd.Timestamp(0)).total_seconds())}])
    db_path = Path(".pytest_cache") / "tte_loc.h5"
    Path(".pytest_cache").mkdir(exist_ok=True)
    with pd.HDFStore(str(db_path), "w") as store:
        store.append("recaptures", recaptures, format="table", data_columns=True)
    project = DummyProject(str(db_path), pd.DataFrame({"rel_date": [pd.Timestamp("2019-07-01")]}, index=pd.Index(["F1"], name="freq_code")))
    with pytest.raises(ValueError, match="rel_loc='Holyoke' was requested"):
        time_to_event({"R1": 1}, project, rel_loc="Holyoke")
    with pytest.raises(ValueError, match="cap_loc='Cabot' was requested"):
        time_to_event({"R1": 1}, project, cap_loc="Cabot")
