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
