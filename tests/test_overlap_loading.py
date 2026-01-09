import pandas as pd
import pytest

from pymast.overlap_removal import overlap_reduction


class DummyProject:
    def __init__(self, db):
        self.db = db


@pytest.mark.unit
@pytest.mark.overlap
def test_overlap_loading_summarizes_presence(tmp_path):
    db_path = tmp_path / "overlap_loading.h5"

    presence = pd.DataFrame([
        {
            "freq_code": "F1",
            "epoch": 10.0,
            "time_stamp": pd.Timestamp("2020-01-01T00:00:10"),
            "power": 50.0,
            "rec_id": "R1",
            "bout_no": 1,
        }
    ])
    classified = pd.DataFrame([
        {
            "freq_code": "F1",
            "epoch": 10.0,
            "time_stamp": pd.Timestamp("2020-01-01T00:00:10"),
            "power": 50.0,
            "rec_id": "R1",
            "iter": 1,
            "test": 1,
            "posterior_T": 0.9,
            "posterior_F": 0.1,
        }
    ])

    with pd.HDFStore(str(db_path), "w") as store:
        store.append("presence", presence, format="table", data_columns=True)
        store.append("classified", classified, format="table", data_columns=True)

    project = DummyProject(str(db_path))
    overlap = overlap_reduction(nodes=["R1"], edges=[], radio_project=project)

    summarized = overlap.node_pres_dict["R1"]
    assert len(summarized) == 1
    row = summarized.iloc[0]
    assert row["min_epoch"] == 10.0
    assert row["max_epoch"] == 10.0
    assert row["median_power"] == 50.0
