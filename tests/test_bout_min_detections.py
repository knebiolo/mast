"""bout(min_detections=...) drops bouts with too few detections; the default keeps every bout."""

import numpy as np
import pandas as pd
import pytest

from pymast.overlap_removal import bout


class _Project:
    def __init__(self, db, tags):
        self.db = db
        self.tags = tags


@pytest.fixture()
def project(tmp_path):
    db = str(tmp_path / "bouts.h5")
    start = 1_560_000_000
    # one fish at one receiver: a 3-detection bout, then (after a long gap) a 6-detection bout, pulses 3 s apart
    epochs = list(start + 3 * np.arange(3)) + list(start + 1000 + 3 * np.arange(6))
    classified = pd.DataFrame({
        "freq_code": "148.340 20", "epoch": epochs, "time_stamp": pd.to_datetime(epochs, unit="s"),
        "power": np.float32(50), "rec_id": "R1", "iter": 1, "test": 1,
    })
    classified.to_hdf(db, key="classified", format="table", data_columns=True)
    tags = pd.DataFrame({"pulse_rate": [3.0]}, index=pd.Index(["148.340 20"], name="freq_code"))
    return _Project(db, tags)


def test_default_keeps_every_bout(project):
    obj = bout(project, "R1", eps_multiplier=5)
    assert obj.presence_df.bout_no.nunique() == 2 and len(obj.presence_df) == 9


def test_min_detections_drops_small_bouts(project):
    obj = bout(project, "R1", eps_multiplier=5, min_detections=5)
    assert obj.presence_df.bout_no.nunique() == 1 and len(obj.presence_df) == 6


def test_min_detections_larger_than_any_bout_leaves_nothing(project):
    obj = bout(project, "R1", eps_multiplier=5, min_detections=10)
    assert obj.presence_df.empty
