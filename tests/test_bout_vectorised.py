"""The vectorised bout detection must give the same bouts as one DBSCAN(eps, min_samples=1) fit per tag."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import DBSCAN

from pymast.overlap_removal import bout


def _classified(rng, rec="R1"):
    frames = []
    for i, fish in enumerate(["150.0 1", "150.0 2", "151.0 7", "152.0 9"]):
        gaps = rng.choice([1, 2, 2, 3, 9, 11, 30, 600, 2000], size=800)
        epoch = (1_500_000_000 + np.cumsum(gaps)).astype("int64")
        frames.append(pd.DataFrame({"freq_code": fish, "epoch": epoch, "time_stamp": pd.to_datetime(epoch, unit="s"),
                                    "power": rng.integers(40, 120, len(epoch)).astype("float32"), "rec_id": rec, "iter": np.int32(1),
                                    "test": np.int32(1)}))
    return pd.concat(frames, ignore_index=True)


def _project(tmp_path, classified, pulse):
    db = str(tmp_path / "bouts.h5")
    with pd.HDFStore(db, "w") as store:
        store.append("classified", classified, format="table", data_columns=True, min_itemsize={"freq_code": 20, "rec_id": 20})
    tags = pd.DataFrame({"pulse_rate": list(pulse.values())}, index=pd.Index(list(pulse.keys()), name="freq_code"))
    return SimpleNamespace(db=db, tags=tags)


def _oracle(classified, pulse, eps_multiplier, min_detections):
    rows = []
    for fish, g in classified.sort_values(["freq_code", "time_stamp"]).groupby("freq_code", sort=True):
        eps = pulse.get(fish, 8.0) * eps_multiplier
        labels = DBSCAN(eps=eps, min_samples=1).fit(g[["epoch"]].values).labels_
        g = g.assign(bout_no=labels)
        if min_detections > 1:
            g = g[g.groupby("bout_no").bout_no.transform("size") >= min_detections]
        rows.append(g[["freq_code", "epoch", "bout_no"]])
    return pd.concat(rows, ignore_index=True)


@pytest.mark.unit
@pytest.mark.parametrize("eps_multiplier,min_detections", [(5, 1), (5, 5), (2, 3), (9, 1)])
def test_same_bouts_as_dbscan(tmp_path, eps_multiplier, min_detections):
    rng = np.random.default_rng(5)
    classified = _classified(rng)
    pulse = {"150.0 1": 2.0, "150.0 2": 3.0, "151.0 7": 8.0}                  # the fourth tag is missing from the tag table
    project = _project(tmp_path, classified, pulse)
    got = bout(project, "R1", eps_multiplier=eps_multiplier, lag_window=2, min_detections=min_detections).presence_df
    want = _oracle(classified, pulse, eps_multiplier, min_detections)
    got = got[["freq_code", "epoch", "bout_no"]].sort_values(["freq_code", "epoch"]).reset_index(drop=True)
    want = want.sort_values(["freq_code", "epoch"]).reset_index(drop=True)
    got, want = got.astype({"freq_code": str, "bout_no": "int64"}), want.astype({"freq_code": str, "bout_no": "int64"})
    pd.testing.assert_frame_equal(got, want)


@pytest.mark.unit
def test_empty_receiver(tmp_path):
    classified = _classified(np.random.default_rng(1), rec="OTHER")
    project = _project(tmp_path, classified, {"150.0 1": 2.0})
    assert bout(project, "R1").presence_df.empty


@pytest.mark.unit
def test_only_the_last_classification_round_is_used(tmp_path):
    rng = np.random.default_rng(9)
    last = _classified(rng)
    earlier = last.assign(iter=np.int32(1))
    earlier.loc[earlier.index[::3], "test"] = np.int32(0)              # round 1 called a third of the detections false
    middle = last.assign(iter=np.int32(2), epoch=last.epoch + 10_000)   # different detections: must not leak into the result
    final = last.assign(iter=np.int32(3))
    final.loc[final.index[::5], "test"] = np.int32(0)                  # the last round drops a fifth of them
    classified = pd.concat([earlier, middle, final], ignore_index=True)
    pulse = {"150.0 1": 2.0, "150.0 2": 3.0, "151.0 7": 8.0}
    got = bout(_project(tmp_path, classified, pulse), "R1", eps_multiplier=5, lag_window=2, min_detections=1).presence_df
    want = _oracle(final[final.test == 1], pulse, 5, 1)
    key = ["freq_code", "epoch", "bout_no"]
    got = got[key].sort_values(["freq_code", "epoch"]).reset_index(drop=True).astype({"freq_code": str, "bout_no": "int64"})
    want = want[key].sort_values(["freq_code", "epoch"]).reset_index(drop=True).astype({"freq_code": str, "bout_no": "int64"})
    pd.testing.assert_frame_equal(got, want)
