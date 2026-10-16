"""train_receiver (one read, one write per receiver) must reproduce the original per-tag train."""
import numpy as np
import pandas as pd
import pytest

from test_classify_receiver import FISH, _original_predictors, project  # noqa: F401  (project is a fixture)


def _reference_train(proj, fish):
    orig = _original_predictors()
    with pd.HDFStore(proj.db, mode="r") as store:
        dat = store.select("raw_data", where=f'(freq_code == "{fish}") & (rec_id == "R1")')
    dat["time_stamp"] = pd.to_datetime(dat.time_stamp)
    dat["epoch"] = (dat.time_stamp.astype("int64") // 10**9).astype("int64")
    dat.sort_values(by="epoch", inplace=True)
    dat.drop_duplicates(subset="time_stamp", keep="first", inplace=True)
    dat["detection"] = 1 if fish in proj.study_tags else 0
    dat["lag"] = dat.epoch.diff()
    dat["lag_diff"] = dat.lag.diff()
    pulse = proj.tags.at[fish, "pulse_rate"] if fish in proj.tags.index else 673.
    hist, hit, cons_det, mx = orig.detection_history(dat.epoch.values, pulse, proj.det_count, dat.channels.values, dat.scan_time.values)
    dat["det_hist"] = np.array(["".join(r.astype(str)) for r in hist])
    dat["hit_ratio"], dat["cons_det"], dat["cons_length"] = hit, cons_det, mx
    dat.fillna(value=9999999, inplace=True)
    return dat.reset_index(drop=True)


@pytest.mark.unit
def test_matches_the_original_per_tag_train(project):          # noqa: F811
    got = project.train_receiver("R1", write=False)
    # the beacon tag is skipped; the three study tags are trained
    assert set(got.freq_code) == set(FISH)
    for fish in FISH:
        ref = _reference_train(project, fish)
        mine = got[got.freq_code == fish].sort_values("epoch").reset_index(drop=True)
        assert len(ref) == len(mine) > 0
        for column in ["epoch", "detection", "det_hist", "cons_det"]:
            assert list(mine[column]) == list(ref[column]), column
        for column in ["lag", "lag_diff", "hit_ratio", "cons_length"]:
            np.testing.assert_allclose(mine[column].values, ref[column].astype("float32").values, rtol=1e-6)


@pytest.mark.unit
def test_train_wrapper_and_write(project):                     # noqa: F811
    project.train(FISH[0], "R1")
    project.train_receiver("R1", freq_codes=FISH[1:])
    with pd.HDFStore(project.db, mode="r") as store:
        stored = store.select("trained")
    assert set(stored.freq_code) == set(FISH)


@pytest.mark.unit
def test_unknown_receiver_raises(project):                     # noqa: F811
    with pytest.raises(ValueError, match="not found in receiver_data"):
        project.train_receiver("NOPE")
