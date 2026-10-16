"""Non-interactive repeat classification and the expected-false-share report."""
import numpy as np
import pandas as pd
import pytest

from pymast.radio_project import radio_project

FISH = ["150.000 1", "150.000 2", "150.000 3"]
NOISE_CODE = "150.000 50"             # decoded by the receiver but never deployed: the known-false class


def _raw(rng):
    frames = []
    for i, fish in enumerate(FISH):
        gaps = rng.choice([2, 2, 2, 2, 4, 30], size=400)             # regular pulses: a real tag
        epoch = (1_500_000_000 + i * 90_000 + np.cumsum(gaps)).astype(np.int64)
        frames.append((fish, epoch))
    noise_epoch = np.sort(rng.integers(1_500_000_000, 1_500_300_000, size=500)).astype(np.int64)      # scattered: noise
    frames.append((NOISE_CODE, noise_epoch))
    out = []
    for fish, epoch in frames:
        out.append(pd.DataFrame({"power": rng.integers(40, 120, len(epoch)).astype("float32"),
                                 "time_stamp": pd.to_datetime(epoch, unit="s").astype("datetime64[ns]"), "epoch": epoch, "freq_code": fish,
                                 "noise_ratio": np.round(rng.random(len(epoch)), 1).astype("float32"), "scan_time": np.float32(1.0),
                                 "channels": np.int32(1), "rec_id": "R1", "rec_type": "orion"}))
    return pd.concat(out, ignore_index=True)


@pytest.fixture()
def trained_project(tmp_path):
    rng = np.random.default_rng(21)
    tags = pd.DataFrame({"freq_code": FISH + ["150.000 99"], "pulse_rate": 2.0, "mort_rate": 9999.0,
                         "tag_type": ["study"] * 3 + ["beacon"], "rel_date": pd.Timestamp("2017-01-01")})
    receivers = pd.DataFrame({"rec_id": ["R1"], "rec_type": ["orion"], "node": ["N1"]})
    proj = radio_project(str(tmp_path), "stable", 5, 1, tags, receivers, pd.DataFrame({"node": ["N1"], "X": [0], "Y": [0]}))
    with pd.HDFStore(proj.db, mode="a") as store:
        store.append("raw_data", _raw(rng), format="table", data_columns=True, index=False, min_itemsize={"freq_code": 20, "rec_id": 20, "rec_type": 20})
    proj.train_receiver("R1")
    return proj


@pytest.mark.unit
def test_runs_to_a_stable_result_and_records_the_rounds(trained_project):
    rounds = trained_project.classify_until_stable(max_iter=8)
    assert set(rounds.columns) == set(trained_project.ROUNDS_COLUMNS)
    assert rounds.iter.max() >= 2, "stability needs at least two rounds to compare"
    last = rounds[rounds.iter == rounds.iter.max()].iloc[0]
    assert last.flip_share < 0.005 and last.true_change < 0.01
    with pd.HDFStore(trained_project.db, mode="r") as store:
        stored = store.select("classification_rounds")
        classified = store.select("classified", columns=["iter", "test"])
    assert len(stored) == len(rounds) and set(classified.iter) == set(rounds.iter)
    assert classified[classified.iter == 1].test.sum() > 0


@pytest.mark.unit
def test_finished_run_is_not_repeated_and_interrupted_run_resumes(trained_project):
    with pytest.warns(RuntimeWarning, match="max_iter=1"):
        first = trained_project.classify_until_stable(max_iter=1)
    assert list(first.iter.unique()) == [1]
    resumed = trained_project.classify_until_stable(max_iter=8)         # continues at round 2; round 1 is not redone
    assert (resumed.iter == 1).sum() == 1
    again = trained_project.classify_until_stable(max_iter=8)           # finished: returns the stored table, adds no round
    assert len(again) == len(resumed)


@pytest.mark.unit
def test_unknown_receiver_is_an_error(trained_project):
    with pytest.raises(ValueError, match="not found in receiver_data"):
        trained_project.classify_until_stable(rec_ids=["NOPE"])


@pytest.mark.unit
def test_expected_false_share_counts_noise_from_undeployed_codes(trained_project):
    trained_project.classify_until_stable(max_iter=8)
    report = trained_project.expected_false_share()
    assert {"iter", "rec_id", "kept", "expected_false", "expected_false_share", "retained_cons_ge_3"} <= set(report.columns)
    trained = pd.read_hdf(trained_project.db, "trained")
    n_noise = int((trained.freq_code == NOISE_CODE).sum())
    ratio = 3 / (99 - 4)                    # 3 deployed codes; valid codes 1..99; four are in the tag table
    assert report.expected_false.iloc[0] == pytest.approx(ratio * n_noise)
    assert (report.kept > 0).all()


@pytest.mark.unit
def test_create_training_data_columns_gives_the_same_rows(trained_project):
    trained_project.classify_until_stable(max_iter=2)
    fields = ["hit_ratio", "cons_length", "noise_ratio", "lag_diff"]
    for label in (None, 2):
        full = trained_project.create_training_data(rec_list=["R1"], reclass_iter=label)
        lean = trained_project.create_training_data(rec_list=["R1"], reclass_iter=label, columns=fields)
        cols = ["detection"] + fields
        a = full[cols].astype(float).sort_values(cols).reset_index(drop=True)
        b = lean[cols].astype(float).sort_values(cols).reset_index(drop=True)
        assert a.equals(b)
        assert list(lean.columns) == cols
