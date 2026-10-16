"""classify_receiver (one read, one write per receiver, prebuilt likelihood tables) must reproduce the original per-fish classify."""
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

from pymast import naive_bayes as nb
from pymast import predictors
from pymast.radio_project import radio_project

FIELDS = ["hit_ratio", "cons_length", "noise_ratio", "lag_diff"]
FISH = ["150.000 1", "150.000 2", "150.000 3"]


def _original_predictors():
    path = pathlib.Path(__file__).parent / "data" / "predictors_original.py"
    spec = importlib.util.spec_from_file_location("predictors_original", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _raw_data(rng):
    frames = []
    for i, fish in enumerate(FISH + ["150.000 99"]):            # the last code is not a study tag
        gaps = rng.choice([2, 2, 2, 4, 30, 400], size=300 + 50 * i)
        epoch = (1_500_000_000 + np.cumsum(gaps)).astype(np.int64)
        frames.append(pd.DataFrame({
            "power": rng.integers(40, 120, size=len(epoch)).astype("float32"),
            "time_stamp": pd.to_datetime(epoch, unit="s").astype("datetime64[ns]"),
            "epoch": epoch, "freq_code": fish,
            "noise_ratio": np.round(rng.random(len(epoch)), 1).astype("float32"),
            "scan_time": np.float32(1.0), "channels": np.int32(1), "rec_id": "R1", "rec_type": "orion"}))
    return pd.concat(frames, ignore_index=True)


def _training(rng, n=4000):
    hit = (rng.integers(1, 12, size=n) / 11).astype("float32")
    return pd.DataFrame({
        "detection": rng.integers(0, 2, size=n).astype("int32"), "hit_ratio": hit,
        "cons_length": rng.integers(1, 12, size=n).astype("float32"),
        "noise_ratio": np.round(rng.random(n), 1).astype("float32"),
        "lag_diff": rng.choice([-2, 0, 2, 26, 99999999], size=n).astype("float32")})


@pytest.fixture()
def project(tmp_path):
    rng = np.random.default_rng(11)
    tags = pd.DataFrame({"freq_code": FISH + ["150.000 99"], "pulse_rate": 2.0, "mort_rate": 9999.0,
                         "tag_type": ["study"] * 3 + ["beacon"], "rel_date": pd.Timestamp("2017-01-01")})
    receivers = pd.DataFrame({"rec_id": ["R1"], "rec_type": ["orion"], "node": ["N1"]})
    nodes = pd.DataFrame({"node": ["N1"], "X": [0], "Y": [0]})
    proj = radio_project(str(tmp_path), "synthetic", 5, 1, tags, receivers, nodes)
    with pd.HDFStore(proj.db, mode="a") as store:
        store.append("raw_data", _raw_data(rng), format="table", data_columns=True, index=False,
                     min_itemsize={"freq_code": 20, "rec_id": 20, "rec_type": 20})
    proj.training = _training(rng)
    return proj


def _reference(proj, fish, training, legacy_lookup):
    """The original per-fish classify, written out with the original primitives (calculate_likelihood, tile-based predictors)."""
    orig = _original_predictors()
    with pd.HDFStore(proj.db, mode="r") as store:
        dat = store.select("raw_data", where=f'(freq_code == "{fish}") & (rec_id == "R1")')[proj.CLASSIFY_INPUT_COLUMNS]
    dat = dat.drop_duplicates()
    dat["time_stamp"] = pd.to_datetime(dat["time_stamp"])
    dat.sort_values(by="time_stamp", inplace=True)
    dat["epoch"] = (dat.time_stamp.astype("int64") // 10**9).astype("int64")
    dat = dat.drop_duplicates(subset="time_stamp")
    dat["lag"] = dat.epoch.diff()
    dat["lag_diff"] = dat.lag.diff()
    dat.fillna(value=99999999, inplace=True)
    hist, hit, cons_det, mx = orig.detection_history(dat.epoch.values, proj.tags.at[fish, "pulse_rate"], proj.det_count,
                                                     dat.channels.values, dat.scan_time.values)
    dat["det_hist"] = np.array(["".join(r.astype(str)) for r in hist])
    dat["hit_ratio"], dat["cons_det"], dat["cons_length"] = hit, cons_det, mx
    binned = dict(zip(["hit_ratio", "power", "lag_diff", "cons_length", "noise_ratio"],
                      nb.bin_predictors(dat.hit_ratio, dat.power, dat.lag_diff, dat.cons_length, dat.noise_ratio)))
    labels = training.detection.values
    prior = nb.calculate_priors(labels)
    lik = {True: {}, False: {}}
    for a in (True, False):
        for f in FIELDS:
            obs = dat[f] if legacy_lookup else dat[f].astype(training[f].dtype)
            lik[a][f] = nb.calculate_likelihood(training[f].values, labels, a, obs)
    like_T, like_F = np.prod([lik[True][f] for f in FIELDS], axis=0), np.prod([lik[False][f] for f in FIELDS], axis=0)
    evidence = nb.calculate_evidence([binned[f] for f in FIELDS])
    pri = {True: np.repeat(prior[0], evidence.shape), False: np.repeat(prior[1], evidence.shape)}
    post_T, post_F = nb.calculate_posterior(pri, evidence, lik, True), nb.calculate_posterior(pri, evidence, lik, False)
    dat["likelihood_T"], dat["likelihood_F"], dat["posterior_T"], dat["posterior_F"] = like_T, like_F, post_T, post_F
    dat["test"] = nb.classify_with_threshold(post_T, post_F, 1.0)
    return dat.reset_index(drop=True)


@pytest.mark.unit
@pytest.mark.parametrize("legacy", [True, False])
def test_matches_the_original_per_fish_computation(project, legacy):
    new = project.classify_receiver("R1", FIELDS, training_data=project.training, legacy=legacy, write=False)
    assert set(new.freq_code) == set(FISH), "only study tags are classified"
    for fish in FISH:
        ref = _reference(project, fish, project.training, legacy_lookup=legacy)
        got = new[new.freq_code == fish].sort_values("epoch").reset_index(drop=True)
        assert len(ref) == len(got)
        np.testing.assert_array_equal(got.epoch.values, ref.epoch.values)
        np.testing.assert_array_equal(got.hit_ratio.values, ref.hit_ratio.astype("float32").values)
        np.testing.assert_array_equal(got.cons_length.values, ref.cons_length.astype("float32").values)
        assert list(got.det_hist) == list(ref.det_hist)
        np.testing.assert_allclose(got.likelihood_T.values, ref.likelihood_T.astype("float32").values, rtol=1e-6)
        np.testing.assert_allclose(got.likelihood_F.values, ref.likelihood_F.astype("float32").values, rtol=1e-6)
        np.testing.assert_array_equal(got.test.values, ref.test.astype("int32").values)


@pytest.mark.unit
def test_writes_one_table_and_old_classify_still_works(project):
    project.classify_receiver("R1", FIELDS, training_data=project.training)
    with pd.HDFStore(project.db, mode="r") as store:
        stored = store.select("classified")
    assert set(stored.freq_code) == set(FISH) and (stored.iter == 1).all()
    project.classify(FISH[0], "R1", FIELDS, project.training, reclass_iter=None)      # compatibility wrapper appends the same fish again
    with pd.HDFStore(project.db, mode="r") as store:
        assert store.get_storer("classified").nrows == len(stored) + int((stored.freq_code == FISH[0]).sum())


@pytest.mark.unit
def test_corrected_lookup_changes_some_calls_and_legacy_flag_restores_the_old_ones(project):
    legacy = project.classify_receiver("R1", FIELDS, training_data=project.training, legacy=True, write=False)
    fixed = project.classify_receiver("R1", FIELDS, training_data=project.training, legacy=False, write=False)
    assert not np.allclose(legacy.likelihood_T.values, fixed.likelihood_T.values)


@pytest.mark.unit
def test_threshold_ratio_is_honoured(project):
    strict = project.classify_receiver("R1", FIELDS, training_data=project.training, threshold_ratio=1e9, write=False)
    default = project.classify_receiver("R1", FIELDS, training_data=project.training, write=False)
    assert strict.test.sum() <= default.test.sum() and strict.test.sum() < len(strict)


@pytest.mark.unit
def test_clear_errors(project):
    with pytest.raises(ValueError, match="training data is empty"):
        project.classification_tables(project.training.iloc[0:0], FIELDS)
    one_class = project.training.assign(detection=1)
    with pytest.raises(ValueError, match="only one class"):
        project.classification_tables(one_class, FIELDS)
    with pytest.raises(ValueError, match="lacks columns"):
        project.classification_tables(project.training.drop(columns=["hit_ratio"]), FIELDS)
    with pytest.raises(ValueError, match="not found in receiver_data"):
        project.classify_receiver("NOPE", FIELDS, training_data=project.training)
