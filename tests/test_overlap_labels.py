"""Overlap reduction must not mix up detections that share a row label.

Detections are appended to the HDF5 tables one file or one fish at a time, each with its own 0..n index, so the labels read back are not
unique. The original code looked detections up by label: a bout's posterior statistics were then computed from other fish's rows, and every
row sharing a flagged label was marked as overlapping (79 to 93 percent of the rows at two real receivers shared a label).
"""
import numpy as np
import pandas as pd
import pytest

from pymast.overlap_removal import _bout_summaries, overlap_reduction


def _detections(rec, fish, start, n, posterior, rng):
    epoch = start + np.arange(n) * 5
    return pd.DataFrame({"freq_code": fish, "epoch": epoch, "time_stamp": pd.to_datetime(epoch, unit="s"),
                         "posterior_T": posterior + rng.normal(0, 0.01, n), "power": 80.0 + rng.normal(0, 1, n), "rec_id": rec})


def _bouts(rec, fish, start, n):
    return pd.DataFrame([{"freq_code": fish, "bout_no": 0, "rec_id": rec, "min_epoch": start, "max_epoch": start + (n - 1) * 5}])


@pytest.mark.unit
def test_bout_summaries_match_the_mask_version():
    rng = np.random.default_rng(2)
    detections = pd.DataFrame({"epoch": np.sort(rng.integers(1000, 5000, 300)), "posterior_T": rng.random(300).astype("float32"),
                               "power": rng.integers(40, 120, 300).astype("float32")})
    detections.index = rng.permutation(300)                              # labels need not be in epoch order
    bouts = pd.DataFrame({"min_epoch": [1000, 2000, 2500, 4900, 6000], "max_epoch": [1200, 2300, 2500, 5200, 6100]})
    for expansion in (0, 30):
        got = _bout_summaries(detections, bouts, expansion)
        for row, info in zip(bouts.itertuples(), got):
            lo, hi = row.min_epoch - expansion, row.max_epoch + expansion
            inside = detections[(detections.epoch >= lo) & (detections.epoch <= hi)]
            assert sorted(info["indices"]) == sorted(inside.index)
            if len(inside):
                assert info["posterior"] == pytest.approx(inside.posterior_T.mean())
                assert info["median_power"] == pytest.approx(inside.power.median())
            else:
                assert np.isnan(info["posterior"]) and np.isnan(info["median_power"])


@pytest.mark.unit
def test_only_the_overlapping_fish_is_flagged_when_labels_repeat(tmp_path):
    rng = np.random.default_rng(4)
    # Fish F1 is heard at both receivers at the same time; R1 hears it much better. Fish F2 is heard at each receiver at different times.
    r1 = pd.concat([_detections("R1", "F1", 100, 8, 0.90, rng), _detections("R1", "F2", 1000, 8, 0.50, rng)])      # both frames carry labels 0..7
    r2 = pd.concat([_detections("R2", "F1", 105, 8, 0.20, rng), _detections("R2", "F2", 5000, 8, 0.50, rng)])
    assert not r1.index.is_unique and not r2.index.is_unique

    obj = overlap_reduction.__new__(overlap_reduction)
    obj.edges, obj.nodes, obj.G, obj.db = [("R1", "R2")], ["R1", "R2"], None, str(tmp_path / "overlap.h5")
    obj.node_pres_dict = {"R1": pd.concat([_bouts("R1", "F1", 100, 8), _bouts("R1", "F2", 1000, 8)], ignore_index=True),
                          "R2": pd.concat([_bouts("R2", "F1", 105, 8), _bouts("R2", "F2", 5000, 8)], ignore_index=True)}
    obj.node_recap_dict = {"R1": r1, "R2": r2}
    written = []
    obj.write_results_to_hdf5 = lambda frame: written.append(frame.copy())

    obj.unsupervised_removal(method="posterior", p_value_threshold=0.05, effect_size_threshold=0.2, min_detections=2, bout_expansion=0)

    flagged = pd.concat(written)
    flagged = flagged[flagged.overlapping == 1]
    assert set(flagged.rec_id) == {"R2"}, "R1 hears fish F1 better, so its R2 detections are the ones removed"
    assert set(flagged.freq_code) == {"F1"}, "fish F2 never overlaps and must not be touched"
    assert len(flagged) == 8
