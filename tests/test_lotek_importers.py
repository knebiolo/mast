"""Importers for Lotek text dumps that the SRX600/SRX800 parsers cannot read, plus layout detection, damaged lines and duplicates."""
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from pymast import parsers
from pymast.radio_project import radio_project

CODELOG = """
Environment history:
changed: 43591 16:40:11
Site number 14
Available memory (internal): 55220 bytes
Codeset = 2000

Antenna   Gain
   1       50

Master antenna gain = 50

Scan time = 0:03.00
Active partition = 0
Antennas = 1
Continuous record time-out = 0:00.00
Noise blank level = 50
Upconverter base frequency = 148.000 MHz


Active scan_table:
CHANNEL  FREQUENCY 
   1      148.340 MHz 
   2      148.380 MHz 
   3      148.480 MHz 

Filter: None

Echo delay = 6
Code_log data:
    Start                                                   Stop
 Date   Time   Channel Code Antenna Power  Data  Events  Date   Time 
43591 16:51:18    2     13     1      85    N.A.      1      43591 16:51:18
43591 16:51:23    2     13     1      99    N.A.      1      43591 16:51:23
43591 17:00:00          Battery OK
43591 17:05:15    3     44     1      94    N.A.      1      43591 17:05:15
43591 17:05:15    3     44     1      94    N.A.      1      43591 17:05:15
"""

LEGACY = """Environment History:
changed: 43251 13:01:17
site number 0011

Antenna      Gain
   0          60

# Antennas = 1

scan time = 0:02.2
Active scan_table:
CHANNEL  FREQUENCY
   1      150.560
   2      150.500

Filter Change:
changed: 43251 13:01:17
Filter Type: No Filter

ID Only Records:

 DATE   TIME     CHA   PWR ANT CODE
43256 20:35:45     2    78  A0   166
43256 20:35:48     2    80  A0   166
43256 20:36:10     1    71  A0    13
End of Data
"""


def _write(path, text):
    path.write_text(text)
    return str(path)


@pytest.mark.unit
def test_layout_detection(tmp_path):
    assert parsers.detect_lotek_layout(_write(tmp_path / "a.dmp", CODELOG)) == "codelog"
    assert parsers.detect_lotek_layout(_write(tmp_path / "b.txt", LEGACY)) == "srx800_legacy"
    assert parsers.detect_lotek_layout(_write(tmp_path / "c.TXT", "SRX800 / 800D Information:\nEnvironment History:\n")) == "srx800"
    assert parsers.detect_lotek_layout(_write(tmp_path / "d.txt", "Battery Status:\nchanged: 04/29/15  11:56:05\n")) == "srx800"
    fragment = "14:48    1     79     1     209    N.A.      1      42148 20:14:48\n42148 20:15:00    1     79     1     212    N.A.      1      42148 20:15:00\n"
    assert parsers.detect_lotek_layout(_write(tmp_path / "e.dmp", fragment)) == "codelog"
    assert parsers.detect_lotek_layout(_write(tmp_path / "f.csv", "a,b,c\n1,2,3\n")) is None
    assert parsers.detect_lotek_layout(_write(tmp_path / "g.txt", "")) is None


def _raw(db):
    with pd.HDFStore(db, "r") as store:
        return store.select("raw_data")


@pytest.mark.unit
def test_codelog_import(tmp_path):
    db = str(tmp_path / "db.h5")
    parsers.lotek_codelog(_write(tmp_path / "x.dmp", CODELOG), db, "R14", ["148.380 13", "148.480 44"], scan_time=1., channels=1)
    raw = _raw(db)
    assert len(raw) == 4                                              # battery line skipped, nothing else lost
    assert set(raw.freq_code) == {"148.380 13", "148.480 44"}
    assert set(raw.rec_id) == {"R14"} and set(raw.rec_type) == {"srx600"}
    first = raw.iloc[0]
    assert first.time_stamp == pd.Timestamp("2019-05-06 16:51:18")       # Excel day 43591 (origin 1899-12-30)
    assert int(first.epoch) == int((pd.Timestamp("2019-05-06 16:51:18") - pd.Timestamp("1970-01-01")).total_seconds())
    assert (raw.epoch.diff().dropna() >= 0).all()
    assert float(first.power) == 85.0


@pytest.mark.unit
def test_legacy_srx800_import(tmp_path):
    db = str(tmp_path / "db.h5")
    parsers.srx800_legacy(_write(tmp_path / "y.txt", LEGACY), db, "t11", ["150.500 166"], scan_time=3.5, channels=1)
    raw = _raw(db)
    assert len(raw) == 3 and set(raw.freq_code) == {"150.500 166", "150.560 13"}
    assert set(raw.rec_type) == {"srx800"} and float(raw.scan_time.iloc[0]) == 3.5
    assert raw.time_stamp.iloc[0] == pd.Timestamp("2018-06-05 20:35:45")  # Excel day 43256


@pytest.mark.unit
def test_damaged_lines_are_reported_not_dropped_silently(tmp_path):
    db = str(tmp_path / "db.h5")
    damaged = CODELOG + "43641 0$\x12\x9f\x01OK\n43591 17:09:00    9     13     1      85    N.A.      1      43591 17:09:00\n"   # garbled line; unscanned channel 9
    parsers.lotek_codelog(_write(tmp_path / "x.dmp", damaged), db, "R14", ["148.380 13"])
    assert len(_raw(db)) == 4
    with pd.HDFStore(db, "r") as store:
        report = store.select("import_damaged_lines")
    assert len(report) == 2 and set(report.rec_id) == {"R14"} and set(report.file) == {"x.dmp"}


@pytest.mark.unit
def test_more_than_damage_raises(tmp_path):
    junk = "\n".join(f"{i} ???garbled???" for i in range(40))
    with pytest.raises(ValueError, match="more than file damage"):
        parsers.lotek_codelog(_write(tmp_path / "x.dmp", CODELOG + junk + "\n"), str(tmp_path / "db.h5"), "R14", [])


@pytest.mark.unit
def test_fragment_needs_a_scan_table(tmp_path):
    fragment = _write(tmp_path / "frag.dmp", "42148 20:14:48    1     79     1     209    N.A.      1      42148 20:14:48\n"
                                             "42148 20:15:00    1     79     1     212    N.A.      1      42148 20:15:00\n")
    with pytest.raises(ValueError, match="scan_table"):
        parsers.lotek_codelog(fragment, str(tmp_path / "db.h5"), "R2", [])
    db = str(tmp_path / "db2.h5")
    parsers.lotek_codelog(fragment, db, "R2", ["149.780 79"], scan_table={"1": "149.780"})
    assert set(_raw(db).freq_code) == {"149.780 79"} and len(_raw(db)) == 2


@pytest.mark.unit
def test_conflicting_channel_table_is_an_error(tmp_path):
    conflicting = CODELOG + "\nActive scan_table:\nCHANNEL  FREQUENCY \n   2      149.000 MHz \n\n"
    with pytest.raises(ValueError, match="maps to both"):
        parsers.lotek_codelog(_write(tmp_path / "x.dmp", conflicting), str(tmp_path / "db.h5"), "R14", [])


def _project(tmp_path):
    tags = pd.DataFrame({"freq_code": ["148.380 13", "148.480 44", "150.500 166"], "pulse_rate": 3.0, "mort_rate": 9999.0,
                         "tag_type": "study", "rel_date": pd.Timestamp("2019-01-01")})
    receivers = pd.DataFrame({"rec_id": ["R14", "t11"], "rec_type": ["srx600", "srx800"], "node": ["N1", "N2"]})
    return radio_project(str(tmp_path / "proj"), "p", 5, 1, tags, receivers, pd.DataFrame({"node": ["N1", "N2"], "X": [0, 1], "Y": [0, 1]}))


@pytest.mark.unit
def test_telem_data_import_detects_layouts_and_filters_files(tmp_path):
    project = _project(tmp_path)
    folder = tmp_path / "dumps"
    folder.mkdir()
    _write(folder / "a.dmp", CODELOG)
    _write(folder / "a.hex", "binary-ish companion\x00\x01")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        project.telem_data_import("R14", "srx600", str(folder), project.db, file_pattern="*.dmp")
    assert len(_raw(project.db)) == 4
    folder2 = tmp_path / "legacy"
    folder2.mkdir()
    _write(folder2 / "b.txt", LEGACY)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        project.telem_data_import("t11", "srx800", str(folder2), project.db)
    assert set(_raw(project.db).rec_id) == {"R14", "t11"}
    with pytest.raises(ValueError, match="layout must be"):
        project.telem_data_import("t11", "srx800", str(folder2), project.db, layout="nonsense")


@pytest.mark.unit
def test_duplicates_are_always_reported_and_removed_on_request(tmp_path):
    def overlapping_downloads(name):
        folder = tmp_path / name
        folder.mkdir()
        _write(folder / "first.dmp", CODELOG)
        _write(folder / "second.dmp", CODELOG)          # a second download that repeats the first
        return str(folder)

    kept_project = _project(tmp_path / "keep")
    with pytest.warns(UserWarning, match="repeat an earlier one"):
        kept_project.telem_data_import("R14", "srx600", overlapping_downloads("a"), kept_project.db)
    assert len(_raw(kept_project.db)) == 8              # default: reported, nothing removed

    removed_project = _project(tmp_path / "remove")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        removed_project.telem_data_import("R14", "srx600", overlapping_downloads("b"), removed_project.db, deduplicate=True)
    unique = _raw(removed_project.db)
    assert len(unique) == 3 and not unique.duplicated(["freq_code", "epoch", "power"]).any()     # 4 rows per file, one repeated inside it
