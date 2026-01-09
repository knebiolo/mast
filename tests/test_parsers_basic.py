import pandas as pd
import pytest

from pymast import parsers


def _write_file(path, lines):
    path.write_text("\n".join(lines), encoding="utf-8")


def _fixed_width_line(colspecs, values):
    line_length = max(end for _, end in colspecs)
    buf = [" "] * line_length
    for (start, end), value in zip(colspecs, values):
        text = str(value)
        text = text[: end - start].ljust(end - start)
        buf[start:end] = list(text)
    return "".join(buf).rstrip()


@pytest.mark.unit
def test_ares_parser_basic(tmp_path):
    file_path = tmp_path / "ares.csv"
    db_path = tmp_path / "ares.h5"

    header = (
        "Date,Time,RxID,Freq,Antenna,Protocol,Code,Power,Squelch,Noise Level,"
        "Pulse Width 1,Pulse Width 2,Pulse Width 3,Pulse Interval 1,Pulse Interval 2,Pulse Interval 3"
    )
    row = "2020-01-01,00:00:01,RX1,166.380,ANT1,P,7,50,0,0,0,0,0,0,0,0"
    _write_file(file_path, [header, row])

    parsers.ares(
        file_name=str(file_path),
        db_dir=str(db_path),
        rec_id="R_ARES",
        study_tags=["166.380 7"],
        scan_time=1.0,
        channels=1,
    )

    with pd.HDFStore(str(db_path), "r") as store:
        raw = store.get("raw_data")

    assert len(raw) == 1
    record = raw.iloc[0]
    assert record["rec_type"] == "ares"
    assert record["rec_id"] == "R_ARES"
    assert record["freq_code"] == "166.380 7"
    assert record["channels"] == 1
    assert record["scan_time"] == pytest.approx(1.0)
    assert record["noise_ratio"] == pytest.approx(0.0)


@pytest.mark.unit
def test_orion_parser_with_type_header(tmp_path):
    file_path = tmp_path / "orion.txt"
    db_path = tmp_path / "orion.h5"

    header = "Date Time Site Ant Freq Type Code power"
    colspecs = [(0, 12), (13, 23), (24, 30), (31, 35), (36, 45), (46, 54), (55, 60), (61, 65)]
    data_line = _fixed_width_line(
        colspecs,
        ["2020-01-01", "00:00:01", "S1", "A1", "166.380", "TAG", "7", "50"],
    )
    _write_file(file_path, [header, data_line])

    parsers.orion_import(
        file_name=str(file_path),
        db_dir=str(db_path),
        rec_id="R_ORION",
        study_tags=["166.380 7"],
        scan_time=1.0,
        channels=1,
    )

    with pd.HDFStore(str(db_path), "r") as store:
        raw = store.get("raw_data")

    assert len(raw) == 1
    record = raw.iloc[0]
    assert record["rec_type"] == "orion"
    assert record["rec_id"] == "R_ORION"
    assert record["freq_code"] == "166.380 7"
    assert record["channels"] == 1
    assert record["scan_time"] == pytest.approx(1.0)
    assert record["noise_ratio"] == pytest.approx(0.0)
