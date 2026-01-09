import logging

import pandas as pd
import pytest

from pymast.overlap_removal import overlap_reduction

logging.basicConfig(level=logging.DEBUG)


class DummyProject:
    def __init__(self, db):
        self.db = db


@pytest.mark.integration
def test_overlap_hdf5_integration(tmp_path):
    path = tmp_path / "overlap_integration.h5"

    # presence table: parent R1 has a bout for F1 from epoch 100-200
    presence = pd.DataFrame([
        {'freq_code': 'F1', 'epoch': 110, 'time_stamp': pd.Timestamp('2020-01-01T00:01:50'),
         'power': 50.0, 'rec_id': 'R1', 'bout_no': 1},
        {'freq_code': 'F1', 'epoch': 120, 'time_stamp': pd.Timestamp('2020-01-01T00:02:00'),
         'power': 52.0, 'rec_id': 'R1', 'bout_no': 1},
    ])
    # child has presence (not strictly necessary for algorithm but keep for completeness)
    presence_child = pd.DataFrame([
        {'freq_code': 'F1', 'epoch': 115, 'time_stamp': pd.Timestamp('2020-01-01T00:01:55'),
         'power': 30.0, 'rec_id': 'R2', 'bout_no': 1},
    ])

    # classified table: include posterior_T
    classified_parent = pd.DataFrame([
        {'freq_code': 'F1', 'epoch': 110, 'time_stamp': pd.Timestamp('2020-01-01T00:01:50'),
         'power': 50.0, 'rec_id': 'R1', 'iter': 1, 'test': 1, 'posterior_T': 0.9},
        {'freq_code': 'F1', 'epoch': 120, 'time_stamp': pd.Timestamp('2020-01-01T00:02:00'),
         'power': 52.0, 'rec_id': 'R1', 'iter': 1, 'test': 1, 'posterior_T': 0.92},
    ])
    classified_child = pd.DataFrame([
        {'freq_code': 'F1', 'epoch': 115, 'time_stamp': pd.Timestamp('2020-01-01T00:01:55'),
         'power': 30.0, 'rec_id': 'R2', 'iter': 1, 'test': 1, 'posterior_T': 0.3},
    ])

    with pd.HDFStore(str(path), 'w') as store:
        store.append(
            'presence',
            pd.concat([presence, presence_child], ignore_index=True),
            format='table',
            data_columns=True,
        )
        store.append(
            'classified',
            pd.concat([classified_parent, classified_child], ignore_index=True),
            format='table',
            data_columns=True,
        )

    project = DummyProject(str(path))

    nodes = ['R1', 'R2']
    edges = [('R1', 'R2')]

    ov = overlap_reduction(nodes, edges, project)
    ov.unsupervised_removal(method='posterior')

    with pd.HDFStore(str(path), 'r') as store:
        keys = store.keys()
        if '/overlapping' not in keys and 'overlapping' not in keys:
            print('STORE KEYS:', keys)
        assert ('/overlapping' in keys) or ('overlapping' in keys), 'overlapping key not written'
        df = store.get('/overlapping') if '/overlapping' in keys else store.get('overlapping')
        print('OVERLAPPING TABLE:')
        print(df)
