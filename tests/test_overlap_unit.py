import pandas as pd
import numpy as np
import tempfile
import os
from pymast.overlap_removal import overlap_reduction


class DummyProject:
    def __init__(self, db):
        self.db = db


def write_store(path, presence_df, classified_df):
    with pd.HDFStore(path, 'w') as store:
        store.append('presence', presence_df, format='table', data_columns=True)
        store.append('classified', classified_df, format='table', data_columns=True)


def test_posterior_wins():
    import tempfile
    import os
    td = tempfile.TemporaryDirectory()
    path = os.path.join(td.name, 'proj.h5')

    presence = pd.DataFrame([
        {'freq_code':'F1','epoch':110,'time_stamp':pd.Timestamp('2020-01-01T00:01:50'),'power':50.0,'rec_id':'R1','bout_no':1},
        {'freq_code':'F1','epoch':120,'time_stamp':pd.Timestamp('2020-01-01T00:02:00'),'power':52.0,'rec_id':'R1','bout_no':1},
    ])
    presence_child = pd.DataFrame([
        {'freq_code':'F1','epoch':115,'time_stamp':pd.Timestamp('2020-01-01T00:01:55'),'power':30.0,'rec_id':'R2','bout_no':1}
    ])

    classified_parent = pd.DataFrame([
        {'freq_code':'F1','epoch':110,'time_stamp':pd.Timestamp('2020-01-01T00:01:50'),'power':50.0,'rec_id':'R1','iter':1,'test':1,'posterior_T':0.9},
    ])
    classified_child = pd.DataFrame([
        {'freq_code':'F1','epoch':115,'time_stamp':pd.Timestamp('2020-01-01T00:01:55'),'power':30.0,'rec_id':'R2','iter':1,'test':1,'posterior_T':0.3}
    ])

    write_store(path, pd.concat([presence, presence_child], ignore_index=True), pd.concat([classified_parent, classified_child], ignore_index=True))

    project = DummyProject(path)
    ov = overlap_reduction(['R1','R2'], [('R1','R2')], project)
    ov.unsupervised_removal(method='posterior', confidence_threshold=0.1)

    with pd.HDFStore(path, 'r') as store:
        df = store['/overlapping'] if '/overlapping' in store.keys() else store['overlapping']
        # child rec_id should have overlapping == 1
        child_rows = df[df.rec_id == 'R2']
        assert not child_rows.empty
        assert (child_rows.overlapping == 1).all()
    td.cleanup()


def test_ambiguous_keep_both():
    import tempfile
    import os
    td = tempfile.TemporaryDirectory()
    path = os.path.join(td.name, 'proj2.h5')

    presence = pd.DataFrame([
        {'freq_code':'F2','epoch':200,'time_stamp':pd.Timestamp('2020-01-02T00:03:20'),'power':40.0,'rec_id':'R3','bout_no':1},
        {'freq_code':'F2','epoch':210,'time_stamp':pd.Timestamp('2020-01-02T00:03:30'),'power':41.0,'rec_id':'R3','bout_no':1},
    ])
    presence_child = pd.DataFrame([
        {'freq_code':'F2','epoch':205,'time_stamp':pd.Timestamp('2020-01-02T00:03:25'),'power':39.0,'rec_id':'R4','bout_no':1}
    ])

    # posteriors close together (diff < threshold)
    classified_parent = pd.DataFrame([
        {'freq_code':'F2','epoch':200,'time_stamp':pd.Timestamp('2020-01-02T00:03:20'),'power':40.0,'rec_id':'R3','iter':1,'test':1,'posterior_T':0.55},
    ])
    classified_child = pd.DataFrame([
        {'freq_code':'F2','epoch':205,'time_stamp':pd.Timestamp('2020-01-02T00:03:25'),'power':39.0,'rec_id':'R4','iter':1,'test':1,'posterior_T':0.52}
    ])

    write_store(path, pd.concat([presence, presence_child], ignore_index=True), pd.concat([classified_parent, classified_child], ignore_index=True))

    project = DummyProject(path)
    ov = overlap_reduction(['R3','R4'], [('R3','R4')], project)
    ov.unsupervised_removal(method='posterior', confidence_threshold=0.05)

    with pd.HDFStore(path, 'r') as store:
        df = store['/overlapping'] if '/overlapping' in store.keys() else store['overlapping']
        # both should be kept (overlapping==0)
        assert (df.overlapping == 0).all()
    td.cleanup()


def test_missing_posterior_raises():
    import tempfile
    import os
    td = tempfile.TemporaryDirectory()
    path = os.path.join(td.name, 'proj3.h5')

    presence = pd.DataFrame([
        {'freq_code':'F3','epoch':300,'time_stamp':pd.Timestamp('2020-01-03T00:05:00'),'power':20.0,'rec_id':'R5','bout_no':1},
        {'freq_code':'F3','epoch':310,'time_stamp':pd.Timestamp('2020-01-03T00:05:10'),'power':22.0,'rec_id':'R5','bout_no':1},
    ])
    classified_parent = pd.DataFrame([
        {'freq_code':'F3','epoch':300,'time_stamp':pd.Timestamp('2020-01-03T00:05:00'),'power':20.0,'rec_id':'R5','iter':1,'test':1}
    ])
    classified_child = pd.DataFrame([
        {'freq_code':'F3','epoch':305,'time_stamp':pd.Timestamp('2020-01-03T00:05:05'),'power':21.0,'rec_id':'R6','iter':1,'test':1}
    ])

    write_store(path, pd.concat([presence], ignore_index=True), pd.concat([classified_parent, classified_child], ignore_index=True))

    project = DummyProject(path)
    ov = overlap_reduction(['R5','R6'], [('R5','R6')], project)
    try:
        ov.unsupervised_removal(method='posterior')
        raised = False
    except ValueError:
        raised = True
    assert raised
    td.cleanup()


def test_power_method():
    import tempfile
    import os
    td = tempfile.TemporaryDirectory()
    path = os.path.join(td.name, 'proj4.h5')

    presence = pd.DataFrame([
        {'freq_code':'F4','epoch':400,'time_stamp':pd.Timestamp('2020-01-04T00:06:40'),'power':80.0,'rec_id':'R7','bout_no':1},
        {'freq_code':'F4','epoch':410,'time_stamp':pd.Timestamp('2020-01-04T00:06:50'),'power':82.0,'rec_id':'R7','bout_no':1},
    ])
    presence_child = pd.DataFrame([
        {'freq_code':'F4','epoch':405,'time_stamp':pd.Timestamp('2020-01-04T00:06:45'),'power':30.0,'rec_id':'R8','bout_no':1}
    ])

    classified_parent = pd.DataFrame([
        {'freq_code':'F4','epoch':400,'time_stamp':pd.Timestamp('2020-01-04T00:06:40'),'power':80.0,'rec_id':'R7','iter':1,'test':1},
    ])
    classified_child = pd.DataFrame([
        {'freq_code':'F4','epoch':405,'time_stamp':pd.Timestamp('2020-01-04T00:06:45'),'power':30.0,'rec_id':'R8','iter':1,'test':1}
    ])

    write_store(path, pd.concat([presence, presence_child], ignore_index=True), pd.concat([classified_parent, classified_child], ignore_index=True))

    project = DummyProject(path)
    ov = overlap_reduction(['R7','R8'], [('R7','R8')], project)
    ov.unsupervised_removal(method='power', power_threshold=0.2)

    with pd.HDFStore(path, 'r') as store:
        df = store['/overlapping'] if '/overlapping' in store.keys() else store['overlapping']
        # child should be marked overlapping
        child_rows = df[df.rec_id == 'R8']
        assert not child_rows.empty
        assert (child_rows.overlapping == 1).all()
    td.cleanup()


def test_posterior_ttest_path():
    import tempfile
    import os
    td = tempfile.TemporaryDirectory()
    path = os.path.join(td.name, 'proj5.h5')

    parent_epochs = [100, 110, 120, 130, 140]
    child_epochs = [105, 115, 125, 135, 145]
    presence_parent = pd.DataFrame([
        {'freq_code':'F5','epoch':e,'time_stamp':pd.Timestamp('2020-01-05T00:00:00') + pd.Timedelta(seconds=e),
         'power':60.0,'rec_id':'R9','bout_no':1}
        for e in parent_epochs
    ])
    presence_child = pd.DataFrame([
        {'freq_code':'F5','epoch':e,'time_stamp':pd.Timestamp('2020-01-05T00:00:00') + pd.Timedelta(seconds=e),
         'power':20.0,'rec_id':'R10','bout_no':1}
        for e in child_epochs
    ])

    parent_post = [0.95, 0.94, 0.93, 0.96, 0.92]
    child_post = [0.12, 0.10, 0.09, 0.11, 0.13]
    classified_parent = pd.DataFrame([
        {'freq_code':'F5','epoch':e,'time_stamp':pd.Timestamp('2020-01-05T00:00:00') + pd.Timedelta(seconds=e),
         'power':60.0,'rec_id':'R9','iter':1,'test':1,'posterior_T':p}
        for e, p in zip(parent_epochs, parent_post)
    ])
    classified_child = pd.DataFrame([
        {'freq_code':'F5','epoch':e,'time_stamp':pd.Timestamp('2020-01-05T00:00:00') + pd.Timedelta(seconds=e),
         'power':20.0,'rec_id':'R10','iter':1,'test':1,'posterior_T':p}
        for e, p in zip(child_epochs, child_post)
    ])

    write_store(path, pd.concat([presence_parent, presence_child], ignore_index=True),
                pd.concat([classified_parent, classified_child], ignore_index=True))

    project = DummyProject(path)
    ov = overlap_reduction(['R9','R10'], [('R9','R10')], project)
    ov.unsupervised_removal(method='posterior', p_value_threshold=0.05, effect_size_threshold=0.3)

    with pd.HDFStore(path, 'r') as store:
        df = store['/overlapping'] if '/overlapping' in store.keys() else store['overlapping']
        child_rows = df[df.rec_id == 'R10']
        assert not child_rows.empty
        assert (child_rows.overlapping == 1).all()
    td.cleanup()
