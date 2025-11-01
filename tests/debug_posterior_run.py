import pandas as pd
import os
import tempfile
from pymast.overlap_removal import overlap_reduction

# Build HDF5
fd, path = tempfile.mkstemp(suffix='.h5')
os.close(fd)

presence = pd.DataFrame([
    {'freq_code':'F1','epoch':110,'time_stamp':pd.Timestamp('2020-01-01T00:01:50'),'power':50.0,'rec_id':'R1','bout_no':1},
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

with pd.HDFStore(path,'w') as store:
    store.append('presence', pd.concat([presence, presence_child], ignore_index=True), format='table', data_columns=True)
    store.append('classified', pd.concat([classified_parent, classified_child], ignore_index=True), format='table', data_columns=True)

class DummyProject:
    def __init__(self, db):
        self.db = db

proj = DummyProject(path)
ov = overlap_reduction(['R1','R2'], [('R1','R2')], proj)
ov.unsupervised_removal(method='posterior', confidence_threshold=0.1)

with pd.HDFStore(path,'r') as store:
    print('STORE KEYS:', store.keys())
    print(store['/overlapping'])

os.remove(path)
print('done')
