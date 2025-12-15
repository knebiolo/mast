import tempfile
import os
import pandas as pd
import numpy as np
import logging
import sys

# Ensure repo root is on sys.path so `import pymast` works when running the script
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pymast.overlap_removal import overlap_reduction

logging.basicConfig(level=logging.DEBUG)

# Build a temporary HDF5 project
fd, path = tempfile.mkstemp(suffix='.h5')
os.close(fd)

# presence table: parent R1 has few rows
presence = pd.DataFrame([
    {'freq_code':'F1','epoch':110,'time_stamp':pd.Timestamp('2020-01-01T00:01:50'),'power':50.0,'rec_id':'R1','bout_no':1},
    {'freq_code':'F1','epoch':120,'time_stamp':pd.Timestamp('2020-01-01T00:02:00'),'power':52.0,'rec_id':'R1','bout_no':1}
])
# child has presence
presence_child = pd.DataFrame([
    {'freq_code':'F1','epoch':115,'time_stamp':pd.Timestamp('2020-01-01T00:01:55'),'power':30.0,'rec_id':'R2','bout_no':1}
])

classified_parent = pd.DataFrame([
    {'freq_code':'F1','epoch':110,'time_stamp':pd.Timestamp('2020-01-01T00:01:50'),'power':50.0,'rec_id':'R1','iter':1,'test':1,'posterior_T':0.9},
    {'freq_code':'F1','epoch':120,'time_stamp':pd.Timestamp('2020-01-01T00:02:00'),'power':52.0,'rec_id':'R1','iter':1,'test':1,'posterior_T':0.92}
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

project = DummyProject(path)

nodes = ['R1','R2']
edges = [('R1','R2')]

print('STORE KEYS BEFORE RUN:')
with pd.HDFStore(path,'r') as s:
    print(s.keys())

ov = overlap_reduction(nodes, edges, project)
ov.unsupervised_removal(method='posterior', confidence_threshold=0.1)

print('STORE KEYS AFTER RUN:')
with pd.HDFStore(path,'r') as s:
    print(s.keys())
    if '/overlapping' in s.keys():
        print('overlapping rows:', len(s.select('overlapping')))

# cleanup
os.remove(path)
print('Repro complete')
