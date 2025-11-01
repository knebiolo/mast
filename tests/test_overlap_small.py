import sys
sys.path.insert(0, r'c:\Users\Kevin.Nebiolo\OneDrive - Kleinschmidt Associates\Software\mast')
import pandas as pd
import numpy as np
from pymast.overlap_removal import overlap_reduction

# Build a fake overlap_reduction object without calling __init__
obj = overlap_reduction.__new__(overlap_reduction)

# minimal attributes
obj.edges = [('R1','R2')]
obj.nodes = ['R1','R2']
obj.G = None

# Presence dict: parent R1 has one bout for fish 'F1' between epoch 100 and 200
obj.node_pres_dict = {
    'R1': pd.DataFrame([{'freq_code':'F1','bout_no':1,'rec_id':'R1','min_epoch':100,'max_epoch':200}])
}

# Recap dicts: create detections inside bout with different posterior_T
# Parent R1 detections: high posterior
obj.node_recap_dict = {
    'R1': pd.DataFrame([
        {'freq_code':'F1','epoch':110,'time_stamp':pd.Timestamp('2020-01-01T00:01:50'),'posterior_T':0.9,'rec_id':'R1'},
        {'freq_code':'F1','epoch':120,'time_stamp':pd.Timestamp('2020-01-01T00:02:00'),'posterior_T':0.92,'rec_id':'R1'}
    ]),
    'R2': pd.DataFrame([
        {'freq_code':'F1','epoch':115,'time_stamp':pd.Timestamp('2020-01-01T00:01:55'),'posterior_T':0.3,'rec_id':'R2'},
        {'freq_code':'F1','epoch':125,'time_stamp':pd.Timestamp('2020-01-01T00:02:05'),'posterior_T':0.35,'rec_id':'R2'}
    ])
}

# Monkeypatch write_results_to_hdf5 to avoid HDF5 I/O in this test
def _fake_write(df):
    print('\n--- write_results_to_hdf5 called ---')
    cols = [c for c in ['freq_code','epoch','posterior_T','rec_id','overlapping'] if c in df.columns]
    print(df[cols].head())

obj.write_results_to_hdf5 = _fake_write

# Run
obj.unsupervised_removal()

# Print final in-memory dicts
print('\nR1 in-memory recaps:')
print(obj.node_recap_dict['R1'])
print('\nR2 in-memory recaps:')
print(obj.node_recap_dict['R2'])
