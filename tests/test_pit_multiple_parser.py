"""
Test script for PIT_Multiple parser with Biomark multi-antenna format
"""

import os
import sys
sys.path.append(r"K:\Jobs\3671\014\Analysis\kpn_2025_10_01\mast")
from pymast import parsers
import pandas as pd

# Test file path
test_file = r"K:\Jobs\3671\014\Data\FieldData\QC1\2025_TFalls_BiomarkLadderPIT_251007\XTF_tag_6fd3663c-e612-4151-9125-1716a917e2ef.csv"

# Set up test parameters
project_dir = r"K:\Jobs\3671\014\Analysis\kpn_2025_10_01"
db_name = 'thompson_2025'
db_dir = os.path.join(project_dir, f'{db_name}.h5')

# Antenna to receiver mapping as provided
antennae_to_rec_id = {
    1: "R0001",
    2: "R0002", 
    3: "R0003",
    4: "R0004",
    5: "R0005"
}

study_tags = []  # Empty for now

print("=== Testing PIT_Multiple Parser ===")
print(f"File: {test_file}")
print(f"Database: {db_dir}")
print(f"Antenna mapping: {antennae_to_rec_id}")

# First, let's examine the file structure
try:
    print("\n=== File Structure Analysis ===")
    sample_df = pd.read_csv(test_file, nrows=5)
    print("Detected columns:")
    for i, col in enumerate(sample_df.columns):
        print(f"  {i}: {col}")
    print("\nSample data:")
    print(sample_df.head())
    
except Exception as e:
    print(f"Error reading file: {e}")

# Now test the parser
try:
    print(f"\n=== Running PIT_Multiple Parser ===")
    parsers.PIT_Multiple(file_name=test_file,
                         db_dir=db_dir,
                         ant_to_rec_dict=antennae_to_rec_id,
                         study_tags=study_tags,
                         skiprows=0,  # CSV usually has headers
                         scan_time=1,
                         channels=1,
                         rec_type="PIT_Multiple")
    
    print("\n✅ PIT_Multiple parser completed successfully!")
    
    # Read back the data to verify
    with pd.HDFStore(db_dir, 'r') as store:
        if 'raw_data' in store:
            raw_data = store['raw_data']
            pit_multiple_data = raw_data[raw_data.rec_type == "PIT_Multiple"]
            
            if len(pit_multiple_data) > 0:
                print(f"\n📊 Imported {len(pit_multiple_data)} PIT_Multiple records")
                
                # Show breakdown by receiver
                receiver_counts = pit_multiple_data.groupby('rec_id').size()
                print("\nRecords by receiver:")
                for rec_id, count in receiver_counts.items():
                    print(f"  {rec_id}: {count} records")
                
                print("\nSample records:")
                print(pit_multiple_data[['time_stamp', 'freq_code', 'rec_id']].head(10))
                
                # Show unique tag IDs
                unique_tags = pit_multiple_data['freq_code'].nunique()
                print(f"\nUnique tag IDs detected: {unique_tags}")
                print("Sample tag IDs:")
                print(pit_multiple_data['freq_code'].unique()[:10])
                
            else:
                print("⚠️ No PIT_Multiple records found in database")
        else:
            print("⚠️ No raw_data table found in database")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()