"""
Test script for improved PIT parsers
"""

import os
import sys
sys.path.append(r"K:\Jobs\3671\014\Analysis\kpn_2025_10_01\mast")
from pymast import parsers
import pandas as pd

# Test file path
test_file = r"K:\Jobs\3671\014\Data\FieldData\QC1\2025_TFalls_SubmersibleData_NWE_Received 250917\Submersible Antenna 1695 QC.QA Data 2025.txt"

# Set up test parameters
project_dir = r"K:\Jobs\3671\014\Analysis\kpn_2025_10_01"
db_name = 'thompson_2025'
db_dir = os.path.join(project_dir, f'{db_name}.h5')
rec_id = 'R1695'
study_tags = []  # Empty for now

print("=== Testing Improved PIT Parser ===")
print(f"File: {test_file}")
print(f"Database: {db_dir}")
print(f"Receiver ID: {rec_id}")

try:
    # Test the PIT parser
    parsers.PIT(file_name=test_file,
                db_dir=db_dir,
                rec_id=rec_id,
                study_tags=study_tags,
                skiprows=6,  # Starting guess
                scan_time=1,
                channels=1,
                rec_type="PIT")
    
    print("\n✅ PIT parser completed successfully!")
    
    # Read back the data to verify
    with pd.HDFStore(db_dir, 'r') as store:
        if 'raw_data' in store:
            raw_data = store['raw_data']
            pit_data = raw_data[raw_data.rec_id == rec_id]
            print(f"\n📊 Imported {len(pit_data)} PIT records for {rec_id}")
            print("Sample records:")
            print(pit_data[['time_stamp', 'freq_code', 'rec_id']].head())
        else:
            print("⚠️ No raw_data table found in database")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()