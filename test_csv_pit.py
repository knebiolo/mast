#!/usr/bin/env python3
"""
Simple test for the unified PIT parser CSV functionality
"""

import sys
sys.path.append('k:\\Jobs\\3671\\014\\Analysis\\kpn_2025_10_01\\mast')

import pymast.parsers as parsers
import pandas as pd
import os

def test_csv_pit_parser():
    print("=== Testing CSV PIT Parser ===\n")
    
    # Test file
    csv_file = "K:\\Jobs\\3671\\014\\Data\\FieldData\\QC1\\2025_TFalls_BiomarkLadderPIT_251007\\XTF_tag_6fd3663c-e612-4151-9125-1716a917e2ef.csv"
    test_db = "K:\\Jobs\\3671\\014\\Analysis\\kpn_2025_10_01\\test_csv_pit.h5"
    
    # Clean up any existing test database
    if os.path.exists(test_db):
        os.remove(test_db)
    
    print("🔧 Testing Multi-Antenna CSV Mode")
    print("=" * 50)
    
    try:
        # Test multi-antenna mode with antenna mapping
        ant_mapping = {1: 'R0001', 2: 'R0002', 3: 'R0003', 4: 'R0004', 5: 'R0005'}
        
        parsers.PIT(
            file_name=csv_file,
            db_dir=test_db,
            rec_id=None,  # Not used in multi-antenna mode
            study_tags=None,
            skiprows=0,  # Auto-detected for CSV
            rec_type="PIT_CSV_Test",
            ant_to_rec_dict=ant_mapping
        )
        print("✅ CSV parsing test PASSED\n")
        
        # Check results
        print("📊 Results Summary")
        print("=" * 30)
        
        with pd.HDFStore(test_db, 'r') as store:
            print("Store keys:", store.keys())
            if '/raw_data' in store.keys():
                data = store['raw_data']
                print(f"Total records: {len(data)}")
                
                # Show record counts by receiver
                print("\nRecords by receiver:")
                print(data['rec_id'].value_counts())
                
                print("\nSample records:")
                print(data[['time_stamp', 'freq_code', 'rec_id']].head(10))
                
                print("\nUnique tags found:")
                print(f"Count: {data['freq_code'].nunique()}")
                print("Sample tags:", data['freq_code'].unique()[:5])
        
    except Exception as e:
        print(f"❌ CSV parsing test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_csv_pit_parser()