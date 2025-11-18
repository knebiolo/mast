#!/usr/bin/env python3
"""
Test script for the unified PIT parser
Tests both single antenna and multi-antenna modes
"""

import sys
sys.path.append('k:\\Jobs\\3671\\014\\Analysis\\kpn_2025_10_01\\mast')

import pymast.parsers as parsers
import pandas as pd
import os

def test_unified_pit_parser():
    print("=== Testing Unified PIT Parser ===\n")
    
    # Test files
    single_antenna_file = "k:\\Jobs\\3671\\014\\Analysis\\kpn_2025_10_01\\mast\\data\\srx1200_fwf.txt"
    multi_antenna_file = "K:\\Jobs\\3671\\014\\Data\\FieldData\\QC1\\2025_TFalls_BiomarkLadderPIT_251007\\XTF_tag_6fd3663c-e612-4151-9125-1716a917e2ef.csv"
    test_db = "K:\\Jobs\\3671\\014\\Analysis\\kpn_2025_10_01\\test_unified_pit.h5"
    
    # Clean up any existing test database
    if os.path.exists(test_db):
        os.remove(test_db)
    
    print("🔧 Test 1: Single Antenna Mode (Fixed-Width)")
    print("=" * 50)
    
    try:
        # Test single antenna mode (traditional PIT call)
        parsers.PIT(
            file_name=single_antenna_file,
            db_dir=test_db,
            rec_id="R0001",  # Single receiver
            study_tags=None,
            skiprows=6,
            rec_type="PIT"
        )
        print("✅ Single antenna mode test PASSED\n")
        
    except Exception as e:
        print(f"❌ Single antenna mode test FAILED: {e}\n")
    
    print("🔧 Test 2: Multi-Antenna Mode (CSV)")
    print("=" * 50)
    
    try:
        # Test multi-antenna mode with antenna mapping
        ant_mapping = {1: 'R0001', 2: 'R0002', 3: 'R0003', 4: 'R0004', 5: 'R0005'}
        
        parsers.PIT(
            file_name=multi_antenna_file,
            db_dir=test_db,
            rec_id=None,  # Not used in multi-antenna mode
            study_tags=None,
            skiprows=0,  # Auto-detected for CSV
            rec_type="PIT_MultiAntenna",
            ant_to_rec_dict=ant_mapping
        )
        print("✅ Multi-antenna mode test PASSED\n")
        
    except Exception as e:
        print(f"❌ Multi-antenna mode test FAILED: {e}\n")
    
    print("🔧 Test 3: Legacy PIT_Multiple() Function")
    print("=" * 50)
    
    try:
        # Test backward compatibility with PIT_Multiple
        parsers.PIT_Multiple(
            file_name=multi_antenna_file,
            db_dir=test_db,
            ant_to_rec_dict=ant_mapping,
            rec_type="PIT_Multiple_Legacy"
        )
        print("✅ Legacy PIT_Multiple() test PASSED\n")
        
    except Exception as e:
        print(f"❌ Legacy PIT_Multiple() test FAILED: {e}\n")
    
    # Check results
    print("📊 Final Database Summary")
    print("=" * 50)
    
    try:
        with pd.HDFStore(test_db, 'r') as store:
            print("Store keys:", store.keys())
            if '/raw_data' in store.keys():
                data = store['raw_data']
                print(f"Total records: {len(data)}")
                
                # Show record counts by receiver and type
                print("\nRecords by receiver:")
                print(data['rec_id'].value_counts())
                
                print("\nRecords by type:")
                print(data['rec_type'].value_counts())
                
                print("\nSample records:")
                print(data[['time_stamp', 'freq_code', 'rec_id', 'rec_type']].head(10))
                
    except Exception as e:
        print(f"Error reading results: {e}")
    
    print("\n🎉 Unified PIT parser testing complete!")

if __name__ == "__main__":
    test_unified_pit_parser()