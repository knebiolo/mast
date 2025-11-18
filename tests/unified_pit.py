def PIT(file_name,
        db_dir,
        rec_id=None,
        study_tags=None,
        skiprows=6,
        scan_time=0,
        channels=0,
        rec_type="PIT",
        ant_to_rec_dict=None):
    """
    Unified smart PIT parser that handles both single and multiple antenna formats.
    Automatically detects CSV vs fixed-width formats and adapts parsing accordingly.
    
    Parameters:
    - file_name: Path to the PIT data file
    - db_dir: Path to HDF5 database
    - rec_id: Single receiver ID (for single antenna files, ignored if ant_to_rec_dict provided)
    - study_tags: List of tags to filter (currently unused)
    - skiprows: Number of rows to skip (auto-detected for CSV)
    - scan_time: Scan time value (default 0)
    - channels: Number of channels (default 0)
    - rec_type: Record type identifier
    - ant_to_rec_dict: Dictionary mapping antenna numbers to receiver IDs (for multi-antenna files)
    """
    
    import pandas as pd
    import re
    
    # Determine mode based on parameters
    is_multi_antenna = ant_to_rec_dict is not None
    mode_str = "multi-antenna" if is_multi_antenna else "single antenna"
    print(f"Parsing PIT file ({mode_str}): {file_name}")
    
    # Function to find columns by pattern matching
    def find_column_by_patterns(df, patterns):
        for col in df.columns:
            col_lower = str(col).lower().strip()
            for pattern in patterns:
                if pattern in col_lower:
                    return col
        return None

    # First, analyze the file to determine format
    def analyze_file_format(file_name):
        """Dynamically determine PIT file format and header structure"""
        with open(file_name, 'r') as file:
            lines = []
            for i in range(20):  # Read first 20 lines to analyze format
                try:
                    line = file.readline()
                    if not line:
                        break
                    lines.append(line.rstrip('\n'))
                except:
                    break
        
        # Check if CSV format (look for commas in sample lines)
        csv_indicators = 0
        for line in lines[max(0, len(lines)-10):]:  # Check last 10 lines for data
            if line.count(',') > 3:  # More than 3 commas suggests CSV
                csv_indicators += 1
        
        is_csv = csv_indicators > 2  # If most lines have commas, it's CSV
        
        # For CSV, look for header row
        actual_skiprows = 0
        if is_csv:
            for i, line in enumerate(lines):
                line_lower = line.lower()
                # Look for column headers
                if any(header in line_lower for header in ['tag', 'time', 'date', 'antenna', 'detected']):
                    if ',' in line:  # Make sure it's actually a header row
                        actual_skiprows = i + 1  # Skip past header
                        break
        else:
            # For fixed-width, look for data start
            for i, line in enumerate(lines):
                line_lower = line.lower()
                if 'version' in line_lower or 'ver' in line_lower:
                    print(f"Found version info: {line}")
                
                # Look for data start indicators
                if any(indicator in line_lower for indicator in ['scan date', 'date', 'timestamp', 'tag id']):
                    if i > 0:  # If this looks like a header row
                        actual_skiprows = i + 1
                        break
                
                # Check if this looks like a data line
                if re.search(r'\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}', line):
                    actual_skiprows = i
                    break
        
        return is_csv, actual_skiprows, lines

    # Analyze file format
    is_csv_format, detected_skiprows, sample_lines = analyze_file_format(file_name)
    
    # Use detected skiprows for CSV, keep provided for fixed-width
    if is_csv_format:
        skiprows = detected_skiprows
        print(f"Detected CSV format, using skiprows: {skiprows}")
    else:
        print(f"Detected fixed-width format, using skiprows: {skiprows}")

    # Parse the file based on detected format
    if is_csv_format:
        # CSV Format Parsing
        try:
            # Read CSV with auto-detection
            telem_dat = pd.read_csv(file_name, skiprows=skiprows, dtype=str)
            print(f"Auto-detected columns: {list(telem_dat.columns)}")
            
        except Exception as e:
            print(f"CSV auto-detection failed: {e}")
            # Fallback to predefined column names
            col_names = [
                "FishId", "Tag1Dec", "Tag1Hex", "Tag2Dec", "Tag2Hex", "FloyTag", "RadioTag",
                "Location", "Source", "FishSpecies", "TimeStamp", "Weight", "Length",
                "Antennae", "Latitude", "Longitude", "SampleDate", "CaptureMethod",
                "LocationDetail", "Type", "Recapture", "Sex", "GeneticSampleID", "Comments"
            ]
            telem_dat = pd.read_csv(file_name, names=col_names, header=0, skiprows=skiprows, dtype=str)

        # Find timestamp column dynamically
        timestamp_col = find_column_by_patterns(telem_dat, ['timestamp', 'time stamp', 'date', 'scan date', 'detected'])
        if timestamp_col:
            print(f"Found timestamp column: {timestamp_col}")
            # Try multiple datetime formats
            for fmt in ["%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y", "%Y-%m-%d", None]:
                try:
                    if fmt:
                        telem_dat["time_stamp"] = pd.to_datetime(telem_dat[timestamp_col], format=fmt, errors="coerce")
                    else:
                        telem_dat["time_stamp"] = pd.to_datetime(telem_dat[timestamp_col], errors="coerce")
                    
                    # Check if parsing was successful
                    if not telem_dat["time_stamp"].isna().all():
                        print(f"Successfully parsed timestamps using format: {fmt or 'auto-detect'}")
                        break
                except:
                    continue
        else:
            raise ValueError("Could not find timestamp column")

        # Find tag ID columns dynamically
        hex_tag_col = find_column_by_patterns(telem_dat, ['hex', 'tag1hex', 'tag id', 'tagid', 'tag'])
        dec_tag_col = find_column_by_patterns(telem_dat, ['dec', 'tag1dec', 'decimal'])
        
        if hex_tag_col:
            print(f"Found HEX tag column: {hex_tag_col}")
            telem_dat["freq_code"] = telem_dat[hex_tag_col].astype(str).str.strip()
        elif dec_tag_col:
            print(f"Found DEC tag column: {dec_tag_col}")
            telem_dat["freq_code"] = telem_dat[dec_tag_col].astype(str).str.strip()
        else:
            raise ValueError("Could not find tag ID column")

        # Handle antenna mapping for multi-antenna files
        if is_multi_antenna:
            antenna_col = find_column_by_patterns(telem_dat, ['antenna', 'antennae', 'ant'])
            if antenna_col:
                print(f"Found antenna column: {antenna_col}")
                # Convert antenna column to integer and apply mapping
                telem_dat["antenna_clean"] = telem_dat[antenna_col].astype(str).str.extract(r'(\d+)')[0]
                telem_dat["antenna_clean"] = pd.to_numeric(telem_dat["antenna_clean"], errors='coerce').astype("Int64")
                telem_dat["rec_id"] = telem_dat["antenna_clean"].map(ant_to_rec_dict)
                # Drop rows where antenna values don't match known receivers
                telem_dat = telem_dat.dropna(subset=["rec_id"])
            else:
                raise ValueError("Multi-antenna mode requires antenna column, but none found")
        else:
            # Single antenna mode - use provided rec_id
            telem_dat["rec_id"] = rec_id

    else:
        # Fixed-Width Format Parsing (original logic)
        
        # Read header information for format detection
        with open(file_name, 'r') as file:
            header_lines = []
            for _ in range(max(skiprows, 10)):
                try:
                    line = file.readline()
                    if not line:
                        break
                    header_lines.append(line.rstrip('\n'))
                except:
                    break
            header_text = " ".join(header_lines).lower()

        # Define colspecs for different fixed-width formats
        if 'latitude' in header_text or 'longitude' in header_text:
            colspecs = [(0, 12), (12, 26), (26, 41), (41, 56), (56, 59), (66, 70), 
                       (79, 95), (95, 112), (113, 120), (120, 131), (138, 145), 
                       (145, 155), (155, 166), (166, 175)]
            col_names = ["Scan Date", "Scan Time", "Download Date", "Download Time", 
                        "Reader ID", "Antenna ID", "HEX Tag ID", "DEC Tag ID", 
                        "Temperature_C", "Signal_mV", "Is Duplicate", "Latitude", 
                        "Longitude", "File Name"]
            print("Using format with latitude/longitude")
        else:
            colspecs = [(0, 12), (12, 26), (26, 41), (41, 56), (56, 62), (62, 73), 
                       (73, 89), (89, 107), (107, 122), (122, 132), (136, 136)]
            col_names = ["Scan Date", "Scan Time", "Download Date", "Download Time", 
                        "S/N", "Reader ID", "HEX Tag ID", "DEC Tag ID", 
                        "Temperature_C", "Signal_mV", "Is Duplicate"]
            print("Using format without latitude/longitude")

        # Read the fixed-width file
        telem_dat = pd.read_fwf(
            file_name,
            colspecs=colspecs,
            names=col_names,
            skiprows=skiprows
        )
        
        print(f"Fixed-width parsing complete. Shape: {telem_dat.shape}")
        
        # Build timestamp from Scan Date + Scan Time
        telem_dat["time_stamp"] = pd.to_datetime(
            telem_dat["Scan Date"] + " " + telem_dat["Scan Time"],
            errors="coerce"
        )
        
        # Use HEX Tag ID as freq_code
        telem_dat["freq_code"] = telem_dat["HEX Tag ID"].str.strip()
        
        # For fixed-width, always use single antenna mode
        telem_dat["rec_id"] = rec_id

    # Data cleaning - remove invalid entries
    print(f"\nCleaning data - original records: {len(telem_dat)}")
    
    before_cleanup = len(telem_dat)
    
    # Remove header artifacts
    header_patterns = ['HEX Tag ID', 'DEC Tag ID', '----', '====', 'Tag ID', 'Scan Date']
    for pattern in header_patterns:
        telem_dat = telem_dat[telem_dat['freq_code'] != pattern]
    
    # Remove separator lines
    telem_dat = telem_dat[~telem_dat['freq_code'].str.match(r'^-+$', na=False)]
    
    # Remove rows with invalid timestamps
    telem_dat = telem_dat[~telem_dat['time_stamp'].isna()]
    
    # Remove rows with invalid freq_codes
    telem_dat = telem_dat[telem_dat['freq_code'].str.len() > 3]
    telem_dat = telem_dat[~telem_dat['freq_code'].isna()]
    
    after_cleanup = len(telem_dat)
    print(f"Removed {before_cleanup - after_cleanup} invalid records")
    print(f"Clean records remaining: {after_cleanup}")

    if after_cleanup == 0:
        raise ValueError(f"No valid records found in {file_name}")

    # Standardize columns
    telem_dat["power"] = 0.0
    telem_dat["noise_ratio"] = 0.0
    telem_dat["scan_time"] = scan_time
    telem_dat["channels"] = channels
    telem_dat["rec_type"] = rec_type

    # Calculate epoch time
    telem_dat["epoch"] = (
        telem_dat["time_stamp"] - pd.Timestamp("1970-01-01")
    ) / pd.Timedelta("1s")

    # Convert to standard data types
    telem_dat = telem_dat.astype({
        "power": "float32",
        "freq_code": "object",
        "time_stamp": "datetime64[ns]",
        "scan_time": "float32",
        "channels": "int32",
        "rec_type": "object",
        "epoch": "float32",
        "noise_ratio": "float32",
        "rec_id": "object"
    })

    # Keep only standard columns
    telem_dat = telem_dat[
        ["power", "time_stamp", "epoch", "freq_code", "noise_ratio",
         "scan_time", "channels", "rec_id", "rec_type"]
    ]

    # Append to HDF5 store
    with pd.HDFStore(db_dir, mode='a') as store:
        store.append(
            key="raw_data",
            value=telem_dat,
            format="table",
            index=False,
            min_itemsize={"freq_code": 20, "rec_type": 20, "rec_id": 20},
            append=True,
            chunksize=1000000,
            data_columns=True
        )

    print(f"\nSuccessfully parsed {file_name} and appended to {db_dir}!")
    print(f"Imported {len(telem_dat)} records in {mode_str} mode")

    with pd.HDFStore(db_dir, 'r') as store:
        print("Store keys after append:", store.keys())


# Legacy alias for backward compatibility
def PIT_Multiple(file_name, db_dir, ant_to_rec_dict, study_tags=None, skiprows=0, scan_time=0, channels=0, rec_type="PIT_Multiple"):
    """
    Legacy function for multi-antenna PIT parsing. 
    Now redirects to the unified PIT() function.
    """
    print("Note: PIT_Multiple() is deprecated. Use PIT() with ant_to_rec_dict parameter instead.")
    return PIT(
        file_name=file_name,
        db_dir=db_dir,
        rec_id=None,  # Not used in multi-antenna mode
        study_tags=study_tags,
        skiprows=skiprows,
        scan_time=scan_time,
        channels=channels,
        rec_type=rec_type,
        ant_to_rec_dict=ant_to_rec_dict
    )