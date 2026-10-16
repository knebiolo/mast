# -*- coding: utf-8 -*-
"""
Radio telemetry project management and HDF5 database operations.

This module provides the `radio_project` class, the central object for managing
radio telemetry studies. It handles project initialization, data import, metadata
storage, and database operations using HDF5 format.

Core Responsibilities
---------------------
- **Project Initialization**: Create HDF5 database with standardized table structure
- **Data Import**: Batch import from multiple receiver types and file formats
- **Metadata Management**: Store tags, receivers, recaptures, nodes, lines
- **Recapture Generation**: Process raw detections into spatiotemporal recaptures
- **Query Interface**: Retrieve fish tracks, detection statistics, project metadata

HDF5 Database Structure
-----------------------
The project database contains these primary tables:

- `/raw_data`: Imported receiver detections (time_stamp, freq_code, power, etc.)
- `/tblMasterTag`: Tag metadata (freq_code, pulse_rate, tag_type, release info)
- `/tblMasterReceiver`: Receiver metadata (rec_id, rec_type, latitude, longitude)
- `/recaptures`: Processed detections linked to spatial locations and tags
- `/nodes`: Spatial nodes for state-space modeling
- `/lines`: Connectivity between nodes for movement modeling

Classification and Filtering Tables:

- `/training`: Hand-labeled detections for classifier training
- `/test`: Detections scored by Naive Bayes classifier
- `/overlapping`: Overlapping detection decisions from overlap_reduction
- `/bouts`: Bout summaries from DBSCAN clustering
- `/presence`: Presence/absence by bout and receiver

Statistical Model Tables:

- `/cjs`: Cormack-Jolly-Seber capture history
- `/lrdr`: Live-recapture dead-recovery format
- `/tte`: Time-to-event format for survival analysis

Typical Usage
-------------
>>> from pymast.radio_project import radio_project
>>> 
>>> # Initialize new project
>>> proj = radio_project(
...     project_dir='C:/projects/my_study',
...     db_name='my_study.h5',
...     rec_list='receivers.csv',
...     tag_list='tags.csv',
...     node_list='nodes.csv',
...     line_list='lines.csv'
... )
>>> 
>>> # Import raw receiver data
>>> proj.import_data(
...     file_name='receiver_001.csv',
...     receiver_make='ares',
...     rec_id='REC001',
...     scan_time=1.0,
...     channels=1
... )
>>> 
>>> # Generate recaptures table
>>> proj.make_recaptures_table()
>>> 
>>> # Query fish tracks
>>> tracks = proj.get_fish_tracks(freq_code='166.380 7')

Notes
-----
- HDF5 format provides fast queries, compression, and hierarchical organization
- All tables use indexed columns for performance (freq_code, rec_id, time_stamp)
- Receiver imports are append-only (no overwrites unless db_dir deleted)
- Project metadata stored in HDF5 attributes for provenance

See Also
--------
parsers : Data import from various receiver formats
overlap_removal : Detection filtering and bout analysis
formatter : Statistical model output generation
"""

# import modules required for function dependencies
import numpy as np
import pandas as pd
import os
import h5py
import datetime
import logging
import pymast.naive_bayes as naive_bayes
import pymast.parsers as  parsers
import pymast.predictors as predictors
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import interpolate
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
import shutil
import warnings
# dask imports moved to make_recaptures_table to prevent import errors
# when dask has version issues but is not needed for basic operations

# Initialize logger
logger = logging.getLogger('pymast.radio_project')

font = {'family': 'serif','size': 6}
rcParams['font.size'] = 6
rcParams['font.family'] = 'serif'

push = 'push'

class radio_project():
    '''
    A class to manage and organize data and parameters for a Radio Telemetry project.

    This class is designed to facilitate the handling of datasets and parameters associated with radio telemetry studies. It organizes project data, including tag, receiver, and node information, and manages the project directory structure and database.

    Attributes:
    - project_dir (str): The directory where the project data and outputs will be stored.
    - db_name (str): The name of the project database.
    - db (str): The path to the project database file.
    - tags (DataFrame or similar): Data containing information about the tags used in the project.
    - receivers (DataFrame or similar): Data containing information about the receivers used in the project.
    - nodes (DataFrame or similar, optional): Data containing information about the nodes used in the project, if applicable.
    - data_dir (str): Directory path for storing raw data.
    - training_dir (str): Directory path for storing training files.
    - output_dir (str): Directory path for storing output files.
    - figures_dir (str): Directory path for storing figures.
    - hdf5 (h5py.File): HDF5 file object for the project database.

    Methods:
    - initialize_hdf5: Initializes the HDF5 database with initial data arrays.
    '''
    
    def __init__(self, project_dir, db_name, detection_count, duration, tag_data, receiver_data, nodes_data = None):
        """
        Initialize a radio telemetry project for data management and analysis.
        
        This constructor sets up the complete project infrastructure including:
        - Directory structure for data, training files, and outputs
        - HDF5 database for efficient data storage
        - Tag, receiver, and node metadata
        
        Parameters
        ----------
        project_dir : str
            Root directory for the project. Recommended to avoid spaces in path.
        db_name : str
            Name of the HDF5 database file (without .h5 extension).
        detection_count : int
            Number of detections to include in detection history window for 
            predictor calculation. Typical values: 3-7.
        duration : float
            Time window in seconds for noise ratio calculation. 
            Typical values: 1.0-5.0 seconds.
        tag_data : pandas.DataFrame
            Master tag table with required columns:
            - freq_code (str): Unique frequency-code combination
            - pulse_rate (float): Seconds between tag pulses
            - tag_type (str): 'study', 'BEACON', or 'TEST'
            - rel_date (datetime): Release date and time
            See docs/API_REFERENCE.md for complete schema.
        receiver_data : pandas.DataFrame
            Master receiver table with required columns:
            - rec_id (str): Unique receiver identifier
            - rec_type (str): Receiver type ('srx600', 'srx800', etc.)
            - node (str): Associated network node ID
        nodes_data : pandas.DataFrame, optional
            Network nodes table with columns:
            - node (str): Unique node identifier
            - X (int): X coordinate for visualization
            - Y (int): Y coordinate for visualization
            Required for movement analysis and overlap removal.
        
        Raises
        ------
        ValueError
            If required columns are missing from input DataFrames.
        OSError
            If project directory cannot be created.
        
        Examples
        --------
        >>> import pandas as pd
        >>> from pymast.radio_project import radio_project
        >>> 
        >>> # Load input data
        >>> tags = pd.read_csv('tblMasterTag.csv')
        >>> receivers = pd.read_csv('tblMasterReceiver.csv')
        >>> nodes = pd.read_csv('tblNodes.csv')
        >>> 
        >>> # Create project
        >>> project = radio_project(
        ...     project_dir='/path/to/project',
        ...     db_name='my_study',
        ...     detection_count=5,
        ...     duration=1.0,
        ...     tag_data=tags,
        ...     receiver_data=receivers,
        ...     nodes_data=nodes
        ... )
        
        Notes
        -----
        The project directory structure will be created as:
        - project_dir/
          - Data/ (raw data storage)
            - Training_Files/ (receiver data files)
          - Output/ (processed data and exports)
            - Figures/ (generated plots)
          - my_study.h5 (HDF5 database)
        """
        # set model parameters
        self.project_dir = project_dir     
        self.db_name = db_name
        self.db = os.path.join(project_dir,'%s.h5'%(db_name))
        self.tags = tag_data
        self.study_tags = self.tags[self.tags.tag_type == 'study'].freq_code.values
        self.test_tags = self.tags[self.tags.tag_type == 'TEST'].freq_code.values
        self.beacon_tags = self.tags[self.tags.tag_type == 'BEACON'].freq_code.values
        self.tags.set_index('freq_code', inplace = True)
        self.receivers = receiver_data
        self.receivers.set_index('rec_id', inplace = True)
        self.nodes = nodes_data
        self.det_count = detection_count
        self.noise_window = duration
        
        # create standard BIOTAS project directory if it does not already exist
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
        
        self.data_dir = os.path.join(project_dir,'Data')                                # raw data goes her
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
       
        self.training_dir = os.path.join(self.data_dir,'Training_Files')
        if not os.path.exists(self.training_dir):
            os.makedirs(self.training_dir)
       
        self.output_dir = os.path.join(self.project_dir, 'Output')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.figures_dir = os.path.join(self.output_dir, 'Figures')
        if not os.path.exists(self.figures_dir):
            os.makedirs(self.figures_dir)
        self.figure_ws = os.path.join(project_dir,'Output','Figures')                         

        # When running in automated/non-interactive mode set this flag True to avoid input() prompts
        # By default, leave interactive (False) so user can respond to prompts
        self.non_interactive = False

        # create a project database and write initial arrays to HDF
        self.initialize_hdf5()    
        
        
    def initialize_hdf5(self):
        '''Initialize an HDF5 database for a simulation'''
        
        # Create the group structure first and close the file again: pandas (PyTables) must not write
        # to a file that h5py still has open (HDF5 1.14 refuses the second handle).
        with h5py.File(self.db, 'a') as hdf5:
            new_project = 'project_setup' not in hdf5
            for group in ("project_setup", "raw_data", "trained", "classified",
                          "presence", "overlapping", "recaptures"):
                if group not in hdf5:
                    hdf5.create_group(group)

        if new_project:
            self.tags.to_hdf(self.db, key='/project_setup/tags', mode='a')
            self.receivers.to_hdf(self.db, key='/project_setup/receivers', mode='a')
            self.nodes.to_hdf(self.db, key='/project_setup/nodes', mode='a')
        else:
            # Project already exists - check for new tags and merge if needed
            try:
                existing_tags = pd.read_hdf(self.db, key='/project_setup/tags')
                
                # Reset index on incoming tags for comparison (it gets set later in __init__)
                incoming_tags = self.tags.copy()
                if incoming_tags.index.name == 'freq_code':
                    incoming_tags = incoming_tags.reset_index()
                
                # Find new tags not in existing database
                if 'freq_code' in existing_tags.columns:
                    existing_freq_codes = set(existing_tags['freq_code'])
                else:
                    existing_freq_codes = set(existing_tags.index)
                
                incoming_freq_codes = set(incoming_tags['freq_code'])
                new_freq_codes = incoming_freq_codes - existing_freq_codes
                
                if new_freq_codes:
                    print(f"Found {len(new_freq_codes)} new tags to add to database: {sorted(new_freq_codes)}")
                    
                    # Merge existing and new tags
                    new_tags_only = incoming_tags[incoming_tags['freq_code'].isin(new_freq_codes)]
                    
                    # Ensure existing_tags has freq_code as column, not index
                    if existing_tags.index.name == 'freq_code':
                        existing_tags = existing_tags.reset_index()
                    
                    merged_tags = pd.concat([existing_tags, new_tags_only], ignore_index=True)
                    
                    # Remove the old tags table and write merged version
                    with pd.HDFStore(self.db, mode='a') as store:
                        if '/project_setup/tags' in store:
                            store.remove('/project_setup/tags')
                        store.put('/project_setup/tags', 
                                  merged_tags, 
                                  format='table',
                                  data_columns=True)
                    
                    # Update self.tags with merged data
                    self.tags = merged_tags.copy()
                    self.tags.set_index('freq_code', inplace=True)
                    
                    # Update tag type arrays
                    self.study_tags = self.tags[self.tags.tag_type == 'study'].index.values
                    self.test_tags = self.tags[self.tags.tag_type == 'TEST'].index.values
                    self.beacon_tags = self.tags[self.tags.tag_type == 'BEACON'].index.values
                    
                    print(f"Successfully added {len(new_freq_codes)} new tags to database.")
                else:
                    print("No new tags found - database is up to date.")
                    
            except (KeyError, FileNotFoundError):
                # Tags table doesn't exist yet, write it
                print("Tags table not found in database, creating it now.")
                self.tags.to_hdf(self.db, key='/project_setup/tags', mode='a')
            
            # Check for new receivers and merge if needed
            try:
                existing_receivers = pd.read_hdf(self.db, key='/project_setup/receivers')
                
                # Reset index on incoming receivers for comparison
                incoming_receivers = self.receivers.copy()
                if incoming_receivers.index.name == 'rec_id':
                    incoming_receivers = incoming_receivers.reset_index()
                
                # Find new receivers not in existing database
                if 'rec_id' in existing_receivers.columns:
                    existing_rec_ids = set(existing_receivers['rec_id'])
                else:
                    existing_rec_ids = set(existing_receivers.index)
                
                incoming_rec_ids = set(incoming_receivers['rec_id'])
                new_rec_ids = incoming_rec_ids - existing_rec_ids
                
                if new_rec_ids:
                    print(f"Found {len(new_rec_ids)} new receivers to add to database: {sorted(new_rec_ids)}")
                    
                    # Merge existing and new receivers
                    new_receivers_only = incoming_receivers[incoming_receivers['rec_id'].isin(new_rec_ids)]
                    
                    # Ensure existing_receivers has rec_id as column, not index
                    if existing_receivers.index.name == 'rec_id':
                        existing_receivers = existing_receivers.reset_index()
                    
                    merged_receivers = pd.concat([existing_receivers, new_receivers_only], ignore_index=True)
                    
                    # Remove the old receivers table and write merged version
                    with pd.HDFStore(self.db, mode='a') as store:
                        if '/project_setup/receivers' in store:
                            store.remove('/project_setup/receivers')
                        store.put('/project_setup/receivers', 
                                  merged_receivers, 
                                  format='table',
                                  data_columns=True)
                    
                    # Update self.receivers with merged data
                    self.receivers = merged_receivers.copy()
                    self.receivers.set_index('rec_id', inplace=True)
                    
                    print(f"Successfully added {len(new_rec_ids)} new receivers to database.")
                else:
                    print("No new receivers found - database is up to date.")
                    
            except (KeyError, FileNotFoundError):
                # Receivers table doesn't exist yet, write it
                print("Receivers table not found in database, creating it now.")
                self.receivers.to_hdf(self.db, key='/project_setup/receivers', mode='a')


    def _prompt(self, prompt_text, default="no"):
        """Centralized prompt helper — returns default ONLY when non_interactive is True.
        
        By default (non_interactive=False), this will prompt the user interactively.
        Set project.non_interactive = True to auto-answer with defaults.
        """
        if self.non_interactive:
            logger.debug(f"Non-interactive mode: auto-answering '{prompt_text}' with '{default}'")
            return default
        try:
            return input(prompt_text)
        except (EOFError, OSError) as exc:
            raise RuntimeError(
                "Input prompt failed. Set project.non_interactive = True to use defaults."
            ) from exc
            
    def telem_data_import(self,
                          rec_id,
                          rec_type,
                          file_dir,
                          db_dir,
                          scan_time = 1, 
                          channels = 1, 
                          ant_to_rec_dict = None,
                          ka_format = False,
                          layout = 'auto',
                          scan_table = None,
                          file_pattern = None,
                          deduplicate = False):
        """
        Import raw telemetry data from receiver files into the project database.
        
        Parameters
        ----------
        rec_id : str
            Receiver ID (must exist in receiver_data)
        rec_type : str
            Receiver type. Supported: 'srx600', 'srx800', 'srx1200', 
            'orion', 'ares', 'VR2'
        file_dir : str
            Directory containing raw data files
        db_dir : str
            Path to HDF5 database file
        scan_time : float, optional
            Channel scan time in seconds (default: 1)
        channels : int, optional
            Number of channels (default: 1)
        ant_to_rec_dict : dict, optional
            Mapping of antenna IDs to receiver IDs
        ka_format : bool, optional
            Use Kleinschmidt Associates format (default: False)
        layout : str, optional
            For 'srx600' and 'srx800' receivers: 'auto' (default) identifies each file's layout from its first lines and uses the Lotek
            "Code_log" or older SRX800 importers where the usual parsers cannot read it (files that are not recognised go to the usual
            parser, as before); 'native' always uses the usual parser; 'codelog' or 'srx800_legacy' force one of the new importers.
        scan_table : dict, optional
            Channel -> frequency (MHz) for dump files that have no channel table of their own, for example {'1': '148.340', '2': '148.380'}.
        file_pattern : str or list of str, optional
            Only import files whose names match this glob pattern (or any of these patterns), for example '*.TXT'. Default: every file in the
            directory, as before. Receivers write companion files (.hex, .cfg, .csv) next to the detections; a pattern keeps them out.
        deduplicate : bool, optional
            Overlapping downloads repeat detections. The number of repeated detections (same tag, second and power) is always logged as a
            warning; with deduplicate=True the repeats are removed, keeping the first copy. Default False: results are unchanged.
        
        Raises
        ------
        ValueError
            If rec_type is not supported or rec_id not found
        FileNotFoundError
            If file_dir doesn't exist or contains no data files
        """
        # Validate receiver type (case-insensitive)
        valid_rec_types = {
            'srx600',
            'srx800',
            'srx1200',
            'orion',
            'ares',
            'vr2',
            'pit',
            'pit_multiple',
        }
        rec_type_norm = str(rec_type).strip().lower()
        if rec_type_norm not in valid_rec_types:
            raise ValueError(
                f"Unsupported receiver type: '{rec_type}'. "
                f"Supported types: {', '.join(sorted(valid_rec_types))}"
            )
        
        # Validate receiver ID
        if rec_id not in self.receivers.index:
            raise ValueError(
                f"Receiver '{rec_id}' not found in receiver_data. "
                f"Available receivers: {', '.join(self.receivers.index)}"
            )
        
        # Validate directory exists
        if not os.path.exists(file_dir):
            logger.error(f"Data directory not found: {file_dir}")
            raise FileNotFoundError(
                f"Data directory not found: {file_dir}. "
                f"Expected location: {self.training_dir}"
            )
        
        logger.info(f"Importing data for receiver {rec_id} (type: {rec_type})")
        logger.info(f"  Data directory: {file_dir}")
        
        # list raw data files (not sub-folders, in a fixed order; optionally those matching file_pattern)
        import fnmatch
        patterns = [file_pattern] if isinstance(file_pattern, str) else (list(file_pattern) if file_pattern else None)
        tFiles = sorted(f for f in os.listdir(file_dir) if not os.path.isdir(os.path.join(file_dir, f))
                        and (patterns is None or any(fnmatch.fnmatch(f, pat) for pat in patterns)))
        if layout not in ('auto', 'native', 'codelog', 'srx800_legacy'):
            raise ValueError(f"layout must be 'auto', 'native', 'codelog' or 'srx800_legacy', not {layout!r}")
        
        if not tFiles:
            logger.warning(f"No files found in {file_dir}")
            return
        
        logger.info(f"  Found {len(tFiles)} file(s) to import")
        
        # for every file call the correct text parser and import
        for i, f in enumerate(tqdm(tFiles, desc=f"Importing {rec_id}", unit="file"), 1):
            logger.debug(f"  Processing file {i}/{len(tFiles)}: {f}")
            # get the complete file directory
            f_dir = os.path.join(file_dir,f)
            
            file_layout = 'native'
            if rec_type_norm in ('srx600', 'srx800'):
                file_layout = layout if layout != 'auto' else (parsers.detect_lotek_layout(f_dir) or 'native')
                if file_layout == 'srx800':                # the layout the usual SRX800 parser reads
                    file_layout = 'native'

            if file_layout == 'codelog':
                parsers.lotek_codelog(f_dir, db_dir, rec_id, self.study_tags, scan_time = scan_time, channels = channels,
                                      ant_to_rec_dict = ant_to_rec_dict, scan_table = scan_table)

            elif file_layout == 'srx800_legacy':
                parsers.srx800_legacy(f_dir, db_dir, rec_id, self.study_tags, scan_time = scan_time, channels = channels,
                                      ant_to_rec_dict = ant_to_rec_dict, scan_table = scan_table)

            elif rec_type_norm == 'srx600' :
                parsers.srx600(f_dir, db_dir, rec_id, self.study_tags, scan_time = scan_time, channels = channels, ant_to_rec_dict = ant_to_rec_dict)
            
            elif rec_type_norm == 'srx800':
                parsers.srx800(f_dir, db_dir, rec_id, self.study_tags, scan_time = scan_time, channels = channels, ant_to_rec_dict = ant_to_rec_dict)
            
            elif rec_type_norm == 'srx1200':
                parsers.srx1200(f_dir, db_dir, rec_id, self.study_tags, scan_time = scan_time, channels = channels, ant_to_rec_dict = ant_to_rec_dict, ka_format = ka_format)
            
            elif rec_type_norm == 'orion':
                parsers.orion_import(f_dir,db_dir,rec_id, self.study_tags, scan_time = scan_time, channels = channels, ant_to_rec_dict = ant_to_rec_dict)
            
            elif rec_type_norm == 'vr2':
                parsers.vr2_import(f_dir, db_dir, self.study_tags, rec_id)
                
            elif rec_type_norm == 'ares':
                parsers.ares(f_dir,db_dir,rec_id, self.study_tags, scan_time = scan_time, channels = channels, ant_to_rec_dict = ant_to_rec_dict)

            elif rec_type_norm == 'pit':
                parsers.PIT(f_dir,db_dir,rec_id, self.study_tags, scan_time = scan_time, channels = channels, ant_to_rec_dict = ant_to_rec_dict)
                
            elif rec_type_norm == 'pit_multiple':
                parsers.PIT_Multiple(f_dir, db_dir,
                                     study_tags=self.study_tags,
                                     ant_to_rec_dict=ant_to_rec_dict,
                                     scan_time=scan_time,
                                     channels=channels)

            else:
                logger.error(f"No import routine for receiver type: {rec_type}")
                raise ValueError(f"No import routine available for receiver type: {rec_type}")
        
        logger.info(f"✓ Import complete for receiver {rec_id}: {len(tFiles)} file(s) processed")
        self._handle_duplicate_detections(rec_id, deduplicate)
        
        # Calculate and display import statistics
        try:
            raw_data = pd.read_hdf(self.db, key='raw_data', where=f'rec_id = "{rec_id}"')
            
            # Total Detection Count
            total_detections = len(raw_data)
            
            # Print to console (always visible)
            print(f"\n{'='*60}")
            print(f"IMPORT STATISTICS FOR {rec_id}")
            print(f"{'='*60}")
            print(f"Total Detection Count: {total_detections:,}")
            
            # Also log for logging systems
            logger.info(f"\n{'='*60}")
            logger.info(f"IMPORT STATISTICS FOR {rec_id}")
            logger.info(f"{'='*60}")
            logger.info(f"Total Detection Count: {total_detections:,}")
            
            if total_detections > 0:
                # Detection count summary statistics
                print(f"\nDetection Summary Statistics:")
                print(f"  Mean detections per file: {total_detections / len(tFiles):.1f}")
                print(f"  Files processed: {len(tFiles)}")
                
                logger.info(f"\nDetection Summary Statistics:")
                logger.info(f"  Mean detections per file: {total_detections / len(tFiles):.1f}")
                logger.info(f"  Files processed: {len(tFiles)}")
                
                # Unique Tag Count
                unique_tags = raw_data['freq_code'].nunique()
                print(f"\nUnique Tag Count: {unique_tags}")
                logger.info(f"\nUnique Tag Count: {unique_tags}")
                
                # Check for duplicate freq_codes in the MASTER TAGS TABLE (configuration error)
                tags_with_detections = raw_data['freq_code'].unique()
                tags_in_table = set(self.tags.index if self.tags.index.name == 'freq_code' else self.tags['freq_code'])
                
                # Check if tags table has duplicates (bad!)
                if hasattr(self.tags, 'index') and self.tags.index.name == 'freq_code':
                    if self.tags.index.duplicated().any():
                        dup_count = self.tags.index.duplicated(keep=False).sum()
                        print(f"\n⚠️  WARNING: {dup_count} duplicate freq_codes in TAGS TABLE")
                        logger.warning(f"{dup_count} duplicate freq_codes in TAGS TABLE")
                    else:
                        print(f"\nTag Table Check: ✓ No duplicate freq_codes")
                        logger.info(f"Tag Table Check: No duplicate freq_codes")
                
                # Check for orphan tags (in raw_data but not in tags table)
                orphan_tags = set(tags_with_detections) - tags_in_table
                if orphan_tags:
                    print(f"\n⚠️  WARNING: {len(orphan_tags)} orphan tags (detected but not in tags table)")
                    logger.warning(f"{len(orphan_tags)} orphan tags (detected but not in tags table)")
                else:
                    print(f"\nOrphan Tags Check: ✓ All detected tags in tags table")
                    logger.info(f"Orphan Tags Check: All detected tags in tags table")
                
                # Time Coverage
                if 'time_stamp' in raw_data.columns:
                    raw_data['time_stamp'] = pd.to_datetime(raw_data['time_stamp'])
                    start_time = raw_data['time_stamp'].min()
                    end_time = raw_data['time_stamp'].max()
                    duration = end_time - start_time
                    
                    print(f"\nTime Coverage:")
                    print(f"  Start: {start_time}")
                    print(f"  End:   {end_time}")
                    print(f"  Duration: {duration.days} days, {duration.seconds // 3600} hours")
                    
                    logger.info(f"\nTime Coverage:")
                    logger.info(f"  Start: {start_time}")
                    logger.info(f"  End:   {end_time}")
                    logger.info(f"  Duration: {duration.days} days, {duration.seconds // 3600} hours")
                    
                    # Detection rate
                    if duration.total_seconds() > 0:
                        det_per_hour = total_detections / (duration.total_seconds() / 3600)
                        print(f"  Detection rate: {det_per_hour:.1f} detections/hour")
                        logger.info(f"  Detection rate: {det_per_hour:.1f} detections/hour")
                
                print(f"{'='*60}\n")
                logger.info(f"{'='*60}\n")
            else:
                print(f"No detections found for receiver {rec_id}")
                logger.warning(f"No detections found for receiver {rec_id}")
                
        except KeyError:
            print(f"Could not retrieve statistics - raw_data table not found in database")
            logger.warning(f"Could not retrieve statistics - raw_data table not found in database")
        except Exception as e:
            print(f"Error calculating import statistics: {e}")
            logger.warning(f"Error calculating import statistics: {e}")
    
    def _handle_duplicate_detections(self, rec_id, deduplicate):
        """Report (always) and optionally remove detections repeated by overlapping downloads: same tag, second and power at one receiver."""
        with pd.HDFStore(self.db, mode='r') as store:
            if '/raw_data' not in store.keys():
                return 0
            raw = store.select('raw_data', where=f'rec_id == "{rec_id}"')
        repeated = raw.duplicated(['freq_code', 'epoch', 'power'], keep='first')
        n = int(repeated.sum())
        if n == 0:
            return 0
        message = f"receiver {rec_id}: {n:,} of {len(raw):,} detections repeat an earlier one (same tag, second and power), probably from overlapping downloads"
        if not deduplicate:
            logger.warning(message + "; kept (pass deduplicate=True to remove them)")
            warnings.warn(message + "; kept (pass deduplicate=True to remove them)", UserWarning)
            return 0
        logger.warning(message + "; removing")
        kept = raw[~repeated.values]
        with pd.HDFStore(self.db, mode='a') as store:
            if '/raw_data_dedup_tmp' in store.keys():           # left behind by an interrupted earlier attempt
                store.remove('raw_data_dedup_tmp')
            store.append('raw_data_dedup_tmp', kept, format='table', data_columns=True, index=False,
                         min_itemsize={'freq_code': 20, 'rec_type': 20, 'rec_id': 20})
            if store.get_storer('raw_data_dedup_tmp').nrows < len(kept):
                raise RuntimeError("could not stage the de-duplicated detections; the original detections were not touched")
            store.remove('raw_data', where=f'rec_id == "{rec_id}"')
            store.append('raw_data', store.select('raw_data_dedup_tmp'), format='table', data_columns=True, index=False,
                         min_itemsize={'freq_code': 20, 'rec_type': 20, 'rec_id': 20})
            store.remove('raw_data_dedup_tmp')
        return n

    def get_fish(self, rec_id, train = True, reclass_iter = None):
        logger.info(f"Getting fish for receiver {rec_id}")
        logger.debug(f"  Mode: {'training' if train else 'classification'}, Iteration: {reclass_iter}")
        
        tags_no_idx = self.tags.reset_index(drop = False)

        if reclass_iter == None and train == True:
            dat = pd.read_hdf(self.db, 
                              key = 'raw_data',
                              where = f'rec_id = "{rec_id}"')
            dat = pd.merge(dat, tags_no_idx, on='freq_code', how='left')
            dat = dat[(dat.tag_type != 'TEST') & (dat.tag_type != 'BEACON')]
            dat = dat[(dat.tag_type != 'beacon') & (dat.tag_type != 'test')]
            
        elif reclass_iter == None and train == False:
            dat = pd.read_hdf(self.db, 
                              key = 'raw_data',
                              where = f'rec_id = "{rec_id}"')
            dat = pd.merge(dat, tags_no_idx, on='freq_code', how='left')
            dat = dat[(dat.tag_type == 'study') | (dat.tag_type == 'STUDY')]

        else:
            itr = reclass_iter -1 
            dat = pd.read_hdf(self.db, 
                              key = 'classified',
                              where = f'(rec_id = "{rec_id}") & (iter == {itr})')
            dat = pd.merge(dat, tags_no_idx, on='freq_code', how='left')
            dat = dat[dat.tag_type == 'study']
    
        fish_list = dat.freq_code.unique()
        logger.info(f"  Found {len(fish_list)} unique fish")
        return fish_list

    def orphan_tags(self, return_rows=False):
        """Return orphan tags or their recapture rows.

        By default returns a sorted list of orphan `freq_code` strings (tags
        present in `/recaptures` but missing from `/project_setup/tags`). If
        `return_rows=True` returns the recaptures DataFrame rows for those tags.
        """
        recaps = pd.read_hdf(self.db, 'recaptures')
        recaps['freq_code'] = recaps['freq_code'].astype(str)

        master = self.tags.copy()
        if master.index.name == 'freq_code':
            master_codes = set(master.index.astype(str))
        else:
            master_codes = set(master['freq_code'].astype(str))

        recap_codes = set(recaps['freq_code'].unique())
        orphans = sorted(list(recap_codes - master_codes))

        if return_rows:
            if not orphans:
                return pd.DataFrame(columns=recaps.columns)
            return recaps[recaps['freq_code'].isin(orphans)].copy()
        return orphans
    
    def train(self, freq_code, rec_id):
        """
        Train the Naive Bayes classifier using a specific tag at a receiver.
        
        This method calculates predictor variables for all detections of the 
        specified tag and stores them in the training dataset. Training data
        includes both known true positives (from beacon/test tags) and known
        false positives (miscoded detections).
        
        Parameters
        ----------
        freq_code : str
            Frequency-code combination to train on (e.g., '164.123 45').
            Must exist in the tag_data provided during initialization.
        rec_id : str
            Receiver ID where training data was collected.
            Must exist in the receiver_data provided during initialization.
        
        Returns
        -------
        None
            Training data is written to HDF5 database at /trained key.
        
        Raises
        ------
        KeyError
            If freq_code or rec_id not found in project data.
        ValueError
            If insufficient data for training (e.g., no detections).
        
        Examples
        --------
        >>> # Train on a single tag
        >>> project.train('164.123 45', 'R01')
        
        >>> # Train on all tags at a receiver
        >>> fishes = project.get_fish(rec_id='R01')
        >>> for fish in fishes:
        ...     project.train(fish, 'R01')
        
        See Also
        --------
        training_summary : Generate statistics and plots from training data
        reclassify : Apply trained classifier to classify detections
        
        Notes
        -----
        Predictor variables calculated:
        - hit_ratio: Proportion of expected detections received
        - cons_length: Maximum consecutive detection length
        - noise_ratio: Ratio of miscoded to total detections
        - power: Signal strength
        - lag_diff: Second-order difference in detection timing
        """
        '''A class object for a training dataframe and related data objects.'''
    
        self.train_receiver(rec_id, freq_codes=[freq_code])

    TRAINED_DTYPES = {'power': 'float32', 'time_stamp': 'datetime64[ns]', 'epoch': 'int64', 'freq_code': 'object', 'noise_ratio': 'float32',
                      'scan_time': 'int32', 'channels': 'int32', 'rec_type': 'object', 'rec_id': 'object', 'detection': 'int32',
                      'lag': 'float32', 'lag_diff': 'float32', 'det_hist': 'object', 'hit_ratio': 'float32', 'cons_det': 'int32',
                      'cons_length': 'float32'}

    def _train_fish_frame(self, train_dat, freq_code):
        """Predictor variables and the detection class for one tag at one receiver (the body of the original `train`)."""
        train_dat['time_stamp'] = pd.to_datetime(train_dat.time_stamp)
        train_dat['epoch'] = parsers._epoch_seconds(train_dat.time_stamp).values
        train_dat.sort_values(by='epoch', inplace=True)
        train_dat.drop_duplicates(subset='time_stamp', keep='first', inplace=True)

        plausible = 1 if freq_code in self.study_tags else 0       # detection class: a tag that was deployed is plausible
        pulse_rate = self.tags.at[freq_code, 'pulse_rate'] if freq_code in self.tags.index else 673.
        train_dat['detection'] = np.repeat(plausible, len(train_dat))
        train_dat['lag'] = train_dat.epoch.diff()
        train_dat['lag_diff'] = train_dat.lag.diff()

        detection_history, hit_ratio_arr, cons_det_arr, max_count_arr = predictors.detection_history(
            train_dat['epoch'].values, pulse_rate, self.det_count, train_dat['channels'].values, train_dat['scan_time'].values)
        train_dat['det_hist'] = (detection_history.astype(np.uint8) + 48).view(f'S{detection_history.shape[1]}').ravel().astype('U').astype(object)
        train_dat['hit_ratio'] = hit_ratio_arr
        train_dat['cons_det'] = cons_det_arr
        train_dat['cons_length'] = max_count_arr
        train_dat.fillna(value=9999999, inplace=True)
        return train_dat

    def train_receiver(self, rec_id, freq_codes=None, write=True):
        """Calculate the training predictors for every tag heard at one receiver, in a single pass.

        Reads the receiver's detections once, applies the per-tag computation of `train` to each code, and appends one table to
        `/trained`. Beacon and test tags are skipped, as in `get_fish(train=True)`. Orders of magnitude faster than calling `train`
        per tag, which reads and writes the database once per tag.

        Parameters
        ----------
        rec_id : str
            Receiver to train on.
        freq_codes : iterable of str, optional
            Restrict to these codes (default: every code heard at the receiver except beacon and test tags).
        write : bool, optional
            Append to `/trained` (default); if False the table is returned instead.
        """
        if rec_id not in self.receivers.index and not (pd.api.types.is_integer_dtype(self.receivers.index.dtype) and np.int64(rec_id) in self.receivers.index):
            raise ValueError(f"Receiver '{rec_id}' not found in receiver_data")
        dat = pd.read_hdf(self.db, 'raw_data', where=f'rec_id == "{rec_id}"')
        if freq_codes is not None:
            dat = dat[dat.freq_code.isin(set(freq_codes))]
        else:
            skip = set(self.tags.index[self.tags.tag_type.astype(str).str.lower().isin(['test', 'beacon'])])
            dat = dat[~dat.freq_code.isin(skip)]
        frames = [self._train_fish_frame(group.copy(), freq_code) for freq_code, group in dat.groupby('freq_code', sort=True)]
        if not frames:
            logger.info(f"  nothing to train on at receiver {rec_id}")
            return None
        out = pd.concat(frames, ignore_index=True)
        try:
            out = out.astype(self.TRAINED_DTYPES)
        except ValueError as exc:
            raise ValueError(f"cannot convert the training table for receiver {rec_id} to the stored column types: {exc}") from exc
        if not write:
            return out
        with pd.HDFStore(self.db, mode='a') as store:
            store.append(key='trained', value=out, format='table', index=False,
                         min_itemsize={'freq_code': 20, 'rec_type': 20, 'rec_id': 20, 'det_hist': 20},
                         append=True, chunksize=1000000)
        logger.info(f"Training complete: {len(frames)} codes ({int(out.detection.sum())} plausible detections) at {rec_id}")
        return None

    def training_summary(self,rec_type,site = None):
        logger.info(f"Generating training summary for {rec_type}")
        
        # connect to database and get data
        trained_dat = pd.read_hdf(self.db,key = 'trained')#, mode = 'r')
        trained_dat = trained_dat[(trained_dat.rec_type == rec_type)] 
        
        logger.info(f"  Loaded {len(trained_dat)} detections from {len(trained_dat.rec_id.unique())} receivers")

        det_class_count = trained_dat.groupby('detection')['detection'].count().to_frame()

        logger.info("")
        logger.info("Training Summary Statistics Report")
        logger.info("="*80)
        logger.info(f"Collected {len(trained_dat)} detections from {len(trained_dat.rec_id.unique())} {rec_type} receivers")
        logger.info("="*80)
        logger.info("")
        logger.info(f"{rec_type} detection class statistics:")
        try:
            prior_true = round(float(det_class_count.at[1,'detection'])/float(det_class_count.sum()),3)
            logger.info(f"  Prior P(true detection) = {prior_true}")
        except KeyError:
            logger.warning("  No known true detections found")
            pass
        try:
            prior_false = round(float(det_class_count.at[0,'detection'])/float(det_class_count.sum()),3)
            logger.info(f"  Prior P(false positive) = {prior_false}")
        except KeyError:
            logger.warning("  No known false positives found")
            pass

        logger.info("")
        logger.info("="*80)
        logger.info("")
        trained_dat['detection'] = trained_dat.detection.astype('str')
        sta_class_count = trained_dat.groupby(['rec_id','detection'])['detection'].count().rename('det_class_count').to_frame().reset_index()
        recs = sorted(sta_class_count.rec_id.unique())
        logger.info("Detection Class Counts Across Stations")
        logger.info("             Known          Known")
        logger.info("             False          True")
        logger.info("       ______________________________")
        logger.info("      |              |              |")
        for i in recs:
            trues = sta_class_count[(sta_class_count.rec_id == i) & (sta_class_count.detection == '1')]
            falses = sta_class_count[(sta_class_count.rec_id == i) & (sta_class_count.detection == '0')]
            if len(trues) > 0 and len(falses) > 0:
                logger.info("%6s|   %8s   |   %8s   |"%(i,falses.det_class_count.values[0],trues.det_class_count.values[0]))
            elif len(trues) == 0 and len(falses) > 0:
                logger.info("%6s|   %8s   |   %8s   |"%(i,falses.det_class_count.values[0],0))
            else:
                try:
                    logger.info("%6s|   %8s   |   %8s   |"%(i,0,trues.det_clas_count.values[0]))

                except AttributeError:
                    logger.info("%6s|   %8s   |   %8s   |"%(i,0,0))

        logger.info("      |______________|______________|")
        logger.info("")
        logger.info("="*80)
        logger.info("Compiling training figures...")
        # get data by detection class for side by side histograms
        trained_dat['power']= trained_dat.power.astype(float)
        trained_dat['lag_diff'] = trained_dat.lag_diff.astype(float)
        trained_dat['cons_length'] = trained_dat.cons_length.astype(float)
        trained_dat['noise_ratio'] = trained_dat.noise_ratio.astype(float)

        trues = trained_dat[trained_dat.detection == '1']
        falses = trained_dat[trained_dat.detection == '0']
        # plot hit ratio histograms by detection class
        hitRatioBins =np.linspace(0,1.0,11)

        # plot signal power histograms by detection class
        minPower = trained_dat.power.min()//5 * 5
        maxPower = trained_dat.power.max()//5 * 5
        powerBins =np.arange(minPower,maxPower+20,10)

        # Lag Back Differences - how stdy are detection lags?
        lagBins =np.arange(-100,110,6)

        # Consecutive Record Length ?
        conBins =np.arange(0,12,1)

        # Noise Ratio
        noiseBins =np.arange(0,1.1,0.1)

        # make lattice plot for pubs
        figSize = (3,7)
        plt.figure()
        fig, axs = plt.subplots(5,2,tight_layout = True,figsize = figSize)
        # hit ratio
        axs[0,1].hist(trues.hit_ratio.values, hitRatioBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[0,0].hist(falses.hit_ratio.values, hitRatioBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[0,0].set_xlabel('Hit Ratio')
        axs[0,1].set_title('True')
        axs[0,1].set_xlabel('Hit Ratio')
        axs[0,0].set_title('False Positive')
        axs[0,0].set_title('A',loc = 'left')

        # consecutive record length
        axs[1,1].hist(trues.cons_length.values, conBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[1,0].hist(falses.cons_length.values, conBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[1,0].set_xlabel('Consecutive Hit Length')
        axs[1,1].set_xlabel('Consecutive Hit Length')
        axs[1,0].set_title('B',loc = 'left')

        # power
        axs[2,1].hist(trues.power.values, powerBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[2,0].hist(falses.power.values, powerBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[2,0].set_xlabel('Signal Power')
        axs[2,1].set_xlabel('Signal Power')
        axs[2,0].set_ylabel('Probability Density')
        axs[2,0].set_title('C',loc = 'left')

        # noise ratio
        axs[3,1].hist(trues.noise_ratio.values, noiseBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[3,0].hist(falses.noise_ratio.values, noiseBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[3,0].set_xlabel('Noise Ratio')
        axs[3,1].set_xlabel('Noise Ratio')
        axs[3,0].set_title('D',loc = 'left')

        # lag diff
        axs[4,1].hist(trues.lag_diff.values, lagBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[4,0].hist(falses.lag_diff.values, lagBins, density = True, color = 'grey', edgecolor='black', linewidth=1.2)
        axs[4,0].set_xlabel('Lag Differences')
        axs[4,1].set_xlabel('Lag Differences')
        axs[4,0].set_title('E',loc = 'left')
        
        plt.show()
        

        plt.savefig(os.path.join(self.figure_ws,"%s_lattice_train.png"%(rec_type)),bbox_inches = 'tight', dpi = 900)
        
    def undo_training(self, rec_id, freq_code=None):
        # Read the table from the HDF5 file
        with pd.HDFStore(self.db, 'r+') as store:
            if 'trained' in store:
                df = store['trained']  # Replace with your table name
        
                # Build the condition based on provided arguments
                condition = (df['rec_id'] == rec_id)
                if freq_code is not None:
                    condition &= (df['freq_code'] == freq_code)

                # Update or delete the rows based on your requirement
                # For deletion:
                df = df[~condition]
                
                # Delete the existing table
                store.remove('trained')
        
                # Save the modified DataFrame back to the HDF5 file
                store.put('trained',
                          df,
                          format='table',
                          data_columns=True, 
                          append = False,
                          min_itemsize = {'freq_code':20,
                                          'rec_type':20,
                                          'rec_id':20,
                                          'det_hist':20})      

    def undo_import(self, rec_id):
        # Read the table from the HDF5 file
        with pd.HDFStore(self.db, 'r+') as store:
            if 'raw_data' in store:
                df = store['raw_data']  # Replace with your table name
        
                # Build the condition based on provided arguments
                condition = (df['rec_id'] == rec_id)

                # Update or delete the rows based on your requirement
                # For deletion:
                df = df[~condition]
                
                # Delete the existing table
                store.remove('raw_data')
        
                # Save the modified DataFrame back to the HDF5 file
                store.put('raw_data',
                          df,
                          format='table',
                          data_columns=True, 
                          append = False,
                          min_itemsize = {'freq_code':20,
                                          'rec_type':20,
                                          'rec_id':20})  
        
    def create_training_data(self, rec_type=None, reclass_iter=None, rec_list=None, columns=None):
        """
        Function to create a training dataset for the current round of classification.
        The function supports multiple pathways for generating training data, including 
        using a receiver list (rec_list) and incorporating reclassification methods.
    
        Parameters
        ----------
        rec_type : str, optional
            The type of receiver to filter the data by. This restricts the training data 
            to a specific receiver type (e.g., 'orion', 'srx800'). If not provided and 
            `rec_list` is used, it is ignored.
        reclass_iter : int, optional
            Iteration number for reclassification. If provided, the function pulls the 
            previous classification data and incorporates known false positives and 
            assumed true positives.
        rec_list : list of str, optional
            A list of receiver IDs to filter the data by. If provided, the function 
            queries the HDF database using this list directly rather than the receiver 
            type (`rec_type`).
        columns : list of str, optional
            Predictor columns to return (plus `detection`). Reading only these keeps the training set small enough for seasons of tens of
            millions of detections; the rows are exactly those returned without this option. Default: every column, as before.
    
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the training data for the classification process, 
            incorporating any previous classifications if applicable.
    
        Notes
        -----
        - If both `rec_type` and `rec_list` are provided, the function will prioritize 
          the `rec_list` to restrict the training data.
        - Reclassification logic is based on contributions from T. Castro-Santos.
        """
        logger.debug(f"  Creating training data (rec_type={rec_type}, iter={reclass_iter}, rec_list={rec_list})")
        if columns is not None:
            return self._training_columns(rec_type, reclass_iter, rec_list, list(columns))
        
        if rec_list is not None:
            # Construct the query for multiple receiver IDs using the OR operator
            rec_query = ' | '.join([f'rec_id == "{rec_id}"' for rec_id in rec_list])
            train_dat = pd.read_hdf(self.db, 'trained', where=rec_query)
        elif rec_type is not None:
            # Query based on receiver type directly
            train_dat = pd.read_hdf(self.db, 'trained', where=f'rec_type == "{rec_type}"')
        else:
            raise ValueError("Either 'rec_type' or 'rec_list' must be provided to create training data.")
    
        # Handling reclassification if this is not the first iteration
        if reclass_iter is not None:
            last_class = reclass_iter - 1
            
            # Load the classified dataset and filter by iteration
            class_dat = pd.read_hdf(self.db, 'classified', where=f'iter == {last_class}')
            
            # Further restrict classified data to the receiver list if rec_list is provided
            if rec_list is not None:
                class_query = ' | '.join([f'rec_id == "{rec_id}"' for rec_id in rec_list])
                class_dat = class_dat.query(class_query)
            
            # Selecting relevant columns for the training dataset
            columns = ['test', 'freq_code', 'power', 'noise_ratio', 'lag', 
                       'lag_diff', 'cons_length', 'cons_det', 'det_hist', 
                       'hit_ratio', 'rec_type', 'epoch']
            
            class_dat = class_dat[columns]
            class_dat.rename(columns={'test': 'detection'}, inplace=True)
            
            # Separate known falses (train_dat) and assumed trues (class_dat)
            train_dat = train_dat[train_dat['detection'] == 0]
            class_dat = class_dat[class_dat['detection'] == 1]
    
            # Append the classified data to the training data
            train_dat = pd.concat([train_dat, class_dat], ignore_index=True)
            logger.debug(f"    Combined training data: {len(train_dat)} detections ({sum(train_dat['detection']==0)} false, {sum(train_dat['detection']==1)} true)")
        else:
            logger.debug(f"    Training data: {len(train_dat)} detections")
    
        return train_dat



    
    def reclassify(self, project, rec_id, threshold_ratio, likelihood_model, rec_type=None, rec_list=None):
        """
        Reclassifies fish in a project based on user-defined criteria and threshold ratios.
    
        Parameters
        ----------
        project : object
            The project object that contains methods for managing and classifying fish data.
            
        rec_id : int or str
            The unique identifier for the receiver to be reclassified.
            
        threshold_ratio : float
            The threshold ratio used for determining classification criteria.
            
        likelihood_model : list of str
            The fields to use as the likelihood model for classification.
            
        rec_type : str, optional
            The type of receiver being processed (e.g., 'srx1200', 'orion').
            
        rec_list : list of str, optional
            A list of receiver IDs to filter the data by, used for creating training data.
    
        Notes
        -----
        - The classification process involves interactive user input to determine if additional
          iterations are needed.
        - The fields used for classification are hardcoded as ['hit_ratio', 'cons_length',
          'noise_ratio', 'power', 'lag_diff'].
        """
        logger.info(f"Starting classification for receiver {rec_id}")
        logger.info(f"  Threshold ratio: {threshold_ratio}")
        logger.info(f"  Likelihood model: {', '.join(likelihood_model)}")
        
        # Validate inputs
        if rec_id not in self.receivers.index:
            logger.error(f"Receiver {rec_id} not found")
            raise ValueError(f"Receiver '{rec_id}' not found in receiver_data")
        
        valid_predictors = ['hit_ratio', 'cons_length', 'noise_ratio', 'power', 'lag_diff']
        invalid = set(likelihood_model) - set(valid_predictors)
        if invalid:
            logger.error(f"Invalid predictors: {invalid}")
            raise ValueError(f"Invalid predictors: {', '.join(invalid)}. Valid: {', '.join(valid_predictors)}")
        
        class_iter = None
        
        while True:
            iter_label = f"iteration {class_iter}" if class_iter else "initial classification"
            logger.info(f"Running {iter_label}...")
            
            # Get a list of fish to iterate over
            fishes = project.get_fish(rec_id=rec_id, train=False, reclass_iter=class_iter)
            logger.info(f"  Found {len(fishes)} fish to classify")
            
            # Generate training data for the classifier
            logger.info("  Creating training data...")
            training_data = project.create_training_data(rec_type=rec_type, reclass_iter=class_iter, rec_list=rec_list)
            logger.info(f"  Training data: {len(training_data)} detections")
    
            # Iterate over fish and classify with progress bar
            logger.info("  Classifying detections...")
            for fish in tqdm(fishes, desc=f"  Classifying {rec_id}", unit="fish"):
                project.classify(fish, rec_id, likelihood_model, training_data, class_iter, threshold_ratio)
            
            # Generate summary statistics
            logger.info("  Generating classification summary...")
            project.classification_summary(rec_id, class_iter)
            
            # Show the figures and block execution until they are closed
            plt.show(block=True)
            
            # Ask the user if they need another iteration (use _prompt helper)
            user_input = str(self._prompt("\nDo you need another classification iteration? (yes/no): ", default="no")).strip().lower()
            
            if user_input in ['yes', 'y']:
                # If yes, increase class_iter and reclassify
                if class_iter is None:
                    class_iter = 2
                else:
                    class_iter += 1
                logger.info(f"Starting iteration {class_iter}")
            elif user_input in ['no', 'n']:
                # If no, break the loop
                logger.info(f"✓ Classification complete for {rec_id}")
                break
            else:
                logger.warning("Invalid input, please enter 'yes' or 'no'")


    CLASSIFIED_DTYPES = {'freq_code': 'object', 'epoch': 'int64', 'rec_id': 'object', 'time_stamp': 'datetime64[ns]', 'power': 'float32',
                         'noise_ratio': 'float32', 'scan_time': 'int32', 'channels': 'int32', 'rec_type': 'object', 'lag': 'float32',
                         'lag_diff': 'float32', 'det_hist': 'object', 'hit_ratio': 'float32', 'cons_det': 'int32', 'cons_length': 'float32',
                         'likelihood_T': 'float32', 'likelihood_F': 'float32', 'posterior_T': 'float32', 'posterior_F': 'float32',
                         'test': 'int32', 'iter': 'int32'}
    CLASSIFY_INPUT_COLUMNS = ['freq_code', 'epoch', 'rec_id', 'time_stamp', 'power', 'noise_ratio', 'scan_time', 'channels', 'rec_type']

    def classification_tables(self, training_data, fields, legacy=False):
        """Build, once per classification round, everything `classify_receiver` needs from the training data.

        Parameters
        ----------
        training_data : pandas.DataFrame
            Output of `create_training_data`: a `detection` column (1 true, 0 false) and one column per field.
        fields : list of str
            Predictors used by the likelihood model.
        legacy : bool, optional
            Reproduce PyMAST 1.1.x lookups exactly (floating-point observations are not cast to the training dtype, so many hit-ratio
            values silently receive the default likelihood). Default False.

        Returns
        -------
        tuple
            (tables, priors): a LikelihoodTable per field and the (prior_true, prior_false) pair.
        """
        if len(training_data) == 0:
            raise ValueError("training data is empty: no detections to learn true and false classes from. Check that the receivers in "
                             "rec_list/rec_type were imported and trained (project.train) before classifying")
        missing = [f for f in list(fields) + ['detection'] if f not in training_data.columns]
        if missing:
            raise ValueError(f"training data lacks columns {missing}; it has {list(training_data.columns)}")
        labels = training_data.detection.values
        if labels.min() == labels.max():
            raise ValueError("training data holds only one class (all detections are "
                             f"{'true' if labels[0] else 'false'}); both true and false examples are needed to classify")
        tables = {f: naive_bayes.LikelihoodTable(training_data[f].values, labels, match_training_dtype=not legacy) for f in fields}
        return tables, naive_bayes.calculate_priors(labels)

    def _classify_fish_frame(self, class_dat, freq_code, fields, tables, priors, reclass_iter, threshold_ratio):
        """Predictors, likelihoods, posteriors and the true/false call for the detections of one fish at one receiver."""
        try:
            pulse_rate = self.tags.at[freq_code, 'pulse_rate']
        except KeyError:
            pulse_rate = 9999.
        class_dat = class_dat[self.CLASSIFY_INPUT_COLUMNS].drop_duplicates()
        if len(class_dat) == 0:
            return None
        class_dat['time_stamp'] = pd.to_datetime(class_dat['time_stamp'])
        class_dat.sort_values(by='time_stamp', inplace=True)
        class_dat['epoch'] = parsers._epoch_seconds(class_dat.time_stamp).values
        class_dat = class_dat.drop_duplicates(subset='time_stamp')

        class_dat['lag'] = class_dat.epoch.diff()
        class_dat['lag_diff'] = class_dat.lag.diff()
        class_dat.fillna(value=99999999, inplace=True)

        detection_history, hit_ratio_arr, cons_det_arr, max_count_arr = predictors.detection_history(
            class_dat['epoch'].values, pulse_rate, self.det_count, class_dat['channels'].values, class_dat['scan_time'].values)
        # '0'/'1' string per detection, built from bytes rather than a Python join per row
        class_dat['det_hist'] = (detection_history.astype(np.uint8) + 48).view(f'S{detection_history.shape[1]}').ravel().astype('U').astype(object)
        class_dat['hit_ratio'] = hit_ratio_arr
        class_dat['cons_det'] = cons_det_arr
        class_dat['cons_length'] = max_count_arr

        hit_ratio, power, lag, con_len, noise = naive_bayes.bin_predictors(
            class_dat.hit_ratio, class_dat.power, class_dat.lag_diff, class_dat.cons_length, class_dat.noise_ratio)
        binned_predictors = {'hit_ratio': hit_ratio, 'power': power, 'lag_diff': lag, 'cons_length': con_len, 'noise_ratio': noise}

        likelihoods = {True: dict(), False: dict()}
        for assumption in (True, False):
            for field in fields:
                likelihoods[assumption][field] = tables[field].lookup(assumption, class_dat[field].values)
        likelihood_T = np.repeat(1., len(class_dat))
        for pred in likelihoods[True]:
            likelihood_T = likelihood_T * likelihoods[True][pred]
        likelihood_F = np.repeat(1., len(class_dat))
        for pred in likelihoods[False]:
            likelihood_F = likelihood_F * likelihoods[False][pred]

        observation_arrays = [binned_predictors[f] if f not in ('cons_det', 'series_hit') else class_dat[f].values for f in fields]
        evidence = naive_bayes.calculate_evidence(observation_arrays)
        prior_arrays = {True: np.repeat(priors[0], evidence.shape), False: np.repeat(priors[1], evidence.shape)}
        posterior_T = naive_bayes.calculate_posterior(prior_arrays, evidence, likelihoods, True)
        posterior_F = naive_bayes.calculate_posterior(prior_arrays, evidence, likelihoods, False)
        classification = naive_bayes.classify_with_threshold(posterior_T, posterior_F, threshold_ratio)

        class_dat['likelihood_T'] = likelihood_T
        class_dat['likelihood_F'] = likelihood_F
        class_dat['posterior_T'] = posterior_T
        class_dat['posterior_F'] = posterior_F
        class_dat['test'] = classification
        class_dat['iter'] = np.repeat(1 if reclass_iter is None else reclass_iter, len(class_dat))
        return class_dat.astype(self.CLASSIFIED_DTYPES)

    def classify_receiver(self, rec_id, fields, training_data=None, reclass_iter=None, threshold_ratio=None,
                          freq_codes=None, classifier=None, legacy=False, write=True):
        """Classify every fish detected at one receiver in a single pass.

        Reads the receiver's detections once, applies the same per-fish computation as `classify`, and appends one table to
        `/classified`. Use `classification_tables` once per round and pass the result as `classifier` to avoid rebuilding the
        likelihood tables for every receiver.

        Parameters
        ----------
        rec_id : str
            Receiver to classify.
        fields : list of str
            Predictors in the likelihood model.
        training_data : pandas.DataFrame, optional
            Needed unless `classifier` is given.
        reclass_iter : int, optional
            None for the initial classification (reads `/raw_data`); k >= 2 reclassifies the detections called true in round k-1.
        threshold_ratio : float, optional
            Posterior ratio required to call a detection true (default 1.0, the MAP rule). Earlier versions ignored this argument.
        freq_codes : iterable of str, optional
            Restrict to these tags (default: every study tag detected at the receiver).
        classifier : tuple, optional
            Result of `classification_tables`.
        legacy : bool, optional
            See `classification_tables`.
        write : bool, optional
            Append the result to `/classified` (default). If False the table is returned instead of written.

        Returns
        -------
        pandas.DataFrame or None
            The classified detections when `write` is False, otherwise None.
        """
        if rec_id not in self.receivers.index:
            raise ValueError(f"Receiver '{rec_id}' not found in receiver_data")
        threshold_ratio = 1.0 if threshold_ratio is None else threshold_ratio
        if classifier is None:
            if training_data is None:
                raise ValueError("give either training_data or a classifier built by classification_tables")
            classifier = self.classification_tables(training_data, fields, legacy=legacy)
        tables, priors = classifier
        if reclass_iter is None:
            dat = pd.read_hdf(self.db, 'raw_data', where=f'rec_id == "{rec_id}"')
        else:
            dat = pd.read_hdf(self.db, 'classified', where=f'(rec_id == "{rec_id}") & (iter == {reclass_iter - 1}) & (test == 1)')
        dat = dat[self.CLASSIFY_INPUT_COLUMNS]
        study = set(self.tags.index[self.tags.tag_type.astype(str).str.lower() == 'study'])
        wanted = study if freq_codes is None else set(freq_codes)
        dat = dat[dat.freq_code.isin(wanted)]
        frames = []
        for freq_code, group in dat.groupby('freq_code', sort=True):
            result = self._classify_fish_frame(group, freq_code, fields, tables, priors, reclass_iter, threshold_ratio)
            if result is not None:
                frames.append(result)
        if not frames:
            logger.info(f"  nothing to classify for receiver {rec_id} (iter {reclass_iter})")
            return None
        out = pd.concat(frames, ignore_index=True)
        if not write:
            return out
        with pd.HDFStore(self.db, mode='a') as store:
            store.append(key='classified', value=out, format='table', index=False,
                         min_itemsize={'freq_code': 20, 'rec_type': 20, 'rec_id': 20, 'det_hist': 20},
                         append=True, data_columns=True, chunksize=1000000)
        logger.debug(f"    classified {len(out)} detections of {len(frames)} fish at {rec_id}: {int(out.test.sum())} true")
        return None

    def classify(self,
                 freq_code,
                 rec_id,
                 fields,
                 training_data,
                 reclass_iter=None,
                 threshold_ratio=None,
                 legacy=False):
        """Classify one fish at one receiver (kept for compatibility).

        Builds the likelihood tables from `training_data` on every call, which is slow when called in a loop; use `classify_receiver`
        with `classification_tables` for whole receivers. Differences from PyMAST 1.1.x: `threshold_ratio` is honoured (it was
        ignored and 1.0 always used), and floating-point observations are matched to the training dtype (pass legacy=True to restore
        the old lookups).
        """
        self.classify_receiver(rec_id, fields, training_data=training_data, reclass_iter=reclass_iter,
                               threshold_ratio=threshold_ratio, freq_codes=[freq_code], legacy=legacy)

    def _training_columns(self, rec_type, reclass_iter, rec_list, columns):
        """create_training_data restricted to `columns`; same rows as the full version."""
        if rec_list is not None:
            query = " | ".join(f'rec_id == "{r}"' for r in rec_list)
        elif rec_type is not None:
            query = f'rec_type == "{rec_type}"'
        else:
            raise ValueError("Either 'rec_type' or 'rec_list' must be provided to create training data.")
        with pd.HDFStore(self.db, mode='r') as store:
            train = store.select('trained', where=query, columns=['detection'] + columns)
            if reclass_iter is not None:
                train = train[train['detection'] == 0]
                where = f'iter == {reclass_iter - 1} & test == 1'
                if rec_list is not None:
                    where += f' & ({query})'
                trues = store.select('classified', where=where, columns=['test'] + columns).rename(columns={'test': 'detection'})
                train = pd.concat([train, trues], ignore_index=True)
        return train.reset_index(drop=True)

    ROUNDS_COLUMNS = ['iter', 'rec_id', 'detections', 'true', 'false', 'flipped', 'flip_share', 'true_change']

    def _read_table(self, key):
        with pd.HDFStore(self.db, mode='r') as store:
            return store.select(key) if '/' + key in store.keys() else None

    def classify_until_stable(self, rec_ids=None, fields=('hit_ratio', 'cons_length', 'noise_ratio', 'lag_diff'), threshold_ratio=1.0,
                              flip_tol=0.005, count_tol=0.01, max_iter=8, legacy=False, group_by='rec_type'):
        """Classify, then reclassify, until the result is stable (the non-interactive form of `reclassify`).

        Round 1 classifies the raw detections. Round k >= 2 trains on the known-false training detections plus the detections round k-1
        called true, and classifies those. Receivers are grouped (by default by their `rec_type`) and each group is trained and classified
        together. The loop stops when, for every receiver, fewer than `flip_tol` of the detections change class between two rounds and the
        number called true moves by less than `count_tol`, or after `max_iter` rounds; in the latter case a warning says which receivers
        were still changing. Rounds and per-receiver progress are stored in `/classification_rounds` and `/classification_progress`, so an
        interrupted run resumes at the first unfinished receiver.

        Self-training can keep shedding real detections round after round, so 'nothing changes' is not a safe stopping rule on its own:
        read `expected_false_share` for each round as well.

        Parameters
        ----------
        rec_ids : list of str, optional
            Receivers to classify (default: every receiver with raw data).
        fields : sequence of str
            Predictors in the likelihood model.
        threshold_ratio : float
            Posterior ratio required to call a detection true (1.0 is the MAP rule).
        flip_tol, count_tol : float
            Stability tolerances described above.
        max_iter : int
            Maximum number of rounds.
        legacy : bool
            Reproduce PyMAST 1.1.x likelihood lookups (see `classification_tables`).
        group_by : str
            Column of `receivers` that defines groups pooled for training.

        Returns
        -------
        pandas.DataFrame
            One row per receiver per round: detections, true, false, flipped, flip_share and true_change.
        """
        import time
        fields = list(fields)
        if rec_ids is None:
            with pd.HDFStore(self.db, mode='r') as store:
                rec_ids = sorted(store.select_column('raw_data', 'rec_id').unique())
        groups = {}
        for rec in rec_ids:
            if rec not in self.receivers.index:
                raise ValueError(f"Receiver '{rec}' not found in receiver_data")
            groups.setdefault(self.receivers.at[rec, group_by], []).append(rec)

        rounds = self._read_table('classification_rounds')
        rows = rounds.to_dict('records') if rounds is not None else []
        it_done = max((int(r['iter']) for r in rows), default=0)
        progress_table = self._read_table('classification_progress')
        progress = {(int(r.iter), str(r.rec_id)) for r in progress_table.itertuples()} if progress_table is not None else set()
        with pd.HDFStore(self.db, mode='a') as store:                # remove the rows of any receiver that was cut off mid-round
            if '/classified' in store.keys():
                store.remove('classified', where=f'iter > {it_done + 1}')
                started = set(store.select('classified', where=f'iter == {it_done + 1}', columns=['rec_id']).rec_id.unique())
                for rec in sorted(started):
                    if (it_done + 1, rec) not in progress:
                        logger.info(f"round {it_done + 1}: receiver {rec} was cut off; removing its partial rows")
                        store.remove('classified', where=f'iter == {it_done + 1} & rec_id == "{rec}"')

        def snapshot(rec, it):
            with pd.HDFStore(self.db, mode='r') as store:
                return store.select('classified', where=f'rec_id == "{rec}" & iter == {it}', columns=['freq_code', 'epoch', 'test'])

        def is_stable(round_rows):
            return all(r['flip_share'] < flip_tol and r['true_change'] < count_tol for r in round_rows)

        if it_done > 1 and (it_done >= max_iter or is_stable([r for r in rows if int(r['iter']) == it_done])):
            return pd.DataFrame(rows, columns=self.ROUNDS_COLUMNS)           # already finished: nothing to do

        previous = {rec: (snapshot(rec, it_done) if it_done else None) for rec in rec_ids}
        it = it_done
        while True:
            it += 1
            label = None if it == 1 else it
            started_at = time.time()
            for group, recs in groups.items():
                training = self.create_training_data(rec_type=None, reclass_iter=label, rec_list=recs, columns=fields)
                classifier = self.classification_tables(training, fields, legacy=legacy)
                del training
                for rec in recs:
                    if (it, rec) in progress:
                        continue
                    self.classify_receiver(rec, fields, classifier=classifier, reclass_iter=label, threshold_ratio=threshold_ratio, legacy=legacy)
                    progress.add((it, rec))
                    with pd.HDFStore(self.db, mode='a') as store:
                        store.append('classification_progress', pd.DataFrame({'iter': [it], 'rec_id': [rec]}), format='table',
                                     data_columns=True, min_itemsize={'rec_id': 20})
            stable, new_rows = True, []
            for rec in rec_ids:
                now = snapshot(rec, it)
                row = {'iter': it, 'rec_id': rec, 'detections': len(now), 'true': int((now.test == 1).sum()), 'false': int((now.test == 0).sum()),
                       'flipped': np.nan, 'flip_share': np.nan, 'true_change': np.nan}
                before = previous[rec]
                if before is not None and len(now):
                    joined = now.merge(before, on=['freq_code', 'epoch'], suffixes=('', '_prev'))
                    row['flipped'] = int((joined.test != joined.test_prev).sum())
                    row['flip_share'] = row['flipped'] / max(len(joined), 1)
                    kept_before = int((before.test == 1).sum())
                    row['true_change'] = abs(row['true'] - kept_before) / max(kept_before, 1)
                    if not (row['flip_share'] < flip_tol and row['true_change'] < count_tol):
                        stable = False
                previous[rec] = now
                new_rows.append(row)
            rows.extend(new_rows)
            with pd.HDFStore(self.db, mode='a') as store:
                store.append('classification_rounds', pd.DataFrame(new_rows, columns=self.ROUNDS_COLUMNS).astype(
                    {'iter': 'int32', 'rec_id': 'object', 'detections': 'int64', 'true': 'int64', 'false': 'int64', 'flipped': 'float64',
                     'flip_share': 'float64', 'true_change': 'float64'}), format='table', data_columns=True, min_itemsize={'rec_id': 20})
            logger.info(f"classification round {it} finished in {(time.time() - started_at) / 60:.1f} min; stable={stable and it > 1}")
            if it > 1 and stable:
                break
            if it >= max_iter:
                moving = sorted({r['rec_id'] for r in new_rows if not (r['flip_share'] < flip_tol and r['true_change'] < count_tol)})
                warnings.warn(f"classify_until_stable stopped at max_iter={max_iter}; still changing: {moving}", RuntimeWarning)
                break
        return pd.DataFrame(rows, columns=self.ROUNDS_COLUMNS)

    def expected_false_share(self, rec_ids=None):
        """For each receiver and classification round: how many of the detections called true are probably noise.

        PyMAST learns the false class from codes that were never deployed, so noise that happens to decode to a deployed code is never seen
        as false. Valid codes at the study frequencies that were not deployed give an unbiased noise sample: the number of such detections,
        scaled by (deployed codes / undeployed valid codes), estimates how many deployed-code detections are noise. The valid code range at
        each frequency is taken as the smallest to largest code in the tag table (an assumption: receivers may accept more).

        Report only. The estimate does not depend on the round (its numerator is fixed), so as detections are shed the share can only
        rise: read it as 'is the noise estimate small or large at this receiver', and use `retained_cons_ge_k` / `expected_false_share_cons_ge_k`
        to see what a minimum run length would remove.

        Returns
        -------
        pandas.DataFrame
            Per receiver and round: kept, kept_before_release, expected_false, expected_false_share, and for k = 3 and 5 the detections with
            a run of at least k expected pulses and the expected false share among them.
        """
        tags = self.tags.reset_index()
        parts = tags.freq_code.str.split(' ', expand=True)
        tags['freq'], tags['code'] = parts[0], parts[1].astype(int)
        study = tags[tags.tag_type == 'study']
        span = tags.groupby('freq').code.agg(['min', 'max'])
        study_freqs = sorted(set(study.freq))
        n_valid = int(sum(span.loc[f, 'max'] - span.loc[f, 'min'] + 1 for f in study_freqs))
        n_unused = n_valid - tags[tags.freq.isin(study_freqs)].freq_code.nunique()
        if n_unused <= 0:
            raise ValueError("every valid code at the study frequencies is deployed, so there is no noise sample to scale from")
        ratio = study.freq_code.nunique() / n_unused
        with pd.HDFStore(self.db, mode='r') as store:
            trained = store.select('trained', columns=['freq_code', 'rec_id', 'detection', 'cons_length'])
            if rec_ids is None:
                rec_ids = sorted(store.select_column('trained', 'rec_id').unique())
        pt = trained.freq_code.str.split(' ', expand=True)
        trained['freq'], trained['code'] = pt[0], pt[1].astype(int)
        unused = trained[(trained.detection == 0) & trained.freq.isin(study_freqs)]
        unused = unused[[span.loc[f, 'min'] <= c <= span.loc[f, 'max'] for f, c in zip(unused.freq, unused.code)]]
        release = self.tags['rel_date'] if 'rel_date' in self.tags.columns else None
        rows = []
        for rec in rec_ids:
            with pd.HDFStore(self.db, mode='r') as store:
                kept_all = store.select('classified', where=f'rec_id == "{rec}" & test == 1', columns=['iter', 'freq_code', 'time_stamp', 'cons_length'])
            noise = unused[unused.rec_id == rec].cons_length.to_numpy()
            for it, kept in kept_all.groupby('iter'):
                row = {'iter': int(it), 'rec_id': rec, 'kept': len(kept), 'expected_false': ratio * len(noise),
                       'kept_before_release': (int((kept.time_stamp < kept.freq_code.map(release)).sum()) if release is not None else np.nan)}
                row['expected_false_share'] = row['expected_false'] / len(kept) if len(kept) else np.nan
                for k in (3, 5):
                    retained = int((kept.cons_length >= k).sum())
                    row[f'retained_cons_ge_{k}'] = retained
                    row[f'expected_false_share_cons_ge_{k}'] = ratio * float((noise >= k).sum()) / retained if retained else np.nan
                rows.append(row)
        return pd.DataFrame(rows)

    def classification_summary(self,rec_id,reclass_iter = None): 
        '''if this is not the initial classification we need the trues from the last 
        last classification and falses from the first'''
        
        iter_label = f"iteration {reclass_iter}" if reclass_iter else "initial classification"
        logger.info(f"Generating classification summary for {rec_id} ({iter_label})")
                
        if reclass_iter == None:
            classified_dat = pd.read_hdf(self.db,
                                         key = 'classified',
                                         where = f'(iter == 1) & (rec_id == "{rec_id}")')
        else:
            classified_dat = pd.read_hdf(self.db,
                                         key = 'classified',
                                         where = f'(iter == {reclass_iter}) & (rec_id == "{rec_id}")')
        
        logger.info(f"  Loaded {len(classified_dat)} classified detections")
                    
        logger.info("")
        logger.info(f"Classification Summary Report: {rec_id}")
        logger.info("="*80)
        det_class_count = classified_dat.groupby('test')['test'].count().to_frame()
        if len(det_class_count)>1:
            logger.info("")
            logger.info(f"{rec_id} detection class statistics:")
            prob_true = round(float(det_class_count.at[1,'test'])/float(det_class_count.sum()),3)
            prob_false = round(float(det_class_count.at[0,'test'])/float(det_class_count.sum()),3)
            logger.info(f"  P(classified as true) = {prob_true}")
            logger.info(f"  P(classified as false positive) = {prob_false}")
            logger.info("")
            logger.info("="*80)
            logger.info("")
            sta_class_count = classified_dat.groupby(['rec_id','test'])['test'].count().to_frame()#.reset_index(drop = False)
            recs = list(set(sta_class_count.index.levels[0]))
            logger.info("Detection Class Counts Across Stations")
            logger.info("          Classified     Classified")
            logger.info("             False          True")
            logger.info("       ______________________________")
            logger.info("      |              |              |")
            for i in recs:
                logger.info("%6s|   %8s   |   %8s   |"%(i,sta_class_count.loc[(i,0)].values[0],sta_class_count.loc[(i,1)].values[0]))
            logger.info("      |______________|______________|")
            logger.info("")
            logger.info("="*80)
            logger.info("Compiling classification figures...")

            # Plot the log likelihood ratio
            classified_dat['log_posterior_ratio'] = np.log10(classified_dat.posterior_T / classified_dat.posterior_F)
            
            # Binning and other parameters
            hit_ratio_bins = np.linspace(0, 1.0, 11)
            con_length_bins = np.arange(1, 12, 1)
            power_bins = np.arange(50, 110, 10)
            noise_bins = np.linspace(0, 1.1, 11)
            lag_bins = np.arange(-100, 110, 20)
            post_ratio_bins = np.linspace(classified_dat.log_posterior_ratio.min(), classified_dat.log_posterior_ratio.max(), 10)
            
            trues = classified_dat[classified_dat.test == 1]
            falses = classified_dat[classified_dat.test == 0]
            
            # Create a grid of subplots (3 rows x 4 columns)
            fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10), dpi=300)
            
            # Function to set font sizes
            def set_fontsize(ax, fontsize=6):
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(fontsize)
            
            # Plot hit ratio
            axes[0, 0].hist(falses.hit_ratio.values, hit_ratio_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[0, 0].set_xlabel('Hit Ratio')
            axes[0, 0].set_ylabel('Probability Density')
            axes[0, 0].set_title('Hit Ratio - False Positive')
            set_fontsize(axes[0, 0])
            
            axes[0, 1].hist(trues.hit_ratio.values, hit_ratio_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[0, 1].set_xlabel('Hit Ratio')
            axes[0, 1].set_title('Hit Ratio - Valid')
            set_fontsize(axes[0, 1])
            
            # Plot consecutive record length
            axes[0, 2].hist(falses.cons_length.values, con_length_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[0, 2].set_xlabel('Consecutive Hit Length')
            axes[0, 2].set_ylabel('Probability Density')
            axes[0, 2].set_title('Consecutive Hit Length - False Positive')
            set_fontsize(axes[0, 2])
            
            axes[0, 3].hist(trues.cons_length.values, con_length_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[0, 3].set_xlabel('Consecutive Hit Length')
            axes[0, 3].set_title('Consecutive Hit Length - Valid')
            set_fontsize(axes[0, 3])
            
            # Plot power
            axes[1, 0].hist(falses.power.values, power_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[1, 0].set_xlabel('Signal Power')
            axes[1, 0].set_ylabel('Probability Density')
            axes[1, 0].set_title('Signal Power - False Positive')
            set_fontsize(axes[1, 0])
            
            axes[1, 1].hist(trues.power.values, power_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[1, 1].set_xlabel('Signal Power')
            axes[1, 1].set_title('Signal Power - Valid')
            set_fontsize(axes[1, 1])
            
            # Plot noise ratio
            axes[1, 2].hist(falses.noise_ratio.values, noise_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[1, 2].set_xlabel('Noise Ratio')
            axes[1, 2].set_ylabel('Probability Density')
            axes[1, 2].set_title('Noise Ratio - False Positive')
            set_fontsize(axes[1, 2])
            
            axes[1, 3].hist(trues.noise_ratio.values, noise_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[1, 3].set_xlabel('Noise Ratio')
            axes[1, 3].set_title('Noise Ratio - Valid')
            set_fontsize(axes[1, 3])
            
            # Plot lag differences
            axes[2, 0].hist(falses.lag_diff.values, lag_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[2, 0].set_xlabel('Lag Differences')
            axes[2, 0].set_ylabel('Probability Density')
            axes[2, 0].set_title('Lag Differences - False Positive')
            set_fontsize(axes[2, 0])
            
            axes[2, 1].hist(trues.lag_diff.values, lag_bins, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[2, 1].set_xlabel('Lag Differences')
            axes[2, 1].set_title('Lag Differences - Valid')
            set_fontsize(axes[2, 1])

            # Plot log posterior ratio
            axes[2, 2].hist(falses.log_posterior_ratio.values, bins=20, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[2, 2].set_xlabel('Log Posterior Ratio')
            axes[2, 2].set_ylabel('Probability Density')
            axes[2, 2].set_title('Log Posterior Ratio - False Positive')
            set_fontsize(axes[2, 2])
            
            axes[2, 3].hist(trues.log_posterior_ratio.values, bins=20, density=True, color='grey', edgecolor='black', linewidth=1.2)
            axes[2, 3].set_xlabel('Log Posterior Ratio')
            axes[2, 3].set_title('Log Posterior Ratio - Valid')
            set_fontsize(axes[2, 3])
            
            # Adjust layout
            plt.tight_layout()
            
            # Show the plot
            plt.show()
        else:
           logger.warning("Insufficient data to quantify summary statistics")
           logger.warning(f"All remaining classified as {det_class_count.index[0]} - no more model improvement expected")

    def undo_classification(self, rec_id, freq_code=None, class_iter=None):
        # Read the table from the HDF5 file
        with pd.HDFStore(self.db, 'r+') as store:
            if 'classified' in store:
                df = store['classified']  # Replace with your table name
        
                # Build the condition based on provided arguments
                condition = (df['rec_id'] == rec_id)
                if freq_code is not None:
                    condition &= (df['freq_code'] == freq_code)
                if class_iter is not None:
                    condition &= (df['class_iter'] == class_iter)
        
                # Update or delete the rows based on your requirement
                # For deletion:
                df = df[~condition]
                
                # Delete the existing table
                store.remove('classified')
        
                # Save the modified DataFrame back to the HDF5 file
                store.put('classified',
                          df,
                          format='table',
                          index = False,
                          min_itemsize = {'freq_code':20,
                                          'rec_type':20,
                                          'rec_id':20,
                                          'det_hist':20},
                          data_columns=True,
                          append = False)       

    def undo_bouts(self, rec_id=None):
        """
        Remove bouts from the presence table.
        
        Args:
            rec_id (str, optional): Specific receiver ID to remove bouts for.
                                   If None, removes all bouts.
        """
        # Read the table from the HDF5 file
        with pd.HDFStore(self.db, 'r+') as store:
            if 'presence' in store:
                df = store['presence'] 
        
                # Build the condition based on provided arguments
                if rec_id is not None:
                    condition = (df['rec_id'] == rec_id)
                    df = df[~condition]
                else:
                    # Remove all presence data
                    df = pd.DataFrame(columns=df.columns)
                
                df = df.astype({'freq_code': 'object',
                                'epoch': 'int64',
                                'rec_id': 'object',
                                'class': 'object',
                                'bout_no':'int32',
                                'det_lag':'int32'})
                
                # Delete the existing table
                store.remove('presence')
        
                # Save the modified DataFrame back to the HDF5 file
                store.put('presence', 
                          df,
                          format='table', 
                          index = False,
                          min_itemsize = {'freq_code':20,
                                          'rec_id':20,
                                          'class':20},
                          data_columns=True, 
                          append = False)

    def repack_database(self, output_path=None):
        """
        Repack HDF5 database to reclaim disk space and improve performance.
        
        Uses PyTables to copy all nodes (Groups, Tables, Arrays) recursively
        with compression enabled. This fixes the bloat from repeated append operations.
        
        Args:
            output_path (str, optional): Path for repacked database.
                                        If None, uses '{db_name}_repacked.h5'
        
        Returns:
            str: Path to the repacked database
        """
        import tables
        import logging
        import time
        
        logger = logging.getLogger(__name__)
        
        if output_path is None:
            base_name = os.path.splitext(self.db)[0]
            output_path = f"{base_name}_repacked.h5"
        
        logger.info(f"Repacking database: {self.db} → {output_path}")
        print(f"[repack] Starting database repack...")
        print(f"  Source: {self.db}")
        print(f"  Target: {output_path}")
        
        # Get original size
        orig_size = os.path.getsize(self.db)
        print(f"  Original size: {orig_size / (1024**3):.2f} GB")
        
        start_time = time.time()
        
        # Open both files with PyTables
        with tables.open_file(self.db, mode='r') as h5in:
            with tables.open_file(output_path, mode='w') as h5out:
                
                # Set compression filters
                filters = tables.Filters(complevel=5, complib='blosc:zstd')
                
                # Copy all top-level nodes recursively
                for node in h5in.root:
                    node_path = node._v_pathname
                    print(f"  Copying {node_path}...")
                    
                    try:
                        # Use recursive=True to copy entire subtree (Groups, Tables, Arrays, etc.)
                        h5in.copy_node(
                            where=node_path,
                            newparent=h5out.root,
                            recursive=True,
                            filters=filters
                        )
                    except (tables.NodeError, tables.HDF5ExtError, OSError, ValueError) as e:
                        raise RuntimeError(f"Failed to copy HDF5 node {node_path}: {e}") from e
        
        # Get new size
        new_size = os.path.getsize(output_path)
        savings = (1 - new_size / orig_size) * 100
        elapsed = time.time() - start_time
        
        print(f"\n[repack] ✓ Repack complete in {elapsed:.1f} seconds")
        print(f"  New size: {new_size / (1024**3):.2f} GB")
        print(f"  Savings: {savings:.1f}%")
        
        logger.info(f"Repack complete: {new_size / (1024**3):.2f} GB ({savings:.1f}% reduction)")
        
        return output_path   

    def make_recaptures_table(self, export=True, pit_study=False):
        '''Creates a recaptures key in the HDF5 file, iterating over receivers to manage memory.'''
        # Import dask lazily; fallback to pandas when dask is not installed.
        try:
            import dask.dataframe as dd
            use_dask = True
        except ImportError:
            dd = None
            use_dask = False
            logger.warning("dask is not installed; using pandas fallback for recaptures table generation.")
        
        logger.info("Creating recaptures table")
        logger.info(f"  PIT study mode: {pit_study}")
        logger.info(f"  Processing {len(self.receivers)} receiver(s)")
        # prepare a heartbeat log so long runs can be monitored (one-line per receiver)
        heartbeat_dir = os.path.join(self.project_dir, 'build')
        try:
            os.makedirs(heartbeat_dir, exist_ok=True)
        except OSError as e:
            raise RuntimeError(
                f"Failed to create heartbeat directory '{heartbeat_dir}': {e}"
            ) from e
        heartbeat_path = os.path.join(heartbeat_dir, 'recaptures_heartbeat.log')
        print(f"Starting recaptures: {len(self.receivers)} receivers. Heartbeat -> {heartbeat_path}")
        try:
            with open(heartbeat_path, 'a') as _hb:
                _hb.write(f"START {datetime.datetime.now().isoformat()} receivers={len(self.receivers)}\n")
        except OSError as e:
            raise RuntimeError(
                f"Failed to write heartbeat start to '{heartbeat_path}': {e}"
            ) from e
        
        if not pit_study:
            # RADIO STUDY PATH
            logger.info("  Using RADIO study processing path")
            # Convert release dates to datetime if not already done
            self.tags['rel_date'] = pd.to_datetime(self.tags['rel_date'])
            tags_copy = self.tags.copy()
            
            for rec in tqdm(self.receivers.index, desc="Processing receivers", unit="receiver"):
                logger.info(f"  Processing receiver {rec}...")
                print(f"[recaptures] processing receiver {rec}...", flush=True)
    
                # Read classified data for this receiver
                if use_dask:
                    rec_dat = dd.read_hdf(self.db, key='classified')
                    rec_dat = rec_dat[rec_dat['rec_id'] == rec].compute()
                else:
                    rec_dat = pd.read_hdf(self.db, key='classified', where=f'rec_id == "{rec}"')   # filter while reading, not after
                
                # Convert 'timestamp' column to datetime
                rec_dat['time_stamp'] = pd.to_datetime(rec_dat['time_stamp'])
                
                # Calculate seconds since Unix epoch
                rec_dat['epoch'] = (rec_dat['time_stamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
                logger.debug(f"    Initial load: {len(rec_dat)} detections")
    
                # Merge with release dates to filter out data before release
                rec_dat = rec_dat.merge(tags_copy, left_on='freq_code', right_index=True)
                rec_dat = rec_dat[rec_dat['time_stamp'] >= rec_dat['rel_date']]
                logger.debug(f"    After release date filter: {len(rec_dat)} detections")
    
                # Reset index to avoid ambiguity between index and column labels
                if 'freq_code' in rec_dat.columns and 'freq_code' in rec_dat.index.names:
                    rec_dat = rec_dat.reset_index(drop=True)
    
                # Filter by latest iteration and valid test
                idxmax_values = rec_dat.iter.max()
                rec_dat = rec_dat[rec_dat.iter == idxmax_values]
                rec_dat = rec_dat[rec_dat['test'] == 1]
                logger.debug(f"    After filtering (iter={idxmax_values}, test=1): {len(rec_dat)} detections")

                # Check if 'presence' exists before trying to read it
                try:
                    if use_dask:
                        presence_data = dd.read_hdf(self.db, key='presence')
                        presence_data = presence_data[presence_data['rec_id'] == rec]
                    else:
                        presence_data = pd.read_hdf(self.db, key='presence', where=f'rec_id == "{rec}"')
                    presence_exists = True
                except (KeyError, FileNotFoundError, ValueError):
                    presence_exists = False
                    
                if presence_exists:
                    try:
                        if use_dask:
                            presence_data = presence_data.compute()
                        presence_data = presence_data[presence_data['freq_code'].isin(self.tags[self.tags.tag_type=='study'].index)]
                        presence_data = presence_data[['freq_code', 'epoch', 'rec_id', 'bout_no']]
                        logger.debug(f"    Presence data: {len(presence_data)} records")

                    except KeyError:
                        logger.warning(f"    No presence data found for {rec}, skipping presence merge")
                else:
                    logger.warning("    'presence' key not found in HDF5, skipping presence merge")                    
    
                # Read overlap data - filter immediately to avoid expensive len() compute
                try:
                    if use_dask:
                        overlap_data = dd.read_hdf(self.db, key='overlapping')
                        overlap_data = overlap_data[overlap_data['rec_id'] == rec]
                    else:
                        overlap_data = pd.read_hdf(self.db, key='overlapping', where=f'rec_id == "{rec}"')
                    overlap_exists = True
                except (KeyError, FileNotFoundError, ValueError):
                    overlap_exists = False
        
                if overlap_exists:
                    try:
                        if use_dask:
                            overlap_data = overlap_data.compute()
                        overlap_data = overlap_data[overlap_data['freq_code'].isin(self.tags[self.tags.tag_type=='study'].index)]
                        # Aggregate both overlapping and ambiguous_overlap columns
                        if 'ambiguous_overlap' in overlap_data.columns:
                            overlap_data = overlap_data.groupby(['freq_code', 'epoch', 'rec_id']).agg({
                                'overlapping': 'max',
                                'ambiguous_overlap': 'max'
                            }).reset_index()
                        else:
                            overlap_data = overlap_data.groupby(['freq_code', 'epoch', 'rec_id'])['overlapping'].max().reset_index()
                        logger.debug(f"    Overlap data: {len(overlap_data)} records")

                    except KeyError:
                        logger.warning(f"    No overlap data found for {rec}, skipping overlap merge")
                else:
                    logger.warning("    'overlapping' key not found in HDF5, skipping overlap merge")
    
                # Merge with presence data
                if presence_exists:
                    # Keep only detections that belong to a bout: bout numbers start at 0, so filling a missing
                    # bout with 0 would silently attach bout-less detections (for example those dropped by a
                    # minimum bout size) to a real bout.
                    rec_dat = rec_dat.merge(presence_data, on=['freq_code', 'epoch', 'rec_id'], how='inner')
                    rec_dat['bout_no'] = rec_dat['bout_no'].astype(int)
                else:
                    rec_dat['bout_no'] = 0
    
                # Merge with overlap data
                if overlap_exists:
                    rec_dat = rec_dat.merge(overlap_data, on=['freq_code', 'epoch', 'rec_id'], how='left')
                    rec_dat['overlapping'] = rec_dat['overlapping'].fillna(0).astype(int)
                    # Add ambiguous_overlap if it exists in overlap data
                    if 'ambiguous_overlap' in overlap_data.columns:
                        rec_dat['ambiguous_overlap'] = rec_dat['ambiguous_overlap'].fillna(0).astype('float32')
                    else:
                        rec_dat['ambiguous_overlap'] = np.float32(0)
                else:
                    rec_dat['overlapping'] = 0
                    rec_dat['ambiguous_overlap'] = np.float32(0)
                
                # Filter out overlapping detections (keep only overlapping=0)
                before_filter = len(rec_dat)
                rec_dat = rec_dat[rec_dat['overlapping'] != 1]
                after_filter = len(rec_dat)
                logger.debug(f"    Filtered {before_filter - after_filter} overlapping detections")
    
                logger.debug(f"    After presence/overlap merge: {len(rec_dat)} detections")
    
                # Check for required columns
                required_columns = ['freq_code', 'rec_id', 'epoch', 'time_stamp', 'power', 'noise_ratio',
                                    'lag', 'det_hist', 'hit_ratio', 'cons_det', 'cons_length', 
                                    'likelihood_T', 'likelihood_F', 'bout_no', 'overlapping', 'ambiguous_overlap']
                
                missing_columns = [col for col in required_columns if col not in rec_dat.columns]
                if missing_columns:
                    logger.error(f"    Required columns missing: {missing_columns}")
                    continue
    
                # Sort by freq code and epoch
                rec_dat = rec_dat.sort_values(by=['freq_code', 'epoch'], ascending=[True, True])
    
                # Keep only the necessary columns (including handling missing columns)
                available_columns = [col for col in required_columns if col in rec_dat.columns]
                rec_dat = rec_dat[available_columns]
    
                # Ensure correct data types
                rec_dat = rec_dat.astype({
                    'freq_code': 'object',
                    'epoch': 'int64',
                    'rec_id': 'object',
                    'time_stamp': 'datetime64[ns]',
                    'power': 'float32', 
                    'noise_ratio': 'float32',
                    'lag': 'float32',
                    'det_hist': 'object',
                    'hit_ratio': 'float32',
                    'cons_det': 'int32',
                    'cons_length': 'float32',
                    'likelihood_T': 'float32',
                    'likelihood_F': 'float32',
                    'bout_no': 'int32',
                    'overlapping': 'int32',
                    'ambiguous_overlap': 'float32'
                })
    
                # Show record counts
                logger.debug(f"    Final: {len(rec_dat)} detections for {rec}")
                print(f"[recaptures] {rec}: compiled {len(rec_dat)} rows (overlapping={rec_dat['overlapping'].sum()}, bouts={rec_dat['bout_no'].max()})", flush=True)
                
                # Append to the HDF5 file
                with pd.HDFStore(self.db, mode='a') as store:
                    store.append(key='recaptures', value=rec_dat, format='table', 
                                 index=False, min_itemsize={'freq_code': 20, 'rec_id': 20, 'det_hist': 20},
                                 append=True, chunksize=1000000, data_columns=True)

                logger.info(f"  ✓ Recaps for {rec} compiled and written to HDF5")
                print(f"[recaptures] ✓ {rec} written to database", flush=True)
                # append heartbeat line
                try:
                    with open(heartbeat_path, 'a') as _hb:
                        _hb.write(f"{datetime.datetime.now().isoformat()} rec={rec} rows={len(rec_dat)}\n")
                except OSError as e:
                    raise RuntimeError(
                        f"Failed to write heartbeat for receiver {rec} to '{heartbeat_path}': {e}"
                    ) from e
                
        else:
            # PIT STUDY PATH
            logger.info("  Using PIT study processing path")
            # Loop over each receiver in self.receivers
            for rec in tqdm(self.receivers.index, desc="Processing PIT receivers", unit="receiver"):
                logger.info(f"  Processing {rec} (PIT study)...")
        
                # Read PIT data (already parsed from your text files) from /raw_data in HDF5
                try:
                    pit_data = pd.read_hdf(self.db, key='raw_data')
                except KeyError:
                    logger.error("  No 'raw_data' key found in HDF5 file")
                    continue
        
                # Filter rows so that only the specified receiver is kept
                pit_data = pit_data[pit_data['rec_id'] == rec]
                logger.debug(f"    Filtered PIT data: {len(pit_data)} detections")
        
                # Add any missing columns to align with the acoustic (non-PIT) columns
                missing_cols = [
                    'lag', 'det_hist', 'hit_ratio', 'cons_det', 'cons_length',
                    'likelihood_T', 'likelihood_F', 'bout_no', 'overlapping', 'ambiguous_overlap'
                ]
                for col in missing_cols:
                    if col not in pit_data.columns:
                        if col == 'ambiguous_overlap':
                            pit_data[col] = np.float32(0)
                        else:
                            pit_data[col] = 0
        
                # Check if 'presence' exists before trying to read it
                with pd.HDFStore(self.db, mode='r') as store:
                    presence_exists = 'presence' in store.keys()
        
                if presence_exists:
                    try:
                        if use_dask:
                            presence_data = dd.read_hdf(self.db, key='presence')
                            presence_data = presence_data[presence_data['rec_id'] == rec].compute()
                        else:
                            presence_data = pd.read_hdf(self.db, key='presence')
                            presence_data = presence_data[presence_data['rec_id'] == rec]
                        presence_data = presence_data[presence_data['freq_code'].isin(self.tags[self.tags.tag_type=='study'].index)]
                        presence_data = presence_data[['freq_code', 'epoch', 'rec_id', 'bout_no']]
                        
                        if not presence_data.empty:
                            pit_data = pit_data.merge(presence_data, on=['freq_code', 'epoch', 'rec_id'], how='left')
                            pit_data['bout_no'] = pit_data['bout_no'].fillna(0).astype(int)
                    except KeyError:
                        logger.warning(f"    No presence data found for {rec}, skipping presence merge")
                else:
                    logger.warning("    'presence' key not found in HDF5, skipping presence merge")
        
                # Check if 'overlapping' exists before trying to read it
                with pd.HDFStore(self.db, mode='r') as store:
                    overlap_exists = 'overlapping' in store.keys()
        
                if overlap_exists:
                    try:
                        if use_dask:
                            overlap_data = dd.read_hdf(self.db, key='overlapping')
                            overlap_data = overlap_data[overlap_data['rec_id'] == rec].compute()
                        else:
                            overlap_data = pd.read_hdf(self.db, key='overlapping')
                            overlap_data = overlap_data[overlap_data['rec_id'] == rec]
                        overlap_data = overlap_data[overlap_data['freq_code'].isin(self.tags[self.tags.tag_type=='study'].index)]
                        # Aggregate: take max for both overlapping and ambiguous_overlap
                        agg_dict = {'overlapping': 'max'}
                        if 'ambiguous_overlap' in overlap_data.columns:
                            agg_dict['ambiguous_overlap'] = 'max'
                        overlap_data = overlap_data.groupby(['freq_code', 'epoch', 'rec_id']).agg(agg_dict).reset_index()

                        if not overlap_data.empty:
                            pit_data = pit_data.merge(overlap_data, on=['freq_code', 'epoch', 'rec_id'], how='left')
                            pit_data['overlapping'] = pit_data['overlapping'].fillna(0).astype(int)
                            if 'ambiguous_overlap' in overlap_data.columns:
                                pit_data['ambiguous_overlap'] = pit_data['ambiguous_overlap'].fillna(0).astype('float32')
                    except KeyError:
                        logger.warning(f"    No overlap data found for {rec}, skipping overlap merge")
                else:
                    logger.warning("    'overlapping' key not found in HDF5, skipping overlap merge")
        
                # Sort PIT data by freq_code and epoch
                pit_data = pit_data.sort_values(['freq_code', 'epoch'])
        
                # Keep only the columns needed in `recaptures`
                required_columns = [
                    'freq_code', 'rec_id', 'epoch', 'time_stamp', 'power', 'noise_ratio', 'lag', 'det_hist',
                    'hit_ratio', 'cons_det', 'cons_length', 'likelihood_T', 'likelihood_F', 'bout_no', 'overlapping', 'ambiguous_overlap'
                ]
                pit_data = pit_data[[c for c in required_columns if c in pit_data.columns]]
        
                # Convert each column to the correct dtype
                dtypes_map = {
                    'freq_code': 'object', 'rec_id': 'object', 'epoch': 'int64',
                    'time_stamp': 'datetime64[ns]', 'power': 'float32', 'noise_ratio': 'float32',
                    'lag': 'float32', 'det_hist': 'object', 'hit_ratio': 'float32',
                    'cons_det': 'int32', 'cons_length': 'float32', 'likelihood_T': 'float32',
                    'likelihood_F': 'float32', 'bout_no': 'int32', 'overlapping': 'int32', 'ambiguous_overlap': 'float32'
                }
                for col, dt in dtypes_map.items():
                    if col in pit_data.columns:
                        pit_data[col] = pit_data[col].astype(dt)
        
                # Show record counts
                print(f"[recaptures] {rec}: compiled {len(pit_data)} PIT rows (overlapping={pit_data['overlapping'].sum()}, bouts={pit_data['bout_no'].max()})", flush=True)
        
                # Convert 'det_hist' to string to avoid serialization issues
                if 'det_hist' in pit_data.columns:
                    pit_data['det_hist'] = pit_data['det_hist'].astype(str)        
        
                # Append PIT data to 'recaptures' in HDF5
                with pd.HDFStore(self.db, mode='a') as store:
                    store.append(
                        key='recaptures',
                        value=pit_data,
                        format='table',
                        index=False,
                        min_itemsize={'freq_code': 20, 'rec_id': 20, 'det_hist': 20},
                        append=True,
                        chunksize=1000000,
                        data_columns=True
                    )
        
                logger.info(f"  ✓ PIT recaps for {rec} compiled and written to HDF5")
                print(f"[recaptures] ✓ {rec} PIT data written to database", flush=True)
                try:
                    with open(heartbeat_path, 'a') as _hb:
                        _hb.write(f"{datetime.datetime.now().isoformat()} pit_rec={rec} rows={len(pit_data)}\n")
                except OSError as e:
                    raise RuntimeError(
                        f"Failed to write PIT heartbeat for receiver {rec} to '{heartbeat_path}': {e}"
                    ) from e


        if export:
            logger.info("Exporting recaptures to CSV...")
            print("[recaptures] exporting recaptures to CSV...", flush=True)
            if use_dask:
                rec_data = dd.read_hdf(self.db, 'recaptures').compute()
            else:
                rec_data = pd.read_hdf(self.db, key='recaptures')
            rec_data.to_csv(os.path.join(self.output_dir,'recaptures.csv'), index=False)
            logger.info(f"  ✓ Export complete: {os.path.join(self.output_dir,'recaptures.csv')}")
            print(f"[recaptures] ✓ Export complete: {os.path.join(self.output_dir,'recaptures.csv')}", flush=True)
            try:
                with open(heartbeat_path, 'a') as _hb:
                    _hb.write(
                        f"DONE {datetime.datetime.now().isoformat()} export="
                        f"{os.path.join(self.output_dir, 'recaptures.csv')}\n"
                    )
            except OSError as e:
                raise RuntimeError(
                    f"Failed to write heartbeat completion to '{heartbeat_path}': {e}"
                ) from e

                
    def undo_recaptures(self):
        """
        Remove recaptures data from HDF5 file.
        Note: File size won't shrink until you manually repack the database.
        """
        logger.info("Removing recaptures from database")
        with pd.HDFStore(self.db, mode='a') as store:
            if 'recaptures' in store:
                store.remove('recaptures')
                logger.info("  ✓ Recaptures key removed")
            else:
                logger.info("  No recaptures key found")
        
        logger.info("  Data logically deleted (file size unchanged)")
        logger.info("  To reclaim disk space, manually repack after all deletions complete")
                    
    def undo_overlap(self):
        """
        Remove overlapping data from HDF5 file.
        Note: File size won't shrink until you manually repack the database.
        """
        logger.info("Removing overlapping from database")
        with pd.HDFStore(self.db, mode='a') as store:
            if 'overlapping' in store:
                store.remove('overlapping')
                logger.info("  ✓ Overlapping key removed")
            else:
                logger.info("  No overlapping key found")
        
        logger.info("  Data logically deleted (file size unchanged)")
        logger.info("  To reclaim disk space, manually repack after all deletions complete")
                
    def new_db_version(self, output_h5):
        """
        Create a new version of the working HDF5 database.

        This function creates a copy of the existing working database, allowing you to 
        backtrack or branch your analysis. If there are keys that are in error or conflict 
        with the current understanding of the system, this function helps you remove them 
        from the new version of the database.

        Parameters:
        output_h5 (str): The file name for the new HDF5 file.
        """
        logger.info(f"Creating new database version: {output_h5}")

        # Copy the HDF5 file
        shutil.copyfile(self.db, output_h5)
        logger.info("  Database copied")

        # Open the copied HDF5 file
        with h5py.File(output_h5, 'r+') as hdf:
            # List all keys in the file
            keys = list(hdf.keys())
            logger.info(f"  Keys in HDF5 file: {', '.join(keys)}")

            # Ask the user to input the keys they want to modify
            selected_keys = str(self._prompt("Enter the keys you want to modify, separated by commas: ", default="")).split(',')

            # Clean up the input (remove whitespace)
            selected_keys = [key.strip() for key in selected_keys]

            for key in selected_keys:
                if key in hdf:
                    logger.info(f"  Processing key: '{key}'...")
                    
                    # If it's a group, recursively delete all datasets (subkeys)
                    if isinstance(hdf[key], h5py.Group):
                        logger.info(f"    Key '{key}' is a group, deleting all subkeys...")
                        for subkey in list(hdf[key].keys()):
                            logger.debug(f"      Removing subkey: '{key}/{subkey}'")
                            del hdf[key][subkey]
                        logger.info(f"    All subkeys under '{key}' deleted")
                    else:
                        # It's a dataset, clear the data in the DataFrame
                        logger.info(f"    Clearing data for dataset key: '{key}'")
                        df = pd.read_hdf(output_h5, key)
                        df.drop(df.index, inplace=True)
                        df.to_hdf(output_h5, key, mode='a', format='table', data_columns=True)
                        logger.info(f"    Data cleared for key: '{key}'")
                else:
                    logger.warning(f"  Key '{key}' not found in HDF5 file")

        # Update the project's database to the new copied database
        self.db = output_h5
        logger.info(f"✓ New database version created: {output_h5}")

            
                    
