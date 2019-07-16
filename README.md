# ABTAS

Aquatic Bio-Telemetry Analysis Software (ABTAS) for use in removing false positive and overlap detections from aquatic radio telemetry projects.

# Introduction
ABTAS, Kleinschmidt’s radio telemetry analysis software is comprised a suite of Python scripts and an importable python module (abtas.py).  Each of the scripts carries out a task one may have when analyzing radio telemetry data, including: creating a standardized project database, importing raw data directly from receiver downloads, identifying and removing false positive detections, cross validating and assessing the quality of training data, producing an array of standard project statistics (no. recaptured at receiver, etc.), removing overlap between neighboring receivers with larger detection zones, and producing data appropriate for analysis with Competing Risks and Mark Recapture (Cormack Jolly Seber and Live Recapture Dead Recover) methods.  In future iterations of the software, we will replace the functional sample scripts with a Jupyter Notebook to guide users through training, classification, and data preparation.

The scripts in their current form take advantage of the multiprocessing module when necessary to make the data filtering process more efficient.  When activated, all system resources will be utilized unless otherwise directed by the end user.  **It is not recommended to create more processes than CPUs in your computer, please understand the limitations of your machine before proceeding.**

Radio telemetry projects create vast quantities of data, especially those that employ in-water beacon tags.  Therefore, it is advised to have at least 10 GB of hard disk space to store raw, intermediate, and final data products, and at least 8 GB of RAM.  To handle this data, the software creates a SQLite project database.  SQLite is an in-process library that implements a self-contained, server-less, zero configuration, transactional SQL database engine (SQLite, 2017).  More importantly, SQLite can handle simultaneous reads, so it is well-suited for write-once, read-many largescale data analysis projects that employ parallel processing across multiple cores.  To view the SQLite project database download either: [sqlite browser](http://sqlitebrowser.org/) or [sqlite studio](https://sqlitestudio.pl/index.rvt) .  Both SQLite viewers can execute SQL statements, meaning you can create a query and export to csv file from within the database viewer.  They are not that different from Microsoft Access, users familiar with databases will have an easy transition into SQLite.

Starting in 2019, Kleinschmidt Associates (primary developer) will begin hosting the code on Atlassian Bitbucket.  With internal and external collaborators, version control has become an issue and the software is no longer standard among users.  Bitbucket is a web-based version control repository hosting service, which uses the Git system of version control.  It is the preferred system for developers collaborating on proprietary code, because users must be invited to contribute.  The Git distributed version control system tracks and manages changes in source code during development.  This is important when more than 1 user of the open source software adapts it to use on their project.  

The software is written in Python 3.7.x and uses dependencies outside of the standard packages.  Please make sure you have the following modules installed and updated when running telemetry analysis: Numpy, Pandas, Networkx, Matplotlib, Sqlite3, and Statsmodels.  The software also uses a number of standard packages including: Multiprocessing, Time, Math, Os, Datetime, Operator, Threading and Collections.  

This repository includes sample scripts that guide a user through a telemetry project.  However, you could import abtas into your own proprietary scripts and data management routines.  These scripts are examples only, if you push changes to the script, the owner may not commit them.

# Project Set Up
The simple 4-line script “project_setup.py” will create a standard file structure and project database in a directory of your choosing.  **It is recommended that the directory does not contain any spaces or special characters.**  For example, if our study was of fish migration in the Connecticut River our initial directory could appear as (if saved to your desktop):
> C:\Users\UserName\Desktop\Connecticut_River_Study

When a directory has been created, insert it into the script in line 2 (highlighted below).  Then, edit line 3 to name the database.  **It is recommended to avoid using spaces in the name of the database.**  Once lines 2 and 4 have been edited, run the script.  

projct_setup.py example:
```
import abtas
proj_dir = 'J:\1210\005\Calcs\Studies\3_3_19\2019'
dbName = 'ultrasound_2019.db'
abtas.createTrainDB(proj_dir, dbName)  # create project database
```

Before proceeding to the next step, investigate the folder structure that was created.  The folders should appear as:

* Project_Name
    * Data *for storing raw data and project set up files*
	    * Training Files *for storing raw receiver files the station you are currently working up*
	* Output *for storing figures, modeling files, and scratch data*
	    * Scratch *holds intermediate files that are managed by the software* - **never delete the contents of this folder**
		* Figures *holds figures exported by the software*

**Do not alter the structure of the directory.** Both the sample scripts provided and ABTAS expect that the directory is structured exactly as created.

# Initializing the Project Database

