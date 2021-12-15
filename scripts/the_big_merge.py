# import modules
import biotas.biotas as biotas
import os
import sqlite3

# declare workspaces
proj_dir = r'D:\Manuscript\CT_River_2015'                                      # what is the project directory?
dbName = 'ctr_2015_v2.db'                                                         # whad did you call the database?
                                                    # what is the name of the project database
outputWS = os.path.join(proj_dir,'Output')
projectDB = os.path.join(proj_dir,'Data',dbName)

rec_list = ['T26','T27']
            #'T02','T03','T05',
            #'T06','T07','T11','T12E','T12W',
            #'T08','T09',
            #'T23','T24','T25'
            #'T13','T14','T15','T16','T17',
            #'T19','T20','T30','T33'
            #'T18','T20','T21'
            #'T26','T27'

# the big merge
recaptures = biotas.the_big_merge(outputWS,projectDB, pre_release_Filter = True, rec_list = rec_list)


recaptures.to_sql('tblRecaptures',sqlite3.connect(projectDB, timeout=30.0),if_exists ='append',chunksize = 1000)

recaptures.to_csv(os.path.join(outputWS,'recaptures.csv'))

rec_by_fish = recaptures.groupby(['FreqCode'])['recID'].unique()

rec_by_fish.to_csv(os.path.join(outputWS,'rec_by_fish.csv'))

