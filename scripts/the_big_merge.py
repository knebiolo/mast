# import modules
import biotas
import os
import sqlite3

# declare workspaces
proj_dir = r'E:\Manuscript\CT_River_2015'                                      # what is the project directory?
dbName = 'ctr_2015_v2.db'                                                    # what is the name of the project database
outputWS = os.path.join(proj_dir,'Output')
projectDB = os.path.join(proj_dir,'Data',dbName)

rec_list = ['T13','T18','T21','T22']

# the big merge
recaptures = biotas.the_big_merge(outputWS,projectDB, pre_release_Filter = True, rec_list = rec_list)

recaptures.to_csv(os.path.join(outputWS,'recaptures.csv'))
recaptures.to_sql('tblRecaputres',sqlite3.connect(projectDB),if_exists ='append')

rec_by_fish = recaptures.groupby(['FreqCode'])['recID'].unique()

rec_by_fish.to_csv(os.path.join(outputWS,'rec_by_fish.csv'))

