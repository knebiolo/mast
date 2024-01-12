# import modules
import sys
sys.path.append(r'C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\biotas')
import biotas as biotas
import os
import sqlite3

# declare workspaces
proj_dir = r'J:\2819\005\Calcs\Working_Nuyakuk2023Telem'                                      # what is the project directory?
dbName = '2023Nuyakuk.db'                                                         # whad did you call the database?
                                                    # what is the name of the project database
outputWS = os.path.join(proj_dir,'Output')
projectDB = os.path.join(proj_dir,'Data',dbName)

rec_list = ['R00','R01','R02','R03','R04','R10','R11','R12','R13','R14','R15','R16']

# the big merge
recaptures = biotas.the_big_merge(outputWS,projectDB, pre_release_Filter = True, rec_list = rec_list)

recaptures.to_sql('tblRecaptures',sqlite3.connect(projectDB, timeout=30.0),if_exists ='append',chunksize = 1000)

recaptures.to_csv(os.path.join(outputWS,'recaptures.csv'))

rec_by_fish = recaptures.groupby(['FreqCode'])['recID'].unique()

rec_by_fish.to_csv(os.path.join(outputWS,'rec_by_fish.csv'))

