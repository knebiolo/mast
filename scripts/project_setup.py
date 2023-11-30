import sys
sys.path.append(r'C:\Users\knebiolo\OneDrive - Kleinschmidt Associates, Inc\Software\biotas')
import biotas
proj_dir = r'C:\Users\knebiolo\Desktop\York Haven BIOTIS'
dbName = 'YH_1200_test.db'
biotas.createTrainDB(proj_dir, dbName)  # create project database
