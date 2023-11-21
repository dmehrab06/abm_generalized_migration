############################1
############################1
import pandas as pd
import numpy as np
import sys
import random
import time
import warnings
from file_paths_and_consts import *
import math
import s2sphere

warnings.filterwarnings('ignore')

def getl13(lat,lng,req_level=13):
    p = s2sphere.LatLng.from_degrees(lat, lng) 
    cell = s2sphere.Cell.from_lat_lng(p)
    cellid = cell.id()
    for i in range(1,30):
        #print(cellid)
        if cellid.level()==req_level:
            return cellid
        cellid = cellid.parent()

PLACE_NAME = str(sys.argv[1])

cur_household_data = pd.read_csv(HOUSEHOLD_DIR+'burundi_household_data_ADM2_HDX.csv') ## change here for different country
cur_household_data = cur_household_data[cur_household_data.matching_place_id==PLACE_NAME]
cur_struct = 13
cur_household_data['s2_cell'] = cur_household_data.apply(lambda x: getl13(x['latitude'],x['longitude'],cur_struct),axis=1)
temp_household = cur_household_data[['hid','s2_cell']]
neighbor_household = temp_household.merge(temp_household,on='s2_cell',how='inner')
neighbor_household.to_csv(HOUSEHOLD_DIR+'burundi_neighbor_'+PLACE_NAME+'_'+str(cur_struct)+'_s2.csv',index=False) ##change here
