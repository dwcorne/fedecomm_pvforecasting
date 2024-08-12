##########################################################################
## settings
##########################################################################
MINCC = 0
MAXCC = 100
import sys
import pandas as pd
import matplotlib.pyplot as plt
import pvlib
import math

import datetime

from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
from pvlib.iotools import read_tmy3

from timezonefinder import TimezoneFinder
from datetime import timedelta

import joblib # for loading the machine-learned model

import random # for experiments with cloud cover errors
#from pyhigh import get_elevation

#########  position of the solar array whose output we are predicting
lat = float(sys.argv[1])  
lon = float(sys.argv[2])
elevation = float(sys.argv[3])

# csv file with datetimes and cloudcover;  header is:  datetime,cloudcover,output
dtccfile = sys.argv[4]

trained_model = sys.argv[5] # the model we will use to do the predictions

cc_error_range = float(sys.argv[6])

rand_seed = int(sys.argv[7])

random.seed(rand_seed)

#########--------------------------------------------------------------------------------------------

############################################################################
## figure out the timezone
############################################################################

tzobj =  TimezoneFinder()
tzstr = tzobj.timezone_at(lat=lat,lng=lon)

print(tzstr,elevation)
## make the pvlib location object
loc = pvlib.location.Location(lat, lon, tzstr, elevation, 'xyz')

############################################################################
## get the data
############################################################################

dtcc_data = pd.read_csv(dtccfile)

############################################################################
##  load the model
############################################################################
from os.path import abspath, join, dirname

loaded_model = joblib.load(join(dirname(abspath(__file__)), trained_model))

############################################################################
## make the predictions
############################################################################

#def update_ae(error, pred, actual)

twodays = timedelta(days=2)
mse = 0
ae = 0 
cases = 0
total_power = 0
actual_power = 0
ae5 = 0


def protected(val,minv,maxv):
    if(val<minv):
        return minv
    if(val>maxv):
        return maxv
    return val

def posval(val):
    if(val>=0):
        return val
    return -1*val



for index, row in dtcc_data.iterrows():
    dtim = row['datetime']
    d = dtim[0:10]

    thedate = datetime.date.fromisoformat(d)
    nextdate = thedate + twodays

    hstr = dtim[11:19];
    secs = sum(x * int(t) for x, t in zip([3600, 60, 1], hstr.split(":"))) 
    hindex = int(secs/3600)

    times = pd.date_range(start=thedate.isoformat(),end=nextdate.isoformat(),freq='60min',tz=tzstr)

    irrad = loc.get_clearsky(times)
    myrow = irrad.iloc[hindex]
    #print(myrow['ghi'],myrow['dni'],myrow['dhi'])

    new_ghi = myrow['ghi']
    new_dni = myrow['dni']
    new_dhi = myrow['dhi']        
    new_cloudcover = row['cloudcover']

    # model error in CC prediction

    err = random.uniform(-1*cc_error_range,cc_error_range)
    perturbed_cloudcover = protected(new_cloudcover + err,MINCC,MAXCC)
    
    pred_output = 0
    if(new_ghi > 0.001):
        #pred_row = {'GHI': new_ghi, 'DNI': new_dni, 'DHI': new_dhi, 'CLOUDCOVER': new_cloudcover}
        pred_row = [ new_ghi,  new_dni, new_dhi, perturbed_cloudcover]        
        pred = loaded_model.predict([pred_row])
        pred_output = pred[0]
        print(dtim,pred_output,row['output'])
    else:
        print(dtim,0, row['output'])
    err = pred_output - row['output']
    err2 = err*err
    mse = mse + err2
    ae = ae + posval(err) # absolute err, proportional to actual power 
    total_power += pred_output
    actual_power += row['output']
    cases = cases+1
    

mse /= cases
mae = ae/cases
print("rmse", math.sqrt(mse), "mae", mae, "total", total_power, "actual_power", actual_power, "pc_power_err", 100*(total_power - actual_power)/actual_power)

############################################################################
## here we might some expts
############################################################################




sys.exit(1)




################# FILE FORMAT
###  DATETIME in ISO 8601 format, including timezone

###  CLOUDCOVER as a number from 0 (clearsky) to 100 (full cloudcover)
#######  (the source figures may have a different scale range, and/or there may be multiple indicators == e.g. high cloud and low cloud
########  .. the idea is that these source forecasts have been converted into a single value from 0 to 100 in some sensible way)



