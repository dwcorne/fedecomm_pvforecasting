import sys
import pandas as pd
import matplotlib.pyplot as plt
import pvlib

import datetime

from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
from pvlib.iotools import read_tmy3

from timezonefinder import TimezoneFinder
from datetime import timedelta

import joblib # for loading the machine-learned model

#from pyhigh import get_elevation

#########  position of the solar array whose output we are predicting
lat = float(sys.argv[1])  
lon = float(sys.argv[2])
elevation = float(sys.argv[3])

# csv file with datetimes and cloudcover;  header is:  datetime,cloudcover
dtccfile = sys.argv[4]

trained_model = sys.argv[5] # the model we will use to do the predictions

#########--------------------------------------------------------------------------------------------

############################################################################
## figure out the timezone
############################################################################

tzobj =  TimezoneFinder()
tzstr = tzobj.timezone_at(lat=lat,lng=lon)

print(tzstr,elevation)
## make the pvlib location object
loc = pvlib.location.Location(lat, lon, tzstr, elevation, 'xyz')

###########################################################################
## various settings
############################################################################
sniff_sample = 20 ## how many rows of the input to look at to determine time interval

time_interval_mins = 60    ## these may well change
time_interval_secs = 3600
time_interval_freq = '60min'

def set_interval(interval_array):
    nonzero = 0
    for index in range(sniff_sample):
        if(interval_array[index]>0):
            nonzero = nonzero + 1
    if(nonzero==1):
        time_interval_secs = interval_array[0]
        time_interval_mins = time_interval_secs/60
        print("time_interval_mins set at ", time_interval_mins)
        if((time_interval_mins != 15) and (time_interval_mins != 30) and (time_interval_mins != 60)):
            print("but I can only handle intervals of 15, 30 or 60, sorry.")
            sys.exit(0)
        return time_interval_secs
    print("please fix the data file; it currently involves more  than one time interval between successive rows")
    for index in range(sniff_sample):
        if(interval_array[index]>0):
            print("interval example: ", interval_array[index]/60, " minutes")
    sys.exit(0)



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
## first pass through the data to detect the time interval
## just pick up a small sniff_sample of cases to see if there
## is a consistent interval
############################################################################

twodays = timedelta(days=2)

last_secs = -1

this_interval = -1
last_interval = -1
intervals = [0]*sniff_sample


for index, row in dtcc_data.iterrows():
    dtim = row['datetime']
    d = dtim[0:10]
    thedate = datetime.date.fromisoformat(d)
    nextdate = thedate + twodays

    hstr = dtim[11:19];
    secs = sum(x * int(t) for x, t in zip([3600, 60, 1], hstr.split(":")))
    hindex = int(secs/3600)
    if(index>=sniff_sample): # that's enough - now make a decision
        time_interval_secs = set_interval(intervals)
        break
    if(last_secs>0):
        this_interval = secs - last_secs
        for index2 in range(sniff_sample):
            if(intervals[index2]==0):
                intervals[index2] = this_interval
                break
            elif(intervals[index2]==last_interval):
                break
    last_interval = this_interval
    last_secs = secs


### print intervals as a check
for index2 in range(sniff_sample):
    print( "intervals", index2, intervals[index2])

### if we are here, then we have figured out time_interval_mins
### now set up other things related to time_interval

if(time_interval_secs==900):
    time_interval_mins = 15
    time_interval_freq = '15min'
    print("I set it to 15min")
if(time_interval_secs==1800):
    time_interval_mins = 30
    time_interval_freq = '30min'
    print("I set it to 30min")

print("time_interval_secs is ", time_interval_secs)
print("time_interval_freq is ", time_interval_freq)


############################################################################
## make the predictions
############################################################################


for index, row in dtcc_data.iterrows():
    dtim = row['datetime']
    d = dtim[0:10]

    thedate = datetime.date.fromisoformat(d)
    nextdate = thedate + twodays

    hstr = dtim[11:19];
    secs = sum(x * int(t) for x, t in zip([3600, 60, 1], hstr.split(":"))) 
    hindex = int(secs/3600)
    time_index = int(secs/time_interval_secs) # flexible time interval change
        
    times = pd.date_range(start=thedate.isoformat(),end=nextdate.isoformat(),freq=time_interval_freq,tz=tzstr)

    irrad = loc.get_clearsky(times)
    myrow = irrad.iloc[time_index]
    #print(myrow['ghi'],myrow['dni'],myrow['dhi'])

    new_ghi = myrow['ghi']
    new_dni = myrow['dni']
    new_dhi = myrow['dhi']        
    new_cloudcover = row['cloudcover']

    if(new_ghi > 0.001):
        #pred_row = {'GHI': new_ghi, 'DNI': new_dni, 'DHI': new_dhi, 'CLOUDCOVER': new_cloudcover}
        pred_row = [ new_ghi,  new_dni, new_dhi, new_cloudcover]        
        pred = loaded_model.predict([pred_row])
        print(dtim,pred[0])
    else:
        print(dtim,0)



############################################################################
## here we might some expts
############################################################################

sys.exit(1)




################# FILE FORMAT
###  DATETIME in ISO 8601 format, including timezone

###  CLOUDCOVER as a number from 0 (clearsky) to 100 (full cloudcover)
#######  (the source figures may have a different scale range, and/or there may be multiple indicators == e.g. high cloud and low cloud
########  .. the idea is that these source forecasts have been converted into a single value from 0 to 100 in some sensible way)



