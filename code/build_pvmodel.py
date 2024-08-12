import sys
import pandas as pd
import matplotlib.pyplot as plt
import pvlib
import numpy as np
import datetime

from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
from pvlib.iotools import read_tmy3

from timezonefinder import TimezoneFinder
from datetime import timedelta

import joblib # for saving the machine-learned model for reuse

#from pyhigh import get_elevation

#########  position of the solar array whose output we want to predict
lat = float(sys.argv[1])
lon = float(sys.argv[2])
elevation = float(sys.argv[3])
#########--------------------------------------------------------------------------------------------

###  csv with latest historic data for the solar array, including cloud cover
### this file is of form DATETIME, CLOUDCOVER, GENERATION
### and we assume it is regularly constructed on the local system, pulling together
### local weather data with actual PV output (or PV input data)
### the header line must be: datetime,cloudcover,output

generation_csv = sys.argv[4]
gendata = pd.read_csv(generation_csv)

##########################################################################
##  this is the filename for the learned model
###########################################################################
model_file = sys.argv[5]

###########################################################################
## add the extraterrestrial radiation data to each row of the dataset
## making use of pvlib
############################################################################

tzobj =  TimezoneFinder()
tzstr = tzobj.timezone_at(lat=lat,lng=lon)

print(tzstr,elevation)

loc = pvlib.location.Location(lat, lon, tzstr, elevation, 'xyz')

traindf = pd.DataFrame(columns=['GHI', 'DNI', 'DHI', 'CLOUDCOVER', 'OUTPUT'])

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



twodays = timedelta(days=2)
rownum = 0

############################################################################
## first pass through the data to detect the time interval
## just pick up a small sniff_sample of cases to see if there
## is a consistent interval
############################################################################

last_secs = -1

this_interval = -1
last_interval = -1
intervals = [0]*sniff_sample   


for index, row in gendata.iterrows():
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

for index, row in gendata.iterrows():
    dtim = row['datetime']
    d = dtim[0:10]

    thedate = datetime.date.fromisoformat(d)
    nextdate = thedate + twodays

    hstr = dtim[11:19];
    secs = sum(x * int(t) for x, t in zip([3600, 60, 1], hstr.split(":"))) 
    hindex = int(secs/3600)
    time_index = int(secs/time_interval_secs) # flexible time interval change
    
    times = pd.date_range(start=thedate.isoformat(),end=nextdate.isoformat(),freq=time_interval_freq,tz=tzstr)
    #print(times)
    #print (times[0])
    #print (times[1])
    #print (times[2])
    #print (times[3])
    #print (times[4])    
    #sys.exit(0)
    #ephem = loc.get_solarposition(times)
    irrad = loc.get_clearsky(times)
    print("Now showing irrad")
    print (irrad)
    for ti in range(0,50):
        print (irrad.iloc[ti])
    
    
    myrow = irrad.iloc[time_index] # was hindex
#    print(myrow['ghi'],myrow['dni'],myrow['dhi'])
#    print("done----------------------------")
    #print(row['datetime'],row['ETR'])
    new_ghi = myrow['ghi']
    new_dni = myrow['dni']
    new_dhi = myrow['dhi']        
    new_cloudcover = row['cloudcover']
    new_output = row['output']    
    if(new_ghi > 0.001):
        new_row = {'GHI': new_ghi, 'DNI': new_dni, 'DHI': new_dhi, 'CLOUDCOVER': new_cloudcover, 'OUTPUT':new_output}
        traindf = traindf._append(new_row, ignore_index=True)

print("here is traindf")
print(traindf)



############################################################################
## here we treat the data for ML
############################################################################


y = traindf['OUTPUT'].values
X = traindf.drop(columns=['OUTPUT']).values

print("X here")
print(X)
print("Y here")
print(y)

############################################################################
## here we do the ML
############################################################################

from sklearn.tree import DecisionTreeRegressor 
  
# create a regressor object
regressor = DecisionTreeRegressor(random_state = 0) 
  
# fit the regressor with X and Y data
print("starting regression ....")

regressor.fit(X, y)

from sklearn.metrics import mean_squared_error

y_pred = regressor.predict(X)

mse = mean_squared_error(y,y_pred)
print("predictions")
print(y_pred)
print("mse is ", mse)

# save the model
joblib.dump(regressor, model_file)


############################################################################
## here we might some expts
############################################################################

sys.exit(1)




################# FILE FORMAT
###  DATETIME in ISO 8601 format, including timezone

###  CLOUDCOVER as a number from 0 (clearsky) to 10 (full cloudcover)
#######  (the source figures may have a different scale range, and/or there may be multiple indicators == e.g. high cloud and low cloud
########  .. the idea is that these source forecasts have been converted into a single value from 0 to 10 in some sensible way)

###  GENERATION:  the generation figure that you want to predict.   This might just be IRRADIANCE level reaching the panel,
#########   This might just be IRRADIANCE level reaching the PV, or it could be the output from the PV

