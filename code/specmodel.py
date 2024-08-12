import pvlib
import pandas as pd
import sys

from pvlib.modelchain import ModelChain
from pvlib.location import Location
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS


#tilt.py 50.0 0.0 2023-05-01 2023-05-02 bob 12 15 1 0.2


lat = float(sys.argv[1])
lon = float(sys.argv[2])
tz = sys.argv[3] # a TZ identifier from https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
name = sys.argv[4]
tilt = float(sys.argv[5])
azimuth = float(sys.argv[6])
panels = int(sys.argv[7])
#peakpower = float(sys.argv[10])
array_model = sys.argv[8]
inverter_model = sys.argv[9]
datetime_start = sys.argv[10]
datetime_end = sys.argv[11]
freq = sys.argv[12]

location = Location(latitude=lat,longitude=lon,tz=tz,altitude=0,name=name)

#cec_modules = pvlib.pvsystem.retrieve_sam('CECMod')
cec_modules = pvlib.pvsystem.retrieve_sam(path='data/CECModules.csv')
print(cec_modules)

cec_inverters = pvlib.pvsystem.retrieve_sam(path='data/CECInverters.csv')
print(cec_inverters)

module = cec_modules[array_model]
#module = cec_modules['Trina_Solar_TSM__400DE09']
inverter = cec_inverters[inverter_model]

temperature_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
print(module)
print(inverter)


system = PVSystem(surface_tilt=tilt,surface_azimuth=azimuth,module_parameters=module,inverter_parameters=inverter,temperature_model_parameters=temperature_parameters)

modelchain = ModelChain(system,location,aoi_model="no_loss",spectral_model="no_loss")

#times = pd.date_range(start=datetime_start,end=datetime_end,freq=freq,tz=location.tz)
#clear_sky = location.get_clearsky(times)

print("clearsky figures")
#print (clear_sky)


#modelchain.run_model(clear_sky)
print("modelchain results")
#print(modelchain.results.ac)

## ac gives Watts !!!!

print("now getting data from file")

extdata = pd.read_csv('data/tmy_10m_met_capriasca_cropped.csv',index_col=0)
extdata.index = pd.date_range(start=datetime_start,end=datetime_end,freq=freq,tz=location.tz)

modelchain.run_model(extdata)
print("new modelchain results")
print(panels * modelchain.results.ac)




"""
def obtain_panel_power_data(lat, lon, start, end, name, tilt, azimuth, panels, peakpower):
     ###
    Determine the solarradiation and generated power for a given solar panel configuration.
    For each hour between startdate and enddate the data is retrieved and calculated.
    :param lat: Latitude of the location
    :param lon: Longitude of the location
    :param start: Startdate for data retrievel
    :param end: Endddate for data retrievel
    :param name: Name of the panel location on the object
    :param tilt: Tilt of the solar panels (0 is flat, 90 is standing straight)
    :param azimuth: 
             Orientation (azimuth angle) of the (fixed) plane. Clockwise from north (north=0, east=90, south=180, west=270). 
             This is offset 180 degrees from the convention used by PVGIS. Ignored for tracking systems.
             Direction the panels, 0 is South, negative from south to east, positive from south to west
    :param panels: Number of panels on the location
    :param peakpower: Peakpower per panel
    :return:
    ### 
    if panels > 0:
        poa, _, _ = pvlib.iotools.get_pvgis_hourly(
            latitude=lat, longitude=lon, start=start, end=end,
            surface_tilt=tilt, surface_azimuth=azimuth,
            pvcalculation=True, peakpower=peakpower*panels,
            components=True, raddatabase='PVGIS-SARAH2', url='https://re.jrc.ec.europa.eu/api/v5_2/',
#            components=True, raddatabase='PVGIS-SARAH2', url='https://re.jrc.ec.europa.eu/api/',
        )
    else:
        poa, _, _ = pvlib.iotools.get_pvgis_hourly(
            latitude=lat, longitude=lon, start=start, end=end,
            surface_tilt=tilt, surface_azimuth=-azimuth,
            pvcalculation=False,
            components=True, raddatabase='PVGIS-SARAH2', url='https://re.jrc.ec.europa.eu/api/v5_2/',
            )
        poa['P'] = 0.0
    poa['date'] = pd.to_datetime(poa.index.date)
    poa['location'] = name
    poa['P'] = poa['P'].div(1000)    # change unit to kWh
    return poa

poa = obtain_panel_power_data(lat, lon, start, end, name, tilt, azimuth, panels, peakpower)
print (poa.to_string())

#print("------------------------")
#poa2 = pvlib.iotools.get_pvgis_hourly(latitude=45.51524, longitude=12.148694, surface_tilt=7, surface_azimuth=17, outputformat='csv', map_variables=True)
#print (poa2)

"""
