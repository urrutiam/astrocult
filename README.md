# AstroCult

Is a python package providing basic functionality for archaeoastronomy analysis. It can:
- perform statistical analysis in the form of curvigrams;
- compute the probability of a site to be located in a specific place;
- generate the location error heatmap for one or more horizon landmarks given an intended alignment;
- perform simple machine learning tasks such as clustering together similar declinations/azimuths, assess which features are most important from a dataset.

## Usage examples

Import the modules:

```
from astrocult import Azimuth, Declination, GeoUtils, JulianHelper
```

### Declination functions

Get a reference to our Declination object:

```
dec = Declination.Declination()
```

Compute all declinations for the year 2000 from a latitude of 45 N, longitude of 23 E and elevation of 100 meters:

``` 
df_dec = dec.calculate_declinations_for_entire_year(2000, 45, 23, 100)

print ("Declination on January 1: " , df_dec[0])
print ("Declination on December 31: " , df_dec[366]) # 2000 is a leap year.
```

Plot the declination curvigram for the latitude of 45 N for a list of declinations [23,22,23.5] with associated bin widths [0.1,0.2,0.3]:

```
dec.plot_decl_adaptive([23,22,23.5], 45, [0.1,0.2,0.3])
```

### Azimuth functions

While in practice we prefer declinations, there are cases when azimuths can still be used.

Get a reference to our Azimuth object:

```
az = Azimuth.Azimuth()
```

Plot the azimuth curvigram for the latitude of 45 N for a list of declinations [23,22,23.5] with associated bin widths [0.1,0.2,0.3]:

```
az.plot_az_adaptive([23,22,23.5], 45, [0.1,0.2,0.3])
```

### Error heatmaps and location probability

For a given area we might want to know what is the best place to place an observatory so that the astronomical event is aligned with a given horizon marker. At the same time we might want to calculate the probability of randomly picking a location in the aforementioned area. 
The code below shows an example of how this can be done. It also shows how the GeoUtils can be used to compute the altitude of the horizon from a certain point:

First we need the location of our observatory:

```
vatinLat = 45.210
vatinLon = 21.287
```

Second we need to set our target declinations, in this case the moon and sun at winter solstice:

```
sunDec = -24.1552
moonDec = -30.1094

targetAstroBodies = []
targetAstroBodies.append(['sun', sunDec])
targetAstroBodies.append(['moon', moonDec])
```

We also provide more information such the latitude, longitude, altitude in degrees and distance from the observatory in meters

```
# summer solstice point 6a
target1Lat = 45.141687
target1Lon = 21.425
target1Alt = 1.9
dist1FromVatin = 13380
    
# southern lunastice point 4a
target2Lat = 45.147294
target2Lon = 21.371814
target2Alt = 1.42
dist2FromVatin = 9678
```

Finally, initialize some classes:

```
gu = GeoUtils.GeoUtils()
dec = Declination.Declination()
```

Now it is time to actually start our analysis:

We compute the azimuth and declination given by the direction from the observatory to each of the 2 targets:

```
az1 = gu.get_bearing(vatinLat, target1Lat, vatinLon, target1Lon)
dec1 = dec.calc_decl(target1Lat, target1Alt, az1)
        
az2 = gu.get_bearing(vatinLat, target2Lat, vatinLon, target2Lon)
dec2 = dec.calc_decl(target2Lat, target2Alt, az2)
```

Optionally, we can compute the error between the real declinations and those provided by our observatory. The error is a quadratic sum as both need to happen simultaneously:

```
vatinError = math.sqrt((sunDec-dec1) * (sunDec-dec1) + (moonDec-dec2) * (moonDec-dec2))
```    

We also compute the 2 targets altitudes' in meters as well as the fistance:

```
target1HeightMeters = gu.horizon_height_meters(dist1FromVatin, target1Alt)
target2HeightMeters = gu.horizon_height_meters(dist2FromVatin, target2Alt)
```    

Now we can construct the heatmap by providing a geographical bounding box:

```       
latRange = list(range(44968, 45500, 1))
lonRange = list(range(20871, 21468, 1))
    
latRange = [x/1000 for x in latRange]
lonRange = [x/1000 for x in lonRange]
    
targetGeoLocations = []
targetGeoLocations.append([target1Lat, target1Lon, target1HeightMeters])
targetGeoLocations.append([target2Lat, target2Lon, target2HeightMeters])
   
coordsRange = [latRange, lonRange]
```

Finally we can compute the heatmap and probability of picking our observatory location by chance:
 
```    
df = gu.compute_error_heatmap(targetGeoLocations, targetAstroBodies, coordsRange)
print("Location probability", gu.compute_location_probability(df, 2))    
   
gu.show_heatmap(df, [vatinLat, vatinLon, vatinError], targetGeoLocations, [vatinLat, vatinLon], 11, 4)   
```

We can additionally plot the error heatmap historigram:

```
gu.plot_error_heatmap_histogram(df)
```

As well as exporting the heatmap as geojson files that we can later import in GIS viewers:

```
json_heatmap, json_targets = gu.build_geojson_from_heatmap(df, [[vatinLat, vatinLon, vatinError]], targetGeoLocations)
```
