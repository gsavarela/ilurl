# ILU (I)ntegrative (L)earning for (U)rban data

This project concerns with the sub-task of traffic simulation.  It leverages objects from [flow project](https://github.com/flow-project/flow) for building realistic simulations using [Sumo](https://www.dlr.de/ts/en/sumo/)

# Map Generation Guideline

The guidelines assume the installation steps for installing flow and sumo have already been taken. Otherwise the following [installation procedure](https://flow.readthedocs.io/en/latest/flow_setup.html#installing-flow-and-sumo) is recommended.

- Export from [Open Street Map](https://www.openstreetmap.org) by cropping a region of interest.
- Edit the OSM file using a [JOSM](https://josm.openstreetmap.de/wiki/Introduction) file editor for correctness.
- Convert the map file into a network file using [netconvert](http://sumo.sourceforge.net/userdoc/NETCONVERT.html).
- Generate routes by executing [randomTrips.py](https://sumo.dlr.de/docs/Tools/Trip.html).

While the following steps are optional:

- Import polygon types e.g buildings, water etc using [typemap.xml](http://sumo.sourceforge.net/userdoc/Networks/Import/OpenStreetMap.html).
- Generate polygons by running [polyconvert](http://sumo.sourceforge.net/userdoc/POLYCONVERT.html)
- Add the polymap to the sumo-gui configuration.

See [1](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7906642&isnumber=7904753)

## Example
This example assume the user has already performed the Export, Edition and the optional Import steps and has both a map.osm and a typemap.xml files saved on the current directory.

### Convert the map file into a network file using [netconvert](http://sumo.sourceforge.net/userdoc/NETCONVERT.html).
```
> netconvert --osm-files map.osm --output.street-names -o map.net.xml
```
### Generate routes by executing [randomTrips.py](https://sumo.dlr.de/docs/Tools/Trip.html).
This script generates a trips.xml which is consumed by duarouter application. By default it sets the begin time (option `-b`) to 0 seconds and end time (option `-e`) to 3600 seconds.

```
> python $SUMO_HOME/tools/randomTrips.py -n map.net.xml -r map.rou.xml -o map.trips.xml
```
### Generate polygons by running [polyconvert](http://sumo.sourceforge.net/userdoc/POLYCONVERT.html)
```
> polyconvert --net-file map.net.xml --osm-files map.osm --type-file typemap.xml -o map.poly.xml
```

### Add the polymap to the sumo-gui configuration.
```
<configuration>
     <input>
         <net-file value="map.net.xml"/>
	 <rou-file value="map.rou.xml"/>
         <additional-files value="map.poly.xml"/>
     </input>
</configuration>
```

# References
<sup>1</sup>L. Codeca, R. Frank, S. Faye and T. Engel, "Luxembourg SUMO Traffic (LuST) Scenario: Traffic Demand Evaluation" in IEEE Intelligent Transportation Systems Magazine, vol. 9, no. 2, pp. 52-63, Summer 2017. DOI: 10.1109/MITS.2017.2666585

