# Intersection Scenario

![alt text](figs/intersection.png)

## Recipe

The image was a selected zone in the city of Lisbon. The intersection choosen is from Rua Luciano Cordeiro and Rua Conde do Redondo and intersection.raw.osm was generated.

### 1. Edit intersection.raw.osm using [JOSM editor](https://josm.openstreetmap.de/).

- Remove nodes.
- Restrict edge lanes to the area of interest.
- Remove traffic lights.
- Add a traffic light to the crossing.
- Save file as intersection.osm.

### 2. Transform intersection.osm using [netconvert](http://sumo.sourceforge.net/userdoc/NETCONVERT.html).

- Save file as intersection.raw.net.

### 3. Edit intersection.raw.net using [netedit](https://sumo.dlr.de/docs/NETEDIT.html).

- Remove U turns using connections editor.
- Save file as intersection.net.xml

