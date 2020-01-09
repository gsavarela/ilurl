# Grid Scenario

![alt text](figs/grid.png)

## Recipe

The image was a selected zone in the city of Lisbon. The grid.raw.osm was generated from the intersections between Rua Conde do Redondo and:

- Rua Luciano Cordeiro
- Rua Bernardo Lima
- Rua Ferreira Lapa

### 1. Edit grid.raw.osm using [JOSM editor](https://josm.openstreetmap.de/).

- Remove nodes.
- Restrict edge lanes to the area of interest.
- Remove traffic lights.
- Add a traffic light to the crossing.
- Save file as grid.osm.

### 2. Transform grid.osm using [netconvert](http://sumo.sourceforge.net/userdoc/NETCONVERT.html).

- Save file as grid.raw.net.

### 3. Edit grid.raw.net using [netedit](https://sumo.dlr.de/docs/NETEDIT.html).

- Remove U turns using connections editor.
- Save file as grid.net.xml

