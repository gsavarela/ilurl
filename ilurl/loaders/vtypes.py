import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString

# Vehicle definition stuff
from flow.controllers import GridRouter
from flow.core.params import SumoCarFollowingParams, VehicleParams

NETWORKS_PATH = f"{os.environ['ILURL_HOME']}/data/networks"
VTYPES_PATH = f"{NETWORKS_PATH}/vtypes.add.xml"

def get_vehicle_types():
    if not os.path.isfile(VTYPES_PATH):

        root = ET.Element(
            'routes',
            attrib={
                'xmlns:xsi':
                "http://www.w3.org/2001/XMLSchema-instance",
                'xsi:noNamespaceSchemaLocation':
                "http://sumo.dlr.de/xsd/routes_file.xsd"}
        )

        type_distribution = ET.SubElement(
            root,
            'vTypeDistribution',
            attrib={'id': 'private'}
        )
        vehicles = VehicleParams()
        vehicles.add(
            veh_id="human",
            routing_controller=(GridRouter, {}),
            car_following_params=SumoCarFollowingParams(
                min_gap=2.5,
                decel=7.5,  # avoid collisions at emergency stops
            ),
        )

        for veh in vehicles.types:
            veh['type_params'].update({
                'vClass': 'passager',
                'id': 'human'
            })
            ET.SubElement(
                type_distribution,
                'vType',
                attrib={
                    key: str(val)
                    for key, val in veh['type_params'].items()
                }
             )
    
        dom = parseString(ET.tostring(root))
        etree = ET.ElementTree(ET.fromstring(dom.toprettyxml()))
        etree.write(VTYPES_PATH)
    return VTYPES_PATH
                    
            
