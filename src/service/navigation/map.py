import osmnx as ox
import geopandas as gpd
import math

from src.types.coordinates import Coordinates
from shapely.geometry import Point

CENTRE_COORINDATES = [51.51297635567389, -0.117451976654785]

data = ox.graph_from_point(CENTRE_COORINDATES, dist=2000)

NODES, EDGES = ox.graph_to_gdfs(data)

merged_geometry = NODES['geometry'].unary_union
exterior_polygon = merged_geometry.convex_hull

convex_hull_gdf = gpd.GeoDataFrame(geometry=[exterior_polygon])

def get_map_boundaries():
    """Gets the map boundary as a list of lon-lat points that can plotted as a polygon 

    Returns:
        list(list(float, float))
    """
    
    p = list(convex_hull_gdf['geometry'][0].exterior.coords)
    inverted_list = [[b, a] for a, b in p]
    return inverted_list

def get_closest_node_from_point(coordinates: Coordinates):
    """Returns the closest node in our map to the 

    Args:
        coordinates (Coordinates): _description_
    """
    point = Point(coordinates.latitude, coordinates.longitude)
    idx = NODES.distance(point).idxmin()
    node = NODES.loc[idx]
    node = node.to_dict()
    node.pop('geometry')
    for key, value in node.items():
        if isinstance(value, float) and math.isnan(value):
            node[key] = None
    return node


