
from fastapi import APIRouter
from src.types.coordinates import Coordinates, WaypointRoute
from src.service.navigation.map import get_map_boundaries, get_closest_node_from_point

router = APIRouter(
    prefix="/navigation"
)

@router.post("/get-closest-node")
def get_closest_node(coordinates: Coordinates):
    node =  get_closest_node_from_point(coordinates)
    
    return node

@router.get("/map-limits")
def get_map_limits():
    return get_map_boundaries()

@router.post("/crow")
def crow(waypoint_route_coordinates: WaypointRoute):
    print(waypoint_route_coordinates)
    return {"succes": "is near"}