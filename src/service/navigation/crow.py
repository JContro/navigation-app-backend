
from src.types.coordinates import WaypointRoute


def crow_polyline(coordinates: WaypointRoute):
    polyline = [
        [coordinates.start.latitude, coordinates.start.longitude],
        [coordinates.waypoint.latitude, coordinates.waypoint.longitude],
        [coordinates.end.latitude, coordinates.end.longitude]
    ]
    return polyline