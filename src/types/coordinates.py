from pydantic import BaseModel

class Coordinates(BaseModel):
    longitude: float
    latitude: float 

class SimpleRoute(BaseModel):
    start: Coordinates
    end: Coordinates

class WaypointRoute(BaseModel):
    start: Coordinates
    waypoint: Coordinates
    end: Coordinates


