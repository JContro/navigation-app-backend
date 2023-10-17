
from fastapi import APIRouter
from src.types.coordinates import Coordinates

router = APIRouter(
    prefix="/navigation"
)

@router.post("/get-closest-node")
def get_closest_node(coordinates: Coordinates):
    return {"My Coordinates": coordinates}

