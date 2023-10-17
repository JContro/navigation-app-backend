from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time 

from typing import List
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    time.sleep(2)
    return [52.505, -0.09]


class Coordinate(BaseModel):
    start: List[float]
    waypoint: List[float]
    end: List[float]

@app.post("/crowflies")
def crowflies(coordinate: Coordinate):
    if len(coordinate.start) != 2 or len(coordinate.waypoint) != 2 or len(coordinate.end) != 2:
        raise HTTPException(status_code=400, detail="Each coordinate should have exactly two values")
    return [coordinate.start, coordinate.waypoint, coordinate.end]


