from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time 

from typing import List
from pydantic import BaseModel
from src.route.navigation import router as navigation_router

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


app.include_router(
    navigation_router
)