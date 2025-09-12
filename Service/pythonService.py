from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from fastapi import FastAPI
import math
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Classes')))
from satellite import Satellite
from gs import Gs
from ss import Ss
from network import Network



EARTH_RADIUS = 6371000  # mét
app = FastAPI()

@app.get("/scan")
def scan(lat: float, lon: float, support5G : bool = False):
    network = Network()
    visible = network.find_connectable_nodes_for_location(lat, lon, support5G=support5G)
    return_data = []
    for node in visible:
        mode = ""
        if node.typename == "groundstation" or node.typename == "seastation":
            mode = "surface"
        else:
            mode = "3d"
        return_data.append({
            "type": node.typename,
            "id": node.id,
            "distance": node.calculate_distance(lat, lon, mode = mode)/1000,
            "priority": node.priority
        })

    return_data.sort(key=lambda x: (x["priority"], x["distance"]))
    return visible


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pythonService:app", host="0.0.0.0", port=8000, reload=True)