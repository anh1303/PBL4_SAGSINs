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

import json



EARTH_RADIUS = 6371000  # mét
app = FastAPI()

# uri = "mongodb+srv://longhoi856:UYcdtPdXsoYGFBrT@cluster0.hb5vpf7.mongodb.net/?retryWrites=true&w=majority"
# client = MongoClient(uri, server_api=ServerApi('1'))
# DB_NAME = "sagsins"
# db = client[DB_NAME]
# satellites = []
# ground_stations = []
# sea_stations = []

# #add satellite from json
# # with open("DB/satellites_24.json") as f:
# #     sat = json.load(f)
# # for s in sat:
# #     satellites.append(Satellite(s))


# collection = db["satellites"]
# for sat in collection.find():
#     satellites.append(Satellite(sat))
# collection = db["groundstations"]
# for gs in collection.find():
#     ground_stations.append(Gs(gs))
# collection = db["seastations"]
# for ss in collection.find():
#     sea_stations.append(Ss(ss))


    
# collection = db["satellites"]  # dùng để cập nhật vị trí vệ tinh

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

@app.get("/nodes")
def get_all_nodes():
    return {
        "satellites": [{"id": sat.id, "lat": sat.position["lat"], "lon": sat.position["lon"], "alt": sat.position["alt"]} for sat in satellites],
        "groundstations": [{"id": gs.id, "lat": gs.position["lat"], "lon": gs.position["lon"], "alt": gs.position["alt"]} for gs in ground_stations],
        "seastations": [{"id": ss.id, "lat": ss.position["lat"], "lon": ss.position["lon"], "alt": ss.position["alt"]} for ss in sea_stations],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pythonService:app", host="0.0.0.0", port=8000, reload=True)