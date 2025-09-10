from pymongo import MongoClient
import json
from pymongo.server_api import ServerApi
# -----------------------------
# Config
# -----------------------------
uri = "mongodb+srv://longhoi856:UYcdtPdXsoYGFBrT@cluster0.hb5vpf7.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(uri, server_api=ServerApi('1'))
DB_NAME = "sagsins"
# -----------------------------
# Connect to MongoDB
# -----------------------------
db = client[DB_NAME]
# db = client["sagsins"]
# collection = db["satellites"]
# collection = db["groundstations"]
# collection = db["seastations"]
collection = db["uavs"]

# # Đọc file JSON
# with open("seastations.json") as f:
#     satellites = json.load(f)

with open("uavs.json") as f:
    uavs = json.load(f)

# # Import vào MongoDB
# collection.insert_many(satellites)
collection.insert_many(uavs)

# print("Đã import", collection.count_documents({}), "vệ tinh vào MongoDB")
# print("Đã import", collection.count_documents({}), "trạm mặt đất vào MongoDB")
# print("Đã import", collection.count_documents({}), "trạm biển vào MongoDB")
print("Đã import", collection.count_documents({}), "trạm UAV vào MongoDB")

# collection.update_many(
#     {},
#     {
#         "$set": {
#             "orbit_state.last_theta": 0.0
#         }
#     }
# )

# print("Cập nhật thành công")
