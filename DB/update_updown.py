from pymongo import MongoClient
from pymongo.server_api import ServerApi

# -----------------------------
# Config
# -----------------------------
uri = "mongodb+srv://longhoi856:UYcdtPdXsoYGFBrT@cluster0.hb5vpf7.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1'))
DB_NAME = "sagsins"
db = client[DB_NAME]

# -----------------------------
# Fixed pools (Mbps)
# -----------------------------
FIXED_POOLS = {
    "groundstations": {"uplink": 5000, "downlink": 10000, "cpu": 5000, "power": 8000},
    "seastations": {"uplink": 2000, "downlink": 4000, "isl": 1000},
}

FIXED_POOL2 = {
    "LEO": {"uplink": 1000, "downlink": 2000, "isl": 2000},
    "GEO": {"uplink": 5000, "downlink": 10000, "isl": 2000}
}

# -----------------------------
# Generic migration (GS + SS)
# -----------------------------
def migrate_collection(name, pools):
    collection = db[name]
    docs = collection.find({"resources": {"$exists": True}})
    count = 0

    for doc in docs:
        update_fields = {
            "resources.uplink": pools["uplink"],
            "resources.downlink": pools["downlink"],
        }
        if "cpu" in pools:  # groundstations only
            update_fields["resources.cpu"] = pools["cpu"]
        if "power" in pools:  # groundstations only
            update_fields["resources.power"] = pools["power"]
        if "isl" in pools:  # seastations only
            update_fields["resources.isl"] = pools["isl"]

        collection.update_one(
            {"_id": doc["_id"]},
            {"$set": update_fields}
        )
        count += 1

    print(f"[OK] Migration completed for {name}. Updated {count} documents.")

# -----------------------------
# Satellite migration (LEO/GEO)
# -----------------------------
def migrate_satellites():
    collection = db["satellites"]
    docs = collection.find({"resources": {"$exists": True}})
    count = 0

    for doc in docs:
        sat_type = doc.get("type", "LEO").upper()
        pools = FIXED_POOL2.get(sat_type, FIXED_POOL2["LEO"])

        update_fields = {
            "resources.uplink": pools["uplink"],
            "resources.downlink": pools["downlink"],
            "resources.isl": pools["isl"],
        }

        collection.update_one(
            {"_id": doc["_id"]},
            {"$set": update_fields}
        )
        count += 1

    print(f"[OK] Satellite migration done. Updated {count} docs.")
    
def migrate_satellites_flat():
    collection = db["satellites"]
    docs = collection.find({"resources": {"$exists": True}})
    count = 0

    for doc in docs:
        sat_type = doc.get("type", "LEO").upper()
        pools = FIXED_POOL2.get(sat_type, FIXED_POOL2["LEO"])

        updated_resources = {}

        # Convert existing resource fields
        for key, value in doc["resources"].items():
            if isinstance(value, dict) and "max" in value:
                updated_resources[key] = value["max"]
            else:
                updated_resources[key] = value

        # Set uplink/downlink/isl as plain numbers
        updated_resources["uplink"] = pools["uplink"]
        updated_resources["downlink"] = pools["downlink"]
        updated_resources["isl"] = pools["isl"]

        # Update the document
        collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"resources": updated_resources}}
        )
        count += 1

    print(f"[OK] Satellite migration done. Updated {count} documents.")

# -----------------------------
# Run all migrations
# -----------------------------
# migrate_satellites()
for cname, pools in FIXED_POOLS.items():
    migrate_collection(cname, pools)
# migrate_satellites_flat()
