from pymongo import MongoClient
import os
from datetime import datetime, timezone
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

requests_col = db["client_requests"]
decisions_col = db["ai_decisions"]

# -----------------------------
# Fetch last 10 requests
# -----------------------------
print("=== Last 10 Client Requests ===")
requests_cursor = requests_col.find().sort("timestamp", -1).limit(10)
for req in requests_cursor:
    ts = req.get("timestamp")
    if isinstance(ts, datetime):
        ts = ts.isoformat()
    print(f"Request ID: {req.get('request_id')}")
    print(f"Timestamp : {ts}")
    print(f"Request   : {req.get('request')}")
    print("-" * 40)

# -----------------------------
# Fetch last 10 AI decisions
# -----------------------------
print("\n=== Last 10 AI Decisions ===")
decisions_cursor = decisions_col.find().sort("timestamp", -1).limit(10)
for dec in decisions_cursor:
    ts = dec.get("timestamp")
    if isinstance(ts, datetime):
        ts = ts.isoformat()
    print(f"Request ID: {dec.get('request_id')}")
    print(f"Timestamp : {ts}")
    print(f"Decision  : {dec.get('decision')}")
    print(f"Status    : {dec.get('status')}")
    print("-" * 40)
