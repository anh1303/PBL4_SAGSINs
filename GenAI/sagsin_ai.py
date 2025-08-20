from flask import Flask, request, jsonify
from pymongo import MongoClient
from datetime import datetime, timezone
import uuid
import os
from pymongo.server_api import ServerApi

# -------------------------
# Config
# -------------------------
uri = "mongodb+srv://longhoi856:UYcdtPdXsoYGFBrT@cluster0.hb5vpf7.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(uri, server_api=ServerApi('1'))
# MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:example@localhost:27017/?authSource=admin")
DB_NAME = "sagsins"

# client = MongoClient(uri, server_api='1')
db = client[DB_NAME]

# Collections
requests_col = db["client_requests"]
decisions_col = db["ai_decisions"]

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__)

# -------------------------
# Fake AI decision logic (replace with RL/GenAI later)
# -------------------------
def fake_inference(user_request: dict) -> dict:
    # Simple heuristic decision
    uid = user_request.get("user_id", "unknown")
    base = abs(hash(uid)) % 10
    sat = f"sat{(base % 3) + 1}"
    uav = f"uav{((base + 1) % 4) + 1}"
    ground = f"ground{((base + 2) % 2) + 1}"
    route = [sat, uav, ground]

    payload_size = user_request.get("payload_size", 0.0)
    latency = 10.0 + payload_size * 0.1 + len(route) * 2
    bandwidth = max(100.0, 1000.0 - payload_size * 0.5)

    return {
        "route": route,
        "expected_latency_ms": latency,
        "allocated_bandwidth_kbps": bandwidth,
        "metadata": {"heuristic": "flask_fake_v1"}
    }

# -------------------------
# Endpoints
# -------------------------

@app.route("/allocate", methods=["POST"])
def allocate():
    try:
        req_json = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    request_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc)

    # Save client request history
    request_doc = {
        "request_id": request_id,
        "timestamp": timestamp,
        "request": req_json
    }
    requests_col.insert_one(request_doc)

    # Generate decision
    decision = fake_inference(req_json)

    # Save decision history
    decision_doc = {
        "request_id": request_id,
        "timestamp": timestamp,
        "decision": decision,
        "status": "inferred"
    }
    decisions_col.insert_one(decision_doc)

    return jsonify({
        "request_id": request_id,
        "action": decision,
        "timestamp": timestamp.isoformat()
    })

@app.route("/history/requests", methods=["GET"])
def get_requests_history():
    limit = int(request.args.get("limit", 20))
    docs = list(requests_col.find().sort("timestamp", -1).limit(limit))
    for d in docs:
        d["_id"] = str(d["_id"])
        if isinstance(d["timestamp"], datetime):
            d["timestamp"] = d["timestamp"].isoformat()
    return jsonify(docs)

@app.route("/history/decisions", methods=["GET"])
def get_decisions_history():
    limit = int(request.args.get("limit", 20))
    docs = list(decisions_col.find().sort("timestamp", -1).limit(limit))
    for d in docs:
        d["_id"] = str(d["_id"])
        if isinstance(d["timestamp"], datetime):
            d["timestamp"] = d["timestamp"].isoformat()
    return jsonify(docs)

@app.route("/health", methods=["GET"])
def health():
    try:
        client.admin.command("ping")
        return jsonify({"status": "ok", "mongo": True})
    except Exception:
        return jsonify({"status": "ok", "mongo": False})

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
