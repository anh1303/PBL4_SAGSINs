from flask import Flask, request, jsonify
import random

app = Flask(__name__)

# Demo AI: chọn ngẫu nhiên một vệ tinh
SATELLITES = ["SAT-1", "SAT-2", "SAT-3"]

@app.route("/allocate", methods=["POST"])
def allocate():
    data = request.json
    user_id = data.get("user_id", "unknown")
    position = data.get("position", [0,0])

    # AI logic: chọn vệ tinh "tối ưu" (ở đây random)
    chosen_sat = random.choice(SATELLITES)

    result = {
        "user_id": user_id,
        "position": position,
        "assigned_satellite": chosen_sat
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(port=5000)
