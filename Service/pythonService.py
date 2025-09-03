from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from fastapi import FastAPI
import math


EARTH_RADIUS = 6371000  # mét
app = FastAPI()

uri = "mongodb+srv://longhoi856:UYcdtPdXsoYGFBrT@cluster0.hb5vpf7.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1'))
DB_NAME = "sagsins"
db = client[DB_NAME]
satellites = []
ground_stations = []
sea_stations = []

collection = db["satellites"]
for sat in collection.find():
    satellites.append(sat)
collection = db["groundstations"]
for gs in collection.find():
    ground_stations.append(gs)
collection = db["seastations"]
for ss in collection.find():
    sea_stations.append(ss)

collection = db["satellites"]  # dùng để cập nhật vị trí vệ tinh

@app.get("/scan")
def scan(lat: float, lon: float, support5G : bool = False):
    visible = []
    visible_gs = []
    visible_ss = []
    visible_sat = []
    for gs in ground_stations:
        dist = calculate_distance(lat, lon, gs["location"]["lat"], gs["location"]["lon"])/1000  # km
        if dist <= gs["coverage_radius_km"]:
            visible_gs.append({
                "type": "groundstation",
                "id": gs["gs_id"],
                "distance": dist
            })
    for ss in sea_stations:
        dist = calculate_distance(lat, lon, ss["location"]["lat"], ss["location"]["lon"])/1000  # km
        if dist <= ss["coverage_radius_km"]:
            visible_ss.append({
                "type": "seastation",
                "id": ss["ss_id"],
                "distance": dist
            })
    if support5G:
        for sat in satellites:
            sat = update_satellite_position_obj_db(sat,db_collection= collection, target_time=datetime.now(timezone.utc))
            dist = distance_3d(lat, lon, 0, sat["position"]["lat"], sat["position"]["lon"], sat["position"]["alt"])/1000
            if can_connect_sat(sat, lat, lon):
                visible_sat.append({
                    "type": "satellite",
                    "id": sat["satellite_id"],
                    "distance": dist
                })
    visible_gs.sort(key=lambda x: x["distance"])
    visible_ss.sort(key=lambda x: x["distance"])
    visible_sat.sort(key=lambda x: x["distance"])
    visible.extend(visible_gs)
    visible.extend(visible_ss)
    visible.extend(visible_sat)
    return visible





def update_satellite_position_obj_db(sat_data, db_collection=None, target_time: datetime = None, min_update_interval: float = 1.0):
    """
    Cập nhật vị trí vệ tinh từ object, có thể update trực tiếp vào MongoDB nếu truyền collection.

    :param sat_data: dict vệ tinh
    :param db_collection: pymongo collection (tùy chọn)
    :param target_time: thời điểm tính vị trí mới, mặc định giờ hiện tại
    :param min_update_interval: ngưỡng Δt (giây) để bỏ qua nếu quá nhỏ
    :return: sat_data mới (đã cập nhật position, last_update, last_theta)
    """

    if sat_data["type"] == "GEO":
        return sat_data  # GEO không cần cập nhật
    
    if target_time is None:
        target_time = datetime.now(timezone.utc)

    last_update = datetime.fromisoformat(sat_data["last_update"].replace("Z", "+00:00"))
    dt = (target_time - last_update).total_seconds()

    # --- Nếu Δt quá nhỏ, bỏ qua ---
    if abs(dt) < min_update_interval:
        return sat_data

    # --- Dữ liệu cơ bản ---
    alt = sat_data["position"]["alt"]
    r = EARTH_RADIUS + alt
    T = sat_data["orbit"]["period"]
    inc = math.radians(sat_data["orbit"]["inclination"])
    raan = math.radians(sat_data["orbit"]["raan"])
    theta0 = sat_data["orbit_state"]["last_theta"]

    # --- Góc mới ---
    delta_theta = 2 * math.pi * (dt / T)
    theta = (theta0 + delta_theta) % (2 * math.pi)

    # --- Vị trí trong mặt phẳng quỹ đạo ---
    x_orb = r * math.cos(theta)
    y_orb = r * math.sin(theta)

    # --- Quay sang hệ ECI ---
    x1 = x_orb
    y1 = y_orb * math.cos(inc)
    z1 = y_orb * math.sin(inc)

    x = x1 * math.cos(raan) - y1 * math.sin(raan)
    y = x1 * math.sin(raan) + y1 * math.cos(raan)
    z = z1

    # --- ECI -> Lat/Lon/Alt ---
    r_new = math.sqrt(x**2 + y**2 + z**2)
    lat_new = math.degrees(math.asin(z / r_new))
    lon_new = math.degrees(math.atan2(y, x))
    alt_new = r_new - EARTH_RADIUS

    # --- Cập nhật object ---
    sat_data["position"] = {"lat": lat_new, "lon": lon_new, "alt": alt_new}
    sat_data["last_update"] = target_time.isoformat()
    sat_data["orbit_state"]["last_theta"] = theta

    # --- Nếu truyền MongoDB collection, update DB ---
    if db_collection is not None:
        db_collection.update_one(
            {"satellite_id": sat_data["satellite_id"]},
            {"$set": {
                "position": sat_data["position"],
                "last_update": sat_data["last_update"],
                "orbit_state.last_theta": sat_data["orbit_state"]["last_theta"]
            }}
        )

    return sat_data

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Tính khoảng cách giữa hai điểm trên bề mặt Trái Đất (theo đường tròn lớn).
    Sử dụng công thức Haversine.
    Trả về khoảng cách tính bằng mét.
    """
    R = EARTH_RADIUS  # bán kính Trái Đất (mét)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

def distance_3d(lat1, lon1, alt1, lat2, lon2, alt2):
    """
    Tính khoảng cách 3D giữa hai điểm trên hoặc cách mặt Trái Đất.
    lat/lon tính bằng độ, alt tính bằng mét.
    """
    # --- Chuyển sang radian ---
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    lam1 = math.radians(lon1)
    lam2 = math.radians(lon2)

    # --- Tọa độ Cartesian ---
    x1 = (EARTH_RADIUS + alt1) * math.cos(phi1) * math.cos(lam1)
    y1 = (EARTH_RADIUS + alt1) * math.cos(phi1) * math.sin(lam1)
    z1 = (EARTH_RADIUS + alt1) * math.sin(phi1)

    x2 = (EARTH_RADIUS + alt2) * math.cos(phi2) * math.cos(lam2)
    y2 = (EARTH_RADIUS + alt2) * math.cos(phi2) * math.sin(lam2)
    z2 = (EARTH_RADIUS + alt2) * math.sin(phi2)

    # --- Khoảng cách Euclidean ---
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    return distance

import math

EARTH_RADIUS = 6371000  # bán kính Trái Đất tính bằng mét

def can_connect_sat(sat_data, dev_lat, dev_lon, dev_alt=0):
    """
    Kiểm tra xem thiết bị có nằm trong vùng phủ của vệ tinh hay không.
    
    :param sat_data: dict vệ tinh, cần có keys: type, position {lat, lon, alt}
    :param dev_lat: latitude thiết bị (độ)
    :param dev_lon: longitude thiết bị (độ)
    :param dev_alt: altitude thiết bị (mét, default=0)
    :return: True nếu thiết bị nằm trong vùng phủ, False nếu không
    """
    # --- Chọn el_min theo loại vệ tinh ---
    sat_type = sat_data["type"]
    if sat_type == "LEO":
        el_min_deg = 7.5
    elif sat_type == "GEO":
        el_min_deg = 5.0
    else:
        el_min_deg = 7.5  # mặc định
    
    # --- Lấy vị trí vệ tinh ---
    sat_lat = sat_data["position"]["lat"]
    sat_lon = sat_data["position"]["lon"]
    sat_alt = sat_data["position"]["alt"]
    
    # --- Chuyển sang radian ---
    phi_sat = math.radians(sat_lat)
    lam_sat = math.radians(sat_lon)
    phi_dev = math.radians(dev_lat)
    lam_dev = math.radians(dev_lon)
    
    # --- Tọa độ Cartesian ---
    x_sat = (EARTH_RADIUS + sat_alt) * math.cos(phi_sat) * math.cos(lam_sat)
    y_sat = (EARTH_RADIUS + sat_alt) * math.cos(phi_sat) * math.sin(lam_sat)
    z_sat = (EARTH_RADIUS + sat_alt) * math.sin(phi_sat)
    
    x_dev = (EARTH_RADIUS + dev_alt) * math.cos(phi_dev) * math.cos(lam_dev)
    y_dev = (EARTH_RADIUS + dev_alt) * math.cos(phi_dev) * math.sin(lam_dev)
    z_dev = (EARTH_RADIUS + dev_alt) * math.sin(phi_dev)
    
    # --- Vector từ thiết bị đến vệ tinh ---
    dx = x_sat - x_dev
    dy = y_sat - y_dev
    dz = z_sat - z_dev
    
    # --- Khoảng cách từ thiết bị đến vệ tinh ---
    d = math.sqrt(dx**2 + dy**2 + dz**2)
    
    # --- Vector từ thiết bị đến tâm Trái Đất ---
    r_dev = math.sqrt(x_dev**2 + y_dev**2 + z_dev**2)
    
    # --- Tính góc nâng chính xác ---
    # Sử dụng công thức: sin(el) = (R⃗_sat • R⃗_dev) / (|R⃗_sat| |R⃗_dev|) - R/|R⃗_dev|
    # Hoặc cách đơn giản hơn: cos(zenith_angle) = (dx*x_dev + dy*y_dev + dz*z_dev) / (d * r_dev)
    
    dot_product = dx*x_dev + dy*y_dev + dz*z_dev
    cos_zenith = dot_product / (d * r_dev)
    cos_zenith = max(-1.0, min(1.0, cos_zenith))  # giới hạn trong [-1, 1]
    
    # Góc thiên đỉnh (zenith angle) là góc giữa vector thẳng đứng và vector tới vệ tinh
    # Góc nâng (elevation) = 90° - góc thiên đỉnh
    zenith_angle_rad = math.acos(cos_zenith)
    el_rad = math.pi/2 - zenith_angle_rad
    el_deg = math.degrees(el_rad)
    
    return el_deg >= el_min_deg

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pythonService:app", host="0.0.0.0", port=8000, reload=True)