from node import node
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import math

EARTH_RADIUS = 6371000

class Satellite(node):
    def __init__(self, sat):
        super().__init__(sat["position"], sat["resources"])
        self.last_theta = sat["orbit_state"]["last_theta"]
        self.id = sat["satellite_id"]
        self.last_update = sat["last_update"]
        self.type = sat["type"]
        self.orbit = sat["orbit"]
        self.connections = []
        self.priority = 4
        self.typename = "satellite"
        
    def can_connect(self, dev_lat, dev_lon, dev_alt = 0, el_min_deg=None, collection=None):
        self.update_satellite_position_obj_db(collection)
        # --- chọn el_min ---
        if el_min_deg is None:
            if self.type == "LEO":
                el_min_deg = 7.5
            elif self.type == "GEO":
                el_min_deg = 5.0
            else:
                el_min_deg = 7.5

        # --- vị trí ---
        sat_lat, sat_lon, sat_alt = self.position["lat"], self.position["lon"], self.position["alt"]

        # --- convert to rad ---
        phi_sat, lam_sat = math.radians(sat_lat), math.radians(sat_lon)
        phi_dev, lam_dev = math.radians(dev_lat), math.radians(dev_lon)

        # --- Cartesian ---
        x_sat = (EARTH_RADIUS + sat_alt) * math.cos(phi_sat) * math.cos(lam_sat)
        y_sat = (EARTH_RADIUS + sat_alt) * math.cos(phi_sat) * math.sin(lam_sat)
        z_sat = (EARTH_RADIUS + sat_alt) * math.sin(phi_sat)

        x_dev = (EARTH_RADIUS + dev_alt) * math.cos(phi_dev) * math.cos(lam_dev)
        y_dev = (EARTH_RADIUS + dev_alt) * math.cos(phi_dev) * math.sin(lam_dev)
        z_dev = (EARTH_RADIUS + dev_alt) * math.sin(phi_dev)

        # --- vector từ dev → sat ---
        dx, dy, dz = x_sat - x_dev, y_sat - y_dev, z_sat - z_dev
        d = math.sqrt(dx*dx + dy*dy + dz*dz)

        # --- vector từ dev → tâm Trái Đất ---
        r_dev = math.sqrt(x_dev*x_dev + y_dev*y_dev + z_dev*z_dev)

        # --- cos(zenith) ---
        dot_product = dx*x_dev + dy*y_dev + dz*z_dev
        cos_zenith = dot_product / (d * r_dev)
        cos_zenith = max(-1.0, min(1.0, cos_zenith))

        # --- góc nâng ---
        zenith_angle = math.acos(cos_zenith)
        el_deg = math.degrees(math.pi/2 - zenith_angle)

        return el_deg >= el_min_deg

    
    def update_satellite_position_obj_db(self, db_collection=None, target_time: datetime = None, min_update_interval: float = 1.0, min_db_update_interval: float = 2000):
        """
        Cập nhật vị trí vệ tinh từ object, có thể update trực tiếp vào MongoDB nếu truyền collection.

        :param sat_data: dict vệ tinh
        :param db_collection: pymongo collection (tùy chọn)
        :param target_time: thời điểm tính vị trí mới, mặc định giờ hiện tại
        :param min_update_interval: ngưỡng Δt (giây) để bỏ qua nếu quá nhỏ
        :return: sat_data mới (đã cập nhật position, last_update, last_theta)
        """

        if self.type == "GEO":
            return
        
        if target_time is None:
            target_time = datetime.now(timezone.utc)

        last_update = datetime.fromisoformat(self.last_update.replace("Z", "+00:00"))
        dt = (target_time - last_update).total_seconds()

        # --- Nếu Δt quá nhỏ, bỏ qua ---
        if abs(dt) < min_update_interval:
            return

        # --- Dữ liệu cơ bản ---
        alt = self.position["alt"]
        r = EARTH_RADIUS + alt
        T = self.orbit["period"]
        inc = math.radians(self.orbit["inclination"])
        raan = math.radians(self.orbit["raan"])
        theta0 = self.last_theta

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
        lon_new = (lon_new + 180) % 360 - 180
        alt_new = r_new - EARTH_RADIUS

        # --- Cập nhật object ---
        self.position = {"lat": lat_new, "lon": lon_new, "alt": alt_new}
        self.last_update = target_time.isoformat()
        self.last_theta = theta

        if dt > min_db_update_interval:

            # --- Nếu truyền MongoDB collection, update DB ---
            if db_collection is not None:
                db_collection.update_one(
                    {"satellite_id": self.id},
                    {"$set": {
                        "position": self.position,
                        "last_update": self.last_update,
                        "orbit_state.last_theta": self.last_theta
                    }}
                )

    def scan_neighbor(self,nodes):
        neighbors = []
        for node in nodes:
            if hasattr(node, 'id') and node.id == self.id:
                continue
            if self.can_connect_sat(node.position["lat"], node.position["lon"], node.position["alt"]):
                neighbors.append(node)
        return neighbors
        

