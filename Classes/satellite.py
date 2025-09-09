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
        
    def can_connect_sat(self, dev_lat, dev_lon, dev_alt=0):
        """
        Kiểm tra xem thiết bị có nằm trong vùng phủ của vệ tinh hay không.
        
        :param sat_data: dict vệ tinh, cần có keys: type, position {lat, lon, alt}
        :param dev_lat: latitude thiết bị (độ)
        :param dev_lon: longitude thiết bị (độ)
        :param dev_alt: altitude thiết bị (mét, default=0)
        :return: True nếu thiết bị nằm trong vùng phủ, False nếu không
        """
        # --- Chọn el_min theo loại vệ tinh ---
        sat_type = self.type
        if sat_type == "LEO":
            el_min_deg = 7.5
        elif sat_type == "GEO":
            el_min_deg = 5.0
        else:
            el_min_deg = 7.5  # mặc định
        
        # --- Lấy vị trí vệ tinh ---
        sat_lat = self.position["lat"]
        sat_lon = self.position["lon"]
        sat_alt = self.position["alt"]
        
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
    
    def update_satellite_position_obj_db(self, db_collection=None, target_time: datetime = None, min_update_interval: float = 1.0):
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

        last_update = datetime.fromisoformat(self.last_theta.replace("Z", "+00:00"))
        dt = (target_time - last_update).total_seconds()

        # --- Nếu Δt quá nhỏ, bỏ qua ---
        if abs(dt) < min_update_interval:
            return

        # --- Dữ liệu cơ bản ---
        alt = self.position["alt"]
        r = EARTH_RADIUS + alt
        T = self.orbit["period"]
        inc = math.radians(self.orbit["inclination"])
        raan = math.radians(self.orbit["orbit"]["raan"])
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
        alt_new = r_new - EARTH_RADIUS

        # --- Cập nhật object ---
        self.position = {"lat": lat_new, "lon": lon_new, "alt": alt_new}
        self.last_update = target_time.isoformat()
        self.last_theta = theta

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

