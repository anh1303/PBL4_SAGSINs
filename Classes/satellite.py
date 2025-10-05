from node import node
from datetime import datetime, timedelta, timezone
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
        
    def can_connect(self, dev_lat, dev_lon, dev_alt = 0, el_min_deg=None, collection=None, is_sat =False):
        self.update_satellite_position_obj_db(collection)
        if is_sat:
            return self.can_connect_sat(dev_lat, dev_lon, dev_alt)
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

        :param db_collection: pymongo collection (tùy chọn)
        :param target_time: thời điểm tính vị trí mới, mặc định giờ hiện tại
        :param min_update_interval: ngưỡng Δt (giây) để bỏ qua nếu quá nhỏ
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
            if self.can_connect(node.position["lat"], node.position["lon"], node.position["alt"]):
                neighbors.append(node)
        return neighbors
    
    def compute_elevation(self, dev_lat, dev_lon, dev_alt=0):
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
        
        return el_deg
        
        
    def estimate_visible_time(self, dev_lat, dev_lon, dev_alt=0, max_time = 7200, el_min_deg = None):
        """
        Estimate how long (in seconds) the satellite will remain above the given elevation threshold.

        :param dev_lat: device latitude (deg)
        :param dev_lon: device longitude (deg)
        :param dev_alt: device altitude (m)
        :param el_min_deg: minimum elevation angle (deg)
        :param db_collection: optional MongoDB collection
        :param step: time step for simulation (seconds)
        :param max_time: maximum search window (seconds)
        :return: estimated visible time (seconds)
        """
        # --- Default elevation threshold ---
        if el_min_deg is None:
            el_min_deg = 7.5 if self.type == "LEO" else 5.0

        # --- Start from current time ---
        t_now = datetime.now(timezone.utc)
        visible_time = 0
        
        # Temporarily compute propagated position (do not commit to DB)
        sat_copy = Satellite({
            "position": dict(self.position),
            "resources": self.resources,
            "orbit_state": {"last_theta": self.last_theta},
            "satellite_id": self.id,
            "last_update": self.last_update,
            "type": self.type,
            "orbit": self.orbit
        })
        
        target_time = t_now + timedelta(seconds=max_time)
        sat_copy.update_satellite_position_obj_db(target_time=target_time, min_update_interval=0, min_db_update_interval=float('inf'))
        el_deg = sat_copy.compute_elevation(
            dev_lat,
            dev_lon,
            dev_alt
        )
        if el_deg >= el_min_deg:
            return max_time  # visible throughout the window

        left = 0
        right = max_time
        while left <= right:
            mid = (left + right) // 2
            target_time = t_now + timedelta(seconds=mid)
            sat_copy.position = dict(self.position)
            sat_copy.last_theta = self.last_theta
            sat_copy.last_update = self.last_update
            sat_copy.update_satellite_position_obj_db(target_time=target_time, min_update_interval=0, min_db_update_interval=float('inf'))

            el = sat_copy.compute_elevation(
                dev_lat,
                dev_lon,
                dev_alt
            )

            if el >= el_min_deg:
                visible_time = mid
                left = mid + 1
            else:
                right = mid - 1
            
        return visible_time
    
    def can_connect_sat(self, dev_lat, dev_lon, dev_alt):
        """
        Check if this satellite can see another satellite (line-of-sight not blocked by Earth).
        Uses geometric visibility (no Earth obstruction).
        """
        def ecef_from_latlon(lat, lon, alt):
            phi, lam = math.radians(lat), math.radians(lon)
            x = (EARTH_RADIUS + alt) * math.cos(phi) * math.cos(lam)
            y = (EARTH_RADIUS + alt) * math.cos(phi) * math.sin(lam)
            z = (EARTH_RADIUS + alt) * math.sin(phi)
            return x, y, z

        # Cartesian coordinates
        x1, y1, z1 = ecef_from_latlon(self.position["lat"], self.position["lon"], self.position["alt"])
        x2, y2, z2 = ecef_from_latlon(dev_lat, dev_lon, dev_alt)

        # Vector difference
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        denom = dx*dx + dy*dy + dz*dz
        if denom == 0:
            return False  # same point or numerical issue

        # Parameter of closest point to Earth's center along the line
        t = -(x1*dx + y1*dy + z1*dz) / denom

        # Compute minimum distance from Earth's center to the line segment
        if 0 <= t <= 1:
            # Closest point lies between satellites
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy
            closest_z = z1 + t * dz
            d_min = math.sqrt(closest_x**2 + closest_y**2 + closest_z**2)
        else:
            # Closest point outside segment — use nearest endpoint
            d1 = math.sqrt(x1**2 + y1**2 + z1**2)
            d2 = math.sqrt(x2**2 + y2**2 + z2**2)
            d_min = min(d1, d2)

        # Line of sight if the Earth doesn't block the path
        return d_min > EARTH_RADIUS
    
    def estimate_visible_time_sat(self, other_sat, max_time=7200):
        """
        Estimate how long (in seconds) this satellite can see another satellite.

        :param other_sat: another Satellite object
        :param max_time: maximum search window (seconds)
        :return: estimated visible time (seconds)
        """
        t_now = datetime.now(timezone.utc)
        visible_time = 0

        # Temporarily compute propagated positions (do not commit to DB)
        sat1_copy = Satellite({
            "position": dict(self.position),
            "resources": self.resources,
            "orbit_state": {"last_theta": self.last_theta},
            "satellite_id": self.id,
            "last_update": self.last_update,
            "type": self.type,
            "orbit": self.orbit
        })
        sat2_copy = Satellite({
            "position": dict(other_sat.position),
            "resources": other_sat.resources,
            "orbit_state": {"last_theta": other_sat.last_theta},
            "satellite_id": other_sat.id,
            "last_update": other_sat.last_update,
            "type": other_sat.type,
            "orbit": other_sat.orbit
        })

        target_time = t_now + timedelta(seconds=max_time)
        sat1_copy.update_satellite_position_obj_db(target_time=target_time, min_update_interval=0, min_db_update_interval=float('inf'))
        sat2_copy.update_satellite_position_obj_db(target_time=target_time, min_update_interval=0, min_db_update_interval=float('inf'))
        
        if sat1_copy.can_connect_sat(
            sat2_copy.position["lat"],
            sat2_copy.position["lon"],
            sat2_copy.position["alt"]
        ):
            return max_time  # visible throughout the window

        left = 0
        right = max_time
        while left <= right:
            mid = (left + right) // 2
            target_time = t_now + timedelta(seconds=mid)
            
            sat1_copy.position = dict(self.position)
            sat1_copy.last_theta = self.last_theta
            sat1_copy.last_update = self.last_update
            sat1_copy.update_satellite_position_obj_db(target_time=target_time, min_update_interval=0, min_db_update_interval=float('inf'))

            sat2_copy.position = dict(other_sat.position)
            sat2_copy.last_theta = other_sat.last_theta
            sat2_copy.last_update = other_sat.last_update
            sat2_copy.update_satellite_position_obj_db(target_time=target_time, min_update_interval=0, min_db_update_interval=float('inf'))
            if sat1_copy.can_connect_sat(
                sat2_copy.position["lat"],
                sat2_copy.position["lon"],
                sat2_copy.position["alt"]
            ):
                visible_time = mid
                left = mid + 1
            else:
                right = mid - 1
        return visible_time




