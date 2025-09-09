import math

EARTH_RADIUS = 6371000

class node():
    def __init__(self, position, resources):
        self.position = position
        if len(self.position) != 3:
            self.position["alt"] = 0
        self.resources = resources
        self.connections = []
        self.free_resources = resources.copy()

    def calculate_distance(self, lat2, lon2, alt2=0):
        lat1 = self.position["lat"]
        lon1 = self.position["lon"]
        alt1 = self.position["alt"]
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
    
    def allocate_resource(self):
        raise NotImplementedError("This method will be built later")

    