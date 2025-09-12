import math
from scipy.spatial import KDTree

EARTH_RADIUS = 6371000  # meters

def latlon_to_cartesian(lat, lon):
    """Convert lat/lon to Cartesian coordinates on unit sphere."""
    phi = math.radians(lat)
    lam = math.radians(lon)
    x = math.cos(phi) * math.cos(lam)
    y = math.cos(phi) * math.sin(lam)
    z = math.sin(phi)
    return (x, y, z)

class GroundSpace:
    def __init__(self):
        self.requests = {}       # request_id -> request object
        self.req_coords = []     # cartesian coords
        self.req_ids = []        # parallel list of ids
        self.kdtree = None       # KDTree for fast lookup

    def add_request(self, req_obj):
        """Add or update a request (user)."""
        self.requests[req_obj.request_id] = req_obj
        self._rebuild_index()

    def remove_request(self, request_id):
        """Remove a request (user)."""
        if request_id in self.requests:
            del self.requests[request_id]
            self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild KDTree when requests change."""
        self.req_ids = list(self.requests.keys())
        self.req_coords = [
            latlon_to_cartesian(
                self.requests[rid].source_location["lat"],
                self.requests[rid].source_location["lon"]
            )
            for rid in self.req_ids
        ]
        if self.req_coords:
            self.kdtree = KDTree(self.req_coords)
        else:
            self.kdtree = None

    def nearby_count(self, lat, lon, radius_km):
        """Return number of requests within radius_km of (lat, lon)."""
        if not self.kdtree:
            return 0

        q = latlon_to_cartesian(lat, lon)
        angular_radius = radius_km * 1000 / EARTH_RADIUS
        chord_dist = 2 * math.sin(angular_radius / 2)

        idxs = self.kdtree.query_ball_point(q, chord_dist)
        return len(idxs)

    def nearby_requests(self, lat, lon, radius_km):
        """Return list of request objects within radius_km of (lat, lon)."""
        if not self.kdtree:
            return []

        q = latlon_to_cartesian(lat, lon)
        angular_radius = radius_km * 1000 / EARTH_RADIUS
        chord_dist = 2 * math.sin(angular_radius / 2)

        idxs = self.kdtree.query_ball_point(q, chord_dist)
        return [self.requests[self.req_ids[i]] for i in idxs]

    def nearby_to_request(self, request_id, radius_km):
        """Return list of request objects near an existing request."""
        if request_id not in self.requests:
            return []
        loc = self.requests[request_id].source_location
        return self.nearby_requests(loc["lat"], loc["lon"], radius_km)
