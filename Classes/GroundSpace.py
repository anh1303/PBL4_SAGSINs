import math
import time
import threading
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


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance (meters)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS * c


class GroundSpace:
    def __init__(self, rebuild_threshold=100, max_stale_time=5.0):
        """
        :param rebuild_threshold: number of add/remove before rebuild
        :param max_stale_time: max seconds before forcing rebuild
        """
        self.requests = {}       # request_id -> request object
        self.req_ids = []        # parallel list of ids
        self.req_coords = []     # cartesian coords
        self.kdtree = None

        self.lock = threading.Lock()
        self.rebuild_threshold = rebuild_threshold
        self.max_stale_time = max_stale_time

        self.update_count = 0
        self.dirty = False
        self.rebuilding = False
        self.last_rebuild = 0.0

    def add_request(self, req_obj):
        with self.lock:
            self.requests[req_obj.request_id] = req_obj
            self.update_count += 1
            self.dirty = True
        self._maybe_rebuild()

    def remove_request(self, request_id):
        with self.lock:
            if request_id in self.requests:
                del self.requests[request_id]
                self.update_count += 1
                self.dirty = True
        self._maybe_rebuild()

    def _maybe_rebuild(self):
        """Trigger background rebuild if needed."""
        now = time.time()
        with self.lock:
            need_rebuild = (
                self.dirty
                and not self.rebuilding
                and (
                    self.update_count >= self.rebuild_threshold
                    or now - self.last_rebuild >= self.max_stale_time
                )
            )
            if not need_rebuild:
                return
            self.rebuilding = True

        def _worker():
            # Take snapshot outside lock
            with self.lock:
                snapshot = list(self.requests.items())
            ids = [rid for rid, _ in snapshot]
            coords = [
                latlon_to_cartesian(req.source_location["lat"], req.source_location["lon"])
                for _, req in snapshot
            ]
            tree = KDTree(coords) if coords else None

            # Update structures inside lock
            with self.lock:
                self.req_ids, self.req_coords, self.kdtree = ids, coords, tree
                self.dirty = False
                self.update_count = 0
                self.rebuilding = False
                self.last_rebuild = time.time()

        threading.Thread(target=_worker, daemon=True).start()

    def nearby_requests(self, lat, lon, radius_km, sort_by_distance=False):
        """Return list of request objects within radius_km of (lat, lon)."""
        with self.lock:
            tree, ids = self.kdtree, self.req_ids
        if not tree:
            return []

        q = latlon_to_cartesian(lat, lon)
        angular_radius = radius_km * 1000 / EARTH_RADIUS
        chord_dist = 2 * math.sin(angular_radius / 2)

        idxs = tree.query_ball_point(q, chord_dist)
        results = []
        for i in idxs:
            req = self.requests.get(ids[i])
            if req:
                d = haversine(lat, lon,
                              req.source_location["lat"],
                              req.source_location["lon"])
                if d <= radius_km * 1000:
                    results.append((d, req))

        if sort_by_distance:
            results.sort(key=lambda x: x[0])
        return [r for _, r in results]

    def nearby_count(self, lat, lon, radius_km):
        """Return number of requests within radius_km of (lat, lon)."""
        return len(self.nearby_requests(lat, lon, radius_km, sort_by_distance=False))

    def nearby_to_request(self, request_id, radius_km, sort_by_distance=False):
        """Return requests near an existing request."""
        with self.lock:
            req = self.requests.get(request_id)
        if not req:
            return []
        loc = req.source_location
        return self.nearby_requests(loc["lat"], loc["lon"], radius_km, sort_by_distance)
