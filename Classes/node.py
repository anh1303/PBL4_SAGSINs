import math
from request import ServiceType, request

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
    
    def allocate_resource(self, req: request, allow_partial=True) -> bool:
        """
        Try to allocate resources (bandwidth, CPU, power) for a request.
        Returns True if at least partial allocation succeeded, False otherwise.
        """

        # --- Available resources ---
        bw_free = self.free_resources.get("bandwidth", 0)
        cpu_free = self.free_resources.get("cpu", 0)
        power_free = self.free_resources.get("power", 0)

        # --- Requested resources ---
        bw_needed = req.bandwidth_required
        cpu_needed = req.cpu_required
        power_needed = req.power_required

        # --- Track allocated ---
        bw_alloc = 0
        cpu_alloc = 0
        power_alloc = 0

        # --- Full allocation possible ---
        if bw_free >= bw_needed and cpu_free >= cpu_needed and power_free >= power_needed:
            bw_alloc, cpu_alloc, power_alloc = bw_needed, cpu_needed, power_needed
            self.free_resources["bandwidth"] -= bw_alloc
            self.free_resources["cpu"] -= cpu_alloc
            self.free_resources["power"] -= power_alloc

        else:
            # --- Partial allocation ---
            if allow_partial and req.allow_partial and req.type not in [ServiceType.EMERGENCY, ServiceType.CONTROL]:
                # Allocate as much as possible from each dimension
                bw_alloc = min(bw_free, bw_needed)
                cpu_alloc = min(cpu_free, cpu_needed)
                power_alloc = min(power_free, power_needed)

                # Reduce free pool
                self.free_resources["bandwidth"] -= bw_alloc
                self.free_resources["cpu"] -= cpu_alloc
                self.free_resources["power"] -= power_alloc
            else:
                # Not enough resources and partial not allowed → reject
                return False

        # --- Save allocation in request object ---
        req.bandwidth_allocated = bw_alloc
        req.cpu_allocated = cpu_alloc
        req.power_allocated = power_alloc

        # --- Track connection ---
        self.connections.append({
            "request_id": req.request_id,
            "bandwidth": bw_alloc,
            "cpu": cpu_alloc,
            "power": power_alloc,
            "priority": req.priority,
            "type": req.type,
        })

        return True



    