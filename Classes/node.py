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
        self.resources_used = resources.copy()
        self.free_resources = resources.copy()
        for i in self.resources_used:
            self.resources_used[i] = 0
        self.typename = "node"
        self.type = ""

    import math

    def calculate_distance(self, lat2, lon2, alt2=0, mode="3d"):
        lat1 = self.position["lat"]
        lon1 = self.position["lon"]
        alt1 = self.position["alt"]

        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        lam1, lam2 = math.radians(lon1), math.radians(lon2)

        if mode == "surface":
            # --- Haversine formula ---
            dphi = phi2 - phi1
            dlam = lam2 - lam1
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            return EARTH_RADIUS * c  # mét

        else:  # "3d" hoặc "auto"
            # --- Cartesian ---
            x1 = (EARTH_RADIUS + alt1) * math.cos(phi1) * math.cos(lam1)
            y1 = (EARTH_RADIUS + alt1) * math.cos(phi1) * math.sin(lam1)
            z1 = (EARTH_RADIUS + alt1) * math.sin(phi1)

            x2 = (EARTH_RADIUS + alt2) * math.cos(phi2) * math.cos(lam2)
            y2 = (EARTH_RADIUS + alt2) * math.cos(phi2) * math.sin(lam2)
            z2 = (EARTH_RADIUS + alt2) * math.sin(phi2)

            dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
            return math.sqrt(dx*dx + dy*dy + dz*dz)  # mét

    
    def allocate_resource(self, req: request, allow_partial=True) -> bool:
        """
        Try to allocate resources (bandwidth, CPU, power) for a request.
        Returns True if at least partial allocation succeeded, False otherwise.
        """

        # --- Available resources ---
        up_link_free = self.free_resources.get("uplink", 0)
        down_link_free = self.free_resources.get("downlink", 0)
        cpu_free = self.free_resources.get("cpu", 0)
        power_free = self.free_resources.get("power", 0)

        # --- Requested resources ---
        uplink_needed = req.uplink_required
        downlink_needed = req.downlink_required
        cpu_needed = req.cpu_required
        power_needed = req.power_required

        # --- Track allocated ---
        up_link_alloc = 0
        down_link_alloc = 0
        cpu_alloc = 0
        power_alloc = 0

        # --- Full allocation possible ---
        if cpu_free >= cpu_needed and power_free >= power_needed:
            bw_alloc, cpu_alloc, power_alloc = bw_needed, cpu_needed, power_needed
            self.free_resources["bandwidth"] -= bw_alloc
            self.free_resources["cpu"] -= cpu_alloc
            self.free_resources["power"] -= power_alloc

        else:
            # --- Partial allocation ---
            if allow_partial and req.allow_partial and req.type not in [ServiceType.EMERGENCY, ServiceType.CONTROL]:
                # Allocate as much as possible from each dimension
                up_link_alloc = min(up_link_free, uplink_needed)
                down_link_alloc = min(down_link_free, downlink_needed)
                cpu_alloc = min(cpu_free, cpu_needed)
                power_alloc = min(power_free, power_needed)

                # Reduce free pool
                self.free_resources["uplink"] -= up_link_alloc
                self.free_resources["downlink"] -= down_link_alloc
                self.free_resources["cpu"] -= cpu_alloc
                self.free_resources["power"] -= power_alloc
            else:
                # Not enough resources and partial not allowed → reject
                return False

        # --- Save allocation in request object ---
        req.uplink_allocated = up_link_alloc
        req.downlink_allocated = down_link_alloc
        req.cpu_allocated = cpu_alloc
        req.power_allocated = power_alloc

        # --- Track connection ---
        self.connections.append({
            "request_id": req.request_id,
            "uplink": up_link_alloc,
            "downlink": down_link_alloc,
            "cpu": cpu_alloc,
            "power": power_alloc,
            "priority": req.priority,
            "type": req.type,
        })

        return True
    
    def can_connect(self, dev_lat, dev_lon, dev_alt=0, collection = None, is_sat = False):
        raise NotImplementedError("This method should be implemented in subclasses")



    