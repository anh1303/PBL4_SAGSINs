from node import node

class Ss(node):
    def __init__(self, ss):
        super().__init__(ss["location"], ss["resources"])
        self.id = ss["ss_id"]
        self.coverage_radius_km = ss["coverage_radius_km"]
        self.type = "seastation"
        self.connections = []
        self.priority = 2
        self.typename = self.type = "seastation"

    def can_connect(self, dev_lat, dev_lon, dev_alt=0, collection=None, is_sat=False):
        if is_sat:
            return False
        dist_km = self.calculate_distance(dev_lat, dev_lon, dev_alt, mode="surface") / 1000
        return dist_km <= self.coverage_radius_km

    
    def connect_ss(self):
        raise NotImplementedError("This method will be built later")
    
    def disconnect_ss(self):
        raise NotImplementedError("This method will be built later")