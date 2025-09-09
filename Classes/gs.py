from node import node

class Gs(node):
    def __init__(self, gs):
        super().__init__(gs["location"], gs["resources"])
        self.id = gs["gs_id"]
        self.coverage_radius_km = gs["coverage_radius_km"]
        self.type = "groundstation"
        self.connections = []

    def can_connect_gs(self, dev_lat, dev_lon, dev_alt=0):
        dist = self.calculate_distance(dev_lat, dev_lon, dev_alt)/1000  # km
        return dist <= self.coverage_radius_km
    
    def connect_gs(self):
        raise NotImplementedError("This method will be built later")
    
    def disconnect_gs(self):
        raise NotImplementedError("This method will be built later")