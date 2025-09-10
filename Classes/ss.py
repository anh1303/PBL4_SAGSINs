from node import node

class Ss(node):
    def __init__(self, ss):
        super().__init__(ss["location"], ss["resources"])
        self.id = ss["ss_id"]
        self.coverage_radius_km = ss["coverage_radius_km"]
        self.type = "seastation"
        self.connections = []
        self.priority = 2

    def can_connect_ss(self, dev_lat, dev_lon, dev_alt=0):
        dist = self.calculate_distance(dev_lat, dev_lon, dev_alt)/1000  # km
        return dist <= self.coverage_radius_km
    
    def connect_ss(self):
        raise NotImplementedError("This method will be built later")
    
    def disconnect_ss(self):
        raise NotImplementedError("This method will be built later")