from pymongo import MongoClient
from pymongo.server_api import ServerApi
from satellite import Satellite
from gs import Gs
from ss import Ss



uri = "mongodb+srv://longhoi856:UYcdtPdXsoYGFBrT@cluster0.hb5vpf7.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1'))
DB_NAME = "sagsins"
db = client[DB_NAME]

class Network:
    
    
    nodes = {}
    satellites = {}
    groundstations = {}
    seastations = {}

    def __init__(self):
        # chỉ load dữ liệu lần đầu khi nodes rỗng
        if not Network.nodes:
            # Load satellites
            self.collection = db["satellites"]
            for sat in self.collection.find():
                self.add_node(Satellite(sat))

            # Load ground stations
            self.collection = db["groundstations"]
            for gs in self.collection.find():
                self.add_node(Gs(gs))

            # Load sea stations
            self.collection = db["seastations"]
            for ss in self.collection.find():
                self.add_node(Ss(ss))

        # default collection for satellite updates
        self.collection = db["satellites"]

    def add_node(self, node_obj):
        """Thêm 1 node mới vào hệ thống"""
        Network.nodes[node_obj.id] = node_obj
        if node_obj.typename == "satellite":
            Network.satellites[node_obj.id] = node_obj
        elif node_obj.typename == "groundstation":
            Network.groundstations[node_obj.id] = node_obj
        elif node_obj.typename == "seastation":
            Network.seastations[node_obj.id] = node_obj

    def remove_node(self, node_id):
        """Xóa node theo id"""
        if node_id in Network.nodes:
            del Network.nodes[node_id]

    def get_node(self, node_id):
        """Lấy node theo id"""
        return Network.nodes.get(node_id, None)

    def find_connectable_nodes(self, node_id, mode="auto", support5G=True):
        """
        Tìm các node mà node_id có thể kết nối tới.
        Kết nối được xem là hợp lệ nếu ít nhất 1 trong 2 chiều có thể kết nối.
        
        :param node_id: id của node nguồn
        :param mode: kiểu tính khoảng cách ('3d', 'surface', hoặc 'auto')
        :param support5G: nếu False thì bỏ qua kết nối với satellite
        :return: list các node connectable
        """
        source = self.get_node(node_id)
        if not source:
            return []

        connectable = []
        for target_id, target in Network.nodes.items():
            if target_id == node_id:
                continue

            if not support5G and target.typename == "satellite":
                continue

            try:
                # Kiểm tra 2 chiều
                ok = (
                    source.can_connect(
                        target.position["lat"],
                        target.position["lon"],
                        target.position["alt"],
                        collection=self.collection if target.typename == "satellite" else None,
                    )
                    or target.can_connect(
                        source.position["lat"],
                        source.position["lon"],
                        source.position["alt"],
                        collection=self.collection if source.typename == "satellite" else None,
                    )
                )

                if ok:
                    connectable.append(target)

            except Exception as e:
                print(f"⚠️ Error checking {source.id}->{target.id}: {e}")

        return connectable
    def find_connectable_nodes(self, node_id, mode="auto", support5G=True):
        """
        Tìm các node mà node_id có thể kết nối tới
        :param node_id: id của node nguồn
        :param mode: kiểu tính khoảng cách ('3d', 'surface', hoặc 'auto')
        :return: list các node connectable
        """
        source = self.get_node(node_id)
        if not source:
            return []

        connectable = []
        for target_id, target in Network.nodes.items():
            if target_id == node_id:
                continue

            if not support5G and target.typename == "satellite":
                continue

            # gọi hàm can_connect của source
            try:
                if source.can_connect(
                    target.position["lat"],
                    target.position["lon"],
                    target.position["alt"],
                    collection=self.collection if target.typename == "satellite" else None,
                ):
                    connectable.append(target)
            except Exception as e:
                print(f"⚠️ Error checking {source.id}->{target.id}: {e}")

        return connectable
    

    def find_connectable_nodes_for_location(self, lat, lon, alt=0, mode="auto", support5G=True):
        """
        Tìm các node mà có thể kết nối tới vị trí (lat, lon, alt)
        :param lat: vĩ độ
        :param lon: kinh độ
        :param alt: độ cao (mét)
        :param mode: kiểu tính khoảng cách ('3d', 'surface', hoặc 'auto')
        :return: list các node connectable
        """
        connectable = []
        for target_id, target in self.nodes.items():
            if not support5G and target.typename == "satellite":
                continue

            # gọi hàm can_connect của target
            try:
                if target.can_connect(
                    lat,
                    lon,
                    alt,
                    collection=self.collection if target.typename == "satellite" else None,
                ):
                    connectable.append(target)
            except Exception as e:
                print(f"⚠️ Error checking location->{target.id}: {e}")

        return connectable

    def get_adjacency_list(self, mode="auto"):
        """
        Trả về toàn bộ danh sách kết nối dưới dạng adjacency list
        """
        adj = {}
        for node_id in self.nodes:
            adj[node_id] = [n.id for n in self.find_connectable_nodes(node_id, mode)]
        return adj
    
    def distance_to_nearest_gs(self, node, mode="auto"):
        """
        Tính khoảng cách đến satellite gần nhất từ node
        :param node: đối tượng node (node, gs, ss)
        :param mode: kiểu tính khoảng cách ('3d', 'surface', hoặc 'auto')
        :return: khoảng cách (mét) hoặc None nếu không có satellite nào
        """
        min_dist = None
        for target_id, target in Network.groundstations.items():
            try:
                dist = target.calculate_distance(node.position["lat"], node.position["lon"], node.position["alt"], mode=mode)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
            except Exception as e:
                print(f"⚠️ Error calculating distance to {target.id}: {e}")

        return min_dist
    
    def distance_to_nearest_gs(self, lat, lon, alt=0, mode="auto"):
        """
        Tính khoảng cách đến ground station gần nhất từ vị trí (lat, lon, alt)
        :param lat: vĩ độ
        :param lon: kinh độ
        :param alt: độ cao (mét)
        :param mode: kiểu tính khoảng cách ('3d', 'surface', hoặc 'auto')
        :return: khoảng cách (mét) hoặc None nếu không có ground station nào
        """
        min_dist = None
        for target_id, target in Network.groundstations.items():
            try:
                dist = target.calculate_distance(lat, lon, alt, mode=mode)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
            except Exception as e:
                print(f"⚠️ Error calculating distance to {target.id}: {e}")

        return min_dist
