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
    def __init__(self):
        # Danh sách node, key = id, value = object node
        self.collection = db["satellites"]  # dùng để cập nhật vị trí vệ tinh
        self.nodes = {}      
        for sat in self.collection.find():
            self.add_node(Satellite(sat))
        self.collection = db["groundstations"]
        for gs in self.collection.find():
            self.add_node(Gs(gs))
        self.collection = db["seastations"]
        for ss in self.collection.find():
            self.add_node(Ss(ss))
        self.collection = db["satellites"]  # dùng để cập nhật vị trí vệ tinh

    def add_node(self, node_obj):
        """Thêm 1 node mới vào hệ thống"""
        self.nodes[node_obj.id] = node_obj

    def remove_node(self, node_id):
        """Xóa node theo id"""
        if node_id in self.nodes:
            del self.nodes[node_id]

    def get_node(self, node_id):
        """Lấy node theo id"""
        return self.nodes.get(node_id, None)

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
        for target_id, target in self.nodes.items():
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
                    collection=collection if target.typename == "satellite" else None,
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
