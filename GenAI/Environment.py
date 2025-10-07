import logging
import gym
from gym import spaces
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Classes')))
from satellite import Satellite
from gs import Gs
from ss import Ss
from network import Network
from request import ServiceType, request
from GroundSpace import GroundSpace
from enum import Enum, auto
import math
import random

#rel_link = exp(-gamma * (d_km))

C = 3e8

resource_profile_relay = {
    "cpu" : 5,
    "power" : 3
}

GAMMA_PROFILE = {
    ("groundstation", "user"):      2.0e-4,   # GS <-> user (short ground links)
    ("groundstation", "uav"):        1.0e-4,   # GS <-> UAV (short air/ground links)
    ("groundstation", "LEO"):        4.0e-5,   # GS <-> LEO (typical LEO slant ~500km)
    ("uav", "LEO"):       4.0e-5,   # UAV <-> LEO (similar to GS-LEO)
    ("LEO", "LEO"):       1.5e-5,   # inter-LEO ISL (optical or well-engineered RF)
    ("LEO", "GEO"):       2.0e-6,   # LEO <-> GEO (engineered long link)
    ("groundstation", "GEO"):        2.0e-6,   # GS <-> GEO (engineered uplink/downlink)
    ("seastation", "LEO"):5.0e-5,   # sea-station <-> LEO (conservative)
    ("default", "default"):1.0e-4,  # fallback
}

GAMMA_BY_TYPE = {
    "LEO": 5e-5,
    "GEO": 1e-6,
    "seastation": 1e-4,
    "uav": 8e-5,
    "groundstation": 2e-5
}

PROC_DELAY_BASE_MS = {
    "LEO": 3,            # typical satellite forwarding fast-path ~3 ms
    "GEO": 10,           # GEO ground processing often higher ~10 ms
    "seastation": 6,     # ship/sea gateways ~6 ms
    "uav": 6,            # UAV processing ~6 ms
    "groundstation": 10,  # GS fast-path ~10 ms (but can be reduced for emergency)
    "user": 6
}

# (w_lat, w_rel, w_up, w_down)
BonusProfilesForService = {
    # Latency -> Reliability -> UP/DOWN Link
    ServiceType.EMERGENCY:      (0.5, 0.3, 0.1, 0.1),
    ServiceType.CONTROL:        (0.5, 0.3, 0.1, 0.1),
    ServiceType.VOICE:          (0.5, 0.3, 0.1, 0.1),

    # UP/DOWN Link -> Reliability -> Latency
    ServiceType.VIDEO:          (0.1, 0.2, 0.4, 0.3),
    ServiceType.STREAMING:      (0.1, 0.2, 0.4, 0.3),
    ServiceType.BULK_TRANSFER:  (0.1, 0.2, 0.4, 0.3),

    # DATA: UP/DOWN -> Latency -> Reliability
    ServiceType.DATA:           (0.2, 0.1, 0.35, 0.35),

    # IOT: Reliability -> Latency -> UP/DOWN
    ServiceType.IOT:            (0.3, 0.4, 0.15, 0.15),
}

DEFAULT_WEIGHTS = (0.25, 0.25, 0.25, 0.25)




# NOTE: Using ServiceType from request.py
RelayProfiles = {
    ServiceType.VOICE:     {"cpu": 1,  "power": 2},
    ServiceType.VIDEO:     {"cpu": 5,  "power": 10},
    ServiceType.DATA:      {"cpu": 3,  "power": 6},
    ServiceType.IOT:       {"cpu": 1,  "power": 1},
    ServiceType.STREAMING: {"cpu": 7,  "power": 12},
    ServiceType.BULK_TRANSFER: {"cpu": 8,  "power": 15},
    ServiceType.CONTROL:   {"cpu": 2,  "power": 3},
    ServiceType.EMERGENCY: {"cpu": 4,  "power": 6},
}

QoSProfiles = {
    ServiceType.VOICE: {
        "uplink": (0.1, 0.3),
        "downlink": (0.2, 0.5),
        "latency": (20, 100),
        "reliability": (0.95, 0.99),
        "priority": (2, 4),
        "cpu": (1, 4),
        "power": (2, 6),
    },
    ServiceType.VIDEO: {
        "uplink": (1, 3),
        "downlink": (5, 10),
        "latency": (50, 150),
        "reliability": (0.90, 0.98),
        "priority": (3, 6),
        "cpu": (10, 30),
        "power": (20, 50),
    },
    ServiceType.DATA: {
        "uplink": (1, 5),
        "downlink": (5, 20),
        "latency": (50, 200),
        "reliability": (0.90, 0.97),
        "priority": (4, 7),
        "cpu": (5, 20),
        "power": (10, 40),
    },
    ServiceType.IOT: {
        "uplink": (0.05, 0.3),
        "downlink": (0.05, 0.2),
        "latency": (10, 100),
        "reliability": (0.97, 0.999),
        "priority": (2, 5),
        "cpu": (1, 3),
        "power": (1, 5),
    },
    ServiceType.STREAMING: {
        "uplink": (1, 3),
        "downlink": (8, 15),
        "latency": (50, 150),
        "reliability": (0.90, 0.97),
        "priority": (3, 6),
        "cpu": (15, 40),
        "power": (20, 60),
    },
    ServiceType.BULK_TRANSFER: {
        "uplink": (5, 20),
        "downlink": (20, 100),
        "latency": (100, 500),
        "reliability": (0.85, 0.95),
        "priority": (7, 10),
        "cpu": (20, 50),
        "power": (40, 80),
    },
    ServiceType.CONTROL: {
        "uplink": (0.1, 0.5),
        "downlink": (0.1, 0.5),
        "latency": (5, 50),
        "reliability": (0.99, 0.999),
        "priority": (1, 3),
        "cpu": (2, 6),
        "power": (5, 10),
    },
    ServiceType.EMERGENCY: {
        "uplink": (0.5, 2),
        "downlink": (0.5, 2),
        "latency": (1, 20),
        "reliability": (0.999, 1.0),
        "priority": (1, 1),
        "cpu": (5, 15),
        "power": (10, 20),
    },
}

# các mốc max để kiểm tra có thể truyền (Normalize)
UL_CAP, DL_CAP = 20.0, 100.0
CPU_CAP, PWR_CAP = 50.0, 80.0
LAT_CAP = 500.0
PRIO_CAP = 10.0





class SagsEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Setup logging properly
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('sagsenv.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Internal world state (hidden from agent)
        self.network = Network()
        self.num_nodes = len(self.network.nodes)
        self.connections = []  #List of all Requests currently being served and later we will release resource to keep the environemnt run endlessly
        self.groundspace = GroundSpace() #Store location of each request to quickly scan for nearby users

        # State = request-local view
        #Including: 
        #                                                                       INFO SERVICE REQUEST
        #Type of Service: a vector with 1 at the index of the service type, 0 elsewhere (8) (Vd: 1,0,0,0,0,0,0,0)
        #current hop / 10 (max 10 hops)
        #uplink required / max uplink (20 Mbps)
        #uplink allocated / uplink required normalized
        #downlink required / max downlink (100 Mbps)
        #downlink allocated / downlink required normalized
        #                                                                       INFO LOCATION & QoS
        #source location (lat, lon, alt) Normalized with formula : (sin(lat), cos(lat), sin(lon), cos(lon), norm_alt = alt/10000000) (clip if >= 1)
        #reliability required / max reliability (1.0)
        #current reliability / reliability required normalized
        #required latency / max latency (500 ms)
        #latency required / latency currently normalized
        #priority / max priority (10)
        #cpu required / max cpu (50)
        #power required / max power (100)
        #number of neighbors of current hop / max neighbors (10)
        #number of requests (users) in range of 2500km / 10000
        #timemout remaining / estmate timeout (timeout user need) need more consideration
        #                                                                       TOP 10 NODE NEAREST
        #top 10-nearest not passed nodes info (distance / (10000km), latency to node / max latency (500 ms), reliability to node / max reliability (1.0), 
        # uplink available / required uplink, downlink available/required downlink, cpu available / required cpu, 
        # power available / required power, gs_or_not, timeout/ user estimate timeout, numbers of user in range 2500 km / 10000, distance to nearest GS
        # , remark of neareast GS) * 10
        #neighbor will be (0)*12 if there is not enough neighbor
        self.obs_dim = 148
        self.observation_space = spaces.Box(
            low=0, high=1.0, 
            shape=(self.obs_dim,)
            , dtype=np.float32
        )

        self.action_space = spaces.Discrete(10)


        self.steps = 0
        self.current_request = None
        self.nodes_passed = []
        self.neighbor_ids = [None]*10 #Store the id of 10 nearest not passed nodes
        self.current_node = None #Store the current node
        self.node_passed_ids = set() #Store the id of nodes passed to quickly check if a node has been passed
        
        #Để rollBack
        self.alloc_ledger = {} # {req_id: [("relay", node, up_used, dn_used), ("gs", node, cpu_used, power_used), ...]}
    
    #Hard reset the environment, re-generate the network and groundspace
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset environment-level variables
        self.steps = 0
        self.neighbor_ids = [None] * 10
        self.current_node = None

        # Reset world components 
        #Dont need since we will continue to train in this world
        self.groundspace = GroundSpace()
        self.network = Network()
        self.num_nodes = len(self.network.nodes)

        # Extract node info
        self.nodes_passed.clear()
        self.node_passed_ids.clear()
        self.connections.clear()

        # Generate first request
        self._new_request()
        #nodes have been defined in network class
        #current node initally is none since we start from user
        self.current_node = None
        
        self.alloc_ledger.clear()


        # Return normalized initial observation
        obs = self._get_obs()
        return obs
    
    #Soft reset the environment, keep the network and groundspace
    def soft_reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset environment-level variables
        self.steps = 0
        self.neighbor_ids = [None] * 10
        self.current_node = None

        # Reset world components 
        #Dont need since we will continue to train in this world
        # self.groundspace = GroundSpace()
        # self.network = Network()

        # Extract node info
        self.nodes_passed.clear()
        self.node_passed_ids.clear()
        
        self.alloc_ledger.clear()

        # Generate first request
        self._new_request()
        #nodes have been defined in network class

    #Tạo request
    def _new_request(self):
        """Spawn a new client request"""
        client_location = random_user()
        service_type = random.choice(list(ServiceType))
        QoSProfiles_service = QoSProfiles[service_type]
        uplink_required = round(random.uniform(*QoSProfiles_service["uplink"]), 2)
        downlink_required = round(random.uniform(*QoSProfiles_service["downlink"]), 2)
        latency_required = round(random.uniform(*QoSProfiles_service["latency"]), 2)
        reliability_required = round(random.uniform(*QoSProfiles_service["reliability"]), 4)
        cpu_required = random.randint(*QoSProfiles_service["cpu"])
        power_required = random.randint(*QoSProfiles_service["power"])
        packet_size = random.randint(1, 100)  # MB
        demand_timeout = random.randint(10000, 45000)  # episode
        priority = random.randint(*QoSProfiles_service["priority"])
        new_request = request(
            request_id=self.connections.__len__() + 1,
            type=service_type,
            source_location=client_location,
            uplink_required=uplink_required,
            downlink_required=downlink_required,
            latency_required=latency_required,
            reliability_required=reliability_required,
            cpu_required=cpu_required,
            power_required=power_required,
            packet_size=packet_size,
            priority=priority,
            demand_timeout=demand_timeout
        )
        self.current_request = new_request


    # Mã hóa trạng thái môi trường cho Agent học
    def _get_obs(self):
        """Return request-local state vector"""
        Maximum_resource_usage = 0.9 #Maximum resource usage percentage of a node, to avoid overloading
        if self.current_request.type == ServiceType.EMERGENCY:
            Maximum_resource_usage = 0.95
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        service_index = self.current_request.type.value - 1
        obs[service_index] = 1.0  # One-hot encoding of service type
        obs[8] = len(self.nodes_passed) / 10.0  # Current hop / 10
        obs[9] = min(self.current_request.uplink_required / 20.0, 1.0)  # Uplink required / max uplink
        obs[10] = min((self.current_request.uplink_allocated / self.current_request.uplink_required
                if self.current_request.uplink_required > 0 else 0.0), 1.0)  # Uplink allocated / uplink required
        obs[11] = min(self.current_request.downlink_required / 100.0, 1.0)  # Downlink required / max downlink
        obs[12] = min((self.current_request.downlink_allocated / self.current_request.downlink_required
                if self.current_request.downlink_required > 0 else 0.0), 1.0)  # Downlink allocated / downlink required
        lat_rad = math.radians(self.current_request.source_location["lat"])
        lon_rad = math.radians(self.current_request.source_location["lon"])
        obs[13] = (math.sin(lat_rad) + 1) / 2.0  # Normalized sin(lat)
        obs[14] = (math.cos(lat_rad) + 1) / 2.0  # Normalized cos(lat)
        obs[15] = (math.sin(lon_rad) + 1) / 2.0  # Normalized sin(lon)
        obs[16] = (math.cos(lon_rad) + 1) / 2.0  # Normalized cos(lon)
        obs[17] = min(self.current_request.source_location["alt"] / 10000000.0, 1.0)  # Normalized altitude
        obs[18] = min(self.current_request.reliability_required / 1.0, 1.0)  # Reliability required / max reliability
        obs[19] = min((self.current_request.reliability_actual / self.current_request.reliability_required
                if self.current_request.reliability_required > 0 else 0.0), 1.0)  # Current reliability / reliability required
        obs[20] = min(self.current_request.latency_required / 500.0, 1.0)  # Required latency / max latency
        obs[21] = min((self.current_request.latency_required / self.current_request.latency_actual
                if self.current_request.latency_required > 0 else 0.0), 1.0)  # Latency required / latency currently
        #Lưu ý: Up/down + relibiliaty Càng cao càng tốt nhưng latency thì ngược lại
        # số hop càng ít càng tốt
        obs[22] = min(self.current_request.priority / 10.0, 1.0)  # Priority / max priority
        obs[23] = min(self.current_request.cpu_required / 50.0, 1.0)  # CPU required / max cpu
        obs[24] = min(self.current_request.power_required / 100.0, 1.0)  # Power required / max power
        connectable_nodes = []
        if self.current_node:
            connectable_nodes = self.network.find_connectable_nodes(self.current_node.id)
        else:
            connectable_nodes = self.network.find_connectable_nodes_for_location(
                self.current_request.source_location["lat"],
                self.current_request.source_location["lon"],
                self.current_request.source_location["alt"]
            )
        neighbors = []
        for node in connectable_nodes:
            if node.id not in self.node_passed_ids:
                neighbors.append(node)
        obs[25] = min(len(neighbors) / 10.0, 1.0)  # Number of neighbors / max neighbors
        users_in_range_count = self.groundspace.nearby_count(
            self.current_request.source_location["lat"],
            self.current_request.source_location["lon"],
            2500
        )
        obs[26] = min(users_in_range_count / 10000.0, 1.0)  # Number of users in range of 2500km / 10000
        obs[27] = min(self.current_request.real_timeout / self.current_request.demand_timeout
                if self.current_request.demand_timeout > 0 else 0.0, 1.0)  # Timeout remaining / estimated timeout
        for i in range(10):
            if i < len(neighbors):
                self.neighbor_ids[i] = neighbors[i].id
                current_location = self.current_node.position if self.current_node else self.current_request.source_location
                node = neighbors[i]
                self.neighbor_ids[i] = node.id
                distance = node.calculate_distance(
                    current_location["lat"],
                    current_location["lon"],
                    current_location["alt"]
                )
                obs[28 + i * 12] = min(distance / 1e7, 1.0)  # Distance / 10000km
                link_delay = calc_link_delay_ms(
                    distance,
                    self.current_node.typename if self.current_node else "user",
                    node.typename,
                    self.current_request.type
                )
                obs[29 + i * 12] = min(link_delay / 500.0, 1.0)  # Latency to node / max latency
                link_reliab = link_reliability(
                    self.current_node.typename if self.current_node else "user",
                    node.typename,
                    distance
                )
                obs[30 + i * 12] = min(link_reliab / 1.0, 1.0)  # Reliability to node / max reliability
                uplink_available = max(node.resources["uplink"]*Maximum_resource_usage - node.resources_used["uplink"], 0)
                obs[31 + i * 12] = min(uplink_available / self.current_request.uplink_allocated
                            if self.current_request.uplink_allocated > 0 else 1.0, 1.0)  # Uplink available / current uplink
                downlink_available = max(node.resources["downlink"]*Maximum_resource_usage - node.resources_used["downlink"], 0)
                obs[32 + i * 12] = min(downlink_available / self.current_request.downlink_allocated
                            if self.current_request.downlink_allocated > 0 else 1.0, 1.0)  # Downlink available / current downlink
                if node.typename != "groundstation":
                    obs[33 + i * 12] = 1.0 # CPU available / required cpu always 1 for relay nodes
                    obs[34 + i * 12] = 1.0 # Power available / required power always 1 for relay nodes
                else:
                    cpu_available = max(node.resources["cpu"]*Maximum_resource_usage - node.resources_used["cpu"], 0)
                    obs[33 + i * 12] = min(cpu_available / self.current_request.cpu_required
                                if self.current_request.cpu_required > 0 else 1.0, 1.0)  # CPU available / required cpu
                    power_available = max(node.resources["power"]*Maximum_resource_usage - node.resources_used["power"], 0)
                    obs[34 + i * 12] = min(power_available / self.current_request.power_required
                                if self.current_request.power_required > 0 else 1.0, 1.0)  # Power available / required power
                obs[35 + i * 12] = 1.0 if node.typename == "groundstation" else 0.0  # GS or not
                estimate_timeout = max_timeout = self.current_request.real_timeout
                if self.current_node:
                    if self.current_node.typename == "satellite" and node.typename == "satellite":
                        current_sat = self.network.get_satellite_by_id(self.current_node.id)
                        target_sat = self.network.get_satellite_by_id(node.id)
                        if current_sat and target_sat:
                            estimate_timeout = current_sat.estimate_visible_time_sat(target_sat, max_time=max_timeout)
                    elif self.current_node.typename == "satellite":
                        current_sat = self.network.get_satellite_by_id(self.current_node.id)
                        if current_sat:
                            estimate_timeout = current_sat.estimate_visible_time(node.position["lat"], node.position["lon"], node.position["alt"], max_time=max_timeout)
                obs[36 + i * 12] = min(estimate_timeout / self.current_request.real_timeout
                            if self.current_request.real_timeout > 0 else 1.0, 1.0)  # Timeout / user estimate timeout
                users_in_range_count = self.groundspace.nearby_count(
                    node.position["lat"],
                    node.position["lon"],
                    2500
                )
                obs[37 + i * 12] = min(users_in_range_count / 10000.0, 1.0)  # Numbers of user in range 2500 km / 10000
                gs_distance, gs_id = self.network.distance_to_nearest_gs(
                    node.position["lat"],
                    node.position["lon"],
                    node.position["alt"]
                )
                obs[38 + i * 12] = min(gs_distance / 1e7, 1.0) if gs_distance is not None else 1.0  # Distance to nearest GS
                if gs_id is not None:
                    gs = self.network.get_gs_by_id(gs_id)
                    initial_mark = 0
                    #Rate by uplink, downlink, cpu, power compared to the current allocated and required
                    gs_uplink_available = max(gs.resources["uplink"]*Maximum_resource_usage - gs.resources_used["uplink"], 0)
                    rate_uplink = gs_uplink_available / (self.current_request.uplink_allocated
                                        if self.current_request.uplink_allocated > 0 else self.current_request.uplink_required)
                    initial_mark += int(rate_uplink * 3)  # Up to 3
                    gs_downlink_available = max(gs.resources["downlink"]*Maximum_resource_usage - gs.resources_used["downlink"], 0)
                    rate_downlink = gs_downlink_available / (self.current_request.downlink_allocated
                                        if self.current_request.downlink_allocated > 0 else self.current_request.downlink_required)
                    initial_mark += int(rate_downlink * 3)  # Up to 3
                    gs_cpu_available = max(gs.resources["cpu"]*Maximum_resource_usage - gs.resources_used["cpu"], 0)
                    rate_cpu = gs_cpu_available / (self.current_request.cpu_required
                                        if self.current_request.cpu_required > 0 else 1)
                    initial_mark += int(rate_cpu * 2)  # Up to 2
                    gs_power_available = max(gs.resources["power"]*Maximum_resource_usage - gs.resources_used["power"], 0)
                    rate_power = gs_power_available / (self.current_request.power_required
                                        if self.current_request.power_required > 0 else 1)
                    initial_mark += int(rate_power * 2)  # Up to 2
                    obs[39 + i * 12] = min(initial_mark / 10.0, 1.0)  # Remark of nearest GS
                else:
                    obs[39 + i * 12] = 0.0  # No GS found, set remark to 0
            else:
                self.neighbor_ids[i] = None
                #pad with zeros
                obs[28 + i * 12: 40 + i * 12] = 0.0
        return obs

    
    RESERVE_RATIO = 0.10 # dành cho emergency 10%
    INVALID_ACTION_PENALTY = 0.1
    DONE_ON_MAX_STEPS = True
    INCLUDE_REMARK = False # True nếu đã remark neighbor nearest
    HOP_PENALTY = 0.01  # phạt mỗi bước -> thúc đẩy Agent đi ngắn hơn
    BASE_REWARD = 1.0



    #Cộng dồn tích lũy latency + reliably vào rq
    def _accumulate_link(self, hop_lat_ms, hop_rel, req):
        req.latency_actual += float(hop_lat_ms) #cộng dồn
        #reliably nhân dồn
        prev_rel = float(getattr(req, "reliability_actual", 1.0))
        req.reliability_actual = float(max(0.0, min(prev_rel * hop_rel, 1.0)))
    
    #kiểm tra GS
    def _is_gs(self, node_obj) -> bool:
        return str(getattr(node_obj, 'typename', "")).lower() == "groundstation"
    
    
        
    #Cơ chế phạt khi gần đạt 90% và thưởng dưới 50
    def _penalty_util_near_90_or_bonus_under_50(self, node_obj, req_obj) -> float:
        maximum_usage = 0.9
        if req_obj.type.name == "EMERGENCY":
            maximum_usage = 0.95
        
        total_upLink, total_downLink, total_cpu, total_power = node_obj.get_total_resources()
        free_upLink, free_downLink, free_cpu, free_power = node_obj.get_free_resources()
        
                    
        cpu_need = req_obj.cpu_required
        power_need = req_obj.power_required
        
        upLink_need = req_obj.uplink_required
        downLink_need = req_obj.downlink_requireds
        
        over_1, over_2 = 0.0, 0.0
        
        if node_obj.type != "GroundStation":
            
            #tỉ lệ dùng sau khi cấp đã normalize
            used_cpu_after = (total_cpu - (free_cpu - cpu_need)) / (total_cpu + 1e-6)
            used_power_after = (total_power - (free_power - power_need)) / (total_power + 1e-6)
            
            
            # Thưởng nhẹ nếu dưới 50%
            if used_cpu_after < 0.50 or used_power_after < 0.50:
                return 0.5
            
            #Phạt nặng nếu vượt 90%, phạt nhẹ nếu gần 90%
            over_1 = max(0.0, used_cpu_after - maximum_usage)
            over_2 = max(0.0, used_power_after - maximum_usage)
        
        else:
            used_upLink_after = (total_upLink - (free_upLink - upLink_need)) / (total_upLink + 1e-6)
            used_downLink_after = (total_downLink - (free_downLink - downLink_need)) / (total_downLink + 1e-6)

            # Thưởng nhẹ nếu dưới 50%s
            if used_upLink_after < 0.50 or used_downLink_after < 0.50:
                return 0.5

            #Phạt nặng nếu vượt 90%, phạt nhẹ nếu gần 90%
            over_1 = max(0.0, used_upLink_after - maximum_usage)
            over_2 = max(0.0, used_downLink_after - maximum_usage)
        
        #hệ số penalty
        penalty = 20.0
        return (- penalty * (over_1 ** 2 + over_2 ** 2))
    
    #thưởng nhẹ ở EMER
    def _emergency_bonus(self, node_obj, req_obj)->float:
        if req_obj.type.name == "EMERGENCY":
            return 0.05
        return 0.0
    
    
    
    #lấy free Công suất của node với policy
    def _free_resource(self, node_obj, req_obj)-> tuple:
        maximum_usage = 0.9
        if req_obj.type.name == "EMERGENCY":
            maximum_usage = 0.95
        
        upLink_free, downLink_free, cpu_free, power_free = node_obj.get_free_resources()
        
        #Dịch vụ thường
        if self._is_gs(node_obj) and req_obj.type.name != "EMERGENCY":
            _, _, cpu_total, power_total = node_obj.get_total_resources()
            
            reserve_cpu   = cpu_total  * maximum_usage
            reserve_power = power_total * maximum_usage

            cpu_free   = max(0.0, cpu_free   - reserve_cpu)
            power_free = max(0.0, power_free - reserve_power)
        
        return upLink_free, downLink_free, cpu_free, power_free

    #Tiêu tụ tài nguyên ở GS
    def _consume_gs_resource(self, gs_node, req):
        # free theo policy 90% (95% cho EMERGENCY)
        _, _, cpu_free, power_free = self._free_resource(gs_node, req)

        cpu_need, power_need = req.cpu_required, req.power_required
        cpu_used   = min(cpu_need,  cpu_free)
        power_used = min(power_need, power_free)

        gs_node.free_resources["cpu"]   -= cpu_used
        gs_node.free_resources["power"] -= power_used

        cpu_ratio   = (cpu_used   / cpu_need)   if cpu_need   > 0 else 1.0
        power_ratio = (power_used / power_need) if power_need > 0 else 1.0
        
        return cpu_used, power_used, cpu_ratio, power_ratio



    
    """
        Tính lượng khả dụng trên cả tuyến
        F = min( 1, tỉ lệ Up + down, tỉ lệ cpu + power)
        với allow partial = False -> F mới thành công
    """
    def commit_resources_for_path(self, req) -> bool:
        allocated = []  # [("relay", node, up_used, dn_used), ("gs", node, cpu_used, power_used)]

        #Xác định GS
        
        gs_node = None
        relays = []
        for node_id in req.path:
            n = self.network.get_node(node_id)
            if n is None:
                return False
            if self._is_gs(n):
                gs_node = n
            else:
                relays.append(n)
        if gs_node is None:
            return False

        # Tính tỉ lệ tắc nghẽn
        
        #Trên relay: theo uplink/downlink
        up_frac_min, dn_frac_min = float('inf'), float('inf')
        for rn in relays:
            up_free, dn_free, _, _ = rn.get_free_resources()
            up_frac_min = min(up_frac_min, up_free / max(req.uplink_required, 1e-9))
            dn_frac_min = min(dn_frac_min, dn_free / max(req.downlink_required, 1e-9))
        f_path = min(up_frac_min, dn_frac_min) if relays else 1.0

        #Trên GS: CPU/Power
        _, _, cpu_free, power_free = self._free_resource(gs_node, req)
        f_cpu   = cpu_free   / max(req.cpu_required, 1e-9)
        f_power = power_free / max(req.power_required, 1e-9)
        f_gs = min(f_cpu, f_power)

        #Đồng nhất
        f = max(0.0, min(1.0, f_path, f_gs))

        #Kh Partial
        if (not req.allow_partial) and (f < 1.0 - 1e-9):
            return False
        
        if f <= 0.0:
            return False

        #yêu cầu theo độ tắc nghẽn
        orig = (req.uplink_required, req.downlink_required, req.cpu_required, req.power_required)
        req.uplink_required   = f * orig[0]
        req.downlink_required = f * orig[1]
        req.cpu_required      = f * orig[2]
        req.power_required    = f * orig[3]

        try:
            #Trừ ở từng relay
            for rn in relays:
                up1, dn1, _, _ = rn.get_free_resources()
                check_ok = rn.allocate_resource(req, allow_partial=False)
                up2, dn2, _, _ = rn.get_free_resources()
                
                if not check_ok:
                    # rollback phần đã trừ rồi trả lại req gốc
                    for tag, node, *vals in reversed(allocated):
                        if tag == "relay":
                            node.free_resources["uplink"]  += vals[0]
                            node.free_resources["downlink"] += vals[1]
                    return False
                
                #used = phần đã cấp = x2 - x1
                allocated.append(("relay", rn, max(0.0, up1 - up2), max(0.0, dn1 - dn2)))

            #Trừ ở GS
            cpu_used, power_used, _, _ = self._consume_gs_resource(gs_node, req)
            allocated.append(("gs", gs_node, cpu_used, power_used))

            self.alloc_ledger[req.request_id] = allocated
            return True
        finally:
            #khôi phục required gốc
            req.uplink_required, req.downlink_required, req.cpu_required, req.power_required = orig



    
    #Cấp phát từng chiều nếu có thể cấp phát 1 phần
    def _calculate_partial_min(self, node_obj, req_obj):
        # Verify total resources first
        total_up, total_down, total_cpu, total_power = node_obj.get_total_resources()
        if any(x <= 0 for x in [total_up, total_down, total_cpu, total_power]):
            self.logger.error("Invalid total resources")
            return 0, 0, 0, 0
            
        # Then get free resources
        upLink_free, downLink_free, cpu_free, power_free = self._free_resource(node_obj, req_obj)
        
        upLink_need = req_obj.uplink_required
        downLink_need = req_obj.downlink_required
        cpu_need = req_obj.cpu_required
        power_need = req_obj.power_required
        
        upLink_alloc = min(upLink_free, upLink_need)
        downLink_alloc = min(downLink_free, downLink_need)
        cpu_alloc = min(cpu_free, cpu_need)
        power_alloc = min(power_free, power_need)
        
        return upLink_alloc, downLink_alloc, cpu_alloc, power_alloc
    
    
    
    INVALID_ACTION_PENALTY = 0.1
    MAX_STEP = 10
    HOP_PENALTY = 0.02  # phạt mỗi bước
    BASE_REWARD = 1.0
    
    
    #Tính reward
    """
        R_final = clip((R_base + R_QoS + R_resource + R_path + R_goal + B_extra) - P_hop, -1, 1)

    Phần thưởng cơ bản
        R_base = 0.1

    Phần thưởng Chất lượng dịch vụ (QoS) - 35%
        S_reliability = min(1, R_actual / R_required)
        S_latency = max(0, 1 - (L_actual / L_required))
        R_QoS = 0.35 * (0.6 * S_reliability + 0.4 * S_latency)

    Phần thưởng Hiệu quả tài nguyên - 25%
        R_resource = 0.25 * (1 - min(P_resource / R_base_const, 1))

    Phần thưởng Hiệu quả đường đi - 20%
        R_path = 0.20 * (1 - (N_steps / N_max_steps))

        Phần thưởng Hoàn thành mục tiêu - 20%
        R_goal:
        - 0.5 (nếu đến trạm mặt đất VÀ đạt QoS)
        - 0.2 (nếu đến trạm mặt đất nhưng KHÔNG đạt QoS)
        - 0 (nếu không đến trạm mặt đất)

    Thưởng thêm
        B_extra = (giá trị bonus được truyền vào)

    Phạt cho mỗi bước đi
        P_hop = 0.02 * N_steps
    """
    def _calculate_reward(self, obs, penalty: float, bonus: float) -> float:
        try:
            # Base reward cho việc di chuyển thành công
            
            # QoS component (35%)
            reliability_score  = obs[19]  # Current reliability / reliability required
            latency_score      = obs[21]  # Latency required / latency currently
            qos_reward = 0.35 * (0.6 * reliability_score + 0.4 * latency_score)
            
            # Resource efficiency (25%)
            resource_penalty = min(penalty / self.BASE_REWARD, 1.0)
            resource_reward = 0.25 * (1.0 - resource_penalty)
            
            # Path efficiency (20%)
            path_penalty = self.steps / self.MAX_STEP
            path_reward = 0.20 * (1.0 - path_penalty)
            
            # Goal achievement (20%)
            if self._is_gs(self.current_node):
                #Bonus lớn khi đến được GS
                goal_reward = 5.0
                #extra bonus nếu đạt QoS
                if (reliability_score >= 1.0 and latency_score >= 1.0):
                    goal_reward += 1.0
            else:
                goal_reward = 0
            
            # Sum
            total = self.BASE_REWARD + qos_reward + resource_reward + path_reward + goal_reward + bonus
            
            # Penalty cho mỗi hop -> khuyến khích đường ngắn
            hop_penalty = self.HOP_PENALTY * self.steps
            
            final_reward = total - hop_penalty
            
            return final_reward
        
        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            return -0.5

    # Extra bonus or penalty 
        """
            EMERGENCY = Latency -> Reliability -> UP/DOWN Link
            CONTROL = Latency -> Reliability -> UP/DOWN Link
            VOICE = Latency -> Reliability -> UP/DOWN Link
            
            VIDEO = UP/DOWN Link -> Rieliability -> Latency
            STREAMING = UP/DOWN Link -> Rieliability -> Latency
            DATA_TRANSFER = UP/DOWN Link -> Rieliability -> Latency
            
            DATA = UP/DOWN Link -> Latency -> Rieliability
            
            IOT = Reliability -> Latency -> UP/DOWN Link
        """
    def _extra_bonus_penalty(self, obs, lat, rel) -> float:
        reward = 0.0
        EXTRA_REWARD = 0.5
        
        # 1 khi tốt nhất
        lat_ratio = min((self.current_request.latency_required / lat), 1.0)
        rel_ratio = rel
        upLink_ratio = obs[10]
        downLink_ratio = obs[12]
        
        service = self.current_request.type
        
        #sum = 1.0
        w_lat, w_rel, w_up, w_down = 0.0, 0.0, 0.0, 0.0
        
        #Latency -> Reliability -> Number of Hops
        if service in (ServiceType.EMERGENCY, ServiceType.CONTROL, ServiceType.VOICE):
            w_lat, w_rel, w_up, w_down = BonusProfilesForService.get(service, DEFAULT_WEIGHTS)
            
        elif service in (ServiceType.VIDEO, ServiceType.STREAMING, ServiceType.DATA_TRANSFER):
            w_lat, w_rel, w_up, w_down = BonusProfilesForService.get(service, DEFAULT_WEIGHTS)
            
        elif service in ServiceType.DATA:
            w_lat, w_rel, w_up, w_down = BonusProfilesForService.get(service, DEFAULT_WEIGHTS)
        elif service in ServiceType.IOT:
            w_lat, w_rel, w_up, w_down = BonusProfilesForService.get(service, DEFAULT_WEIGHTS)
            
        reward += EXTRA_REWARD * ( 0.5 - 
                                    w_lat * (1 - lat_ratio) -
                                    w_rel * (1 - rel_ratio) -
                                    w_up * (1 - upLink_ratio) -
                                    w_down * (1 - downLink_ratio)
                                    )
        
        return reward
    
    
    
    #Mô phỏng bước đi của Agent
    def step(self, action):
        try: 
            done = False
            reward = 0.0
            
            req = self.current_request
            obs = self._get_obs()
            
            #K có request
            if req is None:
                self._new_request()
                self.logger.warning("No current request - creating new one")
                reward -= self.INVALID_ACTION_PENALTY
                return self._get_obs(), reward, done, {}
                

            #Lấy Neighbors
            neighbors = []
            for i in range(10):
                neighbor_id = self.neighbor_ids[i]
                if neighbor_id is not None and neighbor_id not in self.node_passed_ids:
                    neighbors.append(self.network.get_node(neighbor_id))

            #K có neighbors
            if len(neighbors) == 0:
                #kiểm tra GS -> Đích
                if self.current_node and self._is_gs(self.current_node):
                    reward += self._calculate_reward(obs, 0.0, 0.0)
                    
                    
                else:
                    self.logger.warning("No available neighbors to step to")
                    
                    reward -= self.INVALID_ACTION_PENALTY
                    done = True
                
                self._clear_episode()
                
                return self.soft_reset, reward, False, {}
            

            # Xử lý action
            try:
                a = int(action)
            except Exception:
                a = -1
                
            if a < 0 or a >= len(neighbors):
                #Action sai -> phạt
                reward -= self.INVALID_ACTION_PENALTY
                return self.soft_reset, reward, False, {}
            
            chosen = neighbors[a]

            hop_lat_norm = obs[29 + a*12]
            hop_rel = obs[30 + a*12]
            hop_lat_ms = hop_lat_norm * LAT_CAP
            
            lat_after = float(getattr(req, "latency_actual", 0.0)) + float(hop_lat_ms) #latency sau khi qua node
            rel_after = hop_rel #reliably sau khi qua node
            
            reward += self._extra_bonus_penalty(obs, lat_after, rel_after) #Thưởng/phạt thêm độ ưu tiên theo dịch vụ
            
            #Kiểm tra latency
            if lat_after > req.latency_required:
                reward -= 0.5 * (lat_after) / (req.latency_required)


            #Tính cấp phát từng chiều
            uplink = obs[31 + a * 12]
            downlink = obs[32 + a * 12]
            
            #kiểm tra up/down link
            if uplink < 0.5 or downlink < 0.5:
                reward -= 2.0 * self.BASE_REWARD * (2.0 - uplink - downlink)  # phạt nặng nếu k đáp ứng 50% required + Kết thúc
                
                return self.soft_reset(), reward, False, {}
            elif uplink < 1.0 or downlink < 1.0:
                reward -= self.BASE_REWARD * (2.0 - uplink - downlink) * 0.5

            #Cập nhật req
            self._accumulate_link(hop_lat_ms, hop_rel, req)

            req.uplink_allocated *= uplink
            req.downlink_allocated *= downlink
            #CPU va POWER khong can cap phat voi relay node vi khong dang ke
            #Chi cap phat cpu va power khi den GS
            

            #Update đường đi + vị trí hiện tại
            req.path.append(getattr(chosen, "id", getattr(chosen, "name", "node")))
            self.nodes_passed.append(chosen)
            self.node_passed_ids.add(getattr(chosen, "id", getattr(chosen, "name", None)))
            self.current_node = chosen
            self.steps += 1
            
            if self._is_gs(chosen):
                check_ok = self.commit_resources_for_path(req)
                if not check_ok:
                    # Kh đủ tài nguyên -> phạt + kết thúc
                    reward -= 2.0 * self.BASE_REWARD
                    done = True
                else:
                    # Lấy bản ghi GS để suy ra tỉ lệ (dựa trên cpu/power đã cấp)
                    gs_items = [t for t in self.alloc_ledger.get(req.request_id, []) if t[0] == "gs"]
                    if gs_items:
                        #số lượng đã cấp thực tế ở GS
                        _, gs_node, cpu_used, power_used = gs_items[-1]
                        
                        f_cpu = cpu_used  / max(1e-9, req.cpu_required)
                        f_pow = power_used/ max(1e-9, req.power_required)
                        
                        f = min(f_cpu, f_pow)
                        
                        if f < 0.5:
                            reward -= 2.0 * self.BASE_REWARD * (1.0 - f)
                        elif f < 1.0:
                            reward -= 0.5 * self.BASE_REWARD * (1.0 - f)

                    done = True
                    self.connections.append(req)


            #Thưởng hoặc phạt theo quy tắc
            penalty_or_bonus = self._penalty_util_near_90_or_bonus_under_50(chosen, req)
            emer_bonus = self._emergency_bonus(chosen, req) 

            reward += self._calculate_reward(obs, penalty_or_bonus, emer_bonus)

            # Kiểm tra QoS (rel + lat) và kết thúc (.)
            if (obs[19] >= 0.5 and obs[19] <= 1.0 and obs[21] >= 0.5 and obs[21] <= 1.0):
                reward += self.BASE_REWARD * (2.0 - obs[19] - obs[21])  # Thưởng nếu đạt QoS
            
            if self.steps >= self.MAX_STEP:
                done = True

            if done:
                #Giải phóng tài nguyên
                if req not in self.connections:
                    for node_id in req.path:
                        node = self.network.get_node(node_id)
                        if node:
                            self.E_release_resources(node, req)
                
                self._clear_episode()
                
            return self._get_obs(), reward, done, {}
        
        #Chu y chi ket thuc episode khi den GS hoac het lua chon hoac cham den hop_limit
        #Neu chua den thoi diem ket thuc episode thi chi cap nhat hop_count, current_node, path ,... va di tiep qua step tiep theo
        #Sau khi ket thuc episode thi duyet connection de cap nhat lai resource
        #Neu cham den GS thi them request hien tai vao connections

        except Exception as e:
            self.logger.error(f"Error in step: {e}")
            return self._get_obs(), -self.INVALID_ACTION_PENALTY, True, {"error": str(e)}


    
    #Xóa episode khi kết thúc
    def _clear_episode(self):
        try:
            if self.current_request and self.current_request.path:
                # Release allocated resources
                if self.current_request not in self.connections:
                    for node_id in self.current_request.path:
                        node = self.network.get_node(node_id)
                        if node:
                            self.E_release_resources(node, self.current_request)
                        
            # Reset state
            self.current_request = None
            self.current_node = None
            self.node_passed_ids.clear()
            self.neighbor_ids = []
            self.steps = 0
            
        except Exception as e:
            self.logger.error(f"Error in episode cleanup: {e}")
    
    #Phân bổ lại tài nguyên cho node
    def E_release_resources(self, node, request):
        try:
            if node and request:
                node.release_resource(request)           
        except Exception as e:
            self.logger.error(f"Error releasing resources: {e}")
    

def service_proc_delay_ms(node_type, service: ServiceType) -> float:
    #get the delay of the node type base on service type
    base = PROC_DELAY_BASE_MS[node_type]

    if service == ServiceType.EMERGENCY:
        return base * 0.5   # e.g. fast-path for emergency
    if service == ServiceType.CONTROL:
        return base * 0.7   # lower, but not as aggressive
    return base

def calc_link_delay_ms(distance_m: float,
                    node_a,
                    node_b,
                    service: ServiceType) -> float:
    """
    Delay (ms) for a single hop = propagation + avg proc delay.
    """
    # propagation delay (ms)
    prop_ms = (distance_m / C) * 1000.0

    # avg processing delay of both endpoints
    proc_ms = 0.5 * (service_proc_delay_ms(node_a, service) +
                    service_proc_delay_ms(node_b, service))

    return prop_ms + proc_ms
    #total path delay = sum of all link delays

def pick_gamma(a: str, b: str):
    key = (a,b)
    if key in GAMMA_PROFILE:
        return GAMMA_PROFILE[key]
    key_rev = (b,a)
    if key_rev in GAMMA_PROFILE:
        return GAMMA_PROFILE[key_rev]
    return GAMMA_PROFILE[("default","default")]

def link_reliability(type_a: str, type_b: str, distance_m: float) -> float:
    d_km = distance_m / 1000.0
    gamma = pick_gamma(type_a, type_b)
    return math.exp(-gamma * d_km)
    #total path reliability = product of all link reliabilities

def random_user():
    regions = [
        {"name": "China",          "latRange": (18, 54),   "lonRange": (73, 135),  "weight": 20},
        {"name": "India",          "latRange": (8, 37),    "lonRange": (68, 97),   "weight": 18},
        {"name": "Europe",         "latRange": (35, 60),   "lonRange": (-10, 40),  "weight": 15},
        {"name": "USA",            "latRange": (25, 50),   "lonRange": (-125, -66),"weight": 15},
        {"name": "Brazil",         "latRange": (-35, 5),   "lonRange": (-74, -34), "weight": 7},
        {"name": "Nigeria",        "latRange": (4, 14),    "lonRange": (3, 15),    "weight": 5},
        {"name": "Japan",          "latRange": (30, 45),   "lonRange": (129, 146), "weight": 5},
        {"name": "SoutheastAsia",  "latRange": (-10, 20),  "lonRange": (95, 120),  "weight": 5},
        {"name": "Other",          "latRange": (-90, 90),  "lonRange": (-180, 180),"weight": 10},
    ]

    total_weight = sum(r["weight"] for r in regions)
    rand = random.random() * total_weight

    # Weighted region selection
    selected_region = None
    for r in regions:
        if rand < r["weight"]:
            selected_region = r
            break
        rand -= r["weight"]

    # Generate latitude and longitude within range
    lat = round(random.uniform(*selected_region["latRange"]), 4)
    lon = round(random.uniform(*selected_region["lonRange"]), 4)
    alt = random.uniform(0, 2000)  # Altitude in meters

    # 60% chance of supporting 5G
    # support_5g = random.random() < 0.6
    support_5g = True

    return {"lat": lat, "lon": lon, "alt" : alt}