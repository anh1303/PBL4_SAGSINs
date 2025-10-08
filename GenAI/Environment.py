from datetime import datetime
import logging
import gymnasium as gym
from gymnasium import spaces
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
from node import node
from enum import Enum, auto
import math
import random

#rel_link = exp(-gamma * (d_km))

C = 3e8

RESERVE_RATIO = 0.10 # dành cho emergency 10%
INVALID_ACTION_PENALTY = -60
DEAD_END_PENALTY = -100.0  # phạt khi không thể tiếp tục
DONE_ON_MAX_STEPS = True
INCLUDE_REMARK = False # True nếu đã remark neighbor nearest
MAX_STEP = 10
STEP_LIMIT_PENALTY = -50.0  # phạt khi vượt quá số bước tối đa
BASE_REWARD = 5
HOP_PENALTY = 5  # phạt mỗi bước
usage_pool = 5
QOS_pool = 48
timeout_pool = 7
finished_pool = 30

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
        self.MAX_STEP = MAX_STEP
        self.path_log_file = f"paths_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.step_log_file = f"steps_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(self.path_log_file, "w") as f:
            f.write("=== SAGs Environment Path Log ===\n\n")
        # with open(self.step_log_file, "w") as f:
        #     f.write("=== SAGs Environment Step Log ===\n\n")

        # Internal world state (hidden from agent)
        self.network = Network()
        self.num_nodes = len(self.network.nodes)
        self.connections = []  #List of all Requests currently being served and later we will release resource to keep the environemnt run endlessly
        self.groundspace = GroundSpace() #Store location of each request to quickly scan for nearby users
        self.obs_dim = 159
        self.observation_space = spaces.Box(
            low=0, high=1.0, 
            shape=(self.obs_dim,)
            , dtype=np.float32
        )

        self.action_space = spaces.Discrete(10)


        self.steps = 0
        self.current_request = None #Store the current request
        self.neighbor_ids = [None]*10 #Store the id of 10 nearest not passed nodes
        self.current_node : node = None #Store the current node
        self.node_passed_ids = [] #Store the id of nodes passed to quickly check if a node has been passed
        
        self.waiting_queue = [] #Store requests waiting for resource
        
        #Để rollBack
        # self.alloc_ledger = {} # {req_id: [("relay", node, up_used, dn_used), ("gs", node, cpu_used, power_used), ...]}
        self._log_reward = 0
        self._log_score = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.current_node == None or self.current_request == None:
            #Initialize environemnt
            self.network = Network()
            self.groundspace = GroundSpace()
            self.connections = []
            self.steps = 0
            self.neighbor_ids = [None]*10
            self.current_node = None
            self.node_passed_ids = []
        else:
            #Reset for new request
            #Release all resource of current request
            #add current request to connections
            #Check if successful:
            if self.current_node.typename == "groundstation":
                self.connections.append(self.current_request)
                self.groundspace.add_request(self.current_request)
                for id in self.node_passed_ids:
                    node = self.network.get_node_by_id(id)
                    if node:
                        node.allocate_resource(self.current_request)
                self.log_path()
        #Check for timeout of each request in connections
        timeout_req = []
        for index, request in enumerate(self.connections):
            request.real_timeout -= 1
            if request.real_timeout <= 0:
                timeout_req.append(index)
        for index in reversed(timeout_req):
            req = self.connections.pop(index)
            self.groundspace.remove_request(req)
            for id in req.path:
                node = self.network.get_node_by_id(id)
                if node:
                    node.release_resource(req)
                    # self.log_remove(node, req)
                
        self.current_node = None
        self.node_passed_ids = []
        self.current_request = None
        self.neighbor_ids = [None]*10
        self._log_reward = 0
        self.steps = 0
        self._log_score = 0
        # if len(self.waiting_queue) == 0:
        #     self._new_request()
        # else:
        #     self.current_request = self.waiting_queue.pop(0)
        self._new_request()
        obs = self._get_obs()
        return obs, {}
    
    def log_info(self, obs):
        #log current observation, current request, all environemnt variables
        with open(self.step_log_file, "a") as f:
            f.write(f"=== New Step ===\n")
            f.write(f"Current Request: {self.current_request}\n")
            f.write(f"Step: {self.steps}\n")
            f.write(f"Current Node: {self.current_node.id if self.current_node else 'None'}\n")
            f.write(f"Node Passed: {self.node_passed_ids}\n")
            f.write(f"Neighbor IDs: {self.neighbor_ids}\n")
            # f.write(f"Observation: {obs}\n")
            f.write(f"Connections: {[req.request_id for req in self.connections]}\n")
            f.write(f"Waiting Queue: {[req.id for req in self.waiting_queue]}\n")
            #Log the raw obs vector
            f.write("*"*60 + "\n\n")
        
                
                

    #Tạo request
    def _new_request(self):
        """Spawn a new client request"""
        client_location = random_user()
        while (not self.network.check_neighbor_exist(
            client_location["lat"],
            client_location["lon"],
            client_location["alt"]
        )):
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
        demand_timeout = random.randint(100, 1000)  # episode
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
        obs[8] = self.steps / 10.0  # Current hop / 10
        obs[9] = min(self.current_request.uplink_required / 20.0, 1.0)  # Uplink required / max uplink
        obs[10] = min((self.current_request.uplink_allocated / self.current_request.uplink_required
                if self.current_request.uplink_required > 0 else 0.0), 1.0)  # Uplink allocated / uplink required
        obs[11] = min(self.current_request.downlink_required / 100.0, 1.0)  # Downlink required / max downlink
        obs[12] = min((self.current_request.downlink_allocated / self.current_request.downlink_required
                if self.current_request.downlink_required > 0 else 0.0), 1.0)  # Downlink allocated / downlink required
        if not self.current_node:
            current_lat = self.current_request.source_location["lat"]
            current_lon = self.current_request.source_location["lon"]
            current_alt = self.current_request.source_location["alt"]
        else:
            current_lat = self.current_node.position["lat"]
            current_lon = self.current_node.position["lon"]
            current_alt = self.current_node.position["alt"]
        lat_rad = math.radians(current_lat)
        lon_rad = math.radians(current_lon)
        obs[13] = (math.sin(lat_rad) + 1) / 2.0  # Normalized sin(lat)
        obs[14] = (math.cos(lat_rad) + 1) / 2.0  # Normalized cos(lat)
        obs[15] = (math.sin(lon_rad) + 1) / 2.0  # Normalized sin(lon)
        obs[16] = (math.cos(lon_rad) + 1) / 2.0  # Normalized cos(lon)
        obs[17] = min(current_alt / 10000000.0, 1.0)  # Normalized altitude
        obs[18] = min(self.current_request.reliability_required / 1.0, 1.0)  # Reliability required / max reliability
        obs[19] = min((self.current_request.reliability_actual / self.current_request.reliability_required
                if self.current_request.reliability_required > 0 else 0.0), 1.0)  # Current reliability / reliability required
        obs[20] = min(self.current_request.latency_required / 500.0, 1.0)  # Required latency / max latency
        obs[21] = min((self.current_request.latency_required / self.current_request.latency_actual
                if self.current_request.latency_actual > 0 else 1.0), 1.0)  # Latency required / latency currently
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
        self.neighbor_ids = [None]*10
        count = 0
        for node in connectable_nodes:
            if node.id not in self.node_passed_ids:
                self.neighbor_ids[count] = node.id
                count += 1
            if count >= 10:
                break
        obs[25] = min(count / 10.0, 1.0)  # Number of neighbors / max neighbors
        users_in_range_count = self.groundspace.nearby_count(
            self.current_request.source_location["lat"],
            self.current_request.source_location["lon"],
            2500
        )
        obs[26] = min(users_in_range_count / 10000.0, 1.0)  # Number of users in range of 2500km / 10000
        obs[27] = min(self.current_request.real_timeout / self.current_request.demand_timeout
                if self.current_request.demand_timeout > 0 else 0.0, 1.0)  # Timeout remaining / estimated timeout
         # Top 10 nearest nodes info
        for i in range(10):
            if i < count:
                current_location = self.current_node.position if self.current_node else self.current_request.source_location
                node = self.network.get_node_by_id(self.neighbor_ids[i])
                self.neighbor_ids[i] = node.id
                distance = node.calculate_distance(
                    current_location["lat"],
                    current_location["lon"],
                    current_location["alt"]
                )
                obs[28 + i * 13] = min(distance / 1e7, 1.0)  # Distance / 10000km
                current_node_type = self.current_node.typename if self.current_node else "user"
                if current_node_type == "satellite":
                    current_node_type = self.current_node.type
                node_type = node.typename
                if node_type == "satellite":
                    node_type = node.type
                link_delay = calc_link_delay_ms(
                    distance,
                    current_node_type,
                    node_type,
                    self.current_request.type
                )
                obs[29 + i * 13] = min(link_delay / 500.0, 1.0)  # Latency to node / max latency
                link_reliab = link_reliability(
                    current_node_type,
                    node_type,
                    distance
                )
                obs[30 + i * 13] = min(link_reliab / 1.0, 1.0)  # Reliability to node / max reliability
                uplink_available = max(node.resources["uplink"]*Maximum_resource_usage - node.resources_used["uplink"], 0)
                obs[31 + i * 13] = min(uplink_available / self.current_request.uplink_allocated
                            if self.current_request.uplink_allocated > 0 else 1.0, 1.0)  # Uplink available / current uplink
                downlink_available = max(node.resources["downlink"]*Maximum_resource_usage - node.resources_used["downlink"], 0)
                obs[32 + i * 13] = min(downlink_available / self.current_request.downlink_allocated
                            if self.current_request.downlink_allocated > 0 else 1.0, 1.0)  # Downlink available / current downlink
                if node.typename != "groundstation":
                    obs[33 + i * 13] = 1.0 # CPU available / required cpu always 1 for relay nodes
                    obs[34 + i * 13] = 1.0 # Power available / required power always 1 for relay nodes
                else:
                    cpu_available = max(node.resources["cpu"]*Maximum_resource_usage - node.resources_used["cpu"], 0)
                    obs[33 + i * 13] = min(cpu_available / self.current_request.cpu_required
                                if self.current_request.cpu_required > 0 else 1.0, 1.0)  # CPU available / required cpu
                    power_available = max(node.resources["power"]*Maximum_resource_usage - node.resources_used["power"], 0)
                    obs[34 + i * 13] = min(power_available / self.current_request.power_required
                                if self.current_request.power_required > 0 else 1.0, 1.0)  # Power available / required power
                obs[35 + i * 13] = 1.0 if node.typename == "groundstation" else 0.0  # GS or not
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
                obs[36 + i * 13] = min(estimate_timeout / self.current_request.real_timeout
                            if self.current_request.real_timeout > 0 else 1.0, 1.0)  # Timeout / user estimate timeout
                users_in_range_count = self.groundspace.nearby_count(
                    node.position["lat"],
                    node.position["lon"],
                    2500
                )
                obs[37 + i * 13] = min(users_in_range_count / 10000.0, 1.0)  # Numbers of user in range 2500 km / 10000
                gs_distance, gs_id = self.network.distance_to_nearest_gs(
                    node.position["lat"],
                    node.position["lon"],
                    node.position["alt"]
                )
                obs[38 + i * 13] = min(gs_distance / 1e7, 1.0) if gs_distance is not None else 1.0  # Distance to nearest GS
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
                    obs[39 + i * 13] = min(initial_mark / 10.0, 1.0)  # Remark of nearest GS
                else:
                    obs[39 + i * 13] = 0.0  # No GS found, set remark to 0
                obs[40+i*13] = min(node.get_mean_usage(), 1.0) # Percent resource usage
            else:
                self.neighbor_ids[i] = None
                #pad with zeros
                obs[28 + i * 13: 40 + i * 13] = 0.0
                
            obs[158] = 0
            if self.current_node:
                obs[158] = 1.0 if self.current_node.typename == "groundstation" else 0.0  # GS or not
        return obs
    
    
    #Tính reward
    def calculate_efficent_usage_bonus(self):
        usage = self.current_node.get_mean_usage() if self.current_node else 0
        if usage < 0.6:
            return usage_pool
        else:
            return usage_pool*(1-usage)/(1-0.6)
        
    def calculate_hop_reward(self):
        return HOP_PENALTY - self.steps
    
    def calculate_base_reward(self):
        return BASE_REWARD * (1 - self.steps / MAX_STEP)
    
    def _calculate_reward(self, obs):
        reward = 0.0
        reward += self.calculate_base_reward()
        reward += self.calculate_efficent_usage_bonus()
        reward += self.calculate_hop_reward()
        #Calculate timeout reward
        timeout_ratio = obs[27]  # Timeout remaining / estimated timeout
        reward += timeout_pool * timeout_ratio
        #Calculate QOS reward
        #get weights
        weights = BonusProfilesForService.get(self.current_request.type, (0.25, 0.25, 0.25, 0.25))
        w_lat, w_rel, w_up, w_down = weights
        #Latency
        lat_ratio = obs[21]  # Latency required / latency currently
        reward += QOS_pool * w_lat * lat_ratio
        #Reliability
        rel_ratio = obs[19]  # Current reliability / reliability required
        reward += QOS_pool * w_rel * rel_ratio
        #Uplink
        up_ratio = obs[10]  # Uplink allocated / uplink required
        reward += QOS_pool * w_up * up_ratio
        #Downlink
        down_ratio = obs[12]  # Downlink allocated / downlink required
        reward += QOS_pool * w_down * down_ratio
        if obs[-1] == 1:
            #Reached groundstation
            reward += finished_pool/2
            cpu_bonus = self.current_request.cpu_allocated / self.current_request.cpu_required if self.current_request.cpu_required > 0 else 1.0
            cpu_bonus *= finished_pool/4
            power_bonus = self.current_request.power_allocated / self.current_request.power_required if self.current_request.power_required > 0 else 1.0
            power_bonus *= finished_pool/4
            reward += cpu_bonus + power_bonus
        if self.neighbor_ids[0] is None and obs[-1] == 0:
            #No where to go
            reward += DEAD_END_PENALTY
        if self.steps > MAX_STEP:
            reward += STEP_LIMIT_PENALTY
        reward /= 50.0  # Normalize reward
        return reward
        
    
    def _apply_action(self, action):
        maximum_resource_usage = 0.9
        if self.current_request.type == ServiceType.EMERGENCY:
            maximum_resource_usage = 0.95
        limit = 0
        for i in range(10):
            if self.neighbor_ids[i]:
                limit += 1
            else:
                break
        if action < 0 or action >= limit:
            return False
        next_node_id = self.neighbor_ids[action]
        next_node = self.network.get_node_by_id(next_node_id)
        if next_node is None:
            return False
        if self.current_node is not None:
            current_lat = self.current_node.position["lat"]
            current_lon = self.current_node.position["lon"]
            current_alt = self.current_node.position["alt"]
        else:
            current_lat = self.current_request.source_location["lat"]
            current_lon = self.current_request.source_location["lon"]
            current_alt = self.current_request.source_location["alt"]
        distance = next_node.calculate_distance(
            current_lat,
            current_lon,
            current_alt
        )
        current_node_type = self.current_node.typename if self.current_node else "user"
        if current_node_type == "satellite":
            current_node_type = self.current_node.type
        node_type = next_node.typename
        if node_type == "satellite":
            node_type = next_node.type
        link_delay = calc_link_delay_ms(
            distance,
            current_node_type,
            node_type,
            self.current_request.type
        )
        link_reliab = link_reliability(
            current_node_type,
            node_type,
            distance
        )
        self.current_request.path.append(next_node.id)
        self.node_passed_ids.append(next_node.id)
        self.current_request.latency_actual += link_delay
        self.current_request.reliability_actual *= link_reliab
        avail_uplink = max(next_node.resources["uplink"] * maximum_resource_usage - next_node.resources_used["uplink"], 0)
        avail_downlink = max(next_node.resources["downlink"] * maximum_resource_usage - next_node.resources_used["downlink"], 0)
        uplink_alloc = min(avail_uplink, self.current_request.uplink_allocated)
        downlink_alloc = min(avail_downlink, self.current_request.downlink_allocated)
        self.current_request.uplink_allocated = uplink_alloc
        self.current_request.downlink_allocated = downlink_alloc
        next_node.resources_used["uplink"] += uplink_alloc
        next_node.resources_used["downlink"] += downlink_alloc
        if next_node.typename == "groundstation":
            avail_cpu = max(next_node.resources["cpu"] * maximum_resource_usage - next_node.resources_used["cpu"], 0)
            avail_power = max(next_node.resources["power"] * maximum_resource_usage - next_node.resources_used["power"], 0)
            cpu_alloc = min(avail_cpu, self.current_request.cpu_required)
            power_alloc = min(avail_power, self.current_request.power_required)
            self.current_request.cpu_allocated = cpu_alloc
            self.current_request.power_allocated = power_alloc
            next_node.resources_used["cpu"] += cpu_alloc
            next_node.resources_used["power"] += power_alloc
        self.current_node = next_node
        self.steps += 1
        return True
    
    def step(self, action):
        state = self._apply_action(action)
        obs = self._get_obs()  # Get the current observation
        # self.log_info(obs)
        reward = self._calculate_reward(obs)
        if state == False:
            #Hành động không hợp lệ
            reward = INVALID_ACTION_PENALTY
            return obs, reward/100, False, False, {} #This is an invalid action, the apply action return False mean it didnt apply it so just reward and keep current state to try another action
        if self.steps > self.MAX_STEP and obs[-1] == 0:
            #Quá số bước
            reward = STEP_LIMIT_PENALTY
            return obs, reward, True, False, {"score" : 0}
        if self.neighbor_ids[0] is None and obs[-1] == 0:
            #Dead end
            return obs, reward, True, False, {"score" : 0}
        done = True if obs[-1] == 1 else False
        if self.current_node:
            if self.current_node.typename == "groundstation":
                done = True
        score = rel = lat = up = down = time = 0
        if done:
            self._log_reward = reward
            #Calculate score base on QOS:
            score = 0
            #get weights
            weights = BonusProfilesForService.get(self.current_request.type, (0.25, 0.25, 0.25, 0.25))
            w_lat, w_rel, w_up, w_down = weights
            #Latency
            lat_ratio = obs[21]  # Latency required / latency currently
            lat = 10 * w_lat * lat_ratio
            score += lat
            #Reliability
            rel_ratio = obs[19]  # Current reliability / reliability required
            rel = 10 * w_rel * rel_ratio
            score += rel
            #Uplink
            up_ratio = obs[10]  # Uplink allocated / uplink required
            up = 10 * w_up * up_ratio
            score += up
            #Downlink
            down_ratio = obs[12]  # Downlink allocated / downlink required
            down = 10 * w_down * down_ratio
            score += down
            #Timeout
            timeout_ratio = obs[27]  # Timeout remaining / estimated timeout
            time = 2 * timeout_ratio
            score += time
            score = round(score, 2)
            self._log_score = score
        return obs, reward, done, False, {"score" : score, "latency": lat, "reliability" : rel, "uplink": up, "downlink" : down, "timeout" : time}
    
    def log_path(self, success = True):
        """Write the path of the finished request to the path log."""
        if not self.current_request:
            return

        # Build readable path (IDs or names)
        path_names = []
        for node_id in self.node_passed_ids:
            path_names.append(node_id)

        with open(self.path_log_file, "a", encoding="utf-8") as f:
            f.write(f"[Request {self.current_request.request_id}] Type={self.current_request.type.name}\n")
            f.write(f"Path: {' → '.join(path_names) if path_names else 'No path'}\n")
            f.write(f"Total hops: {len(path_names)} | Reward = {self._log_reward}  | Score = {self._log_score} | Success={success}\n")
            f.write("-" * 60 + "\n")
            
    def log_remove(self, node: node, request: request):
        """Write the node have been released from request to the path log."""
        if not self.current_request:
            return

        # Build readable path (IDs or names)
        path_names = []
        for node_id in self.node_passed_ids:
            node = self.network.get_node_by_id(node_id)
            if node:
                path_names.append(node.id)
        path_names.reverse()  # Reverse to show from start to end
        
        # Add current node if not already
        if self.current_node and self.current_node.id not in self.node_passed_ids:
            path_names.append(self.current_node.id)

        with open(self.path_log_file, "a", encoding="utf-8") as f:
            f.write(f"[Request {request.request_id}] Released from Node {node.id}\n")
            f.write(f"Current Path: {' → '.join(path_names) if path_names else 'No path'}\n")
            f.write("-" * 60 + "\n")
            
        
        
        
    

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


