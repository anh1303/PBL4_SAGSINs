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
from request import request
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
    "groundstation": 10  # GS fast-path ~10 ms (but can be reduced for emergency)
}

class ServiceType(Enum):
    VOICE = auto()
    VIDEO = auto()
    DATA = auto()
    IOT = auto()
    STREAMING = auto()
    BULK_TRANSFER = auto()
    CONTROL = auto()
    EMERGENCY = auto()

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




class SagsEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Internal world state (hidden from agent)
        self.network = Network()
        self.connections = []  #List of all Requests currently being served and later we will release resource to keep the environemnt run endlessly
        self.groundspace = GroundSpace() #Store location of each request to quickly scan for nearby users

        # State = request-local view
        #Including: 
        #Type of Service: a vector with 1 at the index of the service type, 0 elsewhere (8) (Vd: 1,0,0,0,0,0,0,0)
        #current hop / 10 (max 10 hops)
        #uplink required / max uplink (20 Mbps)
        #uplink allocated / uplink required normalized
        #downlink required / max downlink (100 Mbps)
        #downlink allocated / downlink required normalized
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
        #top 10-nearest not passed nodes info (distance / (10000km), latency to node / max latency (500 ms), reliability to node / max reliability (1.0), uplink available / required uplink, downlink available/required downlink, cpu available / required cpu, 
        # power available / required power, gs_or_not, timeout/ user estimate timeout, numbers of user in range 2500 km / 10000, distance to nearest GS
        # , remark of neareast GS) * 10
        #neighbor will be (0)*12 if there is not enough neighbor
        self.count = 0
        self.obs_dim = 148
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(10)


        self.steps = 0
        self.current_request = None
        self.nodes_passed = []
        self.neighbor_ids = [None]*10 #Store the id of 10 nearest not passed nodes
        self.current_node = None #Store the current node
        self.node_passed_ids = set() #Store the id of nodes passed to quickly check if a node has been passed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset environment-level variables
        self.steps = 0
        self.connections.clear()
        self.neighbor_ids = [None] * 10
        self.current_node = None

        # Reset world components 
        #Dont need since we will continue to train in this world
        # self.groundspace = GroundSpace()
        # self.network = Network()

        # Extract node info
        self.nodes_passed.clear()
        self.node_passed_ids.clear()
        self.num_nodes = len(self.nodes)

        # Generate first request
        self._new_request()
        #nodes have been defined in network class
        #current node initally is none since we start from user
        self.current_node = None

        # Return normalized initial observation
        obs = self._get_obs()
        return obs


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
        demand_timeout = random.randint(5000, 40000)  # steps
        priority = random.randint(*QoSProfiles_service["priority"])
        new_request = request(
            request_id=self.count,
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
        self.count += 1
        

    def _get_obs(self):
        """Return request-local state vector"""
        Maximum_resource_usage = 0.9 #Maximum resource usage percentage of a node, to avoid overloading
        if self.current_request.type == ServiceType.Emergency:
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
                            estimate_timeout = current_sat.estimate_visible_time_gs(node, max_time=max_timeout)
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
                    
                    
                
        
            

    #Khanh's work here, ChatGPT recommended adn then modified
    #Note that you should add some priority profile for some Qos criteria later
    #Also, you should consider the timeout of each request and release resource when timeout
    #Make sure to consider the stability and up/down link separately
    def step(self, action):
        self.steps += 1
        reward = 0

        node_choice, bw_alloc, cpu_alloc, power_alloc = action
        node_choice = int(node_choice * (self.num_nodes-1))

        r = self.current_request
        # Check resources
        if (self.nodes[node_choice,0] >= bw_alloc and
            self.nodes[node_choice,1] >= cpu_alloc and
            self.nodes[node_choice,2] >= power_alloc):

            # allocate
            self.nodes[node_choice,0] -= bw_alloc
            self.nodes[node_choice,1] -= cpu_alloc
            self.nodes[node_choice,2] -= power_alloc

            # register new connection
            self.connections.append([r["src"], r["dst"], [node_choice], r["ttl"]])
            reward += 1.0
        else:
            reward -= 1.0

        # Tick connections (simulate timeout)
        for c in self.connections:
            c[3] -= 1
        expired = [c for c in self.connections if c[3] <= 0]
        for c in expired:
            # Free resources (simplified)
            node = c[2][0]
            self.nodes[node,0] = min(1.0, self.nodes[node,0] + r["bw_req"])
            self.nodes[node,1] = min(1.0, self.nodes[node,1] + r["cpu_req"])
            self.nodes[node,2] = min(1.0, self.nodes[node,2] + r["power_req"])
            self.connections.remove(c)

        # Next request
        self._new_request()
        return self._get_obs(), reward, False, {}
    
    
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