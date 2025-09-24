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
from request import Request
from GroundSpace import GroundSpace
from enum import Enum, auto
import math

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
        #neighbor will be (1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1) if there is not enough neighbor
        self.obs_dim = 148
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(10)


        self.steps = 0
        self.current_request = None
        self.neighbor_ids = [None]*10 #Store the id of 10 nearest not passed nodes
        self.current_node = None #Store the current node

    def reset(self):
        self.current_node = None
        self.neighbor_ids = [None]*10
        self.current_node = None
        self.connections.clear()
        self.steps = 0
        self.groundspace = GroundSpace()
        self._new_request()
        return self._get_obs()

    def _new_request(self):
        """Spawn a new client request"""
        src = np.random.randint(0, self.num_nodes)
        dst = np.random.randint(0, self.num_nodes)
        while dst == src:
            dst = np.random.randint(0, self.num_nodes)

        req = {
            "src": src,
            "dst": dst,
            "bw_req": np.random.uniform(0.05, 0.2),
            "cpu_req": np.random.uniform(0.05, 0.2),
            "power_req": np.random.uniform(0.05, 0.2),
            "ttl": np.random.randint(50, 200)
        }
        self.current_request = req

    def _get_obs(self):
        """Return request-local state vector"""
        r = self.current_request
        src_feat = self.nodes[r["src"]]
        dst_feat = self.nodes[r["dst"]]
        return np.concatenate([
            src_feat,     # 3 features
            dst_feat,     # 3 features
            [r["bw_req"], r["cpu_req"], r["power_req"], r["ttl"]/200.0, r["src"]/self.num_nodes, r["dst"]/self.num_nodes]
        ]).astype(np.float32)

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

