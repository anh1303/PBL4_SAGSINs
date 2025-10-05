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
from request import request as Request, ServiceType
from GroundSpace import GroundSpace
from enum import Enum, auto
import math
import logging

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
        #neighbor will be (1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1) if there is not enough neighbor
        self.obs_dim = 148
        self.observation_space = spaces.Box(
            low=0, high=1.0, 
            shape=(self.obs_dim,)
            , dtype=np.float32
        )

        self.action_space = spaces.Discrete(10)


        self.steps = 0
        self.current_request = None
        self.neighbor_ids = [None]*10 #Store the id of 10 nearest not passed nodes
        self.current_node = None #Store the current node
        self.visited_ids = set()  # track visited node to prevent loops

        self.max_steps = 30 #số bước có thể đi tối đa

        #Giám sát mectrics
        self.metrics = {
            'episodes_completed': 0,
            'avg_steps_per_episode': [],
            'success_rate': [],
            'avg_reward': [],
            'resource_utilization': []
        }

    #reset environment -> initial state
    def reset(self):
        try:
            # Clear old episode
            self._clear_episode()
            
            # Reset metrics
            self.metrics['episodes_completed'] += 1
            
            # Create new request
            if not self._new_request():
                raise RuntimeError("Failed to create new request")
                
            return self._get_obs()
            
        except Exception as e:
            self.logger.error(f"Error in reset: {e}")
            return np.zeros(self.obs_dim, dtype=np.float32)

    #chọn 1 node đích (destination) bất kì
    def _pick_ramdom_groundstation(self):
        gs_list = [n for n in getattr(self.network, 'nodes', []) if isinstance(n, Gs)]
        return gs_list[np.random.randint(len(gs_list))]

    #Tạo request
    def _new_request(self):
        try:
            # Random loại dịch vụ
            svc = np.random.choice(list(ServiceType))
            prof = QoSProfiles[svc]

            # Lấy ngẫu nhiên trong profile
            ul = float(np.random.uniform(*prof["uplink"]))      # Mbps
            dl = float(np.random.uniform(*prof["downlink"]))    # Mbps
            lat = float(np.random.uniform(*prof["latency"]))    # ms
            rel = float(np.random.uniform(*prof["reliability"]))# 0..1
            cpu = float(np.random.uniform(*prof["cpu"]))
            pwr = float(np.random.uniform(*prof["power"]))
            pri = int(np.random.randint(*prof["priority"]))

            # Vị trí user (random toàn cầu)
            src_lat = float(np.random.uniform(-60.0, 60.0))
            src_lon = float(np.random.uniform(-180.0, 180.0))
            src_alt = 0.0

            # Khởi tạo request object
            rid = int(np.random.randint(1_000_000, 9_999_999))
            self.current_request = Request(
                request_id=rid,
                type=svc,
                source_location={"lat": src_lat, "lon": src_lon, "alt": src_alt},
                uplink_required=ul,
                downlink_required=dl,
                latency_required=lat,
                reliability_required=rel,
                cpu_required=cpu,
                power_required=pwr,
                packet_size=int(np.random.randint(512, 4096)),  # bytes
                direct_sat_support=False,
                priority=pri,
                demand_timeout=int(np.random.randint(50, 200)),
                allow_partial=True,
            )
            # reset actual
            self.current_request.uplink_allocated = 0.0
            self.current_request.downlink_allocated = 0.0
            self.current_request.latency_actual = 0.0
            self.current_request.reliability_actual = 1.0
            self.current_request.real_timeout = 0.0
            self.current_request.path = []
            return True
        except Exception as e:
            self.logger.error(f"Error creating new request: {e}")
            return False
        


    # Mã hóa trạng thái môi trường cho Agent học
    def _get_obs(self):
        try:
            if self.current_request is None:
                self._new_request()
                
            obs = self._calculate_obs(self.current_request)
            # Nếu kích thước khác(do thay đổi schema), cập nhật space
            if obs.shape[0] != self.observation_space.shape[0]:
                self.obs_dim = obs.shape[0]
                self.observation_space = spaces.Box(
                    low=0.0, high=1.0, 
                    shape=(self.obs_dim,),
                    dtype=np.float32
                    )
            return obs
        except Exception as e:
            print(f"Error in _get_obs: {e}")
            return np.zeros(self.obs_dim, dtype=np.float32)
    
    #mã hóa vector Service: toàn bộ 0, 1 với dịch vụ tương ứng
    def _vector_service(self, svc: ServiceType):
        vec = np.zeros(8, dtype=np.float32)
        index = int(svc.value) - 1
        if 0 <= index < 8:
            vec[index] = 1.0
        return vec
    
    #chuẩn hóa
    def _normalize(self, x, cap):
        if not isinstance(x, (int, float)) or not isinstance(cap, (int, float)):
            self.logger.error("Input values must be numeric")
            raise ValueError("Input values must be numeric")
        if cap <= 0:
            self.logger.error("Cap must be positive")
            raise ValueError("Cap must be positive")
        return float(max(0.0, min(1.0, x/(cap + 1e-6))))
    
    #Tỉ lệ đáp ứng
    def _safe_ratio(self, num, den):
        # num/den hoặc den xấp xỉ 0 => 0
        return float(0.0 if den <= 1e-6 else max(0.0, min(1.0, num/den)))
    
    # Mã hoá 5 mức: 0=unknown, 1=too_far, 2=unreliable, 3=high_latency, 4=ok
    def _nearest_gs_remark_neighbors(self, node_obj, dis_gs_m: float) -> float:
        if dis_gs_m is None:
            code = 0
        else:
            type_a = str(getattr(node_obj, 'typename', '')).lower()
            # Ước tính reliability & latency từ node -> GS gần nhất
            rel = link_reliability(type_a, 'groundstation', dis_gs_m)
            lat = calc_link_delay_ms(dis_gs_m, type_a, 'groundstation', ServiceType.EMERGENCY)  #Mọi loại Server đều đúng
            if dis_gs_m > 7_500_000:        # > 7,500 km coi là "too far"
                code = 1
            elif rel < 0.70:              # kém tin cậy
                code = 2
            elif lat > 250:               # độ trễ cao
                code = 3
            else:
                code = 4                  # tốt
        return float(code / 4.0)

    
    """Trả về List Neighbor
        Mỗi neighbor có 12 trường: [dist_norm, lat_norm, rel_norm,
                                ul_avail_over_need, dl_avail_over_need, cpu_avail_over_need, pwr_avail_over_need,
                                is_gs, timeout_ratio, users_nearby_norm, dist_to_nearest_gs_norm, mark_nearest_gs]
                                """
    def _neighbors_block(self, cur_lat, cur_lon, cur_alt, req):
        lat = cur_lat
        lon = cur_lon
        alt = cur_alt

        neighs = self._pick_top10_neighbors(lat, lon, alt, support5G=True)
        num_neighbors_normalize = self._safe_ratio(len(neighs), 10.0)
        
        block = []
        
        for n in neighs:
            src_type = "groundstation" if self.current_node is None else str(getattr(self.current_node, 'typename', 'groundstation')).lower()
            node_type = str(getattr(n, 'typename', "")).lower()
            
            #distance từ user-> mỗi neighbors
            dis_m = n.calculate_distance(lat, lon, alt, mode="3d")
            dis_normalize = self._normalize(dis_m, 10_000_000.0)
            
            #độ trễ và reliability 1 hop (user ground -> neighbors type)
            lat_ms = calc_link_delay_ms(dis_m, src_type, node_type, req.type)
            lat_normalize = self._normalize(lat_ms, LAT_CAP)
            
            rel_ms = link_reliability(src_type, node_type, dis_m)
            rel_normalize = float(max(0.0, min(rel_ms, 1.0)))
            
            # Khả dụng tài nguyên so với nhu cầu: >=1 -> đủ (clip về 1)
            upLink_free, downLink_free, cpu_free, power_free = self._free_resource(n,req)
            upLink_ratio = self._safe_ratio(upLink_free, req.uplink_required)
            downLink_ratio = self._safe_ratio(downLink_free, req.downlink_required)
            cpu_ratio = self._safe_ratio(cpu_free, req.cpu_required)
            power_ratio = self._safe_ratio(power_free, req.power_required)
            
            #check GS
            is_gs = 1.0 if self._is_gs(n) else 0.0
            
            #timeout của request
            timeout_est = max(1.0, float(req.demand_timeout))
            timeout_left = 1.0 #các neighbor không giảm TO
            
            #User trong 2500km: chưa có -> 0
            users_nearby_norm = 0.0
            
            #Distance từ node này đến GS gần nhất
            dis_gs_nearest = self.network.distance_to_nearest_gs(n.position["lat"], n.position["lon"], n.position.get("alt", 0.0))
            dis_gs_nearest_norm = self._normalize(0.0 if dis_gs_nearest is None else dis_gs_nearest, 10_000_000.0)
            
            #remark GS nearest
            remark_gs_nearest = self._nearest_gs_remark_neighbors(n, dis_gs_nearest)
            
            block.extend([
                dis_normalize, lat_normalize, rel_normalize,
                upLink_ratio, downLink_ratio, cpu_ratio, power_ratio,
                is_gs, timeout_left, users_nearby_norm, dis_gs_nearest_norm, remark_gs_nearest
            ])
        
        #Tự thêm nếu n<10
        if len(neighs) < 10:
            pad_count = 10 - len(neighs)
            for _ in range(pad_count):
                # [dist, lat, rel, ul, dl, cpu, pwr, is_gs, timeout, users, dist_to_gs, remark]
                block.extend([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        
        return block, num_neighbors_normalize


    #Phần cốt lõi của vector quan sát(OBS) chỉ phụ thuộc vào yêu cầu dịch vụ hiện tại (req)
    #Nó phản ánh trạng thái thực tế ở lần quan sát tiếp theo -> agent xem đã đáp ứng % để điều chỉnh
    def _base_block(self, req):
        """
        Phần BASE của vector quan sát(không tính top-10 láng giềng)
        Gồm:
        - Vector_service: toàn bộ 0, 1 với dịch vụ tương ứng (8)
        - current_hop_norm
        - UL_req_norm, UL_alloc_over_req
        - DL_req_norm, DL_alloc_over_req
        - location (sin(lat), cos(lat), sin(lon), cos(lon), alt_norm)
        - reliability_req_norm, reliability_actual_over_req
        - latency_req_norm, latency_actual_over_req
        - priority_norm
        - cpu_req_norm, power_req_norm
        - num_neighbors_norm (sẽ điền sau)
        - users_in_2500km_norm (chưa có -> 0)
        - timeout_remaining_over_est
        """
        vector_sv = self._vector_service(req.type)
        
        #số hop đã đi
        cur_hop = len(req.path)
        cur_hop_normalize = self._normalize(cur_hop, 10.0)
        
        #up/down Link
        upLink_req_normalize = self._normalize(req.uplink_required, UL_CAP)
        upLink_alloc_over_req = self._safe_ratio(getattr(req, "uplink_allocated", 0.0), max(req.uplink_required, 1e-9))
        downLink_req_normalize = self._normalize(req.downlink_required, DL_CAP)
        downLink_alloc_over_req = self._safe_ratio(getattr(req, "downlink_allocated", 0.0), max(req.downlink_required, 1e-9))
        
        
        #location
        lat = math.radians(req.source_location["lat"])
        lon = math.radians(req.source_location["lon"])
        alt = req.source_location.get("alt", 0.0)
        loc_vec = [
            math.sin(lat), math.cos(lat),
            math.sin(lon), math.cos(lon),
            self._normalize(alt, 10_000_000.0)
            ]
        
        #reliablity / latency
        rel_req = float(req.reliability_required)
        rel_req_normalize = float(max(0.0, min(rel_req, 1.0)))
        rel_actual = float(getattr(req, "reliability_actual", 0.0))
        rel_actual_over_req = self._safe_ratio(rel_actual, max(rel_req, 1e-9))
        
        lat_req_normalize = self._normalize(req.latency_required, LAT_CAP)
        lat_actual = float(getattr(req, "latency_actual", 0.0))
        lat_actual_over_req = self._safe_ratio(lat_actual, max(req.latency_required, 1e-9))
        
        #priority
        priority_normalize = self._safe_ratio(float(req.priority), PRIO_CAP)
        
        #CPU / POWER
        cpu_req_normalize = self._normalize(req.cpu_required, CPU_CAP)
        power_req_normalize = self._normalize(req.power_required, PWR_CAP)
        
        #user trong 2500km -> 0 (chưa có)
        users_in_2500km_norm = 0.0
        
        #timeout
        est_timeout = max(1.0, float(req.demand_timeout))
        remain_timeout = max(0.0, est_timeout - float(getattr(req, "real_timeout", 0.0)))
        ratio_timeout = self._safe_ratio(remain_timeout, est_timeout)
        
        base = np.concatenate([
            vector_sv,
            np.array([
                cur_hop_normalize,
                upLink_req_normalize, upLink_alloc_over_req,
                downLink_req_normalize, downLink_alloc_over_req,
                ], dtype=np.float32),
            np.array(loc_vec, dtype = np.float32),
            np.array([
                rel_req_normalize, rel_actual_over_req,
                lat_req_normalize, lat_actual_over_req,
                priority_normalize,
                cpu_req_normalize, power_req_normalize
                ], dtype = np.float32),
            np.array([
                users_in_2500km_norm,
                #num_neighbors_normalize, -> thêm sau 
                ratio_timeout], dtype=np.float32),
        ])
        
        return base
    
    #Tính chiều dài observation động theo schema
    #Nếu thay đổi Schema thì hàm tự tính lại
    def _calculate_obs(self, req):
        base = self._base_block(req)
        if self.current_node is None:
            cur_lat = req.source_location["lat"]
            cur_lon = req.source_location["lon"]
            cur_alt = req.source_location.get("alt", 0.0)
        else:
            cur_lat = self.current_node.position["lat"]
            cur_lon = self.current_node.position["lon"]
            cur_alt = self.current_node.position.get("alt", 0.0)
        neigh_block, num_neighbors_normalize = self._neighbors_block(cur_lat, cur_lon, cur_alt, req)
        
        #chèn num_neighbors_normalize
        obs = np.concatenate([
            base,
            np.array([num_neighbors_normalize], dtype=np.float32),
            np.array(neigh_block, dtype=np.float32)
            ])
        
        return obs.astype(np.float32)
    
    # Tính chiều dài vector động (để set observation_space an toàn)
    def _calc_obs_len(self):
        # base: vector_service(8) + 5 (hop+UL/DL) + 5 loc + 7 (rel/lat/pri/cpu/pwr) + 2 (users, timeout)
        base_len = 8 + 5 + 5 + 7 + 2
        # +1 cho num_neighbors_norm
        PER_NEI = 12  # [dist, lat, rel, ul, dl, cpu, pwr, is_gs, timeout, users, dist_to_gs, remark]
        return base_len + 1 + (10 * PER_NEI)
    
    RESERVE_RATIO = 0.10 # dành cho emergency 10%
    INVALID_ACTION_PENALTY = 0.1
    DONE_ON_MAX_STEPS = True
    INCLUDE_REMARK = False # True nếu đã remark neighbor nearest
    HOP_PENALTY = 0.01  # phạt mỗi bước -> thúc đẩy Agent đi ngắn hơn
    BASE_REWARD = 1.0

    
    def _hop_metrics_from_source(self, destination_node, req):
        #metrics từ user(ground) -> destination node
        lat = req.source_location["lat"]
        lon = req.source_location["lon"]
        alt = req.source_location.get("alt", 0.0)
        
        distence_m = destination_node.calculate_distance(lat, lon, alt, mode="3d")
        destination_type = str(getattr(destination_node, 'typename', "")).lower()
        
        hot_lat_ms = calc_link_delay_ms(distence_m, "groundstation", destination_type, req.type)
        hot_rel = link_reliability("groundstation", destination_type, distence_m)
        
        return distence_m, hot_lat_ms, hot_rel
    
    def _hop_metrics_from_node(self, src_node, dst_node, req):
        #metrics từ src node -> dst node
        s = src_node.position
        distence_m = dst_node.calculate_distance(s["lat"], s["lon"], s.get("alt", 0.0), mode="3d")
        
        src_type = str(getattr(src_node, 'typename', "")).lower()
        dst_type = str(getattr(dst_node, 'typename', "")).lower()

        hop_lat_ms = calc_link_delay_ms(distence_m, src_type, dst_type, req.type)
        hop_rel = link_reliability(src_type, dst_type, distence_m)
        
        return distence_m, hop_lat_ms, hop_rel

    #Cộng dồn tích lũy latency + reliably vào rq
    def _accumulate_link(self, hop_lat_ms, hop_rel, req):
        req.latency_actual += float(hop_lat_ms) #cộng dồn
        #reliably nhân dồn
        prev_rel = float(getattr(req, "reliability_actual", 1.0))
        req.reliability_actual = float(max(0.0, min(prev_rel * hop_rel, 1.0)))
    
    #kiểm tra GS
    def _is_gs(self, node_obj) -> bool:
        return str(getattr(node_obj, 'typename', "")).lower() == "groundstation"
    
    
    def _get_totals(self, node_obj):
        #tổng công suất src gốc
        total_uplink  = float(node_obj.resources.get("uplink",   0.0))
        total_downlink= float(node_obj.resources.get("downlink", 0.0))
        total_cpu     = float(node_obj.resources.get("cpu",      0.0))
        total_power   = float(node_obj.resources.get("power",    0.0))
        
        return total_uplink, total_downlink, total_cpu, total_power
        
    def _get_free(self, node_obj):
        uplink_free   = float(node_obj.free_resources.get("uplink",   0.0))
        downlink_free = float(node_obj.free_resources.get("downlink", 0.0))
        cpu_free      = float(node_obj.free_resources.get("cpu",      0.0))
        power_free    = float(node_obj.free_resources.get("power",    0.0))
        
        return uplink_free, downlink_free, cpu_free, power_free
    
    #Theo quy luật
    def _respect_reserve(self, node_obj, req_obj) -> bool:
        #moniter metrics
        total_resources = self._get_totals(node_obj)
        free_resources = self._get_free(node_obj)
        utilization = {
            k: 1 - (free_resources[k] / total_resources[k]) 
            for k in total_resources
        }
        
        # Track utilization + giới hạn append
        
        self.metrics['resource_utilization'].append(utilization)
        if len(self.metrics['resource_utilization']) > 1000:
            self.metrics['resource_utilization'] = self.metrics['resource_utilization'][-100:]
        
        #GS để lại 10% cho EMER và EMER có thể vượt quá 10%, node khác free là đủ
        _, _, cpu_free, power_free = self._get_free(node_obj)
        _, _, cpu_total, power_total = self._get_totals(node_obj)

        cpu_need = req_obj.cpu_required
        power_need = req_obj.power_required
        
        #Các node khác thì cho phép
        if not self._is_gs(node_obj):
            return (cpu_free >= cpu_need and power_free >= power_need)
        
        #Dịch vụ EMER
        if req_obj.type.name == "EMERGENCY":
            return (cpu_free >= cpu_need and power_free >= power_need)
        
        #Dịch vụ thường -> cho sự dụng < 90%
        resver_cpu = cpu_total * self.RESERVE_RATIO
        resver_power = power_total * self.RESERVE_RATIO
        
        cpu_left = cpu_free - cpu_need
        power_left = power_free - power_need
        
        return (cpu_left >= resver_cpu and power_left >= resver_power)
        
    #Cơ chế phạt khi gần đạt 90%
    def _penalty_util_near_90(self, node_obj, req_obj) -> float:
        if not self._is_gs(node_obj) or req_obj.type.name == "EMERGENCY":
            return 0.0
        
        total_upLink, total_downLink, total_cpu, total_power = self._get_totals(node_obj)
        free_upLink, free_downLink, free_cpu, free_power = self._get_free(node_obj)
        cpu_need = req_obj.cpu_required
        power_need = req_obj.power_required
        
        #tỉ lệ dùng sau khi cấp(0..1)
        used_cpu_after = (total_cpu - (free_cpu - cpu_need)) / (total_cpu + 1e-6)
        used_power_after = (total_power - (free_power - power_need)) / (total_power + 1e-6)
        
        # <= 90% -> =0
        over_cpu = max(0.0, used_cpu_after - 0.90)
        over_power = max(0.0, used_power_after - 0.90)
        
        #hệ số penalty
        penalty = 20.0
        return penalty * (over_cpu ** 2 + over_power ** 2)
    
    def _emergency_bonus(self, node_obj, req_obj)->float:
        #thưởng nhẹ ở GS + EMER
        if not self._is_gs(node_obj) or req_obj.type.name != "EMERGENCY":
            return 0.0
        return 0.5
    
    def _pick_top10_neighbors(self, lat, lon, alt = 0.0, support5G = True):
        # All nodes visited
        if len(self.visited_ids) >= self.num_nodes:
            return []
        
        candidates = self.network.find_connectable_nodes_for_location(lat, lon, alt, support5G = support5G)

        #loại các node đã qua
        filtered = [n for n in candidates if getattr(n, 'id', getattr(n, 'name', None)) not in self.visited_ids]

        #Sort theo Dis
        filtered.sort(key=lambda n: n.calculate_distance(lat, lon, alt, mode="3d"))

        #Lưu id
        self.neighbor_ids = [getattr(n, 'id', getattr(n, 'name', 'node')) for n in filtered[:10]] + [None] * (10 - len(filtered[:10]))
        return filtered[:10]
    
    
    #lấy free Công suất của node với policy 10%
    def _free_resource(self, node_obj, req_obj)-> tuple:
        upLink_free, downLink_free, cpu_free, power_free = self._get_free(node_obj)
        
        #Dịch vụ thường
        if self._is_gs(node_obj) and req_obj.type.name != "EMERGENCY":
            _, _, cpu_total, power_total = self._get_totals(node_obj)
            
            reserve_cpu   = cpu_total  * self.RESERVE_RATIO
            reserve_power = power_total* self.RESERVE_RATIO

            cpu_free   = max(0.0, cpu_free   - reserve_cpu)
            power_free = max(0.0, power_free - reserve_power)
        
        return upLink_free, downLink_free, cpu_free, power_free
    
    #Cấp phát từng chiều nếu có thể cấp phát 1 phần
    def _calculate_partial_min(self, node_obj, req_obj):
        # Verify total resources first
        total_up, total_down, total_cpu, total_power = self._get_totals(node_obj)
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
    
    
    
    
    #Mô phỏng bước đi của Agent
    def step(self, action):
        try:
            #Kiểm tra timeout
            if self._check_timeout(self.current_request):
                self.logger.info(f"Request {self.current_request.request_id} timed out")
                return self._get_obs(), -self.BASE_REWARD, True, {"blocked": "timeout"}
                
            
            self.current_request.real_timeout += 1
            
            done = False
            reward = 0.0
            
            req = self.current_request
            
            #K có request
            if req is None:
                self._new_request()
                self.logger.warning("No current request - creating new one")
                reward -= self.INVALID_ACTION_PENALTY
                return self._get_obs(), reward, done, {}
                

            #Lấy Neighbors
            if self.current_node is None:
                lat = req.source_location["lat"]
                lon = req.source_location["lon"]
                alt = req.source_location.get("alt", 0.0)
            else:
                lat = self.current_node.position["lat"]
                lon = self.current_node.position["lon"]
                alt = self.current_node.position.get("alt", 0.0)

            neighbors = self._pick_top10_neighbors(lat, lon, alt, support5G=True)

            #K có neighbors
            avail = len(neighbors)
            if avail == 0:
                reward -= self.INVALID_ACTION_PENALTY
                return self._get_obs(), reward, True, {}

            # Xử lý action
            try:
                a = int(action)
            except Exception:
                a = -1
            if a < 0 or a >= avail:
                #Action sai -> phạt
                reward -= self.INVALID_ACTION_PENALTY
                return self._get_obs(), reward, False, {}

            chosen = neighbors[a]

            # Tính metrics hop
            if self.current_node is None or not req.path:
                _, hop_lat_ms, hop_rel = self._hop_metrics_from_source(chosen, req)
            else:
                _, hop_lat_ms, hop_rel = self._hop_metrics_from_node(self.current_node, chosen, req)
            
            lat_after = float(getattr(req, "latency_actual", 0.0)) + float(hop_lat_ms) #latency sau khi qua node
            
            #Kiểm tra latency
            if lat_after > req.latency_required:
                reward -= 0.5
                return self._get_obs(), reward, False, {"blocked": "latency"}

            #Kiểm tra quy luật
            if not self._respect_reserve(chosen, req):
                reward -= self.BASE_REWARD
                return self._get_obs(), reward, False, {"blocked": "reserve"}

            #Tính cấp phát từng chiều
            up_alloc, down_alloc, cpu_alloc, pwr_alloc = self._calculate_partial_min(chosen, req)
            #kiểm tra up/down link
            if up_alloc + 1e-6 < req.uplink_required or down_alloc + 1e-6 < req.downlink_required:
                reward -= self.BASE_REWARD
                return self._get_obs(), reward, False, {"blocked": "bandwidth"}

            #Cập nhật req
            self._accumulate_link(hop_lat_ms, hop_rel, req)

            req.uplink_allocated = up_alloc
            req.downlink_allocated = down_alloc
            req.cpu_allocated = cpu_alloc
            req.power_allocated = pwr_alloc
            

            #Cấp phát
            check = chosen.allocate_resource(req, allow_partial = req.allow_partial)
            if not check:
                reward -= self.BASE_REWARD
                return self._get_obs(), reward, False, {}

            #Update đường đi + vị trí hiện tại
            req.path.append(getattr(chosen, "id", getattr(chosen, "name", "node")))
            #Mark node 
            chosen_id = getattr(chosen, "id", getattr(chosen, "name", None))
            
            if chosen_id is not None:
                self.visited_ids.add(chosen_id)
                
            self.current_node = chosen
            reward -= self.HOP_PENALTY
            
            if self._is_gs(chosen):
                reward = self._calculate_reward(chosen, req, 0, 0)
                return self._get_obs(), reward, True, {}

            penalty = self._penalty_util_near_90(chosen, req)
            emer_bonus = self._emergency_bonus(chosen, req) 

            reward = self._calculate_reward(chosen, req, penalty, emer_bonus)

            # Kiểm tra QoS và kết thúc
            qos_ok = (req.reliability_actual >= req.reliability_required) and (req.latency_actual <= req.latency_required)
            if qos_ok:
                reward += 0.3
                done = True
            
            self.steps += 1
            if self.DONE_ON_MAX_STEPS and self.steps >= self.max_steps:
                done = True


            return self._get_obs(), reward, done, {}

        except Exception as e:
            self.logger.error(f"Error in step: {e}")
            return self._get_obs(), -self.INVALID_ACTION_PENALTY, True, {"error": str(e)}


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
    def _calculate_reward(self, chosen_node, req, penalty: float, bonus: float) -> float:
        try:
            # Base reward cho việc di chuyển thành công
            base_reward = 0.1
            
            # QoS component (35%)
            reliability_score = min(1.0, req.reliability_actual / req.reliability_required)
            latency_score = max(0, 1.0 - (req.latency_actual / req.latency_required))
            qos_reward = 0.35 * (0.6 * reliability_score + 0.4 * latency_score)
            
            # Resource efficiency (25%)
            resource_penalty = min(penalty / self.BASE_REWARD, 1.0)
            resource_reward = 0.25 * (1.0 - resource_penalty)
            
            # Path efficiency (20%)
            path_penalty = self.steps / self.max_steps
            path_reward = 0.20 * (1.0 - path_penalty)
            
            # Goal achievement (20%)
            if self._is_gs(chosen_node):
                #Bonus lớn khi đến được GS
                goal_reward = 0.20
                #extra bonus nếu đạt QoS
                if (req.reliability_actual >= req.reliability_required and 
                    req.latency_actual <= req.latency_required):
                    goal_reward += 0.3
            else:
                goal_reward = 0
            
            # Sum
            total = base_reward + qos_reward + resource_reward + path_reward + goal_reward + bonus
            
            # Penalty cho mỗi hop -> khuyến khích đường ngắn
            hop_penalty = 0.02 * self.steps
            
            final_reward = total - hop_penalty
            
            return float(np.clip(final_reward, -1.0, 1.0))
        
        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            return -0.5
    
    #Xóa episode khi kết thúc
    def _clear_episode(self):
        try:
            if self.current_request and self.current_request.path:
                # Release allocated resources
                for node_id in self.current_request.path:
                    node = self.network.get_node(node_id)
                    if node:
                        self._release_resources(node, self.current_request)
                        
                # Update metrics
                success = (self.current_request.reliability_actual >= self.current_request.reliability_required and 
                            self.current_request.latency_actual <= self.current_request.latency_required)
                self.metrics['success_rate'].append(1.0 if success else 0.0)
                self.metrics['avg_steps_per_episode'].append(self.steps)
                
            # Reset state
            self.current_request = None
            self.current_node = None
            self.visited_ids.clear()
            self.neighbor_ids = [None] * 10
            self.steps = 0
            
        except Exception as e:
            self.logger.error(f"Error in episode cleanup: {e}")
    
    #Phân bổ lại tài nguyên cho node
    def _release_resources(self, node, request):
        try:
            if not node or not request:
                return
                
            # Return allocated resources
            node.free_resources["uplink"] += request.uplink_allocated
            node.free_resources["downlink"] += request.downlink_allocated 
            node.free_resources["cpu"] += request.cpu_allocated
            node.free_resources["power"] += request.power_allocated
            
            # Remove connection record
            node.connections = [c for c in node.connections if c["request_id"] != request.request_id]
            
        except Exception as e:
            self.logger.error(f"Error releasing resources: {e}")
    
    #Check timeout với demand
    #Chưa hoàn thiện nha, chưa lấy biết lấy ở đâu, 
    def _check_timeout(self, req) -> bool:
        try:
            if req.demand_timeout <= 0:
                return False
                
            timeout = float(req.demand_timeout)
            elapsed = float(getattr(req, "real_timeout", 0))
            
            return elapsed >= timeout
            
        except Exception as e:
            self.logger.error(f"Error checking timeout: {e}")
            return True

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

