from enum import Enum

class ServiceType(Enum):
    VOICE = 1
    VIDEO = 2
    DATA = 3
    IOT = 4
    STREAMING = 5
    BULK_TRANSFER = 6
    CONTROL = 7
    EMERGENCY = 8

class request:
    def __init__(self, request_id, type: ServiceType, source_location,
                 uplink_required, downlink_required, latency_required, reliability_required, cpu_required, power_required,
                 packet_size, direct_sat_support = False, priority=10, demand_timeout=0, allow_partial=True):
        self.request_id = request_id
        self.type = type
        self.source_location = source_location


        #Resource
        self.cpu_required = cpu_required
        self.power_required = power_required

        self.cpu_allocated = 0
        self.power_allocated = 0

        # QoS
        self.uplink_required = uplink_required
        self.downlink_required = downlink_required
        self.latency_required = latency_required
        self.reliability_required = reliability_required
        self.priority = priority

        self.uplink_allocated = 0
        self.downlink_allocated = 0
        self.latency_actual = None
        self.reliability_actual = None

        # Data info
        self.packet_size = packet_size

        self.direct_sat_support = direct_sat_support  # thiết bị có thể kết nối trực tiếp vệ tinh hay không
        self.demand_timeout = demand_timeout
        self.real_timeout = 0
        self.allow_partial = allow_partial  # cho phép cấp phát băng thông một phần hay không
        self.path = [] # lưu đường đi đã chọn