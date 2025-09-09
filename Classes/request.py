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
                 bandwidth_required, latency_required, reliability_required,
                 packet_size, direct_sat_support = False, priority=10):
        self.request_id = request_id
        self.type = type
        self.source_location = source_location

        # QoS
        self.bandwidth_required = bandwidth_required
        self.latency_required = latency_required
        self.reliability_required = reliability_required
        self.priority = priority

        # Data info
        self.packet_size = packet_size

        self.direct_sat_support = direct_sat_support  # thiết bị có thể kết nối trực tiếp vệ tinh hay không
        self.demand_timeout = 0
        self.real_timeout = 0