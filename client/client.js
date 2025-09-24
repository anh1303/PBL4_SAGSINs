// client.js
const axios = require("axios");
const { io } = require("socket.io-client");

const PYTHON_SERVICE_URL = "http://localhost:8000/scan";
const SERVER_PORT = 3000; // The socket server port

const SERVER_IP = "http://192.168.206.1:3000"

// Generate random test input
const ServiceType = {
  VOICE: 1,
  VIDEO: 2,
  DATA: 3,
  IOT: 4,
  STREAMING: 5,
  BULK_TRANSFER: 6,
  CONTROL: 7,
  EMERGENCY: 8
};

// QoS + resource profiles by service type
const QoSProfiles = {
  [ServiceType.VOICE]: {
    bandwidth: [0.1, 0.5],
    latency: [20, 100],
    reliability: [0.95, 0.99],
    priority: [2, 4],
    cpu: [1, 4],
    power: [2, 6]
  },
  [ServiceType.VIDEO]: {
    bandwidth: [2, 10],
    latency: [50, 150],
    reliability: [0.90, 0.98],
    priority: [3, 6],
    cpu: [10, 30],
    power: [20, 50]
  },
  [ServiceType.DATA]: {
    bandwidth: [1, 20],
    latency: [50, 200],
    reliability: [0.90, 0.97],
    priority: [4, 7],
    cpu: [5, 20],
    power: [10, 40]
  },
  [ServiceType.IOT]: {
    bandwidth: [0.05, 0.5],
    latency: [10, 100],
    reliability: [0.97, 0.999],
    priority: [2, 5],
    cpu: [1, 3],
    power: [1, 5]
  },
  [ServiceType.STREAMING]: {
    bandwidth: [3, 15],
    latency: [50, 150],
    reliability: [0.90, 0.97],
    priority: [3, 6],
    cpu: [15, 40],
    power: [20, 60]
  },
  [ServiceType.BULK_TRANSFER]: {
    bandwidth: [10, 100],
    latency: [100, 500],
    reliability: [0.85, 0.95],
    priority: [7, 10],
    cpu: [20, 50],
    power: [40, 80]
  },
  [ServiceType.CONTROL]: {
    bandwidth: [0.1, 1],
    latency: [5, 50],
    reliability: [0.99, 0.999],
    priority: [1, 3],
    cpu: [2, 6],
    power: [5, 10]
  },
  [ServiceType.EMERGENCY]: {
    bandwidth: [0.5, 2],
    latency: [1, 20],
    reliability: [0.999, 1.0],
    priority: [1, 1],
    cpu: [5, 15],
    power: [10, 20]
  }
};



// Utility: pick random value in range
function randRange([min, max], decimals = 2) {
  return parseFloat((Math.random() * (max - min) + min).toFixed(decimals));
}

// Function to generate a request object
function generateRequest(user) {
  const serviceTypes = Object.values(ServiceType);
  const type = serviceTypes[Math.floor(Math.random() * serviceTypes.length)];

  const profile = QoSProfiles[type];

  const bandwidth_required = randRange(profile.bandwidth, 2);
  const latency_required = randRange(profile.latency, 0);
  const reliability_required = randRange(profile.reliability, 3);
  const cpu_required = Math.floor(randRange(profile.cpu, 0));
  const power_required = Math.floor(randRange(profile.power, 0));

  const packet_size = Math.floor(Math.random() * 900) + 100; // 100–1000 bytes
  const priority = profile.priority[0] === profile.priority[1]
    ? profile.priority[0]
    : Math.floor(randRange(profile.priority, 0));

  const direct_sat_support = user.support5G;

  return {
    request_id: "req_" + Math.random().toString(36).substring(2, 10),
    type,
    source_location: { lat: user.lat, lon: user.lon },
    bandwidth_required,
    latency_required,
    reliability_required,
    cpu_required,
    power_required,
    packet_size,
    direct_sat_support,
    priority
  };
}


// Example: integrate with your randomUser()
function randomUser() {
  const regions = [
    { name: "China", latRange: [18, 54], lonRange: [73, 135], weight: 20 },
    { name: "India", latRange: [8, 37], lonRange: [68, 97], weight: 18 },
    { name: "Europe", latRange: [35, 60], lonRange: [-10, 40], weight: 15 },
    { name: "USA", latRange: [25, 50], lonRange: [-125, -66], weight: 15 },
    { name: "Brazil", latRange: [-35, 5], lonRange: [-74, -34], weight: 7 },
    { name: "Nigeria", latRange: [4, 14], lonRange: [3, 15], weight: 5 },
    { name: "Japan", latRange: [30, 45], lonRange: [129, 146], weight: 5 },
    { name: "SoutheastAsia", latRange: [-10, 20], lonRange: [95, 120], weight: 5 },
    { name: "Other", latRange: [-90, 90], lonRange: [-180, 180], weight: 10 }
  ];

  const totalWeight = regions.reduce((sum, r) => sum + r.weight, 0);
  let rand = Math.random() * totalWeight;
  let selectedRegion;

  for (let r of regions) {
    if (rand < r.weight) {
      selectedRegion = r;
      break;
    }
    rand -= r.weight;
  }

  const lat = (Math.random() * (selectedRegion.latRange[1] - selectedRegion.latRange[0]) + selectedRegion.latRange[0]).toFixed(4);
  const lon = (Math.random() * (selectedRegion.lonRange[1] - selectedRegion.lonRange[0]) + selectedRegion.lonRange[0]).toFixed(4);

  const support5G = Math.random() < 0.6;
  return { lat: parseFloat(lat), lon: parseFloat(lon), support5G };
}


// Scan satellites / stations from Python service
async function scanVisible(userLat, userLon, support5G = true) {
  try {
    const res = await axios.get(PYTHON_SERVICE_URL, {
      params: { lat: userLat, lon: userLon, support5G }
    });
    const visible = res.data; // already sorted by Python
    return visible || [];
  } catch (err) {
    console.error("❌ Error scanning:", err.message);
    return [];
  }
}

function tryConnect(user, target) {
  return new Promise((resolve) => {
    request = generateRequest(user);
    const socket = io(SERVER_IP, {
      query: { request: JSON.stringify(request), targetId: target.id },
      timeout: 3000
    });

    let accepted = false;

    socket.on("connect", () => {
      console.log(`🔗 Trying connection to ${target.type} ${target.id}`);
    });

    socket.on("connectionResult", (msg) => {
      if (msg === "accepted") {
        accepted = true;
        console.log(`✅ Connection accepted by ${target.id}`);

        // Listen for heartbeat continuously
        socket.on("heartbeat", (data) => {
          console.log(`💓 Heartbeat from ${target.id}:`, data);
        });

        // Resolve with the active socket
        resolve({ socket, target });
      } else {
        socket.disconnect();
      }
    });

    socket.on("disconnect", () => {
      if (!accepted) resolve(null);
    });

    // Timeout fallback
    setTimeout(() => {
      if (!accepted) {
        socket.disconnect();
        resolve(null);
      }
    }, 5000);
  });
}


async function main() {
  let activeSocket = null; // store the successful connection

  while (true) {
    const user = randomUser();
    console.log(`\n🌐 Random user: lat=${user.lat}, lon=${user.lon}, support5G=${user.support5G}`);

    const visibleList = await scanVisible(user.lat, user.lon, user.support5G);
    if (visibleList.length === 0) {
      console.log("❌ No visible targets, retrying in 3s...");
      await new Promise((r) => setTimeout(r, 3000));
      continue;
    }

    let connected = false;
    for (const target of visibleList) {
      const result = await tryConnect(user, target);
      if (result) {
        connected = true;
        activeSocket = result.socket;
        console.log(`🎯 Connected to ${result.target.id} successfully!`);
        break; // stop trying other targets
      } else {
        console.log(`❌ Connection to ${target.id} failed or was ignored.`);
      }
    }

    if (connected) {
      console.log("✅ Maintaining connection. Stopping further scans.");
      break; // stop the while loop entirely
    } else {
      console.log("⚠️ No targets accepted the connection, rescanning...");
      await new Promise((r) => setTimeout(r, 2000));
    }
  }

  // You can now use activeSocket to maintain communication
  activeSocket.on("someServerEvent", (data) => {
    console.log("📥 Received:", data);
  });

  // Keep process alive
  process.stdin.resume();
}


// Start
main().catch(console.error);