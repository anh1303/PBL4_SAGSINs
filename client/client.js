// client.js
const axios = require("axios");
const { io } = require("socket.io-client");

const PYTHON_SERVICE_URL = "http://localhost:8000/scan";
const SERVER_PORT = 3000; // The socket server port

// Generate random test input
function randomUser() {
  const lat = (Math.random() * 180 - 90).toFixed(4); // -90 .. 90
  const lon = (Math.random() * 360 - 180).toFixed(4); // -180 .. 180
  const support5G = Math.random() < 0.6; // true 60% of the time
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
    const socket = io(`http://localhost:${SERVER_PORT}`, {
      query: { userLat: user.lat, userLon: user.lon, targetId: target.id },
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
