// server.js
const http = require("http");
const os = require("os");
const { Server } = require("socket.io");

const PORT = 3000;
const ACCEPT_RATE = 0.3; // 30% chance to accept connection
const HEARTBEAT_INTERVAL = 5000; // ms

// Function to get local LAN IP
function getLocalIP() {
  const nets = os.networkInterfaces();
  for (const name of Object.keys(nets)) {
    for (const net of nets[name]) {
      // Pick IPv4, non-internal (not 127.0.0.1)
      if (net.family === "IPv4" && !net.internal) {
        return net.address;
      }
    }
  }
  return "127.0.0.1"; // fallback
}

const HOST = getLocalIP();

// Create HTTP server
const server = http.createServer();

// Attach Socket.io
const io = new Server(server, {
  cors: { origin: "*" },
  pingInterval: 10000,
  pingTimeout: 5000,
});

// Start server
server.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 Server running at: http://${HOST}:${PORT}`);
});

// Handle connections
io.on("connection", (socket) => {
  const { targetId } = socket.handshake.query;

  console.log(`🔌 New client connected: socketId=${socket.id}, targetId=${targetId}`);

  const accepted = Math.random() < ACCEPT_RATE;

  if (accepted) {
    console.log(`✅ Accepted connection for targetId=${targetId}`);
    socket.emit("connectionResult", "accepted");

    // Heartbeat
    const intervalId = setInterval(() => {
      if (socket.connected) {
        socket.emit("heartbeat", { msg: "ping", timestamp: Date.now() });
      } else {
        clearInterval(intervalId);
      }
    }, HEARTBEAT_INTERVAL);
  } else {
    socket.disconnect(true);
  }

  socket.on("disconnect", () => {
    console.log(`🔌 Client disconnected: socketId=${socket.id}`);
  });
});
