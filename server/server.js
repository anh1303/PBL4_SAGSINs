// server.js
const { Server } = require("socket.io");

const PORT = 3000;
const ACCEPT_RATE = 0.3; // 30% chance to accept connection
const HEARTBEAT_INTERVAL = 5000; // ms

// Create Socket.io server
const io = new Server(PORT, {
  cors: { origin: "*" },
  pingInterval: 10000,
  pingTimeout: 5000,
});

console.log(`🚀 Server listening on port ${PORT}`);

// Handle connections
io.on("connection", (socket) => {
  const { targetId } = socket.handshake.query;

  // Decide immediately whether to accept or ignore
  const accepted = Math.random() < ACCEPT_RATE;

  if (accepted) {
    console.log(`✅ Accepted connection for targetId=${targetId}`);
    socket.emit("connectionResult", "accepted");

    // Keep socket alive: send periodic heartbeat messages
    const intervalId = setInterval(() => {
      if (socket.connected) {
        socket.emit("heartbeat", { msg: "ping", timestamp: Date.now() });
      } else {
        clearInterval(intervalId);
      }
    }, HEARTBEAT_INTERVAL);

  } else {
    // Reject silently: do not emit anything
    // Disconnect immediately to free server resources
    socket.disconnect(true);
  }

  socket.on("disconnect", () => {
    console.log(`🔌 Client disconnected: socketId=${socket.id}`);
  });
});
