// client.js
const { io } = require("socket.io-client");

// Kết nối đến server Node.js
const socket = io("http://localhost:3000", {
    reconnectionAttempts: 5,   // thử reconnect tối đa 5 lần
    timeout: 5000              // timeout kết nối 5s
});

// Khi kết nối thành công
socket.on("connect", () => {
    console.log("Connected to server:", socket.id);

    // Gửi yêu cầu user (ví dụ request từ vehicle_01)
    socket.emit("user_request", {
        user_id: "vehicle_01",
        position: [10.5, 20.3]
    });
});

// Nhận phản hồi từ server
socket.on("server_response", (data) => {
    console.log("Response from server:", data);
});

// Nếu lỗi kết nối
socket.on("connect_error", (err) => {
    console.error("Connection error:", err.message);
});

// Nếu bị mất kết nối
socket.on("disconnect", (reason) => {
    console.warn("Disconnected:", reason);
});
