const { io } = require("socket.io-client");

const socket = io("http://localhost:3000");

// Gửi yêu cầu user
socket.emit("user_request", {
    user_id: "vehicle_01",
    position: [10.5, 20.3]
});

// Nhận phản hồi từ server
socket.on("server_response", (data) => {
    console.log("Response from server:", data);
});
