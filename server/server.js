// server.js
const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const axios = require("axios");

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: { origin: "*" } // cho phép mọi client connect (có thể giới hạn theo domain sau)
});

// Khi client kết nối
io.on("connection", (socket) => {
    console.log("Client connected:", socket.id);

    // Nhận yêu cầu từ client
    socket.on("user_request", async (data) => {
        console.log("Received user_request:", data);

        try {
            // Gọi sang Python AI service
            const response = await axios.post(
                "http://127.0.0.1:5000/allocate",
                data,
                { timeout: 5000 } // timeout 5s tránh treo
            );

            // Gửi phản hồi lại cho đúng client
            socket.emit("server_response", response.data);

        } catch (error) {
            console.error("Error calling Python service:", error.message);
            socket.emit("server_response", {
                error: "Python service unavailable or timeout"
            });
        }
    });

    // Khi client ngắt kết nối
    socket.on("disconnect", (reason) => {
        console.log("Client disconnected:", socket.id, "Reason:", reason);
    });
});

// Chạy server Node.js
const PORT = 3000;
server.listen(PORT, () => {
    console.log(`Node.js Socket.io server running on port ${PORT}`);
});
