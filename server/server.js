const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const axios = require("axios"); // gọi sang Python

const app = express();
const server = http.createServer(app);
const io = new Server(server);

io.on("connection", (socket) => {
    console.log("Client connected:", socket.id);

    // Nhận yêu cầu từ client
    socket.on("user_request", async (data) => {
        console.log("User request:", data);

        try {
            // Gọi sang Python AI service
            const response = await axios.post("http://127.0.0.1:5000/allocate", data);

            // Trả kết quả về client
            socket.emit("server_response", response.data);

        } catch (error) {
            console.error("Error calling Python service:", error);
            socket.emit("server_response", { error: "Python service failed" });
        }
    });

    socket.on("disconnect", () => {
        console.log("Client disconnected:", socket.id);
    });
});

server.listen(3000, () => {
    console.log("Node.js Socket.io server running on port 3000");
});
