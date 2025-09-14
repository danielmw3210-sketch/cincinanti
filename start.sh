#!/bin/bash

# Crypto Price Forecast Startup Script

echo "🚀 Starting Crypto Price Forecast System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p backend/models/saved
mkdir -p frontend/node_modules

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
cd backend
pip install -r ../requirements.txt

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
cd ../frontend
npm install

# Start the backend in background
echo "🔧 Starting backend server..."
cd ../backend
python main.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 5

# Start the frontend
echo "🎨 Starting frontend server..."
cd ../frontend
npm start &
FRONTEND_PID=$!

echo "✅ System started successfully!"
echo "📊 Backend API: http://localhost:8000"
echo "🎨 Frontend: http://localhost:3000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT

# Wait for processes
wait