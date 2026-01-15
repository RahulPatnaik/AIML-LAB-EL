#!/bin/bash

echo "======================================"
echo "EEG Model Explainability Dashboard"
echo "======================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

echo "✓ Python 3 found"

# Check if backend dependencies are installed
echo ""
echo "Checking dependencies..."

# Install dependencies if needed
cd backend
if [ ! -f ".dependencies_installed" ]; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        touch .dependencies_installed
        echo "✓ Dependencies installed"
    else
        echo "❌ Failed to install dependencies"
        exit 1
    fi
else
    echo "✓ Dependencies already installed"
fi

# Start backend
echo ""
echo "Starting backend server on http://localhost:8000 ..."
python main.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Check if backend is running
if kill -0 $BACKEND_PID 2>/dev/null; then
    echo "✓ Backend server started (PID: $BACKEND_PID)"
else
    echo "❌ Failed to start backend server"
    exit 1
fi

# Start frontend
cd ../frontend
echo ""
echo "Starting frontend server on http://localhost:3000 ..."
echo ""
echo "======================================"
echo "✓ Application is running!"
echo "======================================"
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

python3 -m http.server 3000 &
FRONTEND_PID=$!

# Trap Ctrl+C and cleanup
trap cleanup INT

cleanup() {
    echo ""
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✓ Servers stopped"
    exit 0
}

# Wait for user to press Ctrl+C
wait
