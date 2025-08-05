#!/bin/bash

# SD Discord Bot Startup Script (without Docker)
# Requires: temporal CLI, Python 3.8+

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "SD Discord Bot with Temporal"
echo "=========================================="

# Check if temporal CLI is installed
if ! command -v temporal &> /dev/null; then
    echo -e "${RED}❌ Temporal CLI not found${NC}"
    echo "Please install it first:"
    echo "  Mac: brew install temporal"
    echo "  Linux: curl -sSf https://temporal.download/cli.sh | sh"
    echo "  Visit: https://docs.temporal.io/cli#install"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    
    # Kill all background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Kill temporal server
    pkill -f "temporal server" 2>/dev/null || true
    
    echo -e "${GREEN}✅ All services stopped${NC}"
    exit 0
}

# Set trap for cleanup on script exit
trap cleanup EXIT INT TERM

# Start Temporal dev server in background
echo -e "${GREEN}Starting Temporal server...${NC}"
temporal server start-dev \
    --ui-port 8080 \
    --db-filename temporal.db \
    --log-level warn \
    > temporal.log 2>&1 &

TEMPORAL_PID=$!
echo "  PID: $TEMPORAL_PID"

# Wait for Temporal to be ready
echo "Waiting for Temporal to start..."
sleep 5

# Check if Temporal is running
if ! kill -0 $TEMPORAL_PID 2>/dev/null; then
    echo -e "${RED}❌ Temporal failed to start${NC}"
    cat temporal.log
    exit 1
fi

echo -e "${GREEN}✅ Temporal server started${NC}"
echo "  UI: http://localhost:8080"

# Start Temporal worker in background
echo -e "${GREEN}Starting Temporal worker...${NC}"
python3 temporal_worker.py > worker.log 2>&1 &
WORKER_PID=$!
echo "  PID: $WORKER_PID"

sleep 3

# Check if worker is running
if ! kill -0 $WORKER_PID 2>/dev/null; then
    echo -e "${RED}❌ Worker failed to start${NC}"
    cat worker.log
    exit 1
fi

echo -e "${GREEN}✅ Worker started${NC}"

# Start Discord bot in background
echo -e "${GREEN}Starting Discord bot...${NC}"
python3 run.py > bot.log 2>&1 &
BOT_PID=$!
echo "  PID: $BOT_PID"

sleep 2

# Check if bot is running
if ! kill -0 $BOT_PID 2>/dev/null; then
    echo -e "${RED}❌ Discord bot failed to start${NC}"
    cat bot.log
    exit 1
fi

echo -e "${GREEN}✅ Discord bot started${NC}"

echo "=========================================="
echo -e "${GREEN}All services running!${NC}"
echo "  Temporal UI: http://localhost:8080"
echo "  Logs: temporal.log, worker.log, bot.log"
echo -e "  Press ${YELLOW}Ctrl+C${NC} to stop all services"
echo "=========================================="

# Monitor processes
while true; do
    # Check if processes are still running
    if ! kill -0 $TEMPORAL_PID 2>/dev/null; then
        echo -e "${RED}❌ Temporal server crashed${NC}"
        break
    fi
    
    if ! kill -0 $WORKER_PID 2>/dev/null; then
        echo -e "${RED}❌ Worker crashed${NC}"
        break
    fi
    
    if ! kill -0 $BOT_PID 2>/dev/null; then
        echo -e "${RED}❌ Discord bot crashed${NC}"
        break
    fi
    
    sleep 5
done

echo -e "${RED}A service crashed. Check logs for details.${NC}"
cleanup