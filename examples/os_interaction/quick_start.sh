#!/bin/bash
# Quick Start Script for OS Interaction Training
# This script helps you get started with training on the OS task

set -e

echo "=========================================="
echo "OS Interaction Task - Quick Start"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if AgentBench path is set
if [ -z "$AGENTBENCH_PATH" ]; then
    echo -e "${YELLOW}Warning: AGENTBENCH_PATH not set${NC}"
    echo "Please set it to your AgentBench directory:"
    echo "  export AGENTBENCH_PATH=/path/to/AgentBench"
    echo ""
    read -p "Enter AgentBench path (or press Enter to skip): " AGENTBENCH_PATH
    if [ -z "$AGENTBENCH_PATH" ]; then
        echo -e "${RED}Skipping AgentBench checks${NC}"
        AGENTBENCH_PATH="."
    fi
fi

# Check Docker
echo "Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker ps &> /dev/null; then
    echo -e "${RED}✗ Docker daemon not running or no permissions${NC}"
    echo "Please start Docker or add your user to docker group:"
    echo "  sudo usermod -aG docker \$USER && newgrp docker"
    exit 1
fi
echo -e "${GREEN}✓ Docker is ready${NC}"

# Check Docker image
echo ""
echo "Checking Docker image..."
if docker images | grep -q "local-os/default"; then
    echo -e "${GREEN}✓ Docker image 'local-os/default' found${NC}"
else
    echo -e "${YELLOW}! Docker image 'local-os/default' not found${NC}"
    echo "Building Docker image (this may take a few minutes)..."
    if [ -d "$AGENTBENCH_PATH/data/os_interaction/res/dockerfiles" ]; then
        docker build -f "$AGENTBENCH_PATH/data/os_interaction/res/dockerfiles/default" \
                     "$AGENTBENCH_PATH/data/os_interaction/res/dockerfiles" \
                     --tag local-os/default
        echo -e "${GREEN}✓ Docker image built${NC}"
    else
        echo -e "${RED}✗ Cannot find AgentBench dockerfiles${NC}"
        echo "Please check AGENTBENCH_PATH or build manually"
        exit 1
    fi
fi

# Check if task server is running
echo ""
echo "Checking task server..."
if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
    TASK_NAME=$(curl -s http://localhost:5000/api/health | python3 -c "import sys, json; print(json.load(sys.stdin)['task_name'])")
    echo -e "${GREEN}✓ Task server is running: $TASK_NAME${NC}"
else
    echo -e "${YELLOW}! Task server not running${NC}"
    echo ""
    echo "To start the task server, run in another terminal:"
    echo -e "${YELLOW}  cd $AGENTBENCH_PATH${NC}"
    echo -e "${YELLOW}  python -m src.server.task_server_adapter os-dev --port 5000${NC}"
    echo ""
    read -p "Press Enter after you've started the task server..."

    # Check again
    if ! curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
        echo -e "${RED}✗ Task server still not reachable${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Task server is running${NC}"
fi

# Test connection
echo ""
echo "Testing task server connection..."
python3 examples/os_interaction/test_connection.py --server http://localhost:5000

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "=========================================="
    echo "Ready to start training!"
    echo "=========================================="
    echo ""
    echo "Run the following command to start training:"
    echo ""
    echo -e "${GREEN}python -m areal.launcher.local examples/os_interaction/train.py \\${NC}"
    echo -e "${GREEN}    --config examples/os_interaction/config.yaml \\${NC}"
    echo -e "${GREEN}    experiment_name=os_rl_experiment \\${NC}"
    echo -e "${GREEN}    trial_name=run1${NC}"
    echo ""
else
    echo -e "${RED}✗ Connection test failed${NC}"
    echo "Please check the errors above and try again"
    exit 1
fi
