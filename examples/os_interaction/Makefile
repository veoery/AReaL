# Makefile for OS Interaction Task Training
# Convenient commands for common operations

.PHONY: help test-server train test-connection clean

# Default AgentBench path (override with: make AGENTBENCH_PATH=/path/to/agentbench)
AGENTBENCH_PATH ?= ../../../AgentBench
SERVER_PORT ?= 5000
SERVER_URL ?= http://localhost:$(SERVER_PORT)

help:
	@echo "OS Interaction Task - Available Commands"
	@echo "========================================="
	@echo ""
	@echo "  make test-connection  - Test task server connection"
	@echo "  make test-server      - Start task server (requires AgentBench)"
	@echo "  make train            - Start training (server must be running)"
	@echo "  make quick-start      - Run quick start script"
	@echo "  make clean            - Clean up generated files"
	@echo ""
	@echo "Environment Variables:"
	@echo "  AGENTBENCH_PATH=$(AGENTBENCH_PATH)"
	@echo "  SERVER_PORT=$(SERVER_PORT)"
	@echo "  SERVER_URL=$(SERVER_URL)"
	@echo ""

test-connection:
	@echo "Testing connection to task server at $(SERVER_URL)..."
	python test_connection.py --server $(SERVER_URL)

test-server:
	@echo "Starting task server on port $(SERVER_PORT)..."
	@echo "AgentBench path: $(AGENTBENCH_PATH)"
	@if [ ! -d "$(AGENTBENCH_PATH)" ]; then \
		echo "Error: AgentBench directory not found at $(AGENTBENCH_PATH)"; \
		echo "Set AGENTBENCH_PATH environment variable or pass it to make:"; \
		echo "  make test-server AGENTBENCH_PATH=/path/to/AgentBench"; \
		exit 1; \
	fi
	cd $(AGENTBENCH_PATH) && \
		python -m src.server.task_server_adapter os-dev --port $(SERVER_PORT)

train:
	@echo "Starting training..."
	@echo "Server URL: $(SERVER_URL)"
	python -m areal.launcher.local train.py \
		--config config.yaml \
		task_server_url=$(SERVER_URL) \
		experiment_name=os_rl_experiment \
		trial_name=$$(date +%Y%m%d_%H%M%S)

quick-start:
	./quick_start.sh

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -f *.pyc
	@echo "Done"

# Development helpers
.PHONY: build-docker check-docker

build-docker:
	@echo "Building Docker image for OS task..."
	@if [ ! -d "$(AGENTBENCH_PATH)" ]; then \
		echo "Error: AgentBench directory not found"; \
		exit 1; \
	fi
	docker build \
		-f $(AGENTBENCH_PATH)/data/os_interaction/res/dockerfiles/default \
		$(AGENTBENCH_PATH)/data/os_interaction/res/dockerfiles \
		--tag local-os/default

check-docker:
	@echo "Checking Docker setup..."
	@docker --version || (echo "Docker not installed" && exit 1)
	@docker ps > /dev/null || (echo "Docker daemon not running" && exit 1)
	@echo "✓ Docker is ready"
	@if docker images | grep -q "local-os/default"; then \
		echo "✓ Docker image 'local-os/default' found"; \
	else \
		echo "! Docker image 'local-os/default' not found"; \
		echo "  Run 'make build-docker' to build it"; \
	fi
