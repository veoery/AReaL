#!/usr/bin/env python3
"""
Fix AlfWorld indices by querying the task server to get actual sample count.

This script queries the running task server to find out how many samples
actually exist, then regenerates the index files with the correct count.
"""

import json
import os
import requests

TASK_SERVER_URL = "http://localhost:5000"
OUTPUT_DIR = "/home/haizhonz/yizhuod/AReaL/examples/alfworld/data"

def get_actual_sample_count():
    """Query the task server to get the actual number of samples."""
    try:
        response = requests.get(f"{TASK_SERVER_URL}/api/task/info", timeout=10)
        response.raise_for_status()
        info = response.json()
        return info.get("num_samples", 0)
    except Exception as e:
        print(f"Error querying task server: {e}")
        print("\nMake sure the task server is running:")
        print("  cd /home/haizhonz/yizhuod/AgentBench")
        print("  python -m src.server.task_server_adapter alfworld-std --port 5000")
        return None

def generate_indices(num_samples, prefix, output_file):
    """Generate index file with correct number of samples."""
    indices = []
    for i in range(num_samples):
        index = i
        indices.append({"index": index, "messages": []})

    with open(output_file, 'w') as f:
        for item in indices:
            f.write(json.dumps(item) + '\n')

    print(f"✓ Generated {len(indices)} indices -> {output_file}")

def main():
    print("=" * 70)
    print("Querying task server for actual sample count...")
    print("=" * 70)

    num_samples = get_actual_sample_count()

    if num_samples is None:
        return 1

    print(f"\nTask server reports {num_samples} samples in alfworld-std")
    print("\n" + "=" * 70)
    print("Regenerating training indices...")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_output = os.path.join(OUTPUT_DIR, "train_indices.jsonl")
    generate_indices(num_samples, "", train_output)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Training samples: {num_samples}")
    print("\nRestart your training and it should work now!")

    return 0

if __name__ == "__main__":
    exit(main())
