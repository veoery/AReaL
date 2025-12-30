#!/usr/bin/env python3
"""Generate index files for os-std (train) and os-dev (eval) tasks."""

import json
import os
import glob

AGENTBENCH_ROOT = "/home/haizhonz/yizhuod/AgentBench"
OUTPUT_DIR = "/home/haizhonz/yizhuod/AReaL/examples/os_interaction/data"

def count_problems_in_file(json_path):
    """Count how many problems are in a JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return len(data)
    elif isinstance(data, dict):
        return 1
    else:
        raise ValueError(f"Unexpected data type in {json_path}: {type(data)}")

def generate_indices_for_dataset(dataset_num, prefix):
    """Generate indices for all problems in a dataset."""
    pattern = f"{AGENTBENCH_ROOT}/data/os_interaction/data/{dataset_num}/*.json"
    files = sorted(glob.glob(pattern))

    indices = []
    for json_file in files:
        filename = os.path.basename(json_file).removesuffix(".json")
        num_problems = count_problems_in_file(json_file)

        print(f"Dataset {dataset_num}, file {filename}: {num_problems} problems")

        for i in range(num_problems):
            index = f"{prefix}{filename}-{i:05d}"
            indices.append({"index": index, "messages": []})

    return indices

def generate_dev_indices():
    """Generate indices for dev.json."""
    dev_file = f"{AGENTBENCH_ROOT}/data/os_interaction/data/dev.json"

    if not os.path.exists(dev_file):
        print(f"Warning: {dev_file} not found, skipping dev indices")
        return []

    # Dev uses a single JSON file (not a glob pattern)
    num_problems = count_problems_in_file(dev_file)
    print(f"Dev set: {num_problems} problems")

    indices = []
    for i in range(num_problems):
        index = f"dev-001-{i:05d}"
        indices.append({"index": index, "messages": []})

    return indices

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate training indices (os-std: datasets 1-7)
    print("=" * 60)
    print("Generating TRAINING indices (os-std: datasets 1-7)")
    print("=" * 60)
    train_indices = []
    for dataset_num in range(1, 8):  # 1-7 inclusive
        prefix = f"std-{dataset_num:03d}-"
        indices = generate_indices_for_dataset(dataset_num, prefix)
        train_indices.extend(indices)

    train_output = os.path.join(OUTPUT_DIR, "train_indices.jsonl")
    with open(train_output, 'w') as f:
        for item in train_indices:
            f.write(json.dumps(item) + '\n')

    print(f"\n✓ Generated {len(train_indices)} training indices -> {train_output}")

    # Generate validation indices (os-dev: dev.json)
    print("\n" + "=" * 60)
    print("Generating VALIDATION indices (os-dev: dev.json)")
    print("=" * 60)
    valid_indices = generate_dev_indices()

    valid_output = os.path.join(OUTPUT_DIR, "valid_indices.jsonl")
    with open(valid_output, 'w') as f:
        for item in valid_indices:
            f.write(json.dumps(item) + '\n')

    print(f"\n✓ Generated {len(valid_indices)} validation indices -> {valid_output}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Training samples (os-std, datasets 1-7): {len(train_indices)}")
    print(f"Validation samples (os-dev, dev.json):   {len(valid_indices)}")
    print(f"\nTotal samples: {len(train_indices) + len(valid_indices)}")

if __name__ == "__main__":
    main()
