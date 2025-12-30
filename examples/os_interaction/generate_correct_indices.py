#!/usr/bin/env python3
"""
Generate correct index files by counting actual problems in each JSON file.

This script:
1. Reads each JSON file in datasets 1-7
2. Counts how many problems are in each file (array length)
3. Generates indices for ALL problems
4. Creates train_indices.jsonl (datasets 1-6) and valid_indices.jsonl (dataset 7)
"""

import json
import glob
import os
from pathlib import Path

AGENTBENCH_ROOT = Path("/home/haizhonz/yizhuod/AgentBench")
DATA_DIR = AGENTBENCH_ROOT / "data/os_interaction/data"
OUTPUT_DIR = Path("/home/haizhonz/yizhuod/AReaL/examples/os_interaction/data")

def count_problems_in_file(json_path):
    """Count number of problems in a JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return len(data)
    elif isinstance(data, dict):
        return 1
    else:
        raise ValueError(f"Unexpected data type in {json_path}")

def generate_indices_for_dataset(dataset_num, index_prefix):
    """Generate indices for all files in a dataset."""
    indices = []
    dataset_path = DATA_DIR / str(dataset_num)

    # Find all JSON files in this dataset
    json_files = sorted(glob.glob(str(dataset_path / "*.json")))

    print(f"\n  Dataset {dataset_num}:")
    total_problems = 0

    for json_file in json_files:
        filename = os.path.basename(json_file).removesuffix(".json")
        num_problems = count_problems_in_file(json_file)
        total_problems += num_problems

        print(f"    {filename}.json: {num_problems} problems")

        # Generate indices for each problem in this file
        for problem_idx in range(num_problems):
            index = f"{index_prefix}{filename}-{problem_idx:05d}"
            indices.append(index)

    print(f"    Total: {total_problems} problems")
    return indices

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("="*80)
    print("Generating Correct Index Files for OS Interaction")
    print("="*80)

    # Generate training indices (datasets 1-6)
    print("\n→ Training Datasets (1-6):")
    train_indices = []

    for dataset_num in range(1, 7):
        prefix = f"train-{dataset_num:03d}-"
        try:
            indices = generate_indices_for_dataset(dataset_num, prefix)
            train_indices.extend(indices)
        except Exception as e:
            print(f"    Error processing dataset {dataset_num}: {e}")

    # Generate eval indices (dataset 7)
    print("\n→ Evaluation Dataset (7):")
    try:
        eval_indices = generate_indices_for_dataset(7, "eval-007-")
    except Exception as e:
        print(f"    Error processing dataset 7: {e}")
        eval_indices = []

    # Write train indices
    train_file = OUTPUT_DIR / "train_indices.jsonl"
    print(f"\n→ Writing {len(train_indices)} training indices to {train_file}...")
    with open(train_file, "w") as f:
        for idx in train_indices:
            f.write(json.dumps({"index": idx, "messages": []}) + "\n")
    print(f"  ✓ Done")

    # Write eval indices
    eval_file = OUTPUT_DIR / "valid_indices.jsonl"
    print(f"\n→ Writing {len(eval_indices)} eval indices to {eval_file}...")
    with open(eval_file, "w") as f:
        for idx in eval_indices:
            f.write(json.dumps({"index": idx, "messages": []}) + "\n")
    print(f"  ✓ Done")

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Training samples: {len(train_indices)} (datasets 1-6)")
    print(f"Eval samples: {len(eval_indices)} (dataset 7)")
    print(f"Total samples: {len(train_indices) + len(eval_indices)}")
    print("="*80)

    # Show first few indices as examples
    print("\nExample training indices:")
    for idx in train_indices[:10]:
        print(f"  {idx}")
    if len(train_indices) > 10:
        print(f"  ... and {len(train_indices) - 10} more")

    print("\nExample eval indices:")
    for idx in eval_indices[:5]:
        print(f"  {idx}")
    if len(eval_indices) > 5:
        print(f"  ... and {len(eval_indices) - 5} more")

if __name__ == "__main__":
    main()
