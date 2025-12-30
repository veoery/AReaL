"""
Generate dataset files from AgentBench OS interaction data.

This script reads the AgentBench dev.json and generates properly
formatted dataset files for AReaL training.

Usage:
    python generate_dataset.py --agentbench-path /path/to/AgentBench
"""

import argparse
import json
import os
from pathlib import Path


def generate_dataset_files(
    agentbench_path: str,
    output_dir: str,
    train_split: float = 0.8
):
    """
    Generate train/valid dataset files from AgentBench data.

    Parameters
    ----------
    agentbench_path : str
        Path to AgentBench directory
    output_dir : str
        Output directory for dataset files
    train_split : float
        Fraction of data to use for training (rest for validation)
    """
    # Load dev.json
    dev_json_path = os.path.join(
        agentbench_path,
        "data/os_interaction/data/dev.json"
    )

    if not os.path.exists(dev_json_path):
        raise FileNotFoundError(
            f"dev.json not found at {dev_json_path}\n"
            f"Please check AgentBench path"
        )

    with open(dev_json_path, 'r') as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples from dev.json")

    # Generate indices based on AgentBench's indexing scheme
    # From configs/tasks/os.yaml: index_prefix: "dev-001-"
    # From task.py line 311: index = prefix + "%05d" % idx
    # The filename "dev.json" becomes part of the prefix
    # Final format: dev-001-dev-00000, dev-001-dev-00001, etc.
    indices = []
    for idx in range(len(samples)):
        # index_prefix ("dev-001-") + basename("dev") + "-" + number
        index = f"dev-001-dev-{idx:05d}"
        indices.append({
            "index": index,
            "messages": []  # Empty, task server provides prompt
        })

    # Split into train/valid
    split_idx = int(len(indices) * train_split)
    train_indices = indices[:split_idx]
    valid_indices = indices[split_idx:]

    print(f"Split: {len(train_indices)} train, {len(valid_indices)} valid")

    # Write to files
    os.makedirs(output_dir, exist_ok=True)

    train_file = os.path.join(output_dir, "train_indices.jsonl")
    with open(train_file, 'w') as f:
        for item in train_indices:
            f.write(json.dumps(item) + '\n')
    print(f"Wrote train indices to {train_file}")

    valid_file = os.path.join(output_dir, "valid_indices.jsonl")
    with open(valid_file, 'w') as f:
        for item in valid_indices:
            f.write(json.dumps(item) + '\n')
    print(f"Wrote valid indices to {valid_file}")

    # Also create a file with all indices for reference
    all_file = os.path.join(output_dir, "all_indices.jsonl")
    with open(all_file, 'w') as f:
        for item in indices:
            f.write(json.dumps(item) + '\n')
    print(f"Wrote all indices to {all_file}")

    # Print sample information
    print(f"\nSample indices:")
    print(f"  First train: {train_indices[0]['index']}")
    print(f"  Last train:  {train_indices[-1]['index']}")
    print(f"  First valid: {valid_indices[0]['index']}")
    print(f"  Last valid:  {valid_indices[-1]['index']}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate AReaL dataset files from AgentBench data"
    )
    parser.add_argument(
        "--agentbench-path",
        type=str,
        default="../../../AgentBench",
        help="Path to AgentBench directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for dataset files"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)"
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    agentbench_path = Path(args.agentbench_path).resolve()
    output_dir = (script_dir / args.output_dir).resolve()

    print(f"AgentBench path: {agentbench_path}")
    print(f"Output directory: {output_dir}")
    print()

    generate_dataset_files(
        str(agentbench_path),
        str(output_dir),
        args.train_split
    )

    print("\n✓ Dataset generation complete!")


if __name__ == "__main__":
    main()
