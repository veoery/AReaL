"""
OS Interaction Dataset Loader

This dataset loader provides task indices for the OS interaction task,
which runs via an external task server (AgentBench).

The dataset format is simple JSONL with task indices:
    {"index": "dev-001-dev-00000", "messages": []}

The actual task descriptions and execution are handled by the task server.
This keeps the task environment separate from the RL training framework.
"""

from datasets import load_dataset


def get_os_interaction_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    """SFT training is not supported for OS interaction tasks."""
    raise NotImplementedError(
        "OS interaction dataset is designed for RL training only. "
        "Tasks require interactive execution in a shell environment, "
        "which cannot be reduced to static SFT examples."
    )


def get_os_interaction_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    """
    Load OS interaction dataset for RL training.

    Args:
        path: Path to the JSONL file containing task indices
              (e.g., "examples/os_interaction/data/train_indices.jsonl")
        split: Dataset split (ignored for JSONL files, but required by API)
        tokenizer: Tokenizer instance (unused for this dataset)
        max_length: Maximum length (unused for this dataset)

    Returns:
        Dataset with fields:
            - index: Task identifier (e.g., "dev-001-dev-00000")
            - messages: Empty list (actual task data comes from server)

    The workflow will use the 'index' field to request specific tasks
    from the task server via HTTP API.
    """
    # Load JSONL file using HuggingFace datasets
    # For JSONL files, we always use split="train" when loading
    dataset = load_dataset("json", data_files=path, split="train")

    def process(sample):
        """
        Process each sample to ensure consistent format.

        The sample already has 'index' and 'messages' fields from JSONL.
        We keep both for compatibility:
        - 'index': Used by TaskServerWorkflow to identify the task
        - 'messages': Empty list, maintained for standard RL dataset interface
        """
        return {
            "index": sample["index"],
            "messages": sample.get("messages", []),
        }

    # Process the dataset
    dataset = dataset.map(process)

    # No filtering needed - all OS tasks are valid
    # max_length is not applicable since we don't have actual text content yet

    return dataset