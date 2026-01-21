from datasets import load_dataset

def get_webshop_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    raise NotImplementedError

def get_webshop_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
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

    return dataset