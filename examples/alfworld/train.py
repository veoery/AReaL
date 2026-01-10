"""
AlfWorld Task Training with AReaL

This script trains a language model on the AlfWorld household task using
reinforcement learning via AReaL's PPO implementation.

The task server (AgentBench AlfWorld task) runs independently, and AReaL
connects to it via HTTP to collect trajectories for training.

Usage:
    # 1. Start task servers (in separate terminals)
    cd AgentBench
    python -m src.server.task_server_adapter alfworld-std --port 5000  # Training (standard split)
    python -m src.server.task_server_adapter alfworld-dev --port 5001  # Evaluation (dev split)

    # 2. Start training (in another terminal)
    cd AReaL
    python -m areal.launcher.local examples/alfworld/train.py \
        --config examples/alfworld/config.yaml \
        experiment_name=alfworld_rl \
        trial_name=run1
"""

import sys
from typing import Dict, Any

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.experimental.trainer import PPOTrainer
from areal.utils.hf_utils import load_hf_tokenizer
from areal.workflow.task_server import TaskServerWorkflow


class AlfWorldWorkflow(TaskServerWorkflow):
    """
    Custom workflow for AlfWorld task.

    Handles AlfWorld-specific requirements:
    - Converts string indices to integers (AlfWorld uses numeric indices)
    - Applies negative rewards for failures
    - Applies turn discount
    """

    def __init__(self, *args, failure_penalty: float = -0.1, **kwargs):
        """
        Parameters
        ----------
        failure_penalty : float
            Negative reward given when task fails (reward=0.0).
            Default: -0.1 (success=1.0, failure=-0.1)
        """
        super().__init__(*args, **kwargs)
        self.failure_penalty = failure_penalty

    async def _call_server(self, endpoint: str, method: str = "POST", data: Dict[str, Any] = None):
        """
        Override to convert sample_id to integer for AlfWorld.

        AlfWorld uses numeric indices (0, 1, 2, ...) not string indices.
        The parent class converts all sample_ids to strings, but AlfWorld
        expects integers in its start_sample() method.
        """
        # Convert sample_id from string to int if present
        if data and "sample_id" in data:
            try:
                # AlfWorld expects integer indices
                data = data.copy()  # Don't modify original
                data["sample_id"] = int(data["sample_id"])
            except (ValueError, TypeError):
                pass  # Keep as-is if conversion fails

        return await super()._call_server(endpoint, method, data)

    def postprocess_reward(
        self,
        reward: float,
        num_turns: int,
        info: Dict[str, Any],
    ) -> float:
        """
        Apply reward shaping with negative penalty for failures.

        Reward structure:
        - Success (reward=1.0): Apply turn discount
        - Failure (reward=0.0): Apply failure penalty + turn discount

        This ensures failed trajectories get negative signal to learn from.
        """
        # If task failed (reward is 0.0), apply negative penalty
        if reward <= 1e-3:
            reward = self.failure_penalty

        # Apply turn discount (reward decays with more turns)
        return reward * (self.turn_discount ** num_turns)


def main(args):
    """Main training loop."""
    # Load configuration
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Load datasets
    # For AlfWorld task, dataset just needs to return sample IDs
    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )

    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    # Create PPO trainer
    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        # Create workflow that connects to task servers
        train_task_server_url = "http://localhost:5000"  # alfworld-std (standard split)
        eval_task_server_url = "http://localhost:5001"   # alfworld-dev (dev split)

        workflow = AlfWorldWorkflow(
            task_server_url=train_task_server_url,
            gconfig=config.gconfig,
            tokenizer=trainer.tokenizer,
            max_turns=35,  # AlfWorld uses 35 max steps
            turn_discount=0.95,  # Penalize taking more turns
            failure_penalty=-0.1,  # Negative reward for failures (vs +1.0 for success)
            rollout_stat_scope="rollout",
            timeout=300.0,  # AlfWorld tasks need time (5 minutes per request)
        )

        # Eval workflow (different server for dev set)
        eval_workflow = AlfWorldWorkflow(
            task_server_url=eval_task_server_url,
            gconfig=config.gconfig.new(temperature=0.6),
            tokenizer=trainer.tokenizer,
            max_turns=35,
            turn_discount=0.95,
            failure_penalty=-0.1,  # Same penalty for consistency
            rollout_stat_scope="eval-rollout",
            timeout=300.0,
        )

        # Start training!
        trainer.train(workflow, eval_workflow)


if __name__ == "__main__":
    main(sys.argv[1:])
