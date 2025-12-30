"""
OS Interaction Task Training with AReaL

This script trains a language model on the OS interaction task using
reinforcement learning via AReaL's PPO implementation.

The task server (AgentBench OS task) runs independently, and AReaL
connects to it via HTTP to collect trajectories for training.

Usage:
    # 1. Start task servers (in separate terminals)
    cd AgentBench
    python -m src.server.task_server_adapter os-std --port 5000  # Training (datasets 1-7)
    python -m src.server.task_server_adapter os-dev --port 5001  # Evaluation (dev set)

    # 2. Start training (in another terminal)
    cd AReaL
    python -m areal.launcher.local examples/os_interaction/train.py \
        --config examples/os_interaction/config.yaml \
        experiment_name=os_rl \
        trial_name=run1
"""

import sys
from typing import Dict, Any

from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.experimental.trainer import PPOTrainer
from areal.utils.hf_utils import load_hf_tokenizer
from areal.workflow.task_server import TaskServerWorkflow


class OSTaskWorkflow(TaskServerWorkflow):
    """Custom workflow for OS task with negative rewards for failures."""

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
    # For OS task, dataset just needs to return sample IDs
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
        train_task_server_url = "http://localhost:5000"  # os-std (datasets 1-7)
        eval_task_server_url = "http://localhost:5001"   # os-dev (dev set)

        workflow = OSTaskWorkflow(
            task_server_url=train_task_server_url,
            gconfig=config.gconfig,
            tokenizer=trainer.tokenizer,
            max_turns=8,  # OS task uses 8 rounds
            turn_discount=0.95,  # Penalize taking more turns
            failure_penalty=0,  # Negative reward for failures (vs +1.0 for success)
            rollout_stat_scope="rollout",
            timeout=300.0,  # OS tasks need more time (5 minutes per request)
        )

        # Eval workflow (different server for dev set)
        eval_workflow = OSTaskWorkflow(
            task_server_url=eval_task_server_url,
            gconfig=config.gconfig.new(temperature=0.6),
            tokenizer=trainer.tokenizer,
            max_turns=8,
            turn_discount=0.95,
            failure_penalty=0,  # Same penalty for consistency
            rollout_stat_scope="eval-rollout",
            timeout=300.0,  # OS tasks need more time (5 minutes per request)
        )

        # Start training!
        trainer.train(workflow, eval_workflow)


if __name__ == "__main__":
    main(sys.argv[1:])
