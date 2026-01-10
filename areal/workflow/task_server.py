"""
Generic Task Server Workflow

This workflow connects to any task server implementing the standard
Task Server API, enabling AReaL to train on external environments.

Task servers are independent HTTP services that manage environment state,
execute actions, and compute rewards. AReaL handles model inference,
trajectory collection, and RL training.
"""

import asyncio
import uuid
from typing import Optional, Dict, Any, List

import aiohttp
import torch
from transformers import PreTrainedTokenizerFast

from areal.api.workflow_api import RolloutWorkflow
from areal.api.engine_api import InferenceEngine
from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import ModelRequest
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors

logger = logging.getLogger("TaskServerWorkflow")


class TaskServerWorkflow(RolloutWorkflow):
    """
    Generic workflow that connects to external task servers.

    Task servers must implement the standard Task Server API:
    - POST /episode/start  - Initialize episode, return initial observation
    - POST /episode/step   - Execute action, return (obs, reward, done, info)
    - POST /episode/cancel - Cancel episode

    This workflow handles:
    - Communication with task server
    - Prompt formatting (observation → messages)
    - Action generation (messages → model output)
    - Trajectory tracking (collect token-level data)
    - Tensor construction (trajectory → training data)

    Users can subclass to customize:
    - format_observation_to_messages(): Custom prompt formatting
    - format_agent_output_to_action(): Custom action formatting
    - postprocess_reward(): Custom reward shaping
    """

    def __init__(
        self,
        task_server_url: str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        max_turns: int = 20,
        rollout_stat_scope: str = "rollout",
        turn_discount: float = 1.0,
        dump_dir: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Parameters
        ----------
        task_server_url : str
            Base URL of task server (e.g., "http://localhost:5000")
        gconfig : GenerationHyperparameters
            Generation configuration for model
        tokenizer : PreTrainedTokenizerFast
            Tokenizer for encoding/decoding
        max_turns : int, default=20
            Maximum turns per episode
        rollout_stat_scope : str, default="rollout"
            Scope name for stats logging
        turn_discount : float, default=1.0
            Discount factor for multi-turn rewards (reward *= discount^num_turns)
        dump_dir : str, optional
            Directory to dump rollout traces (for debugging)
        timeout : float, default=60.0
            Timeout in seconds for server requests
        """
        self.task_server_url = task_server_url.rstrip("/")
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.rollout_stat_scope = rollout_stat_scope
        self.turn_discount = turn_discount
        self.dump_dir = dump_dir
        self.timeout = timeout

        # Validate task server is reachable
        self._server_info = None

    async def _init_server_info(self):
        """Lazy initialization of server info."""
        if self._server_info is None:
            try:
                self._server_info = await self._call_server("api/task/info", "GET")
                logger.info(f"Connected to task server: {self._server_info['name']}")
            except Exception as e:
                logger.error(f"Failed to connect to task server {self.task_server_url}: {e}")
                raise

    async def _call_server(
        self,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Call task server API.

        Parameters
        ----------
        endpoint : str
            API endpoint (without leading slash)
        method : str
            HTTP method (GET or POST)
        data : dict, optional
            Request body for POST requests

        Returns
        -------
        dict
            Response JSON

        Raises
        ------
        Exception
            If request fails or server returns error
        """
        url = f"{self.task_server_url}/{endpoint}"

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        # Use TCPConnector with force_close=True to prevent lingering connections
        connector = aiohttp.TCPConnector(force_close=True, limit=10)
        session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        try:
            if method == "POST":
                async with session.post(url, json=data) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(
                            f"Server returned {resp.status}: {error_text}"
                        )
                    result = await resp.json()
            elif method == "GET":
                async with session.get(url, params=data) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(
                            f"Server returned {resp.status}: {error_text}"
                        )
                    result = await resp.json()
            else:
                raise ValueError(f"Unsupported method: {method}")

            return result

        except asyncio.TimeoutError:
            raise Exception(f"Request to {url} timed out after {self.timeout}s")
        except aiohttp.ClientError as e:
            raise Exception(f"Network error calling {url}: {e}")
        finally:
            # Explicitly close session and wait for cleanup
            await session.close()
            # Give connector time to cleanup
            await asyncio.sleep(0)

    def format_observation_to_messages(
        self,
        observation: Dict[str, Any],
        history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Convert task server observation to chat messages.

        Override this method to customize how observations are formatted
        into the chat template.

        Parameters
        ----------
        observation : dict
            Observation from task server, formats:
            - {"type": "text", "content": str} - append text as user message
            - {"type": "messages", "content": list} - replace history with full message list
        history : list
            Current message history

        Returns
        -------
        list
            Updated message history
        """
        if observation["type"] == "text":
            return history + [{"role": "user", "content": observation["content"]}]
        elif observation["type"] == "messages":
            # Full conversation history (includes system prompt, examples, etc.)
            return observation["content"]
        else:
            raise NotImplementedError(
                f"Observation type '{observation['type']}' not supported. "
                f"Override format_observation_to_messages() to handle it."
            )

    def format_agent_output_to_action(
        self,
        agent_output: str,
        info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert agent output to task server action format.

        Override this method to customize action formatting.

        Parameters
        ----------
        agent_output : str
            Decoded text from model
        info : dict
            Info dict from last server response

        Returns
        -------
        dict
            Action dict, format: {"type": "text", "content": str}
        """
        return {"type": "text", "content": agent_output}

    def postprocess_reward(
        self,
        reward: float,
        num_turns: int,
        info: Dict[str, Any],
    ) -> float:
        """
        Apply reward shaping.

        Override this method to customize reward computation.

        Parameters
        ----------
        reward : float
            Raw reward from task server
        num_turns : int
            Number of turns taken
        info : dict
            Final info dict from server

        Returns
        -------
        float
            Shaped reward
        """
        # Apply turn discount
        return reward * (self.turn_discount ** num_turns)

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Workflow:
        1. Start episode on server → get initial observation
        2. Loop until done or max_turns:
           a. Convert observation to messages
           b. Generate action with AReaL engine
           c. Send action to server
           d. Receive (observation, reward, done, info)
        3. Build training tensors from trajectory

        Returns
        -------
        dict
            Training tensors: {input_ids, logprobs, loss_mask, versions,
                              rewards, attention_mask}
        """
        # Initialize server connection if needed
        await self._init_server_info()

        # Extract sample ID
        sample_id = data.get("task_id") or data.get("index")
        if sample_id is None:
            raise ValueError("Data must contain 'task_id' or 'index' field")

        episode_id = None

        try:
            # 1. Start episode
            response = await self._call_server(
                "api/episode/start",
                data={"sample_id": str(sample_id), "config": {}},
            )
            logger.info(f"Eposide stated with sample {sample_id}")
            # logger.info(f"First response: {response}")

            episode_id = response["episode_id"]
            observation = response["observation"]
            info = response.get("info", {})

            # 2. Initialize trajectory tracking
            messages = []
            trajectory_steps = []
            total_reward = 0.0
            done = False

            # Add initial observation
            messages = self.format_observation_to_messages(observation, messages)

            # 3. Multi-turn interaction loop
            for turn in range(self.max_turns):
                if done:
                    break

                # a. Convert messages to input_ids
                # enable_thinking=False disables Qwen3's thinking mode
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    enable_thinking=False,
                )

                # b. Generate action with AReaL engine
                req = ModelRequest(
                    rid=uuid.uuid4().hex,
                    input_ids=list(input_ids),
                    gconfig=self.gconfig.new(n_samples=1),
                    tokenizer=self.tokenizer,
                )

                resp = await engine.agenerate(req)

                # c. Decode agent output
                agent_output = self.tokenizer.decode(resp.output_tokens)

                # d. Track trajectory (token-level data for RL)
                trajectory_steps.append(
                    {
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "output_logprobs": resp.output_logprobs,
                        "output_versions": resp.output_versions,
                        "input_len": len(resp.input_tokens),
                        "output_len": len(resp.output_tokens),
                        "agent_output": agent_output,
                    }
                )

                # e. Format action for server
                action = self.format_agent_output_to_action(agent_output, info)

                # f. Send action to server
                logger.info(f"Running turn {turn + 1} / {self.max_turns}")
                # logger.info(f"messages: {messages}")
                logger.info(f"action: {action}")
                step_response = await self._call_server(
                    "api/episode/step",
                    data={"episode_id": episode_id, "action": action},
                )
                logger.info(f"step_response: {step_response}")

                # g. Get next state
                observation = step_response.get("observation")
                reward = step_response.get("reward", 0.0)
                done = step_response["done"]
                info = step_response.get("info", {})

                total_reward += reward

                # h. Update message history
                messages.append({"role": "assistant", "content": agent_output})
                if not done and observation:
                    messages = self.format_observation_to_messages(
                        observation, messages
                    )

            # 4. Apply reward shaping
            num_turns = len(trajectory_steps)
            shaped_reward = self.postprocess_reward(total_reward, num_turns, info)
            logger.info(f"Eposide reward {sample_id}: {shaped_reward}")

            # 5. Log statistics
            stats_tracker.get(self.rollout_stat_scope).scalar(
                reward=shaped_reward,
                num_turns=num_turns,
                success=info.get("success", 0.0),
            )

            # 6. Build training tensors
            return self._build_training_data(trajectory_steps, shaped_reward)

        except Exception as e:
            logger.error(f"Episode failed for sample {sample_id}: {e}")

            # Cancel episode on error
            if episode_id:
                try:
                    await self._call_server(
                        "api/episode/cancel",
                        data={"episode_id": episode_id},
                    )
                except Exception as cancel_error:
                    logger.warning(f"Failed to cancel episode: {cancel_error}")

            # Return None to reject this trajectory
            return None

    def _build_training_data(
        self,
        trajectory_steps: List[Dict[str, Any]],
        reward: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert trajectory to training tensors.

        Concatenates all turns into a single sequence, properly handling
        prompt token deduplication (since chat history grows each turn).

        Parameters
        ----------
        trajectory_steps : list
            List of step data from trajectory
        reward : float
            Final shaped reward for this episode

        Returns
        -------
        dict
            Training tensors compatible with AReaL's PPO trainer
        """
        seq, logprobs, loss_mask, versions = [], [], [], []

        prev_seq_len = 0
        for step in trajectory_steps:
            # Only take NEW prompt tokens (avoid duplication of history)
            new_prompt_len = len(step["input_tokens"]) - prev_seq_len

            if new_prompt_len > 0:
                # Add new prompt tokens
                seq += step["input_tokens"][-new_prompt_len:]
                logprobs += [0.0] * new_prompt_len
                loss_mask += [0] * new_prompt_len  # Don't train on prompts
                versions += [-1] * new_prompt_len

            # Add completion tokens
            seq += step["output_tokens"]
            logprobs += step["output_logprobs"]
            loss_mask += [1] * step["output_len"]  # Train on completions
            versions += step["output_versions"]

            prev_seq_len = len(seq)

        # Return as batch of 1
        result = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "rewards": torch.tensor(reward, dtype=torch.float32),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool),
        }

        # Add batch dimension
        result = {k: v.unsqueeze(0) for k, v in result.items()}

        return result
