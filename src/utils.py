import os
import gym
import torch
import random
import pygame
import numpy as np
import torch.nn as nn
from loguru import logger
from IPython import display
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
from collections import deque
import random
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional


import time
import json

def loss_fn(
        epoch_log_probability_actions: torch.Tensor, epoch_action_rewards: torch.Tensor
    ) -> torch.Tensor:
    return -1.0 * (epoch_log_probability_actions * epoch_action_rewards).mean()

def loss_fn_dqn(
            value_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> torch.Tensor:
        mse = nn.MSELoss()
        return mse(value_batch, target_batch)


def discount_rewards(epoch_action_rewards, gamma):
    discounted_rewards = []
    R = 0
    for r in reversed(epoch_action_rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)

    return discounted_rewards


def set_seed(seed: int = 42):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"Random seed set as {seed}.")


def device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"Using {device} device.")
    return torch.device(device)


DEVICE = device()

def tensor(x: np.array, type=torch.float32, device=DEVICE) -> torch.Tensor:
    return torch.as_tensor(x, dtype=type, device=device)

def save_episode_as_gif(
    env,
    agents,
    max_steps=3_000,
    save_path="episode.gif",
    fps=10,
):
    """
    Record an episode as a GIF with specified frames per second

    Args:
        env: The environment to run the episode in
        agents: {agent_name: agent}
        max_steps: Maximum number of steps before terminating
        primary_agent: Name of the primary agent
        save_path: Path where to save the GIF
        fps: Frames per second for the output GIF (default: 10)

    Returns:
        str: Path where the GIF was saved
    """
    env.reset()

    # List to store frames
    frames = []

    # Capture initial state
    frame = env.render()
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame)
    frames.append(frame)
    observation, reward, _, _, _ = env.last()

    for step, agent_name in enumerate(env.agent_iter()):
        agent = agents[agent_name]

        if hasattr(agent, 'policy'):
            action, log_prob = agent.policy.sample(observation)
        else:  # for dqn
            action = agent.epsilon_greedy_action(observation)

        env.step(action)
        observation, reward, _, _, _ = env.last()

        # Capture frame after action
        frame = env.render()
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        frames.append(frame)

        if step > max_steps or reward == 1:
            break

    # Save frames as GIF
    if frames:
        # Ensure all frames have the same size as the first frame
        first_frame = frames[0]
        frames = [frame.resize(first_frame.size) for frame in frames]

        # Calculate duration based on FPS (duration in milliseconds = 1000/fps)
        duration = int(1000 / fps)

        # Save the GIF
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,  # Duration between frames in milliseconds
            loop=0,  # 0 means loop indefinitely
        )

    return save_path

@dataclass
class SACTransition:
    """Single transition in replay buffer"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool

class ReplayBuffer:
    """Experience replay buffer for SAC"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition: SACTransition):
        """Add transition to buffer"""
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample random batch of transitions"""
        transitions = random.sample(self.buffer, batch_size)
        
        # Separate transitions into batches
        batch = SACTransition(*zip(*transitions))
        
        # Convert to tensors and stack
        states = tensor(np.stack(batch.state))
        actions = tensor(np.stack(batch.action))
        rewards = tensor(np.stack(batch.reward))
        next_states = tensor(np.stack(batch.next_state))
        dones = tensor(np.stack(batch.done), dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)

def sac_critic_loss_fn(
    q_values: torch.Tensor,
    target_q_values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99
) -> torch.Tensor:
    """
    Soft Q-Learning loss
    
    Args:
        q_values: Current Q-values
        target_q_values: Target Q-values
        rewards: Batch of rewards
        dones: Batch of done flags
        gamma: Discount factor
    """
    # Calculate target using Bellman equation
    target = rewards + gamma * (1 - dones) * target_q_values
    
    # MSE loss between current and target Q-values
    return F.mse_loss(q_values, target.detach())

def sac_actor_loss_fn(
    q_values: torch.Tensor,
    log_probs: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """
    Policy loss with entropy regularization
    
    Args:
        q_values: Q-values for the actions
        log_probs: Log probabilities of the actions
        alpha: Temperature parameter for entropy
    """
    # Policy loss is Q-value minus entropy regularization
    return (alpha * log_probs - q_values).mean()

def soft_update(
    target_params: torch.nn.ParameterList,
    source_params: torch.nn.ParameterList,
    tau: float = 0.005
) -> None:
    """
    Soft update for target networks
    
    Args:
        target_params: Parameters to update
        source_params: Source parameters
        tau: Update rate (default: 0.005)
    """
    for target, source in zip(target_params, source_params):
        target.data.copy_(
            tau * source.data + (1.0 - tau) * target.data
        )