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

import time
import json

def loss_fn(
        epoch_log_probability_actions: torch.Tensor, epoch_action_rewards: torch.Tensor
    ) -> torch.Tensor:
    return -1.0 * (epoch_log_probability_actions * epoch_action_rewards).mean()


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

        action, log_prob = agent.policy.sample(observation)

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
