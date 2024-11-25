import torch 
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Dict, List
from .utils import tensor

def get_vision_heatmap(policy: nn.Module, observation: np.ndarray) -> torch.Tensor:
    """
    Get activation heatmap from the vision encoder without modifying policy forward pass.
    """
    obs_tensor = tensor(observation).permute(2, 0, 1).view(-1, policy.in_channels, policy.height, policy.width)
    
    # Get activations from each conv layer
    x1 = policy.vision_enc.activation(policy.vision_enc.conv1(obs_tensor))
    x2 = policy.vision_enc.activation(policy.vision_enc.conv2(x1))
    x3 = policy.vision_enc.activation(policy.vision_enc.conv3(x2))
    
    # Upsample all activation maps to input size
    h, w = obs_tensor.shape[-2:]
    
    x1_heat = F.interpolate(x1.mean(1, keepdim=True), size=(h, w))
    x2_heat = F.interpolate(x2.mean(1, keepdim=True), size=(h, w))
    x3_heat = F.interpolate(x3.mean(1, keepdim=True), size=(h, w))
    
    combined_heat = (x1_heat + x2_heat + x3_heat) / 3.0
    
    # Normalize to [0,1]
    min_val = combined_heat.min()
    max_val = combined_heat.max()
    heatmap = (combined_heat - min_val) / (max_val - min_val)
    
    return heatmap

def create_heatmap_overlay(frame: np.ndarray, heatmap: torch.Tensor, alpha: float = 0.6) -> np.ndarray:
    """
    Create a heatmap overlay on the frame
    """
    heatmap = heatmap.squeeze().detach().cpu().numpy()
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    overlay = cv2.addWeighted(frame_bgr, 1-alpha, heatmap, alpha, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay

def create_multi_agent_frame(
    frame: np.ndarray,
    agent_heatmaps: Dict[str, torch.Tensor],
    alpha: float = 0.6,
    frame_width: int = 320  # Width for each individual view
) -> Image.Image:
    """
    Create a composite frame showing original view and all agents' perspectives side by side
    """
    n_panels = len(agent_heatmaps) + 1  # +1 for original frame
    
    # Calculate dimensions for the composite image
    frame_height = int(frame_width * frame.shape[0] / frame.shape[1])
    total_width = frame_width * n_panels
    
    # Create blank composite image (including space for titles)
    title_height = 30
    composite = Image.new('RGB', (total_width, frame_height + title_height), (255, 255, 255))
    draw = ImageDraw.Draw(composite)
    
    try:
        # Try to load a system font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # First, add the original frame
    original_frame = Image.fromarray(frame).resize((frame_width, frame_height), Image.Resampling.LANCZOS)
    composite.paste(original_frame, (0, title_height))
    
    # Add "Original" title
    original_title = "Original"
    text_width = draw.textlength(original_title, font=font)
    text_x = (frame_width - text_width) // 2
    draw.text((text_x, 5), original_title, fill=(0, 0, 0), font=font)
    
    # Create each agent's view
    for idx, (agent_name, heatmap) in enumerate(agent_heatmaps.items()):
        # Create heatmap overlay
        frame_with_heatmap = create_heatmap_overlay(frame, heatmap, alpha)
        
        # Resize frame to desired width while maintaining aspect ratio
        frame_img = Image.fromarray(frame_with_heatmap).resize((frame_width, frame_height), Image.Resampling.LANCZOS)
        
        # Calculate position for this frame (offset by 1 due to original frame)
        x_offset = (idx + 1) * frame_width
        
        # Paste frame into composite
        composite.paste(frame_img, (x_offset, title_height))
        
        # Add agent name as title
        text_width = draw.textlength(agent_name, font=font)
        text_x = x_offset + (frame_width - text_width) // 2
        draw.text((text_x, 5), agent_name, fill=(0, 0, 0), font=font)
    
    return composite

def save_multi_agent_episode(
    env,
    agents: Dict[str, object],
    max_steps: int = 3_000,
    save_path: str = "multi_agent_episode.gif",
    fps: int = 10,
    alpha: float = 0.6,
    frame_width: int = 320  # Width for each agent's view
):
    """
    Record an episode as a GIF showing original frame and all agents' attention simultaneously
    """
    env.reset()
    frames: List[Image.Image] = []
    
    # Capture initial state
    frame = env.render()
    if isinstance(frame, np.ndarray):
        observation, reward, _, _, _ = env.last()
        
        # Get initial heatmaps for all agents
        agent_heatmaps = {
            agent_name: get_vision_heatmap(agent.policy, observation)
            for agent_name, agent in agents.items()
        }
        
        # Create composite frame
        composite_frame = create_multi_agent_frame(frame, agent_heatmaps, alpha, frame_width)
        frames.append(composite_frame)
    
    for step, agent_name in enumerate(env.agent_iter()):
        agent = agents[agent_name]
        
        # Get action using original sample method
        action, log_prob = agent.policy.sample(observation)
        
        # Get heatmaps for all agents
        agent_heatmaps = {
            name: get_vision_heatmap(agent.policy, observation)
            for name, agent in agents.items()
        }
        
        env.step(action)
        observation, reward, _, _, _ = env.last()
        
        # Capture frame after action
        frame = env.render()
        if isinstance(frame, np.ndarray):
            composite_frame = create_multi_agent_frame(frame, agent_heatmaps, alpha, frame_width)
            frames.append(composite_frame)
        
        if step > max_steps or reward == 1:
            break
    
    # Save frames as GIF
    if frames:
        first_frame = frames[0]
        duration = int(1000 / fps)
        
        first_frame.save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
    
    return save_path