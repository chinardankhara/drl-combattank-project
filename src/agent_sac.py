import os
import torch
from torch.optim import Adam

# local imports
from .policy import PolicySAC
from .utils import (
    DEVICE, tensor, ReplayBuffer, SACTransition,
    soft_update, sac_critic_loss_fn, sac_actor_loss_fn
)

class AgentSAC:
    def __init__(
        self,
        name: str,
        state_dimension: int,
        action_dim: int,
        hidden_dimension: int,
        learning_rate: float,
        obs_dim: tuple,
        alpha: float = 0.2,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        checkpoints_dir: str = "./checkpoints"
    ):
        self.name = name
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.state_dimension = state_dimension
        self.hidden_dimension = hidden_dimension
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.obs_dim = obs_dim
        C, H, W = self.obs_dim
        
        # Initialize policy and optimizers
        self.policy = PolicySAC(
            state_dimension, 
            action_dim,
            hidden_dimension,
            C, H, W
        ).to(DEVICE)
        
        # Create optimizers for actor and critics
        self.actor_optimizer = Adam(self.policy.actor.parameters(), learning_rate)
        self.critic_optimizer = Adam(
            list(self.policy.critic1.parameters()) + 
            list(self.policy.critic2.parameters()), 
            learning_rate
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize tracking variables
        self.total_losses = []
        self.total_rewards = []
        self.score = 0
        self.storage_path = checkpoints_dir
    
    def take_action(self, env, training: bool = True) -> bool:
        observation = env.last()[0]
        action, log_prob = self.policy.sample(observation)
        
        env.step(action.detach().cpu().numpy())
        next_observation = env.last()[0]
        reward = env.last()[1]
        done = env.last()[2]
        
        self.score += reward
        
        if training:
            # Store transition in replay buffer
            self.replay_buffer.push(SACTransition(
                observation,
                action.cpu().numpy(),
                reward,
                next_observation,
                done
            ))
        
        return reward == 1
    
    def optimize(self) -> tuple:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
            
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.policy.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
        
        # Current Q-values
        current_q1, current_q2 = self.policy.critic(states, actions)
        
        # Compute critic losses
        critic1_loss = critic_loss_fn(current_q1, next_q, rewards, dones, self.gamma)
        critic2_loss = critic_loss_fn(current_q2, next_q, rewards, dones, self.gamma)
        critic_loss = critic1_loss + critic2_loss
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actions_new, log_probs = self.policy.sample(states)
        q1, q2 = self.policy.critic(states, actions_new)
        q = torch.min(q1, q2)
        actor_loss = actor_loss_fn(q, log_probs, self.alpha)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        soft_update(
            self.policy.critic1_target.parameters(),
            self.policy.critic1.parameters(),
            self.tau
        )
        soft_update(
            self.policy.critic2_target.parameters(),
            self.policy.critic2.parameters(),
            self.tau
        )
        
        # Track metrics
        self.total_losses.append(critic_loss.item())
        self.total_rewards.append(rewards.mean().item())
        
        return critic_loss.item(), rewards.mean().item()
    
    def save(self, path: str = None):
        if not path:
            path = os.path.join(self.storage_path, self.name + "_sac.pt")
        elif not path.endswith(".pt"):
            path = os.path.join(path, self.name + "_sac.pt")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_losses': self.total_losses,
            'total_rewards': self.total_rewards,
            'score': self.score,
            'state_dimension': self.state_dimension,
            'hidden_dimension': self.hidden_dimension,
            'action_dim': self.action_dim,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'tau': self.tau,
            'alpha': self.alpha,
            'name': self.name,
            'checkpoints_dir': self.storage_path,
            'obs_dim': self.obs_dim,
        }, path) 