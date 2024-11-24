import os
import torch
from torch.optim import Adam

# local imports
from .policy import Policy
from .utils import discount_rewards, DEVICE

class Agent:
    def __init__(self, name, state_dimension, num_actions, hidden_dimension, learning_rate, gamma=0.99, obs_dim = (3, 250, 160), checkpoints_dir="./checkpoints"):
        self.name = name
        self.gamma = gamma
        self.state_dimension = state_dimension
        self.hidden_dimension = hidden_dimension
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.obs_dim = obs_dim  # Observation is 3-channel image with dimensions 250x160
        C, H, W = self.obs_dim
        
        self.policy = Policy(
            state_dimension, 
            num_actions, 
            hidden_dimension, 
            C, H, W
        ).to(DEVICE)
        
        self.optimizer = Adam(self.policy.parameters(), learning_rate)
        self.cache = [] # "actions", "log_probs", "rewards"
        self.total_losses  = [] 
        self.total_rewards = [] 
        self.score = 0
        self.storage_path = checkpoints_dir
        
    
    def update_cache(self, *args):
        self.cache.append(args)
    
    def clear_cache(self):
        self.cache = []
        
    def take_action(self, env, cache=True):
        observation = env.last()[0]
        action, log_prob = self.policy.sample(observation)
        
        env.step(action)
        reward = env.last()[1]
        
        self.score += reward
        
        if cache:
            self.update_cache(action, log_prob, reward)
        
        return reward == 1
        
    def optimize(self, loss_fn):
        actions, probs, rewards = zip(*self.cache)
        self.clear_cache()
        
        discounted_rewards = discount_rewards(rewards, self.gamma)
        
        # Convert to tensors
        action_probs_tensor = torch.stack(probs)
        reward_tensor = torch.tensor(discounted_rewards, dtype=torch.float32)
        
        # Normalize the rewards (advantage)
        reward_tensor = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-18)
        
        # Calculate the policy loss
        advantage = reward_tensor.to(DEVICE)
        policy_loss = loss_fn(action_probs_tensor, advantage) 
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        loss_item   = policy_loss.item()
        reward_item = reward_tensor.sum().item()
        self.total_losses.append(loss_item)
        self.total_rewards.append(reward_item)
        
        return loss_item, reward_item
    
    def save(self, path: str=None):
        
        if not path:
            path = os.path.join(self.storage_path, self.name + ".pt")
        
        elif not path.endswith(".pt"):
            path = os.path.join(path, self.name + ".pt")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save({
            'policy_model': self.policy.network.state_dict(),
            'vision_model': self.policy.vision_enc.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_losses': self.total_losses,
            'total_rewards': self.total_rewards,
            'state_dimension': self.state_dimension,
            'hidden_dimension': self.hidden_dimension,
            'num_actions': self.num_actions,
            'learning_rate': self.learning_rate,
            'score': self.score,
            'gamma': self.gamma,
            'name': self.name,
            'checkpoints_dir': self.storage_path,
        }, path)
        
    
def load(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved agent found at {path}")
    
    checkpoint = torch.load(path)
    
    # Create a new agent instance with the saved parameters
    agent = Agent(
        name=checkpoint['name'],
        state_dimension=checkpoint['state_dimension'],
        num_actions=checkpoint['num_actions'],
        hidden_dimension=checkpoint['hidden_dimension'],
        learning_rate=checkpoint['learning_rate'],
        gamma=checkpoint['gamma'],
        checkpoints_dir=checkpoint['checkpoints_dir'],
    )
    
    # Load the saved state
    agent.policy.network.load_state_dict(checkpoint['policy_model'])
    agent.policy.vision_enc.load_state_dict(checkpoint['vision_model'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.total_losses = checkpoint['total_losses']
    agent.total_rewards = checkpoint['total_rewards']
    agent.score = checkpoint['score']
    
    return agent