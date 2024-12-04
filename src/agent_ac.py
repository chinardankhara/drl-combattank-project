import torch
import torch.nn as nn
from torch.optim import Adam
import os

from .policy import Policy
from .buffer import ReplayBuffer, Transition
from .utils import DEVICE


class Critic(nn.Module):
    def __init__(self, state_dimension, hidden_dimension):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(state_dimension, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, 1)  
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
    

class Agent:
    def __init__(self, name, state_dimension, num_actions, hidden_dimension, learning_rate, obs_dim, gamma=0.99, checkpoints_dir="./checkpoints", buffer_size=10_000, batch_size=64):
        self.name = name
        self.gamma = gamma
        self.state_dimension = state_dimension
        self.hidden_dimension = hidden_dimension
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.obs_dim = obs_dim  # Observation is 3-channel image with dimensions 250x160
        C, H, W = self.obs_dim
        
        # Actor (Policy) Network
        self.policy = Policy(
            state_dimension, 
            num_actions, 
            hidden_dimension, 
            C, H, W
        ).to(DEVICE)

        # Critic (Value) Network
        self.critic = Critic(state_dimension, hidden_dimension).to(DEVICE)
        
        # Optimizers for actor and critic
        self.policy_optimizer = Adam(self.policy.parameters(), learning_rate)
        self.critic_optimizer = Adam(self.critic.parameters(), learning_rate)
        
        self.buffer = ReplayBuffer(buffer_size, batch_size)  # Stores ("state", "action", "log_prob", "reward", "next_state", "done")
        
        self.total_losses = []
        self.total_rewards = []
        self.score = 0
        self.storage_path = checkpoints_dir
        
    def clear_cache(self):
        self.buffer.clear()
        
    def take_action(self, env, cache=True):
        observation = env.last()[0]
        action, log_prob = self.policy.sample(observation)  # Sample action from policy
        
        # add constraint if fired, penalty
        if action == 1 or action > 9:
            penalty = 0#-0.05
        else:
            penalty = 0
        
        env.step(action)
        next_observation, reward, done, _, _ = env.last()
        self.score += reward
        
        if cache:
            latent = self.policy.observe(observation)#.view(-1)
            next_latent = self.policy.observe(next_observation)#.view(-1)
            # log_prob = log_prob.item()
            self.buffer.push(latent, action, log_prob, next_latent, reward + penalty)
        
        return done
    
    def optimize(self, epochs=1, *args):
        total_loss = 0.0
        total_reward = 0.0

        for _ in range(epochs):
            # Sample transitions from the buffer
            states, action, log_probs, next_states, rewards = self.buffer.sample()
            dones = rewards > 0

            # Compute value estimates
            state_values = self.critic(states).squeeze()
            next_state_values = self.critic(next_states).squeeze()
            next_state_values = next_state_values * (~dones).float()  
            td_target = rewards + self.gamma * next_state_values
            td_error = td_target - state_values  
            
            # Critic Loss (Mean Squared Error)
            critic_loss = td_error.pow(2).mean()
            
            # Update Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()


            # Compute TD Target
            with torch.no_grad():
                sv = self.critic(states).squeeze()
                next_sv = self.critic(next_states).squeeze()
            td_target = rewards + self.gamma * next_sv
            td_error = td_target - sv      
            
            
            # td_error = (td_error - td_error.mean()) / (td_error.std() + 1e-8)

            # Actor Loss (Policy Gradient with Advantage)
            actor_loss = -(log_probs * td_error).mean()

            # Update Actor
            self.policy_optimizer.zero_grad()
            actor_loss.backward()
            self.policy_optimizer.step()
            
            
            # Accumulate metrics
            total_loss += actor_loss.item()
            total_reward += rewards.sum().item()

        # Compute average loss and reward
        avg_loss = total_loss / epochs
        avg_reward = total_reward / epochs

        # Log metrics
        self.total_losses.append(avg_loss)
        self.total_rewards.append(avg_reward)

        return avg_loss, avg_reward

    # def optimize(self, epochs=1, *args):
        
    #     # states, next_states, log_probs, rewards, dones
    #     # 'state', 'action', 'log_prob', 'next_state', 'reward'
    #     batch_transitions = self.buffer.sample()
    #     batch = Transition(*zip(*batch_transitions))
            
    #     # Convert to tensors
    #     states = torch.stack(batch.state)
    #     next_states = torch.stack(batch.next_state)
    #     log_probs = torch.stack(batch.log_prob)
    #     rewards = torch.tensor(batch.reward, dtype=torch.float32).to("cuda")
    #     dones = rewards == 1

    #     # Compute value estimates
    #     state_values = self.critic(states).squeeze()
    #     next_state_values = self.critic(next_states).squeeze()
    #     next_state_values = next_state_values * (~dones)
        
    #     # Compute TD Target
    #     td_target = rewards + self.gamma * next_state_values
    #     td_error = td_target - state_values  # Advantage is the TD error
        
    #     # Actor Loss (Policy Gradient with Advantage)
    #     actor_loss = -(log_probs * td_error.detach()).mean()  # Detach TD error for actor update
        
    #     # Critic Loss (Mean Squared Error)
    #     critic_loss = td_error.pow(2).mean()
        
    #     # Update Critic
    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #     self.critic_optimizer.step()
        
    #     # Update Actor
    #     self.policy_optimizer.zero_grad()
    #     actor_loss.backward()
    #     self.policy_optimizer.step()
        
    #     # Log metrics
    #     total_loss = actor_loss.item()
    #     total_reward = rewards.sum().item()
    
    #     loss_avg = total_loss / epochs
    #     reward_avg = total_reward / epochs
    #     self.total_losses.append(loss_avg)
    #     self.total_rewards.append(reward_avg)
        
    #     return loss_avg, reward_avg
    
    
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
            'critic': self.critic.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
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
            'obs_dim': self.obs_dim,
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
        obs_dim=checkpoint['obs_dim'],
        gamma=checkpoint['gamma'],
        checkpoints_dir=checkpoint['checkpoints_dir'],
    )
    
    # Load the saved state
    agent.policy.network.load_state_dict(checkpoint['policy_model'])
    agent.policy.vision_enc.load_state_dict(checkpoint['vision_model'])
    # agent.critic.load_state_dict(checkpoint['critic'])
    
    # agent.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
    # agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
    
    # agent.total_losses = checkpoint['total_losses']
    # agent.total_rewards = checkpoint['total_rewards']
    # agent.score = checkpoint['score']
    
    return agent