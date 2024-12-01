import os
import torch
import numpy as np
import torch.nn as nn
import random
from torch.optim import Adam

# local imports
from .policy import Policy, ValueFunctionQ
from .buffer import ReplayBuffer, Transition
from .utils import discount_rewards, DEVICE

class Agent_dqn:
    def __init__(self, name, state_dimension, num_actions, hidden_dimension, learning_rate, batch_size, obs_dim, gamma=0.99, checkpoints_dir="./checkpoints"):
        self.name = name
        self.gamma = gamma
        self.state_dimension = state_dimension
        self.hidden_dimension = hidden_dimension
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.obs_dim = obs_dim  # Observation is 3-channel image with dimensions 250x160
        C, H, W = self.obs_dim
        
        self.Q = ValueFunctionQ(
            state_dimension, 
            num_actions, 
            hidden_dimension,
            C, H, W
        ).to(DEVICE)
        
        self.target_Q = ValueFunctionQ(
            state_dimension, 
            num_actions, 
            hidden_dimension, 
            C, H, W
        ).to(DEVICE)

        self.target_Q.load_state_dict(self.Q.state_dict())
        self.target_Q.eval()
        
        self.optimizer = Adam(self.Q.parameters(), learning_rate)
        self.cache = [] # "actions", "log_probs", "rewards"
        self.total_losses  = [] 
        self.total_rewards = [] 
        self.score = 0
        self.storage_path = checkpoints_dir

        self.memory = ReplayBuffer(capacity=100_000, batch_size=batch_size)
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.999
        self.epsilon = self.eps_start
        
    
    def update_cache(self, *args):
        self.cache.append(args)
    
    def clear_cache(self):
        self.cache = []

    def epsilon_greedy_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q_values = self.Q(state_tensor)
            return torch.argmax(q_values).item()
        
    def take_action(self, env, cache=True):
        observation = env.last()[0]
        action = self.epsilon_greedy_action(observation)
        
        env.step(action)
        next_observation, reward, _, _, _ = env.last()
        
        self.score += reward
        
        if cache:
            self.update_cache(action, None, reward)

        self.memory.push(observation, action, next_observation, reward if reward is not None else 0.0)
        return reward == 1
        
    def optimize(self, loss_fn):
        if len(self.memory) < self.batch_size:
            return None
        
        batch_transitions = self.memory.sample()
        batch = Transition(*zip(*batch_transitions))

        states = np.stack(batch.state)
        actions = np.stack(batch.action)
        rewards = np.stack(batch.reward)
        valid_next_states = np.stack(tuple(
            filter(lambda s: s is not None, batch.next_state)
        ))

        nonterminal_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            dtype=torch.bool
        )

        rewards = torch.tensor(rewards)

        targets = torch.zeros(size=(self.memory.batch_size, 1), requires_grad=True)
        with torch.no_grad():
            next_q_vals = torch.zeros(self.memory.batch_size)
            next_q_vals[nonterminal_mask] = self.target_Q(valid_next_states).max(-1)[0].squeeze()
            targets = rewards + (self.gamma * next_q_vals)

        states_tensor = torch.tensor(states, dtype=torch.float64).squeeze()
        actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        q_vals = self.Q(states_tensor).gather(1, actions_tensor)

        loss_val = loss_fn(q_vals, targets)
        self.optimizer.zero_grad()
        loss_val.backward()
        self.optimizer.step()

        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        
        loss_item   = loss_val.item()
        reward_item = rewards.sum().item()
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
            'q_network_state_dict': self.Q.state_dict(),
            'target_network_state_dict': self.target_Q.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_rewards': self.total_rewards,
            'score': self.score,
            'state_dimension': self.state_dimension,
            'hidden_dimension': self.hidden_dimension,
            'num_actions': self.num_actions,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'obs_dim': self.obs_dim,
        }, path)
        
    
def load_dqn(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved agent found at {path}")
    
    checkpoint = torch.load(path)
    
    # Create a new agent instance with the saved parameters
    agent = Agent_dqn(
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
    agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.total_rewards = checkpoint['total_rewards']
    agent.score = checkpoint['score']
    agent.epsilon = checkpoint['epsilon']
    
    return agent
