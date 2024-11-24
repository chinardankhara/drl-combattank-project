import torch 
import torch.nn as nn 
import numpy as np
from torch.distributions.categorical import Categorical
from typing import Tuple

# local import 
from .utils import DEVICE, tensor

class Eyes(nn.Module):
    def __init__(self, latent_dim, in_channels):
        super(Eyes, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        self.conv1 = nn.Conv2d(self.in_channels, 4, kernel_size=4, stride=4, padding=0)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=4, stride=4, padding=0)
        self.conv3 = nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1)
        self.activation = nn.ReLU()

        # Calculate the flattened size 
        self._to_linear = 4 * 8 * 5  # New dimensions after convolutions
        
        self.fc = nn.Linear(self._to_linear, self.latent_dim)


    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Network(nn.Module):
    def __init__(self, in_dimension: int, hidden_dimension: int, out_dimension: int):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(in_dimension, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, hidden_dimension)
        self.fc3 = nn.Linear(hidden_dimension, out_dimension)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Policy(nn.Module):
    def __init__(
            self,
            latent_dimension: int,
            num_actions: int,
            hidden_dimension: int,
            in_channels, 
            height, 
            width,
    ):
        super(Policy, self).__init__()
        
        self.in_channels = in_channels
        self.height = height 
        self.width = width
        
        self.vision_enc = Eyes(latent_dimension, self.in_channels)#.to(DEVICE)
        self.network = Network(
            latent_dimension, hidden_dimension, num_actions
        )
        
        self.state_dimension = latent_dimension
        self.num_actions = num_actions
        self.hidden_dimension = hidden_dimension
        

    def forward(self, observation: np.ndarray) -> torch.Tensor:
        """
        Forward pass of the Policy network.

        Args:
            observation (np.ndarray): The input observation.

        Returns:
            torch.Tensor: The output logits for each action.

        You can use the self.network to forward the input.
        """
        obs_tensor = tensor(observation).permute(2, 0, 1).view(-1, self.in_channels, self.height, self.width) # CHW
        letent = self.vision_enc(obs_tensor)
        logits = self.network(letent)
        return logits

    def pi(self, state: np.ndarray) -> Categorical:
        """
        Computes the action distribution π(a|s) for a given state.

        Args:
            state (np.ndarray): The input state.

        Returns:
            Categorical: The action distribution.

        
        """
        logits = self.forward(state)
        return Categorical(logits=logits)

    def sample(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """
        Samples an action from the policy and returns the action along with its log probability.

        Args:
            state (np.ndarray): The input state.

        Returns:
            Tuple[int, torch.Tensor]: The sampled action and its log probability.

        """
        pi = self.pi(state)
        action = pi.sample()
        log_prob = pi.log_prob(action)
        return (int(action), log_prob)

    def sample_multiple(self, states: np.ndarray) -> Tuple[int, torch.Tensor]:
        """
        Samples actions for multiple states and returns the actions along with their log probabilities.

        Args:
            states (np.ndarray): The input states.

        Returns:
            Tuple[int, torch.Tensor]: The sampled actions and their log probabilities.

        """
        logits = self.network(tensor(states))
        categorical = Categorical(logits=logits)
        actions = categorical.sample()  
        return [int(action) for action in actions], categorical.log_prob(actions).to(DEVICE) 

    def action(self, state: np.ndarray) -> torch.Tensor:
        """
        Selects an action based on the policy without returning the log probability.

        Args:
            state (np.ndarray): The input state.

        Returns:
            int: The selected action.

        """
        pi = self.pi(state)
        action = pi.sample()
        return int(action)
