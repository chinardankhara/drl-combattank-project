import torch
from .policy import Policy 

class PolicyZero(Policy):
    def __init__(
            self,
            latent_dimension: int,
            num_actions: int,
            hidden_dimension: int,
            in_channels, 
            height, 
            width,
    ):
        super(PolicyZero, self).__init__(
            latent_dimension,
            num_actions,
            hidden_dimension,
            in_channels, 
            height, 
            width
        )
        
    def sample(self, state):
        return 0, torch.tensor(1.0)