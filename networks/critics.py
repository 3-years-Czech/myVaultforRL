from .utils import layer_init
import torch.nn as nn

class critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.critic_network = nn.Sequential(
            layer_init(nn.Linear(input_dim, 128)),        
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1)
        )
    
    def forward(self,x):
        return self.critic_network(x)

