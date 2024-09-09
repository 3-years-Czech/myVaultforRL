from .utils import layer_init
import torch.nn as nn
from .distributions import Categorical

class actor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.actor_network = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),       
            nn.ReLU(),
            layer_init(nn.Linear(64, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),       
        )
        self.action_out = Categorical(64, action_dim)

    def forward(self,x):
        logits = self.actor_network(x)
        action_logits = self.action_out(logits)
        actions = action_logits.sample()
        action_log_probs = action_logits.log_probs(actions)
        return actions, action_log_probs
    
    def get_probs(self, x):
        action_logits = self.action_out(x)
        action_probs = action_logits.probs
        return action_probs
    
    def evaluate_actions(self, x, action):
        action_logits = self.action_out(x)
        action_log_probs = action_logits.log_probs(action)
        dist_entropy = action_logits.entropy().mean()
        return action_log_probs, dist_entropy