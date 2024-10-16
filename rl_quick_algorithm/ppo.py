import argparse
import torch 
import numpy as np
import torch.nn as nn

# 配置参数
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-minibatches", type=int, default=20,
        help="the number of mini-batches")
    parser.add_argument("--torch-deterministic", type=bool, default=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=100000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--anneal-lr", type=bool, default=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=1.0,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--load-model", type=bool, default=False,
        help="begin with an exist model")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--norm-adv", type=bool, default=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-vloss", type=bool, default=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--global_begin", type=float, default=0,)
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    # fmt: on
    return args

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

class Categorical(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [torch.nn.init.xavier_uniform_, torch.nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: torch.nn.init.constant_(x, 0), gain)

        self.linear = init_(torch.nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)

class actor(nn.Module):
    def __init__(self, input_dim, action_dim, device):
        super().__init__()
        self.device = device
        self.actor_network = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),       
            nn.ReLU(),
            layer_init(nn.Linear(64, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),       
        )
        self.action_out = Categorical(64, action_dim)

    def forward(self,obs):
        x = check(obs).to(self.device)
        logits = self.actor_network(x)
        action_logits = self.action_out(logits)
        actions = action_logits.sample()
        action_log_probs = action_logits.log_probs(actions)
        return actions, action_log_probs
    
    def get_probs(self, x):
        x = check(x).to(self.device)
        logits = self.actor_network(x)
        action_logits = self.action_out(logits)
        action_probs = action_logits.probs
        return action_probs
    
    def evaluate_actions(self, x, action):
        x = check(x).to(self.device)
        action = check(action).to(self.device)
        logits = self.actor_network(x)
        action_logits = self.action_out(logits)
        action_log_probs = action_logits.log_probs(action)
        dist_entropy = action_logits.entropy().mean()
        return action_log_probs, dist_entropy

class critic(nn.Module):
    def __init__(self, input_dim, device):
        super().__init__()
        self.device = device
        self.critic_network = nn.Sequential(
            layer_init(nn.Linear(input_dim, 128)),           # 4v3c:48  5v4c:75  6v4c:90     7v4c:105     (c+1)*3*v
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1)
        )
    
    def forward(self,x):
        x = check(x).to(self.device)
        return self.critic_network(x)

class Agent(object):
    def __init__(self, envs, args, device):
        state_dim = envs.single_observation_space.shape[0]
        action_dim = envs.single_action_space.n
        self.critic_network = critic(state_dim,device).to(device)
        self.actor_network = actor(state_dim,action_dim,device).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(),
                                                lr=args.learning_rate, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(),
                                                lr=args.learning_rate, eps=1e-5)
    
    def get_value(self, x):
        return self.critic_network(x)
    
    def get_probs(self,x):
        return self.actor_network.get_probs(x)
    
    def get_action(self, x):
        return self.actor_network(x)
    
    def evaluate_action(self,x,actions):
        action_log_probs, dist_entropy = self.actor_network.evaluate_actions(x,actions)
        value = self.critic_network(x)
        return value, action_log_probs, dist_entropy

    def prep_training(self):
        self.critic_network.train()
        self.actor_network.train()

    def prep_rollout(self):
        self.critic_network.eval()
        self.actor_network.eval()

    def actor_zero_grad(self):
        self.actor_optimizer.zero_grad()

    def critic_zero_grad(self):
        self.critic_optimizer.zero_grad()