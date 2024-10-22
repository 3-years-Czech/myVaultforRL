import argparse
import torch 
import numpy as np
import torch.nn as nn
import time
import os
import yaml
from torch.utils.tensorboard import SummaryWriter
import logging

# 配置参数
def create_config_dict():
    """
    Creates and returns a dictionary containing all configuration parameters.

    Parameter Descriptions:
    - num_envs: The number of parallel game environments.
    - num_minibatches: The number of mini-batches.
    - torch_deterministic: If toggled, sets `torch.backends.cudnn.deterministic=True` for deterministic behavior.
    - seed: The seed of the experiment.
    - total_timesteps: Total timesteps of the experiments.
    - learning_rate: The learning rate of the optimizer.
    - anneal_lr: Toggle learning rate annealing for policy and value networks.
    - gamma: The discount factor gamma.
    - gae_lambda: The lambda for the general advantage estimation.
    - load_model: Begin with an existing model.
    - update_epochs: The K epochs to update the policy.
    - clip_coef: The surrogate clipping coefficient.
    - norm_adv: Toggles advantages normalization.
    - clip_vloss: Toggles whether or not to use a clipped loss for the value function, as per the paper.
    - ent_coef: Coefficient of the entropy.
    - vf_coef: Coefficient of the value function.
    - max_grad_norm: The maximum norm for the gradient clipping.
    - global_begin: The global begin value.
    - target_kl: The target KL divergence threshold.
    - num_steps: the number of steps in a epoch
    - num_envs: the number of the envs

    Returns:
    - config: A dictionary containing all configuration parameters.
    """

    config = {
        "num_envs": 1,
        "num_minibatches": 20,
        "torch_deterministic": True,
        "seed": 1,
        "total_timesteps": 100000000,
        "learning_rate": 2.5e-4,
        "anneal_lr": True,
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "load_model": False,
        "update_epochs": 10,
        "clip_coef": 0.1,
        "norm_adv": True,
        "clip_vloss": True,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "global_begin": 0,
        "target_kl": None,
        "num_steps": 100,
        "num_envs": 1,
    }

    return config

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
    def __init__(self, envs, config, device):
        state_dim = envs.single_observation_space.shape[0]
        action_dim = envs.single_action_space.n
        self.critic_network = critic(state_dim,device).to(device)
        self.actor_network = actor(state_dim,action_dim,device).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(),
                                                lr=config['learning_rate'], eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(),
                                                lr=config['learning_rate'], eps=1e-5)
    
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

class replay_buffer(object):
    def __init__(self, config, env, agent_num):
        self.num_steps = config['num_steps']     # 游戏终止时的步长
        self.num_envs = config['num_envs']            # 并行交互的环境数量
        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.agent_num = agent_num
        self.obs = np.zeros((self.num_steps,self.num_envs,self.agent_num,)+env.single_observation_space.shape, dtype=np.float32)
        self.value_pred = np.zeros((self.num_steps,self.num_envs,self.agent_num), dtype=np.float32)
        self.returns = np.zeros_like(self.value_pred)
        
        self.actions = np.zeros((self.num_steps,self.num_envs,self.agent_num)+env.single_action_space.shape, dtype=np.int32)
        self.action_log_probs = np.zeros((self.num_steps,self.num_envs,self.agent_num), dtype=np.float32)
        self.rewards = np.zeros((self.num_steps,self.num_envs,self.agent_num), dtype=np.float32)
        self.step = 0
    
    def insert(self,obs,actions,action_log_probs,value_preds,rewards):
        self.obs[self.step] = obs.copy().reshape(self.obs.shape[1:])
        self.value_pred[self.step] = value_preds.copy().reshape(self.value_pred.shape[1:])
        self.actions[self.step] = actions.copy().reshape(self.actions.shape[1:])
        self.action_log_probs[self.step] = action_log_probs.copy().reshape(self.action_log_probs.shape[1:])
        self.rewards[self.step] = rewards.copy().reshape(self.rewards.shape[1:])
        self.step = (self.step + 1) % self.num_steps  # Update step and wrap around if needed
        
    def compute_returns(self,next_value):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        advantages = np.zeros_like(self.rewards)
        nextvalues = next_value.copy().reshape(self.value_pred.shape[1:])
        lastgaelam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 0.0        # Assuming terminal state is always 0 at the end
            else:
                nextnonterminal = 1.0 - 0  # Assuming no terminal states in the middle
                nextvalues = self.value_pred[t+1]
            delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.value_pred[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        self.returns = advantages + self.value_pred
        return advantages

    def feed_forward_generator(self, advantages, update_idx, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        batch_size = self.num_steps * self.num_envs 
        # assert batch_size >= max(num_mini_batch, mini_batch_size), (
        #     "PPO requires the batch size / number of updates per epoch to be greater "
        #     f"than or equal to the number of PPO epochs. Current model: {batch_size} < {max(num_mini_batch, mini_batch_size)}")
        # Ensure that either num_mini_batch or mini_batch_size is provided
        if num_mini_batch is None:
            assert mini_batch_size is not None, "Must provide either num_mini_batch or mini_batch_size"
            num_mini_batch = int(batch_size / mini_batch_size)
        else:
            assert num_mini_batch > 0
            if mini_batch_size is not None:
                raise ValueError("Cannot set both num_mini_batch and mini_batch_size")
            mini_batch_size = int(batch_size / num_mini_batch)

        # Generate random indices for sampling
        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        # Reshape observations, actions, etc., while preserving agent dimension
        obs = self.obs[:,:,update_idx].reshape(-1, *self.obs.shape[3:])
        actions = self.actions[:,:,update_idx].reshape(-1, *self.actions.shape[3:])
        old_action_log_probs = self.action_log_probs[:,:,update_idx].reshape(-1, *self.action_log_probs.shape[3:])
        value_pred = self.value_pred[:,:,update_idx].reshape(-1, *self.value_pred.shape[3:])
        returns = self.returns[:,:,update_idx].reshape(-1, *self.value_pred.shape[3:])
        
        # Flatten advantages for easier processing
        advantages = advantages[:,:,update_idx].reshape(-1, 1)
        
        # Generate batches
        for indices in sampler:
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            old_action_log_probs_batch = old_action_log_probs[indices]
            advantages_batch = advantages[indices]
            returns_batch = returns[indices]
            value_preds_batch = value_pred[indices]

            # Assuming the rest of the data (e.g., masks, etc.) needs to be reshaped similarly
            base_ret = [obs_batch, old_action_log_probs_batch, actions_batch, advantages_batch, returns_batch, value_preds_batch]
            yield base_ret

class runner():
    def __init__(self, env, exp_name="ppo", result_path="exp", algo_config=create_config_dict(), env_config={}) -> None:
        exp_path = f"{result_path}/{time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime())}_{exp_name}/"
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

        config = {'algorithm':vars(algo_config),'env':env_config}
        with open(exp_path+'config.yml',mode='w') as file:
            yaml.dump(config,file)

        self.writer = SummaryWriter(exp_path+'runs')
        self.writer.add_text(
            "model_hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in algo_config.items()])),
        )
        self.writer.add_text(
            "env_parameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in env_config.items()])),
        )

        self.logger = logging.getLogger('exp')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(exp_path+'log.txt',encoding='utf-8')
        sh = logging.StreamHandler()
        fh.setLevel(logging.DEBUG)
        sh.setLevel(logging.DEBUG)
        ffmt = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
        fh.setFormatter(ffmt)
        sh.setFormatter(ffmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

        modelpath = exp_path+'model/'
        if not os.path.exists(modelpath):
            os.mkdir(modelpath)

        np.random.seed(algo_config['seed'])
        torch.manual_seed(algo_config['seed'])
        torch.backends.cudnn.deterministic = algo_config['torch_deterministic']

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.agent = Agent(env, algo_config, self.device)

        self.buffer = replay_buffer(algo_config, env, 1)

        self.env = env
        self.algo_config = algo_config
        self.env_config = env_config

    def run(self):
        global_step = 0
        start_time = time.time()
        temp_obs, _ = self.env.reset()
        next_obs = temp_obs
        # next_done = np.zeros(args.num_envs*env_config['agent_num'])
        num_updates = self.total_timesteps // args.batch_size