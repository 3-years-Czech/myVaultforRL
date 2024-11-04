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
        "num_envs": 100,
        "num_minibatches": 100,
        "torch_deterministic": True,
        "seed": 1,
        "total_timesteps": 1000000,
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
        "num_steps": 10,
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
    output = torch.from_numpy(input).float() if type(input) == np.ndarray else input
    # # 检查输入是否为一维向量
    # if isinstance(output, torch.Tensor) and output.dim() == 1:
    #     output = output.unsqueeze(0)  # 将一维向量转换为二维张量 (1, n)
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
        state_dim = envs.observation_space.shape[0]
        action_dim = envs.action_space.n
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
        self.obs = np.zeros((self.num_steps,self.num_envs,self.agent_num,)+env.observation_space.shape, dtype=np.float32)
        self.value_pred = np.zeros((self.num_steps,self.num_envs,self.agent_num), dtype=np.float32)
        self.returns = np.zeros_like(self.value_pred)
        
        self.actions = np.zeros((self.num_steps,self.num_envs,self.agent_num)+env.action_space.shape, dtype=np.int32)
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

def run(env, exp_name="ppo", result_path="exp", algo_config=create_config_dict(), env_config={}):
    exp_path = f"{result_path}/{time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime())}_{exp_name}/"
    algo_config['batch_size'] = int(algo_config['num_envs'] * algo_config['num_steps'])
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    config = {'algorithm':algo_config,'env':env_config}
    with open(exp_path+'config.yml',mode='w') as file:
        yaml.dump(config,file)

    writer = SummaryWriter(exp_path+'runs')
    writer.add_text(
        "model_hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in algo_config.items()])),
    )
    writer.add_text(
        "env_parameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in env_config.items()])),
    )

    logger = logging.getLogger('exp')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(exp_path+'log.txt',encoding='utf-8')
    sh = logging.StreamHandler()
    fh.setLevel(logging.DEBUG)
    sh.setLevel(logging.DEBUG)
    ffmt = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    fh.setFormatter(ffmt)
    sh.setFormatter(ffmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    modelpath = exp_path+'model/'
    if not os.path.exists(modelpath):
        os.mkdir(modelpath)

    np.random.seed(algo_config['seed'])
    torch.manual_seed(algo_config['seed'])
    torch.backends.cudnn.deterministic = algo_config['torch_deterministic']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = Agent(env, algo_config, device)

    buffer = replay_buffer(algo_config, env, 1)
    
    global_step = 0
    start_time = time.time()
    temp_obs = env.reset()
    next_obs = temp_obs
    # next_done = np.zeros(args.num_envs*env_config['agent_num'])
    num_updates = algo_config['total_timesteps'] // algo_config['batch_size']

    for update in range(0, num_updates + 1):

        if algo_config['anneal_lr']:
            frac = 1.0 - (update - 1.0) / (num_updates)
            lrnow = frac * algo_config['learning_rate']
            agent.actor_optimizer.param_groups[0]["lr"] = lrnow
            agent.critic_optimizer.param_groups[0]['lr'] = lrnow

        for step in range(0, algo_config['num_steps']):
            global_step += 1 * algo_config['num_envs']
            obs = next_obs

            # ALGO LOGIC: action logic
            actions = np.zeros((algo_config['num_envs'],),dtype=np.int64)
            logprobs = np.zeros((algo_config['num_envs'],1),dtype=np.float32)
            value_preds = np.zeros((algo_config['num_envs'],1),dtype=np.float32)
            with torch.no_grad():
                agent.prep_rollout()
                action, logprob= agent.get_action(obs)
                value_pred = agent.get_value(obs)
            logprobs = logprob.cpu().numpy()
            actions = action.cpu().numpy().ravel()
            value_preds = value_pred.cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, done, info = env.step(actions)
            buffer.insert(obs,actions,logprobs,value_preds,rewards)
            obs = next_obs
        
        next_obs = env.reset()


        # 计算return及advantage
        value_preds = np.zeros((algo_config['num_envs'],1),dtype=np.float32)

        with torch.no_grad():
            agent.prep_rollout()
            value_pred = agent.get_value(next_obs)
            value_preds = value_pred.cpu().numpy()
        advantages = buffer.compute_returns(value_preds)

        writer.add_scalar("charts/rew",np.sum(buffer.rewards)/algo_config['num_envs'],global_step)
        # 优势归一化，并不确定是不是一定需要！！！！
        advantages_copy = advantages.copy()
        # advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for epoch in range(algo_config['update_epochs']):
            agent.prep_training()
            data_generator = buffer.feed_forward_generator(advantages, 0, num_mini_batch=algo_config['num_minibatches'])
    
            for sample in data_generator:
                obs_batch, old_action_log_probs_batch, \
                actions_batch, advatages_batch, returns_batch, \
                value_preds_batch = sample

                old_action_log_probs_batch = check(old_action_log_probs_batch).to(device).unsqueeze(1)
                advatages_batch = check(advatages_batch).to(device)
                returns_batch = check(returns_batch).to(device).unsqueeze(1)
                value_preds_batch = check(value_preds_batch).to(device).unsqueeze(1)
                value_real, action_log_probs, dist_entropy = agent.evaluate_action(obs_batch, actions_batch)
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                pg_loss1 = advatages_batch * ratio
                pg_loss2 = advatages_batch * torch.clamp(ratio, 1.0 - algo_config['clip_coef'], 1.0 + algo_config['clip_coef'])
                pg_loss = - torch.min(pg_loss1, pg_loss2).mean()
                actor_loss = pg_loss - dist_entropy * algo_config['ent_coef']

                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(agent.actor_network.parameters(), algo_config['max_grad_norm'])
                agent.actor_optimizer.step()

                # 更新 critic 网络 使用的变量有 returns_batch 计算好的， value_real 带有梯度的  value_preds_batch 记录下来的不带梯度的
                v_loss_unclipped = (value_real - returns_batch) ** 2
                v_clipped = value_preds_batch + torch.clamp(value_real - value_preds_batch, -algo_config['clip_coef'], algo_config['clip_coef'])
                v_loss_clipped = (v_clipped - returns_batch) ** 2
                v_loss = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = torch.mean(v_loss)
                agent.critic_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(agent.critic_network.parameters(), algo_config['max_grad_norm'])
                agent.critic_optimizer.step()

            writer.add_scalar("charts/learning_rate", agent.actor_optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", dist_entropy.item(), global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            print(f"SPS={int(global_step / (time.time() - start_time))}")

        if (update % (num_updates/10) == 0) or (update==num_updates):
            torch.save(agent.actor_network.state_dict(),modelpath+'actor_model.pth')
            torch.save(agent.critic_network.state_dict(),modelpath+'critic_model.pth')
    writer.close()
    endtime = time.time()
    return endtime-start_time

# Example usage
if __name__ == "__main__":
    from env.number import GuessNumberEnv
    env = GuessNumberEnv(max_number=10, max_steps=10)
    
    time = run(env)
    
    print(f'total time = {time}')

    env.close()