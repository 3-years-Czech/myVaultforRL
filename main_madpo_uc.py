import argparse
import setproctitle
import yaml
import os
import logging
from distutils.util import strtobool
import torch
import numpy as np
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter
import random
import importlib
import supersuit as ss

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=100,
        help="the number of parallel game environments")
    parser.add_argument("--num-minibatches", type=int, default=20,
        help="the number of mini-batches")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=100000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
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
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
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
    parser.add_argument("--dtar_kl", type=float, default=0.01,
        help="kl divergence target")
    parser.add_argument("--kl_scale", type=float, default=1.5,
        help="kl divergence available scale")
    parser.add_argument("--kl_para", type=float, default=2.0,
        help="kl coeff scale")
    parser.add_argument("--sqrt_kl_para", type=float, default=2.0,
        help="sqrt kl coeff scale")
    parser.add_argument("--para_lower_bound", type=int, default=0.001,
        help="para lower bound")
    parser.add_argument("--para_upper_bound", type=int, default=1000,
        help="para upper bound")
    args = parser.parse_args()
    # fmt: on
    return args

def set_envs_config():
    env_config = {
        'agent_num': 8,
        'channel_num':4,
        'payload': 80.0*1060,
        'v2i_threshold_per_second_Hz': 3.0,
        'slot':1.0,
        'period':100.0,
        'AWGN_n0':-114.0,
        'bandwidth':180*1e3,
        'max_power_dbm':30.0,
        'aoi_requirement':10.0,
        'aoi_prob_threshold':0.95,
        'outage_prob_threshold':0.95,
        'power_weight': 10.0,
        'v2v_weight': 0.0,
        'v2i_weight': 10.0,
        'global_weight':1.0,
        'action_level' :8,
        'period_num' : 1,
    }
    return env_config

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

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

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
    def __init__(self, env_config, args, device):
        channel_num = env_config['channel_num']
        action_level = env_config['action_level']
        self.outage_prob_threshold = env_config['outage_prob_threshold']
        self.aoi_prob_threshold = env_config['aoi_prob_threshold']
        self.critic_network = critic((channel_num+1)*3,device).to(device)
        self.actor_network = actor((channel_num+1)*3,channel_num*2*action_level,device).to(device)
        self.lambda_weight1 = 0.0
        self.lambda_weight2 = 0.0
        self.kl_coeff = 1.0
        self.sqrt_coeff = 1.0
        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(),
                                                lr=args.learning_rate, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(),
                                                lr=args.learning_rate, eps=1e-5)
    
    def get_value(self, x):
        return self.critic_network(x)

    def update_suc_weitgh(self,accum_overrun):
        self.lambda_weight2 = self.lambda_weight2 + (accum_overrun+self.outage_prob_threshold-1.0)*10.0
        if self.lambda_weight2<0:
            self.lambda_weight2=0
        return self.lambda_weight2
    
    def update_aoi_weitgh(self,accum_overrun):
        self.lambda_weight1 = self.lambda_weight1 + (accum_overrun+self.aoi_prob_threshold-1.0) * 10.0
        if self.lambda_weight1<0:
            self.lambda_weight1=0
        return self.lambda_weight1
    
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
    def __init__(self, args, env, agent_num):
        self.num_steps = args.num_steps     # 游戏终止时的步长
        self.num_envs = args.num_envs       # 并行交互的环境数量
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
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

if __name__ == "__main__":
    experiment_name = "exp_madpo"
    setproctitle.setproctitle(experiment_name+"@GZH")
    args = parse_args()
    env_config = set_envs_config()
    args.num_steps = int(env_config['period'] / env_config['slot'] * env_config['period_num'])
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.kl_lower = args.dtar_kl / args.kl_scale
    args.kl_upper = args.dtar_kl * args.kl_scale
    args.sqrt_dtar_kl = np.sqrt(args.dtar_kl)
    args.sqrt_kl_lower = args.sqrt_dtar_kl / args.kl_scale
    args.sqrt_kl_upper = args.sqrt_dtar_kl * args.kl_scale
    exp_path = f"exp/{time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime())}_{experiment_name}/"    
    
    # parser.add_argument('--beta_kl', type=float, default=1.0)
    # parser.add_argument('--dtar_kl', type=float, default=0.01)
    # parser.add_argument('--kl_para1', type=float, default=1.5)
    # parser.add_argument('--kl_para2', type=float, default=2.0)

    # parser.add_argument('--beta_sqrt_kl', type=float, default=1.0)
    # parser.add_argument('--dtar_sqrt_kl', type=float, default=0.01)
    # parser.add_argument('--sqrt_kl_para1', type=float, default=1.5)
    # parser.add_argument('--sqrt_kl_para2', type=float, default=2.0)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        
    # 设置参数记录
    config = {'algorithm':vars(args),'env':env_config}
    with open(exp_path+'config.yml',mode='w') as file:
        yaml.dump(config,file)
        
    # 设置 tensorboard记录
    writer = SummaryWriter(exp_path+'runs')
    writer.add_text(
        "model_hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    writer.add_text(
        "env_parameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in env_config.items()])),
    )

    # 设置 logger 记录
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
    
    # 源码保存
    import shutil
    code_path = exp_path+'code/'
    saved_file = ['main_madpo.py',
                'env/trafficEnv_fix.py']
    if not os.path.exists(code_path):
        os.makedirs(code_path)
    for file in saved_file:
        if '/' in file:
            filename = file.split('/')[-1]
        else:
            filename = file
        shutil.copyfile(file,code_path+filename)

    modelpath = exp_path+'model/'
    if not os.path.exists(modelpath):
        os.mkdir(modelpath)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = importlib.import_module(f"env.trafficEnv_fix").v2vEnv(env_config)
    env1 = ss.pettingzoo_env_to_vec_env_v1(env)
    # envs = ss.concat_vec_envs_v1(env1, args.num_envs, num_cpus=args.num_envs, base_class="gymnasium")
    envs = ss.concat_vec_envs_v1(env1, args.num_envs, num_cpus=16, base_class="gymnasium")
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True

    agent = [Agent(env_config,args,device) for _ in range(env_config['agent_num'])]

    buffer = replay_buffer(args,envs,env_config['agent_num'])
    
    # if args.load_model:
    #     loadpath = 'exp/exp_move_random_v8_2024-03-18_09:46:26/model/'
    #     for i in range(env_config['agent_num']):
    #         agent[i].load_state_dict(torch.load(loadpath+'model_'+str(i)+'.pth'))
    #         weights = torch.load(loadpath+'weight_'+str(i)+'.pth')
    #         agent[i].weight1 = weights['weight1']
    #         agent[i].weight2 = weights['weight2']
    #     print('load model successfully!')

    # global_step = 0
    global_step = args.global_begin
    start_time = time.time()
    temp_obs, _ = envs.reset()
    next_obs = temp_obs
    # next_done = np.zeros(args.num_envs*env_config['agent_num'])
    num_updates = args.total_timesteps // args.batch_size

   
    # for update in range(1, num_updates + 1):
    for update in range(global_step//args.batch_size+1, num_updates + 1):

        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / (num_updates)
            lrnow = frac * args.learning_rate
            for i in range(env_config['agent_num']):
                agent[i].actor_optimizer.param_groups[0]["lr"] = lrnow
                agent[i].critic_optimizer.param_groups[0]['lr'] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs = next_obs

            # ALGO LOGIC: action logic
            actions = np.zeros((env_config['agent_num']*args.num_envs,),dtype=np.int64)
            logprobs = np.zeros((env_config['agent_num']*args.num_envs,1),dtype=np.float32)
            value_preds = np.zeros((env_config['agent_num']*args.num_envs,1),dtype=np.float32)
            for i in range(env_config['agent_num']):
                ptr = slice(i,i+env_config['agent_num']*args.num_envs,env_config['agent_num'])
                with torch.no_grad():
                    agent[i].prep_rollout()
                    action, logprob= agent[i].get_action(obs[ptr])
                    value_pred = agent[i].get_value(obs[ptr])
                logprobs[ptr] = logprob.cpu().numpy()
                actions[ptr] = action.cpu().numpy().ravel()
                value_preds[ptr] = value_pred.cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, done, _, info = envs.step(actions)
            performance = np.stack([info[i]['cost'] for i in range(len(info))],axis=0)
            buffer.insert(obs,actions,logprobs,value_preds,rewards)
            obs = next_obs


        avg_power = []
        avg_aoi = []
        avg_suc = []
        avg_suc_aoi = []
        #假设环境的输出信息按照功率、V2V成功率、aoi成功率、aoi排列
        with torch.no_grad():
            for i in range(env_config['agent_num']):
                ptr = slice(i,i+env_config['agent_num']*args.num_envs,env_config['agent_num'])
                avg_power.append(np.mean(performance[ptr,0]))
                avg_suc.append(np.mean(performance[ptr,1])*args.num_steps/args.num_envs)
                avg_suc_aoi.append(np.mean(performance[ptr,2])*args.num_steps/args.num_envs)
                avg_aoi.append(np.mean(performance[ptr,3]))
        logger.info(f"global_step={global_step}, aoi={sum(avg_aoi)/env_config['agent_num']}, \
                    succ={1.0-sum(avg_suc)/env_config['agent_num']/env_config['period_num']},\
                    aoi_succ={1.0-sum(avg_suc_aoi)/env_config['agent_num']/env_config['period_num']}, \
                    power={sum(avg_power)/env_config['agent_num']}") 
        writer.add_scalar(f"eval/aoi", sum(avg_aoi)/env_config['agent_num'], global_step)
        writer.add_scalar(f"eval/succ", 1.0-sum(avg_suc)/env_config['agent_num']/env_config['period_num'], global_step) 
        writer.add_scalar(f"eval/aoi_succ", 1.0-sum(avg_suc_aoi)/env_config['agent_num']/env_config['period_num'], global_step) 
        writer.add_scalar(f"eval/power", sum(avg_power)/env_config['agent_num'], global_step)  
        for i in range(env_config['agent_num']):
            writer.add_scalar("agent"+str(i)+"/aoi", avg_aoi[i], global_step)
            writer.add_scalar("agent"+str(i)+"/succ", 1.0-avg_suc[i], global_step) 
            writer.add_scalar("agent"+str(i)+"/aoi_succ", 1.0-avg_suc_aoi[i], global_step) 
            writer.add_scalar("agent"+str(i)+"/power", avg_power[i], global_step)   

        # 计算return及advantage
        value_preds = np.zeros((env_config['agent_num']*args.num_envs,1),dtype=np.float32)
        for i in range(env_config['agent_num']):
            ptr = slice(i,i+env_config['agent_num']*args.num_envs,env_config['agent_num'])
            with torch.no_grad():
                agent[i].prep_rollout()
                value_pred = agent[i].get_value(next_obs[ptr])
                value_preds[ptr] = value_pred.cpu().numpy()
        advantages = buffer.compute_returns(value_preds)

        # 优势归一化，并不确定是不是一定需要！！！！
        advantages_copy = advantages.copy()
        # advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for epoch in range(args.update_epochs):
            for idx in range(env_config['agent_num']):
                agent[idx].prep_training()
                data_generator = buffer.feed_forward_generator(advantages, idx, num_mini_batch=args.num_minibatches)
        
                for sample in data_generator:
                    obs_batch, old_action_log_probs_batch, \
                    actions_batch, advatages_batch, returns_batch, \
                    value_preds_batch = sample

                    old_action_log_probs_batch = check(old_action_log_probs_batch).to(device).unsqueeze(1)
                    advatages_batch = check(advatages_batch).to(device)
                    returns_batch = check(returns_batch).to(device).unsqueeze(1)
                    value_preds_batch = check(value_preds_batch).to(device).unsqueeze(1)
                    value_real, action_log_probs, dist_entropy = agent[idx].evaluate_action(obs_batch, actions_batch)
                    ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                    pg_loss1 = advatages_batch * ratio
                    # 根据论文作者理论使用旋度约束已经包含了或者说替代了clip，所以这里不需要使用clip
                    # pg_loss2 = advatages_batch * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                    with torch.no_grad():
                        old_dist = agent[idx].get_probs(obs_batch)
                    new_dist = agent[idx].get_probs(obs_batch)
                    kl = (old_dist * (torch.log(old_dist + 1e-9) - torch.log(new_dist + 1e-9))).sum(dim=1, keepdim=True)
                    sqrt_kl = torch.sqrt(torch.max(kl + 1e-12,1e-12 * torch.ones_like(kl)))
                    kl_coeff = torch.tensor(agent[idx].kl_coeff).to(device)
                    sqrt_kl_coeff = torch.tensor(agent[idx].sqrt_coeff).to(device)
                    pg_loss = - pg_loss1 + kl * kl_coeff + sqrt_kl * sqrt_kl_coeff

                    pg_loss = torch.mean(pg_loss)
                    actor_loss = pg_loss - dist_entropy * args.ent_coef
                    agent[idx].actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(agent[idx].actor_network.parameters(), args.max_grad_norm)
                    agent[idx].actor_optimizer.step()

                    # 更新kl系数
                    with torch.no_grad():
                        new_new_dist = agent[idx].get_probs(obs_batch)
                    new_kl = (old_dist * (torch.log(old_dist + 1e-9) - torch.log(new_new_dist + 1e-9))).sum(dim=1, keepdim=True)
                    term_kl = torch.mean(new_kl)
                    term_sqrt_kl = torch.sqrt(term_kl + 1e-12)
                    if term_kl < args.kl_lower:
                        agent[idx].kl_coeff /= args.kl_para
                        agent[idx].kl_coeff = agent[idx].kl_coeff if agent[idx].kl_coeff > args.para_lower_bound else args.para_lower_bound
                    elif term_kl > args.kl_upper:
                        agent[idx].kl_coeff *= args.kl_para
                        agent[idx].kl_coeff = agent[idx].kl_coeff if agent[idx].kl_coeff < args.para_upper_bound else args.para_upper_bound
                    
                    if term_sqrt_kl < args.sqrt_kl_lower:
                        agent[idx].sqrt_coeff /= args.sqrt_kl_para
                        agent[idx].sqrt_coeff = agent[idx].sqrt_coeff if agent[idx].sqrt_coeff > args.para_lower_bound else args.para_lower_bound
                    elif term_sqrt_kl > args.sqrt_kl_upper:
                        agent[idx].sqrt_coeff *= args.sqrt_kl_para
                        agent[idx].sqrt_coeff = agent[idx].sqrt_coeff if agent[idx].sqrt_coeff < args.para_upper_bound else args.para_upper_bound

                    # 更新 critic 网络 使用的变量有 returns_batch 计算好的， value_real 带有梯度的  value_preds_batch 记录下来的不带梯度的
                    v_loss_unclipped = (value_real - returns_batch) ** 2
                    v_clipped = value_preds_batch + torch.clamp(value_real - value_preds_batch, -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - returns_batch) ** 2
                    v_loss = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = torch.mean(v_loss)
                    agent[idx].critic_optimizer.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(agent[idx].critic_network.parameters(), args.max_grad_norm)
                    agent[idx].critic_optimizer.step()

        if (update % 100 == 0) or (update==num_updates):
            for i in range(env_config['agent_num']):
                torch.save(agent[i].state_dict(),modelpath+'model_'+str(i)+'.pth')
                torch.save({'weight1':agent[i].weight1, 'weight2':agent[i].weight2}, modelpath+'weight_'+str(i)+'.pth')
    envs.close()
    writer.close()
