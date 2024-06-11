import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import common
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v)
                else:
                    self.data[k] = torch.zeros(buffer_shape + v.shape).to(device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)

def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)

class PlainConv(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_dim=256,
                 pool_feature_map=False,
                 last_act=True, # True for ConvBody, False for CNN
                 ):
        super().__init__()
        # assume input image size is 64x64

        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(32, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(64, 128, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(128, 128, 1, padding=0, bias=True), nn.ReLU(inplace=True),
        )

        if pool_feature_map:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=last_act)
        else:
            self.pool = None
            self.fc = make_mlp(128 * 4 * 4, [out_dim], last_act=last_act)

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor):
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class Agent(nn.Module):
    def __init__(self, envs, sample_obs):
        super().__init__()
        # self.feature_net = NatureCNN(sample_obs=sample_obs)
        # latent_size = np.array(envs.unwrapped.single_observation_space.shape).prod()
        self.encoder = PlainConv(in_channels=4, out_dim=256)
        latent_size = self.encoder.out_dim
        state_dim = envs.single_observation_space['state'].shape[0]
        # self.critic = nn.Sequential(
        #     layer_init(nn.Linear(latent_size + state_dim, 512)),
        #     nn.ReLU(inplace=True),
        #     layer_init(nn.Linear(512, 1)),
        # )
        # self.actor_mean = nn.Sequential(
        #     layer_init(nn.Linear(latent_size + state_dim, 512)),
        #     nn.ReLU(inplace=True),
        #     layer_init(nn.Linear(512, np.prod(envs.unwrapped.single_action_space.shape)), std=0.01*np.sqrt(2)),
        # )
        action_dim = np.prod(envs.unwrapped.single_action_space.shape)
        self.critic = make_mlp(latent_size+state_dim, [512, 256, 1], last_act=False)
        self.actor_mean = make_mlp(latent_size+state_dim, [512, 256, action_dim], last_act=False)
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)
    def get_features(self, obs):
        rgb = obs['rgb'].float() / 255.0 # (B, H, W, 3*k)
        depth = obs['depth'].float() / 1000.0 # (B, H, W, 1*k)
        img = torch.cat([rgb, depth], dim=3) # (B, H, W, C)
        img = img.permute(0, 3, 1, 2) # (B, C, H, W)
        feature = self.encoder(img)
        return torch.cat([feature, obs['state']], dim=1)
    def get_value(self, x):
        x = self.get_features(x)
        return self.critic(x)
    def get_action(self, x, deterministic=False):
        x = self.get_features(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None):
        x = self.get_features(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(obs_mode="rgbd", control_mode="pd_joint_delta_pos", render_mode="all", sim_backend="cpu", sensor_configs=dict(width=64, height=64))
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, **env_kwargs)
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, **env_kwargs)

    # rgbd obs mode returns a dict of data, we flatten it so there is just a rgbd key and state key
    envs = FlattenRGBDObservationWrapper(envs)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
        # eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=False, auto_reset=False, **env_kwargs)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=False, auto_reset=False, **env_kwargs)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # ALGO Logic: Storage setup
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    eps_returns = torch.zeros(args.num_envs, dtype=torch.float, device=device)
    eps_lens = np.zeros(args.num_envs)
    place_rew = torch.zeros(args.num_envs, device=device)
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    agent = Agent(envs, sample_obs=next_obs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))
    # evaluate
    print("Evaluating")
    eval_envs.reset()
    viewer = eval_envs.base_env.render_human()
    while True:
        with torch.no_grad():
            eval_obs = common.to_tensor(eval_obs, device=device)
            eval_obs, _, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_action(eval_obs, deterministic=True).cpu())
            viewer = eval_envs.base_env.render_human()
            if viewer.window.key_press("r"):
                eval_obs, _ = eval_envs.reset()
    envs.close()
