import typing as tt
import torch
from torch import nn
import numpy as np
import gymnasium as gym


TNoise = tt.List[torch.Tensor]


def sample_noise(
        net: nn.Module,
        device: torch.device = torch.device('cpu')
) -> tt.Tuple[TNoise, TNoise]:
    pos = []
    neg = []
    for p in net.parameters():
        noise = np.random.normal(size=p.data.size())
        pos.append(torch.FloatTensor(noise).to(device))
        neg.append(torch.FloatTensor(-noise).to(device))
    return pos, neg


def evaluate(
        env: gym.Env, net: nn.Module,
        get_max_action: bool = True,
        device: torch.device = torch.device('cpu')
) -> tt.Tuple[float, int]:
    obs, _ = env.reset()
    reward = 0.0
    steps = 0
    while True:
        obs_v = torch.FloatTensor(np.expand_dims(obs, 0)).to(device)
        act_v = net(obs_v)
        if get_max_action:
            act = act_v.max(dim=1)[1].data.numpy()[0]
        else:
            act = act_v.data.cpu().numpy()[0]
        obs, r, done, is_tr, _ = env.step(act)
        reward += r
        steps += 1
        if done or is_tr:
            break
    return reward, steps


def eval_with_noise(
        env: gym.Env, net: nn.Module,
        noise: TNoise, noise_std: float,
        get_max_action: bool = True,
        device: torch.device = torch.device("cpu")
) -> tt.Tuple[float, int]:
    old_params = net.state_dict()
    for p, p_n in zip(net.parameters(), noise):
        p.data += noise_std * p_n
    r, s = evaluate(env, net, get_max_action=get_max_action, device=device)
    net.load_state_dict(old_params)
    return r, s
