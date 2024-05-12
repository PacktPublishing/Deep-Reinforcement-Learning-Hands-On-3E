import numpy as np
import torch
import torch.distributions as distr
import gymnasium as gym
import typing as tt
import ptan

from lib import model


ENV_IDS = {
    'cheetah': "HalfCheetahBulletEnv-v0",
    'cheetah-mujoco': "HalfCheetah-v4",
    'ant': "AntBulletEnv-v0",
    'ant-mujoco': "Ant-v4",
}

ENV_PARAMS = {
    'cheetah': ('pybullet_envs.gym_locomotion_envs:HalfCheetahBulletEnv', 1000, 3000.0),
    'ant': ('pybullet_envs.gym_locomotion_envs:AntBulletEnv', 1000, 2500.0),
}


def register_env(name: str, mujoco: bool) -> str:
    if mujoco:
        real_id = ENV_IDS[name + "-mujoco"]
    else:
        # register environment in gymnasium registry, not gym's
        real_id = ENV_IDS[name]
        entry, steps, reward = ENV_PARAMS[name]
        gym.register(
            real_id, entry_point=entry,
            max_episode_steps=steps, reward_threshold=reward,
            apply_api_compatibility=True,
            disable_env_checker=True,
        )
    return real_id


def unpack_batch_a2c(
        batch: tt.List[ptan.experience.ExperienceFirstLast],
        net: model.ModelCritic,
        last_val_gamma: float,
        device: torch.device):
    """
    Convert batch into training tensors
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = torch.FloatTensor(np.array(actions, copy=False)).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
        last_vals_v = net(last_states_v)
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, ref_vals_v


@torch.no_grad()
def unpack_batch_sac(
        batch: tt.List[ptan.experience.ExperienceFirstLast],
        val_net: model.ModelCritic,
        twinq_net: model.ModelSACTwinQ,
        policy_net: model.ModelActor,
        gamma: float, ent_alpha: float,
        device: torch.device):
    """
    Unpack Soft Actor-Critic batch
    """
    states_v, actions_v, ref_q_v = \
        unpack_batch_a2c(batch, val_net, gamma, device)

    # references for the critic network
    mu_v = policy_net(states_v)
    act_dist = distr.Normal(mu_v, torch.exp(policy_net.logstd))
    acts_v = act_dist.sample()
    q1_v, q2_v = twinq_net(states_v, acts_v)
    # element-wise minimum
    ref_vals_v = torch.min(q1_v, q2_v).squeeze() - \
                 ent_alpha * act_dist.log_prob(acts_v).sum(dim=1)
    return states_v, actions_v, ref_vals_v, ref_q_v



def calc_adv_ref(trajectory: tt.List[ptan.experience.Experience],
                 net_crt: model.ModelCritic,
                 states_v: torch.Tensor,
                 gamma: float,
                 gae_lambda: float,
                 device: torch.device):
    """
    By trajectory calculate advantage and 1-step ref value
    :param trajectory: trajectory list
    :param net_crt: critic network
    :param states_v: states tensor
    :return: tuple with advantage numpy array and reference values
    """
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, (exp,) in zip(reversed(values[:-1]),
                                     reversed(values[1:]),
                                     reversed(trajectory[:-1])):
        if exp.done_trunc:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + gamma * next_val - val
            last_gae = delta + gamma * gae_lambda * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(np.array(list(reversed(result_adv)),
                                       copy=False))
    ref_v = torch.FloatTensor(np.array(list(reversed(result_ref)),
                                       copy=False))
    return adv_v.to(device), ref_v.to(device)