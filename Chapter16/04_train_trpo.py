#!/usr/bin/env python3
import os
import ptan
import time
import gymnasium as gym
import argparse
from torch.utils.tensorboard.writer import SummaryWriter

from lib import model, trpo, common

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


GAMMA = 0.99
GAE_LAMBDA = 0.95

TRAJECTORY_SIZE = 2049
LEARNING_RATE_CRITIC = 1e-3

TRPO_MAX_KL = 0.01
TRPO_DAMPING = 0.1

TEST_ITERS = 100000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device to use, default=cpu")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-e", "--env", choices=list(common.ENV_PARAMS.keys()),
                        default='cheetah', help="Environment id, default=cheetah")
    parser.add_argument("--mujoco", default=False, action='store_true',
                        help="If given, MuJoCo env will be used instead of PyBullet")
    parser.add_argument("--no-unhealthy", default=False, action='store_true',
                        help="Disable unhealthy checks in MuJoCo env")
    parser.add_argument("--lr", default=LEARNING_RATE_CRITIC, type=float, help="Critic learning rate")
    parser.add_argument("--maxkl", default=TRPO_MAX_KL, type=float, help="Maximum KL divergence")
    args = parser.parse_args()
    device = torch.device(args.dev)

    name = args.name + ("-mujoco" if args.mujoco else "-pybullet")
    save_path = os.path.join("saves", "trpo-" + name)
    os.makedirs(save_path, exist_ok=True)

    extra = {}
    if args.mujoco and args.no_unhealthy:
        extra['terminate_when_unhealthy'] = False
    env_id = common.register_env(args.env, args.mujoco)
    env = gym.make(env_id, **extra)
    test_env = gym.make(env_id, **extra)

    net_act = model.ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    net_crt = model.ModelCritic(env.observation_space.shape[0]).to(device)
    print(net_act)
    print(net_crt)

    writer = SummaryWriter(comment="-trpo_" + name)
    agent = model.AgentA2C(net_act, device=device)
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)

    opt_crt = optim.Adam(net_crt.parameters(), lr=args.lr)

    trajectory = []
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        for step_idx, exp in enumerate(exp_source):
            rewards_steps = exp_source.pop_rewards_steps()
            if rewards_steps:
                rewards, steps = zip(*rewards_steps)
                writer.add_scalar("episode_steps", np.mean(steps), step_idx)
                tracker.reward(np.mean(rewards), step_idx)

            if step_idx % TEST_ITERS == 0:
                ts = time.time()
                rewards, steps = model.test_net(net_act, test_env, device=device)
                print("Test done in %.2f sec, reward %.3f, steps %d" % (
                    time.time() - ts, rewards, steps))
                writer.add_scalar("test_reward", rewards, step_idx)
                writer.add_scalar("test_steps", steps, step_idx)
                if best_reward is None or best_reward < rewards:
                    if best_reward is not None:
                        print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                        name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                        fname = os.path.join(save_path, name)
                        torch.save(net_act.state_dict(), fname)
                    best_reward = rewards

            trajectory.append(exp)
            if len(trajectory) < TRAJECTORY_SIZE:
                continue

            traj_states = [t[0].state for t in trajectory]
            traj_actions = [t[0].action for t in trajectory]
            traj_states_v = torch.FloatTensor(np.array(traj_states, copy=False)).to(device)
            traj_actions_v = torch.FloatTensor(np.array(traj_actions, copy=False)).to(device)
            traj_adv_v, traj_ref_v = common.calc_adv_ref(
                trajectory, net_crt, traj_states_v,
                GAMMA, GAE_LAMBDA, device=device)
            mu_v = net_act(traj_states_v)
            old_logprob_v = model.calc_logprob(mu_v, net_act.logstd, traj_actions_v)

            # normalize advantages
            traj_adv_v = (traj_adv_v - torch.mean(traj_adv_v)) / torch.std(traj_adv_v)

            # drop last entry from the trajectory, an our adv and ref value calculated without it
            trajectory = trajectory[:-1]
            old_logprob_v = old_logprob_v[:-1].detach()
            traj_states_v = traj_states_v[:-1]
            traj_actions_v = traj_actions_v[:-1]
            sum_loss_value = 0.0
            sum_loss_policy = 0.0
            count_steps = 0

            # critic step
            opt_crt.zero_grad()
            value_v = net_crt(traj_states_v)
            loss_value_v = F.mse_loss(
                value_v.squeeze(-1), traj_ref_v)
            loss_value_v.backward()
            opt_crt.step()

            # actor step
            def get_loss():
                mu_v = net_act(traj_states_v)
                logprob_v = model.calc_logprob(
                    mu_v, net_act.logstd, traj_actions_v)
                dp_v = torch.exp(logprob_v - old_logprob_v)
                action_loss_v = -traj_adv_v.unsqueeze(dim=-1)*dp_v
                return action_loss_v.mean()

            def get_kl():
                mu_v = net_act(traj_states_v)
                logstd_v = net_act.logstd
                mu0_v = mu_v.detach()
                logstd0_v = logstd_v.detach()
                std_v = torch.exp(logstd_v)
                std0_v = std_v.detach()
                v = (std0_v ** 2 + (mu0_v - mu_v) ** 2) / \
                    (2.0 * std_v ** 2)
                kl = logstd_v - logstd0_v + v - 0.5
                return kl.sum(1, keepdim=True)

            trpo.trpo_step(net_act, get_loss, get_kl, args.maxkl,
                           TRPO_DAMPING, device=device)

            trajectory.clear()
            writer.add_scalar("advantage", traj_adv_v.mean().item(), step_idx)
            writer.add_scalar("values", traj_ref_v.mean().item(), step_idx)
            writer.add_scalar("loss_value", loss_value_v.item(), step_idx)

