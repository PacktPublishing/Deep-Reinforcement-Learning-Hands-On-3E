#!/usr/bin/env python3
import gymnasium as gym
import ptan
import numpy as np
import argparse
import collections

import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim
from ray import tune

from lib import common

GAMMA = 0.99
#LEARNING_RATE = 0.0001
#ENTROPY_BETA = 0.02
BATCH_SIZE = 8

#REWARD_STEPS = 10
BASELINE_STEPS = 1000000
#GRAD_L2_CLIP = 0.1
EVAL_STEPS = 1_000_000

ENV_COUNT = 32

PARAMS_SPACE = {
    "lr": tune.loguniform(1e-5, 1e-3),
    "reward_steps": tune.choice([3, 5, 7, 9]),
    "grad_clip": tune.loguniform(1e-2, 1),
    "beta": tune.loguniform(1e-4, 1e-1),
}


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))


class MeanBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.deque = collections.deque(maxlen=capacity)
        self.sum = 0.0

    def add(self, val: float):
        if len(self.deque) == self.capacity:
            self.sum -= self.deque[0]
        self.deque.append(val)
        self.sum += val

    def mean(self) -> float:
        if not self.deque:
            return 0.0
        return self.sum / len(self.deque)


def train(config: dict, device: torch.device) -> dict:
    p_lr = config['lr']
    p_reward_steps = config['reward_steps']
    p_grad_clip = config['grad_clip']
    p_beta = config['beta']

    envs = [make_env() for _ in range(ENV_COUNT)]

    net = common.AtariPGN(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(net)

    agent = ptan.agent.PolicyAgent(net, apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, gamma=GAMMA, steps_count=p_reward_steps)

    optimizer = optim.Adam(net.parameters(), lr=p_lr, eps=1e-3)

    train_step_idx = 0
    baseline_buf = MeanBuffer(BASELINE_STEPS)
    reward_buf = MeanBuffer(100)

    batch_states, batch_actions, batch_scales = [], [], []
    max_reward = None

    for step_idx, exp in enumerate(exp_source):
        if step_idx > EVAL_STEPS:
            break
        baseline_buf.add(exp.reward)
        baseline = baseline_buf.mean()
        batch_states.append(np.array(exp.state, copy=False))
        batch_actions.append(int(exp.action))
        batch_scales.append(exp.reward - baseline)

        # handle new rewards
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            for r in new_rewards:
                reward_buf.add(r)
            max_rw = reward_buf.mean()
            if max_reward is None or max_rw > max_reward:
                print(f"{step_idx}: Max mean reward updated: {max_reward} -> {max_rw:.2f}")
                max_reward = max_rw
        if len(batch_states) < BATCH_SIZE:
            continue

        train_step_idx += 1
        states_v = torch.as_tensor(np.array(batch_states, copy=False)).to(device)
        batch_actions_t = torch.as_tensor(batch_actions).to(device)
        batch_scale_v = torch.as_tensor(batch_scales).to(device)

        optimizer.zero_grad()
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = -p_beta * entropy_v
        loss_v = loss_policy_v + entropy_loss_v
        loss_v.backward()
        nn_utils.clip_grad_norm_(net.parameters(), p_grad_clip)
        optimizer.step()

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()
    for e in envs:
        e.close()
    return {"max_reward": max_reward}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dev", default="cpu", help="Device to use, default=cpu")
    parser.add_argument("--samples", type=int, default=20, help="Count of samples to run")
    args = parser.parse_args()
    device = torch.device(args.dev)

    config = tune.TuneConfig(num_samples=args.samples)
    obj = tune.with_parameters(train, device=device)
    if device.type == 'cuda':
        obj = tune.with_resources(obj, {"gpu": 1})
    tuner = tune.Tuner(
        obj, param_space=PARAMS_SPACE, tune_config=config
    )
    results = tuner.fit()
    best = results.get_best_result(metric="max_reward", mode="max")
    print(best.config)
    print(best.metrics)
