#!/usr/bin/env python3
import gymnasium as gym
import ptan
import collections
from ptan.experience import VectorExperienceSourceFirstLast
import argparse

import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
from ray import tune

from lib import common

MAX_STEPS_TUNE = 4_000_000

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50

REWARD_STEPS = 4
CLIP_GRAD = 0.1

PARAMS_SPACE = {
    "entropy_beta": tune.loguniform(0.001, 0.1),
    "lr": tune.loguniform(1e-5, 1e-2),
    "reward_steps": tune.choice([2, 4, 6, 8]),
    "clip_grad": tune.loguniform(1e-2, 1),
    "batch_size": tune.choice([4, 8, 16, 32, 64, 128, 256]),
    "num_envs": tune.choice([4, 8, 16, 32, 64]),
}


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
    LEARNING_RATE = config['lr']
    BATCH_SIZE = config['batch_size']
    NUM_ENVS = config['num_envs']
    REWARD_STEPS = config['reward_steps']
    CLIP_GRAD = config['clip_grad']
    ENTROPY_BETA = config['entropy_beta']

    env_factories = [
        lambda: ptan.common.wrappers.wrap_dqn(
            gym.make("PongNoFrameskip-v4"))
        for _ in range(NUM_ENVS)
    ]
    env = gym.vector.SyncVectorEnv(env_factories)
    net = common.AtariA2C(env.single_observation_space.shape,
                          env.single_action_space.n).to(device)

    agent = ptan.agent.PolicyAgent(
        lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_source = VectorExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    mean_buf = MeanBuffer(100)
    max_mean_reward = None
    optimizer = optim.Adam(
        net.parameters(), lr=LEARNING_RATE, eps=1e-3)
    batch = []

    for step_idx, exp in enumerate(exp_source):
        if step_idx > MAX_STEPS_TUNE:
            break
        batch.append(exp)

        # handle new rewards
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            for r in new_rewards:
                mean_buf.add(r)
            m = mean_buf.mean()
            if max_mean_reward is None:
                max_mean_reward = m
            elif max_mean_reward < m:
                print(f"{step_idx}: Mean reward "
                      f"updated {max_mean_reward} -> {m}")
                max_mean_reward = m

        if len(batch) < BATCH_SIZE:
            continue

        states_t, actions_t, vals_ref_t = \
            common.unpack_batch(
                batch, net, device=device,
                gamma=GAMMA, reward_steps=REWARD_STEPS)
        batch.clear()

        optimizer.zero_grad()
        logits_t, value_t = net(states_t)
        loss_value_t = F.mse_loss(
            value_t.squeeze(-1), vals_ref_t)

        log_prob_t = F.log_softmax(logits_t, dim=1)
        adv_t = vals_ref_t - value_t.detach()
        log_act_t = log_prob_t[range(BATCH_SIZE), actions_t]
        log_prob_actions_t = adv_t * log_act_t
        loss_policy_t = -log_prob_actions_t.mean()

        prob_t = F.softmax(logits_t, dim=1)
        entropy_loss_t = ENTROPY_BETA * (
                prob_t * log_prob_t).sum(dim=1).mean()

        # calculate policy gradients only
        loss_policy_t.backward(retain_graph=True)

        # apply entropy and value gradients
        loss_v = entropy_loss_t + loss_value_t
        loss_v.backward()
        nn_utils.clip_grad_norm_(
            net.parameters(), CLIP_GRAD)
        optimizer.step()
    env.close()
    return {"max_reward": max_mean_reward}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu",
                        help="Device to use, default=cpu")
    parser.add_argument("--samples", type=int, default=20,
                        help="Count of samples to run")
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
