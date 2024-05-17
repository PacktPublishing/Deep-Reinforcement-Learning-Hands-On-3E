#!/usr/bin/env python3
import gymnasium as gym
from dataclasses import dataclass
import time
import argparse
import numpy as np
import typing as tt

import torch
import torch.nn as nn
from torch import multiprocessing as mp
from torch import optim

from lib import common

from torch.utils.tensorboard.writer import SummaryWriter

NOISE_STD = 0.001
LEARNING_RATE = 0.001
PROCESSES_COUNT = 6
ITERS_PER_UPDATE = 10
MAX_ITERS = 100000

@dataclass(frozen=True)
class RewardsItem:
    """
    Result item from the worker to master. Fields:
    1. random seed used to generate noise
    2. reward obtained from the positive noise
    3. reward obtained from the negative noise
    4. total amount of steps done
    """
    seed: int
    pos_reward: float
    neg_reward: float
    steps: int


def make_env():
    return gym.make("HalfCheetah-v4")


class Net(nn.Module):
    def __init__(self, obs_size: int, act_size: int,
                 hid_size: int = 64):
        super(Net, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, act_size),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mu(x)


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def train_step(optimizer: optim.Optimizer, net: Net,
               batch_noise: tt.List[common.TNoise],
               batch_reward: tt.List[float],
               writer: SummaryWriter, step_idx: int,
               noise_std: float):
    weighted_noise = None
    norm_reward = compute_centered_ranks(np.array(batch_reward))

    for noise, reward in zip(batch_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += reward * p_n
    m_updates = []
    optimizer.zero_grad()
    for p, p_update in zip(net.parameters(), weighted_noise):
        update = p_update / (len(batch_reward) * noise_std)
        p.grad = -update
        m_updates.append(torch.norm(update))
    writer.add_scalar("update_l2", np.mean(m_updates), step_idx)
    optimizer.step()


def worker_func(params_queue: mp.Queue, rewards_queue: mp.Queue,
                device: torch.device, noise_std: float):
    env = make_env()
    net = Net(env.observation_space.shape[0],
              env.action_space.shape[0]).to(device)
    net.eval()

    while True:
        params = params_queue.get()
        if params is None:
            break
        net.load_state_dict(params)

        for _ in range(ITERS_PER_UPDATE):
            seed = np.random.randint(low=0, high=65535)
            np.random.seed(seed)
            noise, neg_noise = common.sample_noise(
                net, device=device)
            pos_reward, pos_steps = common.eval_with_noise(
                env, net, noise, noise_std,
                get_max_action=False, device=device)
            neg_reward, neg_steps = common.eval_with_noise(
                env, net, neg_noise, noise_std,
                get_max_action=False, device=device)
            rewards_queue.put(RewardsItem(
                seed=seed, pos_reward=pos_reward,
                neg_reward=neg_reward, steps=pos_steps+neg_steps))


if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device to train on, default=cpu")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--noise-std", type=float, default=NOISE_STD)
    parser.add_argument("--iters", type=int, default=MAX_ITERS)
    args = parser.parse_args()
    device = torch.device(args.dev)

    writer = SummaryWriter(comment="-cheetah-es_lr=%.3e_sigma=%.3e" % (args.lr, args.noise_std))
    env = make_env()
    net = Net(env.observation_space.shape[0], env.action_space.shape[0])
    print(net)

    params_queues = [
        mp.Queue(maxsize=1)
        for _ in range(PROCESSES_COUNT)
    ]
    rewards_queue = mp.Queue(maxsize=ITERS_PER_UPDATE)
    workers = []

    for params_queue in params_queues:
        p_args = (params_queue, rewards_queue,
                  device, args.noise_std)
        proc = mp.Process(target=worker_func, args=p_args)
        proc.start()
        workers.append(proc)

    print("All started!")
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    zero_noise, _ = common.sample_noise(net)
    for t in zero_noise:
        t.zero_()

    for step_idx in range(args.iters):
        # broadcasting network params
        params = net.state_dict()
        for q in params_queues:
            q.put(params)

        # waiting for results
        t_start = time.time()
        batch_noise = []
        batch_reward = []
        results = 0
        batch_steps = 0
        while True:
            while not rewards_queue.empty():
                reward = rewards_queue.get_nowait()
                np.random.seed(reward.seed)
                noise, neg_noise = common.sample_noise(net)
                batch_noise.append(noise)
                batch_reward.append(reward.pos_reward)
                batch_noise.append(neg_noise)
                batch_reward.append(reward.neg_reward)
                results += 1
                batch_steps += reward.steps

            if results == PROCESSES_COUNT * ITERS_PER_UPDATE:
                break
            time.sleep(0.01)

        dt_data = time.time() - t_start
        m_reward = np.mean(batch_reward)
        train_step(optimizer, net, batch_noise, batch_reward,
                   writer, step_idx, args.noise_std)
        writer.add_scalar("reward_mean", m_reward, step_idx)
        writer.add_scalar("reward_std", np.std(batch_reward), step_idx)
        writer.add_scalar("reward_max", np.max(batch_reward), step_idx)
        writer.add_scalar("batch_episodes", len(batch_reward), step_idx)
        writer.add_scalar("batch_steps", batch_steps, step_idx)
        speed = batch_steps / (time.time() - t_start)
        writer.add_scalar("speed", speed, step_idx)
        dt_step = time.time() - t_start - dt_data

        print("%d: reward=%.2f, speed=%.2f f/s, data_gather=%.3f, train=%.3f" % (
            step_idx, m_reward, speed, dt_data, dt_step))

    for worker, p_queue in zip(workers, params_queues):
        p_queue.put(None)
        worker.join()
