#!/usr/bin/env python3
import gymnasium as gym
import ptan
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
import typing as tt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 8

REWARD_STEPS = 10


class PGN(nn.Module):
    def __init__(self, input_size: int, n_actions: int):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def smooth(old: tt.Optional[float], val: float,
           alpha: float = 0.95) -> float:
    if old is None:
        return val
    return old * alpha + (1-alpha)*val


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    writer = SummaryWriter(comment="-cartpole-pg")

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = ptan.agent.PolicyAgent(
        net, preprocessor=ptan.agent.float32_preprocessor,
        apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_rewards = []
    step_idx = 0
    done_episodes = 0
    reward_sum = 0.0
    bs_smoothed = entropy = l_entropy = l_policy = l_total = None

    batch_states, batch_actions, batch_scales = [], [], []

    for step_idx, exp in enumerate(exp_source):
        reward_sum += exp.reward
        baseline = reward_sum / (step_idx + 1)
        writer.add_scalar("baseline", baseline, step_idx)
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        batch_scales.append(exp.reward - baseline)

        # handle new rewards
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 450:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        if len(batch_states) < BATCH_SIZE:
            continue

        states_t = torch.as_tensor(
            np.array(batch_states, copy=False))
        batch_actions_t = torch.as_tensor(batch_actions)
        batch_scale_t = torch.as_tensor(batch_scales)

        optimizer.zero_grad()
        logits_t = net(states_t)
        log_prob_t = F.log_softmax(logits_t, dim=1)
        act_probs_t = log_prob_t[range(BATCH_SIZE), batch_actions_t]
        log_prob_actions_t = batch_scale_t * act_probs_t
        loss_policy_t = -log_prob_actions_t.mean()

        prob_t = F.softmax(logits_t, dim=1)
        entropy_t = -(prob_t * log_prob_t).sum(dim=1).mean()
        entropy_loss_t = -ENTROPY_BETA * entropy_t
        loss_t = loss_policy_t + entropy_loss_t

        loss_t.backward()
        optimizer.step()

        # calc KL-div
        new_logits_t = net(states_t)
        new_prob_t = F.softmax(new_logits_t, dim=1)
        kl_div_t = -((new_prob_t / prob_t).log() * prob_t).\
            sum(dim=1).mean()
        writer.add_scalar("kl", kl_div_t.item(), step_idx)

        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0
        for p in net.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1

        bs_smoothed = smooth(
            bs_smoothed,
            float(np.mean(batch_scales))
        )
        entropy = smooth(entropy, entropy_t.item())
        l_entropy = smooth(l_entropy, entropy_loss_t.item())
        l_policy = smooth(l_policy, loss_policy_t.item())
        l_total = smooth(l_total, loss_t.item())

        writer.add_scalar("baseline", baseline, step_idx)
        writer.add_scalar("entropy", entropy, step_idx)
        writer.add_scalar("loss_entropy", l_entropy, step_idx)
        writer.add_scalar("loss_policy", l_policy, step_idx)
        writer.add_scalar("loss_total", l_total, step_idx)
        writer.add_scalar("grad_l2", grad_means / grad_count,
                          step_idx)
        writer.add_scalar("grad_max", grad_max, step_idx)
        writer.add_scalar("batch_scales", bs_smoothed, step_idx)

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

    writer.close()
