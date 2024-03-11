#!/usr/bin/env python3
import gymnasium as gym
import ptan
import argparse
import random

import torch
import torch.optim as optim
from torch import nn
import typing as tt
import numpy as np

from ignite.engine import Engine

from lib import dqn_model, common

NAME = "03_double"
STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100


BEST_PONG = common.Hyperparams(
    env_name="PongNoFrameskip-v4",
    stop_reward=18.0,
    run_name="pong",
    replay_size=100_000,
    replay_initial=10_000,
    target_net_sync=1000,
    epsilon_frames=100_000,
    epsilon_final=0.02,
    learning_rate=8.53462665602991e-05,
    gamma=0.98,
    episodes_to_solve=412,
)


def calc_loss_double_dqn(
        batch: tt.List[ptan.experience.ExperienceFirstLast],
        net: nn.Module, tgt_net: nn.Module, gamma: float,
        device: torch.device):
    states, actions, rewards, dones, next_states = \
        common.unpack_batch(batch)

    states_v = torch.as_tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)
    with torch.no_grad():
        next_states_v = torch.as_tensor(next_states).to(device)
        next_state_acts = net(next_states_v).max(1)[1]
        next_state_acts = next_state_acts.unsqueeze(-1)
        next_state_vals = tgt_net(next_states_v).gather(
            1, next_state_acts).squeeze(-1)
        next_state_vals[done_mask] = 0.0
        exp_sa_vals = next_state_vals.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_vals, exp_sa_vals)


def train(params: common.Hyperparams,
          device: torch.device, extra: dict) -> tt.Optional[int]:
    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)

    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(
        epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma, env_seed=common.SEED)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(),
                           lr=params.learning_rate)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        if args.double:
            loss_v = calc_loss_double_dqn(
                batch, net, tgt_net.target_model,
                gamma=params.gamma, device=device)
        else:
            loss_v = common.calc_loss_dqn(
                batch, net, tgt_net.target_model,
                gamma=params.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        if engine.state.iteration % EVAL_EVERY_FRAME == 0:
            eval_states = getattr(engine.state, "eval_states", None)
            if eval_states is None:
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [
                    np.array(transition.state, copy=False)
                    for transition in eval_states
                ]
                eval_states = np.array(eval_states, copy=False)
                engine.state.eval_states = eval_states
            engine.state.metrics["values"] = \
                common.calc_values_of_states(
                    eval_states, net, device)
        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    common.setup_ignite(
        engine, params, exp_source,
        f"{NAME}={args.double}", extra_metrics=('values',))
    r = engine.run(common.batch_generator(
        buffer, params.replay_initial,
        params.batch_size))
    if r.solved:
        return r.episode


if __name__ == "__main__":
    args = common.argparser().parse_args()
    common.train_or_tune(args, train, BEST_PONG)
