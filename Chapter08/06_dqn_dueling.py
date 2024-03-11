#!/usr/bin/env python3
import gymnasium as gym
import ptan
import typing as tt

import torch
from torch import nn
from torch import optim
import numpy as np

from ignite.engine import Engine

from lib import dqn_extra, common

NAME = "06_dueling"

STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100

BEST_PONG = common.GAME_PARAMS['pong']


@torch.no_grad()
def evaluate_states(states: np.ndarray, net: nn.Module,
                    device: torch.device, engine: Engine):
    s_v = torch.as_tensor(states).to(device)
    adv, val = net.adv_val(s_v)
    engine.state.metrics['adv'] = adv.mean().item()
    engine.state.metrics['val'] = val.mean().item()


def train(params: common.Hyperparams,
          device: torch.device, extra: dict) -> tt.Optional[int]:
    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)

    net = dqn_extra.DuelingDQN(env.observation_space.shape,
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
            evaluate_states(eval_states, net, device, engine)
        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME,
                        extra_metrics=('adv', 'val'))
    r = engine.run(common.batch_generator(
        buffer, params.replay_initial, params.batch_size))
    if r.solved:
        return r.episode


if __name__ == "__main__":
    args = common.argparser().parse_args()
    common.train_or_tune(args, train, BEST_PONG)