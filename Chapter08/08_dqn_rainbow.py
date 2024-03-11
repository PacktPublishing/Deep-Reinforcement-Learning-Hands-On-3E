#!/usr/bin/env python3
import gymnasium as gym
import ptan
from ptan.experience import ExperienceFirstLast
import typing as tt
import numpy as np

import torch
from torch import optim, nn

from ignite.engine import Engine

from lib import common, dqn_extra

NAME = "08_rainbow"
N_STEPS = 3
PRIO_REPLAY_ALPHA = 0.6

BEST_PONG = common.Hyperparams(
    env_name="PongNoFrameskip-v4",
    stop_reward=18.0,
    run_name="pong",
    replay_size=100_000,
    replay_initial=10_000,
    target_net_sync=1000,
    epsilon_frames=100_000,
    epsilon_final=0.02,
    learning_rate=8.085421018377671e-05,
    gamma=0.98,
    episodes_to_solve=215,
)


def calc_loss(
        batch: tt.List[ExperienceFirstLast],
        batch_weights: np.ndarray, net: dqn_extra.RainbowDQN,
        tgt_net: dqn_extra.RainbowDQN, gamma: float,
        device: torch.device) -> tt.Tuple[torch.Tensor, np.ndarray]:
    states, actions, rewards, dones, next_states = \
        common.unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_states_v = torch.tensor(next_states).to(device)
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    losses_v = batch_weights_v * (state_action_values - expected_state_action_values) ** 2
    return losses_v.mean(), (losses_v + 1e-5).data.cpu().numpy()


def train(params: common.Hyperparams,
          device: torch.device, extra: dict) -> tt.Optional[int]:
    alpha = extra['alpha']
    n_steps = extra['n_steps']
    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)

    net = dqn_extra.RainbowDQN(env.observation_space.shape,
                        env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma, steps_count=n_steps)
    buffer = dqn_extra.PrioReplayBuffer(
        exp_source, params.replay_size, alpha)
    optimizer = optim.Adam(net.parameters(),
                           lr=params.learning_rate)

    def process_batch(engine, batch_data):
        batch, batch_indices, batch_weights = batch_data
        optimizer.zero_grad()
        loss_v, sample_prios = calc_loss(
            batch, batch_weights, net, tgt_net.target_model,
            gamma=params.gamma**n_steps, device=device)
        loss_v.backward()
        optimizer.step()
        net.reset_noise()
        buffer.update_priorities(batch_indices, sample_prios)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "beta": buffer.update_beta(engine.state.iteration),
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME)
    r = engine.run(common.batch_generator(
        buffer, params.replay_initial,
        params.batch_size))
    if r.solved:
        return r.episode


if __name__ == "__main__":
    args = common.argparser().parse_args()
    common.train_or_tune(
        args, train, BEST_PONG,
        extra_params={
            "alpha": PRIO_REPLAY_ALPHA,
            "n_steps": N_STEPS,
        },
        extra_space={
            "alpha": PRIO_REPLAY_ALPHA,
            "n_steps": N_STEPS,
        }
    )


