#!/usr/bin/env python3
import gymnasium as gym
import ptan
import typing as tt

import torch
import torch.optim as optim

from ignite.engine import Engine

from lib import common, dqn_extra

NAME = "04_noisy"
NOISY_SNR_EVERY_ITERS = 100

BEST_PONG = common.Hyperparams(
    env_name="PongNoFrameskip-v4",
    stop_reward=18.0,
    run_name="pong",
    replay_size=100_000,
    replay_initial=10_000,
    target_net_sync=1000,
    epsilon_frames=100_000,
    epsilon_final=0.02,
    learning_rate=7.142520950425814e-05,
    gamma=0.99,
    episodes_to_solve=273,
)



def train(params: common.Hyperparams,
          device: torch.device, extra: dict) -> tt.Optional[int]:
    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)

    net = dqn_extra.NoisyDQN(
        env.observation_space.shape,
        env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.ArgmaxActionSelector()
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
        net.reset_noise()
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        if engine.state.iteration % NOISY_SNR_EVERY_ITERS == 0:
            for layer_idx, sigma_l2 in enumerate(net.noisy_layers_sigma_snr()):
                engine.state.metrics[f'snr_{layer_idx+1}'] = sigma_l2
        return {
            "loss": loss_v.item(),
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME, extra_metrics=('snr_1', 'snr_2'))
    r = engine.run(common.batch_generator(buffer, params.replay_initial,
                                          params.batch_size))
    if r.solved:
        return r.episode


if __name__ == "__main__":
    args = common.argparser().parse_args()
    common.train_or_tune(args, train, BEST_PONG)
