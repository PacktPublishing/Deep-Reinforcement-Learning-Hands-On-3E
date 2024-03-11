#!/usr/bin/env python3
import gymnasium as gym
import ptan
import typing as tt

import torch
import torch.optim as optim

from ignite.engine import Engine

from lib import dqn_model, common

NAME = "02_n_steps"
DEFAULT_STEPS = 3

BEST_PONG: tt.Dict[int, common.Hyperparams] = {
    2: common.Hyperparams(
        env_name="PongNoFrameskip-v4",
        stop_reward=18.0,
        run_name="pong",
        replay_size=100_000,
        replay_initial=10_000,
        target_net_sync=1000,
        epsilon_frames=100_000,
        epsilon_final=0.02,
        learning_rate=3.9688475460127145e-05,
        gamma=0.98,
        episodes_to_solve=293,
    ),
    3: common.Hyperparams(
        env_name="PongNoFrameskip-v4",
        stop_reward=18.0,
        run_name="pong",
        replay_size=100_000,
        replay_initial=10_000,
        target_net_sync=1000,
        epsilon_frames=100_000,
        epsilon_final=0.02,
        learning_rate=7.82368506822844e-05,
        gamma=0.98,
        episodes_to_solve=260,
    ),
    4: common.Hyperparams(
        env_name="PongNoFrameskip-v4",
        stop_reward=18.0,
        run_name="pong",
        replay_size=100_000,
        replay_initial=10_000,
        target_net_sync=1000,
        epsilon_frames=100_000,
        epsilon_final=0.02,
        learning_rate=6.0739390947756206e-05,
        gamma=0.98,
        episodes_to_solve=290,
    ),
}


def train(params: common.Hyperparams,
          device: torch.device, extra: dict) -> tt.Optional[int]:
    n_steps = extra["n"]

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
        env, agent, gamma=params.gamma, env_seed=common.SEED,
        steps_count=n_steps
    )
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(),
                           lr=params.learning_rate)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss_dqn(
            batch, net, tgt_net.target_model,
            gamma=params.gamma**n_steps, device=device)
        loss_v.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source,
                        f"{NAME}={n_steps}")
    r = engine.run(
        common.batch_generator(buffer, params.replay_initial,
                               params.batch_size)
    )
    if r.solved:
        return r.episode


if __name__ == "__main__":
    parser = common.argparser()
    parser.add_argument(
        "-n", type=int, default=DEFAULT_STEPS,
        help="Steps count on Bellman unroll")
    args = parser.parse_args()

    common.train_or_tune(
        args,
        train,
        best_params=BEST_PONG[args.n],
        extra_params={"n": args.n},
        extra_space={"n": args.n},
    )
