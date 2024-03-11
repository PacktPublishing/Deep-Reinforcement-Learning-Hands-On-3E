#!/usr/bin/env python3
import pathlib
import gymnasium as gym
import ptan
from ptan.experience import ExperienceFirstLast
import typing as tt
import numpy as np
from ray import tune

import torch
from torch import optim
import torch.nn.functional as F

from ignite.engine import Engine

from lib import common, dqn_extra

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pylab as plt


NAME = "07_distrib"
STATES_TO_EVALUATE = 64
EVAL_EVERY_GAME = 10
IMG_EVERY_GAME = 30

BEST_PONG = common.GAME_PARAMS['pong']


def calc_values_of_states(
        states: np.ndarray, net: dqn_extra.DistributionalDQN,
        device: torch.device) -> float:
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.as_tensor(batch).to(device)
        action_values_v = net.qvals(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return float(np.mean(mean_vals))


def save_state_images(
        path_prefix: str, game_idx: int, states: np.ndarray,
        net: dqn_extra.DistributionalDQN,
        device: torch.device
):
    p = np.arange(dqn_extra.Vmin, dqn_extra.Vmax +
                  dqn_extra.DELTA_Z, dqn_extra.DELTA_Z)
    states_v = torch.as_tensor(states).to(device)
    action_prob = net.apply_softmax(net(states_v)).data.cpu().numpy()
    batch_size, num_actions, _ = action_prob.shape
    for batch_idx in range(batch_size):
        plt.clf()
        for action_idx in range(num_actions):
            plt.subplot(num_actions, 1, action_idx+1)
            plt.bar(p, action_prob[batch_idx, action_idx], width=0.5)
        plt.savefig("%s/%05d_%08d.png" % (
            path_prefix, batch_idx, game_idx))


def calc_loss(
        batch: tt.List[ExperienceFirstLast],
        net: dqn_extra.DistributionalDQN,
        tgt_net: dqn_extra.DistributionalDQN,
        gamma: float, device: torch.device) -> torch.Tensor:
    states, actions, rewards, dones, next_states = \
        common.unpack_batch(batch)
    batch_size = len(batch)

    states_v = torch.as_tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.as_tensor(next_states).to(device)

    # next state distribution
    next_distr_v, next_qvals_v = tgt_net.both(next_states_v)
    next_acts = next_qvals_v.max(1)[1].data.cpu().numpy()
    next_distr = tgt_net.apply_softmax(next_distr_v)
    next_distr = next_distr.data.cpu().numpy()

    next_best_distr = next_distr[range(batch_size), next_acts]

    proj_distr = dqn_extra.distr_projection(
        next_best_distr, rewards, dones, gamma)

    distr_v = net(states_v)
    sa_vals = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(sa_vals, dim=1)
    proj_distr_v = torch.tensor(proj_distr).to(device)

    loss_v = -state_log_sm_v * proj_distr_v
    return loss_v.sum(dim=1).mean()


def train(params: common.Hyperparams,
          device: torch.device, extra: dict) -> tt.Optional[int]:
    img_path = extra.get("img_path")
    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)

    net = dqn_extra.DistributionalDQN(env.observation_space.shape, env.action_space.n).to(device)
    print(net)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(lambda x: net.qvals(x), selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma, env_seed=common.SEED)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = calc_loss(batch, net, tgt_net.target_model,
                           gamma=params.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()

        if img_path is not None:
            eval_states = getattr(engine.state, "eval_states", None)
            if eval_states is None:
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                engine.state.eval_states = np.array(eval_states, copy=False)

            if engine.state.episode % EVAL_EVERY_GAME == 0:
                engine.state.metrics["values"] = \
                    calc_values_of_states(eval_states, net, device=device)

            if engine.state.episode % IMG_EVERY_GAME == 0:
                save_state_images(img_path, engine.state.episode,
                                  eval_states, net, device=device)

        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME,
                        tuner_reward_episode=200)
    r = engine.run(common.batch_generator(
        buffer, params.replay_initial, params.batch_size))
    if r.solved:
        return r.episode


if __name__ == "__main__":
    parser = common.argparser()
    parser.add_argument("--img-path", help="Set image path")
    args = parser.parse_args()
    if args.img_path is not None:
        pathlib.Path(args.img_path).mkdir(parents=True, exist_ok=True)
    common.train_or_tune(
        args, train, BEST_PONG,
        extra_params={
            "img_path": args.img_path,
        },
        extra_space={
            "learning_rate": tune.loguniform(5e-5, 1e-3)
        }
    )

