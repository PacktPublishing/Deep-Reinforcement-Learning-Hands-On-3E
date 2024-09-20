#!/usr/bin/env python3
import os

import ptan
import torch
import argparse
from typing import Tuple
import ptan.ignite as ptan_ignite

from torch import optim
from types import SimpleNamespace
from lib import data, model, common
from ignite.engine import Engine


PARAMS = SimpleNamespace(**{
    'run_name':         'battle',
    'stop_reward':      None,
    'replay_size':      1000000,
    'replay_initial':   100,
    'target_net_sync':  1000,
    'epsilon_frames':   5*10**5,
    'epsilon_start':    1.0,
    'epsilon_final':    0.02,
    'learning_rate':    1e-4,
    'gamma':            0.99,
    'batch_size':       32
})


def make_env(args: argparse.Namespace) -> data.magent_parallel_env:
    env = data.BattleEnv(
        map_size=args.map_size,
        count_walls=args.walls,
        count_a=args.a,
        count_b=args.b,
    )
    return env


def test_model(a_net: model.DQNModel, device: torch.device,
               args: argparse.Namespace) -> Tuple[float, float, int]:
    env = make_env(args)
    a_agent = ptan.agent.DQNAgent(
        a_net, ptan.actions.ArgmaxActionSelector(),
        device)
    b_agent = data.RandomMAgent(env, env.handles[1])

    obs = env.reset()
    sum_steps = 0
    a_rewards = 0.0
    b_rewards = 0.0

    while env.agents:
        actions = {}
        a_obs = [
            obs[agent_id]
            for agent_id in env.agents
            if agent_id.startswith("a")
        ]
        a_acts, _ = a_agent(a_obs)
        ofs = 0
        for agent_id in env.agents:
            if agent_id.startswith("a"):
                actions[agent_id] = a_acts[ofs]
                ofs += 1
        b_obs = [
            obs[agent_id]
            for agent_id in env.agents
            if agent_id.startswith("b")
        ]
        b_acts, _ = b_agent(b_obs)
        ofs = 0
        for agent_id in env.agents:
            if agent_id.startswith("b"):
                actions[agent_id] = b_acts[ofs]
                ofs += 1

        obs, rewards, dones, _, _ = env.step(actions)
        sum_steps += 1
        for agent_id, reward in rewards.items():
            if agent_id.startswith("a"):
                a_rewards += reward
            if agent_id.startswith("b"):
                b_rewards += reward

    return a_rewards / env.count_a, b_rewards / env.count_b, sum_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device to train")
    parser.add_argument("-n", "--name", required=True, help="Run name")
    parser.add_argument("--map-size", type=int, default=data.MAP_SIZE,
                        help="Size of the map, default=" + str(data.MAP_SIZE))
    parser.add_argument("--walls", type=int, default=data.COUNT_WALLS,
                        help="Count of walls, default=" + str(data.COUNT_WALLS))
    parser.add_argument("--a", type=int, default=data.COUNT_BATTLERS,
                        help="Count of tigers, default=" + str(data.COUNT_BATTLERS))
    parser.add_argument("--b", type=int, default=data.COUNT_BATTLERS,
                        help="Count of deer, default=" + str(data.COUNT_BATTLERS))
    args = parser.parse_args()
    device = torch.device(args.dev)

    env = make_env(args)
    saves_path = os.path.join("saves", args.name)
    os.makedirs(saves_path, exist_ok=True)

    net = model.DQNModel(
        env.observation_spaces['a_0'].shape,
        env.action_spaces['a_0'].n,
    ).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    print(net)

    action_selector = ptan.actions.EpsilonGreedyActionSelector(
        epsilon=PARAMS.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(action_selector, PARAMS)
    a_agent = ptan.agent.DQNAgent(net, action_selector, device)
    b_agent = data.RandomMAgent(env, env.handles[1])
    exp_source = data.MAgentExperienceSourceFirstLast(
        env,
        agents_by_group={
            'a': a_agent, 'b': b_agent
        },
        track_reward_group="a",
        filter_group="a",
    )
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, PARAMS.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=PARAMS.learning_rate)

    def process_batch(engine, batch):
        res = {}
        optimizer.zero_grad()
        loss_v = model.calc_loss_dqn(
            batch, net, tgt_net.target_model,
            ptan.agent.default_states_preprocessor,
            gamma=PARAMS.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        if epsilon_tracker is not None:
            epsilon_tracker.frame(engine.state.iteration)
            res['epsilon'] = action_selector.epsilon
        if engine.state.iteration % PARAMS.target_net_sync == 0:
            tgt_net.sync()
        res['loss'] = loss_v.item()
        return res

    engine = Engine(process_batch)
    common.setup_ignite(engine, PARAMS, exp_source, args.name,
                        extra_metrics=('test_a_reward', 'test_b_reward', 'test_steps'))
    best_test_reward = None

    @engine.on(ptan_ignite.PeriodEvents.ITERS_10000_COMPLETED)
    def test_network(engine):
        net.train(False)
        a_reward, b_reward, steps = test_model(net, device, args)
        net.train(True)
        engine.state.metrics['test_a_reward'] = a_reward
        engine.state.metrics['test_b_reward'] = b_reward
        engine.state.metrics['test_steps'] = steps
        print("Test done: got %.3f reward (a) vs %.3f reward (b) after %.2f steps" % (
            a_reward, b_reward, steps
        ))

        global best_test_reward
        if best_test_reward is None:
            best_test_reward = a_reward
        elif best_test_reward < a_reward:
            print("Best test reward updated %.3f -> %.3f, save model" % (
                best_test_reward, a_reward
            ))
            best_test_reward = a_reward
            torch.save(net.state_dict(), os.path.join(saves_path, "best_%.3f.dat" % a_reward))

    engine.run(common.batch_generator(buffer, PARAMS.replay_initial,
                                      PARAMS.batch_size))
