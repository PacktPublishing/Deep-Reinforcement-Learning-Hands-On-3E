#!/usr/bin/env python3
import os

import ptan
import torch
import argparse
from typing import Tuple, List
import ptan.ignite as ptan_ignite

from torch import optim
from types import SimpleNamespace
from lib import data, model, common
from ignite.engine import Engine

# As both deer and tigers are in the same replay buffer, need to increase the sampled batch
TRAIN_BATCH_SIZE = 32

PARAMS = SimpleNamespace(**{
    'run_name':         'tigers-deers',
    'stop_reward':      None,
    'replay_size':      3000000,
    'replay_initial':   300,
    'target_net_sync':  1000,
    'epsilon_frames':   5*10**5,
    'epsilon_start':    1.0,
    'epsilon_final':    0.02,
    'learning_rate':    1e-4,
    'gamma':            0.99,
    'batch_size':       TRAIN_BATCH_SIZE * 10
})


def make_env(args: argparse.Namespace) -> data.magent_parallel_env:
    if args.mode == "forest":
        env = data.ForestEnv(
            map_size=args.map_size,
            count_walls=args.walls,
            count_tigers=args.tigers,
            count_deer=args.deer,
        )
    elif args.mode == 'double_attack':
        env = data.DoubleAttackEnv(
            map_size=args.map_size,
            count_walls=args.walls,
            count_tigers=args.tigers,
            count_deer=args.deer,
        )
    else:
        raise RuntimeError("Wrong mode")
    return env


def test_model(tiger_net: model.DQNModel, deer_net: model.DQNModel, device: torch.device, args: argparse.Namespace) -> Tuple[float, float, int]:
    env = make_env(args)
    tiger_agent = ptan.agent.DQNAgent(
        tiger_net, ptan.actions.ArgmaxActionSelector(),
        device)
    deer_agent = ptan.agent.DQNAgent(
        deer_net, ptan.actions.ArgmaxActionSelector(),
        device)

    obs = env.reset()
    total_steps = 0
    tiger_rewards = 0.0
    deer_rewards = 0.0

    while env.agents:
        actions = {}
        tiger_obs = [
            obs[agent_id]
            for agent_id in env.agents
            if agent_id.startswith("tiger")
        ]
        tiger_acts, _ = tiger_agent(tiger_obs)
        ofs = 0
        for agent_id in env.agents:
            if agent_id.startswith("tiger"):
                actions[agent_id] = tiger_acts[ofs]
                ofs += 1

        deer_obs = [
            obs[agent_id]
            for agent_id in env.agents
            if agent_id.startswith("deer")
        ]
        deer_acts, _ = deer_agent(deer_obs)
        ofs = 0
        for agent_id in env.agents:
            if agent_id.startswith("deer"):
                actions[agent_id] = deer_acts[ofs]
                ofs += 1

        obs, rewards, dones, _, _ = env.step(actions)
        total_steps += 1
        for agent_id, reward in rewards.items():
            if agent_id.startswith("tiger"):
                tiger_rewards += reward
            elif agent_id.startswith("deer"):
                deer_rewards += reward

    return tiger_rewards / env.count_tigers, deer_rewards / env.count_deer, total_steps


def filter_batch(batch: List[data.ExperienceFirstLastMARL],
                 group: str, batch_size: int) -> List[data.ExperienceFirstLastMARL]:
    res = []
    for sample in batch:
        if sample.group != group:
            continue
        res.append(sample)
        if len(res) == batch_size:
            break
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device to train")
    parser.add_argument("-n", "--name", required=True, help="Run name")
    parser.add_argument("--mode", default='forest', choices=['forest', 'double_attack'],
                        help="GridWorld mode, could be 'forest' or 'double_attack' default='forest'")
    parser.add_argument("--map-size", type=int, default=data.MAP_SIZE,
                        help="Size of the map, default=" + str(data.MAP_SIZE))
    parser.add_argument("--walls", type=int, default=data.COUNT_WALLS,
                        help="Count of walls, default=" + str(data.COUNT_WALLS))
    parser.add_argument("--tigers", type=int, default=data.COUNT_TIGERS,
                        help="Count of tigers, default=" + str(data.COUNT_TIGERS))
    parser.add_argument("--deer", type=int, default=data.COUNT_DEER,
                        help="Count of deer, default=" + str(data.COUNT_DEER))
    args = parser.parse_args()
    device = torch.device(args.dev)

    env = make_env(args)
    saves_path = os.path.join("saves", args.name)
    os.makedirs(saves_path, exist_ok=True)

    tiger_net = model.DQNModel(
        env.observation_spaces['tiger_0'].shape,
        env.action_spaces['tiger_0'].n,
    ).to(device)
    tiger_tgt_net = ptan.agent.TargetNet(tiger_net)
    print(tiger_net)

    deer_net = model.DQNModel(
        env.observation_spaces['deer_0'].shape,
        env.action_spaces['deer_0'].n,
    ).to(device)
    deer_tgt_net = ptan.agent.TargetNet(deer_net)

    action_selector = ptan.actions.EpsilonGreedyActionSelector(
        epsilon=PARAMS.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(action_selector, PARAMS)
    tiger_agent = ptan.agent.DQNAgent(tiger_net, action_selector, device)
    deer_agent = ptan.agent.DQNAgent(deer_net, action_selector, device)
    exp_source = data.MAgentExperienceSourceFirstLast(
        env,
        agents_by_group={
            'deer': deer_agent, 'tiger': tiger_agent
        },
        track_reward_group="tiger",
    )
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, PARAMS.replay_size)
    tiger_optimizer = optim.Adam(tiger_net.parameters(), lr=PARAMS.learning_rate)
    deer_optimizer = optim.Adam(deer_net.parameters(), lr=PARAMS.learning_rate)

    def process_batch(engine, batch):
        res = {}
        tiger_batch = filter_batch(batch, group="tiger", batch_size=TRAIN_BATCH_SIZE)
        if tiger_batch:
            tiger_optimizer.zero_grad()
            tiger_loss_v = model.calc_loss_dqn(
                tiger_batch, tiger_net, tiger_tgt_net.target_model,
                ptan.agent.default_states_preprocessor,
                gamma=PARAMS.gamma, device=device)
            tiger_loss_v.backward()
            tiger_optimizer.step()
            res['tiger_loss'] = tiger_loss_v.item()

        deer_batch = filter_batch(batch, group="deer", batch_size=TRAIN_BATCH_SIZE)
        if deer_batch:
            deer_optimizer.zero_grad()
            deer_loss_v = model.calc_loss_dqn(
                deer_batch, deer_net, deer_tgt_net.target_model,
                ptan.agent.default_states_preprocessor,
                gamma=PARAMS.gamma, device=device)
            deer_loss_v.backward()
            deer_optimizer.step()
            res['deer_loss'] = deer_loss_v.item()

        if epsilon_tracker is not None:
            epsilon_tracker.frame(engine.state.iteration)
            res['epsilon'] = action_selector.epsilon
        if engine.state.iteration % PARAMS.target_net_sync == 0:
            tiger_tgt_net.sync()
            deer_tgt_net.sync()
        return res

    engine = Engine(process_batch)
    common.setup_ignite(engine, PARAMS, exp_source, args.name,
                        extra_metrics=('test_tiger_reward', 'test_deer_reward', 'test_steps'),
                        loss_metrics=("tiger_loss", "deer_loss"))
    best_test_tiger_reward = None
    best_test_deer_reward = None

    @engine.on(ptan_ignite.PeriodEvents.ITERS_10000_COMPLETED)
    def test_network(engine):
        tiger_net.train(False)
        deer_net.train(False)
        tiger_reward, deer_reward, steps = test_model(tiger_net, deer_net, device, args)
        tiger_net.train(True)
        deer_net.train(True)
        engine.state.metrics['test_tiger_reward'] = tiger_reward
        engine.state.metrics['test_deer_reward'] = deer_reward
        engine.state.metrics['test_steps'] = steps
        print("Test done: got %.3f tiger and %.3f deer reward after %.2f steps" % (
            tiger_reward, deer_reward, steps
        ))

        global best_test_tiger_reward, best_test_deer_reward
        if best_test_tiger_reward is None:
            best_test_tiger_reward = tiger_reward
        elif best_test_tiger_reward < tiger_reward:
            print("Best test tiger reward updated %.3f -> %.3f, save model" % (
                best_test_tiger_reward, tiger_reward
            ))
            best_test_tiger_reward = tiger_reward
            torch.save(tiger_net.state_dict(), os.path.join(saves_path, "best_tiger_%.3f.dat" % tiger_reward))

        if best_test_deer_reward is None:
            best_test_deer_reward = deer_reward
        elif best_test_deer_reward < deer_reward:
            print("Best test deer reward updated %.3f -> %.3f, save model" % (
                best_test_deer_reward, deer_reward
            ))
            best_test_deer_reward = deer_reward
            torch.save(deer_net.state_dict(), os.path.join(saves_path, "best_deer_%.3f.dat" % deer_reward))

    engine.run(common.batch_generator(buffer, PARAMS.replay_initial,
                                      PARAMS.batch_size))
