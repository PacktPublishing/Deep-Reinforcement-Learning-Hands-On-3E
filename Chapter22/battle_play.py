#!/usr/bin/env python3
import argparse
import torch
import ptan
from lib import model, data

from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                        help="Model file to load")
    parser.add_argument("--map-size", type=int, default=data.MAP_SIZE,
                        help="Size of the map, default=" + str(data.MAP_SIZE))
    parser.add_argument("--render", default="render/battle.mp4",
                        help="Name of the video file to render, default=render/battle.mp4")
    parser.add_argument("--walls", type=int, default=data.COUNT_WALLS,
                        help="Count of walls, default=" + str(data.COUNT_WALLS))
    parser.add_argument("--a", type=int, default=data.COUNT_BATTLERS,
                        help="Count of tigers, default=" + str(data.COUNT_BATTLERS))
    parser.add_argument("--b", type=int, default=data.COUNT_BATTLERS,
                        help="Count of deer, default=" + str(data.COUNT_BATTLERS))

    args = parser.parse_args()

    env = data.BattleEnv(
        map_size=args.map_size,
        count_walls=args.walls,
        count_a=args.a,
        count_b=args.b,
        render_mode="rgb_array",
    )
    recorder = VideoRecorder(env, args.render)
    net = model.DQNModel(
        env.observation_spaces['a_0'].shape,
        env.action_spaces['a_0'].n,
    )
    net.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    a_agent = ptan.agent.DQNAgent(
        net, ptan.actions.ArgmaxActionSelector())
    b_agent = data.RandomMAgent(env, env.handles[0])

    obs = env.reset()
    recorder.capture_frame()
    total_reward = 0.0
    total_steps = 0

    while env.agents:
        actions = {}
        b_obs = [
            obs[agent_id]
            for agent_id in env.agents
            if agent_id.startswith("a")
        ]
        a_acts, _ = a_agent(b_obs)
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
        recorder.capture_frame()
        total_steps += 1
        for agent_id, reward in rewards.items():
            if agent_id.startswith("a"):
                total_reward += reward

    print("Episode steps: %d" % total_steps)
    print("Total reward: %.3f" % total_reward)
    print("Mean reward: %.3f" % (total_reward / args.a))
    recorder.close()