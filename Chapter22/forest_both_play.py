#!/usr/bin/env python3
import argparse
import torch
import ptan
from lib import model, data

from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mt", "--model-tiger", required=True,
                        help="Model file to load for tiger agent")
    parser.add_argument("-md", "--model-deer", required=True,
                        help="Model file to load for deer agent")
    parser.add_argument("--map-size", type=int, default=data.MAP_SIZE,
                        help="Size of the map, default=" + str(data.MAP_SIZE))
    parser.add_argument("--render", default="render/video.mp4",
                        help="Name of the video file to render, default=render/video.mp4")
    parser.add_argument("--walls", type=int, default=data.COUNT_WALLS,
                        help="Count of walls, default=" + str(data.COUNT_WALLS))
    parser.add_argument("--tigers", type=int, default=data.COUNT_TIGERS,
                        help="Count of tigers, default=" + str(data.COUNT_TIGERS))
    parser.add_argument("--deer", type=int, default=data.COUNT_DEER,
                        help="Count of deer, default=" + str(data.COUNT_DEER))
    parser.add_argument("--mode", default='forest', choices=['forest', 'double_attack'],
                        help="GridWorld mode, could be 'forest' or 'double_attack', default='forest'")

    args = parser.parse_args()

    if args.mode == 'forest':
        env = data.ForestEnv(
            map_size=args.map_size,
            count_walls=args.walls,
            count_tigers=args.tigers,
            count_deer=args.deer,
            render_mode="rgb_array",
        )
    elif args.mode == 'double_attack':
        env = data.DoubleAttackEnv(
            map_size=args.map_size,
            count_walls=args.walls,
            count_tigers=args.tigers,
            count_deer=args.deer,
            render_mode="rgb_array",
        )
    else:
        raise RuntimeError()
    recorder = VideoRecorder(env, args.render)
    tiger_net = model.DQNModel(
        env.observation_spaces['tiger_0'].shape,
        env.action_spaces['tiger_0'].n,
    )
    tiger_net.load_state_dict(torch.load(args.model_tiger, map_location=torch.device('cpu')))
    tiger_agent = ptan.agent.DQNAgent(
        tiger_net, ptan.actions.ArgmaxActionSelector())

    deer_net = model.DQNModel(
        env.observation_spaces['deer_0'].shape,
        env.action_spaces['deer_0'].n,
    )
    deer_net.load_state_dict(torch.load(args.model_deer, map_location=torch.device('cpu')))
    deer_agent = ptan.agent.DQNAgent(
        deer_net, ptan.actions.ArgmaxActionSelector())

    obs = env.reset()
    recorder.capture_frame()
    total_tiger_reward = 0.0
    total_deer_reward = 0.0
    total_steps = 0

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
        recorder.capture_frame()
        total_steps += 1
        for agent_id, reward in rewards.items():
            if agent_id.startswith("tiger"):
                total_tiger_reward += reward
            if agent_id.startswith("deer"):
                total_deer_reward += reward

    print("Episode steps: %d" % total_steps)
    print("Total tiger reward: %.3f" % total_tiger_reward)
    print("Mean tiger reward: %.3f" % (total_tiger_reward / args.tigers))
    print("Total deer reward: %.3f" % total_deer_reward)
    print("Mean deer reward: %.3f" % (total_deer_reward / args.deer))
    recorder.close()