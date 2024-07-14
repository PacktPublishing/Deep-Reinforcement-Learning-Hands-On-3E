#!/usr/bin/env python3
import argparse
import pathlib

import gymnasium as gym

from lib import common, rlhf
import ptan

import numpy as np
import torch
import torch.nn.functional as F


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default="SeaquestNoFrameskip-v4",
                        help="Environment name to use, default=SeaquestNoFrameskip-v4")
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    parser.add_argument("-n", type=int, help="Count of experiments to run")
    parser.add_argument("--reward", help="Path to reward model, if not given - use env reward")
    args = parser.parse_args()

    rounds = args.n if args.n is not None else 1
    logs = []

    for round in range(rounds):
        video_folder = args.record
        if args.n is not None:
            video_folder += "-" + str(round)
        env = gym.make(args.env, render_mode='rgb_array')
        if args.record is not None:
            env = gym.wrappers.RecordVideo(env, video_folder=video_folder)
        if args.reward is not None:
            p = pathlib.Path(args.reward)
            env = rlhf.RewardModelWrapper(env, p, dev=torch.device("cpu"))
        env = ptan.common.wrappers.wrap_dqn(env, clip_reward=False)
        print(env)

        net = common.AtariA2C(env.observation_space.shape, env.action_space.n)
        net.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

        obs, _ = env.reset()
        total_reward = 0.0
        total_steps = 0
        while True:
            obs_v = torch.FloatTensor(obs).unsqueeze(0)
            policy_v = net(obs_v)[0]
            policy_v = F.softmax(policy_v, dim=1)
            probs = policy_v[0].detach().cpu().numpy()
            action = np.random.choice(len(probs), p=probs)
            obs, reward, done, is_tr, _ = env.step(action)
            total_reward += reward
            total_steps += 1
            if done or is_tr:
                break
            if total_steps > 100000:
                break
        logs.append("%d: %d steps we got %.3f reward" % (round, total_steps, total_reward))
        env.close()
    print("\n".join(logs))
