#!/usr/bin/env python3
"""
Tool saves trajectories from several games using the given model, vectorized version
"""
import sys
sys.path.append(".")
import pathlib
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

from lib import model, wob, demos

ENV_NAME = 'miniwob/count-sides-v1'
N_ENVS = 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file")
    parser.add_argument("-o", "--output", required=True, help="Dir to save screenshots")
    parser.add_argument("-a", type=int, help="If given, this action will be executed")
    args = parser.parse_args()

    envs = [
        lambda: wob.MiniWoBClickWrapper.create(ENV_NAME)
        for _ in range(N_ENVS)
    ]
    env = gym.vector.AsyncVectorEnv(envs)

    net = model.Model(input_shape=wob.WOB_SHAPE, n_actions=env.single_action_space.n)
    net.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    out_dir = pathlib.Path(args.output)
    for i in range(N_ENVS):
        (out_dir / str(i)).mkdir(parents=True, exist_ok=True)

    obs, info = env.reset()
    step_idx = 0
    done_envs = set()

    while len(done_envs) < N_ENVS:
        obs_v = torch.tensor(obs)
        logits_v = net(obs_v)[0]
        policy = F.softmax(logits_v, dim=1).data.numpy()
        actions = [
            np.random.choice(len(policy[i]), p=policy[i]) if args.a is None else args.a
            for i in range(N_ENVS)
        ]

        new_obs, rewards, dones, is_trs, infos = env.step(actions)
        for i, (action, reward, done, is_tr) in enumerate(zip(actions, rewards, dones, is_trs)):
            b_x, b_y = wob.action_to_bins(action)
            print(f"{step_idx}-{i}: act={action}, b={b_x}_{b_y}, r={reward}, done={done}, tr={is_tr}")
            p = out_dir / str(i) / f"scr_{step_idx:03d}_act={action}_b={b_x}-{b_y}_r={reward:.2f}_d={done:d}_tr={is_tr:d}.png"
            demos.save_obs_image(obs[i], action, str(p))
            if is_tr or done:
                done_envs.add(i)
        obs = new_obs
        step_idx += 1

    env.close()
