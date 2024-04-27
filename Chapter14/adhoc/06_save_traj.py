#!/usr/bin/env python3
"""
Tool saves trajectories from several games using the given model
"""
import sys
sys.path.append(".")
import pathlib
import argparse
import torch
import torch.nn.functional as F
import numpy as np

from lib import model, wob, demos

ENV_NAME = 'miniwob/count-sides-v1'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file")
    parser.add_argument("-o", "--output", required=True, help="Dir to save screenshots")
    args = parser.parse_args()

    env = wob.MiniWoBClickWrapper.create(ENV_NAME)

    net = model.Model(input_shape=wob.WOB_SHAPE, n_actions=env.action_space.n)
    net.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs, info = env.reset()
    step_idx = 0

    while True:
        obs_v = torch.tensor(np.expand_dims(obs, axis=0))
        logits_v = net(obs_v)[0]
        policy = F.softmax(logits_v, dim=1).data.numpy()[0]
        action = np.random.choice(len(policy), p=policy)

        new_obs, reward, done, is_tr, info = env.step(action)
        print(f"{step_idx}: act={action}, r={reward}, done={done}, tr={is_tr}: {info}")

        p = out_dir / f"scr_{step_idx:03d}_act={action}_r={reward:.2f}_d={done:d}_tr={is_tr:d}.png"
        demos.save_obs_image(obs, action, str(p))
        obs = new_obs
        step_idx += 1
        if is_tr or done:
            break
    p = out_dir / f"scr_{step_idx:03d}.png"
    demos.save_obs_image(obs, action=None, file_name=str(p))

    env.close()
