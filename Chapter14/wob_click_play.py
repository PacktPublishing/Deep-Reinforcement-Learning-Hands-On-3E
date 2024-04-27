#!/usr/bin/env python3
import argparse
import numpy as np

import torch
import torch.nn.functional as F

from lib import wob, model


ENV_NAME = 'miniwob/click-dialog-v1'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model file to load")
    parser.add_argument("--count", type=int, default=1, help="Count of episodes to play, default=1")
    parser.add_argument("--env", default=ENV_NAME, help="Environment name to solve, default=" + ENV_NAME)
    parser.add_argument("--verbose", default=False, action='store_true', help="Display every step")
    parser.add_argument("--render", default=False, action='store_true', help="Show browser window")
    args = parser.parse_args()

    env_name = args.env
    if not env_name.startswith('miniwob/'):
        env_name = "miniwob/" + env_name

    render_mode = 'human' if args.render else None
    env = wob.MiniWoBClickWrapper.create(env_name, render_mode=render_mode)

    net = model.Model(input_shape=wob.WOB_SHAPE, n_actions=env.action_space.n)
    if args.model:
        net.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

    steps_count = 0
    reward_sum = 0

    for round_idx in range(args.count):
        step_idx = 0
        obs, info = env.reset()
        while True:
            obs_v = torch.tensor(np.expand_dims(obs, axis=0))
            logits_v = net(obs_v)[0]
            policy = F.softmax(logits_v, dim=1).data.numpy()[0]
            action = np.random.choice(len(policy), p=policy)

            obs, reward, done, is_tr, info = env.step(action)
            if args.verbose:
                print(step_idx, reward, done, info)

            step_idx += 1
            reward_sum += reward
            steps_count += 1
            if done:
                print("Round %d done" % round_idx)
                break
    print("Done %d rounds, mean steps %.2f, mean reward %.3f" % (
        args.count, steps_count / args.count, reward_sum / args.count
    ))

    if args.render:
        input("Press enter to close the browser >>> ")
        env.close()

    pass
