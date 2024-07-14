#!/usr/bin/env python3
import sys
sys.path.append(".")
import gymnasium as gym
import pathlib
import torch
import argparse

from lib import rlhf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reward", required=True,
                        help="Path to reward model file")
    parser.add_argument("-d", "--dev", default="cuda")
    args = parser.parse_args()
    dev = torch.device(args.dev)

    e = gym.make("SeaquestNoFrameskip-v4")
    p = pathlib.Path(args.reward)
    e = rlhf.RewardModelWrapper(e, p, dev)
    r, _ = e.reset()
    obs, r, is_done, is_tr, extra = e.step(0)
    print(obs.shape)
    print(r)
