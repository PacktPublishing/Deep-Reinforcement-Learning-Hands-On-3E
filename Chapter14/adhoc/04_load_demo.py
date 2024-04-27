#!/usr/bin/env python3
import sys
sys.path.append(".")
import argparse
import pathlib

from lib import demos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", help="If given, save observations to this image prefix")
    parser.add_argument("-i", "--input", help="Input file to parse",
                        default="demos/click-dialog/click-dialog_0421123844.json")
    args = parser.parse_args()
    p = pathlib.Path(args.input)
    res = demos.load_demo_file(p, gamma=0.99, steps=2)
    for idx, e in enumerate(res):
        print(f"obs={e.state.shape}, act={e.action}, r={e.reward}, last={e.last_state is None}")
        if args.save is not None:
            name = f"{args.save}_{idx:04d}_a={e.action}.png"
            demos.save_obs_image(e.state, e.action, name)
            print("Saved to", name)
