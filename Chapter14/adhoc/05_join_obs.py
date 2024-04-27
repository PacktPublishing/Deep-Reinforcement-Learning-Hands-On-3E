#!/usr/bin/env python3
import sys
sys.path.append(".")
import argparse
import pathlib
import pickle
import json
from lib import demos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dat", required=True, help="Data file in json format")
    parser.add_argument("--obs", required=True, help="Observations in pickle format")
    parser.add_argument("--save", default=False, action="store_true", help="Save images from observations")
    args = parser.parse_args()

    p = pathlib.Path(args.obs)
    rel_obs = pickle.loads(p.read_bytes())
    p = pathlib.Path(args.dat)
    data = json.loads(p.read_text())

    if args.save:
        for k in sorted(rel_obs.keys()):
            f = f"{k:05d}.png"
            demos.save_obs_image(rel_obs[k]['screenshot'], action=None, file_name=f, transpose=False)
    new_data = demos.join_obs(data, rel_obs)
    pass