#!/usr/bin/env python3
import pickle
import argparse
import pathlib
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input file name")
    parser.add_argument("-o", "--output", required=True, help="Output file name")
    args = parser.parse_args()

    dat = pathlib.Path(args.input).read_bytes()
    steps = pickle.loads(dat)
    print(len(steps))
    sh = steps[0].obs.shape
    im = Image.new("RGB", (sh[1], sh[0]), (0, 0, 0))
    images = [
        Image.fromarray(step.obs)
        for step in steps
    ]
    im.save(args.output, save_all=True, append_images=images,
            duration=300, loop=0)
