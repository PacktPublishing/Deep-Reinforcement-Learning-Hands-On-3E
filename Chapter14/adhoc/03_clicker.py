#!/usr/bin/env python3
import sys
import time

sys.path.append(".")
import typing as tt
from lib import wob

RENDER_ENV = True


def close_bins(elems: tt.Tuple[dict, ...]) -> int:
    elem_ids = {e['ref']: e for e in elems}
    close_elem = None
    for e in elems:
        if e['text'] == 'Close':
            close_elem = e
            break
    # need to roll back while ref is negative
    while close_elem['ref'] < 0:
        close_elem = elem_ids[close_elem['parent']]
    print(close_elem)
    x = close_elem['left'][0] #+ close_elem['width'][0] / 2.0
    y = close_elem['top'][0] #+ close_elem['height'][0] / 2.0
    i = int(x // wob.BIN_SIZE)
    j = int((y - wob.Y_OFS) // wob.BIN_SIZE) - 1
    print(f"found elem x={x}, y={y} -> i={i}, j={j} = {i} + {j*16}")
    return i + 16*j

if __name__ == "__main__":
    env = wob.MiniWoBClickWrapper.create(
        'miniwob/click-dialog-v1', keep_obs=True,
        render_mode='human' if RENDER_ENV else None
    )
    print(env)
    print(env.action_space)
    print(env.observation_space)
    try:
        # Start a new episode.
        obs, info = env.reset()
        orig_obs = info.pop(wob.MiniWoBClickWrapper.FULL_OBS_KEY)
        print("Obs shape:", obs.shape)
        print("Info dict:", info)
        action = close_bins(orig_obs['dom_elements'])
        print("action", action)

        # switch between detected close action and brute force mode
        if False:
            obs, reward, is_done, is_trunc, info = env.step(action)
            info.pop(wob.MiniWoBClickWrapper.FULL_OBS_KEY)
            print(reward, is_done, info)
        else:
            is_done = False
            for action in range(env.action_space.n):
                time.sleep(0.01)
                obs, reward, is_done, is_trunc, info = env.step(action)
                info.pop(wob.MiniWoBClickWrapper.FULL_OBS_KEY)
                print(action, "=>", reward, is_done, info)
                if is_done:
                    print("Episode done:", action)
                    break
    finally:
        env.close()
