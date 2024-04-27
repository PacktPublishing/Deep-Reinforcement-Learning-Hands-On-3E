#!/usr/bin/env python3
import numpy as np
import gymnasium
import miniwob
import typing as tt
from miniwob.action import ActionTypes, ActionSpaceConfig

RENDER_ENV = True

BIN_DX = 10
BIN_DY = 10
SIZE_Y = 210
SIZE_X = 160
BINS_X = SIZE_X // BIN_DX
BINS_Y = SIZE_Y // BIN_DY



def close_bins(elems: tt.Tuple[dict, ...]) -> tt.Tuple[int, int]:
    elem_ids = {e['ref']: e for e in elems}
    close_elem = None
    for e in elems:
        if e['text'] == 'Close':
            close_elem = e
            break
    # need to roll back while ref is negative
    while close_elem['ref'] < 0:
        close_elem = elem_ids[close_elem['parent']]
    x = close_elem['left'][0] + close_elem['width'][0] / 2.0
    y = close_elem['top'][0] + close_elem['height'][0] / 2.0
    return x // BIN_DX, y // BIN_DY



if __name__ == "__main__":
    gymnasium.register_envs(miniwob)

    act_cfg = ActionSpaceConfig(
        action_types=(ActionTypes.CLICK_COORDS, ),
        coord_bins=(BINS_X, BINS_Y),
    )
    env = gymnasium.make(
        'miniwob/click-dialog-v1',
        render_mode='human' if RENDER_ENV else None,
        action_space_config=act_cfg,
    )
    print(env)
    print(env.action_space)
    try:
        # Start a new episode.
        obs, info = env.reset()
        print("Obs keys:", list(obs.keys()))
        print("Info dict:", info)
        print("Screenshot shape:", obs['screenshot'].shape)
        coords = close_bins(obs['dom_elements'])

        action = {
            "action_type": 0,
            "coords": np.array(coords, dtype=np.int8)
        }
        print("action", action)
        obs, reward, is_done, is_trunc, info = env.step(action)
        print(reward, is_done, info)

        # Brute force to check that our action is correct (comment step() call above)
        if False:
            is_done = False
            for y in range(BINS_Y):
                for x in range(BINS_X):
                    action = {
                        "action_type": 0,
                        "coords": np.array((x, y), dtype=np.int8)
                    }
                    obs, reward, is_done, is_trunc, info = env.step(action)
                    if is_done:
                        print("Episode done:", action)
                        print(reward, is_done, info)
                        break
                if is_done:
                    break
    finally:
        env.close()
