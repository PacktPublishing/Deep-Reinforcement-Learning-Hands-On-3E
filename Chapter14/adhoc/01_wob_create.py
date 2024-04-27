#!/usr/bin/env python3
import time
import gymnasium as gym
import miniwob
from miniwob.action import ActionTypes

RENDER_ENV = True


if __name__ == "__main__":
    gym.register_envs(miniwob)

    env = gym.make('miniwob/click-test-2-v1',
                   render_mode='human' if RENDER_ENV else None)
    print(env)
    try:
        # Start a new episode.
        obs, info = env.reset()
        print("Obs keys:", list(obs.keys()))
        print("Info dict:", info)
        assert obs["utterance"] == "Click button ONE."
        assert obs["fields"] == (("target", "ONE"),)
        print("Screenshot shape:", obs['screenshot'].shape)
        if RENDER_ENV:
            # to let you look at the environment.
            time.sleep(2)

        # Find the HTML element with text "ONE".
        target_elems = [
            e for e in obs['dom_elements']
            if e['text'] == "ONE"
        ]
        assert target_elems
        print("Target elem:", target_elems[0])

        # Click on the element.
        action = env.unwrapped.create_action(
            ActionTypes.CLICK_ELEMENT, ref=target_elems[0]["ref"]
        )
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward, terminated, info)
    finally:
        env.close()
