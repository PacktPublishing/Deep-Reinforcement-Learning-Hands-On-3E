#!/usr/bin/env python3
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from lib import data
from PIL import Image
import pathlib
import numpy as np

RENDER_DIR = "render"


def save_render(render: np.ndarray, path: pathlib.Path, step: int):
    img = Image.fromarray(render)
    p = path / f"render_{step:04d}.png"
    img.save(str(p))


if __name__ == "__main__":
    env = data.ForestEnv(render_mode="rgb_array")
    recorder = VideoRecorder(env, RENDER_DIR + "/forest-random.mp4")
    sum_rewards = {agent_id: 0.0 for agent_id in env.agents}
    sum_steps = {agent_id: 0 for agent_id in env.agents}
    obs = env.reset()
    recorder.capture_frame()
    assert isinstance(obs, dict)
    print(f"tiger_0: obs {obs['tiger_0'].shape}, "
          f"act: {env.action_space('tiger_0')}")
    print(f"deer_0: obs {obs['deer_0'].shape}, "
          f"act: {env.action_space('deer_0')}\n")
    step = 0
    save_render(env.render(), pathlib.Path(RENDER_DIR), step)

    while env.agents:
        actions = {
            agent_id: env.action_space(agent_id).sample()
            for agent_id in env.agents
        }
        all_obs, all_rewards, all_dones, all_trunc, all_info = \
            env.step(actions)
        recorder.capture_frame()
        for agent_id, r in all_rewards.items():
            sum_rewards[agent_id] += r
            sum_steps[agent_id] += 1
        step += 1
        save_render(env.render(), pathlib.Path(RENDER_DIR), step)

    final_rewards = list(sum_rewards.items())
    final_rewards.sort(key=lambda p: p[1], reverse=True)
    for agent_id, r in final_rewards[:20]:
        print(f"{agent_id}: got {r:.2f} in {sum_steps[agent_id]} steps")
    recorder.close()