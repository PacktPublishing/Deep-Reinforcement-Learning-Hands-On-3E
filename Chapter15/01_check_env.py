#!/usr/bin/env python3
import gymnasium as gym

ENV_ID = "MinitaurBulletEnv-v0"
ENTRY = "pybullet_envs.bullet.minitaur_gym_env:MinitaurBulletEnv"
RENDER = True


if __name__ == "__main__":
    gym.register(
        ENV_ID, entry_point=ENTRY,
        max_episode_steps=1000, reward_threshold=15.0,
        disable_env_checker=True,
    )
    env = gym.make(ENV_ID, render=RENDER)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print(env)
    print(env.reset())
    input("Press any key to exit\n")
    env.close()
