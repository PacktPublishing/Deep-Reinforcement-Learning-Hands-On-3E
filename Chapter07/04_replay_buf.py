from lib import *


if __name__ == "__main__":
    env = ToyEnv()
    agent = DullAgent(action=1)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=1.0, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=100)

    for step in range(6):
        buffer.populate(1)
        # if buffer is small enough, do nothing
        if len(buffer) < 5:
            continue
        batch = buffer.sample(4)
        print("Train time, %d batch samples:" % len(batch))
        for s in batch:
            print(s)
