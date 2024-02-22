from lib import *


if __name__ == "__main__":
    env = ToyEnv()
    s, _ = env.reset()
    print(f"env.reset() -> {s}")
    s = env.step(1)
    print(f"env.step(1) -> {s}")
    s = env.step(2)
    print(f"env.step(2) -> {s}")

    for _ in range(10):
        r = env.step(0)
        print(r)

    agent = DullAgent(action=1)
    print("agent:", agent([1, 2])[0])

    env = ToyEnv()
    agent = DullAgent(action=1)
    exp_source = ptan.experience.ExperienceSource(
        env=env, agent=agent, steps_count=2)
    for idx, exp in enumerate(exp_source):
        if idx > 15:
            break
        print(exp)

    exp_source = ptan.experience.ExperienceSource(
        env=env, agent=agent, steps_count=4)
    print(next(iter(exp_source)))

    exp_source = ptan.experience.ExperienceSource(
        env=[ToyEnv(), ToyEnv()], agent=agent, steps_count=2)
    for idx, exp in enumerate(exp_source):
        if idx > 4:
            break
        print(exp)

    print("ExperienceSourceFirstLast")
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=1.0, steps_count=1)
    for idx, exp in enumerate(exp_source):
        print(exp)
        if idx > 10:
            break
