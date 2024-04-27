#!/usr/bin/env python3
import os
import pathlib
import random
import argparse
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

from lib import wob, model, common, demos

import ptan

import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim


ENVS_COUNT = 8
ENV_NAME = 'miniwob/click-dialog-v1'

GAMMA = 0.99
REWARD_STEPS = 2
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
ENTROPY_BETA = 0.001
CLIP_GRAD = 0.05

SAVES_DIR = "saves"

# probability to add demo samples to the batch
DEMO_PROB = 0.5
# For how many initial frames we train on demo batch
DEMO_FRAMES = 25000

DUMP_INTERVAL = 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True,
                        help="Name of the run")
    parser.add_argument("--dev", default="cpu",
                        help="Device to use, default=cpu")
    parser.add_argument("--env", default=ENV_NAME,
                        help="Environment name to solve, "
                             "default=" + ENV_NAME)
    parser.add_argument("--demo", help="Demo dir to load, "
                                       "default=no demo")
    parser.add_argument("--dump",
                        help="If given save models and screenshots "
                             "every 1k training steps using this dir")
    args = parser.parse_args()
    device = torch.device(args.dev)

    env_name = args.env
    if not env_name.startswith('miniwob/'):
        env_name = "miniwob/" + env_name

    demo_samples = None
    if args.demo:
        demo_samples = demos.load_demo_dir(
            args.demo, gamma=GAMMA, steps=REWARD_STEPS)
        print(f"Loaded {len(demo_samples)} demo samples")

    dump_dir = None
    if args.dump is not None:
        dump_dir = pathlib.Path(args.dump)
        dump_dir.mkdir(parents=True, exist_ok=True)

    name = env_name.split('/')[-1] + "_" + args.name
    writer = SummaryWriter(comment="-wob_click_" + name)
    saves_path = os.path.join(SAVES_DIR, name)
    os.makedirs(saves_path, exist_ok=True)

    envs = [
        lambda: wob.MiniWoBClickWrapper.create(env_name)
        for _ in range(ENVS_COUNT)
    ]
    env = gym.vector.AsyncVectorEnv(envs)

    net = model.Model(input_shape=wob.WOB_SHAPE,
                      n_actions=env.single_action_space.n).to(device)
    print(net)
    optimizer = optim.Adam(net.parameters(),
                           lr=LEARNING_RATE, eps=1e-3)

    agent = ptan.agent.PolicyAgent(
        lambda x: net(x)[0], device=device, apply_softmax=True)
    exp_source = ptan.experience.VectorExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    best_reward = None
    with common.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(
                writer, batch_size=10) as tb_tracker:
            batch = []
            for step_idx, exp in enumerate(exp_source):
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps",
                                     np.mean(steps), step_idx)

                    mean_reward = tracker.reward(
                        float(np.mean(rewards)), step_idx)
                    if mean_reward is not None:
                        if best_reward is None or \
                                mean_reward > best_reward:
                            if best_reward is not None:
                                name = "best_%.3f_%d.dat" % (
                                    mean_reward, step_idx)
                                fname = os.path.join(
                                    saves_path, name)
                                torch.save(net.state_dict(), fname)
                                print("Best reward updated: %.3f "
                                      "-> %.3f" % (
                                    best_reward, mean_reward))
                            best_reward = mean_reward
                batch.append(exp)
                if dump_dir is not None and step_idx % DUMP_INTERVAL == 0:
                    print("Dumping model and screenshots from the batch")
                    p = dump_dir / f"model_{step_idx:06d}.dat"
                    torch.save(net.state_dict(), str(p))
                    p = dump_dir / f"scr_{step_idx:06d}_act={exp.action}_r={exp.reward:.2f}.png"
                    demos.save_obs_image(exp.state, exp.action, str(p))
                if len(batch) < BATCH_SIZE:
                    continue

                if demo_samples and step_idx < DEMO_FRAMES:
                    if random.random() < DEMO_PROB:
                        random.shuffle(demo_samples)
                        demo_batch = demo_samples[:BATCH_SIZE]
                        model.train_demo(
                            net, optimizer, demo_batch, writer,
                            step_idx, device=device
                        )

                states_v, actions_t, vals_ref_v = \
                    common.unpack_batch(
                        batch, net, device=device,
                        last_val_gamma=GAMMA ** REWARD_STEPS)

                optimizer.zero_grad()
                logits_v, value_v = net(states_v)

                loss_value_v = F.mse_loss(
                    value_v.squeeze(-1), vals_ref_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.detach()
                lpa = log_prob_v[range(len(batch)), actions_t]
                log_prob_actions_v = adv_v * lpa
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                ent_v = prob_v * log_prob_v
                entropy_loss_v = ENTROPY_BETA * ent_v
                entropy_loss_v = entropy_loss_v.sum(dim=1).mean()

                loss_v = loss_policy_v + entropy_loss_v + \
                         loss_value_v
                loss_v.backward()
                nn_utils.clip_grad_norm_(
                    net.parameters(), CLIP_GRAD)
                optimizer.step()
                batch.clear()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v,
                                 step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v,
                                 step_idx)
                tb_tracker.track("loss_policy", loss_policy_v,
                                 step_idx)
                tb_tracker.track("loss_value", loss_value_v,
                                 step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
