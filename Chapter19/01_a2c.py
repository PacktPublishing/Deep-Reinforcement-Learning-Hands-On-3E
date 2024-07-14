#!/usr/bin/env python3
import gymnasium as gym
import ptan
from ptan.experience import VectorExperienceSourceFirstLast
from ptan.common.utils import TBMeanTracker
import numpy as np
import argparse
import pathlib
import queue
import typing as tt
from torch.utils.tensorboard.writer import SummaryWriter

import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

from lib import common, rlhf

# Limit of steps in episodes is needed, as sometimes agent leans
# just to refill the oxigen and do nothing :)
# This limit is applied after all the wrappers (so real game
# will be x4 due to frame skip)
TIME_LIMIT = 5000

GAMMA = 0.99
LEARNING_RATE          = 0.0007
LEARNING_RATE_FINETUNE = 0.00007
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 16

REWARD_STEPS = 5
CLIP_GRAD = 0.1

TEST_EPISODES = 10
TEST_EVERY_STEP = 100*BATCH_SIZE


def make_env_func(env_idx: int, db_path: tt.Optional[str],
                  reward_path: tt.Optional[str],
                  dev: torch.device,
                  metrics_queue: tt.Optional[queue.Queue]) -> \
        tt.Callable[[], gym.Env]:
    def make_env() -> gym.Env:
        e = gym.make("SeaquestNoFrameskip-v4")
        if reward_path is not None:
            p = pathlib.Path(reward_path)
            e = rlhf.RewardModelWrapper(
                e, p, dev=dev,
                metrics_queue=metrics_queue)
        if db_path is not None:
            p = pathlib.Path(db_path)
            p.mkdir(parents=True, exist_ok=True)
            e = rlhf.EpisodeRecorderWrapper(e, p, env_idx=env_idx)
        e = ptan.common.wrappers.wrap_dqn(e)
        # add time limit after all wrappers
        e = gym.wrappers.TimeLimit(e, TIME_LIMIT)
        return e
    return make_env


def make_test_env() -> gym.Env:
    e = gym.make("SeaquestNoFrameskip-v4")
    e = ptan.common.wrappers.wrap_dqn(e, clip_reward=False)
    # add time limit after all wrappers
    e = gym.wrappers.TimeLimit(e, TIME_LIMIT)
    return e


def process_metrics(step_idx: int, metrics_queue: queue.Queue,
                    writer: SummaryWriter):
    try:
        while True:
            key, val = metrics_queue.get(block=False)
            writer.add_scalar(key, val, step_idx)
    except queue.Empty:
        pass


def test_model(
        env: gym.Env,
        dev: torch.device,
        net: common.AtariA2C,
        episodes: int = TEST_EPISODES
) -> tt.Tuple[float, int]:
    """
    Test model for given amount of episodes
    :param env: test environment
    :param dev: device to use
    :param net: model to test
    :param episodes: count of episodes
    :return: best reward and count of steps
    """
    best_reward = 0.0
    best_steps = 0
    for _ in range(episodes):
        cur_reward, cur_steps = 0, 0.0
        obs, _ = env.reset()
        while True:
            obs_v = torch.FloatTensor(obs).unsqueeze(0).to(dev)
            policy_v = net(obs_v)[0]
            policy_v = F.softmax(policy_v, dim=1)
            probs = policy_v[0].detach().cpu().numpy()
            action = np.random.choice(len(probs), p=probs)
            obs, reward, done, is_tr, _ = env.step(action)
            cur_reward += reward
            cur_steps += 1
            if done or is_tr:
                break
        if best_reward < cur_reward:
            best_reward = cur_reward
        if best_steps < cur_steps:
            best_steps = cur_steps
    return best_reward, best_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu",
                        help="Device to use, default=cpu")
    parser.add_argument("--use-async", default=False,
                        action='store_true',
                        help="Use async vector env (A3C mode)")
    parser.add_argument("-n", "--name", required=True,
                        help="Name of the run")
    parser.add_argument("-r", "--reward",
                        help="Reward model to load")
    parser.add_argument("-m", "--model",
                        help="Policy model to load")
    parser.add_argument("--finetune", default=False,
                        action="store_true", help="If given, enable finetune mode")
    parser.add_argument("--save", help="If given, dir to save models")
    parser.add_argument(
        "--db-path", help="If given, short episodes will be "
                          "stored in this path")
    args = parser.parse_args()
    device = torch.device(args.dev)
    metrics_queue = queue.Queue(maxsize=0)

    env_factories = [
        make_env_func(env_idx, args.db_path, args.reward,
                      device, metrics_queue)
        for env_idx in range(NUM_ENVS)
    ]
    if args.use_async:
        env = gym.vector.AsyncVectorEnv(env_factories)
    else:
        env = gym.vector.SyncVectorEnv(env_factories)
    test_env = make_test_env()
    writer = SummaryWriter(comment="-a2c_" + args.name)

    net = common.AtariA2C(env.single_observation_space.shape,
                          env.single_action_space.n).to(device)
    print(net)
    if args.model is not None:
        net.load_state_dict(torch.load(args.model))
        print("Loaded model " + args.model)
    agent = ptan.agent.PolicyAgent(
        lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_source = VectorExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    lr = LEARNING_RATE
    if args.finetune:
        lr = LEARNING_RATE_FINETUNE
        # Freeze convolution layers
        net.conv.requires_grad_(False)

    save_path = None
    if args.save:
        save_path = pathlib.Path(args.save)
        if not save_path.exists():
            save_path.mkdir(parents=True)

    optimizer = optim.Adam(
        net.parameters(), lr=lr, eps=1e-5)
    scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=8*(10**7) / BATCH_SIZE)

    batch = []
    best_test_reward = 0.0
    best_test_steps = 0.0

    with common.RewardTracker(writer, stop_reward=None) as tracker:
        with TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)

                # handle new rewards
                solved = False
                for new_reward, new_step in exp_source.pop_rewards_steps():
                    if tracker.reward(new_reward, new_step, step_idx):
                        solved = True
                if solved:
                    break

                process_metrics(step_idx, metrics_queue, writer)
                if step_idx % TEST_EVERY_STEP == 0:
                    print(f"{step_idx}: Testing model...")
                    test_rw, test_steps = test_model(test_env, device, net)
                    writer.add_scalar("test_reward", test_rw, step_idx)
                    writer.add_scalar("test_steps", test_steps, step_idx)
                    print(f"Got best reward {test_rw:.2f} and steps {test_steps} "
                          f"in {TEST_EPISODES} episodes")
                    if save_path is not None:
                        if test_rw > best_test_reward or test_steps > best_test_steps:
                            name = "model_%06d-rw=%.0f-steps=%d.dat" % (step_idx, test_rw, test_steps)
                            torch.save(net.state_dict(), str(save_path / name))
                            best_test_reward = test_rw
                            best_test_steps = test_steps

                if len(batch) < BATCH_SIZE:
                    continue

                states_t, actions_t, vals_ref_t = \
                    common.unpack_batch(
                        batch, net, device=device,
                        gamma=GAMMA, reward_steps=REWARD_STEPS)
                batch.clear()

                optimizer.zero_grad()
                logits_t, value_t = net(states_t)
                loss_value_t = F.mse_loss(
                    value_t.squeeze(-1), vals_ref_t)

                log_prob_t = F.log_softmax(logits_t, dim=1)
                adv_t = vals_ref_t - value_t.detach()
                log_act_t = log_prob_t[range(BATCH_SIZE), actions_t]
                log_prob_actions_t = adv_t * log_act_t
                loss_policy_t = -log_prob_actions_t.mean()

                prob_t = F.softmax(logits_t, dim=1)
                entropy_loss_t = ENTROPY_BETA * (
                        prob_t * log_prob_t).sum(dim=1).mean()

                # calculate policy gradients only
                loss_policy_t.backward(retain_graph=True)
                grads = np.concatenate([
                    p.grad.data.cpu().numpy().flatten()
                    for p in net.parameters()
                    if p.grad is not None
                ])

                # apply entropy and value gradients
                loss_v = entropy_loss_t + loss_value_t
                loss_v.backward()
                nn_utils.clip_grad_norm_(
                    net.parameters(), CLIP_GRAD)
                optimizer.step()
                scheduler.step()

                # get full loss
                loss_v += loss_policy_t

                tb_tracker.track(
                    "lr", scheduler.get_last_lr()[0], step_idx)
                tb_tracker.track(
                    "advantage", adv_t, step_idx)
                tb_tracker.track(
                    "values", value_t, step_idx)
                tb_tracker.track(
                    "batch_rewards", vals_ref_t, step_idx)
                tb_tracker.track(
                    "loss_entropy", entropy_loss_t, step_idx)
                tb_tracker.track(
                    "loss_policy", loss_policy_t, step_idx)
                tb_tracker.track(
                    "loss_value", loss_value_t, step_idx)
                tb_tracker.track(
                    "loss_total", loss_v, step_idx)
                tb_tracker.track(
                    "grad_l2", np.sqrt(np.mean(np.square(grads))),
                    step_idx)
                tb_tracker.track(
                    "grad_max", np.max(np.abs(grads)), step_idx)
                tb_tracker.track(
                    "grad_var", np.var(grads), step_idx)
