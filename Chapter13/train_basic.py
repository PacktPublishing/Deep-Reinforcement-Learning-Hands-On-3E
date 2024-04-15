#!/usr/bin/env python3
from textworld import gym
from textworld.gym import register_games
import ptan
import pathlib
import argparse
import itertools
import numpy as np

from textworld import EnvInfos

from lib import preproc, model, common

import torch
import torch.optim as optim
from ignite.engine import Engine


EXTRA_GAME_INFO = {
    "inventory": True,
    "description": True,
    "intermediate_reward": True,
    "admissible_commands": True,
}


GAMMA = 0.9
LEARNING_RATE = 5e-5
BATCH_SIZE = 64


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--game", default="simple",
                        help="Game prefix to be used during training, default=simple")
    parser.add_argument("--params", choices=list(common.PARAMS.keys()), default='small',
                        help="Training params, could be one of %s" % (list(common.PARAMS.keys())))
    parser.add_argument("-s", "--suffices", type=int, default=1,
                        help="Count of game indices to use during training, default=1")
    parser.add_argument("-v", "--validation", default='-val',
                        help="Suffix for game used for validation, default=-val")
    parser.add_argument("--dev", default="cpu",
                        help="Device to use, default=cpu")
    parser.add_argument("-r", "--run", required=True, help="Run name")
    args = parser.parse_args()
    device = torch.device(args.dev)
    params = common.PARAMS[args.params]

    game_files = [
        "games/%s%s.ulx" % (args.game, s)
        for s in range(1, args.suffices+1)
    ]
    val_game_file = "games/%s%s.ulx" % (args.game, args.validation)
    if not all(map(lambda p: pathlib.Path(p).exists(), game_files)):
        raise RuntimeError(f"Some game files from {game_files} "
                           f"not found! Please run make_games.sh")
    vocab, action_space, observation_space = \
        common.get_games_spaces(game_files + [val_game_file])
    vocab_rev = common.build_rev_vocab(vocab)
    env_id = register_games(
        gamefiles=game_files,
        request_infos=EnvInfos(**EXTRA_GAME_INFO),
        name=args.game
    )
    print(f"Registered env {env_id} for game files {game_files}")
    val_env_id = register_games(
        gamefiles=[val_game_file],
        request_infos=EnvInfos(**EXTRA_GAME_INFO),
        name=args.game
    )
    print(f"Game {val_env_id}, with file {val_game_file} "
          f"will be used for validation")

    env = gym.make(env_id)
    env = preproc.TextWorldPreproc(env, vocab_rev)
    v = env.reset()

    val_env = gym.make(val_env_id)
    val_env = preproc.TextWorldPreproc(val_env, vocab_rev)

    prep = preproc.Preprocessor(
        dict_size=len(vocab),
        emb_size=params.embeddings, num_sequences=env.num_fields,
        enc_output_size=params.encoder_size).to(device)
    tgt_prep = ptan.agent.TargetNet(prep)

    net = model.DQNModel(obs_size=prep.obs_enc_size,
                         cmd_size=prep.cmd_enc_size)
    net = net.to(device)
    tgt_net = ptan.agent.TargetNet(net)

    agent = model.DQNAgent(net, prep, epsilon=1, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, params.replay_size)

    optimizer = optim.RMSprop(itertools.chain(net.parameters(), prep.parameters()),
                              lr=LEARNING_RATE, eps=1e-5)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_t = model.calc_loss_dqn(batch, prep, tgt_prep.target_model,
                                     net, tgt_net.target_model, GAMMA, device=device)
        loss_t.backward()
        optimizer.step()
        eps = 1 - engine.state.iteration / params.epsilon_steps
        agent.epsilon = max(params.epsilon_final, eps)
        if engine.state.iteration % params.sync_nets == 0:
            tgt_net.sync()
            tgt_prep.sync()
        return {
            "loss": loss_t.item(),
            "epsilon": agent.epsilon,
        }

    engine = Engine(process_batch)
    run_name = f"basic-{args.params}_{args.run}"
    save_path = pathlib.Path("saves") / run_name
    save_path.mkdir(parents=True, exist_ok=True)

    common.setup_ignite(engine, exp_source, run_name,
                        extra_metrics=('val_reward', 'val_steps'))

    @engine.on(ptan.ignite.PeriodEvents.ITERS_100_COMPLETED)
    def validate(engine):
        reward = 0.0
        steps = 0

        obs, extra = val_env.reset()

        while True:
            obs_t = prep.encode_observations([obs]).to(device)
            cmd_t = prep.encode_commands(obs['admissible_commands']).to(device)
            q_vals = net.q_values(obs_t, cmd_t)
            act = np.argmax(q_vals)

            obs, r, is_done, _, _ = val_env.step(act)
            steps += 1
            reward += r
            if is_done:
                break
        engine.state.metrics['val_reward'] = reward
        engine.state.metrics['val_steps'] = steps
        print("Validation got %.3f reward in %d steps" % (reward, steps))
        best_val_reward = getattr(engine.state, "best_val_reward", None)
        if best_val_reward is None:
            engine.state.best_val_reward = reward
        elif best_val_reward < reward:
            print("Best validation reward updated: %s -> %s" % (best_val_reward, reward))
            save_prep_name = save_path / ("best_val_%.3f_p.dat" % reward)
            save_net_name = save_path / ("best_val_%.3f_n.dat" % reward)
            torch.save(prep.state_dict(), save_prep_name)
            torch.save(net.state_dict(), save_net_name)
            engine.state.best_val_reward = reward

    @engine.on(ptan.ignite.EpisodeEvents.BEST_REWARD_REACHED)
    def best_reward_updated(trainer: Engine):
        reward = trainer.state.metrics['avg_reward']
        if reward > 0:
            save_prep_name = save_path / ("best_train_%.3f_p.dat" % reward)
            save_net_name = save_path / ("best_train_%.3f_n.dat" % reward)
            torch.save(prep.state_dict(), save_prep_name)
            torch.save(net.state_dict(), save_net_name)
            print("%d: best avg training reward: %.3f, saved" % (
                trainer.state.iteration, reward))

    engine.run(common.batch_generator(buffer, params.replay_initial, BATCH_SIZE))
