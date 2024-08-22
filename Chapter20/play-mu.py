#!/usr/bin/env python3
import sys
import time
import argparse

from lib import game
from lib import muzero as mu

import torch


MCTS_SEARCHES = 10
MCTS_BATCH_SIZE = 8


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs='+', help="The list of models (at least 2) to play against each other")
    parser.add_argument("-r", "--rounds", type=int, default=2, help="Count of rounds to perform for every pair, default=2")
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    params = mu.MuZeroParams(dev=device)

    nets = []
    for fname in args.models:
        net = mu.MuZeroModels(mu.OBS_SHAPE, game.GAME_COLS)
        d = torch.load(fname, map_location=lambda storage, loc: storage)
        net.set_state_dict(d)
        net.to(device)
        nets.append((fname, net))

    total_agent = {}
    total_pairs = {}

    for idx1, n1 in enumerate(nets):
        for idx2, n2 in enumerate(nets):
            if idx1 == idx2:
                continue
            wins, losses, draws = 0, 0, 0
            ts = time.time()
            for _ in range(args.rounds):
                r, _ = mu.play_game(n1[1], n2[1], params, temperature=0)
                if r > 0.5:
                    wins += 1
                elif r < -0.5:
                    losses += 1
                else:
                    draws += 1
            speed_games = args.rounds / (time.time() - ts)
            name_1, name_2 = n1[0], n2[0]
            print("%s vs %s -> w=%d, l=%d, d=%d" % (name_1, name_2, wins, losses, draws))
            sys.stderr.write("Speed %.2f games/s\n" % speed_games)
            sys.stdout.flush()
            game.update_counts(total_agent, name_1, (wins, losses, draws))
            game.update_counts(total_agent, name_2, (losses, wins, draws))
            game.update_counts(total_pairs, (name_1, name_2), (wins, losses, draws))

    # leaderboard by total wins
    total_leaders = list(total_agent.items())
    total_leaders.sort(reverse=True, key=lambda p: p[1][0])

    print("Leaderboard:")
    for name, (wins, losses, draws) in total_leaders:
        print("%s: \t w=%d, l=%d, d=%d" % (name, wins, losses, draws))
