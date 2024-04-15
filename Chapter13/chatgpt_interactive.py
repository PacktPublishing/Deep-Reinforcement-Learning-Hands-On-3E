#!/usr/bin/env python3
import argparse
import textwrap
from textworld import gym, EnvInfos
from textworld.gym import register_game


def play_game(env, max_steps: int = 20) -> bool:
    commands = []

    obs, info = env.reset()

    print(textwrap.dedent("""\
    You're playing the interactive fiction game.
    Here is the game objective: %s
    
    Here is the room description: %s
    
    What command do you want to execute next? Reply with 
    just a command in lowercase and nothing else. 
    """)  % (info['objective'], info['description']))

    print("=== Send this to chat.openai.com and type the reply...")

    while len(commands) < max_steps:
        cmd = input(">>> ")
        commands.append(cmd)
        obs, r, is_done, info = env.step(cmd)
        if is_done:
            print(f"You won in {len(commands)} steps! "
                  f"Don't forget to congratulate ChatGPT!")
            return True

        print(textwrap.dedent("""\
        Last command result: %s
        Room description: %s
        
        What's the next command?
        """) % (obs, info['description']))

        print("=== Send this to chat.openai.com and "
              "type the reply...")

    print(f"Wasn't able to solve after {max_steps} steps, "
          f"commands: {commands}")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--game", default="simple",
                        help="Game prefix to be used during training, default=simple")
    parser.add_argument("indices", nargs='+', type=int, default=[1], help="Game indices to test on, default=1")
    args = parser.parse_args()

    count_games, count_won = 0, 0
    for index in args.indices:
        env_id = register_game(
            gamefile=f"games/{args.game}{index}.ulx",
            request_infos=EnvInfos(
                description=True,
                objective=True,
            ),
        )
        env = gym.make(env_id)
        count_games += 1
        print(f"Starting game {index}\n")
        if play_game(env):
            count_won += 1
    print(f"Played {count_games}, won {count_won}")