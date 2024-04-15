#!/usr/bin/env python3
import argparse
from textworld import gym, EnvInfos
from textworld.gym import register_game
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, \
    MessagesPlaceholder


def play_game(env, max_steps: int = 20) -> bool:
    prompt_init = ChatPromptTemplate.from_messages([
        ("system", "You're playing the interactive fiction game. "
                   "Reply with just a command in lowercase and "
                   "nothing else"),
        ("system", "Game objective: {objective}"),
        ("user", "Room description: {description}"),
        ("user", "What command you want to execute next?"),
    ])
    llm = ChatOpenAI()
    output_parser = StrOutputParser()

    commands = []

    obs, info = env.reset()
    init_msg = prompt_init.invoke({
        "objective": info['objective'],
        "description": info['description'],
    })

    context = init_msg.to_messages()
    ai_msg = llm.invoke(init_msg)
    context.append(ai_msg)
    cmd = output_parser.invoke(ai_msg)

    prompt_next = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Last command result: {result}"),
        ("user", "Room description: {description}"),
        ("user", "What command you want to execute next?"),
    ])

    for _ in range(max_steps):
        commands.append(cmd)
        print(">>>", cmd)
        obs, r, is_done, info = env.step(cmd)
        if is_done:
            print(f"I won in {len(commands)} steps!")
            return True

        user_msgs = prompt_next.invoke({
            "chat_history": context,
            "result": obs.strip(),
            "description": info['description'],
        })
        context = user_msgs.to_messages()
        ai_msg = llm.invoke(user_msgs)
        context.append(ai_msg)
        cmd = output_parser.invoke(ai_msg)

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
        print(f"Starting game {index}\n")
        env_id = register_game(
            gamefile=f"games/{args.game}{index}.ulx",
            request_infos=EnvInfos(
                description=True,
                objective=True,
            ),
        )
        env = gym.make(env_id)
        count_games += 1
        if play_game(env):
            count_won += 1
    print(f"Played {count_games}, won {count_won}")
