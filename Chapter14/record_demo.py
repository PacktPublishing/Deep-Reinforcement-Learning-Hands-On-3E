#!/usr/bin/env python3
import time
import json
import pathlib
import pickle
import argparse
import gymnasium as gym
import miniwob
import bottle
from miniwob.action import ActionTypes, ActionSpaceConfig
import multiprocessing as mp

from lib import demos

DEFAULT_GAME = 'click-tab-v1'

OBS_DELAY = 0.01


def server_proc(queue: mp.Queue):
    app = bottle.Bottle()

    @app.hook("after_request")
    def enable_cors():
        """Enable the browser to request code from any origin."""
        # This is dangerous but whatever:
        bottle.response.headers["Access-Control-Allow-Origin"] = "*"

    @app.post("/record")
    def record():
        data = bottle.request.body.read()
        queue.put(data)
        return "send to the queue"

    app.run(host="localhost", port=8032)
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--game", default=DEFAULT_GAME,
                        help="Game to be solved, default=" + DEFAULT_GAME)
    parser.add_argument("-o", "--out", required=True,
                        help="Output directory for the recordings")
    parser.add_argument("-d", "--delay", type=int,
                        help="If given, wait for this amount of seconds "
                             "between episodes, default=enter in console")
    parser.add_argument("-s", "--save", default=False, action="store_true",
                        help="If given, intermediate raw data will be stored in files")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    assert out_dir.exists() and out_dir.is_dir()

    queue = mp.Queue()
    proc = mp.Process(target=server_proc, kwargs={"queue": queue})
    proc.start()

    try:
        gym.register_envs(miniwob)
        game_name = args.game
        if not game_name.startswith("miniwob/"):
            game_name = "miniwob/" + game_name

        env = gym.make(
            game_name,
            action_space_config=ActionSpaceConfig(
                action_types=(ActionTypes.NONE, )
            ),
            render_mode='human'
        )
        # hack to enable recording of the game
        env.unwrapped.instance.url += "?record=true"
        env.unwrapped.instance.driver.get(env.unwrapped.instance.url)
        try:
            idx = 1
            while True:
                observations = []
                obs, info = env.reset()
                observations.append((obs, time.time_ns()))

                while True:
                    time.sleep(OBS_DELAY)
                    action = {"action_type": 0}
                    obs, reward, is_done, is_tr, info = env.step(action)
                    # do not record fully black observations from the end
                    if obs['screenshot'].max() > 0:
                        observations.append((obs, time.time_ns()))
                    if is_done:
                        break
                data_bytes = queue.get()
                data = json.loads(data_bytes)
                rel_obs = demos.observations_to_delta(observations)
                if args.save:
                    out_obs = pathlib.Path(f"{idx}-obs.dat")
                    out_obs.write_bytes(pickle.dumps(rel_obs))
                    out_data = pathlib.Path(f"{idx}-dat.json")
                    out_data.write_bytes(data_bytes)

                new_data = demos.join_obs(data, rel_obs)
                ts = time.strftime('%m%d%H%M%S', time.gmtime())
                out_path = out_dir / f"{new_data['taskName']}_{ts}.json"
                out_path.write_text(json.dumps(new_data))
                print("Saved in ", out_path)
                if args.delay is None:
                    input("Press enter to start new round...")
                else:
                    print(f"New episode starts in {args.delay} seconds...")
                    time.sleep(args.delay)
                idx += 1
        finally:
            env.close()
    finally:
        print("Stopping server...")
        proc.kill()
        proc.join(1)
        print("Done")