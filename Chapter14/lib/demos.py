import io
import copy
import json
import pathlib

import math
import numpy as np
import bisect
import base64
import typing as tt
from PIL import Image, ImageDraw

from ptan.experience import ExperienceFirstLast

from lib import wob


def encode_screenshot(data: np.ndarray) -> str:
    fd = io.BytesIO()
    np.savez_compressed(fd, data)
    return base64.encodebytes(fd.getvalue()).decode()


def decode_screenshot(s_data: str) -> np.ndarray:
    data = base64.decodebytes(s_data.encode())
    fd = io.BytesIO(data)
    return np.load(fd)['arr_0']


def join_obs(data: dict, delta_obs: tt.Dict[int, dict], ofs_ms: int = 100) -> dict:
    """
    Join events data and recorded observations (with screenshots)
    :param data: events obtained from the website
    :param delta_obs: observations in form delta_ms -> obs_dict
    :param ofs_ms: the gap to make before and after events to prevent weird observations
    :return: events data with screenshots joined
    """
    keys = list(sorted(delta_obs.keys()))
    new_data = copy.deepcopy(data)
    last_time: tt.Dict[tt.Tuple[str, int], int] = dict()
    for idx, state in enumerate(new_data['states']):
        if idx == 0:
            # initial state always copied from the first entry
            src_idx = 0
        else:
            evt_type = state['action']['type']
            evt_timing = state['action']['timing']
            cur_time = state['time']
            last_time[(evt_type, evt_timing)] = cur_time
            # for click event, we need to take screenshot before the mousedown event,
            # because event of click is timed when mouse was released. So, on long clicks
            # image is different
            if evt_type == 'click' and evt_timing == 1:
                cur_time = last_time.get(('mousedown', 1), cur_time)
            # search for index in observations. Events before the action got on the left, after on the right
            if evt_timing == 1:
                src_idx = bisect.bisect_left(keys, cur_time - ofs_ms) - 1
            else:
                src_idx = bisect.bisect_left(keys, cur_time + ofs_ms)
        if src_idx >= len(keys):
            src_idx = len(keys)-1
        src_key = keys[src_idx]
        scr_np = delta_obs[src_key]['screenshot']
        scr = encode_screenshot(scr_np)
        state['screenshot'] = scr
    return new_data


def observations_to_delta(observations: tt.List[tt.Tuple[dict, int]]) -> tt.Dict[int, dict]:
    """
    Convert pairs of observations with nanoseconds into relative miliseconds dict
    :param observations: list of tuples (observation, nanosecond timestamp)
    :return: dict with key of milisecond relative time and value of observation.
    """
    delta_obs = {}
    start_ts = None
    for obs, ts_ns in observations:
        if start_ts is None:
            start_ts = ts_ns
        delta_ms = math.trunc((ts_ns - start_ts) / 1_000_000)
        delta_obs[int(delta_ms)] = obs
    return delta_obs


def load_demo_file(
        file_path: pathlib.Path,
        gamma: float,
        steps: int,
        keep_text: bool = False,
) -> tt.List[ExperienceFirstLast]:
    """
    Load human demonstration from file and generate experience items from it.
    :param file_path: path of file to load
    :param gamma: gamma value to calculate discounted reward
    :param steps: count of steps to compute experience items
    :param keep_text: keep text in observation (besides screenshot)
    :return: list of experience items
    """
    data = json.loads(file_path.read_text())

    text = data['utterance']
    hist = []
    for state in data['states']:
        action = state['action']
        if action is None:
            continue
        # we need only states before the mouse click
        if not (action['type'] == 'click' and action['timing'] == 1):
            continue
        scr = decode_screenshot(state['screenshot'])
        scr = np.transpose(scr, (2, 0, 1))
        obs = (scr, text) if keep_text else scr
        act = wob.coord_to_action(action["x"], action["y"])
        hist.append((obs, act))
    result = []
    last_obs = None
    reward = data['reward']
    for obs, act in reversed(hist):
        result.append(ExperienceFirstLast(
            state=obs,
            action=act,
            reward=reward,
            last_state=last_obs
        ))
        reward *= gamma**steps
        last_obs = obs
    return result


def load_demo_dir(
        dir_name: str, gamma: float,
        steps: int, keep_text: bool = False
) -> tt.List[ExperienceFirstLast]:
    """
    Load all the demo from given directory. They have to belong to the single environment.
    :param dir_name: Directory to load
    :return: list of experience items loaded
    """
    res = []
    env_names = set()
    for file in pathlib.Path(dir_name).glob("*.json"):
        env_name = file.name.split("_", maxsplit=1)[0]
        env_names.add(env_name)
        res.extend(load_demo_file(
            file, gamma=gamma, steps=steps, keep_text=keep_text
        ))
    if len(env_names) > 1:
        raise RuntimeError(f"Directory {dir_name} contains more than one environment samples: {env_names}")
    return res


def save_obs_image(data: np.ndarray, action: tt.Optional[int], file_name: str, transpose: bool = True):
    if transpose:
        data = np.transpose(data, (1, 2, 0))
    img = Image.fromarray(data)
    if action is not None:
        draw = ImageDraw.Draw(img)
        x, y = wob.action_to_coord(action)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), (255, 0, 0, 128))
    img.save(file_name)