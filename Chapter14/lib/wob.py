import time

import gymnasium as gym
from gymnasium import spaces
import typing as tt
import numpy as np
import miniwob
from miniwob.action import ActionTypes, ActionSpaceConfig


# Constants for MiniWoB
WIDTH = 160
HEIGHT = 210
Y_OFS = 50

# default size of the clicking bin - square area we can click
BIN_SIZE = 10

WOB_SHAPE = (3, HEIGHT, WIDTH)


class MiniWoBClickWrapper(gym.ObservationWrapper):
    """
    Converts MiniWoB environment into simple bucketed click environment:
    * observations are stripped to image
    * actions are reduced to bucketed click items of given size,
    performing Y offset to get rid of instruction area
    """
    FULL_OBS_KEY = "full_obs"

    def __init__(self, env: gym.Env, keep_text: bool = False,
                 keep_obs: bool = False, bin_size: int = BIN_SIZE):
        super(MiniWoBClickWrapper, self).__init__(env)
        self.bin_size = bin_size
        self.keep_text = keep_text
        self.keep_obs = keep_obs
        img_space = spaces.Box(
            low=0, high=255, shape=WOB_SHAPE, dtype=np.uint8
        )
        if keep_text:
            self.observation_space = spaces.Tuple(
                (img_space, spaces.Text(max_length=1024))
            )
        else:
            self.observation_space = img_space
        self.x_bins = WIDTH // bin_size
        count = self.x_bins * ((HEIGHT - Y_OFS) // bin_size)
        self.action_space = spaces.Discrete(count)

    @classmethod
    def create(cls, env_name: str, bin_size: int = BIN_SIZE,
               keep_text: bool = False, keep_obs: bool = False,
               **kwargs) -> "MiniWoBClickWrapper":
        """
        Creates miniwob environment wrapped into the click wrapper
        :param env_name: name of the environment
        :param bin_size: size of the click bin
        :param keep_text: preserves instruction in the observation
        :param keep_obs: save original observation in info dict
        :param kwargs: extra args to gym.make()
        :return: environment
        """
        gym.register_envs(miniwob)
        x_bins = WIDTH // bin_size
        y_bins = (HEIGHT - Y_OFS) // bin_size
        act_cfg = ActionSpaceConfig(
            action_types=(ActionTypes.CLICK_COORDS, ),
            coord_bins=(x_bins, y_bins),
        )
        env = gym.make(
            env_name, action_space_config=act_cfg,
            **kwargs
        )
        return MiniWoBClickWrapper(
            env, keep_text=keep_text, keep_obs=keep_obs,
            bin_size=bin_size)

    def _observation(self, observation: dict) -> \
            np.ndarray | tt.Tuple[np.ndarray, str]:
        text = observation['utterance']
        scr = observation['screenshot']
        scr = np.transpose(scr, (2, 0, 1))
        if self.keep_text:
            return scr, text
        return scr

    def reset(
            self, *, seed: int | None = None,
            options: dict[str, tt.Any] | None = None
    ) -> tuple[gym.core.WrapperObsType, dict[str, tt.Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        if self.keep_obs:
            info[self.FULL_OBS_KEY] = obs
        return self._observation(obs), info

    def step(self, action: int) -> tt.Tuple[
        gym.core.WrapperObsType, gym.core.SupportsFloat,
        bool, bool, dict[str, tt.Any]
    ]:
        b_x, b_y = action_to_bins(action, self.bin_size)
        new_act = {
            "action_type": 0,
            "coords": np.array((b_x, b_y), dtype=np.int8),
        }
        obs, reward, is_done, is_tr, info = self.env.step(new_act)
        if self.keep_obs:
            info[self.FULL_OBS_KEY] = obs
        return self._observation(obs), reward, is_done, is_tr, info


def coord_to_action(x: int, y: int, bin_size: int = BIN_SIZE) -> int:
    """
    Convert coordinate of click into binned action
    :param x: x coordinate of click
    :param y: y coordinate of click
    :param bin_size: size of the bin
    :return: action index
    """
    y -= Y_OFS
    y = max(y, 0)
    y //= bin_size
    x //= bin_size
    return x + (WIDTH // bin_size) * y


def action_to_coord(action: int, bin_size: int = BIN_SIZE) -> tt.Tuple[int, int]:
    """
    Convert click action to coords
    :param action: action from 0 to 255 (for bin=10)
    :param bin_size: size of the bins
    :return: x, y of coordinates
    """
    b_x, b_y = action_to_bins(action, bin_size)
    d = bin_size // 2
    return (b_x * bin_size) + d, Y_OFS + (b_y * bin_size) + d


def action_to_bins(action: int,
                   bin_size: int = BIN_SIZE) -> tt.Tuple[int, int]:
    """
    Convert click action to coords
    :param action: action from 0 to 255 (for bin=10)
    :param bin_size: size of the bins
    :return: x, y of coordinates
    """
    row_bins = WIDTH // bin_size
    b_y = action // row_bins
    b_x = action % row_bins
    return b_x, b_y

