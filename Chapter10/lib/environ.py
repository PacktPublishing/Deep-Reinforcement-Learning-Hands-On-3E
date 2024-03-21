import typing as tt
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
import enum
import numpy as np

from . import data

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2


class State:
    def __init__(self, bars_count: int, commission_perc: float,
                 reset_on_close: bool, reward_on_close: bool = True,
                 volumes: bool = True):
        assert bars_count > 0
        assert commission_perc >= 0.0
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes
        self.have_position = False
        self.open_price = 0.0
        self._prices = None
        self._offset = None

    def reset(self, prices: data.Prices, offset: int):
        assert offset >= self.bars_count-1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    @property
    def shape(self) -> tt.Tuple[int, ...]:
        # [h, l, c] * bars + position_flag + rel_profit
        if self.volumes:
            return 4 * self.bars_count + 1 + 1,
        else:
            return 3 * self.bars_count + 1 + 1,

    def encode(self) -> np.ndarray:
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count+1, 1):
            ofs = self._offset + bar_idx
            res[shift] = self._prices.high[ofs]
            shift += 1
            res[shift] = self._prices.low[ofs]
            shift += 1
            res[shift] = self._prices.close[ofs]
            shift += 1
            if self.volumes:
                res[shift] = self._prices.volume[ofs]
                shift += 1
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = self._cur_close() / self.open_price - 1.0
        return res

    def _cur_close(self) -> float:
        """
        Calculate real close price for the current bar
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)

    def step(self, action: Actions) -> tt.Tuple[float, bool]:
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action: action to be executed
        :return: reward, done
        """
        reward = 0.0
        done = False
        close = self._cur_close()
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            reward -= self.commission_perc
        elif action == Actions.Close and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += 100.0 * (close / self.open_price - 1.0)
            self.have_position = False
            self.open_price = 0.0

        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1

        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close / prev_close - 1.0)

        return reward, done


class State1D(State):
    """
    State with shape suitable for 1D convolution
    """
    @property
    def shape(self) -> tt.Tuple[int, ...]:
        if self.volumes:
            return 6, self.bars_count
        else:
            return 5, self.bars_count

    def encode(self) -> np.ndarray:
        res = np.zeros(shape=self.shape, dtype=np.float32)
        start = self._offset-(self.bars_count-1)
        stop = self._offset+1
        res[0] = self._prices.high[start:stop]
        res[1] = self._prices.low[start:stop]
        res[2] = self._prices.close[start:stop]
        if self.volumes:
            res[3] = self._prices.volume[start:stop]
            dst = 4
        else:
            dst = 3
        if self.have_position:
            res[dst] = 1.0
            res[dst+1] = self._cur_close() / self.open_price - 1.0
        return res


class StocksEnv(gym.Env):
    spec = EnvSpec("StocksEnv-v0")

    def __init__(
            self, prices: tt.Dict[str, data.Prices],
            bars_count: int = DEFAULT_BARS_COUNT,
            commission: float = DEFAULT_COMMISSION_PERC,
            reset_on_close: bool = True, state_1d: bool = False,
            random_ofs_on_reset: bool = True,
            reward_on_close: bool = False, volumes=False
    ):
        self._prices = prices
        if state_1d:
            self._state = State1D(
                bars_count, commission, reset_on_close,
                reward_on_close=reward_on_close, volumes=volumes)
        else:
            self._state = State(
                bars_count, commission, reset_on_close,
                reward_on_close=reward_on_close, volumes=volumes)
        self.action_space = spaces.Discrete(n=len(Actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, tt.Any] | None = None,
    ):
        # make selection of the instrument and it's offset. Then reset the state
        super().reset(seed=seed, options=options)
        self._instrument = self.np_random.choice(
            list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(
                prices.high.shape[0]-bars*10) + bars
        else:
            offset = bars
        self._state.reset(prices, offset)
        return self._state.encode(), {}

    def step(self, action_idx: int) -> \
            tt.Tuple[np.ndarray, float, bool, bool, dict]:
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {
            "instrument": self._instrument,
            "offset": self._state._offset
        }
        return obs, reward, done, False, info

    @classmethod
    def from_dir(cls, data_dir: str, **kwargs):
        prices = {
            file: data.load_relative(file)
            for file in data.price_files(data_dir)
        }
        return StocksEnv(prices, **kwargs)
