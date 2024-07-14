import logging
import random
import pickle
import json
import numpy as np
import collections
from dataclasses import dataclass
import typing as tt
import gymnasium as gym
import torch
import queue
from torch import nn
from gymnasium.core import WrapperObsType, WrapperActType, SupportsFloat
import pathlib
from copy import deepcopy
from PIL import Image

log = logging.getLogger("rlhf")


# how many transitions to store in episode
EPISODE_STEPS = 50
# probability to start episode recording
START_PROB = 0.00005
LABELS_FILE_NAME = "labels.json"


@dataclass(frozen=True)
class EpisodeStep:
    obs: np.ndarray
    act: int


@dataclass()
class HumanLabel:
    sample1: pathlib.Path
    sample2: pathlib.Path
    # 1 if sample 1 is better than 2
    # 2 if sample 2 is better than 1
    # 0 if they are equal
    label: tt.Optional[int]

    def to_json(self, extra_id: tt.Optional[int] = None) -> dict:
        res = {
            "sample1": str(self.sample1),
            "sample2": str(self.sample2),
            "label": self.label,
        }
        if extra_id is not None:
            res['id'] = extra_id
        return res

    @classmethod
    def from_json(cls, data: dict) -> "HumanLabel":
        return HumanLabel(
            sample1=pathlib.Path(data['sample1']),
            sample2=pathlib.Path(data['sample2']),
            label=int(data['label']),
        )


@dataclass
class Database:
    db_root: pathlib.Path
    paths: tt.List[pathlib.Path]
    labels: tt.List[HumanLabel]

    def shuffle_labels(self, seed: tt.Optional[int] = None):
        random.seed(seed)
        random.shuffle(self.labels)


class EpisodeRecorderWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, db_path: pathlib.Path,
                 env_idx: int, start_prob: float = START_PROB,
                 steps_count: int = EPISODE_STEPS):
        """
        Constructs the DB episode storage wrapper.
        :param env: environment to wrap
        :param db_path: path to the episode DB
        :param env_idx: index of the environment, used for storing
        :param start_prob: probability to start the episode capture
        :param steps_count: count of steps to capture
        """
        super().__init__(env)
        self._store_path = db_path / f"{env_idx:02d}"
        self._store_path.mkdir(parents=True, exist_ok=True)
        self._start_prob = start_prob
        self._steps_count = steps_count
        self._is_storing = False
        self._steps: tt.List[EpisodeStep] = []
        self._prev_obs = None
        self._step_idx = 0

    def reset(
        self, *, seed: int | None = None,
            options: dict[str, tt.Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, tt.Any]]:
        self._step_idx += 1
        res = super().reset(seed=seed, options=options)
        if self._is_storing:
            self._prev_obs = deepcopy(res[0])
        return res

    def step(
        self, action: WrapperActType
    ) -> tuple[
        WrapperObsType, SupportsFloat, bool, bool, dict[str, tt.Any]
    ]:
        self._step_idx += 1
        obs, r, is_done, is_tr, extra = super().step(action)
        if self._is_storing:
            self._steps.append(
                EpisodeStep(self._prev_obs, int(action))
            )
            self._prev_obs = deepcopy(obs)

            if len(self._steps) >= self._steps_count:
                store_segment(self._store_path, self._step_idx,
                              self._steps)
                self._is_storing = False
                self._steps.clear()
        elif random.random() <= self._start_prob:
            # start recording
            self._is_storing = True
            self._prev_obs = deepcopy(obs)
        return obs, r, is_done, is_tr, extra


def store_segment(root_path: pathlib.Path, step_idx: int,
                  steps: tt.List[EpisodeStep]):
    out_path = root_path / f"{step_idx:08d}.dat"
    dat = pickle.dumps(steps)
    out_path.write_bytes(dat)
    print(f"Stored {out_path}")


def load_labels(path: pathlib.Path) -> tt.List[HumanLabel]:
    """
    Load labels from the file, but keeping only the last entries with the same (source1, source2) key.
    This is needed, as the same samples could be labelled several times.
    :param path: path to read
    :return: list of labels
    """
    res = []
    if not path.exists():
        return res
    seen_s12 = set()
    for l in reversed(path.read_text().splitlines()):
        if not l.strip():
            continue
        d = json.loads(l)
        hl = HumanLabel.from_json(d)
        key = (hl.sample1, hl.sample2)
        if key in seen_s12:
            continue
        seen_s12.add(key)
        res.append(hl)
    return res


def load_db(db_path: str) -> Database:
    db_path = pathlib.Path(db_path)
    labels = load_labels(db_path / LABELS_FILE_NAME)
    paths = [
        p.relative_to(db_path)
        for p in sorted(db_path.glob("*/*.dat"))
    ]
    return Database(db_root=db_path, paths=paths, labels=labels)


def sample_to_label(db: Database, count: int = 20) -> tt.List[HumanLabel]:
    have_pairs = {
        (label.sample1, label.sample2) for label in db.labels
    }
    res = []
    while len(res) < count:
        p1, p2 = random.sample(db.paths, 2)
        if (p1, p2) in have_pairs:
            continue
        res.append(HumanLabel(sample1=p1, sample2=p2, label=None))
        have_pairs.add((p1, p2))
    return res


def get_episode_gif(path: pathlib.Path) -> pathlib.Path:
    """
    Return episode's gif file (if exists). Otherwise create one.
    :param path: path to data file
    :return: gif path
    """
    gif_path = path.with_suffix(".gif")
    if gif_path.exists():
        return gif_path

    dat = path.read_bytes()
    steps = pickle.loads(dat)
    sh = steps[0].obs.shape
    im = Image.new("RGB", (sh[1], sh[0]), (0, 0, 0))
    images = [
        Image.fromarray(step.obs)
        for step in steps
    ]
    im.save(gif_path, save_all=True, append_images=images,
            duration=300, loop=0)
    return gif_path


def store_label(db: Database, label: HumanLabel):
    labels_path = db.db_root / LABELS_FILE_NAME
    with labels_path.open("a") as fd:
        fd.write(json.dumps(label.to_json()) + "\n")


def steps_to_tensors(path: pathlib.Path, total_actions: int) -> \
    tt.Tuple[torch.Tensor, torch.Tensor]:
    dat = path.read_bytes()
    steps = pickle.loads(dat)
    obs = np.stack([s.obs for s in steps])
    # put channels first for pytorch convolution
    obs = np.moveaxis(obs, (3, ), (1, ))
    act_idx = np.array([s.act for s in steps])
    acts = np.eye(total_actions)[act_idx]
    return torch.as_tensor(obs, dtype=torch.uint8), \
        torch.as_tensor(acts, dtype=torch.float32)


class RewardModel(nn.Module):
    def __init__(self, input_shape: tt.Tuple[int, ...],
                 n_actions: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=7, stride=3),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        size = self.conv(torch.zeros(1, *input_shape)).size()[-1]
        self.out = nn.Sequential(
            nn.Linear(size + n_actions, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs: torch.ByteTensor,
                acts: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(obs / 255)
        comb = torch.hstack((conv_out, acts))
        out = self.out(comb)
        return out


class RewardModelWrapper(gym.Wrapper):
    KEY_REAL_REWARD_SUM = "real_reward_sum"
    KEY_REWARD_MU = "reward_mu"
    KEY_REWARD_STD = "reward_std"

    def __init__(self, env: gym.Env, model_path: pathlib.Path,
                 dev: torch.device, reward_window: int = 100,
                 metrics_queue: tt.Optional[queue.Queue] = None):
        """
        Constructs reward model wrapper. Use given model
        to get the reward for the observations. This reward is
        used instead of the real environment reward.
        :param env: environment to wrap
        :param model_path: path to the model weights
        :param reward_window: size of the window for reward normalisation
        :param metrics_queue: queue to send extra metrics
        """
        super().__init__(env)
        self.device = dev
        assert isinstance(env.action_space, gym.spaces.Discrete)
        s = env.observation_space.shape
        self.total_actions = env.action_space.n
        self.model = RewardModel(
            input_shape=(s[2], s[0], s[1]),
            n_actions=self.total_actions
        )
        self.model.load_state_dict(
            torch.load(model_path,
                       map_location=torch.device('cpu'))
        )
        self.model.eval()
        self.model.to(dev)
        self._prev_obs = None
        self._reward_window = collections.deque(maxlen=reward_window)
        self._real_reward_sum = 0.0
        self._metrics_queue = metrics_queue

    def reset(
            self, *, seed: int | None = None,
            options: dict[str, tt.Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, tt.Any]]:
        res = super().reset(seed=seed, options=options)
        self._prev_obs = deepcopy(res[0])
        self._real_reward_sum = 0.0
        return res

    def step(
        self, action: WrapperActType
    ) -> tuple[
        WrapperObsType, SupportsFloat, bool, bool, dict[str, tt.Any]
    ]:
        obs, r, is_done, is_tr, extra = super().step(action)
        self._real_reward_sum += r
        p_obs = np.moveaxis(self._prev_obs, (2, ), (0, ))
        p_obs_t = torch.as_tensor(p_obs).to(self.device)
        p_obs_t.unsqueeze_(0)
        act = np.eye(self.total_actions)[[action]]
        act_t = torch.as_tensor(act, dtype=torch.float32).\
            to(self.device)
        new_r_t = self.model(p_obs_t, act_t)
        new_r = float(new_r_t.item())

        # track reward for normalization
        self._reward_window.append(new_r)
        if len(self._reward_window) == self._reward_window.maxlen:
            mu = np.mean(self._reward_window)
            std = np.std(self._reward_window)
            new_r -= mu
            new_r /= std
            self._metrics_queue.put((self.KEY_REWARD_MU, mu))
            self._metrics_queue.put((self.KEY_REWARD_STD, std))

        if is_done or is_tr:
            self._metrics_queue.put(
                (self.KEY_REAL_REWARD_SUM, self._real_reward_sum)
            )
        self._prev_obs = deepcopy(obs)
        return obs, new_r, is_done, is_tr, extra
