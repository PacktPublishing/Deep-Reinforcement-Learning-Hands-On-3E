import gymnasium as gym
import logging
import typing as tt

import textworld.gym.envs
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from sentence_transformers import SentenceTransformer
from . import common


KEY_ADM_COMMANDS = "admissible_commands"

class TextWorldPreproc(gym.Wrapper):
    """
    Simple wrapper to preprocess text_world game observation

    Observation and action spaces are not handled, as it will
    be wrapped into other preprocessors
    """
    log = logging.getLogger("TextWorldPreproc")

    # field with observation
    OBS_FIELD = "obs"

    def __init__(
            self, env: gym.Env,
            vocab_rev: tt.Optional[tt.Dict[str, int]],
            encode_raw_text: bool = False,
            encode_extra_fields: tt.Iterable[str] = (
                 'description', 'inventory'),
            copy_extra_fields: tt.Iterable[str] = (),
            use_admissible_commands: bool = True,
            keep_admissible_commands: bool = False,
            use_intermediate_reward: bool = True,
            tokens_limit: tt.Optional[int] = None,
            reward_wrong_last_command: tt.Optional[float] = None
    ):
        """
        :param env: TextWorld env to be wrapped
        :param vocab_ver: reverse vocabulary
        :param encode_raw_text: flag to encode raw texts
        :param encode_extra_fields: fields to be encoded
        :param copy_extra_fields: fields to be copied into obs
        :param use_admissible_commands: use list of commands
        :param keep_admissible_commands: keep list of admissible commands in observations
        :param use_intermediate_reward: intermediate reward
        :param tokens_limit: limit tokens in encoded fields
        :param reward_wrong_last_command: if given, this reward will be given if 'last_command' observation field is 'None'.
        """
        super(TextWorldPreproc, self).__init__(env)
        self._vocab_rev = vocab_rev
        self._encode_raw_text = encode_raw_text
        self._encode_extra_field = tuple(encode_extra_fields)
        self._copy_extra_fields = tuple(copy_extra_fields)
        self._use_admissible_commands = use_admissible_commands
        self._keep_admissible_commands = keep_admissible_commands
        self._use_intermedate_reward = use_intermediate_reward
        self._num_fields = len(self._encode_extra_field) + \
                           int(self._encode_raw_text)
        self._last_admissible_commands = None
        self._last_extra_info = None
        self._tokens_limit = tokens_limit
        self._reward_wrong_last_command = reward_wrong_last_command
        self._cmd_hist = []

    @property
    def num_fields(self):
        return self._num_fields

    def _maybe_tokenize(self, s: str) -> str | tt.List[int]:
        """
        If dictionary is present, tokenise the string, otherwise keep intact
        :param s: string to process
        :return: tokenized string or original value
        """
        if self._vocab_rev is None:
            return s
        tokens = common.tokenize(s, self._vocab_rev)
        if self._tokens_limit is not None:
            tokens = tokens[:self._tokens_limit]
        return tokens

    def _encode(self, obs: str, extra_info: dict) -> dict:
        obs_result = []
        if self._encode_raw_text:
            obs_result.append(self._maybe_tokenize(obs))
        for field in self._encode_extra_field:
            extra = extra_info[field]
            obs_result.append(self._maybe_tokenize(extra))
        result = {self.OBS_FIELD: obs_result}
        if self._use_admissible_commands:
            result[KEY_ADM_COMMANDS] = [
                self._maybe_tokenize(cmd)
                for cmd in extra_info[KEY_ADM_COMMANDS]
            ]
            self._last_admissible_commands = \
                extra_info[KEY_ADM_COMMANDS]
        if self._keep_admissible_commands:
            result[KEY_ADM_COMMANDS] = extra_info[KEY_ADM_COMMANDS]
            if 'policy_commands' in extra_info:
                result['policy_commands'] = extra_info['policy_commands']
        self._last_extra_info = extra_info
        for field in self._copy_extra_fields:
            if field in extra_info:
                result[field] = extra_info[field]
        return result

    def reset(self, seed: tt.Optional[int] = None):
        res, extra = self.env.reset()
        self._cmd_hist = []
        return self._encode(res, extra), extra

    def step(self, action):
        if self._use_admissible_commands:
            action = self._last_admissible_commands[action]
            self._cmd_hist.append(action)
        obs, r, is_done, extra = self.env.step(action)
        if self._use_intermedate_reward:
            r += extra.get('intermediate_reward', 0)
        if self._reward_wrong_last_command is not None:
            if action not in self._last_extra_info[KEY_ADM_COMMANDS]:
                r += self._reward_wrong_last_command
        return self._encode(obs, extra), r, is_done, False, extra

    @property
    def last_admissible_commands(self):
        if self._last_admissible_commands:
            return tuple(self._last_admissible_commands)
        return None

    @property
    def last_extra_info(self):
        return self._last_extra_info


class LocationWrapper(gym.Wrapper):
    """
    Wrapper which tracks list of locations we've already seen
    """
    SEEN_LOCATION_FIELD = "location_seen"

    def __init__(self, env: gym.Env):
        super(LocationWrapper, self).__init__(env)
        self._seen_locations = set()
        self._cur_location = None

    def reset(self, *, seed: tt.Optional[int] = None):
        self._seen_locations.clear()
        self._cur_location = None
        obs, extra = self.env.reset(seed=seed)
        self._track_location(extra)
        obs[self.SEEN_LOCATION_FIELD] = int(self.location_was_seen)
        return obs, extra

    @property
    def location_was_seen(self) -> bool:
        return self._cur_location in self._seen_locations

    def _track_location(self, extra_dict: dict):
        if self._cur_location is not None:
            self._seen_locations.add(self._cur_location)
        descr = extra_dict.get('description')
        if descr is None:
            self._cur_location = None
        else:
            self._cur_location = descr.split("\n")[0]

    def step(self, action):
        obs, r, is_done, is_tr, extra = self.env.step(action)
        self._track_location(extra)
        obs[self.SEEN_LOCATION_FIELD] = int(self.location_was_seen)
        return obs, r, is_done, is_tr, extra


class RelativeDirectionWrapper(gym.Wrapper):
    """
    Wrapper which tracks our heading direction (NSWE) and adds support of
    relative navigation ("go right", "go straight", etc).
    We also add heading direction into observation.
    """
    # Directions are made relative to each other, don't update them blindly
    ABSOLUTE_DIRS = ('west', 'north', 'east', 'south')
    RELATIVE_DIRS = ('right', 'forward', 'left', 'back')
    HEADING_FIELDS = tuple("heading_" + d for d in ABSOLUTE_DIRS)
    NEW_ACTIONS = set("go " + d for d in RELATIVE_DIRS)
    OLD_ACTIONS = set("go " + d for d in ABSOLUTE_DIRS)

    OLD_ACTIONS_DIRS = {
        "go " + d: idx
        for idx, d in enumerate(ABSOLUTE_DIRS)
    }

    NEW_ACTIONS_DELTAS = {
        "go " + d: delta
        for d, delta in zip(RELATIVE_DIRS, (1, 0, 3, 2))
    }

    def __init__(self, env: textworld.gym.envs.TextworldGymEnv):
        if not isinstance(env, textworld.gym.envs.TextworldGymEnv):
            raise ValueError(f"{self.class_name()} has to be applied to TextworldGymEnv")
        super(RelativeDirectionWrapper, self).__init__(env)
        # look north
        self._heading_idx = 1

    @classmethod
    def abs_to_rel(cls, abs_action: str, cur_dir: int) -> str:
        act_dir = cls.OLD_ACTIONS_DIRS[abs_action]
        delta = act_dir - cur_dir
        if delta == 0:
            return "go forward"
        elif delta == 1 or delta == -3:
            return "go right"
        elif delta == -1 or delta == 3:
            return "go left"
        elif abs(delta) == 2:
            return "go back"
        else:
            raise RuntimeError("Unhandled value of delta=" + str(delta))

    @classmethod
    def rel_to_abs(cls, rel_action: str, cur_dir: int) -> str:
        delta = cls.NEW_ACTIONS_DELTAS[rel_action]
        new_dir = (cur_dir + delta) % 4
        return "go " + cls.ABSOLUTE_DIRS[new_dir]

    @classmethod
    def rel_execute(cls, rel_action: str, cur_dir: int) -> int:
        delta = cls.NEW_ACTIONS_DELTAS[rel_action]
        new_dir = (cur_dir + delta) % 4
        return new_dir

    @classmethod
    def update_vocabs(cls, vocab: tt.Dict[int, str], vocab_rev: tt.Dict[str, int]):
        """
        Update vocabularies to include new action words
        :param vocab: forward vocabulary
        :param vocab_rev: reverse vocabulary
        :return:
        """
        for word in cls.RELATIVE_DIRS:
            if word not in vocab_rev.keys():
                next_idx = len(vocab)
                vocab[next_idx] = word
                vocab_rev[word] = next_idx

    def _update_info(self, info: dict) -> dict:
        """
        Update information dict: replace admissible_commands into relative
        :param info: dict with extra information
        :return: updated dict (same object)
        """
        new_commands = []
        for cmd in info.get(KEY_ADM_COMMANDS, []):
            if cmd in self.OLD_ACTIONS:
                cmd = self.abs_to_rel(cmd, self._heading_idx)
            new_commands.append(cmd)
        if new_commands:
            info[KEY_ADM_COMMANDS] = new_commands
        for idx, field in enumerate(self.HEADING_FIELDS):
            info[field] = int(idx == self._heading_idx)
        return info

    def reset(self):
        # Look north initially
        self._heading_idx = 1
        obs, extra = self.env.reset()
        return obs, self._update_info(extra)

    def step(self, action: str):
        if action in self.NEW_ACTIONS:
            abs_action = self.rel_to_abs(action, self._heading_idx)
            self._heading_idx = self.rel_execute(action, self._heading_idx)
            action = abs_action
        obs, r, is_done, extra = self.env.step(action)
        return obs, r, is_done, self._update_info(extra)


class Encoder(nn.Module):
    """
    Takes input sequences (after embeddings) and returns
    the hidden state from LSTM
    """
    def __init__(self, emb_size: int, out_size: int):
        super(Encoder, self).__init__()

        self.net = nn.LSTM(
            input_size=emb_size, hidden_size=out_size,
            batch_first=True)

    def forward(self, x):
        self.net.flatten_parameters()
        _, hid_cell = self.net(x)
        # Warn: if bidir=True or several layers,
        # sequeeze has to be changed!
        return hid_cell[0].squeeze(0)


class Preprocessor(nn.Module):
    """
    Takes batch of several input sequences and outputs their
    summary from one or many encoders
    """
    def __init__(self, dict_size: int, emb_size: int,
                 num_sequences: int, enc_output_size: int,
                 extra_flags: tt.Sequence[str] = ()):
        """
        :param dict_size: amount of words is our vocabulary
        :param emb_size: dimensionality of embeddings
        :param num_sequences: count of sequences
        :param enc_output_size: output from single encoder
        :param extra_flags: list of fields from observations
        to encode as numbers
        """
        super(Preprocessor, self).__init__()
        self._extra_flags = extra_flags
        self._enc_output_size = enc_output_size
        self.emb = nn.Embedding(num_embeddings=dict_size,
                                embedding_dim=emb_size)
        self.encoders = []
        for idx in range(num_sequences):
            enc = Encoder(emb_size, enc_output_size)
            self.encoders.append(enc)
            self.add_module(f"enc_{idx}", enc)
        self.enc_commands = Encoder(emb_size, enc_output_size)

    @property
    def obs_enc_size(self):
        return self._enc_output_size * len(self.encoders) + \
            len(self._extra_flags)

    @property
    def cmd_enc_size(self):
        return self._enc_output_size

    def _apply_encoder(self, batch: tt.List[tt.List[int]],
                       encoder: Encoder):
        dev = self.emb.weight.device
        batch_t = [self.emb(torch.tensor(sample).to(dev))
                   for sample in batch]
        batch_seq = rnn_utils.pack_sequence(
            batch_t, enforce_sorted=False)
        return encoder(batch_seq)

    def encode_observations(self, observations: tt.List[dict]) -> \
            torch.Tensor:
        sequences = [
            obs[TextWorldPreproc.OBS_FIELD]
            for obs in observations
        ]
        res_t = self.encode_sequences(sequences)
        if not self._extra_flags:
            return res_t
        extra = [
            [obs[field] for field in self._extra_flags]
            for obs in observations
        ]
        extra_t = torch.Tensor(extra).to(res_t.device)
        res_t = torch.cat([res_t, extra_t], dim=1)
        return res_t

    def encode_sequences(self, batches):
        """
        Forward pass of Preprocessor
        :param batches: list of tuples with variable-length sequences of word ids
        :return: tensor with concatenated encoder outputs for every batch sample
        """
        data = []
        for enc, enc_batch in zip(self.encoders, zip(*batches)):
            data.append(self._apply_encoder(enc_batch, enc))
        res_t = torch.cat(data, dim=1)
        return res_t

    def encode_commands(self, batch):
        """
        Apply encoder to list of commands sequence
        :param batch: list of lists of idx
        :return: tensor with encoded commands in original order
        """
        return self._apply_encoder(batch, self.enc_commands)


class TransformerPreprocessor:
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, num_sequences: int,
                 device: torch.device,
                 extra_flags: tt.Sequence[str] = (),
                 model_name: str = DEFAULT_MODEL):
        self._device = device
        self._transformer = SentenceTransformer(model_name, device=device.type)
        self._emb_size = self._transformer.get_sentence_embedding_dimension()
        self._extra_flags = extra_flags
        self._num_sequences = num_sequences

    @property
    def obs_enc_size(self) -> int:
        return self._emb_size * self._num_sequences + len(self._extra_flags)

    @property
    def cmd_enc_size(self) -> int:
        return self._emb_size

    def encode_observations(self, observations: tt.List[dict]) -> \
            torch.Tensor:
        sequences = [
            obs[TextWorldPreproc.OBS_FIELD]
            for obs in observations
        ]
        res_t = self.encode_sequences(sequences)
        if not self._extra_flags:
            return res_t

        extra = [
            [obs[field] for field in self._extra_flags]
            for obs in observations
        ]
        extra_t = torch.Tensor(extra).to(res_t.device)
        res_t = torch.cat([res_t, extra_t], dim=1)
        return res_t

    def encode_sequences(self, batches: tt.List[tt.Tuple[str, ...]]) -> torch.Tensor:
        """
        Forward pass of Preprocessor
        :param batches: list of tuples with strings
        :return: tensor with concatenated encoder outputs for every batch sample
        """
        data = []
        for b in batches:
            data.extend(b)
        res_t = self._transformer.encode(
            data, convert_to_tensor=True
        )
        res_t = res_t.reshape((len(batches), len(batches[0]) * self._emb_size))
        return res_t

    def encode_commands(self, batch: tt.List[str]) -> torch.Tensor:
        """
        Apply encoder to list of commands sequence
        :param batch: list of string
        :return: tensor with encoded commands in original order
        """
        return self._transformer.encode(
            batch, convert_to_tensor=True)