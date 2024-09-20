import numpy as np
import typing as tt
import collections
from dataclasses import dataclass

from magent2.builtin.config.forest import get_config as forest_config
from magent2.builtin.config.double_attack import get_config as double_attack_config
from magent2.builtin.config.battle import get_config as battle_config
from magent2.gridworld import GridWorld

from gymnasium.utils import EzPickle
from magent2.environments.magent_env import magent_parallel_env

from ptan.experience import ExperienceFirstLast
from ptan.agent import BaseAgent, States, AgentStates


MAP_SIZE = 64
COUNT_WALLS = int(MAP_SIZE * MAP_SIZE * 0.04)
COUNT_DEER = int(MAP_SIZE * MAP_SIZE * 0.05)
COUNT_TIGERS = int(MAP_SIZE * MAP_SIZE * 0.01)
COUNT_BATTLERS = int(MAP_SIZE * MAP_SIZE * 0.02)
MAX_CYCLES = 300


class ForestEnv(magent_parallel_env, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "forest_v4",
        "render_fps": 5,
    }

    def __init__(
        self,
        map_size: int = MAP_SIZE,
        max_cycles: int = MAX_CYCLES,
        extra_features: bool = False,
        render_mode: tt.Optional[str] = None,
        seed: tt.Optional[int] = None,
        count_walls: int = COUNT_WALLS,
        count_deer: int = COUNT_DEER,
        count_tigers: int = COUNT_TIGERS,
    ):
        EzPickle.__init__(
            self, map_size, max_cycles,
            extra_features, render_mode, seed,
        )
        env = GridWorld(
            self.get_config(map_size), map_size=map_size
        )

        handles = env.get_handles()
        self.count_walls = count_walls
        self.count_deer = count_deer
        self.count_tigers = count_tigers

        names = ["deer", "tiger"]
        super().__init__(
            env, handles, names, map_size, max_cycles,
            [-1, 1], False, extra_features, render_mode,
        )

    @classmethod
    def get_config(cls, map_size: int):
        # Standard forest config, but deer get reward after every step
        cfg = forest_config(map_size)
        cfg.agent_type_dict["deer"]["step_reward"] = 1
        return cfg

    def generate_map(self):
        env, map_size = self.env, self.map_size
        handles = env.get_handles()

        env.add_walls(method="random", n=self.count_walls)
        env.add_agents(handles[0], method="random", n=self.count_deer)
        env.add_agents(handles[1], method="random", n=self.count_tigers)


class DoubleAttackEnv(magent_parallel_env, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "tiger_deer_v4",
        "render_fps": 5,
    }

    def __init__(
        self,
        map_size: int = MAP_SIZE,
        max_cycles: int = MAX_CYCLES,
        extra_features: bool = False,
        render_mode: tt.Optional[str] = None,
        seed: tt.Optional[int] = None,
        count_walls: int = COUNT_WALLS,
        count_deer: int = COUNT_DEER,
        count_tigers: int = COUNT_TIGERS,
    ):
        EzPickle.__init__(
            self, map_size, max_cycles,
            extra_features, render_mode, seed,
        )
        env = GridWorld(
            self.get_config(map_size), map_size=map_size
        )

        handles = env.get_handles()
        self.count_walls = count_walls
        self.count_deer = count_deer
        self.count_tigers = count_tigers

        names = ["deer", "tiger"]
        super().__init__(
            env, handles, names, map_size, max_cycles,
            [-1, 1], False, extra_features, render_mode,
        )

    @classmethod
    def get_config(cls, map_size: int):
        # Standard forest config, but deer get reward after every step
        cfg = double_attack_config(map_size)
        cfg.agent_type_dict["deer"]["step_reward"] = 1
        return cfg

    def generate_map(self):
        env, map_size = self.env, self.map_size
        handles = env.get_handles()

        env.add_walls(method="random", n=self.count_walls)
        env.add_agents(handles[0], method="random", n=self.count_deer)
        env.add_agents(handles[1], method="random", n=self.count_tigers)


class BattleEnv(magent_parallel_env, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "battle_v4",
        "render_fps": 5,
    }

    def __init__(
        self,
        map_size: int = MAP_SIZE,
        max_cycles: int = MAX_CYCLES,
        extra_features: bool = False,
        render_mode: tt.Optional[str] = None,
        seed: tt.Optional[int] = None,
        count_walls: int = COUNT_WALLS,
        count_a: int = COUNT_BATTLERS,
        count_b: int = COUNT_BATTLERS,
    ):
        EzPickle.__init__(
            self, map_size, max_cycles,
            extra_features, render_mode, seed,
        )
        env = GridWorld(
            self.get_config(map_size), map_size=map_size
        )

        handles = env.get_handles()
        self.count_walls = count_walls
        self.count_a = count_a
        self.count_b = count_b

        names = ["a", "b"]
        super().__init__(
            env, handles, names, map_size, max_cycles,
            [-1, 1], False, extra_features, render_mode,
        )

    @classmethod
    def get_config(cls, map_size: int):
        cfg = battle_config(map_size)
        return cfg

    def generate_map(self):
        env, map_size = self.env, self.map_size
        handles = env.get_handles()

        env.add_walls(method="random", n=self.count_walls)
        env.add_agents(handles[0], method="random", n=self.count_a)
        env.add_agents(handles[1], method="random", n=self.count_b)


@dataclass(frozen=True)
class ExperienceFirstLastMARL(ExperienceFirstLast):
    group: str


class MAgentExperienceSourceFirstLast:
    """
    2-step experience source for MAgent parallel environment
    """
    def __init__(
            self, env: magent_parallel_env,
            agents_by_group: tt.Dict[str, BaseAgent],
            track_reward_group: str,
            env_seed: tt.Optional[int] = None,
            filter_group: tt.Optional[str] = None,
    ):
        self.env = env
        self.agents_by_group = agents_by_group
        self.track_reward_group = track_reward_group
        self.env_seed = env_seed
        self.filter_group = filter_group
        self.total_rewards = []
        self.total_steps = []

        # forward and inverse map of agent_id -> group
        self.agent_groups = {
            agent_id: self.agent_group(agent_id)
            for agent_id in self.env.agents
        }
        self.group_agents = collections.defaultdict(list)
        for agent_id, group in self.agent_groups.items():
            self.group_agents[group].append(agent_id)

    @classmethod
    def agent_group(cls, agent_id: str) -> str:
        a, _ = agent_id.split("_", maxsplit=1)
        return a

    def __iter__(self) -> \
            tt.Generator[ExperienceFirstLastMARL, None, None]:
        # iterate episodes
        while True:
            # initial observation
            cur_obs = self.env.reset(self.env_seed)

            # agent states are kept in groups
            agent_states = {
                prefix: [
                    self.agents_by_group[prefix].initial_state()
                    for _ in group
                ]
                for prefix, group in self.group_agents.items()
            }

            episode_steps = 0
            episode_rewards = 0.0
            # steps while we have alive agents
            while self.env.agents:
                # calculate actions for the whole group and unpack
                actions = {}
                for prefix, group in self.group_agents.items():
                    gr_obs = [
                        cur_obs[agent_id]
                        for agent_id in group
                        if agent_id in cur_obs
                    ]
                    gr_actions, gr_states = \
                        self.agents_by_group[prefix](
                            gr_obs, agent_states[prefix])
                    agent_states[prefix] = gr_states
                    idx = 0
                    for agent_id in group:
                        if agent_id not in cur_obs:
                            continue
                        actions[agent_id] = gr_actions[idx]
                        idx += 1
                # perform the action
                new_obs, rewards, dones, truncs, _ = \
                    self.env.step(actions)

                # compute and yeld experience items
                # list of agents was updated (deads cleared),
                # need to process returned data for all agents
                for agent_id, reward in rewards.items():
                    group = self.agent_groups[agent_id]
                    if group == self.track_reward_group:
                        episode_rewards += reward
                    if self.filter_group is not None:
                        if group != self.filter_group:
                            continue
                    last_state = new_obs[agent_id]
                    if dones[agent_id] or truncs[agent_id]:
                        last_state = None
                    yield ExperienceFirstLastMARL(
                        state=cur_obs[agent_id],
                        action=actions[agent_id],
                        reward=reward,
                        last_state=last_state,
                        group=group,
                    )
                # update observations
                cur_obs = new_obs
                episode_steps += 1
            # episode ended
            self.total_steps.append(episode_steps)
            tr_group = self.group_agents[self.track_reward_group]
            self.total_rewards.append(
                episode_rewards / len(tr_group))

    def pop_total_rewards(self) -> tt.List[float]:
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self) -> tt.List[tt.Tuple[float, int]]:
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res


class RandomMAgent(BaseAgent):
    def __init__(self, env: magent_parallel_env, handle):
        self.env = env.env
        self.handle = handle

    def __call__(
            self, states: States,
            agent_states: AgentStates = None,
    ) -> tt.Tuple[np.ndarray, AgentStates]:
        n_actions = self.env.get_action_space(self.handle)[0]
        if isinstance(states, list):
            size = len(states)
        else:
            size = states.shape[0]
        res = np.random.randint(n_actions, size=size)
        return res, agent_states
