import numpy as np
from lib import muzero as mu
from lib import game


def test_node():
    n = mu.MCTSNode(0.5, first_plays=True)
    assert not n.is_expanded
    assert n.value == 0


def test_mcts():
    params = mu.MuZeroParams()
    models = mu.MuZeroModels(mu.OBS_SHAPE, game.GAME_COLS)
    min_max = mu.MinMaxStats()
    root = mu.run_mcts(0, game.INITIAL_STATE, params, models,
                       search_rounds=10, min_max=min_max)
    assert root.is_expanded
    assert len(root.children) == game.GAME_COLS
    assert root.visit_count == 11


def test_action_selection():
    params = mu.MuZeroParams()
    root = mu.MCTSNode(0.5, first_plays=True)
    np.random.seed(10)
    v = root.select_action(1, params)
    assert v == 1
    for a in range(params.actions_count):
        root.children[a] = mu.MCTSNode(0.1, first_plays=False)
    root.children[0].visit_count = 100
    v = root.select_action(0.0000001, params)
    assert v == 0
    v = root.select_action(0.1, params)
    assert v == 0


def test_play_game():
    params = mu.MuZeroParams()
    models = mu.MuZeroModels(mu.OBS_SHAPE, game.GAME_COLS)
    reward, episode = mu.play_game(
        models, models, params, temperature=0,
        init_state=8516337133269602564
    )
    assert episode