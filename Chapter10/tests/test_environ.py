import pytest
import pathlib
import typing as tt

import numpy as np

from lib import data, environ



def test_env_simple():
    prices = data.load_relative("data/YNDX_160101_161231.csv")
    env = environ.StocksEnv({"YNDX": prices})
    s = env.reset()
    obs, reward, done, is_tr, info = env.step(0)
    assert reward == pytest.approx(0.0)


@pytest.fixture
def prices() -> data.Prices:
    p = data.Prices(open=np.array([1.0, 2.0, 3.0, 1.0]),
                    high=np.array([2.0, 3.0, 4.0, 2.0]),
                    low=np.array([0.0, 1.0, 2.0, 0.0]),
                    close=np.array([2.0, 3.0, 1.0, 2.0]),
                    volume=np.array([10.0, 10.0, 10.0, 10.0]))
    return data.prices_to_relative(p)



def test_states_basic():
    s = environ.State(bars_count=4, commission_perc=0.0, reset_on_close=False, volumes=False)
    assert s.shape == (4*3+2, )


def test_basic1d(prices):
    s = environ.State1D(bars_count=2, commission_perc=0.0, reset_on_close=False, volumes=True)
    assert s.shape == (6, 2)
    s.reset(prices, 1)
    d = s.encode()
    assert s.shape == d.shape


def test_reset(prices):
    s = environ.State(bars_count=1, commission_perc=0.0, reset_on_close=False)
    s.reset(prices, offset=0)
    assert not s.have_position
    assert s._cur_close() == pytest.approx(2.0)

    r, done = s.step(environ.Actions.Skip)
    assert s._cur_close() == pytest.approx(3.0)
    assert r == pytest.approx(0.0)
    assert not done

    r, done = s.step(environ.Actions.Skip)
    assert s._cur_close() == pytest.approx(1.0)
    assert r == pytest.approx(0.0)
    assert not done

    r, done = s.step(environ.Actions.Skip)
    assert s._cur_close() == pytest.approx(2.0)
    assert r == pytest.approx(0.0)
    assert done


def test_reward(prices):
    s = environ.State(bars_count=1, commission_perc=0.0, reset_on_close=False,
                      reward_on_close=False)
    s.reset(prices, offset=0)
    assert not s.have_position
    assert s._cur_close() == pytest.approx(2.0)

    r, done = s.step(environ.Actions.Buy)
    assert s.have_position
    assert not done
    assert r == pytest.approx(50.0)
    assert s._cur_close() == pytest.approx(3.0)

    r, done = s.step(environ.Actions.Skip)
    assert not done
    assert r == pytest.approx(-2/3 * 100.0)
    assert s._cur_close() == pytest.approx(1.0)

    r, done = s.step(environ.Actions.Skip)
    assert done
    assert r == pytest.approx(100.0)
    assert s._cur_close() == pytest.approx(2.0)


def test_comission(prices):
    s = environ.State(bars_count=1, commission_perc=1.0, reset_on_close=False, reward_on_close=False)
    s.reset(prices, offset=0)
    assert not s.have_position
    assert s._cur_close() == pytest.approx(2.0)

    r, done = s.step(environ.Actions.Buy)
    assert s.have_position
    assert not done

    # execution price is the cur bar close, comission 1%, reward in percent
    assert r == pytest.approx(100.0 * (3.0 - 2.0) / 2.0 - 1.0)
    assert s._cur_close() == pytest.approx(3.0)


def test_final_reward(prices):
    s = environ.State(bars_count=1, commission_perc=0.0, reset_on_close=False, reward_on_close=True)
    s.reset(prices, offset=0)
    assert not s.have_position
    assert s._cur_close() == pytest.approx(2.0)

    r, done = s.step(environ.Actions.Buy)
    assert s.have_position
    assert not done
    assert s._cur_close() == pytest.approx(3.0)

    r, done = s.step(environ.Actions.Skip)
    assert not done
    assert s._cur_close() == pytest.approx(1.0)

    r, done = s.step(environ.Actions.Close)
    assert done
    assert r == pytest.approx(-50.0)
    assert s._cur_close() == pytest.approx(2.0)
