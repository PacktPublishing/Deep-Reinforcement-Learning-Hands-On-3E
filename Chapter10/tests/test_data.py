import numpy as np
import pathlib
from lib import data


def test_read_csv():
    prices = data.read_csv(pathlib.Path("data/YNDX_160101_161231.csv"))
    assert isinstance(prices, data.Prices)


def test_prices_to_relative():
    t = data.Prices(open=np.array([1.0]),
                    high=np.array([3.0]),
                    low=np.array([0.5]),
                    close=np.array([2.0]),
                    volume=np.array([10]))
    rel = data.prices_to_relative(t)
    np.testing.assert_equal(rel.open,  t.open)
    np.testing.assert_equal(rel.volume,  t.volume)
    np.testing.assert_equal(rel.high,  np.array([2.0]))  # 200% growth
    np.testing.assert_equal(rel.low,   np.array([-.5]))  # 50% fall
    np.testing.assert_equal(rel.close, np.array([1.0]))  # 100% growth


def test_price_files():
    files = data.price_files("data")
    assert len(files) > 0

