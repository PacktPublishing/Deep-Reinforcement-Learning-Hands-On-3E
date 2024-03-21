import os
import csv
import glob
import pathlib
import numpy as np
import typing as tt
from dataclasses import dataclass


@dataclass
class Prices:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray


def read_csv(file_path: pathlib.Path, sep: str = ',',
             filter_data: bool = True,
             fix_open_price: bool = False) -> Prices:
    print("Reading", file_path)
    with file_path.open('rt', encoding='utf-8') as fd:
        reader = csv.reader(fd, delimiter=sep)
        h = next(reader)
        if '<OPEN>' not in h and sep == ',':
            return read_csv(file_path, ';')
        indices = [
            h.index(s)
            for s in ('<OPEN>', '<HIGH>', '<LOW>',
                      '<CLOSE>', '<VOL>')
        ]
        o, h, l, c, v = [], [], [], [], []
        count_out = 0
        count_filter = 0
        count_fixed = 0
        prev_vals = None
        filter_func = lambda v: abs(v-vals[0]) < 1e-8
        for row in reader:
            vals = list(map(float, [row[idx] for idx in indices]))
            if filter_data and all(map(filter_func, vals[:-1])):
                count_filter += 1
                continue

            po, ph, pl, pc, pv = vals

            # fix open price for current bar to match close price for the previous bar
            if fix_open_price and prev_vals is not None:
                ppo, pph, ppl, ppc, ppv = prev_vals
                if abs(po - ppc) > 1e-8:
                    count_fixed += 1
                    po = ppc
                    pl = min(pl, po)
                    ph = max(ph, po)
            count_out += 1
            o.append(po)
            c.append(pc)
            h.append(ph)
            l.append(pl)
            v.append(pv)
            prev_vals = vals
    print(f"Read done, got {count_filter + count_out} rows, "
          f"{count_filter} filtered, "
          f"{count_fixed} open prices adjusted")
    return Prices(open=np.array(o, dtype=np.float32),
                  high=np.array(h, dtype=np.float32),
                  low=np.array(l, dtype=np.float32),
                  close=np.array(c, dtype=np.float32),
                  volume=np.array(v, dtype=np.float32))


def prices_to_relative(prices: Prices):
    """
    Convert prices to relative in respect to open price
    :param ochl: tuple with open, close, high, low
    :return: tuple with open, rel_close, rel_high, rel_low
    """
    rh = (prices.high - prices.open) / prices.open
    rl = (prices.low - prices.open) / prices.open
    rc = (prices.close - prices.open) / prices.open
    return Prices(open=prices.open, high=rh, low=rl,
                  close=rc, volume=prices.volume)


def load_relative(csv_path: pathlib.Path | str) -> Prices:
    if isinstance(csv_path, str):
        csv_path = pathlib.Path(csv_path)
    return prices_to_relative(read_csv(csv_path))


def price_files(dir_name: str) -> tt.List[pathlib.Path]:
    result = []
    for path in glob.glob(os.path.join(dir_name, "*.csv")):
        result.append(pathlib.Path(path))
    return result


def load_year_data(
        year: int, basedir: str = 'data'
) -> tt.Dict[str, Prices]:
    y = str(year)[-2:]
    result = {}
    for path in glob.glob(os.path.join(basedir, "*_%s*.csv" % y)):
        result[path] = load_relative(pathlib.Path(path))
    return result
