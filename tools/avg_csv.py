#!/usr/bin/env python3
import csv
import pathlib
import argparse
import itertools
import typing as tt
from dataclasses import dataclass


@dataclass
class Series:
    start_wall: float
    time_deltas: tt.List[float]
    steps: tt.List[int]
    values: tt.List[float]

    @classmethod
    def read(cls, path: pathlib.Path) -> "Series":
        start_wall = None
        deltas = []
        steps = []
        values = []
        with path.open('rt', encoding='utf-8') as fd:
            reader = csv.DictReader(fd)
            for r in reader:
                assert isinstance(r, dict)
                t = float(r["Wall time"])
                if start_wall is None:
                    start_wall = t
                deltas.append(t - start_wall)
                steps.append(int(r["Step"]))
                values.append(float(r["Value"]))
        return Series(start_wall=start_wall, time_deltas=deltas, steps=steps, values=values)

    def write(self, path: pathlib.Path):
        with path.open('wt', encoding='utf-8') as fd:
            writer = csv.DictWriter(fd, ('Wall time', 'Step', 'Value'))
            writer.writeheader()
            for dt, s, v in zip(self.time_deltas, self.steps, self.values):
                writer.writerow({
                    'Wall time': self.start_wall + dt,
                    'Step': s,
                    'Value': v,
                })

    def __iter__(self) -> tt.Generator[tt.Tuple[float, int, float], None, None]:
        yield from zip(self.time_deltas, self.steps, self.values)


def mean_max_step(series: tt.List[Series]) -> float:
    return sum(map(lambda s: s.steps[-1], series)) / len(series)


def avg_entries(entries: tt.Tuple[tt.Optional[tt.Tuple[float, int, float]], ...]) -> tt.Tuple[float, int, float]:
    deltas = []
    steps = []
    values = []
    for entry in entries:
        if entry is None:
            continue
        d, s, v = entry
        deltas.append(d)
        steps.append(s)
        values.append(v)
    return sum(deltas) / len(deltas), int(sum(steps) / len(steps)), sum(values) / len(values)


def average_series(series: tt.List[Series]) -> Series:
    mean_steps = mean_max_step(series)
    start_wall = series[0].start_wall
    deltas = []
    steps = []
    values = []

    for vals in itertools.zip_longest(*series):
        dt, s, v = avg_entries(vals)
        if s <= mean_steps:
            deltas.append(dt)
            steps.append(s)
            values.append(v)
    return Series(start_wall=start_wall, time_deltas=deltas, steps=steps, values=values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True, help="Output csv file to produce")
    parser.add_argument("files", nargs='+', help="Input csv files")
    args = parser.parse_args()

    series = [Series.read(pathlib.Path(n)) for n in args.files]
    res = average_series(series)
    res.write(pathlib.Path(args.output))