#!/usr/bin/env python3
import csv
import argparse
import ballpark
import collections
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True, help="Output file name to produce")
    parser.add_argument("-x", default="Train steps", help="X label")
    parser.add_argument("-y", default="Reward", help="Y label")
    parser.add_argument("-i", "--input", action='append', help="Input files to process")
    parser.add_argument("-l", "--legend", action='append', help="Set label for legend")
    parser.add_argument("--ylog", action='store_true', default=False, help="Set y axis log scale")
    parser.add_argument("--lloc", default='upper left', help="Sets legend location")
    parser.add_argument("--max-dt", type=float, default=None, help="Maximum length in hours, default=No limit")
    parser.add_argument("--use-steps", action="store_true", default=False,
                        help="Use steps for comparison instead of time")
    parser.add_argument("--sma", type=int, default=None, help="Apply SMA with given window, default=disabled")
    args = parser.parse_args()

    data = []
    max_steps = None
    max_hours = None
    deque = None
    if args.sma is not None:
        deque = collections.deque(maxlen=args.sma)

    for file_name in args.input:
        hours = []
        steps = []
        vals = []
        with open(file_name, 'rt', encoding='utf-8') as fd:
            min_time = None
            for row in csv.DictReader(fd):
                t = float(row["Wall time"])
                if min_time is None:
                    min_time = t
                h = float(t - min_time)/3600
                if args.max_dt is not None and args.max_dt < h:
                    continue
                hours.append(h)
                s = float(row["Step"])
                steps.append(s)
                v = float(row["Value"])
                if deque is not None:
                    deque.append(v)
                    v = sum(deque) / len(deque)
                vals.append(v)
                if max_steps is None or max_steps < s:
                    max_steps = s
                if max_hours is None or max_hours < h:
                    max_hours = h
        data.append([hours, steps, vals])

    x_label = "Hours"
    if max_hours < 2/60:
        x_label = "Seconds"
        for hours, _, _ in data:
            for idx, h in enumerate(hours):
                hours[idx] = h * 3600.0
        max_hours *= 3600
    if max_hours < 0.2:
        x_label = "Minutes"
        for hours, _, _ in data:
            for idx, h in enumerate(hours):
                hours[idx] = h * 60.0
        max_hours *= 60

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    if not args.legend:
        ax2 = ax1.twiny()

    for (hours, steps, vals), style in zip(data, ('-', ':', '--', '-.')):
        x = steps if args.use_steps else hours
        ax1.plot(x, vals, color='black', linewidth=.8, linestyle=style)

    def label_formatter(x, pos):
        v = int(x)
        if v < 0:
            return v
        elif v <= 9:
            return ballpark.business(v, precision=1)
        elif v <= 99:
            return ballpark.business(v, precision=2)
        else:
            return ballpark.business(v, precision=3)
        
    ax1.grid(True, axis='both')
    if args.legend:
        ax1.legend(args.legend, loc=args.lloc, fancybox=True)
    if args.use_steps:
        x_label = args.x
        ax1.xaxis.set_major_formatter(FuncFormatter(label_formatter))
    else:
        ax1.set_xlim(0, max_hours)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(args.y)

    if args.ylog:
        plt.yscale('log')
    if not args.legend:
        ax2.xaxis.set_major_formatter(FuncFormatter(label_formatter))

#    new_tick_locations = np.linspace(0, max_steps, num=10)

        ax2.set_xlim(0, max_steps)
#    ax2.set_xticks(new_tick_locations)
#    ax2.set_xticklabels(["%.1f" % v for v in new_tick_locations])
        ax2.set_xlabel(args.x)
        ax2.grid(True, axis='y')
    plt.savefig(args.output)
#    plt.show()
