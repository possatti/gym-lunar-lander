#!/usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_log')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    df = pd.read_csv(args.csv_log)

    sns.set()
    n_cols = len(df.columns)
    fig, axes = plt.subplots(n_cols, 1, sharex=True)
    for i, col in enumerate(df.columns):
        axes[i].plot(df.index, df[col], label=col)
        axes[i].legend()
    plt.show()

if __name__ == "__main__":
    main()