#!/usr/bin/env python3
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--standard_out', required=True)
    parser.add_argument('--tool_in', required=True)
    args = parser.parse_args()

    # Read standard input
    data = pd.read_csv(args.tool_in)

    # drop first column
    data = data.drop(data.columns[0], axis=1)

    data.to_csv(args.standard_out, index=False)

if __name__ == "__main__":
    main()
