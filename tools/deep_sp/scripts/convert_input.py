#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--standard_in', required=True)
    parser.add_argument('--tool_out', required=True)
    args = parser.parse_args()

    # Read standard input
    data = pd.read_csv(args.standard_in)

    # Convert 
    df = pd.DataFrame()
    df['Name'] = [f'mAb{i+1}' for i in range(len(data))]
    df['Heavy_Chain'] = data['sequence_aa_heavy']
    df['Light_Chain'] = data['sequence_aa_light']

    df.to_csv(args.tool_out, index=False)

if __name__ == "__main__":
    main()
