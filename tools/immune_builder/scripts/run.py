#!/usr/bin/env python3
import argparse
from ImmuneBuilder import ABodyBuilder2
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser(description="Run ImmuneBuilder")
    parser.add_argument("--input", required=True, help="Path to the input file")
    parser.add_argument("--output_dir", required=True, help="Path to the output file")
    args = parser.parse_args()

    predictor = ABodyBuilder2()

    df = pd.read_csv(args.input)

    for idx, row in df.iterrows():
        print(f"Processing antibody {idx+1}/{len(df)}...")
        output_file = os.path.join(args.output_dir, f"antibody_{idx+1}.pdb")
        sequences = {
            'H': row["sequence_aa_heavy"],
            'L': row["sequence_aa_light"]
        }

        antibody = predictor.predict(sequences)
        antibody.save(output_file)
        print(f"Antibody structure saved to: {output_file}")

    print("Done: The pipeline steps have completed successfully!")

if __name__ == "__main__":
    main()
