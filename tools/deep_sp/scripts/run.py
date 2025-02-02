#!/usr/bin/env python3
import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="Run DeepSP with standardized I/O")
    parser.add_argument("--input", required=True, help="Path to the input file")
    parser.add_argument("--output", required=True, help="Path to the output file")
    args = parser.parse_args()

    # Step 1: Convert from standard input format to the tool’s format
    cmd_convert_input = [
        "python", "/app/scripts/convert_input.py",
        "--standard_in", args.input,
        "--tool_out", "/app/DeepSP/DeepSP_input.csv"
    ]
    subprocess.run(cmd_convert_input, check=True)

    # Step 2: Run the actual tool (DeepSP)
    cwd = os.getcwd()
    os.chdir("/app/DeepSP")
    cmd_run_tool = [
        "python", "deepsp_predictor.py",
    ]
    subprocess.run(cmd_run_tool, check=True)
    os.chdir(cwd)

    # Step 3: Convert from the tool’s output back to the standard format
    cmd_convert_output = [
        "python", "/app/scripts/convert_output.py",
        "--tool_in", "/app/DeepSP/DeepSP_descriptors.csv",
        "--standard_out", args.output
    ]
    subprocess.run(cmd_convert_output, check=True)

    print("Done: The pipeline steps have completed successfully!")

if __name__ == "__main__":
    main()
