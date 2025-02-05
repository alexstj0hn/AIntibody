import argparse
import subprocess

def main():
    
    parser = argparse.ArgumentParser(description="Run Chai-1")

    parser.add_argument("--input", required=True, help="Path to the input file (fasta format)")
    parser.add_argument("--output_dir", required=True, help="Path to the output file")
    args = parser.parse_args()

    result = subprocess.run(["chai", "fold", args.input, args.output_dir, "--use-msa-server"], stdout=subprocess.PIPE)

    print(result.stdout)

if __name__ == "__main__":
    main()