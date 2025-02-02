#!/usr/bin/env python3
import os
import sys
import subprocess

DOCKER_IMAGE_NAME = "deepsp"

def build_docker_image():
    """Checks if the Docker image exists, and builds it if necessary."""
    result = subprocess.run(["docker", "images", "-q", DOCKER_IMAGE_NAME], capture_output=True, text=True)
    if not result.stdout.strip():
        print(f"Building Docker image '{DOCKER_IMAGE_NAME}'...")
        subprocess.run(["docker", "build", "-t", DOCKER_IMAGE_NAME, "."], check=True)
        print("Docker image built successfully.")

def run_deepsp(input_file, output_file):
    """Runs the DeepSP container with correctly mounted paths."""
    input_path = os.path.abspath(input_file)
    output_path = os.path.abspath(output_file)
    mount_dir = os.path.dirname(input_path)

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{mount_dir}:{mount_dir}",
        "-w", mount_dir,
        DOCKER_IMAGE_NAME,
        "--input", input_path,
        "--output", output_path
    ]

    subprocess.run(cmd, check=True)

def main():
    if len(sys.argv) != 3:
        print("Usage: deepsp <input_file> <output_file>")
        sys.exit(1)

    build_docker_image()
    run_deepsp(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
