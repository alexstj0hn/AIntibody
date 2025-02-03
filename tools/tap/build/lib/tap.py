import os
import sys
import subprocess

DOCKER_IMAGE_NAME = "tap"

def build_docker_image():
    """Checks if the Docker image exists, and builds it if necessary."""
    result = subprocess.run(["docker", "images", "-q", DOCKER_IMAGE_NAME], capture_output=True, text=True)
    if not result.stdout.strip():
        print(f"Building Docker image '{DOCKER_IMAGE_NAME}'...")  # Immediate print
        subprocess.run(["docker", "build", "-t", DOCKER_IMAGE_NAME, "."], check=True)
        print("Docker image built successfully.")  # Immediate print

def run_tap(input_file, output_file):
    """Runs the tap container with correctly mounted paths and forces real-time output."""
    input_path = os.path.abspath(input_file)
    output_path = os.path.abspath(output_file)
    mount_dir = os.path.dirname(input_path)

    cmd = [
        "docker", "run", "--rm", "-t", "-i",
        "-v", f"{mount_dir}:{mount_dir}",
        "-w", mount_dir,
        "--env", "PYTHONUNBUFFERED=1",  # <- Force unbuffered output inside container
        DOCKER_IMAGE_NAME,
        "--input", input_path,
        "--output", output_path
    ]

    # Run and force real-time output streaming
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True) as process:
        for line in iter(process.stdout.readline, ''):  # <- Use `readline()` to force immediate output
            print(line, end="", flush=True)  # <- Ensure immediate print

        process.wait()

        if process.returncode != 0:
            print(f"Error: Process exited with code {process.returncode}", file=sys.stderr)
            sys.exit(process.returncode)

def main():
    if len(sys.argv) != 3:
        print("Usage: tap <input_file> <output_file>")
        sys.exit(1)

    build_docker_image()
    run_tap(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
