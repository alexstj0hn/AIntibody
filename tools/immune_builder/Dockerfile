# Use a conda-based image
FROM continuumio/miniconda3:4.10.3

# Create a working directory
WORKDIR /app

# Copy your conda environment file into the container
COPY environment.yml /app/environment.yml

# Create the environment inside the container
RUN conda env create -f /app/environment.yml

# (Optional) Switch the shell so subsequent commands are in that environment
SHELL ["conda", "run", "-n", "immune_builder", "/bin/bash", "-c"]

# Copy scripts
COPY scripts /app/scripts

# Set the default command to run the Python orchestrator
ENTRYPOINT ["conda", "run", "-n", "immune_builder", "python", "/app/scripts/run.py"]
