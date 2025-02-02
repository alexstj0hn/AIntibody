# Use Ubuntu 22.04 as a base image instead of Miniconda (Debian Buster is outdated)
FROM ubuntu:22.04

# Install Miniconda manually
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Set conda environment variables
ENV PATH="/opt/conda/bin:$PATH"

# Create a working directory
WORKDIR /app

# Install Playwright dependencies
RUN apt-get update && apt-get install -y \
    libenchant-2-2 \
    libicu-dev \
    libjpeg-turbo8 \
    libvpx-dev \
    libevent-dev \
    fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

# Copy your conda environment file into the container
COPY environment.yml /app/environment.yml

# Create the environment inside the container
RUN conda env create -f /app/environment.yml

# Activate conda environment and install Playwright browsers
RUN conda run -n tap playwright install --with-deps

# Copy scripts
COPY scripts /app/scripts

# Set the default command to run the Python orchestrator
ENTRYPOINT ["conda", "run", "-n", "tap", "python", "/app/scripts/run.py"]
