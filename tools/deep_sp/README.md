Inside this directory:
1. Build docker container with `docker build -t deepsp-image .`
2. Run tool on test data with `docker run --rm -v "$(pwd)/../test:/app/data" deepsp-image --input /app/data/sequences.csv --output /app/data/deep_sp.csv`