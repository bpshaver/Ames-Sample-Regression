#!/bin/bash

set -e

# Run interactively by default: -it
# Mount a volume from the data folder under the same directory as the
# Dockerfile. It will be accessible inside the Docker container as ./data
docker run -it -v $(pwd)/data:/data --rm ames-regression:latest