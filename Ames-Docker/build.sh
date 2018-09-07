#!/bin/bash

set -e

# Build the image and give it the name 'proj-name' and tag it 'latest'
docker build -t ames-regression:latest .

echo "Done building Docker image, it can be found under: ames-regression:latest"