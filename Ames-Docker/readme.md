# Template for Python-on-Docker Project

The Dockerfile creates an image built on a Miniconda image with Python 3, then installs dependencies with `conda` and a requirements.txt file. If you're transitioning from a conda environment on your local computer, you may `conda env export --file environment.yml` to create an alternative to a requirements.txt.

Docker run has `-it` flags which means, basically, that the container runs interactively. Combined with the `-i` flag in the Dockerfile ENDPOINT command, which runs the Python session interactively, this allows you to interact with the Python process inside your Docker container after your script has finished executing.

To build:
```
./build.sh
```

To run:
```
./start.sh
```