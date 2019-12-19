#!/bin/bash

# This will run the dip-q image, while mounting the /my_results folder on the host's /tmp/results, meaning that any change to that folder
# will be reflected on the host's /tmp/results folder as well.

# The source of the mount is the path to the file or directory on the Docker daemon host.
# The target takes as its value the path where the file or directory is mounted in the container.

docker run -d \
  -it \
  --rm \
  --name playground \
  --mount type=bind,source="/tmp/results",target="/my_results" \
  playground:latest