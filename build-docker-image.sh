#!/bin/bash

DOCKER_IMAGE_NAME=demulab/pointnet-test:latest

./stop-docker-container.sh
docker build ./docker -t $DOCKER_IMAGE_NAME #--no-cache=true
