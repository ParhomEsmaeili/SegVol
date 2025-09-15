#!/bin/bash
docker_tag=testing:segvolv1_debugging
#docker push ${docker_tag}
docker build --no-cache . -f dockerfile_debugging \
 -t ${docker_tag} \
 --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --network=host
#docker push ${docker_tag}
