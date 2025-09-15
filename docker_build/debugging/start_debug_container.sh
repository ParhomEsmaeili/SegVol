docker container run --gpus all --rm -ti \
--volume /home/parhomesmaeili/IS-Validation-Framework:/workspace \
--volume /home/parhomesmaeili/local_docker_vscode-server:/home/parhomesmaeili/.vscode-server \
--cpus 16 \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--ipc host \
--name segvolv1_debug \
testing:segvolv1_debugging

