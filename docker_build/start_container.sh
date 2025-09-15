docker container run --gpus 1 --rm -ti \
--volume /home/psmeili/IS-Validation-Framework/IS_Validate:/home/psmeili/IS_Validate \
--volume /data/psmeili/Validation_Framework_Datasets/datasets:/home/psmeili/external_mount/datasets \
--volume /data/psmeili/IS_Applications/SegVol_Validate_App/:/home/psmeili/external_mount/input_application/Sample_SegVol \
--volume /data/psmeili/Validation_Results/:/home/psmeili/external_mount/results \
--volume /home/psmeili/IS-Validation-bashscripts:/home/psmeili/validation_bashscripts \
--cpus 10 \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--ipc host \
--name segvolv1_test \
testing:segvolv1

