#/bin/bash

python3 sample_condition_mri.py \
    --model_config=configs/model_config_mri.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/mri_config.yaml\
    --scale_factor=1;