#/bin/bash

python3 sample_condition_mri_test.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/mri_config.yaml;