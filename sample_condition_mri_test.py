from functools import partial
import os
import argparse
import yaml
import json 

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger

# MRI 
from experimentation import *

def ifft2(y, dim=(0, 1), centered=True):
    """
    Perform 2D inverse FFT with optional shifts on the under-sampled k-space.

    Args:
    y (torch.Tensor): Undersampled k-space (frequency domain).
    sampling_mask (torch.Tensor): Sampling mask to be applied.
    dim (tuple): Dimensions for the FFT operations.
    centered (bool): If True, apply FFT shifts.

    Returns:
    torch.Tensor: The transformed tensor after 2D inverse FFT with optional shifts.
    """
    if centered:
        x = torch.fft.ifftshift(y, dim=dim)
    else:
        x = y
    x = torch.fft.ifft2(x, dim=dim)
    if centered:
        x = torch.fft.fftshift(x, dim=dim)
    return x

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
   
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'progress_gradient', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # Converts to tensor (0,1) and scales it from 0-255 to 0-1
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization values for grayscale images 
    ])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    
        
    # Do Inference
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        print(f"Size of image is {ref_img.shape}")
        mean = ref_img.mean(dim=(2, 3))
        std = ref_img.std(dim=(2, 3))
    
        print("Mean of each channel:", mean)
        print("Standard deviation of each channel:", std)

        # Exception) In case of inpainging,
        if measure_config['operator'] ['name'] == 'mri':
            mask = get_mask(torch.zeros([1, 3, 256, 256]), 256, 
                                 1)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n, k-space)
            y = operator.forward(ref_img, mask=mask)
            # y_n = noiser(y)

        else: 
            # Forward measurement model (Ax + n)
            print('this got triggered')
            y = operator.forward(ref_img)
            y_n = noiser(y)
         
        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample, distances = sample_fn(x_start=x_start, measurement=y, record=True, save_root=out_path)

        fname = str(i).zfill(5) + '_acc_factor_2.png'
        k_to_image = ifft2(y, dim=(-2,-1))

        plt.imsave(os.path.join(out_path, 'input', 'k_space.png'), clear_color(torch.log(y+1e-9)))
        plt.imsave(os.path.join(out_path, 'input', 'inversed_image.png'), clear_color(k_to_image))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))

        # Saving distances
        # Saving distances to compare
        filepath = os.path.join(out_path, f"distances_correctnormalization_resize_gradientupdate.json")
        print(f"saved distances in {filepath}")
        with open(filepath, 'w') as file:
            json.dump(distances, file)

if __name__ == '__main__':
    #print("Hello world")
    main()
