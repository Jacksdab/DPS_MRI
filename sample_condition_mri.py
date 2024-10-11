from functools import partial
import os
import argparse
import yaml
import json 
import pickle

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator, normalize_torch, _ifft2
from util.logger import get_logger
from datetime import datetime
# storing results
import json

# Direct
# direct imports
from direct.data.datasets import FastMRIDataset
# Import Subsampling schemes
from direct.common.subsample import FastMRIEquispacedMaskFunc
# Import transforms such as Fourier and inverse Fourier transforms, etc
from direct.data.transforms import fft2, ifft2, modulus, root_sum_of_squares
# Import the mri transforms pipeline which is used in direct for setting up experiments
from direct.data.mri_transforms import build_mri_transforms, ReconstructionType, ToTensor, CreateSamplingMask, Compose, EstimateSensitivityMap, ApplyMask, ComputeImage
# Import some classes needed for typing
from direct.types import KspaceKey, TransformKey
from direct.functionals import PSNRLoss, SSIMLoss

# MRI 
from experimentation import *

class FastMRI_knee_magnitude(Dataset):
    def __init__(self, folder_paths):
        self.folder_paths = folder_paths
        self.file_list = []
        for ind, folder_path in enumerate(folder_paths):
            self.file_list += [file for file in os.listdir(folder_path) if file.endswith('.pt') ]
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]

        img_prior1 = pickle.load(open(os.path.join(self.folder_paths[0], file_name), 'rb'))['img']
        magnitude = np.abs(img_prior1)/np.abs(img_prior1).max() #[0,1]
        phase = (np.angle(img_prior1)-np.angle(img_prior1).min())/(np.angle(img_prior1).max()-np.angle(img_prior1).min())
        normalized_data = np.exp(1j*phase)*magnitude

        kspace = fft2(torch.from_numpy(normalized_data))
        kspace = torch.stack([kspace,kspace,kspace], dim=0).to(torch.complex64)
        
        magnitude = magnitude #[-1,1]
        magnitude = magnitude.astype(np.float32)
        magnitude = np.stack([magnitude, magnitude, magnitude]).astype(np.float32)
        return magnitude, kspace

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
    # accelaration factor
    parser.add_argument('--acc_factor', type=int, default = 2)
    parser.add_argument('--scale_factor', type=int, default = 0.5)
    # saving gradients
    # parser.add_argument('--save_gradients', action=argparse.BooleanOptionalAction)
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
    # Allow for hyperparameter search
    cond_config['params']['scale'] = args.scale_factor
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn) 
    # Working directory
    # create subfolders for each run
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f'./output/{run_id}/'
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'progress_gradient', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),  # Converts to tensor (0,1) and scales it from 0-255 to 0-1
    # transforms.Normalize(mean=[0], std=[1])  # Normalization values for grayscale images 
    ])
    # dataset = get_dataset(**data_config, transforms=transform)
    # loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    transform = build_mri_transforms(
        forward_operator=fft2,
        backward_operator=ifft2,
        mask_func=FastMRIEquispacedMaskFunc(accelerations=[5], center_fractions=[0.1]),
        estimate_sensitivity_maps=True,
        scaling_key=KspaceKey.MASKED_KSPACE,  # This computes a scaling factor based on the masked k-space
        scale_percentile=0.99,  # Use the "max" 99th percentile value for the scaling factor
        crop="reconstruction_size"  # This assumes that `reconstruction_size` is in the sample
    )
    dataset_path = "/home/j.chu/code/DPS_MRI/data/test_folder"
    dataset = FastMRIDataset(data_root=dataset_path, transform=transform)
    loader = DataLoader(dataset, 
                        batch_size=1,  # Set the batch size to what fits your GPU memory
                        shuffle=False,   # Shuffle the data for training
                        num_workers=4)
    print(f"Dataset path: {data_config['root']}")
    psnr_all = []
    ssim_all = []   
    # Do Inference
    # path = "/home/j.chu/code/MRPD/data/example/val"
    # dataset = FastMRI_knee_magnitude(folder_paths=[path])
    # loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img['target'].unsqueeze(0)
        # ref_img =  _ifft2(ref_img['kspace'], dim=(-2,-1)).unsqueeze(0)
        # ref_img = ref_img[0][:, 0:1, :, :]
        _, _, h, w = ref_img.shape
        ref_img = ref_img.view(1, 1, h, w)
        ref_img = ref_img.to(device)
        print(f"Size of image is {ref_img.shape}")
        
        mean = ref_img.mean(dim=(2, 3), keepdim=True) 
        std = ref_img.std(dim=(2, 3), keepdim=True) 
        min = ref_img.min().item()
        max = ref_img.max().item()
        ref_img = (ref_img - min) / (max -min)
        # ref_img = ref_img * 2 - 1
        # The image is now normalized with mean 0 and std 1
        print(f"Image after normalization: Mean={ref_img.mean().item()}, Std={ref_img.std().item()}")
        print("Min value:", ref_img.min().item())
        print("Max value:", ref_img.max().item())

        # Exception) In case of inpainging,
        if measure_config['operator'] ['name'] == 'mri':
            mask = get_mask(torch.zeros([1, 1, 320, 320]), 320, 
                                1, acc_factor= args.acc_factor)
            mask = mask.to(device)
            print(f"ref_img device: {ref_img.device}")
            print(f"mask device: {mask.device}")
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)
            # Forward measurement model (Ax + n, k-space)
            y = operator.forward(ref_img, mask=mask)
            print(f"the shape of y is: {y.shape}")
            # y_n = noiser(y)

        else: 
            # Forward measurement model (Ax + n)
            # print('this got triggered')
            y = operator.forward(ref_img)
            y_n = noiser(y)
        
        # Sampling
        x_start = torch.randn(ref_img.shape[0],
                    2,
                    320,
                    320,
                    device=device).requires_grad_()
        sample, distances = sample_fn(x_start=x_start, measurement=y, record=True, save_root=out_path)
        
        fname = str(i).zfill(5)
        fname_recon = fname + f'_acc_factor_{args.acc_factor}' + f'_scale_{args.scale_factor}' + '.png'

        k_to_image = _ifft2(y, dim=(-2,-1))

        plt.imsave(os.path.join(out_path, 'input', fname + f'k_space.png'), np.abs(clear_color(torch.log(y+1e-9))), cmap='gray')
        plt.imsave(os.path.join(out_path, 'input', fname + f'inversed_measurements.png'), np.abs(clear_color(k_to_image)), cmap='gray')
        plt.imsave(os.path.join(out_path, 'label', f'{fname}.png'), clear_color(ref_img), cmap='gray')
        plt.imsave(os.path.join(out_path, 'recon', fname_recon), np.abs(clear_color(sample)), cmap='gray')
        print("Saved images")
        print('ref image shape:', ref_img.shape)
        print('sample shape:', sample.shape)
        # Evaluation metrics
        ssim = SSIMLoss()
        psnr = PSNRLoss()
        
        # lpips = LPIPS(net_type='vgg')
        # current_lpips = lpips(normalize_torch(sample).cpu(), normalize_torch(ref_img).cpu())
        device_cpu = torch.device('cpu')
        sample = torch.abs(sample).to(device_cpu)
        ref_img = ref_img.to(device_cpu)
        print("ref_img type: ", ref_img.type())
        print("sample type: ", sample.type())
        current_ssim = ssim(sample, ref_img, data_range = torch.Tensor(1))
        current_PSNRLoss = psnr(ref_img, sample)
        print('SSIM:', current_ssim)
        # print('LPIPS:', current_lpips)
        print('PSNR:', current_PSNRLoss)
        psnr_all.append(current_PSNRLoss.item())
        ssim_all.append(current_ssim.item())
        torch.cuda.empty_cache()
        

    print('Final Average PSNR:', np.mean(psnr_all))
    print('Final Average SSIM:', np.mean(ssim_all))

    # File path where the data will be saved
    file_path = os.path.join(out_path,f"metrics_data.json")

    # Create a dictionary to store all lists
    metrics_data = {
        "psnr_all": psnr_all,
        "ssim_all": ssim_all,
        "psnr_avg": np.mean(psnr_all),
        "ssim_avg": np.mean(ssim_all)
    }

    # Write to file using json
    with open(file_path, 'w') as f:
        json.dump(metrics_data, f)

    print(f"Metrics data saved to {file_path}")

if __name__ == '__main__':
    #print("Hello world")
    main()
