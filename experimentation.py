import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from utils_physics import *

device = (
torch.device("cuda")
if torch.cuda.is_available()
else torch.device("cpu")
)

def get_mask(img, size, batch_size, type='uniform1d', acc_factor=2, center_fraction=0.04, fix=False):
    mux_in = size ** 2
    if type.endswith('2d'):
        Nsamp = mux_in // acc_factor
    elif type.endswith('1d'):
        Nsamp = size // acc_factor
    if type == 'uniform1d':
        mask = torch.zeros_like(img)
        if fix:
            Nsamp_center = int(size * center_fraction)
            samples = np.random.choice(size, int(Nsamp - Nsamp_center))
            mask[..., samples] = 1
            # ACS region
            c_from = size // 2 - Nsamp_center // 2
            mask[..., c_from:c_from + Nsamp_center] = 1
        else:
            for i in range(batch_size):
                Nsamp_center = int(size * center_fraction)
                samples = np.random.choice(size, int(Nsamp - Nsamp_center))
                mask[i, :, :, samples] = 1
                # ACS region
                c_from = size // 2 - Nsamp_center // 2
                mask[i, :, :, c_from:c_from+Nsamp_center] = 1
    else:
        NotImplementedError(f'Mask type {type} is currently not supported.')

    return mask

class SinglecoilMRI_comp:
    def __init__(self, image_size, mask):
        self.image_size = image_size
        self.mask = mask

    def A(self, x):
        return fft2_m(x) * self.mask

    def A_dagger(self, x):
        return ifft2_m(x)

    def A_T(self, x):
        return self.A_dagger(x)

def clear(x):
  x = x.detach().cpu().squeeze().numpy()
  return x

def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.
    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform
    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))


def prepare_im(load_dir, image_size, device):
  ref_img = torch.from_numpy(plt.imread(load_dir)[:, :, :3]).to(device)
  ref_img = ref_img.permute(2, 0, 1)
  ref_img = ref_img.view(1, 3, image_size, image_size)
  ref_img = ref_img * 2 - 1
  return ref_img



# # Load the .npy file
fname = './data/samples/001.npy'
saveroot_mri = './data/samples/mri'
# saveroot = './data/samples/medical.png'
# saveroot_mask = './data/samples/medical_mask.png'
# img = torch.from_numpy(np.load(fname))
# ### Taking absolute, but this is now handled in function from torch ###
# # img = torch.abs(img)
# print(img.shape)
# # Mask 
# mask = get_mask(torch.zeros([1, 1, 320, 320]), 320, 
#                                 1)
# mask = mask.to(device)
# ### Mask saving ###
# # plt.imsave(saveroot_mask, clear(mask_img), cmap='gray')
# # # Save the data as an image
# # plt.imsave(saveroot, clear(img), cmap='gray')

# # Create forward 
# A_funcs = SinglecoilMRI_comp(320, mask)
# A = lambda z: A_funcs.A(z)
# AT = lambda z: A_funcs.A_T(z)
# Ap = lambda z: A_funcs.A_dagger(z)

# h, w = img.shape
# x_orig = img.view(1, 1, h, w)
# x_orig = x_orig.to(device)
# y = A(x_orig)
# Apy = Ap(y)
# ATy = AT(y)

# Apy_sv = clear(Apy)
# plt.imsave(saveroot_mri + "_input.png", np.abs(Apy_sv), cmap='gray')
# x_orig_sv = clear(x_orig)
# plt.imsave(saveroot_mri + "_label.png", np.abs(x_orig_sv), cmap='gray')
# import direct
# from direct.data.datasets import FastMRIDataset
# dataset_path = "/projects/0/prjs0756/data/"
# dataset_path = "/data/groups/aiforoncology/archive/reconstruction/reconstruction/fastmri/brain/multicoil_train"
# dataset = FastMRIDataset(data_root=dataset_path)

# random_idx = 100
# sample = dataset[random_idx]
# print("Sample keys: ", sample.keys())