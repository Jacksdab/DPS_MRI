import os
import h5py
import shutil

# Paths to your directories
validation_folder = "/data/groups/public/archive/fastmri/knee/multicoil_val"
test_folder = "/home/j.chu/code/DPS_MRI/test_folder"

# Create the test folder if it doesn't exist
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# Iterate over .h5 files in the validation folder
for file_name in os.listdir(validation_folder):
    if file_name.endswith(".h5"):
        file_path = os.path.join(validation_folder, file_name)

        # Open the .h5 file to inspect its structure
        with h5py.File(file_path, 'r') as h5_file:
            # Assuming the dataset has a 'kspace' or 'reconstruction_rss' key for MRI slices
            if 'kspace' in h5_file.keys():
                kspace_data = h5_file['kspace']
                num_slices = kspace_data.shape[0]  # Number of slices in the volume

                # Drop first 10 and last 5 slices, only keep the middle ones
                slice_indices = list(range(10, num_slices - 5))

                for idx in slice_indices:
                    # Create symlink for each selected slice in the test folder
                    slice_name = f"{file_name}_slice_{idx}.h5"
                    src = os.path.join(validation_folder, file_name)
                    dst = os.path.join(test_folder, slice_name)

                    # Create a symbolic link in the test folder pointing to the original slice
                    os.symlink(src, dst)
                    print(f"Created symlink for {file_name} slice {idx} in {test_folder}")