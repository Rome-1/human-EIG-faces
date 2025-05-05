import os
import numpy as np
import scipy.io

from region_specific_rdm_rsa import STIMULI_DIR, RESULTS_DIR, DATA_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)

# Set your paths
PATH_IMAGES = STIMULI_DIR # 'osfstorage-archive/stimuli' 
PATH_IDENTITY = 'osfstorage-archive/Data/identity_CelebA.txt'

# Load identity file into a dictionary
filename_to_identity = {}
with open(PATH_IDENTITY, 'r') as f:
    for line in f:
        fname, identity = line.strip().split()
        filename_to_identity[fname] = int(identity)

# Get image filenames
image_filenames = sorted([f for f in os.listdir(PATH_IMAGES) if f.endswith('.jpg')])

# Map each filename to its original identity label
original_identity_labels = np.array([filename_to_identity[fname] for fname in image_filenames])
unique_ids, new_identity_labels = np.unique(original_identity_labels, return_inverse=True)

# Save and inspect
print("Original identity labels (first 10):", original_identity_labels[:10])
print("Mapped identity labels (first 10):", new_identity_labels[:10])
print("Total unique identities in subset:", len(unique_ids))

np.save(os.path.join(RESULTS_DIR, 'file_identity_map.npy'), new_identity_labels)

image_code = scipy.io.loadmat(os.path.join(DATA_DIR, 'CelebA_Image_Code.mat'))
np.save(os.path.join(RESULTS_DIR, 'neural_file_identity_map.npy'), image_code['im_code'][0])