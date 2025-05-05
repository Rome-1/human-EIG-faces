"""
plot_vgg_faces_panel.py
======================

This script visualizes the VGG representational dissimilarity matrix (RDM) with face images along the axes, and displays a grid of the first 25 faces. It automatically loads data from canonical locations as defined in region_specific_rdm_rsa.py. No manual path editing is required.

- Loads VGG RDM and faces from canonical .mat files
- Plots the RDM with faces along axes and saves the figure
- Displays the first 25 faces in a 5x5 grid

Run: python plot_vgg_faces_panel.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import canonical paths from region_specific_rdm_rsa.py
import os
import scipy.io
from region_specific_rdm_rsa import RESULTS_DIR, VGG_FEATURES_PATH, STIMULI_DIR, PATH_IDENTITY_LABELS

os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_PATH = RESULTS_DIR


# --- Helper functions ---
def load_vgg_rdm():
    print(f"Loading VGG features from {VGG_FEATURES_PATH} ...")
    if not os.path.exists(VGG_FEATURES_PATH):
        print(f"ERROR: VGG features file not found at {VGG_FEATURES_PATH}")
        sys.exit(1)
    mat_data = scipy.io.loadmat(VGG_FEATURES_PATH)
    vgg_key = next((key for key, value in mat_data.items() if isinstance(value, np.ndarray) and not key.startswith('__')), None)
    if vgg_key is None:
        print("ERROR: Could not find VGG features in the .mat file")
        sys.exit(1)
    vgg_array = mat_data[vgg_key]
    print(f"Using '{vgg_key}' as VGG features key with shape {vgg_array.shape}")
    # Compute RDM (correlation distance)
    from scipy.spatial.distance import pdist, squareform
    vgg_rdm_sorted = squareform(pdist(vgg_array.T, metric='correlation'))
    print(f"Computed VGG RDM with shape {vgg_rdm_sorted.shape}")
    return vgg_rdm_sorted

# def load_faces():
#     # For demonstration, load the first N face images from the canonical stimuli directory
#     stimuli_dir = os.path.join(BASE_DIR, 'osfstorage-archive', 'stimuli')
#     print(f"Loading faces from {stimuli_dir} ...")
#     if not os.path.exists(stimuli_dir):
#         print(f"ERROR: Stimuli directory not found at {stimuli_dir}")
#         sys.exit(1)
#     import glob
#     from PIL import Image
#     face_paths = sorted(glob.glob(os.path.join(stimuli_dir, '*.jpg')))[:500]  # Adjust N as needed
#     if len(face_paths) == 0:
#         print(f"ERROR: No face images found in {stimuli_dir}")
#         sys.exit(1)
#     faces = []
#     for path in face_paths:
#         img = Image.open(path)
#         faces.append(np.array(img))
#     print(f"Loaded {len(faces)} faces with shape {faces[0].shape} (showing first)")
#     return np.array(faces)

# def plot_rdm_with_faces_wide(rdm, faces, output_path='rdm_with_faces_wide.png', n_faces_to_show=50, face_scale=10):
#     """
#     Plot an RDM matrix with faces on the top and side axes. Only plots the first n_faces_to_show faces.
#     Args:
#         rdm: 2D numpy array (n_faces, n_faces)
#         faces: list or array of images (n_faces, H, W, [C])
#         output_path: where to save the figure
#         n_faces_to_show: number of faces to plot along each axis (default 50)
#         face_scale: how much larger to make the faces (default 10)
#     """
#     n_faces = min(n_faces_to_show, len(faces), rdm.shape[0])
#     faces = np.array(faces[:n_faces])
#     rdm = np.array(rdm[:n_faces, :n_faces])
#     # Each face will be scaled up by face_scale
#     fig_size = (n_faces * face_scale, n_faces * face_scale)
#     fig = plt.figure(figsize=fig_size, dpi=100)
#     gs = fig.add_gridspec(n_faces+1, n_faces+1, width_ratios=[face_scale]+[face_scale]*n_faces, height_ratios=[face_scale]+[face_scale]*n_faces, wspace=0.0, hspace=0.0)
#     ax_faces_x = [fig.add_subplot(gs[0, i+1]) for i in range(n_faces)]
#     ax_faces_y = [fig.add_subplot(gs[i+1, 0]) for i in range(n_faces)]
#     ax_rdm = fig.add_subplot(gs[1:, 1:])
#     for ax, face in zip(ax_faces_x, faces):
#         ax.imshow(face)
#         ax.axis('off')
#     for ax, face in zip(ax_faces_y, faces):
#         ax.imshow(face)
#         ax.axis('off')
#     im = ax_rdm.imshow(rdm, cmap='gray', vmin=0, vmax=1)
#     ax_rdm.axis('off')
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     plt.savefig(output_path, bbox_inches='tight')
#     print(f"Saved fixed RDM with faces to {output_path}")

# def display_25_faces(faces_sorted, output_path=None):
#     """
#     Display the first 25 faces in a 5x5 grid using matplotlib and save to disk.
#     faces_sorted should be a numpy array or list of shape (N, H, W, C) or (N, H, W)
#     """
#     faces = np.array(faces_sorted)
#     n = 25
#     if faces.shape[0] < n:
#         print(f"Warning: Only {faces.shape[0]} faces available, displaying all.")
#         n = faces.shape[0]
#     plt.figure(figsize=(10, 10))
#     for i in range(n):
#         plt.subplot(5, 5, i + 1)
#         face = faces[i]
#         if face.ndim == 2:
#             plt.imshow(face, cmap='gray')
#         else:
#             plt.imshow(face)
#         plt.axis('off')
#     plt.tight_layout()
#     if output_path is None:
#         output_path = os.path.join(BASE_DIR, "EIG_Humans", "results", "first_25_faces.png")
#     plt.savefig(output_path, bbox_inches='tight')
#     print(f"Saved first 25 faces grid to {output_path}")
#     plt.show()

# def plot_rdm_with_faces_wide(rdm, faces, output_path='rdm_with_faces_wide.png'):
#     n_faces = len(faces)
#     ratios = [1] + [1]*n_faces
#     fig = plt.figure(figsize=(12, 12), dpi=150)
#     gs = fig.add_gridspec(n_faces+1, n_faces+1, width_ratios=ratios, height_ratios=ratios, wspace=0.0, hspace=0.0)
#     ax_faces_x = [fig.add_subplot(gs[0, i+1]) for i in range(n_faces)]
#     ax_faces_y = [fig.add_subplot(gs[i+1, 0]) for i in range(n_faces)]
#     ax_rdm = fig.add_subplot(gs[1:, 1:])
#     for ax, face in zip(ax_faces_x, faces):
#         ax.imshow(face)
#         ax.axis('off')
#     for ax, face in zip(ax_faces_y, faces):
#         ax.imshow(face)
#         ax.axis('off')
#     im = ax_rdm.imshow(rdm, cmap='gray', vmin=0, vmax=1)
#     ax_rdm.axis('off')
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     plt.savefig(output_path, bbox_inches='tight')
#     print(f"Saved fixed RDM with faces to {output_path}")

# def display_25_faces(faces):
#     faces = np.array(faces)
#     n = min(25, faces.shape[0])
#     if faces.shape[0] < 25:
#         print(f"Warning: Only {faces.shape[0]} faces available, displaying all.")
#     plt.figure(figsize=(10, 10))
#     for i in range(n):
#         plt.subplot(5, 5, i + 1)
#         face = faces[i]
#         if face.ndim == 2:
#             plt.imshow(face, cmap='gray')
#         else:
#             plt.imshow(face)
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     print("\n===== VGG Faces Panel Plotter =====\n")
#     try:
#         vgg_rdm_sorted = load_vgg_rdm()
#         faces = load_faces()
#     except Exception as e:
#         print(f"ERROR: {e}")
#         sys.exit(1)
#     if vgg_rdm_sorted.shape[0] != vgg_rdm_sorted.shape[1]:
#         print(f"ERROR: VGG RDM must be square. Got shape {vgg_rdm_sorted.shape}")
#         sys.exit(1)
#     if len(faces) != vgg_rdm_sorted.shape[0]:
#         print(f"ERROR: Number of faces ({len(faces)}) does not match RDM size ({vgg_rdm_sorted.shape[0]})")
#         sys.exit(1)
#     # Sample every 10th face and corresponding RDM row/col
#     faces_sorted = faces[::10]
#     vgg_rdm_panel = vgg_rdm_sorted[::10, ::10]
#     print("\nPlotting RDM with faces (every 10th face, one per identity)...")
#     plot_rdm_with_faces_wide(vgg_rdm_panel, faces_sorted, OUTPUT_PATH)
#     print("\nDisplaying and saving first 25 faces...")
#     display_25_faces(faces_sorted)
#     print("\nDone.")

# def plot_vgg_faces_panel(vgg_rdm_sorted, faces_sorted, output_path='vgg_feats_with_faces.png'):
#     """
#     Plots the VGG RDM with faces. You must provide your own data arrays:
#     - vgg_rdm_sorted: 2D numpy array (RDM matrix, shape [N, N])
#     - faces_sorted: list or array of N face images (each [H, W, C] or [H, W])
#     """
#     plot_rdm_with_faces_wide(vgg_rdm_sorted, faces_sorted, output_path=output_path)
#     print(f"Panel plot saved to {output_path}")

# (function display_25_faces is defined above; remove this stray incomplete definition)




def plot_rdm_with_faces_wide(rdm, faces, output_path='rdm_with_faces_wide.png'):
    n_faces = len(faces)

    # --- Define width and height ratios ---
    ratios = [1] + [1]*n_faces  # 1 for faces (top and side), and 1 for RDM matrix columns/rows

    fig = plt.figure(figsize=(12, 12), dpi=150)
    gs = fig.add_gridspec(n_faces+1, n_faces+1, 
                          width_ratios=ratios, height_ratios=ratios,
                          wspace=0.0, hspace=0.0)

    # --- Create axes
    ax_faces_x = [fig.add_subplot(gs[0, i+1]) for i in range(n_faces)]  # Top faces
    ax_faces_y = [fig.add_subplot(gs[i+1, 0]) for i in range(n_faces)]  # Side faces
    ax_rdm = fig.add_subplot(gs[1:, 1:])  # Main RDM

    # --- Plot faces on top
    for ax, face in zip(ax_faces_x, faces):
        ax.imshow(face)
        ax.axis('off')

    # --- Plot faces on side
    for ax, face in zip(ax_faces_y, faces):
        ax.imshow(face)
        ax.axis('off')

    # --- Plot RDM
    im = ax_rdm.imshow(rdm, cmap='gray', vmin=0, vmax=1)
    ax_rdm.axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, bbox_inches='tight')
    # plt.show()
    print(f"Saved fixed RDM with faces to {output_path}")


filename_to_identity = np.load(PATH_IDENTITY_LABELS, allow_pickle=True)
stimuli_sorting_indices = np.argsort(filename_to_identity).tolist()
stimuli = np.array(sorted(os.listdir(STIMULI_DIR)))

# Load all faces
all_faces = []
plt.figure(figsize=(15, 2))
for index, fname in enumerate(stimuli[stimuli_sorting_indices]):
    img = Image.open(os.path.join(STIMULI_DIR, str(fname)))
    img = img.resize((50, 50))  # tiny thumbnails
    img = np.array(img) / 255.0  # normalize
    all_faces.append(img) # (500, H, W, 3) or (500, H, W) if grayscale

# drop all but every 10th image
faces_sorted = all_faces[::10]

# sample 25 images and put them in a grid and save as png
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(faces_sorted[i])
    plt.axis('off')
plt.savefig(os.path.join(OUTPUT_PATH, "25_faces.png"), dpi=300)
plt.close()

vgg_rdm_sorted = load_vgg_rdm()

plot_rdm_with_faces_wide(vgg_rdm_sorted, faces_sorted, os.path.join(OUTPUT_PATH, "vgg_feats_with_faces.png"))