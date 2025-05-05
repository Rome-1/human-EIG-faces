# EIG-Humans

The EIG-Humans repository contains code for running RSA between EIG latents, VGG features, and neural data for images. The goal is to investigate whether the compelling evidence by Yildirim et al. (2020) that macaques invert a generative model to process faces extends to humans using data reported by Cao et al. (2025).

## Setup

Clone this repo into a directory of your choice. I recommend using a fresh conda environment. You can also skip this step if you have EIG-Faces installed already, or follow their instructions to set up an environment with Conda or Singularity.

```bash
# Clone the repo and cd into the root directory
cd EIG_Humans

# Create and activate a new conda environment
conda create -n eig-humans python=3.6
conda activate eig-humans

# Install dependencies
conda install -c conda-forge matplotlib
conda install -c anaconda scipy
conda install pytorch torchvision -c pytorch
conda install -c anaconda configparser
conda install h5py
conda install -c anaconda pandas
conda install -c anaconda scikit-learn
```

We will also need EIG-Faces and osfstorage.

To get EIG-Faces, clone it as you did this github repo and put it at the same level as this repo.

To get osfstorage, download the complete https://osf.io/tcbs5/files/osfstorage and put it at the same level as this repo.

At this point, your directory should look like this:

```
EIG_Humans/
EIG_Faces/
osfstorage/
```

In EIG_Faces, download their pretrained weights by running (at the EIG_Faces root):

```
chmod +x download_network_weights.sh
./download_network_weights.sh
```

For more info, see the EIG-Faces `README.md`.

To modify paths, edit `region_specific_rdm_rsa.py`. All other files import their base paths from there. Specific filenames can be modified in their respective files.

## Reproducing the analysis

The analysis has a few steps. First, we need to run out image set through the EIG_Faces EIG model and get the latents. Then, we need to do the same for the VGG model. Then, we use the neural data from Cao et al. (2025) to generate the RDMs, run RSA, etc.

### EIG and VGG latents

To get the EIG latents requires a slightly different `infer.py` script than that in EIG_Faces. Copy of the `infer.py` script from EIG_Humans to EIG_Faces (`EIG_Faces/infer_render_using_eig/infer.py`). To get the activations, run the following:

```bash
# from EIG_Faces directory
python -m infer_render_using_eig.infer --imagefolder ../osfstorage-archive/stimuli --segment --model eig
python -m infer_render_using_eig.infer --imagefolder ../osfstorage-archive/stimuli --segment --model vgg
```

This is written to run on a GPU. The outputs are written to `EIG_Faces/output/infer_output_eig.hdf5` and `EIG_Faces/output/infer_output_vgg.hdf5`.

### Identity maps

To generate the identity maps, first we need to retrieve identities from the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Download it from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs?resourcekey=0-pEjrQoTrlbjZJO2UL8K_WQ). 

Then, use `EIG_Humans/create_identity_maps.py`. From `EIG_Humans`, run:

```bash
python create_identity_maps.py
```

###Figures and Plots

To generate the RDMs, use `EIG_Humans/region_specific_rdm_rsa.py`. From `EIG_Humans`, run any of the following commands listed below. None require GPUs and can run quickly on a CPU (usually under a minute). To switch between HGP 1 second and 0.5s, modify the HGP path in `region_specific_rdm_rsa.py` from `CelebA_HGP_all_1s.mat` to `CelebA_HGP_all_0.5s.mat`.

```bash
# Default: Process all regions together (MTL, VOTC for HGP and la, ra, lah, rah for single-neuron)
python region_specific_rdm_rsa.py

# List all available HGP regions
python region_specific_rdm_rsa.py --list-hgp-regions

# Run only Single-Neuron analysis
python region_specific_rdm_rsa.py --skip-hgp

# Run non-MTL/VOTC HGP regions with more than 100 neurons separately
python region_specific_rdm_rsa.py --hgp-regions middletemporal superiortemporal lateralorbitofrontal insula WhiteMatter precentral medialorbitofrontal superiorfrontal supramarginal --skip-sn --separate

# Run only HGP analysis
python region_specific_rdm_rsa.py --skip-sn

# Analyze each region separately
python region_specific_rdm_rsa.py --separate

# Specify custom HGP regions
python region_specific_rdm_rsa.py --hgp-regions AH Amyg --single-neuron-regions la ra --separate

# Combine predefined and custom HGP regions
python region_specific_rdm_rsa.py --hgp-regions MTL fusiform

# Process only specific regions
python region_specific_rdm_rsa.py --hgp-regions MTL --single-neuron-regions la ra

# Process each MTL, VOTC in HGP and single-neuron regions separately
python region_specific_rdm_rsa.py --hgp-regions AH Amyg PH entorhinal parahippocampal inferiortemporal fusiform lingual lateraloccipital --single-neuron-regions la lah ra rah lph rph rhr --separate
```

To plot RSA values, use `EIG_Humans/plot_rsa_comparison.py`. From `EIG_Humans`, run:

```bash
python plot_rsa_comparison.py
```


CelebA_Label.mat contains a spreadsheet where rows and labels are in the same order as the stimulus data labels in relative order.

## References

Cao, R., Zhang, J., Zheng, J., Wang, Y., Brunner, P., Willie, J.T., Wang, S. (2025). A neural computational framework for face processing in the human temporal lobe. Current Biology, Vol. 35, No. 8, pp. 1765–1778.e6. https://doi.org/10.1016/j.cub.2025.02.063

Yildirim, I., Belledonne, M., Freiwald, W., Tenenbaum J. (2020.) Efficient inverse graphics in biological face processing. Science Advances, Vol. 6, no. 10.

Liu, Z., Luo, P., Wang, X., Tang, X. (2015). Deep learning face attributes in the wild. Proceedings of the IEEE International Conference on Computer Vision (ICCV), December 2015.
