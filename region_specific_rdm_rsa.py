#!/usr/bin/env python3
import os
import numpy as np
import scipy.io
import h5py
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import argparse
import collections
from PIL import Image

# Import paths from the original script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EIG_FACES_DIR = os.path.join(BASE_DIR, 'EIG_faces')
DATA_DIR = os.path.join(BASE_DIR, 'osfstorage-archive', 'Data')
STIMULI_DIR = os.path.join(BASE_DIR, 'osfstorage-archive', 'stimuli')
FEATURES_DIR = os.path.join(BASE_DIR, 'osfstorage-archive', 'Features')
RESULTS_DIR = os.path.join(BASE_DIR, 'human-EIG-faces', 'results')
MODEL_DIR = os.path.join(EIG_FACES_DIR, 'models', 'checkpoints')

EIG_LATENTS_PATH = os.path.join(EIG_FACES_DIR, 'output', 'infer_output_eig.hdf5')
VGG_ACTS_PATH = os.path.join(EIG_FACES_DIR, 'output', 'infer_output_vgg.hdf5')
VGG_FEATURES_PATH = os.path.join(FEATURES_DIR, 'FM_vggface_vgg16_pool5_new.mat')
SORTED_FR_PATH = os.path.join(DATA_DIR, 'SortedFRCelebA_MTL.mat')
HGP_PATH = os.path.join(DATA_DIR, 'CelebA_HGP_all_1s.mat')
PATH_IDENTITY_LABELS = os.path.join(RESULTS_DIR, 'file_identity_map.npy')
PATH_NEURAL_IDENTITY_LABELS = os.path.join(RESULTS_DIR, 'neural_file_identity_map.npy')

os.makedirs(RESULTS_DIR, exist_ok=True)

# Reuse functions from the original script
def load_mat_file(filepath):
    if filepath.endswith('.mat'):
        try:
            mat = scipy.io.loadmat(filepath)
            return mat
        except NotImplementedError:
            mat = {}
            with h5py.File(filepath, 'r') as f:
                for key in f.keys():
                    mat[key] = np.array(f[key])
            return mat
    else:
        raise ValueError("Unsupported file format")

def compute_rdm(features):
    """Compute correlation distance RDM quickly."""
    dists = pdist(features, metric='correlation')
    return squareform(dists)

def compute_rdm_nan_safe(features):
    """Compute NaN-safe correlation distance RDM, moderately fast."""
    n = features.shape[0]
    rdm = np.zeros((n, n))

    # Precompute
    for i in range(n):
        for j in range(i, n):
            x = features[i]
            y = features[j]

            valid = ~np.isnan(x) & ~np.isnan(y)
            if valid.sum() > 1:  # Need at least 2 points to compute correlation
                corr = np.corrcoef(x[valid], y[valid])[0, 1]
                rdm[i, j] = 1 - corr
                rdm[j, i] = rdm[i, j]
            else:
                rdm[i, j] = np.nan
                rdm[j, i] = np.nan

    return rdm

def load_vgg_features():
    """Load precomputed VGG features for RSA."""
    print(f"Loading VGG features from {VGG_FEATURES_PATH}...")
    if not os.path.exists(VGG_FEATURES_PATH):
        raise FileNotFoundError(f"File not found at {VGG_FEATURES_PATH}")
    
    mat_data = scipy.io.loadmat(VGG_FEATURES_PATH)
    vgg_key = next((key for key, value in mat_data.items() if isinstance(value, np.ndarray) and not key.startswith('__')), None)
    if vgg_key is None:
        raise KeyError("Could not find VGG features in the .mat file")
    
    vgg_array = mat_data[vgg_key]  # <- extract numpy array, clean name
    print(f"Using '{vgg_key}' as VGG features key with shape {vgg_array.shape}")

    all_features = {'pool5': vgg_array.T}  # now clean

    print("VGG features loaded successfully.")
    return all_features

def load_vgg_acts():
    """Load precomputed VGG activations from HDF5."""
    if not os.path.exists(VGG_ACTS_PATH):
        raise FileNotFoundError(f"File not found at {VGG_ACTS_PATH}")
    
    with h5py.File(VGG_ACTS_PATH, 'r') as f:
        f3 = f['f3'][:]
        f4 = f['f4'][:]
        f5 = f['f5'][:]

    return {
        'TCL': f3,
        'FFCL': f4,
        'SFCL': f5
    }

def load_eig_latents():
    """Load precomputed EIG latents from HDF5."""
    if not os.path.exists(EIG_LATENTS_PATH):
        raise FileNotFoundError(f"File not found at {EIG_LATENTS_PATH}")
    
    with h5py.File(EIG_LATENTS_PATH, 'r') as f:
        tcl = f['TCL'][:]
        ffcl = f['FFCL'][:]
        latents = f['latents'][:]
    
    return {
        'f3': tcl,
        'f4': ffcl,
        'f5': latents
    }

def load_neural_features(sorted_fr_path, target_areas=None):
    """Load and filter neural firing rate features."""
    print(f"Loading neural data from {sorted_fr_path}...")
    mtl_data = load_mat_file(sorted_fr_path)

    sort_fr = mtl_data['SortFR']  # (2566, 500)
    sort_base = mtl_data['SortBase']  # (2566, 500)
    vKeep = mtl_data['vKeep'].flatten() - 1  # shift to 0-based indexing
    areaCell = [str(s[0]) for s in mtl_data['areaCell'][0]]

    # Good neurons only
    sort_fr_good = sort_fr[vKeep, :]
    areaCell_good = np.array([areaCell[i] for i in vKeep])
    
    # Filter by target areas if specified
    if target_areas:
        area_mask = np.array([
            any(target.lower() in area.lower() for target in target_areas)
            for area in areaCell_good
        ])
        
        sort_fr_target = sort_fr_good[area_mask, :]
        neural_features = sort_fr_target.T  # (500, n_selected_neurons)
        
        # count how many neurons per area
        area_counts = collections.Counter([areaCell[i] for i in np.where(area_mask)[0]])
        print("Neuron counts per brain area:")
        for area, count in area_counts.most_common():
            print(f"{area}: {count}")
    else:
        # Use all neurons if no target areas specified
        neural_features = sort_fr_good.T
        
        # count how many neurons per area
        area_counts = collections.Counter([areaCell[i] for i in vKeep])
        print("Neuron counts per brain area:")
        for area, count in area_counts.most_common():
            print(f"{area}: {count}")
    
    print(f"Loaded {neural_features.shape[1]} neurons across {neural_features.shape[0]} images.")

    return neural_features

def prepare_hgp_features(filepath, baseline_correct=False, brain_region_filter=None):
    """Prepare HGP features for RSA, filtering by brain regions.
    
    Args:
        filepath: Path to the HGP data file
        baseline_correct: Whether to baseline correct the HGP data
        brain_region_filter: List of area names to filter by, OR 'MTL', 'VOTC', or a custom list
    """
    # Define predefined region groups
    region_groups = {
        'MTL': ['AH', 'Amyg', 'PH', 'entorhinal', 'parahippocampal'],
        'VOTC': ['inferiortemporal', 'fusiform', 'lingual', 'lateraloccipital']
    }
    
    # Handle predefined region groups
    if isinstance(brain_region_filter, str) and brain_region_filter in region_groups:
        brain_region_filter = region_groups[brain_region_filter]
    
    print(f"Loading HGP data from {filepath}...")
    data = load_mat_file(filepath)

    # Extract main fields
    hgp = data['data']          # (4948, 500) High Gamma data
    hgp_base = data['dataBase'] # (4948, 500) Baseline
    vKeep = data['vKeep'].flatten() - 1  # 0-based Python indexing
    vLabel = [str(x[0]) for x in data['vLabel']]  # vLabel: area names per channel

    mask = np.zeros(len(vLabel), dtype=bool)
    mask[vKeep] = True

    # Optionally baseline-correct
    if baseline_correct:
        hgp = hgp - hgp_base

    # Filter by brain region if specified
    if isinstance(brain_region_filter, list) and brain_region_filter:
        # Use vLabel for filtering
        region_mask = np.array([
            any(region.lower() in label.lower() for region in brain_region_filter)
            for label in vLabel
        ])
        mask &= region_mask

        # Count how many channels per area
        area_counts = collections.Counter([vLabel[i] for i in np.where(mask)[0]])
        print("Channel counts per brain area:")
        for area, count in area_counts.most_common():
            print(f"{area}: {count}")
    
    # Build neural feature matrix: (images, channels)
    hgp_good = hgp[mask, :]
    vLabel_good = np.array(vLabel)[mask]
    neural_features = hgp_good.T  # (500, n_channels)

    print(f"Final neural feature matrix shape (HGP, {brain_region_filter}): {neural_features.shape}")

    return neural_features, vLabel_good

def flatten_rdm(rdm):
    """Flatten RDM into upper triangle vector."""
    triu_indices = np.triu_indices(rdm.shape[0], k=1)
    return rdm[triu_indices]

def run_rsa_and_plot(neural_rdm, model_features_dict, title, output_path='rsa_results.png'):
    """Run RSA between neural features and model feature sets, and save a bar plot."""
    results = {}
    
    neural_flat = flatten_rdm(neural_rdm)

    model_names = []
    correlations = []
    
    for model_name, model_features in model_features_dict.items():
        model_rdm = compute_rdm(model_features)
        model_flat = flatten_rdm(model_rdm)

        corr, pval = spearmanr(neural_flat, model_flat, nan_policy='omit')
        results[model_name] = (corr, pval)
        model_names.append(model_name)
        correlations.append(corr)
        print(f"{model_name}: Spearman r={corr:.4f}, p={pval:.4g}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, correlations, color='skyblue')
    plt.ylabel('RSA Spearman Correlation')
    plt.title(f'RSA Results: {title}')
    plt.ylim([-0.2, 1.0])
    plt.axhline(0, color='gray', linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Add correlation values
    for i, (corr, model_name) in enumerate(zip(correlations, model_names)):
        plt.text(i, corr + 0.02, f'{corr:.2f}', ha='center', va='bottom', fontsize=10)

    # Save figure
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    
    plt.close()
    
    return results

def plot_rdms_side_by_side(rdm_dict, output_path='rdm_panel.png', neural_region_name=None):
    """
    Plot multiple RDMs side-by-side with RSA values.
    
    Args:
        rdm_dict: dict of {name: RDM_matrix}
        output_path: path to save the figure
        neural_region_name: specific name of the neural region for clearer labeling
    """
    model_names = list(rdm_dict.keys())
    num_models = len(model_names)
    
    # Find neural RDM keys for RSA computation
    neural_keys = [key for key in model_names if 'Neural' in key]
    model_keys = [key for key in model_names if 'Neural' not in key]
    
    # Create figure
    fig, axes = plt.subplots(1, num_models, figsize=(4*num_models, 4), dpi=600)

    if num_models == 1:
        axes = [axes]  # Ensure axes is iterable
    
    # Compute RSA values between neural and model RDMs
    rsa_values = {}
    for neural_key in neural_keys:
        neural_rdm = rdm_dict[neural_key]
        neural_flat = flatten_rdm(neural_rdm)
        
        for model_key in model_keys:
            model_rdm = rdm_dict[model_key]
            model_flat = flatten_rdm(model_rdm)
            
            # Compute Spearman correlation
            corr, pval = spearmanr(neural_flat, model_flat, nan_policy='omit')
            
            if neural_key not in rsa_values:
                rsa_values[neural_key] = {}
            rsa_values[neural_key][model_key] = (corr, pval)
    
    # Plot each RDM
    for ax, model_name in zip(axes, model_names):
        rdm = rdm_dict[model_name]
        
        im = ax.imshow(rdm, cmap='gray', vmin=0, vmax=1)  # Grayscale, distances 0 to 1
        
        # Enhance title with region information for neural data
        title = model_name
        if 'Neural' in model_name and neural_region_name:
            if 'HGP' in model_name:
                title = f"Neural (HGP)\n{neural_region_name}"
            else:
                title = f"Neural (SN)\n{neural_region_name}"
        
        ax.set_title(title, fontsize=12)
        
        # Add RSA values as text below model RDMs
        if 'Neural' not in model_name:
            y_pos = 1.05
            for i, neural_key in enumerate(neural_keys):
                if neural_key in rsa_values and model_name in rsa_values[neural_key]:
                    corr, pval = rsa_values[neural_key][model_name]
                    sig_stars = ''
                    if pval < 0.001:
                        sig_stars = '***'
                    elif pval < 0.01:
                        sig_stars = '**'
                    elif pval < 0.05:
                        sig_stars = '*'
                    
                    # Extract short name for neural data
                    if 'HGP' in neural_key:
                        neural_short = 'HGP'
                    else:
                        neural_short = 'SN'
                    
                    ax.text(0.5, -0.1 - (i * 0.1), 
                            f"{neural_short}: r={corr:.3f}{sig_stars} p={pval:.5g}",
                            ha='center', va='top', transform=ax.transAxes, fontsize=8)
        
        ax.axis('off')  # Hide axes ticks
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=600)
    print(f"Saved RDM panel plot to {output_path}")
    
    return rsa_values

def get_available_hgp_regions():
    """
    Get a list of all available HGP regions from the data file.
    
    Returns:
        List of available region names and a dictionary mapping region names to their counts
    """
    data = load_mat_file(HGP_PATH)
    vLabel = [str(x[0]) for x in data['vLabel']]  # vLabel: area names per channel
    vKeep = data['vKeep'].flatten() - 1  # 0-based Python indexing
    
    # Get valid labels
    valid_labels = [vLabel[i] for i in vKeep]
    
    # Count occurrences of each region
    region_counts = collections.Counter(valid_labels)
    
    # Extract unique region names
    unique_regions = sorted(region_counts.keys())
    
    return unique_regions, region_counts

def load_and_prepare_data(hgp_regions=None, single_neuron_regions=None):
    """
    Load and prepare all necessary data for RDM and RSA analysis.
    
    Args:
        hgp_regions: List of HGP brain regions to filter by
        single_neuron_regions: List of single neuron brain regions to filter by
        
    Returns:
        Dictionary containing all prepared data
    """
    # Load model features
    vgg_features = load_vgg_features()
    vgg_acts = load_vgg_acts()
    eig_latents = load_eig_latents()
    
    # Create dictionaries to store results
    neural_rdm_sorted = None
    neural_rdm_hgp = None
    rdm_dict = {}
    
    # Process sorting indices (needed for both types of data)
    filename_to_identity = np.load(PATH_IDENTITY_LABELS, allow_pickle=True)
    stimuli_sorting_indices = np.argsort(filename_to_identity).tolist()
    
    neural_filename_to_identity = np.load(PATH_NEURAL_IDENTITY_LABELS, allow_pickle=True)
    sorting_indices = np.argsort(neural_filename_to_identity).tolist()
    
    # Create valid rows mask (all rows are valid by default)
    valid_rows = np.ones(500, dtype=bool)  # Assuming 500 images
    
    # Apply sorting
    stimuli_sorting_indices = np.array(stimuli_sorting_indices)[valid_rows]
    sorting_indices = np.array(sorting_indices)[valid_rows]
    
    # Load single-neuron data if requested
    if single_neuron_regions is not None:
        neural_features = load_neural_features(SORTED_FR_PATH, target_areas=single_neuron_regions)
        neural_features_sorted = neural_features[sorting_indices, :]
        neural_rdm_sorted = compute_rdm_nan_safe(neural_features_sorted)
        
        # Format the region name for display
        if isinstance(single_neuron_regions, list):
            if len(single_neuron_regions) == 1:
                sn_region_name = single_neuron_regions[0]
            else:
                sn_region_name = '+'.join(single_neuron_regions)
        else:
            sn_region_name = 'All'
            
        rdm_dict[f'Neural (SN, {sn_region_name})'] = neural_rdm_sorted
    
    # Load HGP data if requested
    if hgp_regions is not None:
        # Handle predefined region groups
        region_groups = {
            'MTL': ['AH', 'Amyg', 'PH', 'entorhinal', 'parahippocampal'],
            'VOTC': ['inferiortemporal', 'fusiform', 'lingual', 'lateraloccipital']
        }
        
        # Convert string region names to actual filter list
        if isinstance(hgp_regions, str):
            if hgp_regions in region_groups:
                hgp_region_filter = region_groups[hgp_regions]
                hgp_region_name = hgp_regions
            else:
                # Single custom region
                hgp_region_filter = [hgp_regions]
                hgp_region_name = hgp_regions
        elif isinstance(hgp_regions, list):
            # Check if it's a list of predefined groups
            expanded_regions = []
            for region in hgp_regions:
                if region in region_groups:
                    expanded_regions.extend(region_groups[region])
                else:
                    expanded_regions.append(region)
            hgp_region_filter = expanded_regions
            if len(hgp_regions) == 1:
                hgp_region_name = hgp_regions[0]
            else:
                hgp_region_name = '+'.join(hgp_regions)
        else:
            hgp_region_filter = None
            hgp_region_name = 'All'
        
        # Load and process HGP data
        neural_features_hgp, _ = prepare_hgp_features(HGP_PATH, brain_region_filter=hgp_region_filter)
        neural_features_hgp = neural_features_hgp[sorting_indices, :]
        neural_rdm_hgp = compute_rdm_nan_safe(neural_features_hgp)
        rdm_dict[f'Neural (HGP, {hgp_region_name})'] = neural_rdm_hgp
    
    # Reorder and compute model RDMs
    vgg_features_sorted = vgg_features['pool5'][stimuli_sorting_indices, :]
    eig_f3_sorted = eig_latents['f3'][stimuli_sorting_indices, :]
    eig_f4_sorted = eig_latents['f4'][stimuli_sorting_indices, :]
    eig_f5_sorted = eig_latents['f5'][stimuli_sorting_indices, :]
    vgg_tcl_sorted = vgg_acts['TCL'][stimuli_sorting_indices, :]
    vgg_ffcl_sorted = vgg_acts['FFCL'][stimuli_sorting_indices, :]
    vgg_sfcl_sorted = vgg_acts['SFCL'][stimuli_sorting_indices, :]
    
    # Compute model RDMs
    vgg_rdm_sorted = compute_rdm(vgg_features_sorted)
    eig_f3_rdm_sorted = compute_rdm(eig_f3_sorted)
    eig_f4_rdm_sorted = compute_rdm(eig_f4_sorted)
    eig_f5_rdm_sorted = compute_rdm(eig_f5_sorted)
    vgg_tcl_rdm_sorted = compute_rdm(vgg_tcl_sorted)
    vgg_ffcl_rdm_sorted = compute_rdm(vgg_ffcl_sorted)
    vgg_sfcl_rdm_sorted = compute_rdm(vgg_sfcl_sorted)
    
    # Add model RDMs to dictionary
    rdm_dict.update({
        'VGG Pool5': vgg_rdm_sorted,
        'EIG f3': eig_f3_rdm_sorted,
        'EIG f4': eig_f4_rdm_sorted,
        'EIG f5': eig_f5_rdm_sorted,
        'VGG TCL': vgg_tcl_rdm_sorted,
        'VGG FFCL': vgg_ffcl_rdm_sorted,
        'VGG SFCL': vgg_sfcl_rdm_sorted
    })
    
    # Merge models into one dictionary for RSA
    all_models = {}
    all_models.update(vgg_acts)
    all_models.update(eig_latents)
    all_models['pool5'] = vgg_features_sorted
    
    # Create a dictionary of all sorted model features
    model_features = {
        'pool5': vgg_features_sorted,
        'f3': eig_f3_sorted,
        'f4': eig_f4_sorted,
        'f5': eig_f5_sorted,
        'TCL': vgg_tcl_sorted,
        'FFCL': vgg_ffcl_sorted,
        'SFCL': vgg_sfcl_sorted
    }
    
    return {
        'neural_rdm_sorted': neural_rdm_sorted,
        'neural_rdm_hgp': neural_rdm_hgp,
        'model_features': model_features,
        'all_models': all_models,
        'rdm_dict': rdm_dict,
        'sorting_indices': sorting_indices,
        'stimuli_sorting_indices': stimuli_sorting_indices
    }

def main():
    parser = argparse.ArgumentParser(description='Generate RDM matrices and RSA values for specific brain regions')
    parser.add_argument('--hgp-regions', type=str, nargs='+', default=['MTL', 'VOTC'],
                      help='HGP brain regions to filter by (e.g., "MTL", "VOTC", or specific region names)')
    parser.add_argument('--single-neuron-regions', type=str, nargs='+', default=['la', 'ra', 'lah', 'rah'], 
                      help='Single neuron brain regions to filter by (default: la ra lah rah)')
    parser.add_argument('--output-prefix', type=str, default='region_specific', help='Prefix for output files')
    parser.add_argument('--separate', action='store_true', help='Analyze each region separately')
    parser.add_argument('--skip-hgp', action='store_true', help='Skip HGP analysis')
    parser.add_argument('--skip-sn', action='store_true', help='Skip Single-Neuron analysis')
    parser.add_argument('--list-hgp-regions', action='store_true', help='List all available HGP regions and exit')
    args = parser.parse_args()
    
    # If requested, list all available HGP regions and exit
    if args.list_hgp_regions:
        print("\nAvailable HGP regions:")
        regions, counts = get_available_hgp_regions()
        for region in regions:
            print(f"  {region}: {counts[region]} channels")
        
        print("\nPredefined region groups:")
        print("  MTL: AH, Amyg, PH, entorhinal, parahippocampal")
        print("  VOTC: inferiortemporal, fusiform, lingual, lateraloccipital")
        print("\nYou can specify any of these regions individually or in combination.")
        return
    
    # Define region groups for easy reference
    region_groups = {
        'MTL': ['AH', 'Amyg', 'PH', 'entorhinal', 'parahippocampal'],
        'VOTC': ['inferiortemporal', 'fusiform', 'lingual', 'lateraloccipital']
    }
    
    # Process single-neuron regions if not skipped
    if not args.skip_sn and args.single_neuron_regions:
        print("\n===== ANALYZING SINGLE-NEURON DATA =====\n")
        
        # Determine which single-neuron regions to process
        sn_regions_to_process = []
        if args.separate:
            # Process each region individually
            sn_regions_to_process = [[region] for region in args.single_neuron_regions]
        else:
            # Process all regions together
            sn_regions_to_process = [args.single_neuron_regions]
        
        # Process each single-neuron region or group
        for sn_regions in sn_regions_to_process:
            sn_region_name = '_'.join(sn_regions)
            print(f"\n=== Processing Single-Neuron regions: {sn_region_name} ===\n")
            
            # Load and prepare data for this specific single-neuron region
            # Pass None for HGP regions to avoid loading unnecessary data
            data = load_and_prepare_data(hgp_regions=None, single_neuron_regions=sn_regions)
            
            # Generate output paths
            rdm_output_path = f"{args.output_prefix}_rdm_panel_SN-{sn_region_name}.png"
            sn_rsa_output_path = f"{args.output_prefix}_rsa_SN-{sn_region_name}.png"
            
            # Plot RDMs side by side with RSA values included
            plot_rdms_side_by_side(
                data['rdm_dict'], 
                output_path=rdm_output_path, 
                neural_region_name=f"Single-Neuron: {sn_region_name}"
            )
            
            # Run RSA for single-neuron data
            sn_rsa_results = run_rsa_and_plot(
                data['neural_rdm_sorted'], 
                data['all_models'], 
                title=f"Single-Neuron RSA ({sn_region_name})", 
                output_path=sn_rsa_output_path
            )
    
    # Process HGP regions if not skipped
    if not args.skip_hgp and args.hgp_regions:
        print("\n===== ANALYZING HGP DATA =====\n")
        
        # Determine which HGP regions to process
        hgp_regions_to_process = []
        if args.separate:
            # Process each region or group individually
            for region in args.hgp_regions:
                if region in region_groups:
                    # This is a predefined group (MTL or VOTC)
                    hgp_regions_to_process.append((region, region_groups[region]))
                else:
                    # This is a custom region
                    hgp_regions_to_process.append((region, [region]))
        else:
            # Process all regions together
            if len(args.hgp_regions) == 1 and args.hgp_regions[0] in region_groups:
                # Single predefined group
                hgp_regions_to_process.append((args.hgp_regions[0], region_groups[args.hgp_regions[0]]))
            else:
                # Custom set of regions
                hgp_regions_to_process.append(('_'.join(args.hgp_regions), args.hgp_regions))
        
        # Process each HGP region or group
        for hgp_region_name, hgp_region_list in hgp_regions_to_process:
            print(f"\n=== Processing HGP regions: {hgp_region_name} ===\n")
            
            # Load and prepare data for this specific HGP region
            # Pass None for single-neuron regions to avoid loading unnecessary data
            data = load_and_prepare_data(hgp_regions=hgp_region_list, single_neuron_regions=None)
            
            # Generate output paths
            rdm_output_path = f"{args.output_prefix}_rdm_panel_HGP-{hgp_region_name}.png"
            hgp_rsa_output_path = f"{args.output_prefix}_rsa_HGP-{hgp_region_name}.png"
            
            # Plot RDMs side by side with RSA values included
            plot_rdms_side_by_side(
                data['rdm_dict'], 
                output_path=rdm_output_path, 
                neural_region_name=f"HGP: {hgp_region_name}"
            )
            
            # Run RSA for HGP data if available
            if data['neural_rdm_hgp'] is not None:
                hgp_rsa_results = run_rsa_and_plot(
                    data['neural_rdm_hgp'], 
                    data['all_models'], 
                    title=f"HGP RSA ({hgp_region_name})", 
                    output_path=hgp_rsa_output_path
                )
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()