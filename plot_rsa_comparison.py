import numpy as np
import matplotlib.pyplot as plt
import os

from region_specific_rdm_rsa import RESULTS_DIR
OUTPUT_PATH = RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_rsa_comparison():
    """
    Plots a side-by-side comparison of RSA values across brain regions and conditions.
    Edit the data below as needed for new experiments.
    """
    # RSA data by region and condition
    regions = ['TCL', 'FFCL', 'SFCL', 'f3', 'f4', 'f5']

    # RSA values from each condition (manually extracted from above)
    human_votc_1s = [0.04, -0.01, -0.02, 0.06, 0.02, 0.02]
    human_mtl_1s = [0.01, 0.00, -0.00, 0.00, 0.00, 0.01]
    monkey_rsa = [0.00, 0.02, 0.03, -0.01, -0.01, -0.01]
    human_votc_05s = [0.04, -0.00, -0.02, 0.06, 0.02, 0.02]
    human_mtl_05s = [0.01, 0.00, -0.00, 0.01, 0.01, 0.01]
    human_single_neuron = [0.03, 0.01, 0.00, 0.07, 0.05, 0.03]

    # Stack all data
    all_data = [
        human_votc_1s,
        human_mtl_1s,
        monkey_rsa,
        human_votc_05s,
        human_mtl_05s,
        human_single_neuron
    ]

    labels = [
        'Human RSA (HGP VOTC, 1s)',
        'Human RSA (HGP MTL, 1s)',
        'Monkey RSA',
        'Human RSA (HGP VOTC, 0.5s)',
        'Human RSA (HGP MTL, 0.5s)',
        'Human RSA (Single-Neuron)'
    ]

    colors = ['orange', 'orangered', 'crimson', 'deeppink', 'deepskyblue', 'turquoise']

    # Plot setup
    x = np.arange(len(regions))
    width = 0.13

    plt.figure(figsize=(15, 6))

    for i, data in enumerate(all_data):
        plt.bar(x + i * width, data, width, label=labels[i], color=colors[i])

    plt.axhline(0, color='gray', linestyle='--')
    plt.ylabel("RSA Spearman Correlation")
    plt.title("RSA Results Across Brain Regions and Conditions")

    # Set flipped x-tick labels
    plt.xticks(x + width * 2.5, regions, rotation=45)

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "rsa_comparison.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_rsa_comparison()
