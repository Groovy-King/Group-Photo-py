import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
import argparse
import pickle
from scipy.stats import shapiro
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from Galaxy import Galaxy
from Group import Group
from True_group import TrueGroup
from constants import *
from loading import load_galaxies, load_true_groups
from candidate_groups import identify_groups
from util import *
from post_processing import proximity_merging
from tree_builder import build_tree

parser = argparse.ArgumentParser()

parser.add_argument("--volume", type = int, default = 1, help = "Volume index to process (1-9)")
parser.add_argument("--slice", type = int, default = 1, help = "Slice index to process (1-3)")
parser.add_argument("--save_output", type = bool, default = True, help = "Whether to save the output pickle files containing the extracted groups")

args = parser.parse_args()
volume = args.volume
slice = args.slice
save_output = args.save_output


# Check that volume and slice are valid integers within the expected range
if not isinstance(volume, int) or not isinstance(slice, int):
    raise TypeError("Volume and slice must be integers")
elif volume < 1 or volume > 9:
    raise ValueError("Volume must be between 1 and 9")
elif slice < 1 or slice > 3:
    raise ValueError("Slice must be between 1 and 3")

# Load galaxies and rebuild the bidirectional tree for galaxies and Hough groups
galaxies = load_galaxies(f"Data/Galaxies/Vol_{volume}_Slice_{slice}.csv", include_core_groups = False, include_associated_groups = True)
extracted_group_dicts = pickle.load(open(f"Output/Hough_Groups/Extracted_Groups_Vol_test_{volume}_Slice_{slice}.pkl", "rb"))  # Load the extracted groups from the pickle file
hough_groups = build_tree(galaxies, extracted_group_dicts, print_out=True)

# Load true groups from the Millennium simulation data
true_groups = load_true_groups("Data/Original Groups/GT_Vol_1_Slice_1_5.csv", galaxies)  # Load the true groups from the CSV file, using the galaxies list to look up the member Galaxy objects

# Setup the linking between true groups and hough groups based on the most massive galaxy (MMG) 
true_positives = 0
for true_group in true_groups:
    # Get the most massive galaxy (MMG) of the true group
    mmg = true_group.central_galaxy

    # Get the associated Hough groups for the MMG
    hough_associated_groups = mmg.associated_groups

    # Link the true group to the associated Hough groups based on the MMG
    TG_dict, EG_dict = true_group.mmg_linking(hough_associated_groups)

    # Store the linking information in the true group instance
    true_group.mmg_linking_hough = TG_dict
    primary_extracted_group = TG_dict.get("primary_extracted_group", None)
    if primary_extracted_group is not None:
        primary_extracted_group.mmg_linking = EG_dict  # Store the linking information in the primary extracted group instance

    if TG_dict.get("identification", None) == "True Positive":
        true_positives += 1

# Go through the list of hough groups, and assign the unlinked groups as 'False Positives'
for hough_group in hough_groups:
    if not hasattr(hough_group, "mmg_linking"):
        hough_group.mmg_linking = {"primary_true_group": None, "identification": "False Positive"}  # Mark the unlinked Hough group as a false positive

# Computing precision and recall metrics
precision = true_positives / len(hough_groups) 
recall = true_positives / len(true_groups) 

# Save the computed precision and recall metrics to a csv file
filename = Path("Output/Hough_Groups/Precision_Recall_Metrics.csv")
df = pd.read_csv(filename, index_col = "index")

# Find the index corresponding to the current volume and slice
i = df[(df["volume"] == volume) & (df["slice"] == slice)].index[0]

# Update the precision and recall values for the current volume and slice
df.at[i, "precision"] = precision
df.at[i, "recall"] = recall

# Save the updated DataFrame
df.to_csv(filename)
print(f"Precision and recall metrics for Volume {volume}, Slice {slice} have been updated.")
