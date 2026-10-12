from scipy.io import loadmat
import numpy as np
import pandas as pd

def convert_mat_to_csv(mat_file_path, csv_file_path):
    """
    Ports the true group data from a MATLAB .mat file to a CSV file. The function reads the .mat file, extracts the group data, and saves it in a structured CSV format.
    CSV Structure:
    - Each row corresponds to a galaxy.
    - The first column contains the group ID (1 to n, where n is the number of groups in the given volume/slice).
    - The next 3 columns contain the galaxy's position (RA, Dec, z).
    - The next 4 columns contain the galaxy's properties (apparent magnitude, stellar mass, absolute magnitude, halo mass).
    """
    # Load the MATLAB file
    data = loadmat(mat_file_path)  # Load the true groups from the .mat file
    Group = data["Group"]

    # Collect all galaxies and their group IDs
    all_galaxies = []

    for i in range(Group.shape[1]):

        # Extract the m×10 array from the i-th cell
        galaxies = Group[0, i]

        # Create a group ID column
        group_id = np.full((galaxies.shape[0], 1), i)

        # Append group ID and the 10 properties
        galaxies_with_id = np.hstack((group_id, galaxies[:, 1:4], galaxies[:, 6:]))  # Only include relevant columns

        all_galaxies.append(galaxies_with_id)

    # Combine all groups into a single array
    all_galaxies = np.vstack(all_galaxies)

    # Convert to a DataFrame
    df = pd.DataFrame(all_galaxies)

    # Save as CSV
    df.to_csv(csv_file_path, index=False)
    return


if __name__ == "__main__":
    for volume in range(1, 10):  # Volumes 1 to 9
        for slice_num in range(1, 4):  # Slices 1 to 3
            mat_file_path = f"Data/Original Groups/GT_Vol_{volume}_Slice_{slice_num}_5.mat"
            csv_file_path = f"Data/Original Groups/GT_Vol_{volume}_Slice_{slice_num}_5.csv"
            convert_mat_to_csv(mat_file_path, csv_file_path)