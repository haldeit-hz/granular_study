# IMPORTANT NOTE: Update all "CHANGE HERE" or "CHANGE ACCORDINGLY HERE" values to match your system and file structure.

# MODULE IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob # For pattern-matching file names
from matplotlib.animation import FuncAnimation # Animation maker

# USER PARAMETERS (CHANGE HERE)
# Region of Interest (ROI)
ww, hh, xx, yy = 1390, 1030, 245, 35  # ROI dimensions and offsets # CHSNGE ACCORDINGLY HERE
# Binning
n_xbins = 8  # Number of bins along X-axis
# File pattern --> CHANGE HERE TO REFLECT OWN REPOSITORY
file_pattern = "Results135/OutputFiles/PsC_*.txt"

# SETUP
# Total ROI area (reference only)
TotalArea = ww * hh
# X-axis bin edges and centers
x_edges = np.linspace(0, ww, n_xbins + 1)
x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
# Get sorted file list
file_list = sorted(
    glob(file_pattern),
    key=lambda x: int(x.split('_')[1].split('.')[0])
)

print(f"Found {len(file_list)} files.") # THis one is just to make sure that the files found correspond to the number of frames one excpects

# FUNCTION: Process a single frame
def process_frame(df):
    """
    Process one frame:
    - Accurately split each particle's area across bins based on its X-span overlap.
    - Compute mean coordination number per bin (still based on particle center).
    """
    # Initialize arrays for this frame
    pf_bins = np.zeros(n_xbins) # Total summed particle area per bin
    coord_sum_bins = np.zeros(n_xbins) # Sum of coordination numbers per bin
    coord_count_bins = np.zeros(n_xbins) # Number of particles per bin (for averaging)
    # Loop through each particle in this frame
    for _, row in df.iterrows():
        # Extract particle position, area, and coordination number.
        x0 = row['X_0'] # X position of particle center
        area = row['Area'] # Particle area
        cn = row['Coordination_Number']# Particle coordination number
        # Approximate radius for X-axis projection.
        # Here: Major_Axis is used as diameter.
        radius = row['Major_Axis'] / 2.0
        # Compute the X-range spanned by this particle.
        particle_left = x0 - radius # Left edge of particle in X
        particle_right = x0 + radius # Right edge
        particle_width = 2 * radius # Total X-width (diameter)
        # Loop over bins to check for overlap
        for bin_idx in range(n_xbins):
            bin_left = x_edges[bin_idx] # Bin's left edge in X
            bin_right = x_edges[bin_idx + 1]# Bin's right edge
            # Determine overlap between particle span and bin.
            overlap_left = max(particle_left, bin_left)
            overlap_right = min(particle_right, bin_right)
            overlap_length = overlap_right - overlap_left
            # If there's overlap, assign fractional area to this bin.
            if overlap_length > 0:
                fraction = overlap_length / particle_width   # Fraction of area in this bin
                pf_bins[bin_idx] += area * fraction          # Add partial area to bin total
        # Coordination number: assign based on particle center.
        # This stays discrete — a particle belongs to one bin only.
        bin_idx = np.searchsorted(x_edges, x0) - 1 # Bin index for center point
        if 0 <= bin_idx < n_xbins:
            coord_sum_bins[bin_idx] += cn # Add to coordination sum
            coord_count_bins[bin_idx] += 1 # Increment count

    # Normalize packing fraction: divide by bin area
    bin_width = x_edges[1] - x_edges[0] # Width of each bin in X
    bin_area = bin_width * (hh - yy) # Bin area = bin width × ROI height
    pf_bins /= bin_area # Normalize summed areas by bin area
    # Compute mean coordination number per bin
    # Use NaN for empty bins to avoid divide by zero
    with np.errstate(invalid='ignore'):
        coord_mean_bins = np.where(
            coord_count_bins > 0, # Where bins have particles,
            coord_sum_bins / coord_count_bins, # Compute mean CN
            np.nan # Else, set as NaN
            )
    # Return both profiles for this frame
    return pf_bins, coord_mean_bins

# FUNCTION: Create & Save Animation

def animate_profiles(data, mean_profile, ylabel, filename, y_max=None):
    """
    Creates an animation for the given quantity across frames.
    Shows each frame's profile and the overall mean profile.
    """
    # Create a figure and axis.
    fig, ax = plt.subplots(figsize=(8, 5))
    # Create a Line2D object for the current frame's profile (will update).
    line, = ax.plot([], [], 'bo--', label='Profile')
    # Plot the mean profile as a static dashed red line.
    ax.plot(x_centers, mean_profile, 'r--', label='Mean')
    # Set X-axis limits to cover ROI.
    ax.set_xlim(0, ww)
    # Set Y-axis limits to a bit higher than max data value if not specified.
    ax.set_ylim(0, y_max if y_max else np.nanmax(data) * 1.2)
    # Axis labels and legend.
    ax.set_xlabel("x-position")
    ax.set_ylabel(ylabel)
    ax.legend()
    # Initialize function: clears the line for each animation cycle.
    def init():
        line.set_data([], [])
        return line,
    # Update function: updates data for each frame.
    def update(frame):
        line.set_data(x_centers, data[frame]) # Set X and Y data for current frame
        ax.set_title(f"Frame {frame+1}/{len(data)}") # Update plot title with frame number
        return line,
    # Create the animation object.
    ani = FuncAnimation(
        fig, update, frames=len(data), init_func=init, blit=True
    )
    # Save the animation to a video file.
    ani.save('Results135/Results/Cluster/' + filename, fps=25)   # CHSNGE ACCORDINGLY HERE
    print(f"Animation saved: {filename}")

# FUNCTION: Static Mean +/- Std Plot

def plot_static_profile(x, mean, std, ylabel,filename):
    """
    Plots the mean ± standard deviation profile for a given quantity.
    """
    # Create a new figure.
    plt.figure(figsize=(8, 5))
    # Plot the mean as points.
    plt.plot(x, mean, "o", label='Mean')
    # Shade the area between mean - std and mean + std.
    plt.fill_between(x, mean - std, mean + std, alpha=0.3, label='Std Dev')
    # Add axis labels and title.
    plt.xlabel("x-position")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig('Results135/Results/Cluster/' + filename,dpi=300) # CHSNGE ACCORDINGLY HERE

# MAIN LOOP (uses functions above)

pf_profiles = []     # List to hold packing fraction profiles for all frames
coord_profiles = []  # List to hold coordination profiles for all frames

# Loop through all input files (one file = one frame).
for fname in file_list:
    # Read data into a DataFrame, skipping comment lines.
    df = pd.read_csv(fname, comment='#', sep='\t', header=None)

    # Assign proper column names based on your file structure.
    df.columns = [
        'X_0', 'Y_0', 'Minor_Axis', 'Major_Axis', 'Angle', 'Area',
        'ID', 'Coordination_Number', 'Cluster_ID',
        'Clustering_Coeff', 'Cluster_Density', 'Betweenness_Centrality'
    ]

    # Process the frame: get packing fraction and coordination profiles.
    pf_bins, coord_bins = process_frame(df)

    # Store the results.
    pf_profiles.append(pf_bins)
    coord_profiles.append(coord_bins)

# Convert lists to numpy arrays for easier stats.
pf_profiles = np.array(pf_profiles)
coord_profiles = np.array(coord_profiles)

# Compute mean and standard deviation for each quantity.
pf_mean, pf_std = pf_profiles.mean(axis=0), pf_profiles.std(axis=0)
coord_mean = np.nanmean(coord_profiles, axis=0)
coord_std = np.nanstd(coord_profiles, axis=0)

# Plot and animate packing fraction.
plot_static_profile(x_centers, pf_mean, pf_std, "$\\phi$", f'MeanStdPackingFractionProfile_{n_xbins}bins.png')
animate_profiles(pf_profiles, pf_mean, "$\\phi$", f'packing_fraction_profile_{n_xbins}bins.mp4')

# Plot and animate coordination number.
plot_static_profile(x_centers, coord_mean, coord_std, "Mean Coordination Number",
                    f'MeanStdCoordinationNumberProfile_{n_xbins}bins.png')
animate_profiles(coord_profiles, coord_mean, "Mean Coordination Number", f'coordination_number_profile_{n_xbins}bins.mp4')
