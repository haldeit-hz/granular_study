# IMPORTANT NOTE: Before running, ensure that you change wherever there is a "CHANGE ACCORDINGLY" to reflect where your files are and where the output files should reside

# Modules Import 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob # Pattern Matching Module through shell
import re # Regular Expression --> Find/Replace type of module
import os # Interaction with OS

# System Characteristics
num_parts = 42 # CHANGE ACCORDINGLY. This is the number of particles in the system
dpmm = 7 # CHANGE ACCORDINGLY. This is the number of dots per mm. Check the configurations of the videos taht you will be using

def rms_comp(filepath):
    '''
    This function goes through ID-specific files generated using the Extraction bash script and outputs
    the particle ID and its corresponding data frame with root mean-squared computed relative to its
    initial position.
    
    Arguments:
    filepath: Path to a file containing particle trajectory data. The expected format is: 'id_<particle_id>_trajectory.txt'
    
    Returns:
    particle_id
    df: Data Frame with additional columns for RMS and XRMS
    '''
    # Read the data --> I am using Pandas
    df = pd.read_csv(filepath, comment='#', sep='\t', header=None) # Typical output structure using the video analysis python script
    df.columns = ['X_0', 'Y_0', 'Minor_Axis', 'Major_Axis', 'Angle', 'Area',
                  'ID', 'Coordination_Number', 'Cluster_ID', 'Clustering_Coeff',
                  'Cluster_Density', 'Betweenness_Centrality', 'Frame_Number'] # Columns of the output files
    
    # Initial position as the reference
    x0, y0 = df.loc[0, 'X_0'], df.loc[0, 'Y_0']
    
    # Compute squared displacement and RMS
    df['r_squared'] = (df['X_0'] - x0)**2 + (df['Y_0'] - y0)**2
    df['RMS'] = np.sqrt(df['r_squared'])
    
    # Compute x-displacement only (XRMS)
    df['XRMS'] = np.sqrt((df['X_0'] - x0)**2)

    # Time Column to make things a bit clearer
    df['Time'] = df['Frame_Number']/25

    # Compute speed between consecutive frames using diff
    dx = df['X_0'].diff() 
    dy = df['Y_0'].diff()
    df['Speed'] = np.sqrt((dx)**2 + (dy)**2)  # Assuming delta t = 1
    
    # Extract ID from filename or set it to unknown if super unclear
    match = re.search(r'id_(\d+\.\d+)_trajectory\.txt', os.path.basename(filepath))
    particle_id = match.group(1) if match else 'unknown'
    
    return particle_id, df

# Search for all trajectory files
file_list = glob.glob("Results148/IDData/A/id_*_trajectory.txt") # CHANGE ACCORDINGLY

rms_data = {}  # Storing particle_id: DataFrame
avg_rms_values = {}  # Dict to store time-averaged RMS for each particle

# Files Processing
for file in file_list:
    particle_id, df = rms_comp(file)  # ID and Data Frame Extraction
    rms_data[particle_id] = df  # Storing particle ID specific data frame
    clean_pid = str(int(float(particle_id)))

    # Plot RMS vs Frame Number (for each individual particle)
    plt.figure(figsize=(8, 5))
    plt.plot(df['Time'], df['RMS']/dpmm, color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\left(x,y\right) (mm)$')
    plt.tight_layout()
    #plt.savefig('Results135/Results/IndParticles/Single/'+f'Particle_{clean_pid}_RMS_vs_time.png', dpi=300) # CHANGE ACCORDINGLY
    plt.close()

    # Plot XRMS vs Frame Number (for each individual particle)
    plt.figure(figsize=(8, 5))
    plt.plot(df['Time'], df['XRMS']/dpmm, color='green')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$x (mm)$')
    plt.tight_layout()
    #plt.savefig('Results135/Results/IndParticles/Single/'+f'Particle_{clean_pid}_XRMS_vs_time.png', dpi=300) # CHANGE ACCORDINGLY
    plt.close()
    # Plot speed vs Frame Number (for each individual particle)
    plt.figure(figsize=(8, 5))
    plt.plot(df['Time'], df['Velocity']/dpmm, color='purple')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (mm/s)')
    #plt.savefig('Results135/Results/IndParticles/Single/' + f'Particle_{clean_pid}_Speed_vs_time.png', dpi=300) # CHANGE ACCORDINGLY
    plt.close()
# Compute Mean & Standard Deviation

# Assuming all particles share the same Frame_Number range: Align by merging on 'Frame_Number'

all_rms = []
all_xrms = []

# Collect time series into lists
for pid, df in rms_data.items():
    all_rms.append(df[['Time', 'RMS']].set_index('Time'))
    all_xrms.append(df[['Time', 'XRMS']].set_index('Time'))

# Concatenate and align on Frame Number
rms_concat = pd.concat(all_rms, axis=1)
xrms_concat = pd.concat(all_xrms, axis=1)

# Columns may have duplicated names; rename
rms_concat.columns = list(rms_data.keys())
xrms_concat.columns = list(rms_data.keys())

# Compute mean and std across particles for each frame
rms_mean = rms_concat.mean(axis=1)/dpmm
rms_se = rms_concat.std(axis=1)/(np.sqrt(num_parts)*dpmm)
xrms_mean = xrms_concat.mean(axis=1)/dpmm
xrms_se = xrms_concat.std(axis=1)/(np.sqrt(num_parts)*dpmm)

# Plot Mean RMS vs Frame Number with SE
plt.figure(figsize=(10, 6))
plt.plot(rms_mean.index, rms_mean.iloc, 'bo',label='Mean RMSD')
plt.fill_between(rms_mean.index, 
                 rms_mean.iloc - rms_se.iloc,
                 rms_mean.iloc + rms_se.iloc,
                 color='blue', alpha=0.3, label='SE')
plt.xlabel('Time (s)')
plt.ylabel(r'$\left(x,y\right) (mm)$')
plt.legend()
plt.tight_layout()
plt.savefig('Results148/Results/IndParticles/Average/'+'AMean_RMS_vs_timeTrunc.png', dpi=300) # CHANGE ACCORDINGLY

# Plot Mean XRMS vs Frame Number with SE
plt.figure(figsize=(10, 6))
plt.plot(xrms_mean.index, xrms_mean.iloc, 'go', label='Mean XRMSD')
plt.fill_between(xrms_mean.index, 
                 xrms_mean.iloc - xrms_se.iloc,
                 xrms_mean.iloc + xrms_se.iloc,
                 color='green', alpha=0.3, label='SE')
plt.xlabel('Time (s)')
plt.ylabel(r'$x (mm)$')
plt.legend()
plt.tight_layout()
plt.savefig('Results148/Results/IndParticles/Average/'+'AMean_XRMS_vs_timeTrunc.png', dpi=300) # CHANGE ACCORDINGLY

# Collect speed for all particles
all_speed = []

for pid, df in rms_data.items():
    all_speed.append(df[['Time', 'Speed']].set_index('Time'))

speed_concat = pd.concat(all_speed, axis=1)
speed_concat.columns = list(rms_data.keys())

# Drop first frame since it has NaNs
speed_concat = speed_concat.dropna()

speed_mean = speed_concat.mean(axis=1)/dpmm
speed_std = speed_concat.std(axis=1)/(np.sqrt(num_parts)*dpmm)

# Plot Mean Speed vs Frame Number with SE
plt.figure(figsize=(10, 6))
plt.plot(speed_mean.index, speed_mean.iloc, color='purple', label='Mean Speed')
plt.fill_between(speed_mean.index,
                 speed_mean.iloc - speed_std.iloc,
                 speed_mean.iloc + speed_std.iloc,
                 color='purple', alpha=0.3, label='SE')
plt.xlabel('Time (s)')
plt.ylabel('Speed (mm/s)')
plt.legend()
#plt.savefig('Results135/Results/IndParticles/Average/'+'PMean_Velocity_vs_timeTrunc.png', dpi=300) # CHANGE ACCORDINGLY
# Combine truncated RMS mean & SE into a single DataFrame
rms_trunc = pd.DataFrame({
    'Time': rms_mean.index,
    'Mean_RMS': rms_mean.iloc.values,
    'SE_RMS': rms_se.iloc.values
})

# Save to CSV or TXT --> I mostly used this so that I can try to fit a pwer law to the initial jump
rms_trunc.to_csv('Results148/Results/IndParticles/Average/AMean_RMS_vs_time.txt', sep='\t' ,index=False)

