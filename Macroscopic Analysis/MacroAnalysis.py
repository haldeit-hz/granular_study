# IMPORTANT NOTE: Before running, ensure that you change wherever there is a "CHANGE ACCORDINGLY"

# Modules Import
import numpy as np
import matplotlib.pyplot as plt

# Load Data
# File containing time series data: columns are assumed as:
# 1st column: time steps (or frame indices)
# 2nd column: packing fraction mean
# 3rd column: mean coordination number
# 4th column: standard deviation or error for coordination number
filename = 'Results135/IDData/PackFract.txt' # CHSNGE ACCORDINGLY HERE
numparts = 70 # CHSNGE ACCORDINGLY HERE. This is the number of particles

# Load the entire file into a NumPy array.
# Assumes no header lines.
data = np.loadtxt(filename, comments='#')
# Extract Columns & Convert X-axis to Time
x = data[:, 0] / 25 # Convert frame index to time units (e.g., fps=25)
y = data[:, 1] # Packing fraction mean values
mean_phi = np.mean(y) # Mean of the packing fraction over the whole time range
z = data[:, 2] # Coordination number mean values
mean_z = np.mean(z) # Mean of the coordination number over the whole time range
a = data[:, 3] # Standard deviation or error for coordination number

# Plot 1: Packing Fraction Over Time
plt.figure(1, figsize=(12, 6))
plt.plot(x, y, marker='o', markersize=3, linewidth=1)
plt.axhline(y=mean_phi, linestyle='--', color='r', label=f'$\\left<\\phi\\right>$:{mean_phi:.3f}')
plt.xlabel('Time (s)')
plt.ylabel(r'$\phi$')
plt.legend()
plt.tight_layout()
plt.savefig('Results135/Results/Macro/PackFractMacro.png', dpi=300) # CHSNGE ACCORDINGLY HERE


# Plot 2: Coordination Number with Error Bars Over Time
plt.figure(2)
plt.figsize=(12, 6)
plt.errorbar(x,z,yerr=a / np.sqrt(numparts),fmt='o')
plt.axhline(y=mean_z, linestyle='--', color='r', label=f'$\\left<z\\right>$:{mean_z:.3f}')
plt.xlabel('Time (s)')
plt.ylabel('$z$')
plt.legend()
plt.tight_layout()
plt.savefig('Results135/Results/Macro/'+'ZMacro.png', dpi=300) # CHSNGE ACCORDINGLY HERE
