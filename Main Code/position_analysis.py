import numpy as np
import matplotlib.pyplot as plt
import os
import glob

#--------------------------------------------- check overall sizes--------------------------------------------------
def check_size_change_between_frames(file_directory, frame1=197, frame2=227, size_bar=2000):
    '''
    Checks the change in particle size between two specific frames.
    
    Parameters:
        file_directory: path to directory containing .txt files with particle positions
        frame1: first frame number (default: 197)
        frame2: second frame number (default: 227)
        size_bar: minimal change in pixel size to be regarded as an active particle
    '''
    # Load all txt files from directory
    file_pattern = os.path.join(file_directory, "*.txt")
    files = sorted(glob.glob(file_pattern))
    
    if len(files) == 0:
        print(f"No .txt files found in {file_directory}")
        return
    
    if frame1 >= len(files) or frame2 >= len(files):
        print(f"Error: Only {len(files)} frames available")
        return
    
    # Load data for both frames
    data1 = np.loadtxt(files[frame1])
    data2 = np.loadtxt(files[frame2])
    
    # Handle single particle case
    if data1.ndim == 1:
        data1 = data1.reshape(1, -1)
    if data2.ndim == 1:
        data2 = data2.reshape(1, -1)
    
    # Sort by column 6 (7th column)
    data1 = data1[data1[:, 6].argsort()]
    data2 = data2[data2[:, 6].argsort()]
    
    # Extract sizes (column 5)
    sizes1 = data1[:, 5]
    sizes2 = data2[:, 5]
    
    # Print results
    print(f"\\n=== Size Change: Frame {frame1} → Frame {frame2} ===")
    print(f"{'Particle ID':<12} {'Frame {frame1} Size':<18} {'Frame {frame2} Size':<18} {'Change':<15}")
    print("-" * 65)
    
    num_particles = min(len(sizes1), len(sizes2))
    for pid in range(num_particles):
        size1 = sizes1[pid]
        size2 = sizes2[pid]
        change = size2 - size1

        if abs(change) > size_bar:
            is_active = 1
        else:
            is_active = 0
        
        size_changes[pid] = {
            'size_frame1': size1,
            'size_frame2': size2,
            'change': change,
            'is_active': is_active
        }
        print(f"{pid + 1:<12} {size1:<18.2f} {size2:<18.2f} {change:<15.2f} {is_active}")
    print("=" * 65)

#------------------------------------------------------- plots -------------------------------------------------------

def plot_particle_trajectories(file_directory, part_num=10, frame1=197, frame2=227, size_threshold=2000, 
                                start_frame=0, end_frame=None, frame_step=1):
    '''
    Analyzes particle trajectories and creates four plots:
    - Active particles (size change >= threshold): trajectories and displacement
    - Passive particles (size change < threshold): trajectories and displacement
    
    Parameters:
        file_directory: path to directory containing .txt files with particle positions
        part_num: number of particles to plot (default: 10)
        frame1: first frame to check size change (default: 197)
        frame2: second frame to check size change (default: 227)
        size_threshold: threshold for active vs passive (default: 2000 pixels)
        start_frame: first frame to include in trajectories (default: 0)
        end_frame: last frame to include in trajectories (default: None, uses all frames)
        frame_step: step between frames to include (default: 1, use every frame)
    '''
    # Load all txt files from directory
    file_pattern = os.path.join(file_directory, "*.txt")
    files = sorted(glob.glob(file_pattern))
    
    if len(files) == 0:
        print(f"No .txt files found in {file_directory}")
        return
    
    # Set end_frame if not specified
    if end_frame is None:
        end_frame = len(files) - 1
    
    # Store trajectories for each particle
    trajectories = {}
    frame_indices = []  # Store actual frame numbers used
    
    for frame_idx in range(start_frame, min(end_frame + 1, len(files)), frame_step):
        file = files[frame_idx]
        data = np.loadtxt(file)
        # Handle single particle case
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Sort by column 6 (7th column)
        data = data[data[:, 6].argsort()]
        
        # Extract only first two columns (x, y)
        xy_data = data[:, :2]
        
        for particle_id, (x, y) in enumerate(xy_data):
            if particle_id not in trajectories:
                trajectories[particle_id] = []
            trajectories[particle_id].append((x, y))
        
        frame_indices.append(frame_idx)
    
    # Convert to numpy arrays
    for pid in trajectories:
        trajectories[pid] = np.array(trajectories[pid])
    
    # Determine active vs passive particles based on size change
    active_particles = set()
    passive_particles = set()
    
    if frame1 < len(files) and frame2 < len(files):
        data1 = np.loadtxt(files[frame1])
        data2 = np.loadtxt(files[frame2])
        
        if data1.ndim == 1:
            data1 = data1.reshape(1, -1)
        if data2.ndim == 1:
            data2 = data2.reshape(1, -1)
        
        # Sort by column 6
        data1 = data1[data1[:, 6].argsort()]
        data2 = data2[data2[:, 6].argsort()]
        
        # Extract sizes (column 5)
        sizes1 = data1[:, 5]
        sizes2 = data2[:, 5]
        
        num_particles = min(len(sizes1), len(sizes2))
        for pid in range(num_particles):
            if pid >= part_num:
                break
            size_change = abs(sizes2[pid] - sizes1[pid])
            if size_change >= size_threshold:
                active_particles.add(pid)
            else:
                passive_particles.add(pid)
    
    # Create plots with 2x2 layout
    fig = plt.figure(figsize=(18, 10))
    
    # PLOT 1: Active Trajectories (top left)
    plt.subplot(2, 2, 1)
    for pid in active_particles:
        if pid in trajectories:
            traj = trajectories[pid]
            relative_traj = traj - traj[0]
            plt.plot(relative_traj[:, 0], relative_traj[:, 1], alpha=0.6, label=f'P{pid}')
    
    plt.xlabel('Δx (pixels)')
    plt.ylabel('Δy (pixels)')
    plt.title(f'Active Particle Trajectories (|ΔSize| ≥ {size_threshold} pixels)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # PLOT 2: Active Displacement (top right)
    plt.subplot(2, 2, 2)
    for pid in active_particles:
        if pid in trajectories:
            traj = trajectories[pid]
            displacements = np.sqrt((traj[:, 0] - traj[0, 0])**2 + 
                                    (traj[:, 1] - traj[0, 1])**2)
            
            # # Normalize by max displacement
            # max_displacement = np.max(displacements)
            # if max_displacement > 0:
            #     normalized_displacements = displacements / max_displacement
            # else:
            #     normalized_displacements = displacements
            
            plt.plot(frame_indices, displacements, alpha=0.6, label=f'P{pid}')
    
    plt.xlabel('Frame')
    plt.ylabel('Displacement (pixels)')
    plt.title('Active Particle Displacement')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # PLOT 3: Passive Trajectories (bottom left)
    plt.subplot(2, 2, 3)
    for pid in passive_particles:
        if pid in trajectories:
            traj = trajectories[pid]
            relative_traj = traj - traj[0]
            plt.plot(relative_traj[:, 0], relative_traj[:, 1], alpha=0.6, label=f'P{pid}')
    
    plt.xlabel('Δx (pixels)')
    plt.ylabel('Δy (pixels)')
    plt.title(f'Passive Particle Trajectories (|ΔSize| < {size_threshold} pixels)')
    plt.axis('equal')
    plt.grid(True)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # PLOT 4: Passive Displacement (bottom right)
    plt.subplot(2, 2, 4)
    for pid in passive_particles:
        if pid in trajectories:
            traj = trajectories[pid]
            displacements = np.sqrt((traj[:, 0] - traj[0, 0])**2 + 
                                    (traj[:, 1] - traj[0, 1])**2)
            
            # # Normalize by max displacement
            # max_displacement = np.max(displacements)
            # if max_displacement > 0:
            #     normalized_displacements = displacements / max_displacement
            # else:
            #     normalized_displacements = displacements
            
            plt.plot(frame_indices, displacements, alpha=0.6, label=f'P{pid}')
    
    plt.xlabel('Frame')
    plt.ylabel('Displacement (pixels)')
    plt.title('Passive Particle Displacement')
    plt.grid(True)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    # Save to same directory
    output_path = os.path.join(file_directory, f"particle_analysis_parts{part_num}_step{frame_step}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")
    print(f"Frames used: {start_frame} to {end_frame} (step: {frame_step})")
    print(f"Total frames plotted: {len(frame_indices)}")
    print(f"Active particles: {sorted(active_particles)}")
    print(f"Passive particles: {sorted(passive_particles)}")


plot_particle_trajectories("./results_0749-6/OutputFiles/", part_num=30, start_frame=197, end_frame=7079, frame_step=60)

size_changes = {}
# check_size_change_between_frames("./results_0749-6/OutputFiles/", 197, 227, 2000)
# plot_particle_trajectories("./results_0749-5/OutputFiles/", 53)
