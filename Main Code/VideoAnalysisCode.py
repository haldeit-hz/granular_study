# Modules Import
from skimage.feature import peak_local_max # Local maxima detection
from skimage.segmentation import watershed # Segmetation/separation of overlapping objects
from matplotlib import pyplot as plt
from scipy import ndimage # Multidimensional image processing
from PIL import Image # Image processing Python Library
import numpy as np
import cv2 as cv # Computer vision image processing
import csv # Comma Separated Values (CSV)
import os # Operating System
from scipy.spatial import distance_matrix # Coordination Number Detection
import networkx as nx  # For cluster (connected component) detection
import community as community_louvain # Library to determine communities
from scipy.optimize import linear_sum_assignment

# Color Channel Definition
bgr = {'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (0, 0, 255),
    'aqua': (255, 255, 0),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255)}

##### FUNCTIONS TO DETECT PARTICLES' CONTOUR AND FIT TO ELLIPSE #####
# Contours (nx1): list of individual particle contours --> JL Addition: Determine # of detected particles
def get_contours(binary, min_dist=25, min_area=2500): # Ihad to play with the minimum area value. 2500 seems to work fine
    '''
    This function cleans the binary image using morphological operations, detects
    particle centers via distance transform and peak detection, separates overlapping
    particles using the watershed algorithm and extracts contours of particles above a
    minimum area (adjustable depending on one's' needs) threshold.
    Arguments:
    I. binary: Binary image with foreground pixels (particles) with a value of 255 (white) and background pixels with a value of 0 (Black)
    II. min_dist: Minimum distance allowed between detected local maxima (potential particle centers)
    III. min_area: Minimum area (in pixels) for a detected contour to be considered a valid particle contour. Contours smaller than this threshold are discarded as noise
    Returns:
    contours: List containing the detected and filtered contours. Each contour is a NumPy array of points representing the boundary of a segmented particle
    '''
    # Remove small noise and clean up the binary image using morphological reseambklkace before starting to detect contours --> Check https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))  # Define a 3x3 elliptical kernel --> Important for determining which nature of operation to perform (typically keeping the objects of the form specified)
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2) # Remove speckle noise
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=1)  # Perform opening --> Useful for removing small objects and smoothing objects

    # Compute the Euclidean distance transform (distance from background)
    D = ndimage.distance_transform_edt(binary)  # White blobs --> distance to nearest background pixel
    D = cv.GaussianBlur(D, (3, 3), 0)  # Slightly smooth the distance map to reduce noise

    # Detect local maxima in the distance map as potential particle centers
    localMax = peak_local_max(D, min_distance=min_dist, labels=binary, footprint=np.ones((25, 25)))  # Find peaks within a 25x25 neighborhood (This is adjustable depending on sizes) at least `min_dist` apart
    # Create a mask and label the local maxima as markers
    mask = np.zeros(binary.shape, dtype=bool)  # Initialize an empty mask
    mask[tuple(localMax.T)] = True  # Mark local maxima as True in the mask
    markers, _ = ndimage.label(mask, structure=np.ones((3, 3)))  # Label connected peaks

    # Apply watershed algorithm using negative distance map and markers to segment overlapping particles
    labels = watershed(-D, markers, mask=binary)  # Label each particle uniquely

    contours = []  # Initialize list for storing final contours
    for label in np.unique(labels):  # Loop over all unique labels
        if label == 0:  # Label 0 corresponds to the background
            continue

        # Create a binary mask for the current label/particle
        temp = np.zeros(binary.shape, dtype="uint8")
        temp[labels == label] = 255

        # Find contours in the mask of this single particle
        cnts, _ = cv.findContours(temp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # Quick note: cv.RETR_EXTERNAL is perfect for detecting outermost contours, check other options for other purposes

        # Filtering out small contours based on area --> No need to keep most likely nosey contours
        for cnt in cnts:
            if cv.contourArea(cnt) >= min_area:
                contours.append(cnt)  # Append contour if it exceeds the minimum area

    return contours  # Return all valid particle contours

# particle list(mx5): x, y, minor axis, major axis, angle, outline(nx2): coordinates of all particle outlines. use for finding wall
def get_particle_list(binary,min_dist=25):
  '''
  This function calls upon get_contours and then fits the contours to ellipses.
  It also counts the number of particles.
  Arguments:
  I. binary: Array representing a binary image
  II. min_dist: Minimum distance for peak detection, which is passed to the get_contours function
  Returns:
  I. ellipse_list: List with each element a list of 5 floating-point numbers representing the parameters of a fitted ellipse for a single particle: [center_x, center_y, axis1_length, axis2_length, angle]
  II. outline: Array of shape (N, 2), where N is the total number of points in all the detected contours. It contains the concatenated coordinate points of all the valid particle outlines
  III. contours: List containing the detected and filtered contours, sorted based on the top-left corner of their bounding box.
  '''
  contours=get_contours(binary) # Gets contours from get_contours
  #contours = sorted(contours, key=lambda x: (x[0], x[1]))
  contours = sorted(contours, key=lambda ctr: (cv.boundingRect(ctr)[1], cv.boundingRect(ctr)[0]))
  outline=np.array([[0,0],[0,0]]) # Initialization of ellipse outline
  ellipse_list=[] # Intiializiation of ellipse list
  particle_count = 0  # Initialize a counter for particles
  for cnt in contours:  # Looping over the contours
    particle_count += 1 # Increment the counter for each particle
    outline=np.vstack((outline,cnt.reshape(-1,2))) # Vertically stacks the points of the current contour onto the outline array.
    ellipse=cv.fitEllipse(cnt) # Ellipse fitting
    ellipse_list.append([ellipse[0][0],ellipse[0][1],ellipse[1][0], ellipse[1][1],ellipse[2]]) # Appends the extracted ellipse parameters (center coordinates, axis lengths and angle) to the ellipse_list
  outline=outline[2:,:] # Removes original placeholders
  return ellipse_list,outline,contours


# --- Labeling ------------------------------------------
def label_image(img, frame_no, ellipses=None, contours=None, walls=None, xx=0, yy=0, ww=0, hh=0):
    '''This function annotates particles and walls on image.
    Arguments:
    I. img: Array representing the image on which annotations will be drawn
    II. frame_no: Integer representing the current frame number drawn on the image
    III. ellipses: Optional list of ellipse parameters (typically obtained from get_particle_list), where each element contains [center_x, center_y, axis1_length, axis2_length, angle]
    IV. contours: Optional list of particle contours (typically obtained from get_particle_list)
    V. walls: Optional list of (x, y) coordinate pairs representing the vertices of a wall or boundary. The function draws lines connecting these points to form a polygon.
    VI. xx, yy, ww, hh: Optional integers (default is 0). These likely represent the top-left x and y coordinates (xx, yy), width (ww) and height (hh) of a Region of Interest (ROI)
    Returns:
    img: The modified image with the added annotations.
    '''
    h, w = img.shape[:2] # Height and width + First two elements to determine dimensions
    # The line below uses the cv.putText function from the OpenCV library (cv) to draw the frame number on the image.
    cv.putText(img, f'{frame_no:04d}', (10, h - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, bgr['white'], 2)
    label = 1 # Initialization line

    if ellipses:
        for e in ellipses:
            x, y, maj, minr, ang = map(int, e) # Unpacking ellipse "e" specs
            cv.ellipse(img, ((x, y), (maj, minr), ang), bgr['red'], 2) # Drawing the ellipse on the image in RED
            cv.putText(img, str(label), (x, y - 15), cv.FONT_HERSHEY_SIMPLEX, 0.6, bgr['black'], 2) # Labeling the particles
            label += 1

    if contours:
        cv.drawContours(img, contours, -1, bgr['blue'], 1) # Draw particles contours in BLUE

    if walls:
        # walls = [ (x1,y1), (x2,y2), ... ] in order
        for i in range(len(walls)):
            p1 = walls[i] # Point i of the wall list
            p2 = walls[(i+1) % len(walls)] # Next point modulo length of the walls
            cv.line(img, tuple(p1), tuple(p2), bgr['yellow'], 2) # Draw line between the wall points in YELLOW
        # Draw the Region of Interest (ROI) rectangle in black
    #if ww > 0 and hh > 0:
        #cv.rectangle(img, (xx, yy), (xx + ww, yy + hh), bgr['black'], 2)

    return img

def match_particles(prev_particles, curr_particles, prev_ids, next_id, max_dist=200):
    """
    This function tracks particles by matching current frame ellipses to previous ones 
    based on the globally optimal centroid distance matching (Hungarian Algorithm) to ensure 
    that each particle keeps a consistent ID across frames. 

    It solves the assignment problem by minimizing the total distance between particles 
    in consecutive frames. If a particle does not find a match within the maximum distance, 
    it is assigned a new ID.

    Arguments:
    I. prev_particles: List of ellipses from the immediate previous frame. 
       Each element is [center_x, center_y, axis1, axis2, angle].
    II. curr_particles: List of ellipses from the current frame.
    III. prev_ids: List of unique IDs assigned to particles in the previous frame.
    IV. next_id: Integer representing the next available unique ID for any newly detected particle.
    V. max_dist: Maximum allowed distance (in pixels) for a particle to be considered 
       the same between two consecutive frames.

    Returns:
    I. curr_ids: List of IDs assigned to the current frame's particles.
    II. next_id: Updated next available unique ID to assign to any new particles in future frames.
    """

    # Initialize all IDs to -1 (unassigned)
    curr_ids = [-1] * len(curr_particles)

    # If there are no particles in the previous frame (first frame),
    # assign new unique IDs to all particles in the current frame.
    if not prev_particles:
        curr_ids = list(range(next_id, next_id + len(curr_particles)))
        return curr_ids, next_id + len(curr_particles)

    # Extract just the x,y coordinates for distance calculations
    prev_coords = np.array([p[:2] for p in prev_particles])  # Shape: (N_prev, 2)
    curr_coords = np.array([p[:2] for p in curr_particles])  # Shape: (N_curr, 2)

    # Compute the full distance matrix: each element (i,j) is the distance between
    # previous particle i and current particle j.
    D = np.linalg.norm(prev_coords[:, None, :] - curr_coords[None, :, :], axis=-1)

    # Solve the optimal assignment using the Hungarian Algorithm.
    # row_ind gives indices of previous particles,
    # col_ind gives indices of assigned current particles.
    row_ind, col_ind = linear_sum_assignment(D)

    # Keep track of which current and previous particles are matched
    assigned_curr = set()
    assigned_prev = set()

    # For each matched pair (i,j):
    for i, j in zip(row_ind, col_ind):
        # Only accept the match if the distance is less than the allowed maximum
        if D[i, j] < max_dist:
            curr_ids[j] = prev_ids[i]  # Assign the ID from the previous particle
            assigned_curr.add(j)       # Mark this current particle as matched
            assigned_prev.add(i)       # Mark this previous particle as matched

    # For any unmatched current particles, assign new unique IDs
    for k in range(len(curr_particles)):
        if k not in assigned_curr:
            curr_ids[k] = next_id
            next_id += 1  # Increment the next available unique ID

    return curr_ids, next_id


#def analyze_movies(video_path, output_folder, xx, yy, ww, hh, in_range1, in_range2 ,walls):
#def analyze_movies(video_path, output_folder, xx, yy, ww, hh, in_range1, in_range2, in_range3 ,walls):
def analyze_movies(video_path, output_folder, xx, yy, ww, hh, in_range1, in_range2, in_range3, in_range4a, in_range4b, walls):
    '''
    This function processes a given video to detect and label particles in each frame within a specified
    region of interest. It detects particles in two different color spaces (RGB and HSV), fits ellipses
    to them, saves coordinates and labeled images and optionally handles wall/box information.
    Arguments:
    I. video_path: File path to the video to be analyzed
    II. output_folder: Path to the directory where output data and images will be saved
    III. xx, yy, ww, hh: Integers defining the Region of Interest (ROI) within each video frame. (xx, yy) is the top-left corner coordinate, ww is the width and hh is the height of the ROI. The analysis will only be performed on this region.
    IV. in_range1: A list containing two NumPy arrays or lists, representing the lower and upper bounds for the first color thresholding operation
    V. in_range2: A tuple or list containing two NumPy arrays or lists, representing the lower and upper bounds for the second color thresholding operation
    VI. walls: A list of (x, y) coordinate pairs representing the vertices of the walls
    '''
    
    # Open the video file for reading
    cap = cv.VideoCapture(video_path)
    curr_frame = 1  # Frame counter initialization

    # Try to read the first frame to get video dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error")  # Exit if video can't be read
        return

    height, width = frame.shape[:2]  # Get height and width of the video frame

    # Create output folders if they don’t already exist
    os.makedirs(f"{output_folder}LabeledFrames", exist_ok=True)   # To save labeled frames with ellipses and IDs
    os.makedirs(f"{output_folder}BinaryImages", exist_ok=True)    # To save binary threshold masks
    os.makedirs(f"{output_folder}OutputFiles", exist_ok=True)    # To save text files with particle coordinates
    os.makedirs(f"{output_folder}IDData", exist_ok=True)    # To save ID specific data
    os.makedirs(f"{output_folder}ContactNetworks", exist_ok=True) # Contact Networks Outputs
    # To store per frame macroscopic data
    with open(f"{output_folder}SystemSummary.txt", 'w') as f:
        f.write("Frame\tNumParticles\tNumClusters\tMaxClusterSize\tGlobalEfficiency\tModularity\n")

    prev_particles = []  # List to store particle ellipses from previous frame
    prev_ids = []        # List to store particle IDs from previous frame
    next_id = 1          # Next available unique particle ID

    while True:
        print(f"{curr_frame}")  # Display current frame number for tracking
        ret, frame = cap.read()  # Read the next frame from the video
        if frame is None:        # Break loop if no frame is returned (end of video)
            break

        # Crop the frame to the region of interest (ROI)
        frame = frame[yy:yy + hh, xx:xx + ww]

        # Convert cropped frame to two different color spaces
        hsv1 = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Used for detecting white/transparent particles
        hsv2 = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # Used for detecting yellow particles
        hsv3 = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # Used for detecting green particles or any other color type --> CHANGE ACCORDINGLY
        hsv4 = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # Used for detecting green particles or any other color type --> CHANGE ACCORDINGLY
  

        # Create an exclusion mask for light blue hues (out-of-focus tubing)
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        blue_mask = cv.inRange(hsv_frame, np.array([90, 20, 100]), np.array([130, 255, 255]))  # light/dull blues

        # Apply color thresholding to isolate particles within specified color ranges
        binary1 = cv.inRange(hsv1, np.array(in_range1[0]), np.array(in_range1[1]))  # Mask from RGB space
    

        # Subtract the blue_mask from binary1
        binary1 = cv.bitwise_and(binary1, cv.bitwise_not(blue_mask))
        binary2 = cv.inRange(hsv2, np.array(in_range2[0]), np.array(in_range2[1]))  # Mask from HSV space
        binary3 = cv.inRange(hsv3, np.array(in_range3[0]), np.array(in_range3[1]))  # Mask for HSV range
        # Threshold both red ranges
        binary4a = cv.inRange(hsv4, np.array(in_range4a[0]), np.array(in_range4a[1]))
        binary4b = cv.inRange(hsv4, np.array(in_range4b[0]), np.array(in_range4b[1]))
        # Combine them
        binary4 = cv.bitwise_or(binary4a, binary4b)  # Mask for HSV range

        # Detect particles in each binary mask and fit ellipses to their contours
        ellipses_1, outline_1, contours_1 = get_particle_list(binary1)
        ellipses_2, outline_2, contours_2 = get_particle_list(binary2)
        ellipses_3, outline_3, contours_3 = get_particle_list(binary3)
        ellipses_4, outline_4, contours_4 = get_particle_list(binary4)

        # Combine detections from both masks
        ellipses_combined = ellipses_1 + ellipses_2 + ellipses_3 + ellipses_4
        contours_combined = contours_1 + contours_2 + contours_3 + contours_4

        # Match detected particles to previous frame’s particles for ID continuity
        particle_ids, next_id = match_particles(prev_particles, ellipses_combined, prev_ids, next_id)

        # Update the previous frame data for use in the next iteration
        prev_particles = ellipses_combined
        prev_ids = particle_ids

        # Make a copy of the frame to draw ellipses and text
        image = frame.copy()
        for ellipse, pid in zip(ellipses_combined, particle_ids):
            x, y, maj, minr, ang = map(int, ellipse)  # Extract ellipse properties
            cv.ellipse(image, ((x, y), (maj, minr), ang), bgr['red'], 2)  # Draw ellipse on frame
            cv.putText(image, f'ID #{pid}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, bgr['black'], 2)  # Add ID label

        # Draw contours of detected particles
        cv.drawContours(image, contours_combined, -1, bgr['blue'], 1)

        # Draw wall outlines if provided
        if walls:
            #for i in range(len(walls)):
                #p1 = walls[i]
                #p2 = walls[(i + 1) % len(walls)]  # Connect walls in a closed loop
                #cv.line(image, tuple(p1), tuple(p2), bgr['yellow'], 2)
            cv.rectangle(image, (00, 00), (ww, hh), bgr['black'], 10)

        # Save particle data to a text file if any were detected
        if ellipses_combined:
            # Get centers
            centers = np.array([[e[0], e[1]] for e in ellipses_combined])

            # Compute pairwise distances
            distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
            # Particle-specific cutoffs (based on their minor axis)
            cutoffs = 1.05 * np.array([e[2] for e in ellipses_combined])

            # Coordination number calculation using asymmetric cutoffs
            coord_nums = np.zeros(len(centers), dtype=int)
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    cutoff_ij = 0.5 * (cutoffs[i] + cutoffs[j])
                    if distances[i, j] <= cutoff_ij:
                        coord_nums[i] += 1
                        coord_nums[j] += 1
            # Contact graph
            G = nx.Graph()
            G.add_nodes_from(range(len(centers)))
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    avg_cutoff = 0.5 * (cutoffs[i] + cutoffs[j])
                    if 0 < distances[i, j] < avg_cutoff:
                        G.add_edge(i, j)

            # Clustering coefficient
            clustering_coeffs = nx.clustering(G)
            clustering_list = [clustering_coeffs[i] for i in range(len(centers))]

            # Connected components → clusters
            clusters = list(nx.connected_components(G))
            cluster_labels = np.zeros(len(centers), dtype=int)
            for label, cluster in enumerate(clusters):
                for idx in cluster:
                    cluster_labels[idx] = label

            # Cluster density
            cluster_densities = []
            density_map = {}
            for i, cluster in enumerate(clusters):
                subG = G.subgraph(cluster)
                d = nx.density(subG)
                cluster_densities.append(d)
                for idx in cluster:
                    density_map[idx] = d
            density_list = [density_map[i] for i in range(len(centers))]

            # Betweenness centrality
            node_centrality = nx.betweenness_centrality(G)
            centrality_list = [node_centrality[i] for i in range(len(centers))]

            # Louvain modularity
            try:
                partition = community_louvain.best_partition(G)
                modularity = community_louvain.modularity(partition, G)
            except ValueError:
                modularity = 0.0
            
            # Global efficiency
            try:
                efficiency = nx.global_efficiency(G)
            except:
                efficiency = 0.0

            # Save per-frame system stats
            with open(f"{output_folder}SystemSummary.txt", 'a') as f:
                f.write(f"{curr_frame}\t{len(centers)}\t{len(clusters)}\t{max(len(c) for c in clusters)}\t{efficiency:.4f}\t{modularity:.4f}\n")

            # Save particle data
            areas = [cv.contourArea(cnt) for cnt in contours_combined]
            data = np.array([
                e + [area, pid, cn, cluster_labels[i], clustering_list[i], density_list[i], centrality_list[i]]
                for i, (e, area, pid, cn)
                in enumerate(zip(ellipses_combined, areas, particle_ids, coord_nums))
            ])
            np.savetxt(
                f"{output_folder}OutputFiles/PsC_{curr_frame:01d}.txt",
                data,
                delimiter='\t',
                fmt='%f',
                header='X_0\tY_0\tMinor_Axis\tMajor_Axis\tAngle\tArea\tID\tCoordination_Number\tCluster_ID\tClustering_Coeff\tCluster_Density\tBetweenness_Centrality'
            )# THIS PORTION FOCUSES ON CONTACT NETWORK IMAGE GENERATION
            # Create a black canvas the size of the ROI
            network_img = np.zeros((hh, ww), dtype=np.uint8)

            # Draw connection lines for neighbors
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    avg_cutoff = 0.5 * (cutoffs[i] + cutoffs[j])
                    if 0 < distances[i, j] <= avg_cutoff:
                        pt1 = tuple(np.round(centers[i]).astype(int))
                        pt2 = tuple(np.round(centers[j]).astype(int))
                        cv.line(network_img, pt1, pt2, 255, 1)

            # Draw particle nodes and label them by ID
            for (x, y), pid in zip(centers, particle_ids):
                cx, cy = int(round(x)), int(round(y))
                cv.circle(network_img, (cx, cy), 3, 255, -1)  # White dot at particle center
                cv.putText(network_img, str(pid), (cx + 5, cy - 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)

            # Save the network image
            cv.imwrite(f"{output_folder}ContactNetworks/ContactNet_{curr_frame:01d}.png", network_img)

        # Save labeled frame image
        cv.imwrite(f"{output_folder}LabeledFrames/Frame_{curr_frame:01d}.png", image)

        # Save binary masks used for detection
        cv.imwrite(f"{output_folder}BinaryImages/ActivePartsBinaryFrame_{curr_frame:01d}.png", binary1)
        cv.imwrite(f"{output_folder}BinaryImages/PassivePartsBinaryFrame_{curr_frame:01d}.png", binary4)

        curr_frame += 1  # Increment frame counter


# IMPORTANT NOTE: Depending on the type of color, change in_range accordingly

walls_fixed = [(0,0), (1145,0), (1145,1030), (0,1030)] # The origin is on the top left
white_range = [(155, 155, 155), (255, 255, 255)]   # Transparent/white
#yellow_range = [[18, 70,70], [70, 255, 255]] # Yellow
yellow_range = [[18, 70, 70], [70, 255, 255]] 
green_range = [[65, 60, 40], [85, 255, 255]]
red_range1 = [[0, 80, 80], [8, 255, 255]]
red_range2 = [[142, 100, 100], [180, 255, 255]]

#analyze_movies("DSC_0147.MOV", "Results147/", 245, 35, 1400, 1050, in_range1=white_range, in_range2=green_range,walls=walls_fixed )
analyze_movies("DSC_0169.MOV", "Results169/", 253, 0, 1386, 1030, in_range1=white_range,in_range2=yellow_range, in_range3=green_range, in_range4a=red_range1, in_range4b = red_range2, walls=walls_fixed )
