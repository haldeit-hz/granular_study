#!/bin/bash

# Here one specifies the range of particles IDs for which they would like to extract data --> QUICK NOTE: Check PsC outputfiles and labeled images to get a sense of which particles are active and which are passive
for j in {13..45};do
  ID_TO_EXTRACT=$(printf "%.6f" "$j")  # Rewriting it in a way connectable to the output file format
  OUTPUT_FILE="Results148/IDData/P/id_${ID_TO_EXTRACT}_trajectory.txt" # OUtput file --> CHANGE ACCORDINGLY
  PREFIX="Results148/OutputFiles/PsC_"     # File format from which one extracts data --> CHANGE ACCORDINGLY

  # Create it if it doesn't exist
  mkdir -p "$(dirname "$OUTPUT_FILE")"

  # Clear output file if it exists
  > "$OUTPUT_FILE"

  # Header to make things clear
  echo -e "# X_0\tY_0\tMinor_Axis\tMajor_Axis\tAngle\tArea\tID\tCoordination_Number\tCluster_ID\tClustering_Coeff\tCluster_Density\tBetweenness_Centrality\tFrame_Number" > "$OUTPUT_FILE"

  # Loop through all matching files
  for i in {1..1520}; do # Make sure to know the very last frame number
    gawk -v ID="$ID_TO_EXTRACT" -v file_index="$i" -F'\t' '
      BEGIN { found = 0 }
      {
        # Filter lines that are all numbers and have 6 tab-separated fields
        if (!/[a-df-zA-DF-Z]/ && NF == 12) {
          if ($7 == ID) {
            found += 1
            print $0 "\t" file_index
          }
        }
      }' "${PREFIX}${i}.txt" >> "$OUTPUT_FILE"
  done

  echo "Extraction complete --> Data saved in $OUTPUT_FILE"
done
