#!/bin/bash

# Define constants --> CHANGE ACCORDINGLY
xx=245
ww=1400
yy=35
hh=1050
pi=3.141592653589793
numb=42 # Total Number of Particles

# Compute TotalArea
TotalArea=$(( (ww) * (hh) ))

# Define output file
OUTPUT_FILE="Results148/IDData/PackFract.txt" # CHSNGE ACCORDINGLY HERE
PREFIX="Results148/OutputFiles/PsC_" # CHSNGE ACCORDINGLY HERE

# Create directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Empty the output file if it exists
> "$OUTPUT_FILE"

# Header with new column
echo -e "# Frame_Number\tPackFract\tAvg_Coord_Number\tStDev_Coord_Number" > "$OUTPUT_FILE"

# Loop through files
for i in {1..1520}; do # CHSNGE ACCORDINGLY HERE
    FILE="${PREFIX}${i}.txt"

    if [[ -f "$FILE" ]]; then
        # Compute packing fraction using awk
        packing_fraction=$(awk -v pi="$pi" -v TotalArea="$TotalArea" '
            BEGIN {sum = 0}
            NR > 1 {sum += $6}
            END {printf "%.8f", sum / TotalArea}
        ' "$FILE")

        # Compute avg and stdev of coordination number
        read avg_coord_numb stdev_coord_numb <<< $(awk -v numb="$numb" '
            BEGIN {sum = 0; sumsq = 0; count = 0}
            NR > 1 {
                sum += $8
                sumsq += ($8)^2
                count++
            }
            END {
                if (count > 0) {
                    mean = sum / count
                    variance = (sumsq / count) - (mean^2)
                    if (variance < 0) variance = 0 # guard against tiny negatives
                    stdev = sqrt(variance)
                    printf "%.8f %.8f", mean, stdev
                } else {
                    printf "0 0"
                }
            }
        ' "$FILE")

        # Output all columns
        echo "$i $packing_fraction $avg_coord_numb $stdev_coord_numb" >> "$OUTPUT_FILE"
    else
        echo "Warning: File $FILE does not exist, skipping." >&2
    fi
done
