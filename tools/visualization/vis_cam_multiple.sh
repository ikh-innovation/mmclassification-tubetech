#!/bin/bash

# RUN FROM ROOT
# Set the folder path
folder_path="/home/innovation/Projects/tubetech/mmclassification-tubetech/layer_activations"
config="/home/innovation/Projects/tubetech/mmclassification-tubetech/configs/efficientnet/efficientnet-b5_2xb4_in1k-456px_boiler_defects_tiled_v1.py"
checkpoint="/home/innovation/Projects/tubetech/mmclassification-tubetech/work_dirs/efficientnet-b5_2xb4_in1k-456px_boiler_defects_tiled_v1/best_multi-label_precision_top1_epoch_42.pth"

# Add the module directory to the Python module search path
module_directory="/mnt/storage/Projects/tubetech/mmclassification-tubetech"
export PYTHONPATH="$module_directory:$PYTHONPATH"

# Iterate over each file in the folder
for file_path in "$folder_path"/*
do
    # Check if the file is a regular file
    if [ -f "$file_path" ]
    then
        # Extract the file name without the extension
        file_name=$(basename "$file_path")
        file_name="${file_name%.*}"

        # Execute the Python script with the file name as a parameter
        python_interpreter="/home/innovation/anaconda3/envs/mmclass/bin/python"
        python_script="/home/innovation/Projects/tubetech/mmclassification-tubetech/tools/visualization/vis_cam.py"
        save_path="$folder_path/$file_name-CAM.jpg"
        "$python_interpreter" "$python_script" "$file_path" "$config" "$checkpoint" --method FullGrad --save-path "$save_path"
        # --target-category 0
        # Add any additional commands or operations you want to perform on each file
        # ...
    fi
done