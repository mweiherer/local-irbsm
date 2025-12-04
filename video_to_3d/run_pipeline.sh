#!/bin/bash

# Script to execute 3D reconstruction pipeline.


filename=$(basename -- "$1")

if [[ "$*" == *"--metrical"* ]]; then
    METRICAL=true
else
    METRICAL=false
fi

NIPPLE_DIST_OPTION=""
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[i]}" == "--nipple_to_nipple_dist" ]]; then
        NIPPLE_DIST_OPTION="--nipple_to_nipple_dist ${args[i+1]}"
        break
    fi
done

if [[ "$*" == *"--colorize"* ]]; then
    COLORIZE=true
else
    COLORIZE=false
fi

if [[ "$COLORIZE" = true && "$*" == *"--remove_invisible"* ]]; then
    REMOVE_INVISIBLE=true
else
    REMOVE_INVISIBLE=false
fi

base_output_dir="./${filename%.*}_reconstruction/"

python video_to_3d/extract_frames.py $1 $base_output_dir
python video_to_3d/run_sfm.py $base_output_dir
python video_to_3d/pick_2d_landmarks.py $base_output_dir/images

landmark_file=$(find $base_output_dir -maxdepth 1 -type f -name "*.txt" | head -n 1)
if [ -z "$landmark_file" ]; then
  echo "No landmarks file found in $base_output_dir. Aborting."
  exit 1
fi

python video_to_3d/align_and_prune.py $base_output_dir $landmark_file

RECONSTRUCT_CMD="python ./reconstruct.py $base_output_dir/aligned_pt_pruned.ply --landmarks $base_output_dir/aligned_landmarks_3d.ply --output_dir $base_output_dir --latent_weight 0.1 --anchor_weight 0.1"

if [ "$METRICAL" = true ]; then
   $RECONSTRUCT_CMD --metrical $NIPPLE_DIST_OPTION
else
   $RECONSTRUCT_CMD
fi

COLORIZE_CMD="python video_to_3d/colorize_mesh.py $base_output_dir"

if [ "$COLORIZE" = true ]; then
  if [ "$REMOVE_INVISIBLE" = true ]; then
    $COLORIZE_CMD --remove_invisible
  else
    $COLORIZE_CMD
  fi
fi