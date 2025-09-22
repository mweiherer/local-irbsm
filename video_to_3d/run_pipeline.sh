#!/bin/bash -l


filename=$(basename -- "$1")
base_output_dir="./${filename%.*}_reconstruction/"

python video_to_3d/extract_frames.py $1 $base_output_dir
python video_to_3d/run_sfm.py $base_output_dir
python video_to_3d/pick_2d_landmarks.py $base_output_dir

landmark_file=$(find $base_output_dir -maxdepth 1 -type f -name "*.txt" | head -n 1)
if [ -z "$landmark_file" ]; then
  echo "No landmarks file found in $base_output_dir. Aborting."
  exit 1
fi

python video_to_3d/align_and_prune.py $base_output_dir $landmark_file
python ../reconstruct.py $base_output_dir/aligned_pt_pruned.ply --landmarks $base_output_dir/aligned_landmarks_3d.ply --output_dir $base_output_dir --metrical --latent_weight 0.1 --anchor_weight 0.1