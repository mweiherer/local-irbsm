import argparse
import torch
import numpy as np
import os
import point_cloud_utils as pcu
import time
import pykdtree.kdtree as kdtree
import cv2
import time

from utils import project_2d_points_to_3d, rigid_landmark_alignment


def main(args):
    print('Aligning and pruning point cloud.')

    model_landmarks = torch.load('artifacts/average_anchors.pt') 
    mean_shape_points = torch.load('artifacts/lirbsm-mean_pts.pt')

    camera_id = int(args.landmarks.split('/')[-1].split('_')[0])
    landmarks_uv = np.loadtxt(args.landmarks, dtype = int, delimiter = ',')

    predictions = torch.load(os.path.join(args.input_base_path, 'vggsfm_predictions.pt'))
    raw_point_cloud = predictions['points3D'].cpu().numpy()
    camera_matrices = predictions['extrinsics_opencv'].cpu().numpy()
    intrinsics_matrices = predictions['intrinsics_opencv'].cpu().numpy()
    dense_depth_maps = predictions['depth_dict'] if 'depth_dict' in predictions else None
   
    if args.debug:
        pcu.save_mesh_v(os.path.join(args.input_base_path, 'raw_pt.ply'), raw_point_cloud)
        
        if dense_depth_maps is not None:
            depth_map = dense_depth_maps[f'{camera_id:03d}.png'] 

            depth_colored = cv2.applyColorMap(cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                                              cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(args.input_base_path, f'depth_map_{camera_id:03d}_color.png'), depth_colored)
  
    num_cameras = len(camera_matrices)
    extrinsics_matrices = np.zeros((num_cameras, 4, 4))
    extrinsics_matrices[:, :3, :4] = camera_matrices
    extrinsics_matrices[:, 3, 3] = 1

    s = time.time()  

    # First step is to get 3D landmark positions from the 2D landmarks in the image.
    # To do so, we cast rays from the camera through the 2D landmark positions into 
    # the 3D scene and find the closest point on the point cloud to the ray. If dense
    # depth maps are available, we can directly use the depth information to find the 
    # closest point (in fact, in this case, we do both).   
    landmarks_3d, landmarks_3d_depth = project_2d_points_to_3d(
        landmarks_uv, raw_point_cloud, intrinsics_matrices[camera_id], extrinsics_matrices[camera_id],
        dense_depth_maps[f'{camera_id:03d}.png'] if dense_depth_maps is not None else None
    )

    if args.debug:
        pcu.save_mesh_v(os.path.join(args.input_base_path, 'landmarks_3d.ply'), landmarks_3d)
        if landmarks_3d_depth is not None:
            pcu.save_mesh_v(os.path.join(args.input_base_path, 'landmarks_3d_depth.ply'), landmarks_3d_depth)

    if args.use_depth_if_available and landmarks_3d_depth is not None:
        print('Using depth map to get 3D landmark positions.')
        landmarks_3d = landmarks_3d_depth

    # Second step is to align the raw point cloud, based on the 3D landmark positions, to the model.
    landmarks_3d = torch.from_numpy(landmarks_3d)

    R, t, c = rigid_landmark_alignment(landmarks_3d, model_landmarks, estimate_scale = True)
    aligned_point_cloud = (torch.from_numpy(raw_point_cloud) * c) @ R.T + t
    aligned_landmarks_3d = (landmarks_3d * c) @ R.T + t

    if args.debug: 
        pcu.save_mesh_v(os.path.join(args.input_base_path, 'aligned_pt.ply'), aligned_point_cloud.numpy())

    # Last step is to prune away points that are farther away from the mean shape than a certain threshold.
    tree = kdtree.KDTree(mean_shape_points.numpy())
    distances, _ = tree.query(aligned_point_cloud.numpy(), k = 1)
    aligned_point_cloud_pruned = aligned_point_cloud[distances < args.pruning_dist_threshold]
    e = time.time()
    
    pcu.save_mesh_v(os.path.join(args.input_base_path, 'landmarks_3d.ply'), landmarks_3d.numpy())
    pcu.save_mesh_v(os.path.join(args.input_base_path, 'aligned_landmarks_3d.ply'), aligned_landmarks_3d.numpy())
    pcu.save_mesh_v(os.path.join(args.input_base_path, 'aligned_pt_pruned.ply'), aligned_point_cloud_pruned.numpy())
    
    print(f'Done. Took {e - s:.2f} seconds.')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('input_base_path', type = str,
                           help = 'Base path for reconstruction output.')
    argparser.add_argument('landmarks', type = str, 
                           help = 'Path to the landmarks file. This file must be named <camera_id>_landmarks.txt.')
    argparser.add_argument('--pruning_dist_threshold', type = float,
                           help = 'Distance threshold for pruning points from the aligned point cloud.',
                           default = 0.2)
    argparser.add_argument('--use_depth_if_available', action = 'store_true',
                           help = 'If set, use the depth map to get 3D landmark positions from the 2D landmarks provided.' \
                           'Otherwise, use the point cloud.')
    argparser.add_argument('--debug', action = 'store_true',
                           help = 'If set, save intermediate results for debugging.')
    
    args, _ = argparser.parse_known_args()

    main(args)