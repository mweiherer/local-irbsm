import argparse
import torch
import numpy as np
import os
import cv2
import point_cloud_utils as pcu
from glob import glob
from tqdm import tqdm

from utils import rigid_landmark_alignment


def color_vertices_from_images(vertices, normals, images, camera_matrices, intrinsics_matrices):
    '''
    Colors the reconstructed mesh from the input multi-view images by projecting its vertices 
    into each image, sampling the color, and (weighted) averaging the colors from all images.
    :param vertices: Mesh vertices as numpy array of size [num_vertices, 3]
    :param normals: Mesh vertex normals as numpy array of size [num_vertices, 3]
    :param images: List of input images as numpy arrays of size [H, W, 3]
    :param camera_matrices: Camera extrinsics as numpy array of size [num_cameras, 3, 4]
    :param intrinsics_matrices: Camera intrinsics as numpy array of size [num_cameras, 3, 3]
    :return vertex_colors: Vertex colors as numpy array of size [num_vertices, 3]
    :return valid_mask: Boolean mask indicating which vertices were seen in at least one view
    '''
    num_vertices = len(vertices)

    vertex_colors = np.zeros((num_vertices, 3))
    vertex_weights = np.zeros(num_vertices)
    
    for (image, E, K) in tqdm(zip(images, camera_matrices, intrinsics_matrices), total = len(images)):
        H, W = image.shape[:2]
        
        # Project vertices from world coordinates to camera coodinates.
        vertices_hom = np.concatenate([vertices, np.ones((num_vertices, 1))], axis = 1)
        vertices_cam = (E @ vertices_hom.T).T[:, :3]

        # Get depths (distance from camera; z coordinate).
        depths = vertices_cam[:, 2]
        
        # Project camera coordinates to image space.
        vertices_proj = (K @ vertices_cam.T).T
        vertices_img = vertices_proj[:, :2] / (vertices_proj[:, 2:3] + 1e-8) # [num_vertices, 2]

        # Check if vertex is facing the camera (back-face culling).
        R, t = E[:3, :3], E[:3, 3]
        camera_pos = -R.T @ t
        view_dirs = camera_pos - vertices
        view_dirs = view_dirs / (np.linalg.norm(view_dirs, axis = 1, keepdims = True) + 1e-8)
        facing_camera = np.sum(normals * view_dirs, axis = 1) > 0 

        visible = (depths > 0) & facing_camera & \
                  (vertices_img[:, 0] >= 0) & (vertices_img[:, 0] < W) & \
                  (vertices_img[:, 1] >= 0) & (vertices_img[:, 1] < H) 
        
        if visible.sum() == 0: continue
        
        # Sample RGB colors from image.
        for v_idx in np.where(visible)[0]:
            x, y = vertices_img[v_idx]
            depth = depths[v_idx]
            
            # Get color at sub-pixel position (x, y) using bilinear interpolation. 
            x0, y0 = int(x), int(y) # floor
            x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1) # ceil within image bounds
          
            fx, fy = x - x0, y - y0
       
            color = (image[y0, x0] * (1 - fx) * (1 - fy) +
                     image[y0, x1] * fx * (1 - fy) +
                     image[y1, x0] * (1 - fx) * fy +
                     image[y1, x1] * fx * fy)
            
            # Weight by inverse distance (closer views have more influence).
            weight = 1.0 / (depth + 1e-6)
            
            vertex_colors[v_idx] += weight * color 
            vertex_weights[v_idx] += weight

    # Normalize by weights.
    valid_mask = vertex_weights > 0

    vertex_colors[valid_mask] /= vertex_weights[valid_mask, None]
    
    # For vertices not seen in any view, use grey color.
    vertex_colors[~valid_mask] = 128
 
    return (np.clip(vertex_colors, 0, 255)).astype(np.uint8), valid_mask

def main(args):
    print('Colorizing reconstruction from multi-view images.')

    images_path = os.path.join(args.input_base_path, 'images')
    sfm_predictions_path = os.path.join(args.input_base_path, 'vggsfm_predictions.pt')
    mesh_path = os.path.join(args.input_base_path, 'reconstruction.ply')
    landmarks_path = os.path.join(args.input_base_path, 'landmarks_3d.ply') # Landmarks in scene. 
    scaled_aligned_landmarks_path = os.path.join(args.input_base_path, 'scaled_aligned_landmarks_3d.ply')
    aligned_landmarks_path = os.path.join(args.input_base_path, 'aligned_landmarks_3d.ply')

    # Load data.
    images = [cv2.imread(f, cv2.IMREAD_COLOR_RGB) for f in sorted(glob(f'{images_path}/*.png'))]

    predictions = torch.load(sfm_predictions_path)
    camera_matrices = predictions['extrinsics_opencv'].cpu().numpy()
    intrinsics_matrices = predictions['intrinsics_opencv'].cpu().numpy()

    v, f, n = pcu.load_mesh_vfn(mesh_path)

    landmarks_3d = torch.from_numpy(pcu.load_mesh_v(landmarks_path)).double()

    if os.path.exists(scaled_aligned_landmarks_path):
        # If exists, then we are in the metrical case...
        recon_landmarks_3d = torch.from_numpy(pcu.load_mesh_v(scaled_aligned_landmarks_path)).double()
    else:
        # ... otherwise it must be a non-metrical reconstruction (see reconstruct.py). 
        recon_landmarks_3d = torch.from_numpy(pcu.load_mesh_v(aligned_landmarks_path)).double()

    # Align the reconstruction with the scene based on the landmarks.  
    R, t, c = rigid_landmark_alignment(recon_landmarks_3d, landmarks_3d, estimate_scale = True) 
    v_scene_aligned = ((v * c.numpy()) @ R.numpy().T) + t.numpy()
    n_scene_aligned = n @ R.numpy().T
    
    # Color mesh vertices from multi-view images.
    vertex_colors, valid_mask = color_vertices_from_images(v_scene_aligned, n_scene_aligned, images,
                                                           camera_matrices, intrinsics_matrices)

    if args.remove_invisible:
        print('Removing invisible regions from the mesh.')
        
        vi_map = np.cumsum(valid_mask) - 1
        face_valid = valid_mask[f].all(axis = 1)

        pcu.save_mesh_vfnc(os.path.join(args.input_base_path, 'reconstruction_without_invisible.ply'),
                           v[valid_mask], vi_map[f[face_valid]], n[valid_mask], vertex_colors[valid_mask])
    else:
        pcu.save_mesh_vfnc(mesh_path, v, f, n, vertex_colors)

    print('Done.')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('input_base_path', type = str,
                           help = 'Base path for reconstruction output.')
    argparser.add_argument('--remove_invisible', action = 'store_true',
                           help = 'Whether to remove invisible parts not seen from ' \
                           'any camera from the reconstructed mesh (usually the back).')
    
    args, _ = argparser.parse_known_args()

    main(args)