import numpy as np
import torch


def rigid_landmark_alignment(src_pts, tar_pts, estimate_scale = True): 
    '''
    Performs a landmark-based similarity alignment between two corresponding point sets. See:
        Shinji Umeyama, Least-squares estimation of transformation 
                        parameters between two point patterns, TPAMI'91.
    :param src_pts: The source points as torch.Tensor of size [n, 3]
    :param tar_pts: The target points as torch.Tensor of size [n, 3]
    :param estimate_scale: If True, estimates the scale factor during alignment
    :return: The rotation matrix R as torch.Tensor of shape [3, 3]
    :return: The translation vector t as torch.Tensor of shape [3]
    :return: The scaling factor c as scalar
    '''
    assert src_pts.shape == tar_pts.shape \
        and src_pts.dtype == tar_pts.dtype
 
    mu_src = torch.mean(src_pts, dim = 0)
    mu_tar = torch.mean(tar_pts, dim = 0)
    src_pts_white = src_pts - mu_src
    tar_pts_white = tar_pts - mu_tar
    cov_m = (src_pts_white.t() @ tar_pts_white) / src_pts.shape[0]
    U, D, V = torch.svd(cov_m)
    S = torch.eye(3).to(src_pts)
    S[2, 2] = torch.det(V @ U)
    R = V @ S @ U.t()

    if estimate_scale:
        c = (1 / torch.var(src_pts, dim = 0, unbiased = False).sum()) * (torch.diag(D) @ S).trace()
    else: c = 1.0
 
    t = mu_tar - c * R @ mu_src
    return R, t, c

def closest_point_along_ray(points, ray_origin, ray_direction):
    '''
    Shots a ray into the scene and selects the point as back-projected landmark
    that first hits the scene (Eq. 9 in the paper). This is implemented by searching
    for a point in the scene that is near to the ray and closest to the camera.
    :param points: Scene point cloud as numpy array of size [n, 3]
    :param ray_origin: Ray origins in world coordinates as numpy array of size [K, 3]
    :param ray_direction: Normalized ray directions in world coordinates as numpy array of size [K, 3]
    :return: The back-projected landmarks as numpy array of size [K, 3]
    '''
    if not np.isclose(np.linalg.norm(ray_direction), 1.0):
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

    # We first select a set of 'candidate' points in the scene. These are points 
    # that are near (closer than a certain threshold) to the ray. We don't set 
    # this threshold explicitly, but implicitly by selecting the 'num_smallest' 
    # points that are closest to the ray.
    distances = np.linalg.norm(np.cross(points - ray_origin, ray_direction) , axis = 1) # Eq. 10 in the paper.
    num_smallest = 5
    indices = np.argpartition(distances, num_smallest)[:num_smallest]
    valid_points = points[indices]

    # From this candidate set, we then select the point as back-projected landmark that 
    # is closest to the camera. 
    t = np.dot(valid_points - ray_origin, ray_direction)
    closest_idx = np.argmin(t)

    return valid_points[closest_idx]  

def project_2d_points_to_3d(landmarks_uv, vertices_3d, intrinsics_matrix, extrinsics_matrix, depth_map):   
    '''
    Back-projects 2D landmarks to obtain corresponding 3D landmark positions.
    :param landmarks_uv: 2D landmark positions as numpy array of size [K, 2]
    :param vertices_3d: Scene point cloud as numpy array of size [n, 3]
    :param intrinsics_matrix: Camera intrinsics as numpy array of size [3, 3]
    :param extrinsics_matrix: Camera extrinsics as numpy array of size [4, 4]
    :param depth_map: Depth map as numpy array
    :return: 3D landmark positions as numpy array of size [K, 3]
    ''' 
    K_inv = np.linalg.inv(intrinsics_matrix)

    R = extrinsics_matrix[:3, :3]  
    t = extrinsics_matrix[:3, -1]  

    o = -R.T @ t # Ray origin in world coordinates.

    closest_points, closest_points_depth = [], []
    for (u, v) in landmarks_uv:
        ray_camera =  K_inv @ np.array([u, v, 1.0])
        ray_world = R.T @ ray_camera # Ray direction in world coordinates.
        ray_world = ray_world / np.linalg.norm(ray_world) 
      
        closest_point = closest_point_along_ray(vertices_3d, o, ray_world)

        closest_points.append(closest_point)

        if depth_map is not None: 
            closest_point_depth = o + depth_map[v, u] * ray_world
            closest_points_depth.append(closest_point_depth)

    return np.array(closest_points), np.array(closest_points_depth) if depth_map is not None else None