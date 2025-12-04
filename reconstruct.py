import os
import torch
import argparse
import point_cloud_utils as pcu

from models.implicit_model import ImplicitShapeModel
from utils.io import cond_mkdir, read_landmarks


# Scaling factor to obtain approximate metrical reconstructions.
# It's the average scaling factor obtained by transforming all 
# training scans into [-1, 1]^3.
MEAN_SCALE = 2.0 * 0.0017137007706792716

def main(args):
    if args.device not in ['cpu', 'cuda']:
        print('Invalid device.'); exit()

    if not args.ckpt:
        print('No checkpoint found.'); exit()

    lirbsm = ImplicitShapeModel(cfg_file = args.config, ckpt_file = args.ckpt, device = args.device, 
                                voxel_resolution = args.voxel_resolution, chunk_size = args.chunk_size)           

    cond_mkdir(args.output_dir) 

    point_cloud = torch.from_numpy(pcu.load_mesh_v(args.point_cloud)).double()

    if args.landmarks:
        landmarks = read_landmarks(args.landmarks)
        print(f'Loaded {len(landmarks)} landmarks from {args.landmarks}.')
    else:
        landmarks = None
        print('No landmarks given. The point cloud will be reconstructed as is.')

    if args.metrical:
        # Use exact metrical scale if nipple-to-nipple distance is given.
        if args.nipple_to_nipple_dist: 
            if landmarks is None:
                raise ValueError('Landmarks must be given to determine metrical scale from nipple-to-nipple distance.')
            
            print(f'Will perform exact metrical reconstruction based on provided nipple-to-nipple distance: {args.nipple_to_nipple_dist:.2f}')
            scale = torch.norm(landmarks[2, :] - landmarks[3, :]).item() / args.nipple_to_nipple_dist
        else: 
            # Use approximate metrical if no nipple-to-nipple distance given.
            print('Will perform approximate metrical reconstruction based on average scaling factor.')
            scale = MEAN_SCALE
    else: scale = -1

    reconstruction = lirbsm.reconstruct(point_cloud, landmarks, scale = scale, latent_weight = args.latent_weight,
                                        anchor_weight = args.anchor_weight)
    
    # Scale landmarks to align with reconstructed mesh if metrical reconstruction 
    # is performed and save them (might be required for the colorization step later).
    if args.metrical and landmarks is not None: 
        landmarks /= scale
        pcu.save_mesh_v(f'{args.output_dir}/scaled_{os.path.basename(args.landmarks)}.ply', landmarks.numpy())
        
    reconstruction.export(f'{args.output_dir}/reconstruction.ply')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('point_cloud', type = str, 
                           help = 'Path to point cloud that should be reconstructed.')
    argparser.add_argument('--landmarks', type = str, 
                           help = 'Path to the landmarks file.')
    argparser.add_argument('--output_dir', type = str, 
                           help = 'The directory to save the reconstruction to.',
                           default = './')
    argparser.add_argument('--metrical', action = 'store_true',
                           help = 'If set, the reconstruction is performed in metrical (real-world) scale.')
    argparser.add_argument('--nipple_to_nipple_dist', type = float,
                           help = 'The nipple-to-nipple distance measured on the real subject used to determine the metrical scale.' \
                                  'If not given, approximate metrical scale is used if --metrical is set.')
    argparser.add_argument('--latent_weight', type = float,
                           help = 'The regularization weight that penalizes deviation from the mean shape.',
                           default = 0.05)
    argparser.add_argument('--anchor_weight', type = float,
                           help = 'The weight of the anchor loss. If negative, no anchor loss is applied.',
                           default = -1)
    argparser.add_argument('--device', type = str, 
                           help = 'The device to run the model on.',
                           default = 'cuda')
    argparser.add_argument('--voxel_resolution', type = int,
                           help = 'The resolution of the voxel grid.',
                           default = 256)
    argparser.add_argument('--chunk_size', type = int,
                           help = 'Size of the chunks to split the voxel grid into.',
                           default = 100_000)
    argparser.add_argument('--config', type = str,
                           help = 'The path to the config file.', 
                           default = './configs/local_ensembled_deep_sdf_576.yaml')
    argparser.add_argument('--ckpt', type = str,
                           help = 'The path to the checkpoint file.', 
                           default = './checkpoints/lirbsm_576.pth')
  
    args, _ = argparser.parse_known_args()

    main(args)