import argparse
import subprocess
import time


def run_vggsfm(output_base_path, camera_type, shared_camera, query_method, query_frame_num, max_query_pts, 
               num_ba_iterations, predict_dense_depth, extra_pt_pixel_interval, concat_extra_points):
    script_path = 'video_to_3d/vggsfm_runner.py'
    env_name = 'vggsfm_tmp'

    args = [f'SCENE_DIR={output_base_path}',
            'camera_type=' + camera_type,
            'shared_camera=' + str(shared_camera),
            'query_method=' + query_method,
            'query_frame_num=' + str(query_frame_num),
            'max_query_pts=' + str(max_query_pts),
            'BA_iters=' + str(num_ba_iterations),
            'dense_depth=' + str(predict_dense_depth),
            'extra_pt_pixel_interval=' + str(extra_pt_pixel_interval),
            'concat_extra_points=' + str(concat_extra_points)]

    cmd = [
        'conda', 'run', '-n', env_name, 'python', script_path
    ] + args

    print('Running VGGSfM with command:', ' '.join(cmd))

    s = time.time()
    subprocess.run(cmd)
    e = time.time()
    
    print(f'Done. Took {e - s:.2f} seconds.')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('input_base_path', type = str,
                           help = 'Base path for reconstruction output.')
    argparser.add_argument('--camera_type', type = str,
                           help = 'Camera model type to use. Default is SIMPLE_RADIAL.',
                           default = 'SIMPLE_RADIAL')
    argparser.add_argument('--shared_camera', type = str, 
                           help = 'Whether to use a shared camera model. Default is True.',
                           default = True)
    argparser.add_argument('--query_method', type = str, 
                           help = 'Method for querying frames. Default is sp+sift+aliked.',
                           default = 'sp+sift+aliked')
    argparser.add_argument('--query_frame_num', type = int,
                           help = 'Number of frames to query. Default is 3.',
                           default = 3)
    argparser.add_argument('--max_query_pts', type = int, 
                           help = 'Maximum number of query points. Default is 8192.',
                           default = 8_192)
    argparser.add_argument('--num_ba_iterations', type = int, 
                           help = 'Number of bundle adjustment iterations. Default is 3.',
                           default = 3)
    argparser.add_argument('--predict_dense_depth', type = bool, 
                           help = 'Whether to predict dense depth. Default is False.',
                           default = False)
    argparser.add_argument('--extra_pt_pixel_interval', type = int,
                           help = 'Interval for extra points in pixels. Default is -1 (disabled).',
                           default = -1)
    argparser.add_argument('--concat_extra_points', type = bool,
                           help = 'Whether to concatenate extra points. Default is False.',
                           default = False)

    args, _ = argparser.parse_known_args()

    run_vggsfm(args.input_base_path, args.camera_type, args.shared_camera, args.query_method, args.query_frame_num, args.max_query_pts, 
               args.num_ba_iterations, args.predict_dense_depth, args.extra_pt_pixel_interval, args.concat_extra_points)