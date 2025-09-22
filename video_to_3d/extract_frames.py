import argparse
import os
import cv2
import time
import numpy as np
from pathlib import Path


def extract_frames_uniformly(input_file, output_base_path, num_frames):
    '''
    Extracts a specified number of frames uniformly from a video, regardless of image quality.
    :param input_file: Path to the input video file
    :param output_base_path: Base path for output frames
    :param num_frames: Number of frames to extract
    '''
    frames_path = os.path.join(output_base_path, 'images')
    Path(frames_path).mkdir(parents = True, exist_ok = True)

    capture = cv2.VideoCapture(input_file)

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    timestamps = np.linspace(0, duration, num_frames + 2)[1:-1]
 
    for i, t in enumerate(timestamps):
        capture.set(cv2.CAP_PROP_POS_MSEC, 1_000 * t)
        ret, frame = capture.read()
        if not ret: continue
        cv2.imwrite(os.path.join(frames_path, f'{i:03d}.png'), frame)

    capture.release()

def calculate_sharpness(frame):
    '''
    Calculates the sharpness of an image as the variance of its Laplacian.
    Higher means sharper.
    :param frame: Input image
    :return: Sharpness value
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def determine_auto_threshold(input_file, sample_size = 100, print_statistics = False):
    '''
    Auto-determines a sharpness threshold from a given video by analyzing statistics of 
    frames uniformly extracted from the video. The threshold is set to the 75th percentile
    of sharpness values from sampled frames.
    :param input_file: Path to the input video file
    :param sample_size: Number of frames to sample for threshold determination
    :param print_statistics: Whether to print sharpness statistics computed from video
    :return: Auto-determined sharpness threshold
    '''
    capture = cv2.VideoCapture(input_file)

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    timestamps = np.linspace(0, duration, sample_size + 2)[1:-1]

    sharpness_values = []
    for t in timestamps:
        capture.set(cv2.CAP_PROP_POS_MSEC, 1_000 * t)
        ret, frame = capture.read()
        if not ret: continue
            
        sharpness = calculate_sharpness(frame)
        sharpness_values.append(sharpness)
  
    capture.release()
    
    sharpness_values = np.array(sharpness_values)
    auto_threshold = np.percentile(sharpness_values, 75)

    if print_statistics:
        mean_sharpness = np.mean(sharpness_values)
        std_sharpness = np.std(sharpness_values)
        median_sharpness = np.median(sharpness_values)
        print(mean_sharpness, std_sharpness, median_sharpness, auto_threshold)
  
    return auto_threshold

def extract_only_sharp_frames(input_file, output_base_path, num_frames, sharpness_threshold = None,
                              num_window_samples = 20, print_statistics = False):
    '''
    Extracts a specified number of frames from a video, ensuring both sharpness and temporal coverage.
    To do so, we first select a set of candidate frames uniformly distributed over the video duration.
    Then, for each candidate frame, we search in a local window around it for the sharpest frame
    that exceeds a given sharpness threshold. If no such frame is found, we lower the threshold and try 
    again, up to three times. If still no frame is found, we take the sharpest available frame in the window.
    :param input_file: Path to the input video file
    :param output_base_path: Base path for output frames
    :param num_frames: Number of frames to extract
    :param sharpness_threshold: Minimum sharpness threshold for frame selection. If None, auto-determined.
    :param num_window_samples: Number of frames to sample within each local window for sharpness evaluation
    :param print_statistics: Whether to print sharpness statistics computed from video
    '''
    frames_path = os.path.join(output_base_path, 'images')
    Path(frames_path).mkdir(parents = True, exist_ok = True)

    # Auto-determine threshold if not provided.
    if sharpness_threshold is None:
        initial_threshold = determine_auto_threshold(input_file)
        print('Using auto-determined initial threshold:', initial_threshold)
    else:
        initial_threshold = sharpness_threshold
        print('Using manual initial threshold:', initial_threshold)

    capture = cv2.VideoCapture(input_file)
    
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    target_timestamps = np.linspace(0, duration, num_frames + 2)[1:-1]
    
    # For each target timestamp, we'll search for the sharpest frame in a local window.
    window_size = duration / (3 * num_frames) 
    
    selected_frames = []
    current_threshold = initial_threshold
    
    for target_time in target_timestamps:
        # Define search window around target timestamp.
        window_start = max(0, target_time - window_size / 2)
        window_end = min(duration, target_time + window_size / 2)
        
        # Sample frames within this window (20 per default).
        window_timestamps = np.linspace(window_start, window_end, num_window_samples)
        
        # Now, try to find a good frame in this window, lowering the threshold if needed.
        best_frame, best_sharpness, attempts = None, -1, 0
        local_threshold = current_threshold
    
        while best_frame is None and attempts < 3:
            # Search for the sharpest frame within the local window using the current local threshold.
            for t in window_timestamps:
                capture.set(cv2.CAP_PROP_POS_MSEC, 1_000 * t)
                ret, frame = capture.read()
                if not ret: continue
                    
                sharpness = calculate_sharpness(frame)
                
                if sharpness >= local_threshold and sharpness > best_sharpness:
                    best_sharpness = sharpness
                    best_frame = (t, sharpness, frame.copy())
            
            # If we didn't find any frame above the threshold, lower it by 30% and try again.
            if best_frame is None:
                local_threshold *= 0.7
                attempts += 1
            else: break

        # If still no frame found after three attempts of adapting the threshold, 
        # take the sharpest available frame in window (regardless of threshold).
        if best_frame is None:
            for t in window_timestamps:
                capture.set(cv2.CAP_PROP_POS_MSEC, 1_000 * t)
                ret, frame = capture.read()
                if not ret: continue
                    
                sharpness = calculate_sharpness(frame)

                if sharpness > best_sharpness:
                    best_sharpness = sharpness
                    best_frame = (t, sharpness, frame.copy())
        
        # 'best_frame' should never be None here, but I check it anyway because I'm paranoid.
        if best_frame is not None:
            selected_frames.append(best_frame)

            # Adaptive threshold adjustment: if we found a good frame, slightly raise threshold.
            if best_sharpness > (1.2 * current_threshold):
                current_threshold = min(initial_threshold, 1.05 * current_threshold)
    
    capture.release()
    
    if len(selected_frames) == 0:
        print('Error: Could not extract any frames.'); return
    
    # Finally, save selected frames to disk.
    selected_frames.sort(key = lambda x: x[0])
    for i, (t, sharpness, frame) in enumerate(selected_frames):
        cv2.imwrite(os.path.join(frames_path, f'{i:03d}.png'), frame)
    
    if print_statistics:
        print(f'Selected {len(selected_frames)} frames from {num_frames} requested.')

        sharpness_values = [_[1] for _ in selected_frames]
        mean_sharpness = np.mean(sharpness_values)
        min_sharpness = np.min(sharpness_values)
        print('Mean/Min sharpness:', mean_sharpness, min_sharpness)

        print(f'Temporal coverage: {selected_frames[0][0]:.2f}s to {selected_frames[-1][0]:.2f}s. Total duration: {duration:.2f}s.')
      
def main(args):
    if args.uniform:
        print(f'Extracting {args.num_frames} frames uniformly from video.'); 
        s = time.time()
        extract_frames_uniformly(args.input_file, args.output_base_path, args.num_frames)
        e = time.time()
    else:
        print(f'Extracting {args.num_frames} sharp frames with good temporal coverage from video.');  
        threshold = args.sharpness_threshold if args.sharpness_threshold != -1 else None
        s = time.time()
        extract_only_sharp_frames(args.input_file, args.output_base_path, args.num_frames, threshold)
        e = time.time()

    print(f'Done. Took {e - s:.2f} seconds.')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('input_file', type = str,
                           help = 'Path to the input video file.')
    argparser.add_argument('output_base_path', type = str,
                           help = 'Base path for reconstruction output.')
    argparser.add_argument('--num_frames', type = int,
                           help = 'Number of frames to extract. Default is 30.',
                           default = 30)
    argparser.add_argument('--sharpness_threshold', type = float, 
                           help = 'Sharpness threshold for frame selection. Use -1 for auto-determination. Default is -1.',
                           default = -1)
    argparser.add_argument('--uniform', action = 'store_true',
                           help = 'Whether to extract frames uniformly, regardless of image quality.')

    args, _ = argparser.parse_known_args()

    main(args)