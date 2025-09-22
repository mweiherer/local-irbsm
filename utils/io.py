import torch
import numpy as np
import yaml
from pathlib import Path
import xml.etree.ElementTree as ET
import point_cloud_utils as pcu


def cond_mkdir(path):
    Path(path).mkdir(parents = True, exist_ok = True)

def read_config(config):
    '''
    Loads a given config file in yaml format.
    :param config: The config file in yaml format
    :return: The config file as dictionary
    '''
    with open(config, 'r') as stream:
        cfg = yaml.safe_load(stream)
    return cfg

def dtype_from_string(dtype_str, as_type = 'torch'):
    '''
    Converts dtype given as string to dtype in PyTorch or Numpy format.
    :param dtype_str: The dtype as string
    :param as_type: The target dtype, either 'torch' or 'numpy' (default 'torch')
    :return: Converted dtype
    '''
    if as_type not in ['torch', 'numpy']:
        raise ValueError(f"Unkown type '{as_type}' given. Must be either 'torch' or 'numpy'.")

    if dtype_str == 'float16':
        return torch.float16 if as_type == 'torch' else np.float16
    if dtype_str == 'float32':
        return torch.float32 if as_type == 'torch' else np.float32
    if dtype_str == 'float64':
        return torch.float64 if as_type == 'torch' else np.float64

def read_landmarks(landmarks_file):  
    '''
    Loads m landmark positions stored in either .pp, .csv file, or .ply format.
    :param landmarks_file: Path to the file containing landmarks
    :return: Landmark positions as torch.Tensor of size [m, 3]
    ''' 
    file_extension = landmarks_file.split('/')[-1].split('.')[-1]     
    
    if file_extension == 'pp':
        tree = ET.parse(landmarks_file)
        root = tree.getroot()

        points = []
        for point in root.findall('.//point'):
            x = float(point.get('x'))
            y = float(point.get('y'))
            z = float(point.get('z'))
            points.append([x, y, z])
        return torch.from_numpy(np.array(points))
    
    if file_extension == 'csv':
        return torch.from_numpy(np.loadtxt(landmarks_file, delimiter = ','))
    
    if file_extension == 'ply':
        return torch.from_numpy(pcu.load_mesh_v(landmarks_file))
   
    raise ValueError('Invalid file extension. Supported extensions are .pp, .csv, and .ply.')