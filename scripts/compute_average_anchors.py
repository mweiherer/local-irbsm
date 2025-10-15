import argparse
import h5py
import torch


def main(args):
    file = h5py.File(args.dataset_file, 'r')

    landmarks = []  
    for i in range(len(file.keys())):
        landmarks.append(torch.from_numpy(file[str(i)]['landmarks'][:]))
    file.close()

    print(f'Number of landmarks: {len(landmarks)}.')

    mean_landmarks = 2.0 * torch.mean(torch.stack(landmarks), dim = 0) # Scale it to [-1, 1]^3.
    torch.save(mean_landmarks, './average_anchors.pt')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('dataset_file', type = str,
                           help = 'Path to the hdf5 dataset file.')
 
    args, _ = argparser.parse_known_args()

    main(args)