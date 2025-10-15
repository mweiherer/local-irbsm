# Learning Neural Parametric 3D Breast Shape Models for Metrical Surface Reconstruction From Monocular RGB Videos

**[Paper]() | [Project page](https://rbsm.re-mic.de/local-implicit/)** 

[Maximilian Weiherer](https://mweiherer.github.io)$^{1,2}$, [Antonia von Riedheim]()$^3$, [Vanessa Brébant](https://www.linkedin.com/in/vanessa-brebant-0a391843/)$^3$, [Bernhard Egger](https://eggerbernhard.ch)$^{1*}$, [Christoph Palm](https://re-mic.de/en/head/)$^{2*}$\
$^1$ Friedrich-Alexander-Universität Erlangen-Nürnberg\
$^2$ OTH Regensburg\
$^3$ University Hospital Regensburg

Official implementation of the paper "Learning Neural Parametric 3D Breast Shape Models for Metrical Surface Reconstruction From Monocular RGB Videos".

This repository contains code for the local implicit Regensburg Breast Shape Model (liRBSM).
Along with the inference code (sampling from our model and reconstructing point clouds) and code that has been used to train our model, we also fully open-source the proposed 3D breast surface reconstruction pipeline.

**We also provide an easy-to-use graphical user interface for our 3D reconstruction pipeline that runs on macOS and Windows and (optionally) without a graphics card. Download it [here](https://rbsm.re-mic.de/local-implicit/)!**

Abstract:
*We present a neural parametric 3D breast shape model and, based on this model, introduce a low-cost and accessible 3D surface reconstruction pipeline capable of recovering accurate breast geometry from a monocular RGB video. In contrast to widely used, commercially available yet prohibitively expensive 3D breast scanning solutions and existing low-cost alternatives, our method requires neither specialized hardware nor proprietary software and can be used with any device that is able to record RGB videos. The key building blocks of our pipeline are a state-of-the-art, off-the-shelf Structure-from-motion pipeline, paired with a parametric breast model for robust and metrically correct surface reconstruction. Our model, similarly to the recently proposed implicit Regensburg Breast Shape Model (iRBSM), leverages implicit neural representations to model breast shapes. However, unlike the iRBSM, which employs a single global neural signed distance function (SDF), our approach---inspired by recent state-of-the-art face models---decomposes the implicit breast domain into multiple smaller regions, each represented by a local neural SDF anchored at anatomical landmark positions. When incorporated into our surface reconstruction pipeline, the proposed model, dubbed liRBSM (short for localized iRBSM), significantly outperforms the iRBSM in terms of reconstruction quality, yielding more detailed surface reconstruction than its global counterpart. Overall, we find that the introduced pipeline is able to recover high-quality 3D breast geometry within an error margin of less than 2 mm. Our method is fast (requires less than six minutes), fully transparent and open-source, and---together with the model---publicly available.*

## Table of Contents
- [Setup](#setup)
- [Usage](#usage)
  - [Randomly Sampling From the Model](#randomly-sampling-from-the-model)
  - [Fitting the Model to a Point Cloud](#fitting-the-model-to-a-point-cloud)
    - [Providing Landmarks](#providing-landmarks)
- [Surface Reconstruction From Video](#surface-reconstruction-from-video)
  - [Setup](#setup)
  - [Running the Pipeline](#running-the-pipeline)
- [Training Our Model on Your Own Data](#training-our-model-on-your-own-data)
- [Citation](#citation)

## Setup
We're using Python 3.9, PyTorch 2.0.1, and CUDA 11.7.
To install all dependencies within a conda environment, simply run: 
```
conda env create -f environment.yaml
conda activate local-irbsm
```
This may take a while.

You can download the liRBSM [here](https://rbsm.re-mic.de/local-implicit/).
After downloading, make sure to place the `.pth` file in the `./checkpoints` folder. 

## Usage

### Randomly Sampling From the Model
To produce random samples from the liRBSM, use
```
python sample.py <number-of-samples>
```
This generates `<number-of-samples>` random breast shapes which are saved as `.ply` files.

Optional arguments:
- `--output_dir`: The output directory where to save samples. Default: `./`.
- `--device`: The device the model should run on. Default: `cuda`.
- `--voxel_resolution`: The resolution of the voxel grid on which the implicit model is evaluated. Default: `256`.
- `--chunk_size`: The size of the chunks the voxel grid should be split into. Default: `100_000`. If you have a small GPU with only little VRAM, lower this number.

### Fitting the Model to a Point Cloud
To reconstruct a point cloud using the liRBSM, type
```
python reconstruct.py <path-to-point-cloud>
```
Optional arguments:
- `--landmarks`: Path to a file containing landmarks. Please see details below.
- `--output_dir`: The output directory where to save the reconstruction. Default: `./`.
- `--latent_weight`: The regularization weight that penalizes the L2 norm of the latent code. Default: `0.01`. If you have noisy inputs, increase this number.
- `--anchor_weight`: The weight of the anchor loss that penalizes the deviation from the predicted anchors and supplied landmarks. If negative, no anchor loss is applied. Default `-1`.
- `--device`: The device the model should run on. Default: `cuda`.
- `--voxel_resolution`: The resolution of the voxel grid on which the implicit model is evaluated. Default: `256`.
- `--chunk_size`: The size of the chunks the voxel grid should be split into. Default: `100_000`. If you have a small GPU with only little VRAM, lower this number.

If you want a metrical (real-world scale) reconstruction, add `--metrical`.

#### Providing Landmarks 
Whenever the orientation of the given point cloud differs significantly from the model's orientation, you should first *roughly* align both coordinate systems (this is necessary because our model is not invariant to rigid transformations).
The easiest way to achieve this is by providing certain landmark positions. 
These points can then be used to rigidly align the given point cloud to the model.
Please provide the following landmarks in *exactly* this order:
1. Sternal notch
2. Belly button
3. Left nipple (from the patient's perspective; so it's actually *right* from your perspective!)
4. Right nipple (from the patient's perspective; so it's actually *left* from your perspective!)
5. Left coracoid process (from the patient's perspective; so it's actually *right* from your perspective!)
6. Right coracoid process (from the patient's perspective; so it's actually *left* from your perspective!)

We recommend using [MeshLab](https://www.meshlab.net)'s PickPoints (PP) tool, which allows you to export selected point positions as XML file with `.pp` extension. 
You can directly pass this file to `reconstruct.py`.
Alternatively, you can use your favorite point picker tool and pass points as comma-separated `.csv` file.
Lastly, we also provide a simple application to interactively select points, just run
```
python scripts/pick_landmarks.py <path-to-point-cloud>
```
Please also see the README file in `./scripts`.

## Surface Reconstruction From Video

### Setup
Our reconstruction pipeline heavily relies on [VGGSfM](https://vggsfm.github.io), a state-of-the-art differentiable Structure-from-Motion pipeline.
To set it up, run
```
source scripts/setup_vggsfm.sh
```
This downloads the official GitHub repository into the `./extern` folder and installs all its dependencies.

**If your GPU has less than 32 GB of VRAM you'd need to adapt the following two hardcoded hyperparameters to avoid an out-of-memory error while running VGGSfM: `max_points_num = 163840` in predict_tracks and `max_tri_points_num = 819200` in triangulate_tracks. Please see [here](https://github.com/facebookresearch/vggsfm?tab=readme-ov-file#10-faqs) for more information.**

### Running the Pipeline
To run our pipeline using default parameters, simply type:
```
sh video_to_3d/run_pipeline.sh <path-to-video>
```
This will automatically (1) extract 30 frames from the input video, (2) run SfM on the extracted frames, (3) opens a window which prompts you to pick the six landmarks in a single image, (4) aligns the SfM-generated point cloud to our model's mean shape and prunes away points in the background, and finally (5) reconstructs a metrically correct surface mesh by fitting our model to the aligned and pruned point cloud.
The whole pipeline runs in about six minutes on a single NVIDIA RTX A5000 with 20 GB of VRAM. 

Important mouse and key controls for interactive landmark selection: 

Left click to add a new landmark. Right click to delete last added landmark. Press `Enter` to save and continue. Hit `q` or `ESC` to quit without saving (this will exit the whole pipeline!). Edit a selected landmark position by hovering over the landmark and drag and drop to desired position.

#### Step-by-Step Execution
If you prefer to run our pipeline step-by-step with full control over its parameters, follow the instructions below.

<details>
<summary>Click to expand</summary>

##### 1. Extract Frames
First, extract frames using
```
python video_to_3d/extract_frames.py <path-to-video> <base-output-dir>
```
This extracts 30 frames into `<base-output-dir>/images` according to the selection strategy explained in the paper.
If you want to extract frames uniformly in time regardless of image quality, add `--uniform`.
If you want to extract an arbitrary number of images, set `--num_frames` accordingly.

##### 2. Run SfM
Next, run SfM on the extracted frames by typing
```
python video_to_3d/run_sfm.py <base-output-dir>
```
Depending on the available hardware, this may take a while (around six minutes for 30 frames on a single NVIDIA RTX A5000 with 20 GB of VRAM).

Optional arguments (to VGGSfM, see also [here](https://github.com/facebookresearch/vggsfm)):
- `--camera_type`: Camera model. Can be either `SIMPLE_PINHOLE` or `SIMPLE_RADIAL`. Default: `SIMPLE_RADIAL`.
- `--shared_camera`: Set this flag if you want to use shared camera intrinsics across all images. Usually valid for images extracted from a video. Default: `True`.
- `--query_method`: Query point method. Choose `sp`, `sift`, `aliked`, or any combination of them, like `sp+sift`. Default: `sp+sift+aliked`.
- `--query_frame_num`: Number of query frames. Default: `3`.
- `--max_query_pts`: Maximum number of query points per frame. Default: `8_192`.
- `--num_ba_iterations`: Number of bundle adjustment (BA) iterations: Default: `3`.
- `--predict_dense_depth`: Whether to predict dense depth maps. **Works only if you follow the instructions [here](https://github.com/facebookresearch/vggsfm?tab=readme-ov-file#6-dense-depth-prediction-beta) before running the pipeline**. Default: `False`.
- `--extra_pt_pixel_interval`: Pixel interval used to predict denser point clouds. Note that extra points are not optimized during BA. If negative, don't predict extra points. Default: `-1`.
- `--concat_extra_points`: If set, concatenates extra points (if any) with existing points. Default: `False`.

##### 3. Select Landmarks
Select the following six landmarks in a frontal-facing image in *exactly* this order:
1. Sternal notch
2. Belly button
3. Left nipple (from the patient's perspective; so it's actually *right* from your perspective!)
4. Right nipple (from the patient's perspective; so it's actually *left* from your perspective!)
5. Left coracoid process (from the patient's perspective; so it's actually *right* from your perspective!)
6. Right coracoid process (from the patient's perspective; so it's actually *left* from your perspective!)

To do so, you can either employ your favorite 2D landmarking tool or use our supplied 2D Landmark Picker by running
```
python video_to_3d/pick_2d_landmarks.py <base-output-dir>/images
```
which will automatically select a frontal-facing image and prompts you to annotate the six landmarks.

If you use a custom landmarking tool, make sure to follow the following convention for the landmarks file.
The file is required to be a text file that contains the comma-separated pixel coordinates of the selected landmarks, one row for each landmark position:
```
x_1,y_1
x_2,y_2
...
x_6,y_6
```
The naming convention for this file is: `<camera-id>_landmarks.txt`, where `<camera-id>` is the ID of the image in which landmarks have been selected (e.g., `005`).

##### 4. Align and Prune
Next, based on the previously annotated 2D landmarks, run
```
python video_to_3d/align_and_prune.py <base-output-dir> <path-to-landmarks>
```

Optional arguments:
- `--pruning_dist_threshold`: Maximum distance beyond which points farther from our model's mean shape are removed. Default: `0.2` (about 12 cm in real-world scale). 
- `--use_depth_if_available`: Whether to use predicted dense depth maps for back-projecting 2D landmarks. Works only if `--predict_dense_depth` is set during SfM (see above). Default: `False`.
- `--debug`: If set, outputs intermediate results. Default: `False`.

##### 5. Reconstruct Surface
Finally, run
```
python reconstruct.py <base-output-dir>/aligned_pt_pruned.ply --landmarks <base-output-dir>/aligned_landmarks_3d.ply --output_dir <base-output-dir> --metrical --latent_weight 0.1 --anchor_weight 0.1
```
to reconstruct a metrical surface mesh using the default parameters.

Optional arguments:
See above.
</details>

## Training Our Model on Your Own Data
To train our model on your own data, you'd need a dataset of watertight 3D breast scans with corresponding landmarks (the same as described above), clicked on each scan.
We recommend using the supplied landmarking tool to select landmarks on meshes; see the `./scripts` folder for further information.
Once selected, please save landmark files under `<path-to-your-scans>/landmarks`.

### Preprocess Your Data
First, you need to bring your data into the file format we're using (we're expecting training data to be stored in `.hdf5` files). 
The following script does that for you; it first scales raw meshes into the unit cube, and then optionally discards inner structures. 
Finally, it produces a ready-to-use `.hdf5` file that you can later plug into our training pipeline.
Simply type
```
python scripts/preprocess_dataset.py <path-to-your-scans>
```
Optional arguments:
- `--output`: The name of the output `.hdf5` file. Default: `./dataset.hdf5`.
- `--padding`: The padding to add to the unit cube. Default: `0.1`.

After preprocessing, make sure to place the resulting `.hdf5` file(s) in the `./data` folder.

Secondly, our model's architecture requires average anchor positions, computed as mean landmark positions over the dataset.
You can do so by calling
```
python scripts/compute_average_anchors.py data/dataset.hdf5
```
Place the resulting file in the `./artifacts` folder.

### Train the Model
Finally, to train the model, type
```
python train.py configs/local_ensembled_deep_sdf_576.yaml
```
For training, you'd also need a wandb account. 
To log in to your account, simply type `wandb login` and follow the instructions.

## Citation
If you use the liRBSM or our 3D surface reconstruction pipeline, please cite
```bibtex
@misc{weiherer2025lirbsm,
    title={Learning Neural Parametric 3D Breast Shape Models for Metrical Surface Reconstruction From Monocular RGB Videos},
    author={Weiherer, Maximilian and von Riedheim, Antonia and Brébant, Vanessa and Egger, Bernhard and Palm, Christoph},
    archivePrefix={arXiv},
    eprint={},
    year={2025}
}
```
Also, in case you have any questions, feel free to contact Maximilian Weiherer, Bernhard Egger, or Christoph Palm.
