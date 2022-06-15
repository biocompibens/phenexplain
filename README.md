# Phenexplain &mdash; Official PyTorch implementation

## Requirements

* Python libraries: pip install click tqdm ninja torch torchvision opencv-pthon mako
* The official[ StyleGAN2 repository]( https://github.com/NVlabs/stylegan2-ada-pytorch/) should be cloned somewhere, preferably inside phenexplain's directory. If installed elsewhere, be sure to use the --stylegan-path option.

## Getting started

### Preparing a dataset

* Your dataset should be a directory containing a subdirectory for each class:
    * DATASET/class1
    * DATASET/class2
	* ...
* Call `python make_datasetjson.py DATASET` to prepare the dataset. This will create a JSON file associating images with their classes, and call StyleGAN2's dataset_tool to create a ZIP file containing everything. If needed, you can pass additional options to StyleGAN's tools using the -o option, such as `-o "--width 128 --height 128"`.

### Training StyleGAN2

Once the dataset has been prepared as a DATASET.zip file, you can train StyleGAN using a command line like:

`python [stylegan-path]/train.py --data DATASET.zip --outdir runs --mirror 1 --cond 1`

### Using Phenexplain on a trained network

The preceding command should produce a subdirectory in the `runs` directory, containing backups of the network called `network-snapshot-xxx.pkl`. Use one of these snapshots with Phenexplain to explore transitions between classes.

* Get the indices of the available classes:  
`python phenexplain.py DATASET -w snapshot.pkl -l`
* Generate videos of transitions between class 0 and class 1 in 20 steps, for 5 examples:  
`python phenexplain.py DATASET -w snapshot.pkl -t 0,1 -n 5 -s 20 -o video.avi`


## Licence

## Citation
