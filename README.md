# Phenexplain &mdash; Official PyTorch implementation

## Requirements

* Python libraries: pip install click tqdm ninja torch torchvision opencv-python mako
* The official[ StyleGAN2 repository]( https://github.com/NVlabs/stylegan2-ada-pytorch/) should be cloned inside phenexplain's directory. If installed elsewhere, be sure to use the --stylegan-path option.

## Using your own dataset

### Preparing the dataset

* Your dataset should be a directory containing a subdirectory for each condition with corresponding images inside:
    * DATASET/condition1
    * DATASET/condition2
    * ...
* Call `python make_datasetjson.py DATASET` to prepare the dataset. This will create a JSON file associating images with their condition, and call StyleGAN2's dataset_tool to create a ZIP file containing the images for training and this json file. If needed, you can pass additional options to StyleGAN's tools using the -o option, such as `-o "--width 128 --height 128"`.

### Training a conditional StyleGAN2

Once the dataset has been prepared as a DATASET.zip file, you can train a conditional StyleGAN2 using the following command line:

`python [stylegan-path]/train.py --data DATASET.zip --outdir runs --mirror 1 --cond 1`

## Using Phenexplain on a trained network

The previous command produces a subdirectory in the `runs` directory of StyleGAN2, containing backups of the network called `network-snapshot-xxx.pkl`. Use one of these snapshots with Phenexplain to explore transitions between classes.

* Get the indices of the available classes:
`python phenexplain.py DATASET -w snapshot.pkl -l`
* Generate videos of transitions between class 0 and class 1 in 20 steps, for 5 examples:  
`python phenexplain.py DATASET -w snapshot.pkl -t 0,1 -n 5 -s 20 -o video.avi`

## Example Dataset (BBBC021)

An example dataset, based on the BBBC021 dataset that contains compounds we used in the paper is available here: [BBBC021_selection.zip](https://phenexplain.bio.ens.psl.eu/datasets/BBBC021_selection.zip). It has already been prepared for training with StyleGAN2, and it contains 72 conditions, each compound_concentration being a separate condition.

Training can be performed with:  
`python stylegan2-ada-pytorch/train.py --data BBBC021_selection.zip --outdir runs --mirror 1 --cond 1`

Alternatively, you can download a set of pretrained weights here:  
(...)

## Licence

This work is released under the MIT licence.

## Citation
