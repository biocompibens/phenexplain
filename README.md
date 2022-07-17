# Phenexplain &mdash; Official PyTorch implementation

![Teaser image](./docs/teaser.png)

## Requirements

* Python libraries: pip install -r requirements.txt  
  (Main requirements are pytorch, opencv-python, mako. StyleGAN2 currently does not work with the latest PyTorch version.)
* The official[ StyleGAN2 repository]( https://github.com/NVlabs/stylegan2-ada-pytorch/) should be cloned inside phenexplain's directory. If installed elsewhere, make sure to use the --stylegan-path option.


## Using Phenexplain on a pretrained network

* Download a subset of 73 conditions we prepared (each compound_concentration being a separate condition) from the BBBC021 dataset: [BBBC021_selection.zip](https://phenexplain.bio.ens.psl.eu/datasets/BBBC021_selection.zip) (5.4G). It contains images cropped around each nucleus and condition annotations.

* Download the weights of a conditional StyleGAN2 pretrained on this dataset: [BBBC021_weights.pkl](https://phenexplain.bio.ens.psl.eu/datasets/BBBC021_weights.pkl) (279M)

* Get the list of condition indices

`python phenexplain.py BBBC021_selection.zip -l`

* The following command will generate a video (an .avi file that Fiji can read) of a grid of 5 examples of translations from DMSO (condition index 0) to taxol at concentration 3 µM/ml (condition index 72), use --gpu=cpu if you don't have a GPU:

`python phenexplain.py BBBC021_selection.zip -w BBBC021_weights.pkl -M grid -s 50 -n 5 -t 0,72 -o synthetic.avi`

* You may also display real images from the dataset file for comparison this way:

`python phenexplain.py BBBC021_selection.zip -M grid -n 5 -t 0,72 -o real.png -R`

* You may take a look on additional options to build grids, save to other output formats etc.:

`python phenexplain.py --help`

## Using Phenexplain on your own dataset (GPU required!)

### Preparing your dataset for training

* Your dataset should be a directory containing a subdirectory for each condition with corresponding images inside:
    * DATASET/condition1
    * DATASET/condition2
    * ...
* Call `python make_datasetjson.py DATASET` to prepare the dataset. This will create a JSON file associating images with their condition, and call StyleGAN2's dataset_tool to create a ZIP file containing the images for training and this json file. If needed, you can pass additional options to StyleGAN's tools using the -o option, such as `-o "--width 128 --height 128"`.

### Training a conditional StyleGAN2

Once the dataset has been prepared as a DATASET.zip file, you can train a conditional StyleGAN2 using the following command line:

`python [stylegan-path]/train.py --data DATASET.zip --outdir runs --mirror 1 --cond 1`

Make sure you trained StyleGAN2 long enough to consistently generate good images. The FID you may observe during training through a tensorboard instance must end up very low. 

### Using Phenexplain on your trained network

The previous command produces a subdirectory in the `runs` directory of StyleGAN2, containing regular backups of the network called `network-snapshot-xxx.pkl`. Use one of the last snapshots with Phenexplain to explore transitions between classes.

* Get the available condition indices:

`python phenexplain.py DATASET.zip -l`

* Generate videos of transitions between condition 0 and condition 1 in 20 steps for 5 sample: 

`python phenexplain.py DATASET.zip -w snapshot.pkl -t 0,1 -n 5 -s 20 -o video.avi`

## Licence

This work is released under the MIT licence.

## Citation
