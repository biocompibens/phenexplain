import os
import sys
import json
import click
import numpy as np
from glob import glob


@click.command()
@click.option('-s', '--stylegan-path', default='stylegan2-ada-pytorch', show_default=True,
              help="Location of the SyleGAN repository.")
@click.option('-d', '--dataset-destination', default='',
              help="If provided, will output the dataset zip and JSON files there instead of using PATH.")
@click.option('-o', '--options', help="Additional options for StyleGAN's dataset-tool")
@click.argument('path', type=click.Path(exists=True))
def main(path, stylegan_path, dataset_destination, options):
    """
    PATH: dataset root directory, containing one subdirectory per class.
    """
    mydir = os.path.dirname(os.path.abspath(__file__))
    
    print('Creating JSON file containing the classes')
    os.chdir(path)
    
    file_list = glob(os.path.join('**', '*.png')) + glob(os.path.join('**', '*.jpg'))
    
    file_class = np.array([[f, f.split('/')[-2]] for f in file_list])
    clazz, index = np.unique(file_class[:,1], return_inverse=True)
    class_index = list(zip(clazz, range(clazz.shape[0])))
    dataset_dict = {'labels' : [[a,int(b)] for (a,b) in zip(file_class[:,0], index)],
                    'class_index': class_index}
    json_string = json.dumps(dataset_dict)
    if dataset_destination == '':
            dataset_destination = os.path.join(path, 'dataset.json')
        else:
            dataset_destination = os.path.join(mydir, dataset_destination, 'dataset.json')
    with open(dataset_destination, 'w') as outfile:
        outfile.write(json_string)

    print("Calling StyleGAN's dataset_tool")

    if stylegan_path[0] == '/':
        dataset_tool_path = os.path.join(stylegan_path, 'dataset_tool.py')
    else:
        dataset_tool_path = os.path.join(mydir, stylegan_path, 'dataset_tool.py')
    if os.path.exists(dataset_tool_path):
        if dataset_destination == '':
            dataset_destination = os.path.join(path, 'dataset.zip')
        else:
            dataset_destination = os.path.join(mydir, dataset_destination, 'dataset.zip')
        cmd = "python {} --source {} --dest {} {}".format(dataset_tool_path,
                                                          path, dataset_destination,
                                                          options)
        print(cmd)
        os.system(cmd)
    else:
        print(dataset_tool_path)
        print("StyleGAN dataset_tool.py not found", file=sys.stderr)
        sys.exit(1)

    print("Done. Your dataset is in {}.".format(dataset_destination))
    print("You can train a StyleGAN model with the following command: ")
    cmd = os.path.join(stylegan_path, 'train.py')
    print("python {} --data {} --outdir runs --mirror 1 --cond 1".format(cmd, dataset_destination))


if __name__ == '__main__':
    main()
