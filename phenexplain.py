import os
import sys
import click

from src.utils import load_classes, sample_grid
import src.phenexplain as phenexplain

@click.command()
@click.option('-m', '--method', default='3', show_default=True,
              type=click.Choice(['1', '2', '3']),
              help="Method for the interpolation. See paper for the details.")
@click.option('-M', '--mode', default='grid', show_default=True,
              type=click.Choice(['grid', 'single']),
              help="grid mode generates a grid of images. single mode generates "+
              "individual images, but only works for two targets at a time.")
@click.option('-l', '--list-classes', is_flag=True,
              help="List the available classes with their indices and exit.")
@click.option('-t', '--targets', default='0,1', show_default=True,
              help="Indices of the targets, separated by a comma. You can find "+
              "the indices/labels associations with the -l option")
@click.option('-n', '--samples', default=5, show_default=True,
              help="Number of sample vectors to use")
@click.option('-s', '--steps', default=8, show_default=True,
              help="Number of steps for the interpolation/animation.")
@click.option('-g', '--gpu', default='cuda:0', show_default=True,
              help="Which GPU(s) to use. Can be anything PyTorch understands, even 'cpu'")
@click.option('-o', '--out', default='out.avi', show_default=True,
              help="Output filename. If it doesn't end in .avi, .png or .gif, a "+
              "directory of that name will be created, containing separate pictures for each frame.")
@click.option('-w', '--weights', help="Path to the weights of a trained model [.pkl]")
@click.option('-p', '--stylegan-path', default='stylegan2-ada-pytorch', show_default=True,
              help="Location of the SyleGAN repository.")
@click.option('-R', '--real-images', is_flag=True,
              help="Extract a sample of images from the dataset for easier viewing.")
@click.argument('dataset')
def main(dataset, weights, method, list_classes, targets, samples, steps,
         gpu, out, mode, stylegan_path, real_images):
    # add stylegan's path to our include path in order
    # to be able to import dnnlib and torch_utils
    if os.path.exists(stylegan_path) and os.path.isdir(stylegan_path):
        sys.path.append(stylegan_path)
    else:
        print("StylGAN not found. Please use the --stylegan-path option.",
              file=sys.stderr)
        sys.exit(-2)

    labels, classes = load_classes(dataset)
    if list_classes:
        print("The following classes are available:")
        print(" index\tname")
        for k in range(len(labels)):
            print(" {}\t{}".format(k, labels[k]))
        sys.exit(0)

    method = int(method)
    targets = list(map(int, targets.split(',')))

    if real_images:
        if not out.endswith('.png'):
            print('--real-images option requires a PNG output',
                  file=sys.stderr)
            sys.exit(-1)
            
        sample_grid(dataset, targets, out, samples)
        print("Sample of real images generated in {}".format(out))
        sys.exit(0)

    if mode == 'grid':
        phenexplain.grid(weights, classes, targets, out,
                         samples=samples, method=method,
                         steps=steps, gpu=gpu)
    elif mode == 'single':
        if len(targets) != 2:
            print("This mode only works for two targets", file=sys.stderr)
            sys.exit(-1)
        phenexplain.single(weights, classes, targets, out,
                           samples=samples, method=method,
                           steps=steps, gpu=gpu)




if __name__ == '__main__':
    main()
