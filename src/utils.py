import os
import sys
import json
import numpy as np
import PIL.Image
from io import BytesIO
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from zipfile import ZipFile
import cv2
import torchvision.transforms.functional as TF
from torchvision import transforms, utils


def load_classes(path):
    if path.endswith('.zip'):
        json_path = path.replace('.zip', '.json')
    else:
        json_path = os.path.join(path, 'dataset.json')
    if not os.path.exists(json_path):
        # if we didn't find a JSON file, try to find it in the ZIP file.
        found = False
        if path.endswith('.zip'):
            with ZipFile(path) as myzip:
                with myzip.open('dataset.json') as f:
                    data = json.load(f)
                    if 'class_index' in data.keys():
                        found = True
        if not found:
            print("Could not find the JSON file containing the id/class association",
                  file=sys.stderr)
            sys.exit(1)
    else:
        with open(json_path, 'r') as f:
            data = json.load(f)

    if not 'class_index' in data:
        print("The JSON file we found did not contain the id/class association",
              file=sys.stderr)
        sys.exit(2)
    labels = {}
    for k, v in data['class_index']:
        labels[v] = k
    return labels, np.array(data['class_index'])


# convert one image tensor to numpy [0,255]
def convert_image(im):
    im = im.cpu().numpy()
    im = (im*255).astype('uint8')
    im = np.swapaxes(np.swapaxes(im,0,2),0,1)
    im = im[:,:,::-1]
    return im


def convert(images, labels = None, labels_sequential = False, labels_outside = True):
    height, width = images[0].shape[1:]
    if (labels is not None) and (not labels_sequential):
        thumbsize = width/labels.shape[0]
        fontsize  = int(thumbsize / 11)
    else:
        fontsize = 12
    font = ImageFont.truetype("DejaVuSans.ttf", fontsize)
    output_images = []	
    for j, ti in enumerate(np.arange(len(images))):
        im = convert_image(images[ti])
        if labels is not None:
            if labels_sequential:
                im_pil = Image.fromarray(im)
                d = ImageDraw.Draw(im_pil)
                d.text((10,10), labels[j], font=font, fill='white', anchor="lt")
                im = np.array(im_pil)
            else:
                if labels_outside:
                    im_shape = list(im.shape)
                    im_shape[0] = fontsize * 2
                    im_white = np.ones(im_shape, dtype='uint8') * 255
                    im_pil = Image.fromarray(im_white)
                    d = ImageDraw.Draw(im_pil)
                    for i, txt in enumerate(labels):
                        d.text((thumbsize/2+i*thumbsize,fontsize), txt, font=font, fill='black', anchor="mm")
                    im_white = np.array(im_pil)
                    im = np.vstack((im_white, im))
                else:
                    im_pil = Image.fromarray(im)
                    d = ImageDraw.Draw(im_pil)
                    for i, txt in enumerate(labels):
                        d.text((thumbsize/2+i*thumbsize,10), txt, font=font, fill='white', anchor="mt")
                    im = np.array(im_pil)
        output_images.append(im)
    return output_images


def save_png(image, filename):
    im = image[:,:,::-1]
    Image.fromarray(im).save(filename)


def save_gif(images, filename, fps=5):
    Image.fromarray(images[0]).save(filename, save_all=True,
                                    append_images=[Image.fromarray(im) for im in images[1:]],
                                    duration=1000//fps, loop=0)


def save_video(images, filename, fps=5):
    height, width = images[0].shape[:2]
    if filename.endswith('.avi'):
        fourcc=0
    elif filename.endswith('.mp4v'):
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    else:
        raise Exception('Unknown video format')
    video = cv2.VideoWriter(filename, fourcc, fps, (width,height))
    for im in images:
        video.write(im)
    cv2.destroyAllWindows()
    video.release()


def sample_grid(zipfilename, targets, output, samples=3):
    with ZipFile(zipfilename, mode='r') as zipobj:
        with zipobj.open("dataset.json") as datasetobj:
            jsondata = json.load(datasetobj)
        data = np.array(jsondata['labels'])
        clazz = np.array(jsondata['class_index'])
        imgs = []
        for i in range(samples):
            for idx in targets:
                filename = np.random.choice(data[data[:,1] == str(idx),0],1)[0]
                image_data = zipobj.read(filename)
                fh = BytesIO(image_data)
                imgs.append(TF.to_tensor(Image.open(fh)))
        grid = utils.make_grid(imgs, nrow = len(targets), normalize=True, pad_value=1)
        output_images = convert([grid], labels=clazz[np.array(targets)][:,0])
        save_png(output_images[0], output)
