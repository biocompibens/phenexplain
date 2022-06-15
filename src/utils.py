import os
import sys
import json
import numpy as np
import PIL.Image
from PIL import Image
from PIL import Image, ImageDraw, ImageFont	
import cv2


def load_classes(path):
    if path.endswith('.zip'):
        json_path = path.replace('.zip', '.json')
    else:
        json_path = os.path.join(path, 'dataset.json')
    if not os.path.exists(json_path):
        print("Could not find the JSON file containing the id/class association",
              file=sys.stderr)
        sys.exit(1)
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
