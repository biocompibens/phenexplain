import pickle
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, utils

from src.utils import save_gif, save_png, save_video, convert, convert_image
from src.html import make_simple_index, make_list_index


def single(stylegan_weights, clazz, class_idxs, output=None,
           samples=3, method=3, steps=5, gpu='cuda:0'):
    assert(len(class_idxs)==2)

    if gpu == 'cpu':
        fp32 = True
    else:
        fp32 = False
	
    with open(stylegan_weights, 'rb') as f:
        SG = pickle.load(f)
        G = SG['G_ema'].to(gpu)
	
        id = np.random.randint(1,10000000000,1)[0]
        outputdir = os.path.join(output, '_'.join(clazz[class_idxs,0]), str(id))
        os.makedirs(outputdir)

        class_oh   = F.one_hot(torch.tensor(class_idxs), num_classes=G.c_dim).type(torch.DoubleTensor).to(gpu)

        for k in range(samples):
            Z = torch.randn([1, G.z_dim]).to(gpu)
            # torch.save(Z, os.path.join(outputdir, 'z_%02d.pt' % k))
            W = G.mapping(Z.repeat(class_oh.shape[0], 1), class_oh)

            Wj = []
            if method == 3:
                # straight lines	
                W_start = W[0]
                W_end   = W[1]
			
                Wj = []
                for j in range(steps):	
                    Wj.append(torch.lerp(W_start, W_end, j / (steps-1)))
            elif method == 1:
                E_start = G.mapping.embed(C_start)
                E_end = G.mapping.embed(C_end)

                G.mapping.embed = Identity()
                G.mapping.c_dim = 512

                for j in range(steps):	
                    Ej = torch.lerp(E_start, E_end, j / (steps-1))
                    Wj.append(G.mapping(Z, Ej))
            elif method == 2:
                W_center = []
                for i in range(class_oh.shape[0]):
                    W = G.mapping(torch.randn([100000, G.z_dim]).to(gpu),
                                  class_oh[i].repeat(100000,1))
                    W_center.append(torch.mean(W,axis=0))
                W_center = torch.stack(W_center)

                W_start = G.mapping(Z, C_start)
                W_shift = W_center - W_center[0]

                W_end   = W_start + W_shift.repeat(n_samples,1,1)

                for j in range(steps):	
                    Wj.append(torch.lerp(W_start, W_end, j / (steps-1)))
	    
            Wj = torch.stack(Wj)
	    
            imgs = G.synthesis(Wj, noise_mode='const', force_fp32=fp32)
            _min = torch.min(imgs)
            _max = torch.max(imgs)
		
            for j in range(steps):
                im = convert_image((imgs[j] - _min) / (_max - _min))
                save_png(im, os.path.join(outputdir, '%02d_%06d.png' % (k, j)))
    make_list_index(outputdir, samples, steps)
    print("The results can be seen in {}".format(os.path.join(outputdir, 'index.html')))

    
def grid(stylegan_weights, clazz, class_idxs, output=None,
         samples=3, method=3, steps=5, gpu='cuda:0'):
    if not(output is None) and (output.endswith('.png')):
        steps=2
	
    if gpu == 'cpu':
        fp32 = True
    else:
        fp32 = False

    z = None
    images = []

    with open(stylegan_weights, 'rb') as f:
        SG = pickle.load(f)
        G = SG['G_ema'].to(gpu)

    if z == None:
        if type(samples) is int:
            z = torch.randn([samples, G.z_dim]).to(gpu)
        elif type(samples) is torch.Tensor:
            z = samples.to(gpu)  
        else:
            raise Exception('samples should be an int or a torch tensor')

    n_samples = z.shape[0]

    class_oh   = F.one_hot(torch.tensor(class_idxs), num_classes=G.c_dim).type(torch.DoubleTensor).to(gpu)

    Z = torch.repeat_interleave(z, class_oh.shape[0],dim=0)
    C_start = class_oh[0].repeat(n_samples*class_oh.shape[0], 1)
    C_end   = class_oh.repeat(n_samples, 1)
	
    W_path = []
    
    def append(imgs, Wj):
        images.append(utils.make_grid(imgs, nrow = len(class_idxs),
                                      normalize=True, pad_value=1))
        W_path.append(Wj)

    if method==3:
        W_start = G.mapping(Z, C_start)
        W_end   = G.mapping(Z, C_end)	
        for j in range(steps):	
            Wj = torch.lerp(W_start, W_end, j / (steps-1))
            imgs = G.synthesis(Wj, noise_mode='const', force_fp32=fp32)
            append(imgs, Wj)
    elif method==1:
        E_start = G.mapping.embed(C_start)
        E_end = G.mapping.embed(C_end)

        G.mapping.embed = Identity()
        G.mapping.c_dim = 512

        for j in range(steps):	
            Ej = torch.lerp(E_start, E_end, j / (steps-1))
            Wj = G.mapping(Z, Ej)
            imgs = G.synthesis(Wj, noise_mode='const', force_fp32=fp32)
            append(imgs, Wj)
    elif method==2:
        W_center = []
        for i in range(class_oh.shape[0]):
            W = G.mapping(torch.randn([100000, G.z_dim]).to(gpu),
                          class_oh[i].repeat(100000,1))
            W_center.append(torch.mean(W,axis=0))
        W_center = torch.stack(W_center)

        W_start = G.mapping(Z, C_start)
        W_shift = W_center - W_center[0]

        W_end   = W_start + W_shift.repeat(n_samples,1,1)

        for j in range(steps):	
            Wj = torch.lerp(W_start, W_end, j / (steps-1))
            imgs = G.synthesis(Wj, noise_mode='const', force_fp32=fp32)
            append(imgs, Wj)
    else:
        raise Exception('Invalid method id')

    _min = min([torch.min(imgs) for imgs in images])
    _max = max([torch.max(imgs) for imgs in images])
    for i in range(len(images)):
        images[j] = (images[j] - _min) / (_max - _min)

    labels = clazz[np.array(class_idxs)]
    output_images = convert(images, labels=labels[:,0])

    if not (output is None):
        if output.endswith('.avi') or output.endswith('.mp4v'):
            save_video(output_images, output, fps=10)
            print("Saved in file {}".format(output))
        elif output.endswith('.gif'):
            save_gif(output_images, output, fps=10)
            print("Saved in file {}".format(output))
        elif output.endswith('.png'):
            save_png(output_images[1], output)
            print("Saved in file {}".format(output))
        else:
            id = np.random.randint(1,10000000000,1)[0]
            os.makedirs(os.path.join(output, str(id)))
            paths = []
            for j, im in enumerate(output_images):
                save_png(im, os.path.join(output, str(id), '%02d.png' % j))
                paths.append('%02d.png' % j)
            make_simple_index(os.path.join(output, str(id)), paths)
            print("The results can be seen in {}".format(os.path.join(output, str(id),
                                                                      'index.html')))

    return images, W_path
	

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
