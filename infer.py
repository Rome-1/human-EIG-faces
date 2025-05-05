import argparse
import os
import glob
import time
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets

from PIL import Image
import h5py

from models.eig.networks.network import EIG
from models.eig.networks.network_classifier import EIG_classifier
from models.id.networks.network import VGG

from utils import config

# copy of infer.py from EIG_Faces at EIG_Faces/infer_render_using_eig/infer.py

CONFIG = config.Config()

def load_image(image, size):
    image = image.resize(size)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = np.moveaxis(image, 2, 0)
    image = image.astype(np.float32)
    return image

models_d = {

    'eig' : EIG(),
    'vgg' : VGG(500),

}

image_sizes = {

    'eig' : (227, 227),
    'vgg' : (224, 224),
}


def main():
    """
    Example usage: 
    
    cd EIG_faces # from EIG-faces dir
    python -m infer_render_using_eig.infer --imagefolder ../osfstorage-archive/stimuli --segment --model eig
    python -m infer_render_using_eig.infer --imagefolder ../osfstorage-archive/stimuli --segment --model vgg
    """

    parser = argparse.ArgumentParser(description='Predictions of the models on the neural image test sets')
    parser.add_argument('--imagefolder',  type=str, default='./demo_images/',
                        help='Folder containing the input images.')
    parser.add_argument('--segment', help='whether to initially perform segmentation on the input images.',
                       action='store_true')
    parser.add_argument('--addoffset', help='whether to add offset away from the image boundary to the output of the segmentation step.',
                       action='store_true')
    parser.add_argument('--resume', type = str, default='', 
                        help='Where is the model weights stored if other than where the configuration file specifies.')
    parser.add_argument('--model', type = str, default='eig', 
                        help='Which model to use. Default: eig')

    global args
    args = parser.parse_args()

    print("=> Construct the model...")
    model = models_d[args.model]
    model.cuda()

    resume_path = args.resume
    if resume_path == '':
        resume_path = os.path.join(CONFIG['PATHS', 'checkpoints'], args.model, 'checkpoint_bfm.pth.tar')
    checkpoint = torch.load(resume_path)
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(resume_path, checkpoint['epoch']))

    # test
    if not os.path.exists('./output'):
        os.mkdir('./output')
    outfile = os.path.join('./output', f'infer_output_{args.model}.hdf5')

    test(model, outfile, args.model)

def test(model, outfile, model_type):

    dtype = torch.FloatTensor

    path = args.imagefolder

    filenames = sorted(
        chain(
            glob.glob(os.path.join(path, '*.jpg')),
            glob.glob(os.path.join(path, '*.png'))
        )
    )
    N = len(filenames)


    latents = []
    attended = []
    tcl = []
    ffcl = []
    for i in range(N):
        fname = filenames[i]
        # print(fname)

        v = Image.open(fname)
        image = load_image(v, image_sizes[model_type])
        image = torch.from_numpy(image).type(dtype).cuda()
        image = image.unsqueeze(0)

        if model_type == 'eig':
            att, out_1, out_2, latent = model(image, segment=args.segment, add_offset=args.addoffset and args.segment, test=True)
        elif model_type == 'vgg':
            out_1, out_2, out_3 = model(image, test=True) # f3, f4, f5
            latent = out_3 # f5

        tcl.append(out_1.detach()[0].cpu().numpy().flatten())
        ffcl.append(out_2.detach()[0].cpu().numpy().flatten())
        latents.append(latent.detach()[0].cpu().numpy().flatten())
        if model_type == 'eig':
            attended.append(att.detach()[0].cpu().numpy().flatten())

    f = h5py.File(outfile, 'w')
    f.create_dataset('number_of_layers', data=np.array([2]))
    f.create_dataset('latents' if model_type == 'eig' else 'f5', data=np.array(latents))
    f.create_dataset('TCL' if model_type == 'eig' else 'f3', data=np.array(tcl))
    f.create_dataset('FFCL' if model_type == 'eig' else 'f4', data=np.array(ffcl))
    f.create_dataset('Att', data=np.array(attended)) # empty list if EIG

    asciiList = [n.split('/')[-1][:-4].encode("ascii", "ignore") for n in filenames]
    f.create_dataset('filenames', (len(asciiList), 1), 'S10', data=asciiList)
    f.close()
    print(f"Outputs (N={len(latents)}) saved to {outfile}")


if __name__ == '__main__':
    main()


