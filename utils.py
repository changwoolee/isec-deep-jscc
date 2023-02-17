import os
import torch
import numpy as np
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.autograd as autograd

import json


def save_to_json(stats, args):
    params = {
                'nc': args.num_channels,
                'lr': args.lr,
                'stats': stats
                }

    if os.path.exists(args.json_file_path):
        with open(args.json_file_path) as f:
            data = json.load(f)
    else:
        data = {}

    ind = "snr_train_{:.1f}_snr_{:.1f}_nc_{}".format(args.snr_train, args.snr, args.num_channels)
    data.update({ind: params})
    with open(args.json_file_path, "w") as f:
        json.dump(data, f, indent=4)



def load_images(files):
    images = []
    for f in sorted(files):
        images.append(read_image(f, ImageReadMode.RGB))
    return images

def save_image_collections(img_prefix, num_images, output_dir, nrow=4):
    print("Saving image collections...")
    target_files = [os.path.join(output_dir, img_prefix, "targets","files", "targets{:04d}.png".format(f_ind)) for f_ind in range(num_images)]
    orig_files = [os.path.join(output_dir, img_prefix, "orig","files", "orig{:04d}.png".format(f_ind)) for f_ind in range(num_images)]
    updated_files = [os.path.join(output_dir, img_prefix, "updated","files", "updated{:04d}.png".format(f_ind)) for f_ind in range(num_images)]
    target_ims = make_grid(load_images(target_files), nrow=nrow).permute(1,2,0).numpy()
    orig_ims = make_grid(load_images(orig_files), nrow=nrow).permute(1,2,0).numpy()
    updated_ims = make_grid(load_images(updated_files), nrow=nrow).permute(1,2,0).numpy()
    plt.imsave(os.path.join(output_dir, img_prefix, "targets_collection.png"), target_ims) 
    plt.imsave(os.path.join(output_dir, img_prefix, "orig_collection.png"), orig_ims) 
    plt.imsave(os.path.join(output_dir, img_prefix, "updated_collection.png"), updated_ims) 



def tensor2im(x, min_val=0., max_val=1., return_type='numpy'):
    with torch.no_grad():
        #x = torch.clamp(x, min_val, max_val)
        x = (x - min_val) / (max_val - min_val) * 255.
    if return_type == 'numpy':
        return x.permute(0,2,3,1).clone().detach().cpu().numpy().round().astype('uint8')
    elif return_type == 'torch':
        return x.round()

def batch2im(x, n_row, n_col, min_val=-1., max_val=1., im_height=32, im_width=32):
    x = tensor2im(x, min_val, max_val)
    
    img = np.zeros((im_height*n_row, im_width*n_col, x.shape[-1]), dtype=np.uint8)
    
    for r in range(n_row):
        for c in range(n_col):
            img[r*im_height:(r+1)*im_height, c*im_width:(c+1)*im_width, :] = x[r*n_col + c, :, :, :]
    return img.squeeze()


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self, reduction='mean'):
        self.name = "PSNR"
        self.reduction = reduction

    #@staticmethod
    def __call__(self, img1, img2, min_val=0., max_val=1., mean_weight=1., cuda=False, offset=0):
        if cuda:
            with torch.no_grad():
                img1 = torch.clamp(img1, min_val, max_val)
                img1 = (img1 - min_val) / (max_val - min_val) * 255.
                img2 = torch.clamp(img2, min_val, max_val)
                img2 = (img2 - min_val) / (max_val - min_val) * 255.
                img1, img2 = img1.round(), img2.round()
                mse = torch.mean((img1 - img2)**2, dim=(1,2,3))
                psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
                psnr = psnr.detach().cpu().numpy()
        else:
            img1, img2 = tensor2im(img1, min_val, max_val).astype('float64'), tensor2im(img2, min_val, max_val).astype('float64')
            if offset > 0:
                img1 = img1[:,offset:-offset,offset:-offset,:]
                img2 = img2[:,offset:-offset,offset:-offset,:]
            mse = np.mean((img1 - img2)**2, axis=(1,2,3))
            psnr = 10 * np.log10(255.0**2 / mse)#np.sqrt(mse))
        if self.reduction == 'mean':
            return np.mean(psnr)*mean_weight
        elif self.reduction == 'sum':
            return np.sum(psnr)
        elif self.reduction == 'none':
            return psnr
        else:
            raise NotImplementedError()

