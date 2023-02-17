import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import tensor2im, PSNR, save_image_collections, save_to_json

from pytorch_msssim import ms_ssim, ssim
from pytorch_fid.fid_score import calculate_fid_given_paths
import lpips


import loader
import config
import models.autoencoders as ae
from models.bfcnn import BF_CNN

parser = config.get_common_parser()

parser.add_argument('--jscc_model_path', '-jmp', type=str, default=None, help='model path')
parser.add_argument('--bfcnn_model_path', '-bmp', type=str, default=None, help='model path')
parser.add_argument('--loss_type', type=str, default='l2', help='l2|l1 default=l2')


parser.add_argument('--num_iter', '-ni', type=int, default=100, help="Number of SEC iterations.")
parser.add_argument('--save_images', action='store_true', help='Save output images')
parser.add_argument('--max_batch', '-mb', type=int, default=100, help='Number of maximum batch')
parser.add_argument('--img_prefix',type=str, default="", help='Saved images prefix')

parser.add_argument('--save_json', action='store_true', help='Save JSON file')
parser.add_argument('--json_file_path',type=str, default="", help='path for JSON file')
parser.add_argument('--snr_train', '-st', type=int, default=0, help="Trained SNR")

parser.add_argument('--output_dir', '-od', type=str, default='./outputs', help='output directory')
parser.add_argument('--no_denoiser', action='store_true', help='Do not use denoiser')
parser.add_argument('--alpha', '-al', type=float, default=0.0, help='modified MAP parameter')
parser.add_argument('--delta', '-de', type=float, default=1.5, help='delta')
parser.add_argument('--stop_ratio', '-sr', type=float, default=0.0, help='Stopping Criterion')
parser.add_argument('--num_experiment', '-ne', type=int, default=10, help="Number of experiment")
parser.add_argument('--distribution', '-dist', type=str, default='Gaussian', help='Noise distribution. |Gaussian (default)|Laplace|')


args = parser.parse_args()
if args.debug:
    args.print_freq = 1
    args.display_freq = 1

dev = "cuda:{}".format(args.gpu) if args.gpu>=0 else "cpu"

device = torch.device(dev)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

test_dataloader = loader.get_test_dataloader(args)
image_range=(-1, 1)
print(len(test_dataloader))

if args.loss_type == 'l2':
    criterion = nn.MSELoss(reduction='sum')
elif args.loss_type == 'l1':
    criterion = nn.L1Loss(reduction='sum')
else:
    raise NotImplementedError()

loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

def print_update(i, 
                 t, 
                 scaled_denoiser_sqnorm, 
                 obj,
                 B,
                 outputs, 
                 inputs, 
                 psnr_orig, 
                 msssim_orig,
                 avg_psnr,
                 avg_msssim,
                 lpips_orig=None,
                 avg_lpips=None,
                 last_iter=False):
    with torch.no_grad():
        lpips = loss_fn_vgg(outputs, inputs).mean()
    inputs_255 = tensor2im(inputs, *image_range, 'torch')
    psnr = PSNR(reduction='sum')(outputs, inputs, *image_range, offset=0)
    try:
        msssim = ms_ssim(tensor2im(outputs, *image_range,'torch'), 
        inputs_255, data_range=255, size_average=True)
        label = "MS-SSIM"
    except AssertionError:
        msssim = ssim(tensor2im(outputs, *image_range, 'torch'), 
        inputs_255, data_range=255, size_average=True)
        label = "SSIM"
        
    if last_iter:
        avg_psnr += psnr
        avg_msssim += msssim 
        avg_lpips += lpips
    message = "[{:4d}, {:4d}] sigma_t^2: {:.4f} Obj: {:.4f} PSNR: {:.2f} PSNR Orig: {:.2f} {}: {:.4f} {} Orig: {:.4f}".format(
        i+1, t+1, scaled_denoiser_sqnorm.item(), obj.item(), psnr / B, psnr_orig / B, label, msssim, label, msssim_orig)
    message += " LPIPS: {:.4f} LPIPS Orig: {:.4f}".format(lpips, lpips_orig)
    print(message)
    return avg_psnr, avg_msssim, avg_lpips


def print_avg(i, 
              count, 
              avg_psnr,
              avg_psnr_orig,
              avg_msssim,
              avg_msssim_orig,
              avg_lpips,
              avg_lpips_orig):

    stats = {'PSNR': avg_psnr.item()/count, 
             'PSNR Orig': avg_psnr_orig.item()/count, 
             'PSNR Gain': (avg_psnr - avg_psnr_orig).item()/count, 
             'MS SSIM': avg_msssim.item()/i, 
             'MS SSIM Orig': avg_msssim_orig.item()/i, 
             'MS SSIM Gain': (avg_msssim - avg_msssim_orig).item()/i,
             'lpips': avg_lpips.item()/i,
             'lpips Orig': avg_lpips_orig.item()/i,
             'lpips Gain': (-avg_lpips + avg_lpips_orig).item()/i,
             }
    message = '[{:4d}] Average'.format(i)
    for name, val in stats.items():
        message += " - {}: {:.4f}".format(name, val)
    print(message)
    return stats


def test_latent(net, stddev=0., saved_dir=None, writer=None, epoch=0):
    net.eval()
    avg_psnr_orig = 0.
    avg_psnr = 0.
    avg_msssim_orig = 0.
    avg_msssim = 0.
    avg_lpips_orig = 0.
    avg_lpips = 0.
    avg_fid = 0.
    avg_fid_orig = 0.
    count = 0.

    base_var = 10**(-0.1*args.snr_train)
    print(args.distribution)

    if args.distribution == 'Gaussian' or args.distribution == 'Fading':
        dist = torch.distributions.normal.Normal(0.0, stddev)
    elif args.distribution == 'Laplace':
        dist = torch.distributions.laplace.Laplace(0.0, stddev/np.sqrt(2))
    else:
        raise NotImplementedError()


    decoder = net.decoder

    L = args.num_experiment
    psnr_vals = np.zeros(L)
    psnr_orig_vals = np.zeros(L)
    ssim_vals = np.zeros(L)
    ssim_orig_vals = np.zeros(L)
    lpips_vals = np.zeros(L)
    lpips_orig_vals = np.zeros(L)
    fid_vals = np.zeros(L)
    fid_orig_vals = np.zeros(L)
    for e in range(L):
        for i, data in enumerate(test_dataloader):
            if i==args.max_batch:
                break
            with torch.no_grad():
                # Sample Test Image
                inputs = data[0].to(device)
                B,C,H,W = inputs.size()
                # Encode
                codeword = net.encoder(inputs)
                _,zC,zH,zW = codeword.size()
                # Corrupted codeword
                noise = dist.sample(codeword.size()).to(device)
                if args.distribution == 'Fading':
                    h = torch.randn(B, 1, 1, 1)
                    codeword = h * codeword
                y = codeword + noise
                # One-shot Decoding
                outputs = decoder(y)
                

                psnr_orig = PSNR(reduction='sum')(outputs, inputs, *image_range, offset=0)
                inputs_255 = tensor2im(inputs, *image_range, 'torch')
                try:
                    msssim_orig = ms_ssim(tensor2im(outputs, *image_range, 'torch'), 
                        inputs_255, data_range=255, size_average=True)
                    label = "MS-SSIM"
                except AssertionError:
                    msssim_orig = ssim(tensor2im(outputs, *image_range, 'torch'), 
                    inputs_255, data_range=255, size_average=True)
                    label = "SSIM"

                lpips_orig = loss_fn_vgg(outputs, inputs).sum()
         
                avg_psnr_orig += psnr_orig
                avg_msssim_orig += msssim_orig
                avg_lpips_orig += lpips_orig / B
                count += B
                outputs_orig = outputs.clone()

            # Initialize zt with y
            with torch.no_grad():
                init_p = y
            zt = init_p.detach().clone().requires_grad_()

            with torch.no_grad():
                # Logging Purpose, sqnorm of denoiser output
                dt = net.denoiser(zt)
                vart = torch.sum(dt**2, dim=(1,2,3), keepdim=True)/(zC*zH*zW)
                var_scale = stddev**2 / vart.mean()
                vart *= var_scale
                varL = args.stop_ratio * vart.mean().item()

            # Compute delta
            delta = args.delta if (stddev**2/base_var) > 1 else 1.0
            for t in range(args.num_iter):
                with torch.no_grad():
                    # Logging Purpose, sqnorm of denoiser output
                    dt = net.denoiser(zt)
                    vart = torch.sum(dt**2, dim=(1,2,3), keepdim=True)/(zC*zH*zW)
                    vart *= var_scale
                    scaled_denoiser_sqnorm = vart.mean()

                # Evaluate NLL
                z_p = net.encoder(decoder(zt))
                obj = 1/(2*(stddev**2)) * criterion(z_p, y)
                obj.backward()

                with torch.no_grad():
                    # Gradient of NLL
                    zt_grad = -zt.grad
                    # Scale the output of the denoiser (approximate gradient of the log prior)
                    zt_grad += args.alpha * max(0.1, (stddev**2/base_var)**2) * dt
                    lr = args.lr / max(0.1, (stddev**2/base_var)**delta)
                    zt.data = zt + lr * zt_grad
                    zt.grad.zero_()

                    if t % args.print_freq  == args.print_freq - 1 or t == args.num_iter - 1 or t==0:
                        outputs = decoder(zt)
                        avg_psnr, avg_msssim, avg_lpips = print_update(i, 
                                t, scaled_denoiser_sqnorm, obj, B, outputs, inputs, 
                                psnr_orig, msssim_orig, 
                                avg_psnr, avg_msssim, 
                                lpips_orig/B, avg_lpips, last_iter=t==args.num_iter-1)

            
            stats = print_avg(e*min(len(test_dataloader), args.max_batch) + i+1, count, avg_psnr, avg_psnr_orig, avg_msssim, avg_msssim_orig, avg_lpips, avg_lpips_orig)
            psnr_vals[e] = avg_psnr/count
            psnr_orig_vals[e] = avg_psnr_orig/count
            ssim_vals[e] = avg_msssim/(e+1)
            ssim_orig_vals[e] = avg_msssim_orig/(e+1)
            lpips_vals[e] = avg_lpips/(e+1)
            lpips_orig_vals[e] = avg_lpips_orig/(e+1)
            if args.save_images:
                output_dir = args.output_dir
                subdirs = ['targets', 'orig', 'updated']
                for sd in subdirs:
                    if not os.path.exists(os.path.join(output_dir, args.img_prefix, sd, "files")):
                        os.makedirs(os.path.join(output_dir, args.img_prefix, sd, "files"))
                targets = tensor2im(inputs, *image_range)
                orig = tensor2im(outputs_orig, *image_range)
                updated = tensor2im(outputs, *image_range)
                for b in range(B):
                    plt.imsave(os.path.join(output_dir, 
                        args.img_prefix, "targets", "files", "targets{:04d}.png".format(i*args.batch_size + b)), targets[b,:,:,:])
                    plt.imsave(os.path.join(output_dir, args.img_prefix, "orig", "files", "orig{:04d}.png".format(i*args.batch_size + b)), orig[b,:,:,:])
                    plt.imsave(os.path.join(output_dir, args.img_prefix, "updated", "files", "updated{:04d}.png".format(i*args.batch_size + b)), updated[b,:,:,:])

        if args.save_images and args.image_size > 0:
            save_image_collections(args.img_prefix, np.minimum(36, i*args.batch_size), output_dir, nrow=6)

        try:
            fid_orig = calculate_fid_given_paths(
                    ["{}/{}/orig/files/".format(output_dir, args.img_prefix), "{}/{}/targets/files/".format(output_dir, args.img_prefix)],
                    min(args.batch_size, 16),
                    device,
                    2048,
                    3)
            fid_updated = calculate_fid_given_paths(
                    ["{}/{}/updated/files/".format(output_dir, args.img_prefix), "{}/{}/targets/files/".format(output_dir, args.img_prefix)],
                    min(args.batch_size, 16),
                    device,
                    2048,
                    3)

            avg_fid += fid_updated
            avg_fid_orig += fid_orig
            #stats["FID Gain"] += fid_updated - fid_orig
            print("FID: {:.2f}, FID Orig: {:.2f}, FID Gain: {:.2f}".format(fid_updated, fid_orig, fid_orig - fid_updated))

            fid_vals[e] = fid_updated
            fid_orig_vals[e] = fid_orig

        except RuntimeError as e:
            print("FID unavailable")
            print(e)
            pass

    try:
        stats["FID"] = avg_fid / L
        stats["FID Orig"] = avg_fid_orig / L
        stats["FID Gain"] = -stats["FID"] + stats["FID Orig"]
        stats['PSNR Numpy'] = psnr_vals.tolist()
        stats['PSNR Orig Numpy'] = psnr_orig_vals.tolist()
        stats['SSIM Numpy'] = ssim_vals.tolist()
        stats['SSIM Orig Numpy'] = ssim_orig_vals.tolist()
        stats['LPIPS Numpy'] = lpips_vals.tolist()
        stats['LPIPS Orig Numpy'] = lpips_orig_vals.tolist()
        stats['FID Numpy'] = fid_vals.tolist()
        stats['FID Orig Numpy'] = fid_orig_vals.tolist()
    except KeyError:
        pass

    print(stats)
    return stats


def main():

    if 'cifar' in args.dataset:
        Enc = ae.Encoder_CIFAR
        Dec = ae.Decoder_CIFAR
    else:
        Enc = ae.Encoder
        Dec = ae.Decoder
    encoder = Enc(num_out=args.num_channels,
                      num_hidden=args.num_hidden,
                      num_conv_blocks=args.num_conv_blocks,
                      num_residual_blocks=args.num_residual_blocks,
                      normalization=nn.BatchNorm2d,
                      activation=nn.PReLU,
                      power_norm=args.power_norm)
       
    decoder = Dec(num_in=args.num_channels,
                      num_hidden=args.num_hidden,
                      num_conv_blocks=args.num_conv_blocks,
                      num_residual_blocks=args.num_residual_blocks,
                      normalization=nn.BatchNorm2d,
                      activation=nn.PReLU,
                      no_tanh=False)

    net = ae.Generator(encoder, decoder)
    print(args)

    try:
        filepath = args.jscc_model_path
        print("Try loading "+filepath)
        net.load_state_dict(torch.load(filepath, map_location=dev))
    except Exception as e: 
        print(e)
        print("Loading Failed...")
        exit()

    bfcnn = BF_CNN(1, 64, 3, 20, args.num_channels)
    try:
        filepath = args.bfcnn_model_path
        print("Try loading "+filepath)
        bfcnn.load_state_dict(torch.load(filepath, map_location=dev))
    except Exception as e: 
        print(e)
        print("Loading Failed...")
        exit()

    bfcnn.to(device)
    net.denoiser = lambda z_: -bfcnn(z_)
    net.to(device)

    if args.save_json:
        args.img_prefix = os.path.join(args.dataset, 
                "{}_snr_train_{:.1f}_snr_{:.1f}_nc_{}".format(args.img_prefix, 
                                                              args.snr_train,
                                                              args.snr,
                                                              args.num_channels))

    stats = test_latent(net, 10**(-0.05*args.snr))

    if args.save_json:
        save_to_json(stats, args)


if __name__=='__main__':
    main()
