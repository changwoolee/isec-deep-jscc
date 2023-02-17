import numpy as np
import torch
import torch.nn as nn
import os
import time
import matplotlib.pyplot as plt
from PIL import Image
import json
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import argparse

import config
from loader import get_train_dataloader, get_test_dataloader
from models.autoencoders import Generator, Encoder_CIFAR, Decoder_CIFAR, Encoder, Decoder
from models.bfcnn import BF_CNN
from utils import batch2im, PSNR

parser = config.get_common_parser()
parser = config.get_train_parser(parser)

args = parser.parse_args()
dev = "cuda:{}".format(args.gpu) if args.gpu>=0 else "cpu"
device = torch.device(dev)


np.random.seed(args.seed)
torch.manual_seed(args.seed)

train_dataloader = get_train_dataloader(args)
test_dataloader = get_test_dataloader(args)
   
print(len(train_dataloader))
print(len(test_dataloader))

criterion = nn.MSELoss()

def test(net, bfcnn, stddev=0., saved_dir=None, writer=None, epoch=0):
    net.eval()
    bfcnn.eval()
    avg_psnr = 0.
    avg_loss = 0.

    count = 0.
    batch_count = 0.
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            inputs = data[0].to(device)
            B = inputs.size(0)

            code = net.encoder(inputs)
            noise = torch.randn_like(code)*stddev
            residual = bfcnn(code + noise)
            loss = criterion(residual, noise)
            outputs = net.decoder(code+noise-residual)

            avg_psnr += PSNR(reduction='sum')(outputs, inputs, -1, 1, offset=0)
            avg_loss += loss.item()
            count += inputs.size(0)
            batch_count += 1
            if args.show_outputs:
                plt.imsave('test.png', batch2im(outputs, 8, 8, 
                    im_height=args.image_size, im_width=args.image_size))
                plt.imsave('test_target.png', batch2im(inputs, 8, 8,
                    im_height=args.image_size, im_width=args.image_size))
                break
            
        print('Average PSNR: {:.4f}, Average loss: {:.4f}'.format(avg_psnr/count, avg_loss / batch_count) )
        if writer is not None:
            writer.add_scalar('PSNR/test', avg_psnr/count, epoch+1)


def train(net, bfcnn, optimizer, num_epoch, stddev=0.,
        saved_dir=None, model_name=None, which_epoch=0, writer=None, clip_val=5):
    
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 2, gamma=0.1)
    for epoch in range(which_epoch, num_epoch):  # loop over the dataset multiple times
        start_time = time.time()
        net.eval()
        bfcnn.train()
        running_loss = 0.

        for i, data in enumerate(train_dataloader):
            if args.debug and i==10:
                break
            optimizer.zero_grad()

            inputs = data[0].to(device)
            B = inputs.size(0)
            with torch.no_grad():
                code = net.encoder(inputs)
                noise = torch.randn_like(code)*stddev

            residual = bfcnn(code + noise)
            loss_codeword = criterion(residual, noise)
            loss = loss_codeword 
            loss.backward()
            optimizer.step()

                   
            # print statistics
            running_loss += loss.item()

            if i % args.print_freq == args.print_freq-1 or i == len(train_dataloader)-1:
                with torch.no_grad():
                    code = code[:B,:,:,:]
                    noise = noise[:B,:,:,:]
                    residual = residual[:B,:,:,:]
                    mse_y = criterion(net.decoder(code + noise), inputs)
                    mse_z_star = criterion(net.decoder(code), inputs)
                    mse_dn = criterion(net.decoder(code+noise-residual), inputs)

                log_message = "[{:4d}, {:5d}] loss: {:.5f}, MSE y: {:.5f}, MSE z*: {:.5f}, MSE denoised: {:.5f}, ".format(epoch+1, i+1,
                                               running_loss / (i+1),
                                               mse_y.item(),
                                               mse_z_star.item(),
                                               mse_dn.item()
                                            )
                print(log_message)

        scheduler.step()
        if writer is not None:
            writer.add_scalar("Loss/train", running_loss / (i+1), epoch+1)
        if epoch % args.display_freq == args.display_freq-1:
            with torch.no_grad():
                outputs_y = net.decoder(code+noise)
                outputs_z_star = net.decoder(code)
                outputs_dn = net.decoder(code+noise-residual)
                
                targets = Image.fromarray(batch2im(inputs, 2, 2, 
                    im_height=args.train_image_size, im_width=args.train_image_size))
                targets.save(os.path.join(saved_dir, "e{:03d}_targets.png".format(epoch+1)))
                outputs = Image.fromarray(batch2im(outputs_y, 2, 2,
                    im_height=args.train_image_size, im_width=args.train_image_size))
                outputs.save(os.path.join(saved_dir, "e{:03d}_outputs_y.png".format(epoch+1)))
                outputs = Image.fromarray(batch2im(outputs_z_star, 2, 2,
                    im_height=args.train_image_size, im_width=args.train_image_size))
                outputs.save(os.path.join(saved_dir, "e{:03d}_outputs_z_star.png".format(epoch+1)))
                outputs = Image.fromarray(batch2im(outputs_dn, 2, 2,
                    im_height=args.train_image_size, im_width=args.train_image_size))
                outputs.save(os.path.join(saved_dir, "e{:03d}_outputs_dn.png".format(epoch+1)))

        if epoch % args.save_freq == args.save_freq-1:
            torch.save(net.state_dict(), 
                    os.path.join(saved_dir,"{}_{}_e{:03d}.pb".format(model_name, 
                                                                     args.num_channels, 
                                                                     epoch+1)))
            torch.save(bfcnn.state_dict(), 
                    os.path.join(saved_dir,"{}_{}_e{:03d}_bfcnn.pb".format(model_name,
                                                                           args.num_channels, 
                                                                           epoch+1)))
        
        if epoch % args.test_freq == args.test_freq - 1:
            test(net, bfcnn, stddev, saved_dir=saved_dir, writer=writer, epoch=epoch)

        print('Time Taken: %d sec' % (time.time() - start_time))
        
    print('Finished Training')
    test(net, bfcnn, stddev, saved_dir=saved_dir)
    
    
def train_model(net, bfcnn, epoch=30, stddev=0., wd=0., model_name="", saved_dir=None, writer=None):
    params = bfcnn.parameters()
    optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    train(net, bfcnn, optimizer,
            epoch, 
            stddev=stddev, 
            saved_dir=saved_dir,
            model_name=model_name, 
            writer=writer)  


def main():
    if 'cifar' in args.dataset:
        Enc = Encoder_CIFAR
        Dec = Decoder_CIFAR
    else:
        Enc = Encoder
        Dec = Decoder
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

   
    net = Generator(encoder, decoder)

    bfcnn = BF_CNN(1, 64, 3, 20, args.num_channels)
    
    print(args)

    if args.pretrained_model_path is not None:
        try:
            filepath = args.pretrained_model_path
            print("Try loading "+filepath)
            net.load_state_dict(torch.load(filepath, map_location=dev))
        except Exception as e: 
            print(e)
            print("Loading Failed. Initializing Networks...")
            exit()

    net.to(device)
    bfcnn.to(device)

    if args.eval:
        test(net, bfcnn, 10**(-0.05*args.snr))
        exit()

    saved_dir = args.model_path
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
        
    writer = SummaryWriter(saved_dir)

    with open(os.path.join(saved_dir, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    train_model(net, 
                bfcnn,
                epoch=args.epochs,
                stddev=10**(-0.05*args.snr),
                model_name=args.model_name,
                saved_dir=saved_dir,
                writer=writer)


    torch.save(net.state_dict(),
            os.path.join(saved_dir,args.model_name+"_{}.pb".format(args.num_channels)))
    torch.save(bfcnn.state_dict(),
            os.path.join(saved_dir,args.model_name+"_{}_bfcnn.pb".format(args.num_channels)))



if __name__=='__main__':
    main()
