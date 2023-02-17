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

import loader
import models.autoencoders as ae
import utils
import config

parser = config.get_common_parser()
parser = config.get_train_parser(parser)


args = parser.parse_args()

dev = "cuda:{}".format(args.gpu) if args.gpu>=0 else "cpu"

device = torch.device(dev)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

train_dataloader = loader.get_train_dataloader(args)
test_dataloader = loader.get_test_dataloader(args)
    
print(len(train_dataloader))
print(len(test_dataloader))

criterion = nn.MSELoss()

def test(net, stddev=0., saved_dir=None, writer=None, epoch=0):
    net.eval()
    avg_psnr = 0.
    count = 0.
    start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            inputs = data[0].to(device)
            outputs = net(inputs, stddev)
            avg_psnr += utils.PSNR(reduction='sum')(outputs, inputs, -1, 1, cuda=True)
            count += inputs.size(0)
            if args.show_outputs:
                plt.imsave('test.png', utils.batch2im(outputs, 3, 3, -1, 1, im_height=args.image_size, im_width=args.image_size))
                break
            
            if i%10==9:
                print('[{:4d} / {:4d}] - Average PSNR: {}, time taken: {:.2f} sec'.format(i+1, 
                    len(test_dataloader), avg_psnr/count, time.time()-start_time))

        print('[{:4d} / {:4d}] - Average PSNR: {}, time taken: {:.2f} sec'.format(i+1, 
            len(test_dataloader), avg_psnr/count, time.time()-start_time))
        if writer is not None:
            writer.add_scalar('PSNR/test', avg_psnr/count, epoch+1)


def train(net, 
          optimizer_G, 
          num_epoch, 
          stddev=0.,
          saved_dir=None,
          model_name=None,
          which_epoch=0,
          writer=None):
    
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    
    if 'cifar' in args.dataset:
        scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=args.epochs // 2, gamma=0.1)
    elif 'openimages' in args.dataset:
        scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.1)
    else:
        raise NotImplementedError()

    for epoch in range(which_epoch, num_epoch):  # loop over the dataset multiple times
        start_time = time.time()
        #train_iter = iter(train_dataloader)
        running_loss_mse = 0.0
        running_loss = 0.0
        net.train()

        for i, data in enumerate(train_dataloader):
            if args.debug and i==10:
                break
            inputs= data[0].to(device)

            # zero the parameter gradients
            optimizer_G.zero_grad()
            # forward
            # stddev: max(0, 𝜎±eps) where eps~N(0, 0.01*I).
                
            outputs, z = net(inputs, stddev, return_latent=True)
            mse_loss = criterion(outputs, inputs)

            loss = mse_loss 
            loss.backward()
            optimizer_G.step()
            
            # print statistics
            running_loss_mse += mse_loss.item()
            running_loss += loss.item()
            if  i % args.print_freq == args.print_freq-1 or i == len(train_dataloader)-1:
                print('[%d, %5d] loss: %.4f, mse: %.4f' %
                    (epoch + 1, i + 1, running_loss / (i+1), running_loss_mse/(i+1)))
   
        if writer is not None:
            writer.add_scalar("Loss/train", running_loss / (i+1), epoch+1)
            writer.add_scalar("MSE/train", running_loss_mse / (i+1), epoch+1)
        if epoch % args.display_freq == args.display_freq-1:
            with torch.no_grad():
                targets = Image.fromarray(utils.batch2im(inputs, 2, 2, -1, 1, 
                    im_height=args.train_image_size, im_width=args.train_image_size))
                targets.save(os.path.join(saved_dir, "e{:03d}_targets.png".format(epoch+1)))
                outputs = Image.fromarray(utils.batch2im(outputs, 2, 2, -1, 1,
                    im_height=args.train_image_size, im_width=args.train_image_size))
                outputs.save(os.path.join(saved_dir, "e{:03d}_outputs.png".format(epoch+1)))

        scheduler_G.step()
        print('Time Taken: %d sec' % (time.time() - start_time))

        if epoch % args.save_freq == args.save_freq-1:
            torch.save(net.state_dict(), 
                    os.path.join(saved_dir,"{}_{}_e{:03d}.pb".format(model_name, 
                        args.num_channels, 
                        epoch+1)))
        
        if epoch % args.test_freq == args.test_freq - 1:
            test(net, stddev, saved_dir=saved_dir, writer=writer, epoch=epoch)
        
    print('Finished Training')

    test(net, stddev, saved_dir=saved_dir)
    
def train_model(net, epoch=30, stddev=0., wd=0., model_name="", saved_dir=None, writer=None):
    optimizer_G = optim.Adam(net.parameters(), 
                            lr=args.lr, 
                            betas=(0.0, 0.9), 
                            weight_decay=args.weight_decay)
    train(net, 
          optimizer_G, 
          epoch, 
          stddev=stddev, 
          saved_dir=saved_dir, 
          model_name=model_name, 
          writer=writer)  
    
     


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

    print(encoder)
    print(decoder)
    print(args)
    net = ae.Generator(encoder, decoder)

    if args.pretrained_model_path is not None:
        try:
            filepath = args.pretrained_model_path
            print("Try loading "+filepath)
            net.load_state_dict(torch.load(filepath, map_location=dev))
        except Exception as e: 
            print(e)
            print("Loading Failed. Initializing Networks...")
            pass

    net.to(device)

    if args.eval:
        test(net, 10**(-0.05*args.snr))
        exit()

    saved_dir = args.model_path
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
        
    writer = SummaryWriter(saved_dir)

    with open(os.path.join(saved_dir, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    train_model(net, 
                epoch=args.epochs,
                stddev=10**(-0.05*args.snr),
                model_name=args.model_name,
                saved_dir=saved_dir,
                writer=writer)
    torch.save(net.state_dict(), 
            os.path.join(saved_dir,args.model_name+"_{}.pb".format(args.num_channels)))



if __name__=='__main__':
    main()
