import numpy as np
import torch
import torch.nn as nn

class UnitNorm(nn.Module):
    def __init__(self, max_power=1, eps=1e-5):
        super(UnitNorm, self).__init__()
        self.max_power = max_power

    def forward(self, x):
        x_shape = x.size()
        x = x.reshape(x_shape[0], -1)
        multiplier = self.max_power * np.sqrt(x.size(1))
        proj_idx = torch.norm(x, p=2, dim=1) > multiplier
        x[proj_idx] = multiplier * x[proj_idx] / torch.norm(x[proj_idx], p=2, dim=1, keepdim=True)
        return x.reshape(*x_shape)


def unitnorm(x, max_power=1.0):
	x_shape = x.size()
	x = x.reshape(x_shape[0], -1)
	multiplier = max_power * np.sqrt(x.size(1))
	proj_idx = torch.norm(x, p=2, dim=1) > multiplier
	x[proj_idx] = multiplier * x[proj_idx] / torch.norm(x[proj_idx], p=2, dim=1, keepdim=True)
	return x.reshape(*x_shape)


class ResBlock(nn.Module):
    def __init__(self, dim1, dim2, normalization=nn.BatchNorm2d, activation=nn.ReLU, r=2, conv=None, kernel_size=5):
        super(ResBlock, self).__init__()
        
        if conv is None:
            conv = nn.Conv2d
        self.block1 = nn.Sequential(conv(dim1, dim1//r, kernel_size, stride=1, padding=(kernel_size-1)//2), normalization(dim1//r), activation())
        self.block2 = nn.Sequential(conv(dim1//r, dim2, kernel_size, stride=1, padding=(kernel_size-1)//2), normalization(dim2))
        if dim1 != dim2:
            self.shortcut = conv(dim1, dim2, 1, bias=False)
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        x_short = self.shortcut(x)
        x = self.block2(self.block1(x))
        return x+x_short
        



class Encoder(nn.Sequential):
    """Encoder Module"""
    def __init__(self, 
                 num_out,
                 num_hidden, 
                 num_conv_blocks=2,
                 num_residual_blocks=3,
                 normalization=nn.BatchNorm2d, 
                 activation=nn.PReLU, 
                 power_norm="hard", 
                 primary_latent_power=1.0,
                 r=2,
                 conv_kernel_size=5,
                 residual=True):
        conv = nn.Conv2d
        layers = [conv(3, num_hidden, 7, stride=2, padding=3),
                    normalization(num_hidden),
                    activation()]

        channels = num_hidden
        for _ in range(num_conv_blocks-1):
            layers += [conv(channels, channels, conv_kernel_size, stride=2, padding=(conv_kernel_size-1)//2)]
            layers += [normalization(channels),
                       activation()]
            if residual:
                layers += [ResBlock(channels, channels, normalization, activation, conv=conv, r=r, kernel_size=conv_kernel_size), activation()]

        for _ in range(num_residual_blocks-1): 
            if residual:
                layers += [ResBlock(channels, channels, normalization, activation, conv=conv, r=r, kernel_size=conv_kernel_size), activation()]
            else:
                layers += [conv(channels, channels, conv_kernel_size, stride=1, padding=(conv_kernel_size-1)//2), normalization(channels), activation()]
                                 
        layers += [conv(channels, num_out, conv_kernel_size, stride=1, padding=(conv_kernel_size-1)//2), 
                normalization(num_out)]
        if power_norm == "hard":
            layers += [UnitNorm()]
        elif power_norm == "soft":
            layers += [nn.BatchNorm2d(num_out, affine=False)]
        else:
            raise NotImplementedError()

        super(Encoder, self).__init__(*layers)


class Encoder_CIFAR(nn.Sequential):
    """Encoder Module"""
    def __init__(self, 
                 num_out,
                 num_hidden, 
                 num_conv_blocks=2,
                 num_residual_blocks=2,
                 conv = nn.Conv2d,
                 normalization=nn.BatchNorm2d, 
                 activation=nn.PReLU, 
                 power_norm="hard", 
                 primary_latent_power=1.0,
                 r=2,
                 max_channels=512,
                 first_conv_size=7,
                 conv_kernel_size=3,
                 residual=True,
                 **kwargs):

        bias = normalization==nn.Identity
        layers = [conv(3, num_hidden, first_conv_size, stride=1, padding=(first_conv_size-1)//2, bias=bias),
                    normalization(num_hidden),
                    activation()]

        channels = num_hidden
        for _ in range(num_conv_blocks):
            channels *= 2
            layers += [conv(np.minimum(channels//2, max_channels), np.minimum(channels, max_channels), 3, stride=2, padding=1, bias=bias)]
            layers += [normalization(np.minimum(channels, max_channels)),
                       activation()]
            if residual:
                layers += [ResBlock(np.minimum(channels, max_channels), np.minimum(channels, max_channels), normalization, activation, conv=conv, r=r, kernel_size=3), activation()]
            else:
                layers += [conv(np.minimum(channels, max_channels), np.minimum(channels, max_channels), 3, stride=1, padding=1), normalization(np.minimum(channels, max_channels)), activation()]

        for _ in range(num_residual_blocks): 
            if residual:
                layers += [ResBlock(np.minimum(channels, max_channels), np.minimum(channels, max_channels), normalization, activation, conv=conv, r=r, kernel_size=3), activation()]
            else:
                layers += [conv(np.minimum(channels, max_channels),np.minimum(channels, max_channels), conv_kernel_size, stride=1, padding=(conv_kernel_size-1)//2), 
                        normalization(np.minimum(channels, max_channels)), 
                        activation()]

        if residual:
            layers += [ResBlock(np.minimum(channels, max_channels), num_out, normalization, activation, conv=conv, r=r, kernel_size=3)]
        else:
            layers += [conv(np.minimum(channels, max_channels), num_out, 3, stride=1, padding=1)]
        if power_norm == "hard":
            layers += [UnitNorm()]
        elif power_norm == "soft":
            layers += [nn.BatchNorm2d(num_out, affine=False)]
        elif power_norm == "none":
            pass
        else:
            raise NotImplementedError()

        super(Encoder_CIFAR, self).__init__(*layers)


class Decoder(nn.Sequential):
    def __init__(self, 
                num_in, 
                num_hidden, 
                num_conv_blocks=2,
                num_residual_blocks=3,
                normalization=nn.BatchNorm2d, 
                activation=nn.PReLU, 
                no_tanh=False,
                r=2,
                conv_kernel_size=5,
                residual=True):

        channels = num_hidden

        layers = [nn.Conv2d(num_in, channels, conv_kernel_size, stride=1, padding=(conv_kernel_size-1)//2),
                  normalization(channels),
                  activation()] 

        for _ in range(num_residual_blocks-1):
            if residual:
                layers += [ResBlock(channels, channels, normalization, activation, r=r, kernel_size=conv_kernel_size), activation()]
            else:
                layers += [nn.Conv2d(channels, channels, conv_kernel_size, stride=1, padding=(conv_kernel_size-1)//2), normalization(channels), activation()]

        for _ in range(num_conv_blocks-1):
            layers += [nn.Upsample(scale_factor=(2,2), mode='bilinear')]
            layers += [nn.Conv2d(channels, channels, conv_kernel_size, 1, padding=(conv_kernel_size-1)//2)]
            layers += [normalization(channels), activation()]
            if residual:
                    layers += [ResBlock(channels, channels, normalization, activation, r=r, kernel_size=conv_kernel_size), activation()]

        layers += [nn.Upsample(scale_factor=(2,2), mode='bilinear'),
                nn.Conv2d(num_hidden, 3, 7, stride=1, padding=3)]
        layers += [normalization(3)]
        if not no_tanh:
            layers += [nn.Tanh()]
        super(Decoder, self).__init__(*layers)


class Decoder_CIFAR(nn.Sequential):
    def __init__(self, 
                num_in, 
                num_hidden, 
                num_conv_blocks=2,
                num_residual_blocks=2,
                normalization=nn.BatchNorm2d, 
                activation=nn.PReLU, 
                no_tanh=False,
                bias_free=False,
                r=2,
                residual=True,
                max_channels=512,
                last_conv_size=5,
                normalize_first=False,
                conv_kernel_size=3,
                **kwargs):

        channels = num_hidden * (2**num_conv_blocks)

        layers = [nn.Conv2d(num_in, min(max_channels, channels), 3, stride=1, padding=1, bias=False),
                  normalization(channels),
                  activation()] 

        for _ in range(num_residual_blocks):
            if residual:
                layers += [ResBlock(min(max_channels, channels), min(max_channels, channels), normalization, activation, r=r, kernel_size=conv_kernel_size), activation()]
            else:
                layers += [nn.Conv2d(min(max_channels, channels), min(max_channels, channels), conv_kernel_size, stride=1, padding=(conv_kernel_size-1)//2), normalization(channels), activation()]

        for _ in range(num_conv_blocks):
            channels = channels // 2
            layers += [nn.Upsample(scale_factor=(2,2), mode='bilinear'),
                    nn.Conv2d(min(max_channels, channels*2), min(max_channels, channels), 3, 1, 1, bias=False),
                    normalization(min(max_channels, channels)),
                    activation()]
            if residual:
                layers += [ResBlock(min(max_channels, channels), min(max_channels, channels), normalization, activation, r=r, kernel_size=3), activation()]
            else:
                layers += [nn.Conv2d(min(max_channels, channels), min(max_channels, channels), conv_kernel_size, stride=1, padding=(conv_kernel_size-1)//2), normalization(channels), activation()]

        layers += [nn.Conv2d(num_hidden, 3, last_conv_size, stride=1, padding=(last_conv_size-1)//2, bias=False)]

        if not normalize_first:
            layers += [normalization(3)]
        if not no_tanh:
            layers += [nn.Tanh()]

        super(Decoder_CIFAR, self).__init__(*layers)


class Generator(nn.Module):
    def __init__(self, encoder, decoder):
        super(Generator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inp, stddev, return_latent=False):
        code = self.encoder(inp)
        chan_noise = torch.randn_like(code) * stddev
        y = code + chan_noise
        reconst = self.decoder(y)

        if return_latent:
            return reconst, (y, code)
        else:
            return reconst


