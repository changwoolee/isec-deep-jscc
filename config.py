import argparse

def get_common_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    ### Common Arguments ###
    parser.add_argument('--num_channels', '-nc', type=int, default=6, help='number of channels')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id. -1: cpu')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--batch_size', '-bs', type=int, default=128, help='batch_size')
    parser.add_argument('--model_name',type=str, default='openimages', help='model name')
    parser.add_argument('--data_dir', '-dd', type=str, default='./data', help='data directory')
    parser.add_argument('--display_freq', '-df', type=int, default=10, help='save_freq')
    parser.add_argument('--print_freq', '-pf', type=int, default=50, help='print freq')
    parser.add_argument('--image_size', '-is', type=int, default=128, help='image_size')

    parser.add_argument('--snr', '-s', type=float, default=0., help='SNR for Training')
    parser.add_argument('--lr', '-lr', type=float, default=0.0005, help='Learning rate')

    parser.add_argument('--dataset', '-dset', type=str, default="openimages", help='Specify Dataset. "cifar", "openimages", "kodak"')

    ### Model Configuration ###
    parser.add_argument('--num_hidden',type=int, default=32, help='Number of hidden nodes.')
    parser.add_argument('--power_norm',type=str, default='hard', help='Power normalization type. "hard|soft|none", default: "hard".')
    parser.add_argument('--num_conv_blocks', type=int, default=2, help='Number of hidden nodes.')
    parser.add_argument('--num_residual_blocks', type=int, default=3, help='Number of hidden nodes.')

    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    return parser


def get_train_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, default=150, help='number of epochs')
    parser.add_argument('--test_batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--save_freq', type=int, default=25, help='save_freq')
    parser.add_argument('--test_freq', type=int, default=5, help='test freq')
    parser.add_argument('--show_outputs', action='store_true', help='Show test output if ture.')

    parser.add_argument('--model_path',type=str, default='saved_models/imagenet/', help='model path')

    parser.add_argument('--eval', action='store_true', help='Eval mode')
    parser.add_argument('--pretrained_model_path',type=str, default=None, help='model path')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay parameter')

    parser.add_argument('--train_image_size', type=int, default=128, help='image size')

    return parser

