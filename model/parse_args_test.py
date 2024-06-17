from model.utils import *

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Dense_Nested_Attention_Network_For_SIRST')
    # choose model
    parser.add_argument('--model', type=str, default='UNet',
                        help='model name:  UNet')
    parser.add_argument('--channel_size', type=str, default='two',
                        help='one,  two,  three,  four')
    parser.add_argument('--backbone', type=str, default='resnet_18',
                        help='vgg10, resnet_10,  resnet_18,  resnet_34 ')

    # data and pre-process
    parser.add_argument('--dataset', type=str, default='ICPR_Track2',
                        help='dataset name: ICPR_Track2')
    parser.add_argument('--st_model', type=str, default='ICPR_Track2')
    parser.add_argument('--model_dir', type=str,
                        default = './result_WS/ICPR_Track2/model_weight.pth.tar')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.5', help='when --mode==Ratio')
    parser.add_argument('--root', type=str, default='./dataset')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--split_method', type=str, default='img_idx',
                        help='70_20')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=512,
                        help='256, 512, 1024')
    parser.add_argument('--crop_size', type=int, default=512,
                        help='256, 512, 1024')


    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')

    # cuda and logging
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')

    # ROC threshold
    parser.add_argument('--ROC_thr', type=int, default=10,
                        help='crop image size')


    args = parser.parse_args()

    # the parser
    return args
