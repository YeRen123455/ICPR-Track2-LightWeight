from model.parse_args_train import *
from model.load_param_data import *

from model.model_res_Unet import *
from torchstat import stat
from model.utils import *

args = parse_args()
nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

#设置模型
net = res_UNet(num_classes=1, input_channels=args.in_channels, block='Res_block', num_blocks= num_blocks, nb_filter=nb_filter)

#输入图像尺度为3通道，宽512，高512
stat(net, (3, 512, 512))

