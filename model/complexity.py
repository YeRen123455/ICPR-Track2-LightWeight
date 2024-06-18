# from thop import profile
# import torch
# from net import *


# if __name__ == '__main__':

#     input_img = torch.rand(1,3,512,512).cuda()
#     net = LightWeightNetwork().cuda()
#     flops, params = profile(net, inputs=(input_img, ))
#     print('Params: %2fM' % (params/1e6))
#     print('FLOPs: %2fGFLOPs' % (flops/1e9))

