from PIL import Image, ImageOps, ImageFilter
import platform, os
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import  torch
from torch.nn import init
from datetime import datetime
import argparse
import shutil
from  matplotlib  import pyplot as plt
class TrainSetLoader(Dataset):

    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id ,base_size=512,crop_size=480,transform=None,suffix='.png'):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.transform   = transform
        self._items = img_id
        self.masks  = dataset_dir+'/'+'masks'
        self.images = dataset_dir+'/'+'images'

        self.base_size   = base_size
        self.crop_size   = crop_size
        self.suffix      = suffix



    def _sync_transform(self, img, mask, img_id):
        # random mirror
        if random.random() < 0.5:
            img   = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask  = mask.transpose(Image.FLIP_LEFT_RIGHT)



        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img  = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img  = ImageOps.expand(img,  border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1   = random.randint(0, w - crop_size)
        y1   = random.randint(0, h - crop_size)
        img  = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        img, mask = np.array(img), np.array(mask, dtype=np.float32)


        return img, mask

    def __getitem__(self, idx):

        img_id     = self._items[idx]                      # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path   = self.images+'/'+img_id+self.suffix    # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks +'/'+img_id+self.suffix

        img  = Image.open(img_path).convert('RGB')         ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        mask = Image.open(label_path)


        # synchronized transform
        img, mask = self._sync_transform(img, mask, img_id)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        mask = np.expand_dims(mask[:,:,0] if len(np.shape(mask))>2 else mask, axis=0).astype('float32')/ 255.0


        return img, torch.from_numpy(mask)  #img_id[-1]

    def __len__(self):
        return len(self._items)


class TestSetLoader(Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id, transform=None, base_size=512, crop_size=480, suffix='.png'):
        super(TestSetLoader, self).__init__()
        self.transform = transform
        self._items    = img_id
        self.masks     = dataset_dir+'/'+'masks'
        self.images    = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)


        # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)  # images: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        return img, mask

    def __getitem__(self, idx):
        # print('idx:',idx)
        img_id     = self._items[idx]  # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path   = self.images + '/' + img_id + self.suffix    # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks  + '/' + img_id + self.suffix

        img  = Image.open(img_path).convert('RGB')  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        mask = Image.open(label_path)
        # synchronized transform
        img, mask = self._testval_sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        mask = np.expand_dims(mask[:,:,0] if len(np.shape(mask))>2 else mask, axis=0).astype('float32')/ 255.0


        return img, torch.from_numpy(mask)  # img_id[-1]

    def __len__(self):
        return len(self._items)

class DemoLoader (Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, transform=None,base_size=512,crop_size=480,suffix='.png'):
        super(DemoLoader, self).__init__()
        self.transform = transform
        self.images    = dataset_dir
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _demo_sync_transform(self, img):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)

        # final transform
        img = np.array(img)
        return img

    def img_preprocess(self):
        img_path   =  self.images
        img  = Image.open(img_path).convert('RGB')

        # synchronized transform
        img  = self._demo_sync_transform(img)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img



def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal(m.weight.data)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_ckpt(state, save_path, filename):
    torch.save(state, os.path.join(save_path,filename))

def save_train_log(args, save_dir):
    dict_args=vars(args)
    args_key=list(dict_args.keys())
    args_value = list(dict_args.values())
    with open('result_WS/%s/train_log.txt'%save_dir ,'w') as  f:
        now = datetime.now()
        f.write("time:--")
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string)
        f.write('\n')
        for i in range(len(args_key)):
            f.write(args_key[i])
            f.write(':--')
            f.write(str(args_value[i]))
            f.write('\n')
    return

def save_model_and_result(dt_string, epoch, train_loss, test_loss, best_iou, recall, precision, save_mIoU_dir, save_other_metric_dir):

    with open(save_mIoU_dir, 'a') as f:
        f.write('{} - {:04d}:\t - train_loss: {:04f}:\t - test_loss: {:04f}:\t mIoU {:.4f}\n' .format(dt_string, epoch,train_loss, test_loss, best_iou))
    with open(save_other_metric_dir, 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epoch))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')

def save_model( best_iou, save_dir, save_prefix, train_loss, test_loss, recall, precision, epoch, net):
        save_mIoU_dir = 'result_WS/' + save_dir + '/' + save_prefix + '_best_IoU_IoU.log'
        save_other_metric_dir = 'result_WS/' + save_dir + '/' + save_prefix + '_best_IoU_other_metric.log'
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        save_model_and_result(dt_string, epoch, train_loss, test_loss, best_iou,
                              recall, precision, save_mIoU_dir, save_other_metric_dir)
        save_ckpt({
            'epoch': epoch,
            'state_dict': net,
            'loss': test_loss,
            'mean_IOU': best_iou,
        }, save_path='result_WS/' + save_dir,
            filename='mIoU_' + '_' + save_prefix + '_epoch' + '.pth.tar')


def split_evaluation(dataset_dir, original_mask_dir, target_image_path, save_train_result_dir, base_size, FA, PD, supervision, split_method):
    evaluation_mode = ['test_point', 'test_spot', 'test_extended']
    with open('/media/gfkd/software/SIRST_Detection/ICPR_Track2/' + save_train_result_dir, 'a') as  f:
        f.write("FA:")
        f.write(str(FA))
        f.write("\n")

    with open('/media/gfkd/software/SIRST_Detection/ICPR_Track2/' + save_train_result_dir, 'a') as  f:
        f.write("PD:")
        f.write(str(PD))
        f.write("\n")

    for item in range(len(evaluation_mode)):
        mode = evaluation_mode[item]


        txt_dir = dataset_dir + '/' + split_method +'/' + 'test.txt'

        test_img = []
        with open(txt_dir, "r") as f:
            line = f.readline()
            while line:
                test_img.append(line.split('\n')[0])
                line = f.readline()
            f.close()
        mini  = 1
        maxi  = 1  # nclass
        nbins = 1  # nclass
        total_inter = 0
        total_union = 0
        for k in range(len(test_img)):

            WS_label   = Image.open(target_image_path + '/' + test_img[k] + '.png').convert('RGB')
            Full_label = Image.open(original_mask_dir + '/' + test_img[k] + '.png').convert('RGB')

            Full_label = Full_label.resize((base_size, base_size), Image.NEAREST)

            WS_label   = np.array(WS_label)
            Full_label = np.array(Full_label)

            if WS_label.ndim == 3:
                WS_label   = WS_label[:, :, 0]
            if Full_label.ndim == 3:
                Full_label = Full_label[:, :, 0]


            WS_label = (WS_label > 0).astype('float32')
            Full_label = (Full_label > 0).astype('float32')
            intersection = WS_label * ((WS_label == Full_label))  # TP

            # areas of intersection and union
            area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
            area_pred, _  = np.histogram(WS_label, bins=nbins, range=(mini, maxi))
            area_lab, _   = np.histogram(Full_label, bins=nbins, range=(mini, maxi))
            area_union    = area_pred + area_lab - area_inter
            total_inter  += area_inter
            total_union  += area_union


        with open('/media/gfkd/software/SIRST_Detection/ICPR_Track2/' + save_train_result_dir,'a') as  f:
            f.write("mIoU:")
            f.write(mode)
            f.write(str(total_inter / total_union))
            f.write("\n")


        print("mIoU:", '--' ,mode, str(total_inter / total_union))


def split_test_evaluation(dataset_dir, original_mask_dir, target_image_path, save_train_result_dir, base_size, FA, PD, mean_IOU, st_model):
    if   'NUDT-SIRST'  in save_train_result_dir:
       evaluation_mode = ['test_point', 'test_spot',  'test_spot_tiny', 'test_spot_small',  'test_spot_medium', 'test_spot_big', 'test_extended']
    elif 'NUAA-SIRST'  in save_train_result_dir:
       evaluation_mode = ['test_point', 'test_spot', 'test_extended']
    elif 'IRSTD-SIRST' in save_train_result_dir:
        evaluation_mode = ['test_point', 'test_spot', 'test_extended']

    with open('/media/gfkd/software/SIRST_Detection/WS_SIRST/result_WS/' + st_model+ '/' + save_train_result_dir, 'a') as  f:
        f.write("FA:")
        f.write(str(FA))
        f.write("\n")

    with open('/media/gfkd/software/SIRST_Detection/WS_SIRST/result_WS/'+ st_model+ '/' + save_train_result_dir, 'a') as  f:
        f.write("PD:")
        f.write(str(PD))
        f.write("\n")

    with open('/media/gfkd/software/SIRST_Detection/WS_SIRST/result_WS/'+ st_model+ '/' + save_train_result_dir, 'a') as  f:
        f.write("IoU:")
        f.write(str(mean_IOU))
        f.write("\n")


    for item in range(len(evaluation_mode)):
        mode = evaluation_mode[item]


        if mode   == 'test_point':
            txt_dir = dataset_dir + '/' + '70_20/' + 'point_target_test.txt'
        elif mode == 'test_spot':
            txt_dir = dataset_dir + '/' + '70_20/' + 'spot_target_test.txt'
        elif mode == 'test_extended':
            txt_dir = dataset_dir + '/' + '70_20/' + 'extended_target_test.txt'
        elif mode == 'test_spot_tiny':
            txt_dir = dataset_dir + '/' + '70_20/' + 'spot_tiny_target_test.txt'
        elif mode == 'test_spot_small':
            txt_dir = dataset_dir + '/' + '70_20/' + 'spot_small_target_test.txt'
        elif mode == 'test_spot_medium':
            txt_dir = dataset_dir + '/' + '70_20/' + 'spot_medium_target_test.txt'
        elif mode == 'test_spot_big':
            txt_dir = dataset_dir + '/' + '70_20/' + 'spot_big_target_test.txt'

        test_img = []
        with open(txt_dir, "r") as f:
            line = f.readline()
            while line:
                test_img.append(line.split('\n')[0])
                line = f.readline()
            f.close()
        mini  = 1
        maxi  = 1  # nclass
        nbins = 1  # nclass
        total_inter = 0
        total_union = 0
        for k in range(len(test_img)):
            WS_label   = Image.open(target_image_path + '/' + test_img[k] + '.png').convert('RGB')
            Full_label = Image.open(original_mask_dir + '/' + test_img[k] + '.png').convert('RGB')
            Full_label = Full_label.resize((base_size, base_size), Image.NEAREST)

            WS_label   = np.array(WS_label)
            Full_label = np.array(Full_label)

            if WS_label.ndim == 3:
                WS_label   = WS_label[:, :, 0]
            if Full_label.ndim == 3:
                Full_label = Full_label[:, :, 0]


            WS_label = (WS_label > 0).astype('float32')
            Full_label = (Full_label > 0).astype('float32')
            intersection = WS_label * ((WS_label == Full_label))  # TP

            # areas of intersection and union
            area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
            area_pred, _  = np.histogram(WS_label, bins=nbins, range=(mini, maxi))
            area_lab, _   = np.histogram(Full_label, bins=nbins, range=(mini, maxi))
            area_union    = area_pred + area_lab - area_inter
            total_inter  += area_inter
            total_union  += area_union


        with open('/media/gfkd/software/SIRST_Detection/WS_SIRST/' + 'result_WS' + '/' + st_model+ '/' + save_train_result_dir,'a') as  f:
            f.write("mIoU:")
            f.write(mode)
            f.write(str(total_inter / total_union))
            f.write("\n")


        print("mIoU:", '--' ,mode, str(total_inter / total_union))


def save_result_for_test_together(dataset_dir, epochs, best_iou, recall, precision ):
    with open(dataset_dir + '/' + 'best_IoU.log', 'a') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('{} - {:04d}:\t{:.4f}\n'.format(dt_string, epochs, best_iou))

    with open(dataset_dir + '/' + 'best_other_metric.log', 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epochs))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')
    return

def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def make_dir(gpu, deep_supervision, dataset, model):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    if deep_supervision:
        save_dir = "%s_%s_%s_%s_wDS" % (gpu,dataset, model, dt_string)
    else:
        save_dir = "%s_%s_%s_%s_woDS" % (gpu, dataset, model, dt_string)
    os.makedirs('result_WS/%s' % save_dir, exist_ok=True)
    return save_dir


def total_visulization_generation(dataset_dir, mode, test_txt, suffix, target_image_path, target_dir):
    source_image_path = dataset_dir + '/images'

    txt_path = test_txt
    ids = []
    with open(txt_path, 'r') as f:
        ids += [line.strip() for line in f.readlines()]

    for i in range(len(ids)):
        source_image = source_image_path + '/' + ids[i] + suffix
        target_image = target_image_path + '/' + ids[i] + suffix
        shutil.copy(source_image, target_image)
    for i in range(len(ids)):
        source_image = target_image_path + '/' + ids[i] + suffix
        img = Image.open(source_image)
        img = img.resize((256, 256), Image.ANTIALIAS)
        img.save(source_image)
    for m in range(len(ids)):
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        img = plt.imread(target_image_path + '/' + ids[m] + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Raw Imamge", size=11)

        plt.subplot(1, 3, 2)
        img = plt.imread(target_image_path + '/' + ids[m] + '_GT' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Ground Truth", size=11)

        plt.subplot(1, 3, 3)
        img = plt.imread(target_image_path + '/' + ids[m] + '_Pred' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Predicts", size=11)

        plt.savefig(target_dir + '/' + ids[m].split('.')[0] + "_fuse" + suffix, facecolor='w', edgecolor='red')



def make_visulization_dir(target_image_path, target_dir):
    if os.path.exists(target_image_path):
        shutil.rmtree(target_image_path)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_image_path)

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)  # 删除目录，包括目录下的所有文件
    os.mkdir(target_dir)

def save_Pred_GT(pred, labels, target_image_path, val_img_ids, num, suffix):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) +suffix)
    img = Image.fromarray(labelsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)

def save_Ori_intensity_Pred_GT(pred, labels, target_image_path, val_img_ids, num, suffix, crop_size):

    predsss = np.array((pred > 0).cpu()).astype('int64')*np.array((pred).cpu()).astype('int64')
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(crop_size, crop_size))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) +suffix)
    img = Image.fromarray(labelsss.reshape(crop_size, crop_size))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)

def save_Pred_GT_for_split_evalution(pred, labels, target_image_path, val_img_ids, num, suffix, crop_size):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)

    img = Image.fromarray(predsss.reshape(crop_size, crop_size))
    img.save(target_image_path + '/' + '%s' % (val_img_ids[num]) +suffix)



def save_Pred_GT_visulize(pred, img_demo_dir, img_demo_index, suffix):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)

    img = Image.fromarray(predsss.reshape(256, 256))
    img.save(img_demo_dir + '/' + '%s_Pred' % (img_demo_index) +suffix)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    img = plt.imread(img_demo_dir + '/' + img_demo_index + suffix)
    plt.imshow(img, cmap='gray')
    plt.xlabel("Raw Imamge", size=11)

    plt.subplot(1, 2, 2)
    img = plt.imread(img_demo_dir + '/' + '%s_Pred' % (img_demo_index) +suffix)
    plt.imshow(img, cmap='gray')
    plt.xlabel("Predicts", size=11)


    plt.savefig(img_demo_dir + '/' + img_demo_index + "_fuse" + suffix, facecolor='w', edgecolor='red')
    plt.show()



def save_and_visulize_demo(pred, labels, target_image_path, val_img_ids, num, suffix):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) +suffix)
    img = Image.fromarray(labelsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)

    return


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

### compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
