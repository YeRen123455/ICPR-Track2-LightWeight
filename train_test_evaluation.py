# torch and visulization
import os
import time

from tqdm             import tqdm
import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader
from model.parse_args_train import  parse_args

# metric, loss .etc
from model.utils  import *
from model.metric import *
from model.loss   import *
from model.load_param_data         import  load_dataset, load_param

# model

from model.net          import  LightWeightNetwork

import scipy.io as scio


class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args  = args
        self.ROC   = ROCMetric(1, 10)
        self.PD_FA = PD_FA(1, 10, args.crop_size)
        self.mIoU  = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = args.save_dir
        nb_filter, num_blocks= load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode == 'TXT':
            self.train_dataset_dir = args.root + '/' + args.dataset
            self.test_dataset_dir  = args.root + '/' + args.dataset

        self.train_img_ids, self.val_img_ids, self.test_txt = load_dataset(args.root, args.dataset, args.split_method)

        if args.dataset=='ICPR_Track2':
            mean_value = [0.2518, 0.2518, 0.2519]
            std_value  = [0.2557, 0.2557, 0.2558]

        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_value, std_value)])
        trainset        = TrainSetLoader(self.train_dataset_dir,img_id=self.train_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        testset         = TestSetLoader (self.test_dataset_dir, img_id=self.val_img_ids,  base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        evalset         = TestSetLoader (self.test_dataset_dir, img_id=self.val_img_ids,  base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)

        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers,drop_last=True)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)
        self.eval_data  = DataLoader(dataset=evalset,  batch_size=args.eval_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)

        if args.model == 'UNet':
            model       = LightWeightNetwork()

        model           = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # Optimizer and lr scheduling

        if args.optimizer == 'Adagrad':
            self.optimizer  = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        if args.scheduler   == 'CosineAnnealingLR':
            self.scheduler  = lr_scheduler.CosineAnnealingLR( self.optimizer, T_max=args.epochs, eta_min=args.min_lr)


        # DATA_Evaluation metrics
        self.best_iou       = 0
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

    # Training
    def training(self, epoch):
        lr = self.scheduler.get_lr()[0]
        save_lr_dir = 'result_WS/' + self.save_dir + '/' + self.save_prefix + '_learning_rate.log'
        with open(save_lr_dir, 'a') as f:
            f.write(' learning_rate: {:04f}:\n'.format(lr))
        print('learning_rate:',lr)


        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()
        for i, ( data, labels) in enumerate(tbar):
            data   = data.cuda()
            labels = labels.cuda()
            pred = self.model(data)
            loss = SoftIoULoss(pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))
        self.train_loss = losses.avg
        if args.lr_mode == 'adjusted_lr':
            self.scheduler.step()


    # Testing
    def testing (self, epoch):
        tbar   = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        losses = AverageMeter()

        with torch.no_grad():
            for i, ( data, labels) in enumerate(tbar):
                data   = data.cuda()
                labels = labels.cuda()
                pred   = self.model(data)
                loss   = SoftIoULoss(pred, labels)
                losses.update(loss.item(), pred.size(0))
                self.ROC.update(pred, labels)
                self.mIoU.update(pred, labels)
                _, mean_IOU = self.mIoU.get()
                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg, mean_IOU))

            self.test_loss = losses.avg

            save_train_test_loss_dir = 'result_WS/' + self.save_dir + '/' + self.save_prefix + '_train_test_loss.log'
            with open(save_train_test_loss_dir, 'a') as f:
                f.write('epoch: {:04f}:\t'.format(epoch))
                f.write('train_loss: {:04f}:\t'.format(self.train_loss))
                f.write('test_loss: {:04f}:\t'.format(self.test_loss))
                f.write('\n')

        # save high-performance model
        if mean_IOU > self.best_iou:
            self.best_iou = mean_IOU
            save_model(self.best_iou, self.save_dir, self.save_prefix,
                   self.train_loss, self.test_loss, recall, precision, epoch, self.model.state_dict())

    def evaluation(self,epoch):
        candidate_model_dir = os.listdir('result_WS/' + self.save_dir )
        for model_num in range(len(candidate_model_dir)):
            model_dir = 'result_WS/' + self.save_dir + '/' + candidate_model_dir[model_num]
            if '.pth.tar' in model_dir:
                model_path = model_dir

        checkpoint        = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to('cuda')

        evaluation_save_path =  './result_WS/' + self.save_dir
        target_image_path    =  evaluation_save_path + '/' +'visulization_result'
        target_dir           =  evaluation_save_path + '/' +'visulization_fuse'

        make_visulization_dir(target_image_path, target_dir)

        # Load trained model
        # Test
        self.model.eval()
        tbar = tqdm(self.eval_data)
        losses = AverageMeter()
        with torch.no_grad():
            num = 0
            for i, (data, labels) in enumerate(tbar):
                data   = data.cuda()
                labels = labels.cuda()
                pred = self.model(data)
                loss = SoftIoULoss(pred, labels)
                save_Pred_GT_for_split_evalution(pred, labels, target_image_path, self.val_img_ids, num, args.suffix, args.crop_size)
                num += 1

                losses.    update(loss.item(), pred.size(0))
                self.ROC.  update(pred, labels)
                self.mIoU. update(pred, labels)
                self.PD_FA.update(pred, labels)
                _, mean_IOU = self.mIoU.get()

            FA, PD    = self.PD_FA.get(len(self.val_img_ids), args.crop_size)
            test_loss = losses.avg
            scio.savemat(evaluation_save_path + '/' + 'PD_FA_' + str(255), {'number_record1': FA, 'number_record2': PD})

            print('test_loss, %.4f' % (test_loss))
            print('mean_IOU:', mean_IOU)
            print('PD:', PD)
            print('FA:', FA)
            self.best_iou = mean_IOU


def main(args):
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)
        if (epoch+1) ==args.epochs:
           trainer.evaluation(epoch)


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)





