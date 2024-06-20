# Basic module
from tqdm                  import tqdm
from model.parse_args_test import parse_args
import scipy.io as scio

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from model.utils  import *
from model.metric import *
from model.loss   import *
from model.load_param_data import load_dataset1, load_param, load_dataset_eva

# Model
from model.net import *

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        self.ROC   = ROCMetric(1, args.ROC_thr)
        # self.PD_FA = PD_FA(1,255)
        self.PD_FA = PD_FA(1,10, args.crop_size)
        self.mIoU  = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            #train_img_ids, val_img_ids, test_txt=load_dataset_eva(args.root, args.dataset,args.split_method)
            val_img_ids, test_txt = load_dataset_eva(args.root, args.dataset, args.split_method)

        self.val_img_ids, _ = load_dataset1(args.root, args.dataset, args.split_method)

        if args.dataset=='ICPR_Track2':
            Mean_Value = [0.2518, 0.2518, 0.2519]
            Std_value  = [0.2557, 0.2557, 0.2558]

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(Mean_Value, Std_value)])
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        model       = LightWeightNetwork()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # DATA_Evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Checkpoint
        #checkpoint        = torch.load(args.root.split('dataset')[0] +args.model_dir)
        checkpoint = torch.load(args.model_dir)
        target_image_path = dataset_dir + '/' +'visulization_result' + '/' + args.st_model + '_visulization_result'
        target_dir        = dataset_dir + '/' +'visulization_result' + '/' + args.st_model + '_visulization_fuse'
        eval_image_path   = './result_WS/'+ args.st_model +'/'+ 'visulization_result'
        eval_fuse_path    = './result_WS/'+ args.st_model +'/'+ 'visulization_fuse'

        #make_visulization_dir(target_image_path, target_dir)
        make_visulization_dir(eval_image_path, eval_fuse_path)

        # Load trained model
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to('cuda')
        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()
        with torch.no_grad():
            num = 0
            for i, ( data, labels, size) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                pred = self.model(data)


                loss = SoftIoULoss(pred, labels)
                #save_Ori_intensity_Pred_GT(pred, labels,target_image_path, val_img_ids, num, args.suffix,args.crop_size)
                save_resize_pred(pred, size, args.crop_size, eval_image_path, self.val_img_ids, num, args.suffix)
                #save_Pred_GT_for_split_evalution(pred, labels, eval_image_path, self.val_img_ids, num, args.suffix, args.crop_size)

                num += 1

                losses.    update(loss.item(), pred.size(0))
                self.ROC.  update(pred, labels)
                self.mIoU. update(pred, labels)
                self.PD_FA.update(pred, labels)
                _, mean_IOU = self.mIoU.get()

            FA, PD    = self.PD_FA.get(len(val_img_ids), args.crop_size)
            test_loss = losses.avg

            # scio.savemat(dataset_dir + '/' +  'value_result'+ '/' +args.st_model  + '_PD_FA_' + str(255),
            #              {'number_record1': FA, 'number_record2': PD})

            print('test_loss, %.4f' % (test_loss))
            print('mean_IOU:', mean_IOU)
            print('PD:',PD)
            print('FA:',FA)
            self.best_iou = mean_IOU

''
def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    main(args)
