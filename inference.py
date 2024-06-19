# Basic module
from tqdm                  import tqdm
from model.parse_args_test import parse_args

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from model.utils  import *
from model.load_param_data import load_dataset1, load_param, load_dataset_eva

# Model
from model.net import *

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        self.save_prefix = '_'.join([args.model, args.dataset])

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            val_img_ids, test_txt = load_dataset_eva(args.root, args.dataset, args.split_method)

        self.val_img_ids, _ = load_dataset1(args.root, args.dataset, args.split_method)

        if args.dataset=='ICPR_Track2':
            Mean_Value = [0.2518, 0.2518, 0.2519]
            Std_value  = [0.2557, 0.2557, 0.2558]

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(Mean_Value, Std_value)])
        testset         = InferenceSetLoader(dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        model       = LightWeightNetwork()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model


        # Checkpoint
        checkpoint = torch.load(args.model_dir)

        eval_image_path   = './result_WS/'+ args.st_model +'/'+ 'visulization_result'
        eval_fuse_path    = './result_WS/'+ args.st_model +'/'+ 'visulization_fuse'


        make_visulization_dir(eval_image_path, eval_fuse_path)

        # Load trained model
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to('cuda')
        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)

        with torch.no_grad():
            num = 0
            for i, ( data, size) in enumerate(tbar):
                data = data.cuda()
                pred = self.model(data)
                save_resize_pred(pred, size, args.crop_size, eval_image_path, self.val_img_ids, num, args.suffix)

                num += 1



def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    main(args)

