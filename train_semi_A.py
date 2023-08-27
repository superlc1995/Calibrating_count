import argparse
import datetime
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Dataset
import os.path as osp
from PIL import Image
from crowd_A import build_dataset_partial
from crowd_A import build_dataset_unsup
# from engine_semi_ema import *
from models import build_model_confi
import os
import time
from tensorboardX import SummaryWriter
import pickle
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from engine_semi_A import *

import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training P2PNet', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=3500, type=int)
    parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)

    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    # dataset parameters
    parser.add_argument('--dataset_file', default='SHHA')
    parser.add_argument('--data_root', default='./new_public_density_data',
                        help='path where the dataset is')
    
    parser.add_argument('--output_dir', default='./log',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./ckpt',
                        help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='./runs',
                        help='path where to save, empty for no saving')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')
    parser.add_argument('--confi_weight', default=1.0, type = float)
    parser.add_argument('--label_pro',type=str, default='10')
    parser.add_argument('--un_weight', default=1.0, type = float)

    parser.add_argument('--in_epoch',type=int, default=10)
    parser.add_argument('--end_pro',type=float, default=0.5)
    parser.add_argument('--engine_type', type = str, default = 'type_0')
    return parser



def crowd_wasserstein(pt1, pt2, punish_side = 64):
    if len(pt1) == 0 and len(pt2) == 0:
        return 0
    elif len(pt1) == 0 and len(pt2) > 0:
        return  punish_side * np.sqrt(2) 
    elif len(pt1) > 0 and len(pt2) == 0:
        return  punish_side * np.sqrt(2) 

    cost = pairwise_distances(pt1, pt2)
    row_ind, col_ind = linear_sum_assignment(cost)

    diff = np.maximum(len(pt1), len(pt2)) - np.minimum(len(pt1), len(pt2))
    
    total_cost = cost[row_ind, col_ind].sum() + diff * punish_side * np.sqrt(2) 
    total_cost = total_cost/ np.maximum(len(pt1), len(pt2))


    return total_cost

def confi_gen(img_ink, tar_ink, pre_tar):
    # img_ink = img_ink
    img_pr = {}
    pre_pr = {}
    conf = np.zeros([2,2])
    for xindex in range(2):
        for yindex in range(2):
            tar_part = tar_ink[xindex*64:(xindex*64 + 64), yindex*64:(yindex*64 + 64)]
            pre_part = pre_tar[xindex*64:(xindex*64 + 64), yindex*64:(yindex*64 + 64)]
            tar_dots = np.array(np.where( tar_part >0 ) ).T
            pre_dots = np.array(np.where( pre_part >0 ) ).T
            conf[xindex, yindex] = crowd_wasserstein(tar_dots, pre_dots, punish_side = 64)
            img_pr[(xindex, yindex)] = img_ink[:, xindex*64:(xindex*64 + 64), yindex*64:(yindex*64 + 64)].numpy()
            pre_pr[(xindex, yindex)] = tar_ink[ xindex*64:(xindex*64 + 64), yindex*64:(yindex*64 + 64)].numpy()
    return conf, img_pr, pre_pr


def confi_acumu( model, t_loader,  epoch ):

    model.eval()
    img_list_pp = []
    tar_list_pp = []
    conf_list_pp = []
    for img, tar, names in t_loader:
        outputs,_ = model(img.cuda())
        for ink in range(img.size()[0]):
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][ink]

            outputs_points = outputs['pred_points'][ink]

            points = outputs_points[outputs_scores >0.5].detach().cpu().numpy().tolist()
            points = np.array(points)
            points[points >= 128] = 127
            predict_tar = np.zeros([128,128])
            if len(points) == 0:
                pass
            else:
                predict_tar[points[:,1].astype(int), points[:,0].astype(int)] = 1    

            conf_s, img_ppp, tar_ppp = confi_gen(img[ink] ,tar[ink][0], predict_tar)

            if epoch == 0:
                train_dict[names[ink]] = conf_s
            else:
                train_dict[names[ink]] = train_dict[names[ink]] + conf_s
            
            for x_index in range(2):
                for y_index in range(2):
                    img_list_pp.append( img_ppp[(x_index,y_index)] )
                    tar_list_pp.append( tar_ppp[(x_index,y_index)] )
                    conf_list_pp.append( train_dict[names[ink]][x_index, y_index] )


            # train_dict_detail[names[ink]] = detail_img
            # train_dict_conf[names[ink]] = train_dict[names[ink]]
    return img_list_pp, tar_list_pp, conf_list_pp
    # print('accumu_time: ' + str(time.time() - start))

def ema_models(model_1, model_2, factor = 0):

    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        key_item_1[1].copy_(key_item_1[1] * factor + (1 - factor) *  key_item_2[1])



def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    # create the logging file
    run_log_name = os.path.join(args.output_dir, 'run_log.txt')
    with open(run_log_name, "w") as log_file:
        log_file.write('Eval Log %s\n' % time.strftime("%c"))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    # backup the arguments



    print(args)
    with open(run_log_name, "a") as log_file:
        log_file.write("{}".format(args))
    device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # get the P2PNet model
    print(1)
    model, criterion = build_model_confi(args, training=True)
    teacher_model,_ = build_model_confi(args, training=True)
    # move to GPU
    print(1)
    model.to(device)
    teacher_model.to(device)
    criterion.to(device)
    print(1)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # use different optimation params for different parts of the model
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # Adam is used by default
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # create the dataset
    loading_data = build_dataset_partial(args=args)
    unloading_data = build_dataset_unsup(args=args)
    # create the training and valiation set
    if args.label_pro == '10':
        i_list = 'label_list/sha-10.txt'
    elif args.label_pro == '5':
        i_list = 'label_list/sha-5.txt'
    elif args.label_pro == '40':
        i_list = 'label_list/sha-40.txt'

    train_set, val_set = loading_data(args.data_root, i_list)
    untrain_set, _ = unloading_data(args.data_root, i_list)

    # create the sampler used during training
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)
    unsampler_train = torch.utils.data.RandomSampler(untrain_set)
    # unsampler_val = torch.utils.data.SequentialSampler(unval_set)


    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=False)
    unbatch_sampler_train = torch.utils.data.BatchSampler(
        unsampler_train, args.batch_size, drop_last=False)      

    # the dataloader for training
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    undata_loader_train = DataLoader(untrain_set, batch_sampler=unbatch_sampler_train,
                                   collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
    # resume the weights and training state if exists
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    # save the performance during the training
    mae = []
    mse = []
    # the logger writer
    writer = SummaryWriter(args.tensorboard_dir)
    
    step = 0
    # training starts here
    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()
        img_list_add, tar_list_add, conf_list_add = confi_acumu( model, data_loader_t, epoch )

        # if epoch ==50:
        #     with open('conf_img_10/res_img.pickle', 'wb') as handle:
        #         pickle.dump(img_list_add, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #     # with open('conf_img_10/res_dict.json', 'w') as f:
        #     #     json.dump(train_dict_detail, f)
        #     with open('conf_img_10/res_tar.pickle', 'wb') as handle:
        #         pickle.dump(tar_list_add, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #     with open('conf_img_10/res_conf.pickle', 'wb') as handle:
        #         pickle.dump(conf_list_add, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        #     np.save( 'conf_img_10/res_conf.npy' , np.array(conf_list_add) )

        stat, c_loss = train_one_epoch(
            model, teacher_model, criterion, data_loader_train, optimizer, device, epoch, train_dict, data_loader_t,
            undata_loader_train,img_list_add, tar_list_add, conf_list_add,  args.clip_max_norm, args.confi_weight, args.un_weight, args.in_epoch, args.end_pro)

        ema_decay = min( (0.01* epoch) ** (1/2.5)  ,0.99)
        ema_models(teacher_model, model, ema_decay)
        # record the training states after every epoch
        if writer is not None:
            with open(run_log_name, "a") as log_file:
                log_file.write("loss/loss@{}: {}".format(epoch, stat['loss']))
                log_file.write("loss/loss_ce@{}: {}".format(epoch, stat['loss_ce']))
                log_file.write("loss/loss_confi@{}: {}".format(epoch, c_loss))
                
            writer.add_scalar('loss/loss', stat['loss'], epoch)
            writer.add_scalar('loss/loss_ce', stat['loss_ce'], epoch)
            writer.add_scalar('loss/loss_confi', c_loss, epoch)

        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % \
              (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        with open(run_log_name, "a") as log_file:
            log_file.write('[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        # change lr according to the scheduler
        lr_scheduler.step()
        # save latest weights every epoch
        checkpoint_latest_path = os.path.join(args.checkpoints_dir, 'latest.pth')
        torch.save({
            'model': model_without_ddp.state_dict(),
        }, checkpoint_latest_path)
        torch.save({
            'model': teacher_model.state_dict(),
        },  os.path.join(args.checkpoints_dir, 'latest_teacher.pth'))
        # run evaluation
        
        if epoch % args.eval_freq == 0 and epoch != 0:
            t1 = time.time()
            result = evaluate_crowd_no_overlap(model, data_loader_val, device)
            t2 = time.time()

            mae.append(result[0])
            mse.append(result[1])
            # print the evaluation results
            print('=======================================test=======================================')
            print("mae:", result[0], "mse:", result[1], "time:", t2 - t1, "best mae:", np.min(mae), )
            with open(run_log_name, "a") as log_file:
                log_file.write("mae:{}, mse:{}, time:{}, best mae:{}".format(result[0], 
                                result[1], t2 - t1, np.min(mae)))
            print('=======================================test=======================================')
            # recored the evaluation results
            if writer is not None:
                with open(run_log_name, "a") as log_file:
                    log_file.write("metric/mae@{}: {}".format(step, result[0]))
                    log_file.write("metric/mse@{}: {}".format(step, result[1]))
                writer.add_scalar('metric/mae', result[0], step)
                writer.add_scalar('metric/mse', result[1], step)
                step += 1

            # save the best model since begining
            if abs(np.min(mae) - result[0]) < 0.01:
                checkpoint_best_path = os.path.join(args.checkpoints_dir, 'best_' + str(epoch) + '_mae.pth')
                torch.save({
                    'model': model_without_ddp.state_dict(),
                }, checkpoint_best_path)
                torch.save({
                    'model': teacher_model.state_dict(),
                }, os.path.join(args.checkpoints_dir, 'best_teacher_mae.pth'))                
    # total time for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



class crowd_test(Dataset):
    def __init__(self, imgdir, maskdir, img_trans, mask_trans):
        super(crowd_test, self).__init__()
        self.imgdir = imgdir
        self.maskdir = maskdir
        self.imglist = os.listdir(imgdir)
        self.masklist = [item.replace('.png', '_label.png') for item in self.imglist]       
        self.img_trans = img_trans
        self.mask_trans = mask_trans 

#         print(train_transforms)
    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        image = cv2.imread( osp.join(self.imgdir, self.imglist[idx]) )
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
        mask = Image.open( osp.join(self.maskdir, self.masklist[idx]) )
        image = self.img_trans(image)
        # print(image.size())
        # print(self.imglist[idx])
        mask = self.mask_trans(mask)

        return image, mask, self.imglist[idx].split('.png')[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet-confi training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    mask_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        # standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.label_pro == '10':
        uncertain_folder = '../CrowdCounting-P2PNet/part_A/uncertain_data_10/'
    elif args.label_pro == '5':
        uncertain_folder = '../CrowdCounting-P2PNet/part_A/uncertain_data_5/'
    elif args.label_pro == '40':
        uncertain_folder = '../CrowdCounting-P2PNet/part_A/uncertain_data_40/'
     
    dataset_t_ik = crowd_test(
        imgdir = uncertain_folder,
        maskdir='../CrowdCounting-P2PNet/part_A/uncertain_label/',
        img_trans = img_transform,
        mask_trans = mask_transform
        )
    
    t_sampler = torch.utils.data.RandomSampler(dataset_t_ik)


    data_loader_t = torch.utils.data.DataLoader(
        dataset_t_ik, batch_size=32,
        sampler=t_sampler, num_workers=15)  

    train_dict = {}
    train_dict_detail = {}
    train_dict_conf = {}

    # for _,_, namm in data_loader_t:
    #     for lio in range(len(namm)):
    #         train_dict[namm[lio]] = np.zeros([2,2])
    
    # for _,_, namm in data_loader_v:
    #     for lio in range(len(namm)):
    #         test_dict[namm[lio]] = np.zeros([2,2])




    main(args)
    import pickle
    # with open('conf_img_10/res_dict.pickle', 'wb') as handle:
    #     pickle.dump(train_dict_detail, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('conf_img_10/res_dict.json', 'w') as f:
    #     json.dump(train_dict_detail, f)
    # with open('conf_img_10/res_conf.pickle', 'wb') as handle:
    #     pickle.dump(train_dict_conf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('conf_img_10/res_conf.json', 'w') as f:
    #     json.dump(train_dict_conf, f)
    # np.save( 'conf_img_10/res_dict.npy' , train_dict_detail)
    # np.save( 'conf_img_10/res_conf.npy' , train_dict_conf)
    # for na in train_dict.keys():

    #     np.save( 'conf_train_un10/' + na +'.npy', train_dict[na] )

    # for na in test_dict.keys():

    #     np.save( 'conf_test_f_un10/' + na +'.npy', test_dict[na] )  