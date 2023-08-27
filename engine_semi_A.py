# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import math
import os
from re import A
import sys
from typing import Iterable

import torch

import util.misc as utils
from util.misc import NestedTensor
import numpy as np
import time
import torchvision.transforms as standard_transforms
import cv2
from pytorchltr.loss import LambdaARPLoss2
from pytorchltr.loss import LambdaARPLoss1
from pytorchltr.loss import PairwiseHingeLoss

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def vis(samples, targets, pred, vis_dir, des=None):
    '''
    samples -> tensor: [batch, 3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    '''
    gts = [t['point'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    # draw one by one
    for idx in range(samples.shape[0]):
        sample = restore_transform(samples[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        max_len = np.max(sample_gt.shape)

        size = 2
        # draw gt
        for t in gts[idx]:
            sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
        # draw predictions
        for p in pred[idx]:
            sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        name = targets[idx]['image_id']
        # save the visualized images
        if des is not None:
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_gt.jpg'.format(int(name), 
                                                des, len(gts[idx]), len(pred[idx]))), sample_gt)
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_pred.jpg'.format(int(name), 
                                                des, len(gts[idx]), len(pred[idx]))), sample_pred)
        else:
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_gt.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_gt)
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_pred.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_pred)

def loss_imgplement(model_output, ground):
    norm_gt = (torch.max(ground) - ground) / (torch.max(ground) - torch.min(ground))
    norm_gt = norm_gt[None, :]
    model_output = model_output[None, :]
    norm_gt = norm_gt.cuda()
    loss_fn = LambdaARPLoss2(10)
    loss_fn = loss_fn.cuda()
    n_count = torch.tensor([len(norm_gt[0]) ])
    
    return loss_fn(model_output, norm_gt, n_count.cuda())/ (len(norm_gt[0])*len(norm_gt[0]))

    
    

def listmle_loss( tra_dict, img, name, model ):
    img = img.cuda()
    _, out_1 = model(img)
    gt_list = []
    for ind in range(len(name)):
        gt_list.append( tra_dict[name[ind]] )

    gt_list = np.array(gt_list)
    gt_list = torch.tensor(gt_list)
    model_o = out_1.squeeze()
    model_o = model_o.flatten()
    gt_list = gt_list.flatten()

    return loss_imgplement(model_o, gt_list)


def val_tar_gen(imgs, model_t):
    model_t.eval()
    outputs, gen_con = model_t(imgs)
    all_tar = []
    for ink in range(imgs.size()[0]):
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

        all_tar.append(predict_tar)
    return torch.tensor( np.array(all_tar) ), gen_con

def patch_gen(s_co):
    s_co = np.array(s_co)
    s_co = (s_co - np.min(s_co)) /( np.max(s_co) - np.min(s_co) )
    s_co = s_co/np.sum(s_co)

    chn = np.random.choice(np.arange(len(s_co) ), size = 1, p = s_co)

    return chn


import copy
def crop_emp(img_s, tar_s):
    x = np.random.randint(1, [128, 128, 128])
    y = np.random.randint(1, [128, 128, 128])
    img_new = copy.deepcopy(img_s)
    tar_new = copy.deepcopy(tar_s)
    for i in range(3):
        # print()
        red1 = np.random.randint(low = 20,high = 30, size=1)[0]
        red2 = np.random.randint(low = 20,high = 30, size=1)[0]
        img_new[0, x[i]: np.minimum( 128,x[i]+red1 ), y[i]: np.minimum( 128,y[i]+red2 )] = -2.1179
        img_new[1, x[i]: np.minimum( 128,x[i]+red1 ), y[i]: np.minimum( 128,y[i]+red2 )] = -2.0357
        img_new[2, x[i]: np.minimum( 128,x[i]+red1 ), y[i]: np.minimum( 128,y[i]+red2 )] = -1.8044
        tar_new[ x[i]: np.minimum( 128,x[i]+red1 ), y[i]: np.minimum( 128,y[i]+red2 )] = 0
    return img_new, tar_new


# def unsupervise_loss( imgs, model, sup_patch, sup_ptar, sup_pconf, creti , epoch ):
    
#     psedu_tar, cen_1 = val_tar_gen(imgs.cuda(), model)
#     new_imgs_list = []

#     new_tar_ll = []

#     con_threshold = 0.5
#     # con_threshold = np.maximum(0.8 - (0.6 /20) * (epoch -10) , 0.2)

#     for co in range(imgs.size()[0]):
#         # new_imgs = torch.zeros(imgs[0].size())
#         complete = 1
#         for i_1 in range(2):
#             for i_2 in range(2):
#                 if cen_1[co][0][i_1,i_2] > con_threshold:
#                     pass

#                 else:
#                     chon = patch_gen(sup_pconf)[0]
#                     imgs[co, :, i_1*64:(i_1*64 + 64), i_2*64:(i_2*64 + 64)] = torch.tensor( sup_patch[chon] )
#                     psedu_tar[co, i_1*64:(i_1*64 + 64), i_2*64:(i_2*64 + 64)] = torch.tensor( sup_ptar[chon] )

#                     complete = 0

#         if complete == 1:
#             if np.random.choice(2) == 0:
#                 imgs[co], psedu_tar[co] = crop_emp(imgs[co], psedu_tar[co])

#             else:
#                 x_pos = np.random.choice(64)
#                 y_pos = np.random.choice(64)
#                 chon = patch_gen(sup_pconf)[0]

#                 imgs[co,:, x_pos:(x_pos + 64), y_pos:(y_pos+64)] = torch.tensor( sup_patch[chon] )

#                 psedu_tar[co, x_pos:(x_pos + 64), y_pos:(y_pos+64)] = torch.tensor( sup_ptar[chon] )


#         nee_tar = {}
#         fff = np.array(np.where(psedu_tar[co].numpy() > 0)).T
#         ff_n = np.zeros(fff.shape)

#         ff_n[:,1] = fff[:,0]
#         ff_n[:,0] = fff[:,1]

#         nee_tar['point'] = torch.tensor(ff_n).float()
#         nee_tar['labels'] = torch.tensor( np.ones(len(ff_n)) ).long()

#         new_tar_ll.append(nee_tar)
    
#     new_tar_ll = tuple(new_tar_ll)
#     model.train()
#     outputs,_ = model(imgs.cuda())

#     targets = [{k: v.cuda() for k, v in t.items()} for t in new_tar_ll]

#     loss_dict = creti(outputs, targets)

#     weight_dict = creti.weight_dict

#     losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) 

#     return losses



def unsupervise_loss( imgs, model, model_teacher,creti , epoch, end_pro ):
    
    psedu_tar, cen_1 = val_tar_gen(imgs.cuda(), model_teacher)
    new_imgs_list = []

    new_tar_ll = []

    # con_threshold = 0.35
    # con_threshold = np.maximum(0.9 - (0.5 /50) * (epoch -100) , 0.5)
    # con_threshold = np.maximum(0.9 - (0.5 /140) * (epoch -10) , 0.5)
    con_threshold = np.maximum(0.9 - ( (1-end_pro) /140) * (epoch -10) , end_pro)
    in_input = []
    # in_tar = []
    for co in range(imgs.size()[0]):
        # new_imgs = torch.zeros(imgs[0].size())
        complete = 1
        
        for i_1 in range(2):
            for i_2 in range(2):
                if cen_1[co][0][i_1,i_2] > con_threshold:
                    complete = 0
                    # pass

                else:
                    # chon = patch_gen(sup_pconf)[0]
                    imgs[co, 0, i_1*64:(i_1*64 + 64), i_2*64:(i_2*64 + 64)] = -2.1179
                    imgs[co, 1, i_1*64:(i_1*64 + 64), i_2*64:(i_2*64 + 64)] = -2.0357
                    imgs[co, 2, i_1*64:(i_1*64 + 64), i_2*64:(i_2*64 + 64)] = -1.8044
                    psedu_tar[co, i_1*64:(i_1*64 + 64), i_2*64:(i_2*64 + 64)] = 0

        imgs[co], psedu_tar[co] = crop_emp(imgs[co], psedu_tar[co])

        if complete == 0:
            in_input.append( imgs[co] )
        # if torch.min(cen_1[co][0]) > con_threshold:
        #     # imgs[co], psedu_tar[co] = crop_emp(imgs[co], psedu_tar[co])
        #     in_input.append( imgs[co] )
        #     # in_tar.append( psedu_tar[co] )

            nee_tar = {}
            fff = np.array(np.where(psedu_tar[co].numpy() > 0)).T
            ff_n = np.zeros(fff.shape)

            ff_n[:,1] = fff[:,0]
            ff_n[:,0] = fff[:,1]

            nee_tar['point'] = torch.tensor(ff_n).float()
            nee_tar['labels'] = torch.tensor( np.ones(len(ff_n)) ).long()

            new_tar_ll.append(nee_tar)
    
    if len(in_input) == 0:
        return 0
    
    new_im = torch.zeros( [ len(in_input), imgs[co].size()[0], imgs[co].size()[1], imgs[co].size()[2] ] )
    for ikk in range(len(in_input)):
        new_im[ikk] = in_input[ikk]

    new_tar_ll = tuple(new_tar_ll)
    # model.train()
    outputs,_ = model(new_im.cuda())

    targets = [{k: v.cuda() for k, v in t.items()} for t in new_tar_ll]

    loss_dict = creti(outputs, targets)

    weight_dict = creti.weight_dict

    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) 

    return losses






# the training routine
def train_one_epoch(model: torch.nn.Module, model_teacher: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, train_d: dict, tloader: Iterable, unloader:Iterable,
                    sup_patch: list, sup_ptar: list, sup_pconf: list,
                    max_norm: float = 0, confi_weight: float = 1., un_weight:float = 1., in_epoch:int = 50, end_pro:float = 0.5
                    ):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # iterate all training samples
    dataloader_iterator = iter(tloader)
    dataloader_sup = iter(data_loader)

    for unsamples, _ in unloader:

    # for samples, targets in data_loader:

        try:
            img_1, _, name_1 = next(dataloader_iterator)
            samples, targets = next(dataloader_sup)
        except StopIteration:
            dataloader_iterator = iter(tloader)
            dataloader_sup = iter(data_loader)
            img_1, _, name_1 = next(dataloader_iterator)
            samples, targets = next(dataloader_sup)


        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # forward
        outputs,_ = model(samples)
        # calc the losses
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        confi_losses = listmle_loss( train_d, img_1, name_1, model )
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) 
        if epoch > in_epoch:
            unlosses = unsupervise_loss( unsamples, model,model_teacher, criterion , epoch, end_pro )
            losses = losses + confi_weight * confi_losses + un_weight* unlosses
        else:
            losses = losses + confi_weight * confi_losses

        # reduce all losses
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        # backward
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # update logger
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, confi_losses.item()

# the inference routine
@torch.no_grad()
def evaluate_crowd_no_overlap(model, data_loader, device, vis_dir=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes = []
    mses = []
    for samples, targets in data_loader:
        samples = samples.to(device)

        outputs,_ = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]

        gt_cnt = targets[0]['point'].shape[0]
        # 0.5 is used by default
        threshold = 0.5

        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())
        # if specified, save the visualized images
        if vis_dir is not None: 
            vis(samples, targets, [points], vis_dir)
        # accumulate MAE, MSE
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(float(mae))
        mses.append(float(mse))
    # calc MAE, MSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    return mae, mse