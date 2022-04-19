import shutil
import os
import time
import multiprocessing

import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tqdm import tqdm
from .video_transforms import (GroupRandomHorizontalFlip,
                               GroupMultiScaleCrop, GroupScale, GroupCenterCrop, GroupRandomCrop,
                               GroupNormalize, Stack, ToTorchFormatTensor, GroupRandomScale)


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


def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print("@@@@@@pred", pred)
        # print("@@@@@@correct", correct)

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracy_action(verb_output, verb_target, noun_output, noun_target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = verb_output.size(0)

        _, verb_pred = verb_output.topk(maxk, 1, True, True)
        verb_pred = verb_pred.t()
        verb_correct = verb_pred.eq(verb_target.view(1, -1).expand_as(verb_pred))
        # print("@@@@@@vvvpred", verb_pred)
        # print("@@@@@@vvvcorrect", verb_correct)

        _, noun_pred = noun_output.topk(maxk, 1, True, True)
        noun_pred = noun_pred.t()
        noun_correct = noun_pred.eq(noun_target.view(1, -1).expand_as(noun_pred))
        # print("@@@@@@nnnpred", noun_pred)
        # print("@@@@@@nnncorrect", noun_correct)
        correct = verb_correct * noun_correct

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, filepath=''):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'),
                        os.path.join(filepath, 'model_best.pth.tar'))


def get_augmentor(is_train, image_size, mean=None,
                  std=None, disable_scaleup=False, is_flow=False,
                  threed_data=False, version='v1', scale_range=None):

    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std
    scale_range = [256, 320] if scale_range is None else scale_range
    augments = []

    if is_train:
        if version == 'v1':
            augments += [
                GroupMultiScaleCrop(image_size, [1, .875, .75, .66])
            ]
        elif version == 'v2':
            augments += [
                GroupRandomScale(scale_range),
                GroupRandomCrop(image_size),
            ]
        augments += [GroupRandomHorizontalFlip(is_flow=is_flow)]
    else:
        scaled_size = image_size if disable_scaleup else int(image_size / 0.875 + 0.5)
        augments += [
            GroupScale(scaled_size),
            GroupCenterCrop(image_size)
        ]
    augments += [
        Stack(threed_data=threed_data),
        ToTorchFormatTensor(),
        GroupNormalize(mean=mean, std=std, threed_data=threed_data)
    ]

    augmentor = transforms.Compose(augments)
    return augmentor


def build_dataflow(dataset, is_train, batch_size, workers=36, is_distributed=False):
    workers = min(workers, multiprocessing.cpu_count())
    shuffle = False

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    if is_train:
        shuffle = sampler is None

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=workers, pin_memory=True, sampler=sampler)

    return data_loader


def train(data_loader, model, criterion, optimizer, epoch, display=100,
          steps_per_epoch=99999999999, clip_gradient=None, gpu_id=None, rank=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # set different random see every epoch
    if dist.is_initialized():
        data_loader.sampler.set_epoch(epoch)

    # switch to train mode
    model.train()
    #for param in model.parameters():
        #print("!", param, param.requires_grad)
    end = time.time()
    num_batch = 0
    with tqdm(total=len(data_loader)) as t_bar:
        for i, (images, target) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # compute output
            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)

            #Two fc layer code
            two_fc_action = False

            #####################
            if two_fc_action:
                #Create noun and verb tensors from combined tensor
                # print("@@@@@@", target)
                verbmask = target > 1000000
                verb_target = target * verbmask
                verb_target.apply_(lambda x: int(str(x)[:str(x).index('000000')] if x > 0 else 0))
                # print("verb tensor!", verb_target, verb_target.shape)
                noun_target = target
                noun_target.apply_(lambda x: x % 1000000)
                # print("noun tensor!", noun_target, noun_target.shape)

                verb_target = verb_target.cuda(gpu_id, non_blocking=True)
                noun_target = noun_target.cuda(gpu_id, non_blocking=True)
            #####################

            output = model(images)

            ####################
            if two_fc_action:
                verb_output = output[0]
                noun_output = output[1]
                # print("!", output)
                # print("!!VERB OUTPUT", verb_output, verb_output.shape)
                # print("!!NOUN OUTPUT", noun_output, noun_output.shape)
                output = output[0] #TEMP verb
            #####################

            target = target.cuda(gpu_id, non_blocking=True)

            loss = criterion(output, target)

            #####################
            if two_fc_action:
                verb_loss = criterion(verb_output, verb_target)
                noun_loss = criterion(noun_output, noun_target)
                loss = 0.5*(verb_loss+noun_loss)
                # print("!!!!!!!verbloss", verb_loss)
                # print("!!!!!!!nounloss", noun_loss)
                # print("!!!!!!!loss", loss)
            #####################



            # measure accuracy and record loss
            
            prec1, prec5 = accuracy(output, target)

            #####################
            if two_fc_action:
                prec1, prec5 = accuracy_action(verb_output, verb_target, noun_output, noun_target)
            #####################

            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size

            losses.update(loss.item(), images.size(0))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))
            # compute gradient and do SGD step
            loss.backward()

            if clip_gradient is not None:
                _ = clip_grad_norm_(model.parameters(), clip_gradient)

            optimizer.step()
            optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % display == 0 and rank == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(data_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5), flush=True)
            num_batch += 1
            t_bar.update(1)
            if i > steps_per_epoch:
                break

    return top1.avg, top5.avg, losses.avg, batch_time.avg, data_time.avg, num_batch


def validate(data_loader, model, criterion, gpu_id=None, mAP_softmax_return = False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    #####################
    softmax_iter = []
    #####################

    with torch.no_grad(), tqdm(total=len(data_loader)) as t_bar:
        end = time.time()

        for i, (images, target) in enumerate(data_loader):

            if gpu_id is not None:
                images = images.cuda(gpu_id, non_blocking=True)

            two_fc_action = False

            #####################
            if two_fc_action:
                #Create noun and verb tensors from combined tensor
                verbmask = target > 1000000
                verb_target = target * verbmask
                verb_target.apply_(lambda x: int(str(x)[:str(x).index('000000')] if x > 0 else 0))
                noun_target = target
                noun_target.apply_(lambda x: x % 1000000)

                verb_target = verb_target.cuda(gpu_id, non_blocking=True)
                noun_target = noun_target.cuda(gpu_id, non_blocking=True)
            #####################


            target = target.cuda(gpu_id, non_blocking=True)



            # compute output
            output = model(images)

            #####################
            if mAP_softmax_return:
	            # print("TARGET", target)
	            # print("OUTPUT", output)
	            m = torch.nn.Softmax(dim=1)
	            softmax_output = m(output)
	            softmax_output_list = softmax_output.tolist()
	            #print("smOUTPUT_LIST", softmax_output_list)
	            for i in softmax_output_list:
	                softmax_iter.append(map(str,i))
            #####################

            ####################
            if two_fc_action:
                verb_output = output[0]
                noun_output = output[1]
                output = output[0] #TEMP verb
            #####################

            loss = criterion(output, target)

            #####################
            if two_fc_action:
                verb_loss = criterion(verb_output, verb_target)
                noun_loss = criterion(noun_output, noun_target)
                loss = 0.5*(verb_loss+noun_loss)
            #####################


            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target)

            #####################
            if two_fc_action:
                prec1, prec5 = accuracy_action(verb_output, verb_target, noun_output, noun_target)
            #####################

            if dist.is_initialized():
                world_size = dist.get_world_size()
                dist.all_reduce(prec1)
                dist.all_reduce(prec5)
                prec1 /= world_size
                prec5 /= world_size
            losses.update(loss.item(), images.size(0))
            top1.update(prec1[0], images.size(0))
            top5.update(prec5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)

    
    #Here, softmax_iter is empty of bool is False
    return top1.avg, top5.avg, losses.avg, batch_time.avg, softmax_iter
