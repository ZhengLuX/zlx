import os
import torch
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from utils import metrics
from collections import OrderedDict


def Min_Max_normalization(image_array):
    return (image_array-np)

def Z_Score_normalization(image_array):
    return (image_array-np.mean(image_array)) / (np.std(image_array) + 1e-8)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' %num_params)
def load_file_name_list(root_dir):

    # 构建图像和掩码文件夹的路径
    image_dir = os.path.join(root_dir, 'image')
    mask_dir = os.path.join(root_dir, 'mask')

    # 获取图像文件夹下所有文件的完整路径，筛选出具有指定扩展名的文件
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.mhd', '.nii.gz'))]

    # 获取掩码文件夹下所有文件的完整路径，筛选出具有指定扩展名的文件
    mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.lower().endswith(('.mhd', '.nii.gz'))]

    return image_paths, mask_paths

def load_file_name_list_test(root_dir):

    # 构建图像和掩码文件夹的路径
    image_dir = os.path.join(root_dir, 'image')

    # 获取图像文件夹下所有文件的完整路径，筛选出具有指定扩展名的文件
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.nii.gz', '.mhd'))]

    return image_paths


def one_hot_encode_3d(tensor, n_classes=2):

    n, s, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, s, h, w).scatter_(1, tensor.view(n, 1, s, h, w), 1)

    return one_hot


def train(model, train_loader, optimizer, loss_func, n_classes, alpha, epoch, device):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_classes)

    for idx, (image, mask) in tqdm(enumerate(train_loader), total=len(train_loader)):
        image, mask = image.float(), mask.long()
        mask = one_hot_encode_3d(mask, n_classes)
        image, mask = image.to(device), mask.to(device)
        optimizer.zero_grad()

        output = model(image)
        # loss0 = loss_func(output[0], mask)
        # loss1 = loss_func(output[1], mask)
        # loss2 = loss_func(output[2], mask)
        # loss3 = loss_func(output[3], mask)
        #
        # loss = loss3 + alpha * (loss0 + loss1 + loss2)

        loss = loss_func(output, mask)
        # print(loss)

        loss.backward()
        optimizer.step()

        # train_loss.update(loss3.item(), image.size(0))
        # train_dice.update(output[3], mask)

        train_loss.update(loss.item(), image.size(0))
        train_dice.update(output, mask)

    train_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice_liver': train_dice.avg[1]})
    if n_classes == 3: train_log.update({'Train_dice_tumor': train_dice.avg[2]})
    return train_log


def val(model, val_loader, loss_func, n_classes, device):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_classes)
    with torch.no_grad():
        for idx, (data, mask) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, mask = data.float(), mask.long()
            mask = one_hot_encode_3d(mask, n_classes)
            data, mask = data.to(device), mask.to(device)
            output = model(data)
            loss = loss_func(output, mask)

            val_loss.update(loss.item(), data.size(0))
            val_dice.update(output, mask)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_liver': val_dice.avg[1]})
    if n_classes == 3: val_log.update({'Val_dice_tumor': val_dice.avg[2]})
    return val_log


def predict(model, test_loader, n_classes, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for idx, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
            data = data.float()
            data = data.to(device)
            output = model(data)
            output = output.cpu().numpy()  # 将输出转换为 numpy 数组

            # 将输出转为nii
            output = sitk.GetImageFromArray(output)
            predictions.append(output)

    return predictions








