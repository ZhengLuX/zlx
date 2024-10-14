import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict

from utils.my_dataset import NiftiDataset
from models.UNet import UNet
from models.layers import softmax_dim1
from training.weight_initi import weights_init_kaiming

from training.loss.combined_losses import DC_and_CE_loss

from training.lr_scheduler import CosineAnnealingWarmupRestarts

from utils.common import load_file_name_list, val, one_hot_encode_3d
from utils import metrics

from utils.log import Train_Logger

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

def main(args):

    device_ids = [0, 1]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_data_path = args.train_data_path
    val_data_path = args.val_data_path

    # data info
    train_image_paths, train_mask_paths = load_file_name_list(train_data_path)
    val_image_paths, val_mask_paths = load_file_name_list(val_data_path)
    train_dataset = NiftiDataset(train_image_paths, train_mask_paths, transform=None)
    val_dataset = NiftiDataset(val_image_paths, val_mask_paths, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=len(device_ids), shuffle=False, pin_memory=True)

    # model info
    model = UNet(kernel_size=3, in_channel=1, out_channel=args.n_classes).to(device)
    model.apply(weights_init_kaiming)
    model = nn.DataParallel(model, device_ids=device_ids)
    # print_network(model)

    # loss
    soft_dice_kwargs = {'activation': softmax_dim1}  # 如果需要，可以指定激活函数
    ce_kwargs = {'ignore_index': -100}  # 如果有忽略标签，可以设置

    # 初始化损失函数
    loss = DC_and_CE_loss(
        soft_dice_kwargs=soft_dice_kwargs,
        ce_kwargs=ce_kwargs,
        weight_ce=1.0,  # 交叉熵损失权重
        weight_dice=1.0,  # Dice损失权重
        ignore_label=None  # 如果有忽略标签，可以设置
    )

    # optimizer + lr_sch

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # lr_scheduler
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=200,  # 第一个周期的步数
        cycle_mult=1.5,  # 周期乘数
        max_lr=0.01,  # 最大学习率
        min_lr=1e-5,  # 最小学习率
        warmup_steps=20,  # warmup步数
        gamma=0.9  # 减少率
    )

    log = Train_Logger(args.log_path, "train_log")

    best = [-float('inf'), 0]  # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器
    alpha = 0.4  # 深监督衰减系数初始值

    for epoch in range(1, args.epochs + 1):

        train_log = train(model, train_loader, optimizer, loss, args.n_classes, alpha, epoch, device)
        val_log = val(model, val_loader, loss, args.n_classes, device)
        log.update(epoch, train_log, val_log)

        scheduler.step()

        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(args.model_save_path, 'latest_model.pth'))

        if val_log['Val_dice_liver'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(args.model_save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice_liver']
            trigger = 0
        else:
            trigger += 1
            if trigger >= 300:
                print('Early Stop !')
                break
        print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))

        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    parser.add_argument('--training-data-path', type=str, default="/home/zlx/PycharmProjects/pytorch_segment3D/data/training")
    parser.add_argument('--val-data-path', type=str, default="/home/zlx/PycharmProjects/pytorch_segment3D/data/val")

    # 模型保存地址
    parser.add_argument('--model-save-path', type=str, default="/home/zlx/PycharmProjects/pytorch_segment3D/weights")

    parser.add_argument('--log-path', type=str, default="/home/zlx/PycharmProjects/pytorch_segment3D/log")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weadd_argumentights', type=str, default='', help='initial weights path')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    # 设置Loss曲线存储路径
    parser.add_argument('--loss-path', type=str, default='/Loss', help='Path to save the loss curve plot')

    opt = parser.parse_args()

    main(opt)
