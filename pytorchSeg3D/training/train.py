import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.common import load_file_name_list, val, one_hot_encode_3d
from utils.my_dataset import NiftiDataset
from utils.log import Train_Logger

from models.layers import softmax_dim1
from models.UNet import UNet
from weight_initi import weights_init_kaiming

from training.loss.combined_losses import DC_and_CE_loss
from training.lr_scheduler import CosineAnnealingWarmupRestarts

from collections import OrderedDict
from utils import metrics

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
        # loss = loss3 + alpha * (loss0 + loss1 + loss2)

        # loss.backward()
        # optimizer.step()

        # train_loss.update(loss3.item(), image.size(0))
        # train_dice.update(output[3], mask)

        loss = loss_func(output, mask)

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), image.size(0))
        train_dice.update(output, mask)

    train_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice_liver': train_dice.avg[1]})
    if n_classes == 3: train_log.update({'Train_dice_tumor': train_dice.avg[2]})
    return train_log

def main(args):

    device_ids = [0, 1]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("../weights") is False:
        os.makedirs("../weights")

    train_data_path = args.train_data_path
    val_data_path = args.val_data_path

    # data
    train_image_paths, train_mask_paths = load_file_name_list(train_data_path)
    val_image_paths, val_mask_paths = load_file_name_list(val_data_path)
    train_dataset = NiftiDataset(train_image_paths, train_mask_paths, transform=None)
    val_dataset = NiftiDataset(val_image_paths, val_mask_paths, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=len(device_ids), shuffle=False, pin_memory=True)

    # model
    model = UNet(kernel_size=3, in_channel=1, out_channel=args.n_classes).to(device)
    model.apply(weights_init_kaiming)
    # model  = sUNet(in_channel=1, out_channel=args.n_classes, training=True).to(device)
    model = nn.DataParallel(model,device_ids=device_ids)
    # print_network(model)

    # loss
    soft_dice_kwargs = {'activation': softmax_dim1}
    ce_kwargs = {'ignore_index': -100}  # 设置为一个不在标签中的值

    loss = DC_and_CE_loss(soft_dice_kwargs=soft_dice_kwargs,
                           ce_kwargs=ce_kwargs,
                           weight_ce=1.0,
                           weight_dice=1.0,
                           ignore_label=None)

    # optimizer + lr_scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=100,
        cycle_mult=1.0,
        max_lr=args.lr,
        min_lr=1e-3,
        warmup_steps=10,
        gamma=1.0
    )

    log = Train_Logger(args.log_path, "train_log")

    best = [-float('inf'), 0]  # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器
    alpha = 0.4  # 深监督衰减系数初始值

    for epoch in range(1, args.epochs + 1):

        # adjust_learning_rate(optimizer, epoch, args)

        train_log = train(model, train_loader, optimizer, loss, args.n_classes, alpha, epoch, device)
        val_log = val(model, val_loader, loss, args.n_classes, device)
        log.update(epoch, train_log, val_log)

        # 更新学习率
        scheduler.step()  # 调用调度器的step方法更新学习率

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
