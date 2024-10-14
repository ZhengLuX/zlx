import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.UNet import UNet,sUNet
from models.ResUNet import ResUNet
from models.KiUNet import KiUNet_min
from utils import log
from utils.log import Test_Logger
import nibabel as nib
from utils.my_dataset import NiftiDataset, NiftiDataset_test
from utils.common import load_file_name_list_test, predict
from collections import OrderedDict
import SimpleITK as sitk

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # 移除 'module.' 前缀
        new_state_dict[name] = v
    return new_state_dict

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    test_data_path = args.test_data_path

    # data info
    test_image_paths = load_file_name_list_test(test_data_path)
    test_dataset = NiftiDataset_test(test_image_paths, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # model info
    model = UNet(kernel_size=3, in_channel=1, out_channel=args.n_classes).to(device)
    # model = ResUNet(in_channel=1, out_channel=args.n_classes, training=False).to(device)
    checkpoint = torch.load(args.weight)
    model.load_state_dict(remove_module_prefix(checkpoint['net']))
    model.eval()

    # print_network(model)

    predictions = predict(model, test_loader, args.n_classes, device)

    for i in range(len(predictions)):
        # 保存 NIfTI 文件
        sitk.WriteImage(predictions[i], f'./results/output/predicted_result{i}.nii.gz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)

    # 数据集所在根目录
    parser.add_argument('--test-data-path', type=str, default="/home/zlx/PycharmProjects/pytorch_segment3D/data/test")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weight', type=str, default=r'/home/zlx/PycharmProjects/pytorch_segment3D/weights/latest_model.pth', help='initial weights path')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
