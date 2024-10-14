import torch
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.common import load_file_name_list
import os

class NiftiDataset(Dataset):

    def __init__(self, image_paths, mask_paths, transform=None):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path, sitk.sitkInt16))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path, sitk.sitkUInt8))

        # Normalize image
        image = (image - np.mean(image)) / (np.std(image) + 1e-10)

        # Apply transformations if specified
        if self.transform:
            image, mask = self.transform(image, mask)

        # Convert to PyTorch tensor
        image_tensor = torch.FloatTensor(image).unsqueeze(0)  # Add channel dimension
        mask_tensor = torch.LongTensor(mask)

        return image_tensor, mask_tensor


class NiftiDataset_test(Dataset):

    def __init__(self, image_paths, transform=None):

        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"文件不存在: {image_path}")
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path, sitk.sitkInt16))

        # Normalize image to range [0, 1]
        image = (image - np.mean(image)) / (np.std(image) + 1e-10)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        # Convert to PyTorch tensor
        image_tensor = torch.FloatTensor(image).unsqueeze(0)  # Add channel dimension

        return image_tensor

# # 定义数据集路径
# data_dir = "D:/zlx/pytorch_segment/data/training"
#
#
# #读取文件名
# image_paths, mask_paths = load_file_name_list(data_dir)
#
# print(image_paths,mask_paths)
#
# # 实例化数据集加载器
# dataset = NiftiDataset(image_paths, mask_paths, transform=None)
#
# # 创建数据加载器
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#
# # 遍历数据加载器
# for images, masks in dataloader:
#     print('images num: ', images.size()[0],'','masks num:',masks.size()[0])
#     for mask in masks:
#         print('mask size: ',[mask.size()[0],mask.size()[1],mask.size()[2]])
#         print(mask.size())
#     for image in images:
#         print('image size: ',[image.size()[0],image.size()[1],image.size()[2],image.size()[3]])
#         print(image.size())


# def crop_and_save_nii_image(input_file, output_file, x_start, y_start, z_start, width, height, depth):
#     # 加载nii文件
#     nii_image = nib.load(input_file)
#     image_data = nii_image.get_fdata()
#
#     # 裁剪图像
#     cropped_image_data = image_data[x_start:x_start+width,
#                                     y_start:y_start+height,
#                                     z_start:z_start+depth]
#
#     # 创建新的nii图像对象
#     cropped_nii_image = nib.Nifti1Image(cropped_image_data, affine=nii_image.affine)
#
#     # 保存裁剪后的图像到新的nii文件
#     cropped_nii_image.to_filename(output_file)
#
#     print("图像裁剪完成，并保存到", output_file)
#
#
# input_file = r'D:\zlx\pytorch_segment3D\data\test\image\61408463_8_r.nii'
# output_file = r'D:\zlx\pytorch_segment3D\data\test\image\img_clipping\61408463_8_r.nii.gz'
# x_start, y_start, z_start = 103, 186, 0
# width, height, depth = 128, 128, 128
# crop_and_save_nii_image(input_file, output_file, x_start, y_start, z_start, width, height, depth)

# image_path = 'D:/zlx/pytorch_segment/data/training/mask/F1.nii.gz'
# image = sitk.ReadImage(image_path)
#
# direction_matrix = image.GetDirection()
# print(direction_matrix)